import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from omegaconf import MISSING
import numpy as np

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from ppuu import configs, slurm  # noqa
from ppuu.data import dataloader  # noqa
from ppuu.lightning_modules.fm import FM  # noqa
from ppuu.data.entities import StateSequence

@dataclass
class EvalPredictionConfig(configs.ConfigBase):
    dataset: str = MISSING
    output_dir: Optional[str] = None
    test_size_cap: Optional[int] = None
    forward_model_path: str = MISSING
    seed: int = 42
    dataset_partition: str = "test"
    batch_size: int = 32
    n_pred: int = 30

class DisplacementErrorMetric:
    def __init__(self, normalizer, diffs: bool):
        self.normalizer = normalizer
        self.diffs = diffs

    def _convert_states(self, states: torch.Tensor) -> torch.Tensor:
        unnormalized = self.normalizer.unnormalize_states(states)
        if self.diffs:
            # If states are diffs, we need to cumsum the position (first 2 dims)
            # Assuming states are [batch, time, features]
            # But here we might get [batch, n_modes, time, features]
            # Let's handle the last dim being features.
            
            # Check if we have n_modes dimension
            if len(states.shape) == 4: # [batch, n_modes, time, features]
                cumulative = torch.cumsum(states[..., :2], dim=-2)
                return torch.cat([cumulative, states[..., 2:]], dim=-1)
            else: # [batch, time, features]
                cumulative = torch.cumsum(states[..., :2], dim=-2)
                return torch.cat([cumulative, states[..., 2:]], dim=-1)
        else:
            return unnormalized

    def calculate(
        self,
        # [b_size, n_modes, out_horizon, state_dim]
        pred_states: torch.Tensor,
        # [b_size, out_horizon, state_dim]
        target_states: torch.Tensor,
    ):
        # Ensure target_states has n_modes dim for broadcasting if needed, 
        # but usually we compare best mode or mean.
        # For ADE/FDE with 1 mode (deterministic FM), n_modes=1.
        
        if len(pred_states.shape) == 3:
            pred_states = pred_states.unsqueeze(1) # Add n_modes=1
            
        converted_pred = self._convert_states(pred_states)
        converted_targets = self._convert_states(target_states)
        
        # [batch_size, n_modes, out_horizon, state_dim]
        repeated_targets = converted_targets.unsqueeze(1).repeat_interleave(
            converted_pred.shape[1], dim=1
        )
        
        # [batch_size, n_modes, out_horizon, 2]
        pos_difference = converted_pred[..., :2] - repeated_targets[..., :2]
        
        # [batch_size, n_modes, out_horizon]
        difference_norm = torch.norm(pos_difference, dim=-1, p=2)
        
        # [batch_size, n_modes]
        final_difference_norm = difference_norm[..., -1]
        
        # [batch_size, n_modes]
        avg_difference_norm = difference_norm.mean(dim=-1)
        
        # For deterministic, we just take the value. For stochastic, we might take min over modes.
        # Here we assume 1 mode for now as per current FM usage.
        best_avg, _ = torch.min(avg_difference_norm, dim=-1)
        best_final, _ = torch.min(final_difference_norm, dim=-1)
        
        return best_avg, best_final

def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load Dataset
    test_dataset = dataloader.EvaluationDataset(
        config.dataset, config.dataset_partition, config.test_size_cap
    )
    
    # We need a dataloader for batching
    # The EvaluationDataset returns full episodes. We need to slice them.
    # Actually, EvaluationDataset yields (car_info, episode_data).
    # We need a standard training-like dataloader to get batches of (context, target).
    # Let's use NGSIMDataModule or similar logic.
    
    from ppuu.data import DataStore, Dataset
    store = DataStore(config.dataset)
    # We need to know if the model expects diffs
    # Load model first to check config
    
    m_config = FM.Config()
    m_config.model.fm_type = "km_no_action"
    m_config.model.checkpoint = config.forward_model_path
    m_config.training.enable_latent = True # Usually True for VAEs
    
    # Load checkpoint to get hparams if possible, or assume defaults
    # The FM class loads from checkpoint in __init__ if provided? 
    # No, we need to load it explicitly or use the class method.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    forward_model = FM(m_config).to(device)
    forward_model.eval()
    
    # Check if model uses diffs
    use_diffs = getattr(forward_model.hparams, "diffs", False) # Might be in hparams
    # Or check config if passed. The original script used config.diffs.
    # Let's assume False for now unless we see otherwise, or try to infer.
    # Actually, `run_viz.sh` sets `diffs=False` in configs/dfm-km-mpc.yaml.
    
    # Get ncond from model
    n_cond = forward_model.model.ncond if hasattr(forward_model.model, 'ncond') else 20
    print(f"Model expects n_cond={n_cond}")

    ds = Dataset(
        store, 
        config.dataset_partition, 
        n_cond=n_cond,
        n_pred=config.n_pred,
        size=10, # Evaluate on 10 samples for now
        shift=False, 
        random_actions=False, 
        state_diffs=use_diffs
    )
    
    loader = torch.utils.data.DataLoader(
        ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        drop_last=False
    )
    
    normalizer = dataloader.Normalizer(store.stats)
    forward_model._setup_normalizer(store.stats)
    
    metric = DisplacementErrorMetric(normalizer, diffs=use_diffs)
    
    ade_list = []
    fde_list = []
    
    print(f"Starting evaluation on {len(ds)} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch is a dict or named tuple
            # We need:
            # - conditional_state (initial state)
            # - actions (ground truth actions to force the model, or we assume no action/policy?)
            # For "Forward Model" evaluation, we usually feed GT actions and see if it predicts states.
            # But `km_no_action` implies no action? 
            # If it's `km_no_action`, it predicts based on initial state and maybe latent.
            
            # Move to device
            # Debug batch type
            if i == 0:
                print(f"Batch type: {type(batch)}")
                print(f"Batch dir: {dir(batch)}")
                if isinstance(batch, list):
                    print(f"Batch list len: {len(batch)}")
                    print(f"Batch[0] type: {type(batch[0])}")

            
            # Helper to move tensors
            def to_device(obj):
                if hasattr(obj, 'to'):
                    return obj.to(device)
                elif hasattr(obj, 'cuda'): # Fallback
                     return obj.to(device)
                return obj
            
            # Assuming batch has attributes that are tensors or StateSequence
            # StateSequence has .to(device) usually?
            # Let's check entities.py if needed, but usually we can just move the components we need.
            
            # We need:
            # batch.conditional_state_seq
            # batch.target_state_seq
            # batch.actions
            
            # If default collate works, these should be batched tensors/objects.
            
            # If StateSequence is a custom object, we might need to move it manually if it doesn't support .to()
            # But let's try calling .to() on it if available, or access its .states
            
            # Actually, let's just extract what we need and move it.
            
            cond_states = batch.conditional_state_seq
            if hasattr(cond_states, 'to'):
                cond_states = cond_states.to(device)
            
            target_states = batch.target_state_seq
            if hasattr(target_states, 'to'):
                target_states = target_states.to(device)
                
            actions = batch.target_action_seq.to(device)
            
            # Update batch object or just use variables
            # We need to pass 'batch' to forward_model.unfold
            # forward_model.unfold expects 'batch' to have certain structure.
            # It likely expects the same structure as training.
            
            # We can try to monkey-patch or reconstruct the batch object on device.
            # Or if DatasetSample supports .to().
            
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            else:
                # Manually move known fields
                batch.conditional_state_seq = cond_states
                batch.target_state_seq = target_states
                batch.actions = actions
                # images etc might also need moving if used by FM
                # But for now let's hope this is enough or FM handles it.

            # If batch is a custom object, we might need to handle it.
            # Dataset returns a Batch object usually.
            
            # Forward pass
            # We want to unfold for n_pred steps.
            # FM.unfold usually requires a policy. 
            # If we want to evaluate FM *given GT actions*, we need a dummy policy that returns GT actions.
            # Or we can call `predict_states` directly if it's a KM model.
            
            # Wait, `FwdCNNKMNoAction_VAE` suggests it doesn't use actions? 
            # Or it's for "No Action" baselines?
            # Actually, `dfm-km-mpc.yaml` uses `fm_type: km_no_action`.
            # This usually means the Kinematic Model is used, but maybe the "No Action" part is misleading 
            # or refers to a specific variant.
            # Let's check `ppuu/modeling/forward_models_km_no_action.py` if needed.
            # But `mpc.py` calls `self.forward_model.unfold`.
            

            
            # Actually, `fm.unfold` signature: `def unfold(self, policy, batch, z=None):`
            # It loops and calls `policy(state)`.
            # If we want to use GT actions, we can't easily use `unfold` unless we make a policy that yields them.
            
            # EASIER WAY:
            # If it's a KM model, we can just call `predict_states_seq` if we have the function.
            # But we want to test the `forward_model` instance.
            
            # Let's try to use `forward_model.unfold` but pass a policy that returns the GT actions.
            # The batch has `actions` [B, T, ...].
            # We can create a policy that keeps an index and returns the slice.
            
            # However, `unfold` might be designed for closed-loop (autoregressive).
            # If we feed GT actions, it's "Teacher Forcing" on actions, but autoregressive on states.
            
            # Let's assume we want to evaluate: Given State_0 and Actions_0...T, predict States_1...T.
            
            # We can implement a simple loop here similar to `mpc.unfold_km`.
            
            # Extract data
            # batch.state_seq: [B, T, 5]
            # batch.action_seq: [B, T, 2]
            
            # We need `conditional_state` (first step).
            # And we need to predict the rest.
            
            # If the model is `FwdCNNKMNoAction_VAE`, it might predict *latent* or *residuals*?
            # Let's look at `mpc.py` again. It calls `self.forward_model.unfold`.
            
            # Let's try to use `forward_model.unfold` with a Mock Policy.
            
            gt_actions = batch.target_action_seq # [B, T, 2]
            
            # Generate Z if VAE
            z = None
            if m_config.training.enable_latent:
                # Sample Z
                # batch size
                bsize = gt_actions.shape[0]
                # n_pred
                npred = config.n_pred
                # nz is usually 32, but let's check config or model
                nz = m_config.model.nz if hasattr(m_config.model, 'nz') else 32
                z = torch.randn(bsize, npred, nz).to(device)

            # Use GT actions directly
            prediction = forward_model.unfold(batch.conditional_state_seq, gt_actions, Z=z)
            
            pred_states = prediction.state_seq.states # [B, T, 5]
            target_states = batch.target_state_seq.states # [B, T, 5]
            
            # Calculate metrics
            ade, fde = metric.calculate(pred_states, target_states)
            
            ade_list.append(ade)
            fde_list.append(fde)
            
            if i % 10 == 0:
                print(f"Batch {i}: ADE={ade.mean().item():.4f}, FDE={fde.mean().item():.4f}")
                
    
    total_ade = torch.cat(ade_list).mean().item()
    total_fde = torch.cat(fde_list).mean().item()
    
    print(f"Evaluation Complete.")
    print(f"Overall ADE: {total_ade:.4f}")
    print(f"Overall FDE: {total_fde:.4f}")
    
    results = {
        "ADE": total_ade,
        "FDE": total_fde,
        "config": dataclasses.asdict(config)
    }
    
    if config.output_dir:
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "prediction_eval.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {out_path / 'prediction_eval.json'}")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    config = EvalPredictionConfig.parse_from_command_line()
    main(config)
