"""Train a policy / controller"""
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
from pathlib import Path

import torch
from omegaconf import MISSING

from ppuu.costs.policy_costs_km import PolicyCostKMTaper
from ppuu.data.entities import DatasetSample
from ppuu.lightning_modules.policy.mpur import MPURModule, inject
from ppuu.modeling.forward_models import FwdCNN_VAE
from ppuu.modeling.forward_models_km_no_action import FwdCNNKMNoAction_VAE
from ppuu.modeling.policy.mpc import MPCKMPolicy
from ppuu.wrappers import ForwardModelKM


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelKM)
class MPURKMTaperModule(MPURModule):
    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold_km(
            self.policy_model,
            batch,
            augmenter=self.augmenter,
            npred=self.config.model.n_pred,
        )
        return predictions


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class MPURKMTaperV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "km_taper_v3"
        forward_model_path: str = MISSING

    @dataclass
    class TrainingConfig(MPURModule.TrainingConfig):
        detach_context: bool = False

    def build_log_dict(
        self, predictions: FwdCNN_VAE.Unfolding, cost: PolicyCostKMTaper.Cost
    ) -> dict:
        """Builds a dictionary of values to be logged from
        predictions and costs."""
        # we first use superclass' method, then add km cost specific stuff.
        result = super().build_log_dict(predictions, cost)
        result.update(
            {
                "reference_distance": cost.reference_distance.mean(),
                "speed": cost.speed.mean(),
                "destination": cost.destination.mean(),
            }
        )
        return result

    def calculate_cost(
        self, batch: DatasetSample, predictions: FwdCNN_VAE.Unfolding
    ) -> PolicyCostKMTaper.Cost:
        if self.config.training.detach_context:
            context = predictions.state_seq.map(lambda x: x.detach())
        else:
            context = predictions.state_seq

        return self.policy_cost.calculate_cost(
            batch.conditional_state_seq,
            predictions.state_seq.states,
            predictions.actions,
            context,
        )


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class GTMPURKMTaperV3Module(MPURModule):
    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold(
            self.policy_model,
            batch,
            augmenter=self.augmenter,
            npred=self.config.model.n_pred,
        )
        return predictions

    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "gt_km_taper_v3"
        forward_model_path: str = MISSING


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class MPURKMTaperV3Module_TargetProp(MPURKMTaperV3Module):
    """In this setup target prop is run as follows:
    1. Get an unfolding for the policy.
    2. Get mpc predictions of the same length as unfolding. Use
    unfolding for context.
    3. Penalize L2 distance between MPC output and policy output.
    """

    @dataclass
    class ModelConfig(MPURKMTaperV3Module.ModelConfig):
        model_type: str = "km_taper_v3_target_prop"
        mpc: MPCKMPolicy.Config = field(default_factory=MPCKMPolicy.Config)
        mpc_cost: PolicyCostKMTaper.Config = field(default_factory=PolicyCostKMTaper.Config)

    def _setup_mpc(self):
        self.mpc_cost = PolicyCostKMTaper(
            self.config.model.mpc_cost, self.forward_model, self.normalizer
        )
        self.mpc_cost.config.u_reg = 0
        self.mpc = MPCKMPolicy(
            self.forward_model,
            self.mpc_cost,
            self.normalizer,
            self.config.model.mpc,
        )

    def on_train_start(self):
        super().on_train_start()
        self._setup_mpc()

    def shared_step(
        self, batch: DatasetSample, predictions: FwdCNN_VAE.Unfolding
    ) -> Tuple[torch.Tensor, dict]:
        # Returns total loss and a dict to be logged.

        with torch.no_grad():
            self.mpc.reset()
            target_actions = self.mpc(
                batch.conditional_state_seq,
                gt_future=predictions.state_seq,
                full_plan=True,
            )
            loss = self.calculate_cost(batch, predictions)
            logged_losses = self.build_log_dict(predictions, loss)

        assert predictions.actions.shape == target_actions.shape, (
            f"expected policy actions and mpc actions to be of the same shape,"
            f"got {predictions.actions.shape=} and {target_actions.shape=}"
        )
        action_diff = (predictions.actions - target_actions) ** 2
        action_diff = self.policy_cost.apply_gamma(action_diff)

        # Sum across time and actions dimensions, mean over batch.
        action_diff = action_diff.sum(dim=-1).sum(dim=-1).mean()
        logged_losses["cloning_l2_loss"] = action_diff

        return action_diff, logged_losses

    def training_step(self, batch, batch_idx):
        batch = DatasetSample.from_tuple(batch)

        opt = self.optimizers()
        predictions = self(batch)

        loss, logs = self.shared_step(batch, predictions)

        for k, v in logs.items():
            self.log(
                "train/" + k,
                v,
                on_step=True,
                logger=True,
                prog_bar=True,
            )

        # We retain the gradient of actions to later log it to wandb.
        opt.zero_grad()
        predictions.actions.retain_grad()
        self.manual_backward(loss, optimizer=opt.optimizer)
        self.log_action_grads(predictions.actions.grad)
        self.clip_gradients()

        opt.step()

        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        batch = DatasetSample.from_tuple(batch)
        predictions = self(batch)
        loss = self.calculate_cost(batch, predictions)
        loss, logs = self.shared_step(batch, predictions)

        for k, v in logs.items():
            self.log(
                "val/" + k,
                v,
                on_epoch=True,
                logger=True,
            )

        return loss


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class MPURKMTaperV3Module_TargetPropOneByOne(MPURKMTaperV3Module_TargetProp):
    """In this setup, we run target prop as follows:
    For one particular dataset example:
    1. Unfold the future using the policy.
    2. Randomly, run MPC *separately* for some of the steps. Meaning
    MPC here takes as input the sequence including the unfolded states,
    and produces just one action.
    3. Optionally, dump the data into a file to read later.

    """

    @dataclass
    class ModelConfig(MPURKMTaperV3Module_TargetProp.ModelConfig):
        model_type: str = "km_taper_v3_target_prop_one_by_one"
        use_dumped: bool = False

    @dataclass
    class TrainingConfig(MPURKMTaperV3Module_TargetProp.TrainingConfig):
        dump_path: Optional[str] = None
        mpc_per_sample: int = 1
        mpc_first_only: bool = False
        use_gt_targets: bool = False

    def get_indices_to_annotate(self) -> List[int]:
        if self.config.training.mpc_first_only:
            return [0]
        else:
            return random.sample(
                range(self.config.model.n_pred),
                self.config.training.mpc_per_sample,
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.model.use_dumped:
            # Load the one episode labels. TBR
            mpc_actions_list = []
            for i in range(307):
                x = torch.load(
                    f"/home/us441/vlad_5/results/mpc/mpc_actions_2/{i}.t"
                )
                mpc_actions_list.append(x["mpc_actions"][0])
            self.mpc_actions = torch.stack(mpc_actions_list)

    def get_target_actions(
        self,
        batch: DatasetSample,
        predictions: FwdCNN_VAE.Unfolding,
        indices_to_annotate: List[int],
    ):
        # We first sample what steps we want to run mpc on.
        target_actions = []

        with torch.no_grad():
            conditional_state_seq = batch.conditional_state_seq
            for i in range(self.config.model.n_pred):
                if i in indices_to_annotate:
                    if not self.config.training.use_gt_targets:
                        # TBR
                        if (
                            i == 0
                            and (batch.episode_id == 2294).all().item()
                            and self.config.model.use_dumped
                        ):
                            result = []
                            for j in batch.timestep:
                                result.append(self.mpc_actions[j].cuda())
                            target_actions.append(torch.stack(result))
                        else:
                            self.mpc.reset()
                            target_actions.append(
                                self.mpc(
                                    conditional_state_seq,
                                )
                            )
                    else:
                        target_actions.append(batch.target_action_seq[:, i])
                # shift it by one
                conditional_state_seq = conditional_state_seq.shift_add(
                    predictions.state_seq.images[:, i],
                    predictions.state_seq.states[:, i],
                )

        return target_actions

    def dump_expert_annotation(
        self,
        batch: DatasetSample,
        predictions: FwdCNN_VAE.Unfolding,
        indices_to_annotate: List[int],
        mpc_actions: torch.Tensor,
    ):
        # Save to the path specified in config for using later.
        p = Path(
            f"{os.path.join(self.config.training.dump_path, str(batch.timestep))}.t"
        )
        p.parent.mkdir(exist_ok=True)

        torch.save(
            dict(
                mpc_actions=mpc_actions,
            ),
            p,
        )

    def shared_step(
        self,
        batch: DatasetSample,
        predictions: FwdCNN_VAE.Unfolding,
        save_result: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        # Returns total loss and a dict to be logged.
        indices_to_annotate = self.get_indices_to_annotate()

        target_actions = self.get_target_actions(
            batch, predictions, indices_to_annotate
        )

        if save_result and self.config.training.dump_path is not None:
            self.dump_expert_annotation(
                batch, predictions, indices_to_annotate, target_actions
            )

        with torch.no_grad():
            loss = self.calculate_cost(batch, predictions)
            logged_losses = self.build_log_dict(predictions, loss)

        total_action_diff = 0
        for i, a in zip(indices_to_annotate, target_actions):
            total_action_diff += (predictions.actions[:, i] - a) ** 2 / len(
                indices_to_annotate
            )
        # Sum across time and actions dimensions, mean over batch.
        total_action_diff = total_action_diff.sum(dim=-1).mean()

        logged_losses["cloning_l2_loss"] = total_action_diff
        return total_action_diff, logged_losses

    def training_step(self, batch, batch_idx):
        batch = DatasetSample.from_tuple(batch)
        opt = self.optimizers()
        predictions = self(batch)

        loss, logs = self.shared_step(batch, predictions, save_result=True)

        for k, v in logs.items():
            self.log(
                "train/" + k,
                v,
                on_step=True,
                logger=True,
                prog_bar=True,
            )

        # We retain the gradient of actions to later log it to wandb.
        opt.zero_grad()
        predictions.actions.retain_grad()
        self.manual_backward(loss, optimizer=opt.optimizer)
        self.log_action_grads(predictions.actions.grad)
        self.clip_gradients()
        opt.step()

        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        batch = DatasetSample.from_tuple(batch)
        predictions = self(batch)

        loss, logs = self.shared_step(batch, predictions)

        for k, v in logs.items():
            self.log(
                "val/" + k,
                v,
                on_step=True,
                logger=True,
                prog_bar=True,
            )

        return loss


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class MPURKMTaperV3Module_UrbanDriverLoss(MPURKMTaperV3Module_TargetProp):
    """Similarly to urban driver, we simulate what happens,
    and instead of pushing the actions to be the same as
    targets, we push states to be the same as targets.
    """

    @dataclass
    class ModelConfig(MPURKMTaperV3Module_TargetProp.ModelConfig):
        model_type: str = "km_taper_v3_urban_driver"

    @dataclass
    class TrainingConfig(MPURKMTaperV3Module_TargetProp.TrainingConfig):
        use_gt_targets: bool = False

    def get_target_states(
        self,
        batch: DatasetSample,
        predictions: FwdCNN_VAE.Unfolding,
    ):
        if self.config.training.use_gt_targets:
            return batch.target_state_seq.states
        else:
            with torch.no_grad():
                self.mpc.reset()
                target_actions = self.mpc(
                    batch.conditional_state_seq,
                    gt_future=predictions.state_seq,
                    full_plan=True,
                )
            target_states = self.mpc.unfold_km(
                batch.conditional_state_seq.states[:, -1], target_actions
            )
            return target_states

    def shared_step(
        self,
        batch: DatasetSample,
        predictions: FwdCNN_VAE.Unfolding,
        save_result: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        # Returns total loss and a dict to be logged.
        target_states = self.get_target_states(batch, predictions)

        with torch.no_grad():
            loss = self.calculate_cost(batch, predictions)
            logged_losses = self.build_log_dict(predictions, loss)

        total_states_diff = (
            (target_states[..., :2] - predictions.state_seq.states[..., :2])
            .pow(2)
            .mean()
        )

        logged_losses["states_l2_loss"] = total_states_diff
        return total_states_diff, logged_losses
