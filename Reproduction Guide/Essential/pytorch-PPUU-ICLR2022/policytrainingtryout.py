"""Simplified Policy Training Script - Bypasses complex config system"""
import os
import dataclasses
import pytorch_lightning as pl
import torch.multiprocessing
from pytorch_lightning.callbacks import LearningRateMonitor

from ppuu import lightning_modules
from ppuu.data import NGSIMDataModule
from ppuu.train_utils import CustomLoggerWB


# Hardcoded configuration
@dataclasses.dataclass
class SimplifiedConfig:
    # Forward model
    forward_model_path: str = "/Users/skyyyyyyy/Desktop/Final Project/pytorch-PPUU-ICLR2022/results/fm/fm/seed=42_9/checkpoints/last.ckpt"
    
    # Dataset
    dataset: str = "/Users/skyyyyyyy/Desktop/Final Project/pytorch-PPUU-ICLR2022/traffic-data/processed/data_i80_v0"
    
    # Training parameters
    output_dir: str = "results/policy"
    experiment_name: str = "cfm_km_policy"
    n_steps: int = 100
    n_epochs: int = 1
    batch_size: int = 10
    epoch_size: int = 500
    validation_size: int = 25
    learning_rate: float = 0.0001
    gpus: int = 1
    num_nodes: int = 1
    distributed_backend: str = "auto"
    seed: int = 8888
    
    # Model parameters
    model_type: str = "vanilla_v3"
    n_cond: int = 20
    n_pred: int = 30
    diffs: bool = False
    
    # Cost parameters (from cfm-km.yaml)
    lambda_l: float = 0.2
    lambda_o: float = 1.0
    lambda_p: float = 1.0
    u_reg: float = 0.05
    gamma: float = 0.99


def main():
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    config = SimplifiedConfig()
    
    print("=" * 80)
    print("Simplified Policy Training")
    print("=" * 80)
    print(f"Forward Model: {config.forward_model_path}")
    print(f"Dataset: {config.dataset}")
    print(f"Output Dir: {config.output_dir}")
    print(f"Epochs: {config.n_epochs}, Steps: {config.n_steps}")
    print("=" * 80)
    
    # Get the policy module
    module = lightning_modules.policy.get_module(config.model_type)
    
    # Create config using OmegaConf structured config
    print("\nCreating module config...")
    from omegaconf import OmegaConf
    
    # Create a structured config from the dataclass
    module_config_omega = OmegaConf.structured(module.Config)
    
    # Now we can set values
    module_config_omega.training.dataset = config.dataset
    module_config_omega.training.output_dir = config.output_dir
    module_config_omega.training.experiment_name = config.experiment_name  
    module_config_omega.training.n_steps = config.n_steps
    module_config_omega.training.n_epochs = config.n_epochs
    module_config_omega.training.distributed_backend = config.distributed_backend
    module_config_omega.training.gpus = config.gpus
    module_config_omega.training.num_nodes = config.num_nodes
    module_config_omega.training.batch_size = config.batch_size
    module_config_omega.training.epoch_size = config.epoch_size
    module_config_omega.training.validation_size = config.validation_size
    module_config_omega.training.learning_rate = config.learning_rate
    module_config_omega.training.seed = config.seed
    module_config_omega.training.diffs = config.diffs
    
    # Update model parameters
    module_config_omega.model.forward_model_path = config.forward_model_path
    module_config_omega.model.n_cond = config.n_cond
    module_config_omega.model.n_pred = config.n_pred
    
    # Update cost parameters
    module_config_omega.cost.lambda_l = config.lambda_l
    module_config_omega.cost.lambda_o = config.lambda_o
    module_config_omega.cost.lambda_p = config.lambda_p
    module_config_omega.cost.u_reg = config.u_reg
    module_config_omega.cost.gamma = config.gamma
    
    # Convert back to dataclass
    module_config = module.Config(**OmegaConf.to_container(module_config_omega, resolve=True))
    
    print(f"\nConfig created successfully!")
    print(f"Training dataset: {module_config.training.dataset}")
    print(f"Forward model: {module_config.model.forward_model_path}")
    
    # Create datamodule
    datamodule = NGSIMDataModule(
        module_config.training.dataset,
        module_config.training.epoch_size,
        module_config.training.validation_size,
        module_config.training.batch_size,
        workers=0,
        diffs=module_config.training.diffs,
        n_episodes=getattr(module_config.training, 'n_episodes', None),
    )
    
    pl.seed_everything(module_config.training.seed)
    
    # Create logger
    logger = CustomLoggerWB(
        save_dir=module_config.training.output_dir,
        experiment_name=module_config.training.experiment_name,
        seed=f"seed={module_config.training.seed}",
        version=None,
        project="PPUU_policy",
        offline=True,
    )
    
    print(f"\nLogger directory: {logger.log_dir}")
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if module_config.training.gpus > 0 else "cpu",
        devices=module_config.training.gpus if module_config.training.gpus > 0 else "auto",
        num_nodes=module_config.training.num_nodes,
        max_epochs=module_config.training.n_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        fast_dev_run=False,
        strategy=module_config.training.distributed_backend if module_config.training.distributed_backend != "auto" else "auto",
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(logger.log_dir, "checkpoints"),
                filename="{epoch}_{sample_step}",
                every_n_epochs=1,
                save_last=True,
            ),
        ],
        logger=logger,
        default_root_dir=logger.log_dir,
    )
    
    # Create model
    print("\nInitializing policy model...")
    model = module(module_config)
    
    # Log hyperparameters
    logger.log_hyperparams(model.hparams)
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)
    
    print("\nTraining completed!")
    print(f"Checkpoints saved to: {logger.log_dir}/checkpoints/")
    return model


if __name__ == "__main__":
    main()
