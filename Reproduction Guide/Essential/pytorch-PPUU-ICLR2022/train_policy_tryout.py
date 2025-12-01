"""Train a policy / controller"""
import dataclasses
import os

import pytorch_lightning as pl
import torch.multiprocessing
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor

from ppuu import lightning_modules, slurm
from ppuu.data import NGSIMDataModule
from ppuu.train_utils import CustomLoggerWB


def main(config):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    config.training.auto_batch_size()

    if config.training.debug or config.training.fast_dev_run:
        config.training.epoch_size = 10
        config.training.n_epochs = 10
        config.cost.uncertainty_n_batches = 10

    module = lightning_modules.policy.get_module(config.model.model_type)
    datamodule = NGSIMDataModule(
        config.training.dataset,
        config.training.epoch_size,
        config.training.validation_size,
        config.training.batch_size,
        workers=0,
        diffs=config.training.diffs,
        n_episodes=config.training.n_episodes,
    )

    pl.seed_everything(config.training.seed)

    logger = CustomLoggerWB(
        save_dir=config.training.output_dir,
        experiment_name=config.training.experiment_name,
        seed=f"seed={config.training.seed}",
        version=config.training.version,
        project="PPUU_policy",
        offline=config.training.wandb_offline,
    )

    n_checkpoints = 5
    if config.training.n_steps is not None:
        n_checkpoints = max(n_checkpoints, int(config.training.n_steps / 1e5))

    period = max(1, config.training.n_epochs // n_checkpoints)
    print(f"training {period=} {logger.log_dir=}")

    trainer = pl.Trainer(
        accelerator="gpu" if config.training.gpus > 0 else "cpu",
        devices=config.training.gpus if config.training.gpus > 0 else "auto",
        num_nodes=config.training.num_nodes,
        max_epochs=config.training.n_epochs,
        check_val_every_n_epoch=period,
        num_sanity_val_steps=0,
        fast_dev_run=config.training.fast_dev_run,
        strategy=config.training.distributed_backend if config.training.distributed_backend != "auto" else "auto",
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(logger.log_dir, "checkpoints"),
                filename="{epoch}_{sample_step}",
                every_n_epochs=period,
                save_last=True,
            ),
        ],
        logger=logger,
        default_root_dir=logger.log_dir,
        # terminate_on_nan=True,
    )

    model = module(config)
    logger.log_hyperparams(model.hparams)
    trainer.fit(model, datamodule=datamodule)
    return model


if __name__ == "__main__":
    # Bypass command-line config parsing - load directly from YAML
    import yaml
    from omegaconf import OmegaConf
    
    print("=" * 80)
    print("TRYOUT: Policy Training with Manual Config")
    print("=" * 80)
    
    # Load base config from cfm-km.yaml
    with open("configs/cfm-km.yaml", "r") as f:
        base_config_dict = yaml.safe_load(f)
    
    # Add our specific parameters
    base_config_dict['training']['dataset'] = '/Users/skyyyyyyy/Desktop/Final Project/pytorch-PPUU-ICLR2022/traffic-data/processed/data_i80_v0'
    base_config_dict['training']['output_dir'] = 'results/policy'
    base_config_dict['training']['experiment_name'] = 'cfm_km_policy_tryout'
    base_config_dict['training']['n_steps'] = 100
    base_config_dict['training']['n_epochs'] = 1
    base_config_dict['training']['distributed_backend'] = 'auto'
    base_config_dict['training']['gpus'] = 1
    base_config_dict['training']['wandb_offline'] = True
    
    # Add forward model path
    if 'model' not in base_config_dict:
        base_config_dict['model'] = {}
    base_config_dict['model']['forward_model_path'] = '/Users/skyyyyyyy/Desktop/Final Project/pytorch-PPUU-ICLR2022/results/fm/fm/seed=42_9/checkpoints/last.ckpt'
    
    print("\nLoaded config:")
    print(yaml.dump(base_config_dict, default_flow_style=False, sort_keys=False))
    
    # Get the module FIRST to access nested config classes
    module = lightning_modules.policy.get_module('vanilla_v3')
    
    # Use the ConfigBase.parse_from_dict() method to properly create nested dataclasses
    print("\nCreating config using parse_from_dict...")
    config = module.Config.parse_from_dict(base_config_dict)
    
    print("\n✓ Config created successfully!")
    # Note: config.training etc are Field objects until accessed by the module
    # The module will properly instantiate them


    
    # Now run training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    main(config)
