"""Training script for Uber's trajectory predictor"""
import os
import yaml
import dataclasses

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch

from ppuu.lightning_modules.policy import UberPolicyModule, UberPolicyConfig
from ppuu import slurm
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

    datamodule = NGSIMDataModule(
        config.training.dataset,
        config.training.epoch_size,
        config.training.validation_size,
        config.training.batch_size,
        npred=config.model.out_horizon,
        val_npred=config.model.out_horizon,
        workers=0,
        diffs=config.training.diffs,
    )

    pl.seed_everything(config.training.seed)

    logger = CustomLoggerWB(
        save_dir=config.training.output_dir,
        experiment_name=config.training.experiment_name,
        seed=f"seed={config.training.seed}",
        version=config.training.version,
        project="Uber",
        offline=config.training.wandb_offline,
    )

    trainer = pl.Trainer(
        gpus=config.training.gpus,
        num_nodes=config.training.num_nodes,
        max_epochs=config.training.n_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        fast_dev_run=config.training.fast_dev_run,
        distributed_backend=config.training.distributed_backend,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(logger.log_dir, "checkpoints"),
                filename="{epoch}_{sample_step}",
                period=1,
                save_last=True,
            ),
        ],
        logger=logger,
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        weights_save_path=logger.log_dir,
        # terminate_on_nan=True,
    )

    model = UberPolicyModule(config)
    logger.log_hyperparams(model.hparams)
    trainer.fit(model, datamodule=datamodule)
    return model


if __name__ == "__main__":
    config = UberPolicyConfig.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    print("parsed config")
    print(yaml.dump(dataclasses.asdict(config)))
    if use_slurm:
        executor = slurm.get_executor(
            job_name=config.training.experiment_name,
            cpus_per_task=4,
            nodes=config.training.num_nodes,
            gpus=config.training.gpus,
            constraint=config.training.slurm_constraint,
            logs_path=config.training.slurm_logs_path,
            prince=config.training.prince,
        )
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)
