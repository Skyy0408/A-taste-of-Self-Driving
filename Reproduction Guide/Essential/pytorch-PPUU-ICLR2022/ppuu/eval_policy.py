import os

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging  # noqa
from dataclasses import dataclass  # noqa
from typing import Optional, Union, Dict  # noqa

import torch  # noqa
import torch.multiprocessing  # noqa
from omegaconf import MISSING  # noqa

from ppuu import configs, slurm  # noqa
from ppuu.data import dataloader  # noqa
from ppuu.data.entities import StateSequence  # noqa
from ppuu.eval import PolicyEvaluator  # noqa
from ppuu.lightning_modules.policy import load_policy  # noqa


def get_optimal_pool_size():
    available_processes = len(os.sched_getaffinity(0))
    # we can't use more than 10, as in that case we don't fit into Gpu.
    optimal_pool_size = min(10, available_processes)
    return optimal_pool_size


@dataclass
class EvalConfig(configs.ConfigBase):
    checkpoint_path: Optional[str] = None
    alternative_checkpoint_path: Optional[str] = None
    dataset: str = MISSING
    dummy: bool = False
    save_gradients: bool = False
    debug: bool = False
    num_processes: int = -1
    output_dir: Optional[str] = None
    test_size_cap: Optional[int] = None
    slurm: bool = False
    model_type: Optional[str] = None
    diffs: bool = False
    env_policy_path: Optional[str] = None
    old_stats_path: str = MISSING
    slurm_constraint: Optional[str] = None
    dataset_partition: str = "train"
    use_gt_policy: bool = False

    def __post_init__(self):
        if self.num_processes == -1:
            self.num_processes = get_optimal_pool_size()
            logging.info(
                f"Number of processes wasn't speicifed, "
                f"going to use {self.num_processes}"
            )

        if self.output_dir is None:
            self.checkpoint_path = os.path.normpath(self.checkpoint_path)
            components = self.checkpoint_path.split(os.path.sep)
            components[-2] = "evaluation_results"
            self.output_dir = os.path.join(*components)
            if self.checkpoint_path[0] == os.path.sep:
                self.output_dir = os.path.sep + self.output_dir
            logging.info(
                f"Output dir wasn't specified, "
                f"going to save to {self.output_dir}"
            )


def main(config):
    if config.num_processes > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")

    if config.dummy:
        test_dataset = dataloader.DummyFixedEvalDataset(config.dataset)
    else:
        test_dataset = dataloader.EvaluationDataset(
            config.dataset, config.dataset_partition, config.test_size_cap
        )

    model = load_policy(
        config.checkpoint_path, test_dataset.stats, config.old_stats_path
    )

    evaluator = PolicyEvaluator(
        test_dataset,
        config.num_processes,
        build_gradients=config.save_gradients,
        enable_logging=True,
        env_policy_model=load_policy(
            config.env_policy_path, test_dataset.stats, config.old_stats_path
        ),
        use_gt_policy=config.use_gt_policy,
    )
    result = evaluator.evaluate(
        model,
        output_dir=config.output_dir,
        alternative_module=None,
    )
    print(result["stats"])
    return result


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    config = EvalConfig.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor(
            "eval", 8, constraint=config.slurm_constraint
        )
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)
