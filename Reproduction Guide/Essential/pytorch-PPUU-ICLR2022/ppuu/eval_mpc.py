import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch.multiprocessing
import yaml
import numpy as np
from omegaconf import MISSING

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa
os.environ["OMP_NUM_THREADS"] = "1"  # noqa

from ppuu import configs, slurm  # noqa
from ppuu.costs.policy_costs_km import PolicyCostKMTaper  # noqa
from ppuu.data import dataloader  # noqa
from ppuu.eval import PolicyEvaluator  # noqa
from ppuu.eval_mpc_visualizer import EvalVisualizer  # noqa
from ppuu.lightning_modules.fm import FM  # noqa
from ppuu.lightning_modules.policy import load_policy  # noqa
from ppuu.modeling.policy.mpc import MPCKMPolicy  # noqa


def get_optimal_pool_size():
    available_processes = len(os.sched_getaffinity(0))
    # we can't use more than 10, as in that case we don't fit into Gpu.
    optimal_pool_size = min(10, available_processes)
    return optimal_pool_size


@dataclass
class EvalMPCConfig(configs.ConfigBase):
    comment: str = ""
    dataset: str = MISSING
    save_gradients: bool = False
    debug: bool = False
    num_processes: int = -1
    output_dir: Optional[str] = None
    test_size_cap: Optional[int] = None
    slurm: bool = False
    cost_type: str = "km_taper"
    diffs: bool = False
    cost: PolicyCostKMTaper.Config = field(default_factory=PolicyCostKMTaper.Config)
    mpc: MPCKMPolicy.Config = field(default_factory=MPCKMPolicy.Config)
    visualizer: Optional[Any] = None
    forward_model_path: Optional[str] = None
    seed: int = 42
    dataset_partition: str = "test"
    dataset_one_episode: Optional[int] = None
    pass_gt_future: bool = False
    env_policy_path: Optional[str] = None
    old_stats_path: str = MISSING
    dummy: bool = False


def construct_mpc(config: EvalMPCConfig, stats: dict) -> MPCKMPolicy:
    if config.forward_model_path is not None:
        m_config = FM.Config()
        m_config.model.fm_type = "km_no_action"
        m_config.model.checkpoint = config.forward_model_path
        m_config.training.enable_latent = True
        m_config.training.diffs = config.diffs
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        forward_model = FM(m_config)
        forward_model._setup_normalizer(stats)
    else:
        forward_model = None

    if config.output_dir is not None:
        print("config", config)
        print("dict config", dataclasses.asdict(config))
        print(type(config))
        path = Path(config.output_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = path / "hparams.yml"
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)

    if config.visualizer == "dump":
        config.visualizer = EvalVisualizer(config.output_dir)

    normalizer = dataloader.Normalizer(stats)
    cost = PolicyCostKMTaper(config.cost, None, normalizer)
    policy = MPCKMPolicy(
        forward_model.model, cost, normalizer, config.mpc, config.visualizer
    )
    return policy


def main(config):
    logging.getLogger().setLevel(logging.INFO)

    if config.num_processes > 0:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    torch.manual_seed(config.seed)

    if config.dummy:
        test_dataset = dataloader.DummyFixedEvalDataset(config.dataset)
    else:
        test_dataset = dataloader.EvaluationDataset(
            config.dataset, config.dataset_partition, config.test_size_cap
        )

    if config.dataset_one_episode:
        # a workaround to limit the dataset to just one episode
        test_dataset.splits[config.dataset_partition] = [
            test_dataset.splits[config.dataset_partition][
                config.dataset_one_episode
            ]
        ]
        test_dataset.size = 1

    policy = construct_mpc(config, test_dataset.stats)

    if config.env_policy_path == "mpc":
        env_policy_model = policy
    else:
        env_policy_model = load_policy(
            config.env_policy_path, test_dataset.stats, config.old_stats_path
        )

    evaluator = PolicyEvaluator(
        test_dataset,
        config.num_processes,
        build_gradients=config.save_gradients,
        enable_logging=True,
        visualizer=config.visualizer,
        pass_gt_future=config.pass_gt_future,
        env_policy_model=env_policy_model,
    )
    result = evaluator.evaluate(policy, output_dir=config.output_dir)
    print(result["stats"])

    return result


if __name__ == "__main__":
    config = EvalMPCConfig.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor("mpc", 8, logs_path=config.output_dir)
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)
