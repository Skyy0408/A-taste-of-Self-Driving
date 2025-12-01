from typing import Optional, Dict, NamedTuple

from dataclasses import dataclass  # noqa
import logging
from pathlib import Path
import time

import torch
from omegaconf import MISSING
from tqdm import tqdm

from ppuu import configs
from ppuu.data.dataloader import DataStore, Dataset

from ppuu.lightning_modules.policy import load_policy  # noqa
from ppuu.eval_mpc import EvalMPCConfig, construct_mpc
from ppuu.modeling.policy.old_policy_wrapper import OldPolicyWrapper


@dataclass
class Config(configs.ConfigBase):
    checkpoint_path: Optional[str] = None
    old_stats_path: Optional[str] = None
    dataset: str = MISSING
    debug: bool = False
    output_path: Optional[str] = None
    test_size_cap: int = 1000
    dataset_split: str = "test"
    mpc_config_path: Optional[str] = None


def load_mpc_or_policy(config: Config, stats: Dict):
    if config.mpc_config_path is None:
        policy = load_policy(config.checkpoint_path, stats, config.old_stats_path)
    else:
        mpc_config = EvalMPCConfig.parse_from_file(config.mpc_config_path)
        policy = construct_mpc(mpc_config, stats)
    return policy


class StateEvalInfo(NamedTuple):
    action: torch.Tensor  # shape [2]
    episode_id: int
    timestep: int
    execution_time: float


def main(config: Config):
    store = DataStore(config.dataset)
    ds = Dataset(
        store,
        "test",
        20,
        30,
        config.test_size_cap,
        shift=False,
        random_actions=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    policy = load_mpc_or_policy(config, store.stats)
    policy = policy.cuda()

    eval_infos = []

    for batch in tqdm(loader):
        t_start = time.time()
        if hasattr(policy, "reset"):
            policy.reset()
        a = (
            policy(
                batch.conditional_state_seq.with_ego().cuda(),
                normalize_inputs=True,
                normalize_outputs=False,
            )
            .detach()
            .cpu()
        )
        execution_time = time.time() - t_start
        eval_infos.append(
            StateEvalInfo(
                action=a[0],
                episode_id=batch.episode_id[0].item(),
                timestep=batch.timestep[0].item(),
                execution_time=execution_time,
            )
        )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(eval_infos, output_path)

    logging.info("Finished")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    config = Config.parse_from_command_line()
    main(config)
