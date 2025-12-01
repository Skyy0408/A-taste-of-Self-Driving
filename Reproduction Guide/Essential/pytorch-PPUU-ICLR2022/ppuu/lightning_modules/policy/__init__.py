import argparse
from typing import Optional
from pathlib import Path

import torch

from ppuu.modeling.policy.old_policy_wrapper import OldPolicyWrapper  # noqa

from ppuu.lightning_modules.policy.mpur import (
    MPURContinuousModule,
    MPURContinuousV3Module,
    MPURModule,
    MPURVanillaV3Module,
)
from ppuu.lightning_modules.policy.mpur_dreaming import (
    MPURDreamingLBFGSModule,
    MPURDreamingModule,
    MPURDreamingV3Module,
)
from ppuu.lightning_modules.policy.mpur_km import (
    MPURKMTaperModule,
    MPURKMTaperV3Module,
    MPURKMTaperV3Module_TargetProp,
    MPURKMTaperV3Module_TargetPropOneByOne,
    MPURKMTaperV3Module_UrbanDriverLoss,
)

from ppuu.lightning_modules.policy.uber import (
    UberPolicyModule,
    UberPolicyConfig,
)

MODULES_DICT = dict(
    vanilla=MPURModule,
    dreaming=MPURDreamingModule,
    dreaming_lbfgs=MPURDreamingLBFGSModule,
    km_taper=MPURKMTaperModule,
    km_taper_v3=MPURKMTaperV3Module,
    continuous=MPURContinuousModule,
    vanilla_v3=MPURVanillaV3Module,
    continuous_v3=MPURContinuousV3Module,
    dreaming_v3=MPURDreamingV3Module,
    km_taper_v3_target_prop=MPURKMTaperV3Module_TargetProp,
    km_taper_v3_target_prop_one_by_one=MPURKMTaperV3Module_TargetPropOneByOne,
    km_taper_v3_urban_driver=MPURKMTaperV3Module_UrbanDriverLoss,
)


def get_module_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        help="Pick the model type to run",
        required=True,
    )
    args, _ = parser.parse_known_args()
    return get_module(args.model_type)


def get_module(name):
    return MODULES_DICT[name]


def load_new_policy(path: Optional[str], stats) -> torch.nn.Module:
    if path is None:
        return None

    checkpoint = torch.load(path)

    suffix = ""
    if "model_config" in checkpoint["hyper_parameters"]:
        suffix = "_config"

    Module = get_module(
        checkpoint["hyper_parameters"][f"model{suffix}"]["model_type"]
    )

    mpur_module = Module(checkpoint["hyper_parameters"])
    mpur_module.cuda()
    mpur_module._setup_mixout()
    mpur_module.load_state_dict(checkpoint["state_dict"], strict=False)
    mpur_module.policy_model.diffs = checkpoint["hyper_parameters"][
        f"training{suffix}"
    ]["diffs"]
    mpur_module._setup_normalizer(stats)

    return mpur_module.policy_model


def load_policy(
    checkpoint_path: Optional[str],
    stats: Optional[dict] = None,
    old_stats_path: Optional[str] = None,
):
    if checkpoint_path is None:
        return None

    if Path(checkpoint_path).suffix == ".model":  # we're loading the old model
        assert old_stats_path is not None
        policy = OldPolicyWrapper(checkpoint_path, old_stats_path)
    else:
        assert stats is not None
        policy = load_new_policy(checkpoint_path, stats)
    return policy
