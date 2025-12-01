"""An implementation of the policy proposed here:
https://arxiv.org/abs/1809.10732

Reference implementation:
https://github.com/woven-planet/l5kit/tree/master/l5kit/l5kit/planning/rasterized
"""
from dataclasses import dataclass, field
import dataclasses
from typing import Dict, Tuple

import torch
from torch import nn
import pytorch_lightning as pl
import torchvision

from ppuu import configs
from ppuu.data.entities import DatasetSample, StateSequence
from ppuu.data.dataloader import Normalizer
from ppuu.modeling.km import predict_states_diff, predict_states


@dataclass
class UberPolicyModelConfig(configs.ConfigBase):
    arch: str = "resnet50"
    raster_dim: int = 3
    state_dim: int = 5
    in_horizon: int = 20
    out_dim: int = 2
    out_horizon: int = 30
    pretrained: bool = False
    n_modes: int = 3
    alpha: float = 1.0
    beta: float = 1.0
    action_hinge_val: float = 5.0
    history_dropout_p: float = 0.5


@dataclass
class UberPolicyConfig(configs.ConfigBase):
    model: UberPolicyModelConfig = field(default_factory=UberPolicyModelConfig)
    training: configs.TrainingConfig = field(default_factory=configs.TrainingConfig)


class PredictionModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.model.arch == "resnet50":
            self.model = torchvision.models.resnet.resnet50(
                pretrained=self.config.model.pretrained
            )
            self.model.fc = torch.nn.Sequential()
            self.fc = torch.nn.Sequential(
                nn.Linear(
                    in_features=2048
                    + (
                        self.config.model.in_horizon
                        * self.config.model.state_dim
                    ),
                    out_features=4096,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(4096),
                nn.Linear(
                    in_features=4096,
                    out_features=(
                        self.config.model.out_horizon
                        * self.config.model.out_dim
                        + 1
                    )
                    * self.config.model.n_modes,
                ),
            )
            in_channels = (
                self.config.model.in_horizon * self.config.model.raster_dim
            )
            if in_channels != 3:
                self.model.conv1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            # TODO: implement input of states
        else:
            raise NotImplementedError(
                f"{self.config.model.arch}: unknown architecture"
            )

    def dropout_history(
        self,
        images: torch.Tensor,  # [batch_size, history_size, nchannels, 117, 24]
        states: torch.Tensor,  # [batch_size, history_size, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """When training, randomly drop out the history of images and states."""
        if not self.training:
            return images, states

        images_history, image_last = images[:, :-1], images[:, -1:]
        states_history, state_last = states[:, :-1], states[:, -1:]

        if torch.rand(1)[0] < self.config.model.history_dropout_p:
            images_history = torch.zeros_like(images_history)
        if torch.rand(1)[0] < self.config.model.history_dropout_p:
            states_history = torch.zeros_like(states_history)

        images = torch.cat([images_history, image_last], dim=1)
        states = torch.cat([states_history, state_last], dim=1)
        return images, states

    def forward(
        self,
        images: torch.Tensor,  # [batch_size, history_size, nchannels, 117, 24]
        states: torch.Tensor,  # [batch_size, history_size, state_dim]
    ) -> torch.Tensor:
        images, states = self.dropout_history(images, states)

        # [batch_size, history_size * nchannels, 117, 24]
        images = images.view(images.shape[0], -1, *images.shape[-2:])

        # [batch_size, 2048]
        x = self.model(images)

        # [batch_size, history_horizon * state_dim]
        states = states.view(states.shape[0], -1)

        # [batch_size, 2048 + history_horizon * state_dim]
        x = torch.cat([x, states], dim=1)

        x = self.fc(x)
        return x


class DisplacementErrorMetric:
    def __init__(self, normalizer: Normalizer, *, diffs: bool):
        self.normalizer = normalizer
        self.diffs = diffs

    def _convert_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.diffs:
            unnormalized = self.normalizer.unnormalize_states_diffs(states)
            cumulative = torch.cumsum(states[..., :2], dim=-2)
            return torch.cat([cumulative, states[..., 2:]], dim=-1)
        else:
            unnormalized = self.normalizer.unnormalize_states(states)
            return unnormalized

    def build_log_dict(
        self,
        # [b_size, n_modes, out_horizon, state_dim]
        pred_states: torch.Tensor,
        # [b_size, n_modes, out_horizon, state_dim]
        target_states: torch.Tensor,
    ) -> Dict[str, float]:
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
        # ([batch_size], [batch_size])
        best_avg, best_avg_indices = torch.min(avg_difference_norm, dim=-1)
        # ([batch_size], [batch_size])
        best_final, best_final_indices = torch.min(
            final_difference_norm, dim=-1
        )
        return {
            "ADE": best_avg.mean().item(),
            "FDE": best_final.mean().item(),
            "ADE_idx_std": best_avg_indices.float().std().item(),
            "FDE_idx_std": best_final_indices.float().std().item(),
        }


class UberPolicyModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.set_hparams(hparams)
        self.model = PredictionModel(self.config)

    def predict(
        self, conditional_state_seq: StateSequence
    ) -> torch.Tensor:  # [batch_size, pred_horizon, action_dim]
        """Assumes the input sequence is normalized."""

        # [batch_size, (out_dim * out_horizon + 1) * n_modes]
        output = self.model(
            conditional_state_seq.images, conditional_state_seq.states
        )

        # [batch_size, n_modes, (out_dim * out_horizon + 1)]
        output = output.view(output.shape[0], self.config.model.n_modes, -1)

        # [batch_size, n_modes]
        mode_logits = output[..., -1]

        # [batch_size, n_modes, out_horizon, out_idm]
        out_actions = output[..., :-1].view(
            output.shape[0],
            self.config.model.n_modes,
            self.config.model.out_horizon,
            self.config.model.out_dim,
        )

        return mode_logits, out_actions

    def actions_to_states_diffs(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,  # [b_size, n_modes, out_horizon, action_dim]
    ) -> torch.Tensor:  # [batch_size, n_modes, out_horizon, state_dim]
        pred_states_list = []
        # [batch_size, state_dim]
        states = conditional_state_seq.states[:, -1]

        # [batch_size * n_modes, state_dim]
        states = states.repeat_interleave(actions.shape[1], dim=0)

        # [batch_size * n_modes, out_horizon, action_dim]
        original_actions_shape = actions.shape
        actions = actions.view(-1, *actions.shape[2:])

        for i in range(self.config.model.out_horizon):
            if self.config.training.diffs:
                states = predict_states_diff(
                    states,
                    actions[:, i],
                    self.normalizer,
                )
            else:
                states = predict_states(
                    states,
                    actions[:, i],
                    self.normalizer,
                )

            pred_states_list.append(states)

        # [batch_size * n_modes, out_horizon, state_dim]
        pred_states = torch.stack(pred_states_list, dim=1)
        # [batch_size, n_modes, out_horizon, state_dim]
        pred_states = pred_states.view(*original_actions_shape[:3], -1)
        return pred_states

    @torch.no_grad()
    def assign_mode(
        self,
        # [batch_size, n_modes, out_horizon, state_dim]
        pred_states: torch.Tensor,
        # [batch_size, out_horizon, state_dim]
        target_states: torch.Tensor,
    ) -> torch.Tensor:  # [batch_size, n_modes] (one-hot in dim=-1)
        # [batch_size, n_modes, out_horizon, state_dim]
        target_states = target_states.unsqueeze(1).repeat_interleave(
            pred_states.shape[1], dim=1
        )
        diffs = torch.nn.functional.mse_loss(
            pred_states[..., :2], target_states[..., :2], reduction="none"
        ).mean(dim=(-1, -2))
        assignment = torch.argmin(diffs, dim=-1)
        return assignment

    def common_step(
        self, batch: DatasetSample, batch_idx: int, *, log_prefix: str
    ) -> torch.Tensor:
        # Let's feed stacked images into the model,
        # and make it output the states.
        # Or actions? Actions. We'll need that later
        # for actually running the simulation.

        # ([batch_size, n_modes], [batch_size, n_modes, out_horizon, out_idm])
        input_seq = batch.conditional_state_seq
        if self.config.model.raster_dim == 4:
            input_seq = input_seq.with_ego()

        mode_logits, out_actions = self.predict(input_seq)
        # [batch_size, n_modes, out_horizon, state_dim]
        out_states = self.actions_to_states_diffs(
            batch.conditional_state_seq, out_actions
        )

        target_states = batch.target_state_seq.states
        assignment = self.assign_mode(out_states, target_states)

        # [batch_size, n_modes, out_horizon, state_dim]
        rep_target_states = target_states.unsqueeze(1).repeat_interleave(
            out_states.shape[1], dim=1
        )
        loss_per_mode = torch.nn.functional.mse_loss(
            rep_target_states[..., :2], out_states[..., :2], reduction="none"
        ).mean(dim=(-1, -2))

        loss_best_mode = loss_per_mode[
            torch.arange(assignment.shape[0], device=out_states.device),
            assignment,
        ].mean()

        loss_assignment = torch.nn.functional.cross_entropy(
            mode_logits, assignment
        )

        abs_actions = out_actions.abs()
        hinge_loss = nn.functional.relu(
            abs_actions - self.config.model.action_hinge_val
        ).mean()
        loss_total = (
            self.config.model.alpha * loss_best_mode
            + loss_assignment
            + self.config.model.beta * hinge_loss
        )

        self.log(
            f"{log_prefix}/loss_best_mode",
            loss_best_mode.item(),
            on_step=(log_prefix == "train"),
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{log_prefix}/loss_assignment",
            loss_assignment.item(),
            on_step=(log_prefix == "train"),
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{log_prefix}/loss_hinge",
            hinge_loss.item(),
            on_step=(log_prefix == "train"),
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{log_prefix}/loss_total",
            loss_assignment.item(),
            on_step=(log_prefix == "train"),
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{log_prefix}/action_norm",
            torch.norm(out_actions, p=2, dim=-1).mean(),
            on_step=(log_prefix == "train"),
            logger=True,
            prog_bar=True,
        )
        metrics_logs = self.metric.build_log_dict(out_states, target_states)
        for k, v in metrics_logs.items():
            self.log(
                f"{log_prefix}/{k}",
                v,
                on_step=True,
                logger=True,
                prog_bar=True,
            )
        return loss_total

    def training_step(
        self, batch: DatasetSample, batch_idx: int
    ) -> torch.Tensor:
        return self.common_step(batch, batch_idx, log_prefix="train")

    def validation_step(self, batch: DatasetSample, batch_idx: int):
        self.common_step(batch, batch_idx, log_prefix="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{"params": self.model.parameters()}],
            self.config.training.learning_rate
            * self.config.training.gpus
            * self.config.training.num_nodes,
        )
        if self.config.training.scheduler is not None:
            if self.config.training.scheduler == "step":
                # we want to have 0.1 learning rate after 70% of training
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(self.config.training.n_epochs * 0.7),
                    gamma=0.1,
                )
            elif self.config.training.scheduler == "cosine":
                scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=int(self.config.training.n_epochs / 5)
                    )
                )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def set_hparams(self, hparams=None):
        if hparams is None:
            hparams = UberPolicyConfig()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
            self.config = UberPolicyConfig.parse_from_dict(hparams)
        else:
            self.hparams.update(dataclasses.asdict(hparams))
            self.config = hparams

    def on_train_start(self):
        """
        The setup below happens after everything was moved
        to the correct device, and after the data was loaded.
        """
        self._setup_normalizer()
        self._setup_metrics()
        print("train start")

    def _setup_normalizer(self, stats=None):
        if stats is None:
            stats = self.trainer.datamodule.data_store.stats
        self.normalizer = Normalizer(stats)

    def _setup_metrics(self):
        self.metric = DisplacementErrorMetric(
            self.normalizer, diffs=self.config.training.diffs
        )

    @property
    def sample_step(self):
        return (
            self.trainer.global_step
            * self.config.training.batch_size
            * self.config.training.num_nodes
            * self.config.training.gpus
        )


class EvaluationWrapper:
    """A wrapper around the model to be used in evaluation."""

    def __init__(self, module: UberPolicyModule):
        self.module = module

    @torch.no_grad()
    def __call__(
        self,
        conditional_state_seq: StateSequence,
        *,
        normalize_inputs: bool,
        normalize_outputs: bool,
    ):
        if normalize_inputs:
            n_states = conditional_state_seq.states
            if self.module.config.training.diffs:
                n_states = self.module.normalizer.states_to_diffs(n_states)
                n_states = self.module.normalizer.normalize_states_diffs(
                    n_states
                )
            else:
                n_states = self.module.normalizer.normalize_states(n_states)

            n_images = self.module.normalizer.normalize_images(
                conditional_state_seq.images
            )
            if (
                n_images.shape[-3] == 4
                and self.module.config.model.raster_dim == 3
            ):
                n_images = n_images[..., :3, :, :].contiguous()

            n_seq = StateSequence(
                n_images, n_states, *conditional_state_seq[2:]
            )
        else:
            n_seq = conditional_state_seq

        mode_logits, pred_actions = self.module.predict(n_seq)
        best_idx = torch.argmax(mode_logits, dim=1)
        best_actions = pred_actions[
            torch.arange(end=pred_actions.shape[0]), best_idx, 0
        ]

        if normalize_outputs:
            best_actions = self.module.normalizer.unnormalize_actions(
                best_actions
            )
        return best_actions
