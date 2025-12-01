import torch


class OldPolicyWrapper:
    def __init__(self, model_path, stats_path):
        self.model = torch.load(model_path)["model"].policy_net.cuda()
        self.model.stats = {
            k: v.cuda() for k, v in torch.load(stats_path).items()
        }
        if 'offroad' in model_path: # hacky hackkk
            self.model.encoder.n_channels = 4
        else:
            self.model.encoder.n_channels = 3

    def __call__(
        self, conditional_state_seq, normalize_inputs, normalize_outputs
    ):
        states_coords = conditional_state_seq.states[..., :2]
        states_velocity = (
            conditional_state_seq.states[..., 2:-1]
            * conditional_state_seq.states[..., -1:]
        )
        states = torch.cat([states_coords, states_velocity], dim=-1)

        if self.model.encoder.n_channels == 3:
            images = torch.cat(
                [
                    conditional_state_seq.images[..., :2, :, :],
                    conditional_state_seq.images[..., -1:, :, :],
                ],
                dim=-3,
            )
        else:
            images = conditional_state_seq.images

        return self.model(
            images,
            states,
            normalize_inputs=normalize_inputs,
            normalize_outputs=normalize_outputs,
        )[0]

    def cuda(self):
        return self
