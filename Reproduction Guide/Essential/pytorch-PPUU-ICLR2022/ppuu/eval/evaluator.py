import copy
import json
import logging
import math
import os
import time
from collections import deque, namedtuple
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gym  # noqa
import pandas as pd  # noqa
import torch  # noqa
import numpy as np  # noqa

from ppuu.data import dataloader  # noqa
from ppuu.data.entities import StateSequence  # noqa

MAX_ENV_QUEUE_SIZE = 5


class DummyExecutor:
    def submit(self, f, *args, **kwargs):
        return DummyResult(f(*args, **kwargs))

    def shutdown(self):
        pass


class DummyResult:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


class CarController:
    def __call__(
        state_seq: StateSequence,
        *,
        normalize_inputs: bool = False,
        normalize_outputs: bool = False,
    ) -> torch.Tensor:  # [2]
        raise NotImplementedError()


class PolicyEvaluator:
    def __init__(
        self,
        dataset: dataloader.EvaluationDataset,
        num_processes: int,
        build_gradients: bool = False,
        return_episode_data: bool = False,
        enable_logging: bool = True,
        rollback_seconds: int = 3,
        visualizer=None,
        pass_gt_future: bool = False,
        env_policy_model: CarController = None,
        use_gt_policy: bool = False,
    ):
        self.dataset = dataset
        self.build_gradients = build_gradients
        self.return_episode_data = return_episode_data
        self.num_processes = num_processes
        self.enable_logging = enable_logging
        self.rollback_seconds = rollback_seconds
        self.visualizer = visualizer
        self.normalizer = dataloader.Normalizer(dataset.stats)
        self.pass_gt_future = pass_gt_future
        self.env_policy_model = env_policy_model
        self.use_gt_policy = use_gt_policy

        i80_env_id = "I-80-v1"
        if i80_env_id not in gym.envs.registry:
            gym.envs.registration.register(
                id=i80_env_id,
                entry_point="ppuu.simulator.map_i80_ctrl:ControlledI80",
                kwargs=dict(
                    fps=10,
                    nb_states=20,
                    display=False,
                    delta_t=0.1,
                    store_simulator_video=False,
                    store=(env_policy_model is not None),
                    show_frame_count=False,
                    dataset_path=dataset.data_dir,
                    data_dir=dataset.data_dir,
                    env_policy_model=env_policy_model,
                ),
            )
        self.env = gym.make(i80_env_id, disable_env_checker=True)

    def get_performance_stats(self, results_per_episode):
        results_per_episode_df = pd.DataFrame.from_dict(
            results_per_episode, orient="index"
        )
        return dict(
            mean_distance=results_per_episode_df["distance_travelled"].mean(),
            mean_time=results_per_episode_df["time_travelled"].mean(),
            success_rate=results_per_episode_df["road_completed"].mean(),
            success_rate_without_fallback=results_per_episode_df[
                "road_completed_without_fallback"
            ].mean(),
            success_rate_alt=results_per_episode_df[
                "road_completed_alt"
            ].mean(),
            collision_rate=results_per_episode_df["has_collided"].mean(),
            collision_ahead_rate=results_per_episode_df[
                "has_collided_ahead"
            ].mean(),
            collision_behind_rate=results_per_episode_df[
                "has_collided_behind"
            ].mean(),
            off_screen_rate=results_per_episode_df["off_screen"].mean(),
            alternative_better=results_per_episode_df[
                "alternative_better"
            ].mean(),
            alternative_distance_diff=results_per_episode_df[
                "alternative_distance_diff"
            ].mean(),
            succeeded=int(results_per_episode_df["road_completed"].sum()),
            mean_proximity_cost=results_per_episode_df[
                "mean_proximity_cost"
            ].mean(),
            mean_pixel_proximity_cost=results_per_episode_df[
                "mean_pixel_proximity_cost"
            ].mean(),
            mean_lane_cost=results_per_episode_df["mean_lane_cost"].mean(),
        )

    def unfold(self, env, inputs, policy, car_size, t_limit=None, device="cpu"):
        Unfolding = namedtuple(
            "Unfolding",
            [
                "images",
                "states",
                "costs",
                "actions",
                "env_copies",
                "done",
                "has_collided",
                "off_screen",
                "road_completed",
                "road_completed_without_fallback",
                "has_collided_ahead",
                "has_collided_behind",
                "mean_proximity_cost",
                "mean_pixel_proximity_cost",
                "mean_lane_cost",
            ],
        )
        TimeCapsule = namedtuple("TimeCapsule", ["env", "inputs", "cost"])

        images, states, costs, actions = (
            [],
            [],
            [],
            [],
        )
        has_collided = False
        has_collided_ahead = False
        has_collided_behind = False
        off_screen = False
        road_completed = False
        road_completed_without_fallback = False
        done = False

        env_copies = deque(maxlen=self.rollback_seconds)

        t = 0

        if hasattr(policy, "reset"):
            policy.reset()

        total_cost = {
            "proximity_cost": 0.0,
            "pixel_proximity_cost": 0.0,
            "lane_cost": 0.0,
        }

        while not done:
            input_images = inputs["context"].contiguous()
            input_states = inputs["state"].contiguous()

            conditional_state_seq = StateSequence(
                images=input_images.unsqueeze(0),
                states=input_states.unsqueeze(0),
                car_size=torch.tensor(car_size).unsqueeze(0),
                ego_car_image=input_images.unsqueeze(0),
            ).to(device)

            if self.use_gt_policy:
                a = None
            else:
                if self.pass_gt_future:
                    a = policy(
                        conditional_state_seq=conditional_state_seq,
                        normalize_inputs=True,
                        normalize_outputs=True,
                        gt_future=lambda: self._get_future_with_no_action(
                            env, policy.config.unfold_len, car_size
                        ),
                    )
                else:
                    a = policy(
                        conditional_state_seq=conditional_state_seq,
                        normalize_inputs=True,
                        normalize_outputs=True,
                    )
                a = a.cpu().view(1, 2).numpy()[0]

            # Here we check if the mpc had to use forward model fallback.
            # If it did, we count this as a success.
            if hasattr(policy, "fm_fallback") and policy.fm_fallback:
                road_completed_without_fallback = True

            # env_copy = copy.deepcopy(self.env)
            inputs, cost, done, info = env.step(a)

            for k in total_cost:
                if k in cost:
                    total_cost[k] += cost[k]

            if info.collisions_per_frame > 0:
                has_collided = True
            if info.collisions_per_frame_ahead > 0:
                has_collided_ahead = True

            if info.collisions_per_frame_behind > 0:
                has_collided_behind = True

            if cost["arrived_to_dst"]:
                road_completed = True
                road_completed_without_fallback = True

            done = (
                done
                or (has_collided and has_collided_ahead)
                or road_completed
                or off_screen
            )

            if self.visualizer is not None:
                self.visualizer.update_main_image(
                    inputs["context"][-1].contiguous()
                )
                if hasattr(policy.cost, "t_image"):
                    self.visualizer.update_trajectory(
                        policy.cost.t_image.contiguous(),
                        policy.cost.t_image_data,
                    )
                if policy.cost.config.build_overlay:
                    self.visualizer.update_mask_overlay(
                        policy.cost.get_last_overlay()[
                            0
                        ]  # we just get the first
                    )
                if self.visualizer:
                    self.visualizer.update_cost_profile_and_traj(
                        *policy.cost.get_last_cost_profile_and_traj()
                    )
                self.visualizer.update_plot()

            # Every second, we save a copy of the environment.
            if t % 10 == 0:
                # need to remove lane surfaces because they're unpickleable
                env._lane_surfaces = dict()
                env_copies.append(
                    TimeCapsule(copy.deepcopy(env), inputs, cost)
                )
            t += 1

            off_screen = info.off_screen
            images.append(input_images[-1])
            states.append(input_states[-1])
            costs.append(cost)
            if a is not None:
                actions.append(
                    (
                        (torch.tensor(a[0]) - self.dataset.stats["a_mean"])
                        / self.dataset.stats["a_std"]
                    )
                )
            else:
                actions.append(torch.tensor([.0, .0]))

            if t_limit is not None and t >= t_limit:
                break

        for k in total_cost:
            total_cost[k] /= t

        images = torch.stack(images)
        states = torch.stack(states)
        actions = torch.stack(actions)
        return Unfolding(
            images,
            states,
            costs,
            actions,
            env_copies,
            done,
            has_collided,
            off_screen,
            road_completed,
            road_completed_without_fallback,
            has_collided_ahead,
            has_collided_behind,
            total_cost["proximity_cost"],
            total_cost["pixel_proximity_cost"],
            total_cost["lane_cost"],
        )

    def _get_future_with_no_action(self, env, t, car_size):
        """Build state and images for the future if all actions are 0"""
        env._lane_surfaces = dict()
        env = copy.deepcopy(env)
        images = []
        states = []
        for i in range(t):
            inputs, cost, done, info = env.step([0, 0])
            images.append(inputs["context"].contiguous()[-1])
            states.append(inputs["state"].contiguous()[-1])
            if done:
                return None  # we fall back to using the forward model in this case.
        images = torch.stack(images).unsqueeze(0)

        return StateSequence(
            images,
            torch.stack(states).unsqueeze(0),
            car_size=torch.tensor(car_size).unsqueeze(0),
            ego_car_image=images,
        ).cuda()

    def _build_episode_data(self, unfolding):
        return dict(
            action_sequence=unfolding.actions,
            state_sequence=unfolding.states,
            cost_sequence=unfolding.costs,
            images=(unfolding.images[:, :3] + unfolding.images[:, 3:]).clamp(
                max=255
            ),
            gradients=None,
        )

    def _build_result(self, unfolding):
        return dict(
            time_travelled=len(unfolding.images),
            distance_travelled=(
                unfolding.states[-1][0] - unfolding.states[0][0]
            ).item(),
            road_completed=(
                unfolding.road_completed and not unfolding.has_collided
            ),
            road_completed_without_fallback=(
                unfolding.road_completed_without_fallback
                and not unfolding.has_collided
            ),
            road_completed_alt=(
                unfolding.road_completed and not unfolding.has_collided_ahead
            ),
            off_screen=(
                unfolding.off_screen
                and not (
                    (unfolding.road_completed and not unfolding.has_collided)
                    or (
                        unfolding.road_completed
                        and not unfolding.has_collided_ahead
                    )
                )
            ),
            has_collided=unfolding.has_collided,
            has_collided_ahead=unfolding.has_collided_ahead,
            has_collided_behind=unfolding.has_collided_behind,
            mean_proximity_cost=unfolding.mean_proximity_cost,
            mean_pixel_proximity_cost=unfolding.mean_pixel_proximity_cost,
            mean_lane_cost=unfolding.mean_lane_cost,
        )

    def _process_one_episode(
        self,
        policy_model,
        policy_cost,
        car_info,
        index,
        output_dir,
        alternative_policy=None,
        device="cpu",
    ):
        # Move models to the correct device within the worker process
        policy_model.to(device)
        if alternative_policy is not None:
            alternative_policy.to(device)
        if output_dir is not None:
            episode_data_dir = os.path.join(output_dir, "episode_data")
            episode_output_path = os.path.join(episode_data_dir, str(index))
        else:
            episode_output_path = None

        if episode_output_path is None or not os.path.exists(
            episode_output_path
        ):
            if self.visualizer is not None:
                self.visualizer.episode_reset()

            inputs = self.env.reset(
                time_slot=car_info["time_slot"], vehicle_id=car_info["car_id"]
            )
            unfolding = self.unfold(
                self.env, inputs, policy_model, car_info["car_size"], device=device
            )
            alternative_unfolding = None
            if unfolding.has_collided and alternative_policy is not None:
                alternative_unfolding = self.unfold(
                    unfolding.env_copies[0].env,
                    unfolding.env_copies[0].inputs,
                    alternative_policy,
                    car_info["car_size"],
                    device=device,
                )

            result = self._build_result(unfolding)
            episode_data = self._build_episode_data(unfolding)
            episode_data["result"] = result

            if alternative_unfolding is not None:
                result["alternative"] = self._build_result(
                    alternative_unfolding
                )
                result["alternative_better"] = int(
                    not unfolding.road_completed
                    and alternative_unfolding.road_completed
                )
                alternative_distance = (
                    alternative_unfolding.states[-1][0]
                    - unfolding.states[0][0]
                ).item()
                result["alternative_distance_diff"] = (
                    alternative_distance - result["distance_travelled"]
                )
                episode_data["alternative"] = self._build_episode_data(
                    unfolding
                )
            else:
                result["alternative_better"] = math.nan
                result["alternative_distance_diff"] = math.nan

            if self.build_gradients:
                episode_data["gradients"] = policy_cost.get_grad_vid(
                    policy_model,
                    dict(
                        input_images=unfolding.images[:, :3].contiguous(),
                        input_states=unfolding.states,
                        car_sizes=torch.tensor(
                            car_info["car_size"], dtype=torch.float32
                        ),
                    ),
                )[0]

            if episode_output_path is not None:
                torch.save(episode_data, episode_output_path)

            if self.visualizer is not None:
                self.visualizer.save_video(index)

            result["index"] = index

            print(f"episode {index} success: {result['road_completed']}")

        else:
            episode_data = torch.load(episode_output_path, weights_only=False)
            result = episode_data["result"]
            print(
                f"loaded episode {index}, success: {result['road_completed']}"
            )

        if self.return_episode_data:
            result["episode_data"] = episode_data

        return self._to_cpu(result)

    def _to_cpu(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: self._to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._to_cpu(v) for v in obj)
        else:
            return obj

    def evaluate(
        self,
        module: torch.nn.Module,
        output_dir: Optional[str] = None,
        alternative_module: Optional[torch.nn.Module] = None,
    ):
        if output_dir is not None:
            os.makedirs(
                os.path.join(output_dir, "episode_data"), exist_ok=True
            )

        time_started = time.time()
        if self.num_processes > 0:
            executor = ProcessPoolExecutor(max_workers=self.num_processes)
        else:
            # executor = ThreadPoolExecutor(max_workers=1)
            executor = DummyExecutor()
        async_results = []

        # We create a copy of the cost module, but don't pass in the forward
        # model because we don't need it unless we calculate uncertainty.
        if self.build_gradients:
            policy_cost = module.CostType(
                module.config.cost,
                None,
                self.dataset.stats,
            )
        else:
            policy_cost = None

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        device_str = str(device)

        if hasattr(module, "policy_model"):
            # module.policy_model.to(device)  # Keep on CPU for pickling
            module.policy_model.stats = self.dataset.stats
            if alternative_module is not None:
                # alternative_module.policy_model.to(device) # Keep on CPU
                pass
            policy_model = module.policy_model
        else:
            policy_model = module

        for j, data in enumerate(self.dataset):
            # Debug: Check devices
            if j == 0:
                print(f"DEBUG: policy_model device: {next(policy_model.parameters()).device}")
                if hasattr(policy_cost, 'normalizer') and isinstance(policy_cost.normalizer, dict):
                     for k, v in policy_cost.normalizer.items():
                         if isinstance(v, torch.Tensor):
                             print(f"DEBUG: policy_cost.normalizer[{k}] device: {v.device}")
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            print(f"DEBUG: data[{k}] device: {v.device}")
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            print(f"DEBUG: data[{k}][0] device: {v[0].device}")

            async_results.append(
                executor.submit(
                    self._process_one_episode,
                    policy_model,
                    policy_cost,
                    data,
                    j,
                    output_dir,
                    alternative_policy=alternative_module.policy_model
                    if alternative_module is not None
                    else None,
                    device=device_str,
                )
            )

        results_per_episode = {}

        total_images = 0
        for j in range(len(async_results)):
            simulation_result = async_results[j].result()
            results_per_episode[j] = simulation_result
            total_images += simulation_result["time_travelled"]
            stats = self.get_performance_stats(results_per_episode)

            if self.enable_logging:
                log_string = " | ".join(
                    (
                        f"ep: {j + 1:3d}/{len(self.dataset)}",
                        f"time: {simulation_result['time_travelled']}",
                        (
                            f"distance:"
                            f" {simulation_result['distance_travelled']:.0f}"
                        ),
                        f"success: {simulation_result['road_completed']:d}",
                        f"success_without_fm_fallback: {simulation_result['road_completed_without_fallback']:d}",
                        f"success_alt: {simulation_result['road_completed_alt']:d}",
                        f"success rate: {stats['success_rate']:.2f}",
                        f"success rate without fallback: {stats['success_rate_without_fallback']:.2f}",
                        f"success rate alt: {stats['success_rate_alt']:.2f}",
                    )
                )
                logging.info(log_string)

        executor.shutdown()

        stats = self.get_performance_stats(results_per_episode)
        result = dict(
            results_per_episode=results_per_episode,
            stats=stats,
        )

        diff_time = time.time() - time_started
        eval_speed = total_images / diff_time
        result["stats"]["time"] = diff_time
        result["stats"]["steps evaluated per second"] = eval_speed

        if output_dir is not None:
            with open(
                os.path.join(output_dir, "evaluation_results_symbolic.json"),
                "w",
            ) as f:
                json.dump(result, f, indent=4, cls=NumpyJSONEncoder)

        return result

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)
