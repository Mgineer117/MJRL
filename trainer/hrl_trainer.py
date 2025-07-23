import gc
import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.elementary_policy.uniform_random import UniformRandom
from policy.layers.base import Base
from trainer.base_trainer import BaseTrainer
from utils.rl import estimate_advantages
from utils.sampler import OnlineSampler


def compare_weights(policy1, policy2):
    diffs = {}
    for (name1, param1), (name2, param2) in zip(
        policy1.named_parameters(), policy2.named_parameters()
    ):
        assert name1 == name2, "Parameter names do not match"
        diff = torch.norm(param1.data - param2.data).item()
        diffs[name1] = diff
    return diffs


# on-policy trainer
class HRLOnPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        hl_policy: Base,
        policies: Base,
        intrinsic_reward_fn,
        hl_sampler: OnlineSampler,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_timesteps: int,
        args,
    ) -> None:
        self.env = env
        self.hl_policy = hl_policy
        self.policies = policies

        self.intrinsic_reward_fn = intrinsic_reward_fn

        self.trainable_options = [
            i for i, policy in enumerate(self.policies) if hasattr(policy, "actor")
        ]
        self.num_trainable_options = len(self.trainable_options)

        self.hl_sampler = hl_sampler
        self.sampler = sampler

        self.eval_num = args.eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_timesteps = init_timesteps
        self.timesteps = args.timesteps
        self.hl_timesteps = args.hl_timesteps

        self.log_interval = args.log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)
        self.hl_eval_interval = int(self.hl_timesteps / self.log_interval)

        # initialize the essential training components
        self.last_max_return_mean = -1e10
        self.last_min_return_std = 1e10

        self.episode_len = args.episode_len
        self.rendering = args.rendering
        self.seed = args.seed

        self.args = args

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0
        total_timesteps = int(
            self.timesteps * self.num_trainable_options + self.init_timesteps
        )
        with tqdm(
            total=total_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.hl_policy.name} Training (Timesteps)",
        ) as pbar:
            for option_idx in self.trainable_options:
                while pbar.n < int(
                    (option_idx + 1) * (self.timesteps + self.init_timesteps)
                ):
                    # --- START OF EPOCH/ITERATION ---
                    current_step = pbar.n

                    policy = self.policies[option_idx]
                    policy.train()

                    # === Initial Iteration ===
                    batch, sample_time = self.sampler.collect_samples(
                        env=self.env,
                        policy=policy,
                        seed=self.seed,
                    )

                    # classify the option_idx needs intrinsic rewards
                    if policy.name == "HRL_intrinsic_options":
                        states, next_states = batch["states"], batch["next_states"]
                        intrinsic_rewards = self.intrinsic_reward_fn(
                            states, next_states, option_idx
                        )
                        batch["rewards"] = intrinsic_rewards.cpu().numpy()
                    loss_dict, timesteps, update_time = policy.learn(batch)

                    # add timesteps
                    current_step += timesteps
                    pbar.update(timesteps)

                    # Calculate expected remaining time
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / current_step
                    remaining_time = avg_time_per_iter * (
                        total_timesteps - current_step
                    )

                    # Update environment steps and calculate time metrics
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/timesteps"
                    ] = (current_step + timesteps)
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/sample_time"
                    ] = sample_time
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/update_time"
                    ] = update_time
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/remaining_time (hr)"
                    ] = (
                        remaining_time / 3600
                    )  # Convert to hours
                    loss_dict[f"{self.policies[0].name}/analytics/return"] = (
                        self.average_discounted_return(
                            batch["rewards"], batch["terminals"], self.hl_policy.gamma
                        )
                    )

                    self.write_log(loss_dict, step=current_step)

                    #### EVALUATIONS ####
                    if current_step >= self.eval_interval * eval_idx:
                        ### Eval Loop ###
                        self.policies[option_idx].eval()
                        eval_idx += 1

                        eval_dict, running_video = self.evaluate(option_idx)

                        # Manual logging
                        if self.policies[option_idx].state_visitation is not None:
                            visitation_map = self.policies[option_idx].state_visitation
                            vmin, vmax = visitation_map.min(), visitation_map.max()
                            visitation_map = (visitation_map - vmin) / (
                                vmax - vmin + 1e-8
                            )
                            visitation_map = self.visitation_to_rgb(visitation_map)
                            self.write_image(
                                image=visitation_map,
                                step=current_step,
                                logdir="Image",
                                name=f"sub-policy visitation map {option_idx}",
                            )

                        self.write_log(eval_dict, step=current_step, eval_log=True)
                        self.write_video(
                            running_video,
                            step=current_step,
                            logdir=f"Video",
                            name=f"sub-policy running_video {option_idx}",
                        )

                        self.save_model(
                            current_step,
                            self.policies[option_idx],
                            f"sub-policy {option_idx}",
                        )

        # assign trained option policies
        eval_idx = 0
        init_timesteps = current_step
        total_tiemesteps = init_timesteps + self.hl_timesteps
        self.hl_policy.update_options(self.policies)
        with tqdm(
            total=total_tiemesteps,
            initial=init_timesteps,
            desc=f"{self.hl_policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total_tiemesteps:
                current_step = pbar.n
                self.hl_policy.train()

                batch, sample_time = self.hl_sampler.collect_samples(
                    env=self.env, policy=self.hl_policy, seed=self.seed
                )
                loss_dict, timesteps, update_time = self.hl_policy.learn(batch)

                # add timesteps
                current_step += timesteps
                pbar.update(timesteps)

                # Calculate expected remaining time
                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / current_step
                remaining_time = avg_time_per_iter * (total_tiemesteps - current_step)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.hl_policy.name}/analytics/timesteps"] = (
                    current_step + timesteps
                )
                loss_dict[f"{self.hl_policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.hl_policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.hl_policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=current_step)

                #### EVALUATIONS ####
                if current_step >= self.hl_eval_interval * (eval_idx + 1):
                    ### Eval Loop ###
                    self.hl_policy.eval()
                    eval_idx += 1

                    eval_dict, running_video = self.hl_evaluate()

                    # Manual logging
                    if self.hl_policy.state_visitation is not None:
                        visitation_map = self.hl_policy.state_visitation
                        vmin, vmax = visitation_map.min(), visitation_map.max()
                        visitation_map = (visitation_map - vmin) / (vmax - vmin + 1e-8)
                        visitation_map = self.visitation_to_rgb(visitation_map)
                        self.write_image(
                            image=visitation_map,
                            step=current_step,
                            logdir="Image",
                            name="visitation map",
                        )

                    self.write_log(eval_dict, step=current_step, eval_log=True)
                    self.write_video(
                        running_video,
                        step=current_step,
                        logdir=f"Video",
                        name="running_video",
                    )

                    self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                    self.last_return_std.append(eval_dict[f"eval/return_std"])

                    self.save_model(current_step, self.hl_policy, "hl_policy")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.hl_policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

        return current_step

    def evaluate(self, option_idx: int):
        policy = self.policies[option_idx]

        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            for t in range(self.episode_len):
                with torch.no_grad():
                    a, _ = policy(state, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    image = self.env.render()
                    image_array.append(image)

                next_state, _, term, trunc, infos = self.env.step(a)
                rew = (
                    self.intrinsic_reward_fn(state, next_state, option_idx)
                    .cpu()
                    .numpy()
                )
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(ep_reward, policy.gamma),
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)

        eval_dict = {
            f"eval/sub-policy intrinsic return_mean {option_idx}": return_mean,
            f"eval/sub-policy intrinsic return_std {option_idx}": return_std,
        }

        return eval_dict, image_array

    def hl_evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            for t in range(self.episode_len):
                with torch.no_grad():
                    [option_idx, a], metaData = self.hl_policy(
                        state, None, deterministic=True
                    )
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    image = self.env.render()
                    image_array.append(image)

                if metaData["is_option"]:
                    option_termination = False
                    for i in range(10):
                        next_state, rew, term, trunc, infos = self.env.step(a)
                        done = term or trunc
                        ep_reward.append(rew)

                        if done or option_termination:
                            break
                        else:
                            with torch.no_grad():
                                [_, a], optionMetaData = self.hl_policy(
                                    next_state,
                                    option_idx=option_idx,
                                    deterministic=True,
                                )
                                a = (
                                    a.cpu().numpy().squeeze(0)
                                    if a.shape[-1] > 1
                                    else [a.item()]
                                )
                            option_termination = optionMetaData["option_termination"]
                else:
                    # env stepping
                    next_state, rew, term, trunc, infos = self.env.step(a)
                    done = term or trunc
                    ep_reward.append(rew)

                state = next_state

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(
                                ep_reward, self.hl_policy.gamma
                            ),
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
        }

        return eval_dict, image_array

    def discounted_return(self, rewards, gamma):
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = image if isinstance(image, list) else [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        if len(image) > 0:
            tensor = np.stack(image, axis=0)
            video_path = os.path.join(logdir, name)
            self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    def save_model(self, e: int, model: nn.Module, name: str):
        ### save checkpoint
        name = f"{name}_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
            if len(self.last_return_mean) > 0 and len(self.last_return_std) > 0:
                if (
                    np.mean(self.last_return_mean) >= self.last_max_return_mean
                    and np.mean(self.last_return_std) <= self.last_min_return_std
                ):
                    name = f"best_model.pth"
                    path = os.path.join(self.logger.log_dir, name)
                    torch.save(model.state_dict(), path)

                    self.last_max_return_mean = np.mean(self.last_return_mean)
                    self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")

    def visitation_to_rgb(self, visitation_map: np.ndarray) -> np.ndarray:
        visitation_map = np.squeeze(visitation_map)  # Make sure it's 2D
        H, W = visitation_map.shape

        rgb_map = np.ones((H, W, 3), dtype=np.float32)  # Start with white

        # Zero visitation → gray
        zero_mask = visitation_map == 0
        rgb_map[zero_mask] = [0.5, 0.5, 0.5]

        # Nonzero visitation → white → blue gradient
        nonzero_mask = visitation_map > 0
        blue_intensity = visitation_map[nonzero_mask]

        rgb_map[nonzero_mask] = np.stack(
            [
                1.0 - blue_intensity,  # Red
                1.0 - blue_intensity,  # Green
                np.ones_like(blue_intensity),  # Blue
            ],
            axis=-1,
        )

        return rgb_map


# off-policy trainer
from utils.replay_buffer import ReplayBuffer


class HRLOffPolicyTrainer(HRLOnPolicyTrainer):
    def __init__(self, replay_buffer: ReplayBuffer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.random_policy = UniformRandom(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )

        self.hl_replay_buffer = deepcopy(replay_buffer)
        self.replay_buffers = [
            deepcopy(replay_buffer) for _ in range(self.num_trainable_options)
        ]

        self.warmup_samples = self.args.warmup_samples

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0
        total_timesteps = int(self.timesteps + self.init_timesteps)
        with tqdm(
            total=total_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.hl_policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total_timesteps:
                current_step = pbar.n + 1  # + 1 to avoid zero division

                # selecting a policy
                if current_step < self.warmup_samples:
                    policy = self.random_policy
                else:
                    # randomly select given self.policies
                    option_idx = np.random.choice(self.trainable_options)
                    policy = self.policies[option_idx]

                ### === Env initialization === ###
                ep_reward = []
                state, infos = self.env.reset(seed=self.seed)
                for t in range(self.episode_len):
                    with torch.no_grad():
                        a, _ = policy(state, deterministic=False)
                        action = (
                            a.cpu().numpy().squeeze(0)
                            if a.shape[-1] > 1
                            else [a.item()]
                        )

                    next_state, reward, term, trunc, infos = self.env.step(action)
                    if t == self.episode_len - 1:
                        # safe truncation
                        trunc = True
                    done = term or trunc

                    # === SAVE THE DATA === #
                    for i, idx in enumerate(self.trainable_options):
                        if self.policies[idx].name == "HRL_intrinsic_options":
                            reward = (
                                self.intrinsic_reward_fn(state, next_state, idx)
                                .cpu()
                                .numpy()
                            )

                        if current_step >= self.warmup_samples:
                            if idx == option_idx:
                                # when random sampling we don't record the return
                                ep_reward.append(reward)

                        self.replay_buffers[i].append(
                            state, action, next_state, reward, done
                        )

                    # === UPDATE STATE === #
                    state = next_state
                    pbar.update(1)

                    # === UPDATE THE ALL POLICY === #
                    if current_step >= self.warmup_samples:
                        total_update_time = 0
                        loss_dict_list = []
                        for i, idx in enumerate(self.trainable_options):
                            loss_dict, update_time = self.policies[idx].learn(
                                self.replay_buffers[i]
                            )
                            total_update_time += update_time
                            loss_dict_list.append(loss_dict)

                        loss_dict = self.average_dict_values(loss_dict_list)
                        loss_dict[f"{policy.name}/analytics/update_time"] = (
                            total_update_time
                        )
                        self.write_log(loss_dict, step=current_step)

                    if done:
                        if current_step >= self.warmup_samples:
                            return_dict = {
                                f"{policy.name}/sub-policy intrinsic return {option_idx}": self.discounted_return(
                                    ep_reward, self.args.gamma
                                ),
                            }
                            self.write_log(return_dict, step=current_step)
                        break

                    #### EVALUATIONS ####
                    if current_step >= self.warmup_samples:
                        if (
                            current_step - self.warmup_samples
                            >= self.eval_interval * eval_idx
                        ):
                            ### Eval Loop ###
                            for i, idx in enumerate(self.trainable_options):
                                self.policies[idx].eval()

                                eval_dict, running_video = self.evaluate(idx)

                                self.write_log(eval_dict, step=current_step)
                                self.write_video(
                                    running_video,
                                    step=current_step,
                                    logdir=f"Video",
                                    name=f"sub-policy running_video {idx}",
                                )
                                self.save_model(
                                    current_step,
                                    self.policies[idx],
                                    f"sub-policy {idx}",
                                )

                                self.policies[idx].train()

                            eval_idx += 1

        ### === CLEAR PREVIOUS REPLAY BUFFER === ###
        del self.replay_buffers

        # assign trained option policies
        eval_idx = 0
        init_timesteps = current_step
        total_tiemesteps = init_timesteps + self.hl_timesteps
        self.hl_policy.update_options(self.policies)
        with tqdm(
            total=total_tiemesteps,
            initial=init_timesteps,
            desc=f"{self.hl_policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total_tiemesteps:
                current_step = pbar.n + 1  # + 1 to avoid zero division
                self.hl_policy.train()

                # Env initialization
                state, infos = self.env.reset(seed=self.seed)

                for t in range(self.episode_len):
                    with torch.no_grad():
                        [option_idx, a], metaData = self.hl_policy(
                            state, None, deterministic=True
                        )
                        a = (
                            a.cpu().numpy().squeeze(0)
                            if a.shape[-1] > 1
                            else [a.item()]
                        )

                    if metaData["is_option"]:
                        option_termination = False
                        for i in range(10):
                            next_state, rew, term, trunc, infos = self.env.step(a)
                            done = term or trunc
                            ep_reward.append(rew)

                            if done or option_termination:
                                break
                            else:
                                with torch.no_grad():
                                    [_, a], optionMetaData = self.hl_policy(
                                        next_state,
                                        option_idx=option_idx,
                                        deterministic=True,
                                    )
                                    a = (
                                        a.cpu().numpy().squeeze(0)
                                        if a.shape[-1] > 1
                                        else [a.item()]
                                    )
                                option_termination = optionMetaData[
                                    "option_termination"
                                ]
                    else:
                        # env stepping
                        next_state, rew, term, trunc, infos = self.env.step(a)
                        done = term or trunc
                        ep_reward.append(rew)

                        self.hl_replay_buffer.append(
                            state, action, next_state, reward, done
                        )

                    # === UPDATE POLICY === #
                    if current_step >= init_timesteps + self.warmup_samples:
                        loss_dict, update_time = self.hl_policy.learn(
                            self.hl_replay_buffer
                        )
                        loss_dict[f"{self.hl_policy.name}/analytics/update_time"] = (
                            update_time
                        )

                        self.write_log(loss_dict, step=current_step)

                    # === UPDATE STATE === #
                    state = next_state
                    pbar.update(1)

                    if done:
                        if current_step >= init_timesteps + self.warmup_samples:
                            return_dict = {
                                f"{self.hl_policy.name}/return": self.discounted_return(
                                    ep_reward, self.args.gamma
                                ),
                            }
                            self.write_log(return_dict, step=current_step)
                        break

                #### EVALUATIONS ####
                if current_step >= init_timesteps + self.warmup_samples:
                    if (
                        current_step - (init_timesteps + self.warmup_samples)
                        >= self.hl_eval_interval * eval_idx
                    ):
                        ### Eval Loop ###
                        self.hl_policy.eval()
                        eval_idx += 1

                        eval_dict, running_video = self.hl_evaluate()

                        # Manual logging
                        if self.hl_policy.state_visitation is not None:
                            visitation_map = self.hl_policy.state_visitation
                            vmin, vmax = visitation_map.min(), visitation_map.max()
                            visitation_map = (visitation_map - vmin) / (
                                vmax - vmin + 1e-8
                            )
                            visitation_map = self.visitation_to_rgb(visitation_map)
                            self.write_image(
                                image=visitation_map,
                                step=current_step,
                                logdir="Image",
                                name="visitation map",
                            )

                        self.write_log(eval_dict, step=current_step, eval_log=True)
                        self.write_video(
                            running_video,
                            step=current_step,
                            logdir=f"Video",
                            name="running_video",
                        )

                        self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                        self.last_return_std.append(eval_dict[f"eval/return_std"])

                        self.save_model(current_step, self.hl_policy, "hl_policy")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.hl_policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

        return current_step
