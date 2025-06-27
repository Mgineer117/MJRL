import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.layers.base import Base
from utils.sampler import OnlineSampler


# model-free policy trainer
class OnPolicyTrainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_timesteps: int,
        args,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = args.eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.episode_len = args.episode_len
        self.init_timesteps = init_timesteps
        self.timesteps = args.timesteps

        self.log_interval = args.log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_max_return_mean = 1e10
        self.last_min_return_std = 1e10

        self.rendering = args.rendering
        self.seed = args.seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0
        with tqdm(
            total=self.timesteps + self.init_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps + self.init_timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division
                self.policy.train()

                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )
                loss_dict, timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(timesteps)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = step + timesteps
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.policy.name}/analytics/avg_discounted_return"] = (
                    self.average_discounted_return(
                        batch["rewards"], batch["terminals"], self.policy.gamma
                    )
                )

                self.write_log(loss_dict, step=step)

                #### EVALUATIONS ####
                if step >= self.eval_interval * (eval_idx + 1):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, running_video = self.evaluate()

                    # Manual logging
                    if self.policy.state_visitation is not None:
                        visitation_map = self.policy.state_visitation
                        vmin, vmax = visitation_map.min(), visitation_map.max()
                        visitation_map = (visitation_map - vmin) / (vmax - vmin + 1e-8)
                        visitation_map = self.visitation_to_rgb(visitation_map)
                        self.write_image(
                            image=visitation_map,
                            step=step,
                            logdir="Image",
                            name="visitation map",
                        )

                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_video(
                        running_video,
                        step=step,
                        logdir=f"videos",
                        name="running_video",
                    )

                    self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                    self.last_return_std.append(eval_dict[f"eval/return_std"])

                    self.save_model(step)

                torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

    def evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            for t in range(self.episode_len):
                with torch.no_grad():
                    a, _ = self.policy(state, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    image = self.env.render()
                    image_array.append(image)

                next_state, rew, term, trunc, infos = self.env.step(a)
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(
                                ep_reward, self.policy.gamma
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

    def average_discounted_return(self, rewards, terminals, gamma):
        episode_returns = []
        G = 0.0
        for r, done in zip(rewards, terminals):
            G = r + gamma * G
            if done:
                episode_returns.append(G)
                G = 0.0  # Reset for next episode

        if not episode_returns:
            return 0.0
        return sum(episode_returns) / len(episode_returns)

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
        image_list = [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        if len(image) > 0:
            tensor = np.stack(image, axis=0)
            video_path = os.path.join(logdir, name)
            self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    def save_model(self, e):
        ### save checkpoint
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        model = self.policy.actor

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
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
