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
from policy.uniform_random import UniformRandom
from trainer.base_trainer import BaseTrainer
from utils.replay_buffer import ReplayBuffer


# model-free policy trainer
class OffPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        replay_buffer: ReplayBuffer,
        logger: WandbLogger,
        writer: SummaryWriter,
        args,
    ) -> None:
        self.env = env
        self.random_policy = UniformRandom(
            state_dim=policy.actor.state_dim,
            action_dim=policy.actor.action_dim,
            is_discrete=policy.actor.is_discrete,
            device=policy.device,
        )
        self.policy = policy
        self.eval_num = args.eval_num

        self.replay_buffer = replay_buffer
        self.logger = logger
        self.writer = writer

        # training parameters
        self.episode_len = args.episode_len
        self.timesteps = args.timesteps
        self.warmup_samples = args.warmup_samples

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
            total=self.timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division
                self.policy.train()

                policy = (
                    self.random_policy if step < self.warmup_samples else self.policy
                )

                # Env initialization
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

                    self.replay_buffer.append(state, action, next_state, reward, done)

                    if step >= self.warmup_samples:
                        loss_dict, update_time = policy.learn(self.replay_buffer)

                    state = next_state
                    pbar.update(1)

                    if done:

                        break

                if step >= self.warmup_samples:
                    # Update environment steps and calculate time metrics
                    loss_dict[f"{self.policy.name}/analytics/timesteps"] = step
                    loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
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
                            visitation_map = (visitation_map - vmin) / (
                                vmax - vmin + 1e-8
                            )
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
                np.mean(self.last_return_mean) < self.last_max_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_max_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")
