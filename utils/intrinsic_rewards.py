import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from utils.rl import call_env
from utils.sampler import OnlineSampler


class IntrinsicRewardFunctions(nn.Module):
    def __init__(self, logger, writer, args):
        super(IntrinsicRewardFunctions, self).__init__()

        # === Parameter saving === #
        self.episode_len_for_sampling = 200_000
        self.num_trials = 10

        self.extractor_env = call_env(deepcopy(args), self.episode_len_for_sampling)
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        self.num_rewards = self.args.num_options
        self.extractor_mode = "allo"
        self.define_extractor()
        self.define_eigenvectors()
        self.define_intrinsic_reward_normalizer()

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, i: int):
        states = states[:, self.args.positional_indices]
        next_states = next_states[:, self.args.positional_indices]

        with torch.no_grad():
            feature, _ = self.extractor(states)
            next_feature, _ = self.extractor(next_states)
            difference = next_feature - feature

            eigenvector_idx, eigenvector_sign = self.eigenvectors[i]
            intrinsic_rewards = eigenvector_sign * difference[
                :, eigenvector_idx
            ].unsqueeze(-1)

        # === INTRINSIC REWARD NORMALIZATION === #
        if hasattr(self, "reward_rms") and self.sources[i] != "drnd":
            # drnd has its own normalizer in itself
            self.reward_rms[i].update(intrinsic_rewards.cpu().numpy())
            var_tensor = torch.as_tensor(
                self.reward_rms[i].var,
                device=intrinsic_rewards.device,
                dtype=intrinsic_rewards.dtype,
            )
            intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

        return intrinsic_rewards, self.sources[i]

    def define_extractor(self):
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer
        from utils.rl import get_extractor

        if not os.path.exists("model"):
            os.makedirs("model")

        model_path = f"model/{self.args.env_name}-{self.extractor_mode}-{self.args.num_random_agents}-extractor.pth"
        extractor = get_extractor(self.args)

        if not os.path.exists(model_path):
            uniform_random_policy = UniformRandom(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                is_discrete=self.args.is_discrete,
                device=self.args.device,
            )
            sampler = OnlineSampler(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                episode_len=self.episode_len_for_sampling,
                batch_size=self.num_trials * self.episode_len_for_sampling,
                verbose=False,
            )
            trainer = ExtractorTrainer(
                env=self.extractor_env,
                random_policy=uniform_random_policy,
                extractor=extractor,
                sampler=sampler,
                logger=self.logger,
                writer=self.writer,
                epochs=self.args.extractor_epochs,
            )
            final_timesteps = trainer.train()
            self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)
        else:
            extractor.load_state_dict(
                torch.load(model_path, map_location=self.args.device)
            )
            extractor.to(self.args.device)

        self.extractor = extractor

    def define_eigenvectors(self):
        from utils.rl import get_vector

        # === Define eigenvectors === #
        eigenvectors, heatmaps = get_vector(
            self.extractor_env, self.extractor, self.args
        )
        self.eigenvectors = eigenvectors
        self.logger.write_images(
            step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
        )

    def define_intrinsic_reward_normalizer(self):
        from utils.wrapper import RunningMeanStd

        self.reward_rms = []
        # DRND method has its own rms its own class
        for _ in range(self.args.num_options):
            self.reward_rms.append(RunningMeanStd(shape=(1,)))
