import torch
import torch.nn as nn

from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.psne_learner import PSNE_Learner
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.sampler import OnlineSampler


class PSNE_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(PSNE_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=self.args.batch_size,
        )

        trainer = OnPolicyTrainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            args=self.args,
        )

        trainer.train()

    def define_policy(self):
        from policy.uniform_random import UniformRandom

        self.uniform_random_policy = UniformRandom(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=self.args.batch_size,
            verbose=False,
        )
        batch, _ = sampler.collect_samples(
            env=self.env, policy=self.uniform_random_policy, seed=self.args.seed
        )

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.policy = PSNE_Learner(
            actor=actor,
            critic=critic,
            states=batch["states"],
            nupdates=self.args.nupdates,
            critic_lr=self.args.critic_lr,
            entropy_scaler=self.args.entropy_scaler,
            batch_size=self.args.batch_size,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            self.policy.grid = self.env.get_grid()
