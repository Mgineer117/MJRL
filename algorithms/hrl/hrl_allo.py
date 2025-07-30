import os
from copy import deepcopy

import torch
import torch.nn as nn

from policy.ddpg_learner import DDPG_Learner
from policy.elementary_policy.uniform_random import UniformRandom
from policy.hrl_learner import HRL_PPO_Learner, HRL_SAC_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.layers.sac_networks import SAC_Actor, SAC_Critic
from policy.layers.td3_networks import TD3_Actor, TD3_Actor_From_Critic, TD3_Critic
from policy.ppo_learner import PPO_Learner
from policy.sac_learner import SAC_Learner
from trainer.hrl_trainer import HRLOffPolicyTrainer, HRLOnPolicyTrainer
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.replay_buffer import ReplayBuffer
from utils.sampler import HLSampler, OnlineSampler


class HRL_ALLO(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(HRL_ALLO, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.intrinsic_reward_fn = IntrinsicRewardFunctions(
            logger=logger,
            writer=writer,
            args=args,
        )

        self.current_timesteps = 0

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        hl_sampler = HLSampler(
            state_dim=self.args.state_dim,
            action_dim=len(self.policies),
            episode_len=self.args.episode_len,
            batch_size=self.args.on_policy_batch_size,
            max_option_len=self.args.max_option_duration,
            gamma=self.args.gamma,
            verbose=False,
        )

        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=self.args.on_policy_batch_size,
            verbose=False,
        )

        if self.args.hrl_base_algorithm == "ppo":
            trainer = HRLOnPolicyTrainer(
                env=self.env,
                hl_policy=self.hl_policy,
                policies=self.policies,
                intrinsic_reward_fn=self.intrinsic_reward_fn,
                hl_sampler=hl_sampler,
                sampler=sampler,
                logger=self.logger,
                writer=self.writer,
                init_timesteps=self.current_timesteps,
                args=self.args,
            )
        elif self.args.hrl_base_algorithm in ("ddpg", "sac"):
            replay_buffer = ReplayBuffer(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                buffer_size=self.args.buffer_size,
                batch_size=self.args.off_policy_batch_size,
                device=self.args.device,
            )
            trainer = HRLOffPolicyTrainer(
                replay_buffer=replay_buffer,
                env=self.env,
                hl_policy=self.hl_policy,
                policies=self.policies,
                intrinsic_reward_fn=self.intrinsic_reward_fn,
                hl_sampler=hl_sampler,
                sampler=sampler,
                logger=self.logger,
                writer=self.writer,
                init_timesteps=self.current_timesteps,
                args=self.args,
            )

        trainer.train()

    def define_policy(self):
        # === Define low-level policies === #
        self.policies = nn.ModuleList([])
        for i in range(self.args.num_options):
            if self.args.hrl_base_algorithm == "ppo":
                actor = PPO_Actor(
                    input_dim=self.args.state_dim,
                    hidden_dim=self.args.actor_fc_dim,
                    action_dim=self.args.action_dim,
                    is_discrete=self.args.is_discrete,
                )
                critic = PPO_Critic(
                    self.args.state_dim, hidden_dim=self.args.critic_fc_dim
                )

                policy = PPO_Learner(
                    actor=actor,
                    critic=critic,
                    actor_lr=self.args.actor_lr,
                    critic_lr=self.args.critic_lr,
                    num_minibatch=self.args.num_minibatch,
                    minibatch_size=self.args.on_policy_minibatch_size,
                    eps_clip=self.args.eps_clip,
                    entropy_scaler=self.args.entropy_scaler,
                    target_kl=self.args.target_kl,
                    gamma=self.args.gamma,
                    gae=self.args.gae,
                    K=self.args.K_epochs,
                    device=self.args.device,
                )
            elif self.args.hrl_base_algorithm == "sac":
                actor = SAC_Actor(
                    input_dim=self.args.state_dim,
                    hidden_dim=self.args.actor_fc_dim,
                    action_dim=self.args.action_dim,
                    action_space=self.env.action_space,
                    is_discrete=self.args.is_discrete,
                    activation=nn.ReLU(),
                    device=self.args.device,
                )
                critic1 = SAC_Critic(
                    self.args.state_dim,
                    self.args.action_dim,
                    hidden_dim=self.args.critic_fc_dim,
                    is_discrete=self.args.is_discrete,
                )
                critic2 = SAC_Critic(
                    self.args.state_dim,
                    self.args.action_dim,
                    hidden_dim=self.args.critic_fc_dim,
                    is_discrete=self.args.is_discrete,
                )

                policy = SAC_Learner(
                    actor=actor,
                    critic1=critic1,
                    critic2=critic2,
                    actor_lr=self.args.actor_lr,
                    critic_lr=self.args.critic_lr,
                    gamma=self.args.gamma,
                    tau=self.args.tau,
                    entropy_scaler=self.args.sac_entropy_scaler,
                    is_discrete=self.args.is_discrete,
                    device=self.args.device,
                )

            policy.name = "HRL_intrinsic_options"
            self.policies.append(policy)

        if self.args.fine_grained_option is None and not self.args.is_discrete:
            uniform_random_policy = UniformRandom(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                is_discrete=self.args.is_discrete,
                device=self.args.device,
            )

            self.policies.append(uniform_random_policy)
            print(
                "[INFO] Fine-grained options added:",
                self.policies[-1],
            )
        else:
            # add left right up down stay policy
            from policy.elementary_policy.policies import (
                DownPolicy,
                LeftPolicy,
                RightPolicy,
                StayPolicy,
                UpPolicy,
            )

            option_map = {
                "down": DownPolicy,
                "left": LeftPolicy,
                "right": RightPolicy,
                "stay": StayPolicy,
                "up": UpPolicy,
            }

            for option in self.args.fine_grained_option:
                assert option in option_map, f"Unknown fine-grained option: {option}"
                policy_class = option_map[option]
                self.policies.append(
                    policy_class(
                        state_dim=self.args.state_dim,
                        action_dim=self.args.action_dim,
                        is_discrete=self.args.is_discrete,
                        device=self.args.device,
                    )
                )
            print(
                "[INFO] Fine-grained options added:",
                self.policies[-len(self.args.fine_grained_option) :],
            )

        ### === Define high-level policy === ###
        if self.args.hrl_base_algorithm == "ppo":
            actor = PPO_Actor(
                input_dim=self.args.state_dim,
                hidden_dim=self.args.actor_fc_dim,
                action_dim=len(self.policies),
                is_discrete=True,
                device=self.args.device,
            )
            critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

            self.hl_policy = HRL_PPO_Learner(
                actor=actor,
                critic=critic,
                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                num_minibatch=self.args.num_minibatch,
                minibatch_size=self.args.on_policy_minibatch_size,
                eps_clip=self.args.eps_clip,
                entropy_scaler=self.args.entropy_scaler,
                target_kl=self.args.target_kl,
                gamma=self.args.gamma,
                gae=self.args.gae,
                K=self.args.K_epochs,
                device=self.args.device,
            )
        elif self.args.hrl_base_algorithm == "sac":
            actor = SAC_Actor(
                input_dim=self.args.state_dim,
                hidden_dim=self.args.actor_fc_dim,
                action_dim=len(self.policies),
                action_space=self.env.action_space,
                is_discrete=True,
                activation=nn.ReLU(),
                device=self.args.device,
            )
            critic1 = SAC_Critic(
                self.args.state_dim,
                len(self.policies),
                hidden_dim=self.args.critic_fc_dim,
                is_discrete=True,
            )
            critic2 = SAC_Critic(
                self.args.state_dim,
                len(self.policies),
                hidden_dim=self.args.critic_fc_dim,
                is_discrete=True,
            )

            self.hl_policy = HRL_SAC_Learner(
                actor=actor,
                critic1=critic1,
                critic2=critic2,
                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                gamma=self.args.gamma,
                tau=self.args.tau,
                entropy_scaler=self.args.sac_entropy_scaler,
                is_discrete=True,
                device=self.args.device,
            )

        if hasattr(self.env, "get_grid"):
            for p in self.policies:
                p.grid = self.env.get_grid()
            self.hl_policy.grid = self.env.get_grid()
