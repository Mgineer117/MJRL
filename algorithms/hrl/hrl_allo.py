import os
from copy import deepcopy
from policy.sac_learner import SAC_Learner
from policy.layers.sac_network import SAC_Actor, SAC_Critic
import torch
import torch.nn as nn
from policy.ddpg_learner import DDPG_Learner
from utils.replay_buffer import ReplayBuffer
from policy.elementary_policy.uniform_random import UniformRandom
from policy.hrl_learner import HRL_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.ppo_learner import PPO_Learner
from trainer.hrl_trainer import HRLOffPolicyTrainer, HRLOnPolicyTrainer
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.sampler import HLSampler, OnlineSampler
from policy.layers.td3_network import TD3_Actor, TD3_Actor_From_Critic, TD3_Critic


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

        self.args.nupdates = args.timesteps // (
            args.minibatch_size * args.num_minibatch
        )
        self.args.hl_nupdates = args.hl_timesteps // (
            args.minibatch_size * args.num_minibatch
        )

        self.current_timesteps = 0

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        hl_sampler = HLSampler(
            state_dim=self.args.state_dim,
            action_dim=len(self.policies),
            episode_len=self.args.episode_len,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
            max_option_len=self.args.max_option_duration,
            gamma=self.args.gamma,
            verbose=False,
        )

        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
            verbose=False,
        )

        if self.args.option_algorithm == "ppo":
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
        elif self.args.option_algorithm in ("ddpg", "sac"):
            replay_buffer = ReplayBuffer(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                buffer_size=self.args.buffer_size,
                batch_size=self.args.batch_size,
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
        # === Define policy === #
        self.policies = nn.ModuleList([])
        for i in range(self.args.num_options):
            if self.args.option_algorithm == "ppo":
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
                    nupdates=self.args.nupdates,
                    actor_lr=self.args.actor_lr,
                    critic_lr=self.args.critic_lr,
                    num_minibatch=self.args.num_minibatch,
                    minibatch_size=self.args.minibatch_size,
                    eps_clip=self.args.eps_clip,
                    entropy_scaler=self.args.entropy_scaler,
                    target_kl=self.args.target_kl,
                    gamma=self.args.gamma,  # 1.0,  # gamma for option is 1 to find maxima
                    gae=self.args.gae,
                    K=self.args.K_epochs,
                    device=self.args.device,
                )
            elif self.args.option_algorithm == "ddpg":
                if self.args.is_discrete:
                    if i == 0:
                        # to print once
                        print(
                            "[INFO] DDPG for discrete action space is implemented using twin-critic Q-values. "
                            "[INFO] This works ok, but not widely used discrete method. "
                            "[INFO] Consider using PPO or SAC for discrete action space."
                        )
                    critic = TD3_Critic(
                        self.args.state_dim,
                        self.args.action_dim,
                        hidden_dim=self.args.critic_fc_dim,
                    )
                    # actor is a wrapper that chooses over critic
                    actor = TD3_Actor_From_Critic(critic)
                else:
                    actor = TD3_Actor(
                        input_dim=self.args.state_dim,
                        hidden_dim=self.args.actor_fc_dim,
                        action_dim=self.args.action_dim,
                        action_space=self.env.action_space,
                        action_noise_coeff=self.args.action_noise_coeff,
                        activation=nn.ReLU(),
                        device=self.args.device,
                    )
                    critic = TD3_Critic(
                        self.args.state_dim,
                        self.args.action_dim,
                        hidden_dim=self.args.critic_fc_dim,
                    )

                policy = DDPG_Learner(
                    actor=actor,
                    critic=critic,
                    nupdates=self.args.nupdates,
                    actor_lr=self.args.actor_lr,
                    critic_lr=self.args.critic_lr,
                    policy_freq=self.args.policy_freq,
                    gamma=self.args.gamma,
                    tau=self.args.tau,
                    is_discrete=self.args.is_discrete,
                    device=self.args.device,
                )
            elif self.args.option_algorithm == "sac":
                actor = SAC_Actor(
                    input_dim=self.args.state_dim,
                    hidden_dim=self.args.actor_fc_dim,
                    action_dim=self.args.action_dim,
                    action_space=self.env.action_space,
                    is_discrete=self.args.is_discrete,
                    activation=nn.ReLU(),
                    device=self.args.device,
                )
                critic = SAC_Critic(
                    self.args.state_dim,
                    self.args.action_dim,
                    hidden_dim=self.args.critic_fc_dim,
                    is_discrete=self.args.is_discrete,
                )

                policy = SAC_Learner(
                    actor=actor,
                    critic=critic,
                    nupdates=self.args.nupdates,
                    actor_lr=self.args.actor_lr,
                    critic_lr=self.args.critic_lr,
                    K_epochs=self.args.K_epochs,
                    gamma=self.args.gamma,
                    tau=self.args.tau,
                    entropy_automation=self.args.entropy_automation,
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

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=len(self.policies),
            is_discrete=True,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.hl_policy = HRL_Learner(
            actor=actor,
            critic=critic,
            nupdates=self.args.hl_nupdates,
            num_options=self.args.num_options,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            num_minibatch=self.args.num_minibatch,
            minibatch_size=self.args.minibatch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            for p in self.policies:
                p.grid = self.env.get_grid()
            self.hl_policy.grid = self.env.get_grid()
