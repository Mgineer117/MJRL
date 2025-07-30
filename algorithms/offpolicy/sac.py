import torch
import torch.nn as nn

from policy.layers.sac_networks import SAC_Actor, SAC_Critic
from policy.sac_learner import SAC_Learner
from trainer.offpolicy_trainer import OffPolicyTrainer
from utils.replay_buffer import ReplayBuffer


class SAC_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(SAC_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        replay_buffer = ReplayBuffer(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            buffer_size=self.args.buffer_size,
            batch_size=self.args.off_policy_batch_size,
            device=self.args.device,
        )
        trainer = OffPolicyTrainer(
            env=self.env,
            policy=self.policy,
            replay_buffer=replay_buffer,
            logger=self.logger,
            writer=self.writer,
            args=self.args,
        )

        trainer.train()

    def define_policy(self):
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
            activation=nn.ReLU(),
        )
        critic2 = SAC_Critic(
            self.args.state_dim,
            self.args.action_dim,
            hidden_dim=self.args.critic_fc_dim,
            is_discrete=self.args.is_discrete,
            activation=nn.ReLU(),
        )

        self.policy = SAC_Learner(
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

        if hasattr(self.env, "get_grid"):
            self.policy.grid = self.env.get_grid()
