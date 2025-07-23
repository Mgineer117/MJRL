import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.layers.base import Base
from policy.layers.sac_network import SAC_Actor, SAC_Critic
from utils.replay_buffer import ReplayBuffer
from utils.rl import estimate_advantages


class SAC_Learner(Base):
    def __init__(
        self,
        actor: SAC_Actor,
        critic: SAC_Critic,
        nupdates: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        entropy_scaler: float = 1e-3,
        is_discrete: bool = False,
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        # constants
        self.name = "SAC"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim

        self.gamma = gamma
        self.tau = tau
        self.entropy_scaler = entropy_scaler
        self.nupdates = nupdates

        # trainable networks
        self.is_discrete = is_discrete

        if self.is_discrete:
            # actor share the same memory with critic
            # since actor is not trainable but relies on the critic1
            self.actor = actor

            self.critic1 = critic
            self.critic2 = deepcopy(critic)

            self.critic_target1 = deepcopy(critic)
            self.critic_target2 = deepcopy(critic)

            self.critic_optimizer = torch.optim.Adam(
                [
                    {"params": self.critic1.parameters(), "lr": critic_lr},
                    {"params": self.critic2.parameters(), "lr": critic_lr},
                ]
            )
        else:
            self.actor = actor

            self.critic1 = critic
            self.critic2 = deepcopy(critic)

            self.critic_target1 = deepcopy(critic)
            self.critic_target2 = deepcopy(critic)

            self.actor_optimizer = torch.optim.Adam(
                params=self.actor.parameters(), lr=actor_lr
            )
            self.critic_optimizer = torch.optim.Adam(
                [
                    {"params": self.critic1.parameters(), "lr": critic_lr},
                    {"params": self.critic2.parameters(), "lr": critic_lr},
                ]
            )
        #
        self.steps = 0
        self.to(self.dtype).to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def _update_target_network(self, target: nn.Module, origin: nn.Module, tau: float):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(
                tau * origin_param.data + (1.0 - tau) * target_param.data
            )

    def learn(self, replay_buffer: ReplayBuffer):
        if self.is_discrete:
            return self.learn_for_discrete(replay_buffer)
        else:
            return self.learn_for_continuous(replay_buffer)

    def learn_for_discrete(self, replay_buffer: ReplayBuffer):
        """Performs a single training step using DDPG TD3, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        loss_dict = {}

        ### === PREPARE SAMPLES === ###
        states, actions, next_states, rewards, terminals = replay_buffer.sample()

        ### === CRITIC UPDATE === ###
        critic_loss, td_error = self.critic_loss(
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            terminals=terminals,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        critic_grad_dict = self.compute_gradient_norm(
            [self.critic1, self.critic2],
            ["critic1", "critic2"],
            dir=f"{self.name}",
            device=self.device,
        )
        critic_norm_dict = self.compute_weight_norm(
            [self.critic1, self.critic2, self.critic_target1, self.critic_target2],
            ["critic1", "critic2", "critic_target1", "critic_target2"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.critic_optimizer.step()

        self.steps += 1

        ### === POLYAK AVERAGING === ###
        self._update_target_network(self.critic_target1, self.critic1, self.tau)
        self._update_target_network(self.critic_target2, self.critic2, self.tau)

        ### === LOGGING === ###
        loss_dict[f"{self.name}/critic_loss"] = critic_loss.item()
        loss_dict[f"{self.name}/td_error"] = td_error.item()

        loss_dict.update(critic_grad_dict)
        loss_dict.update(critic_norm_dict)

        # Cleanup
        del states, actions, next_states, rewards, terminals
        self.eval()

        update_time = time.time() - t0

        return loss_dict, update_time

    def learn_for_continuous(self, replay_buffer: ReplayBuffer):
        """Performs a single training step using DDPG TD3, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        loss_dict = {}

        ### === PREPARE SAMPLES === ###
        states, actions, next_states, rewards, terminals = replay_buffer.sample()

        ### === CRITIC UPDATE === ###
        critic_loss, td_error = self.critic_loss(
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            terminals=terminals,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        critic_grad_dict = self.compute_gradient_norm(
            [self.critic1, self.critic2],
            ["critic1", "critic2"],
            dir=f"{self.name}",
            device=self.device,
        )
        critic_norm_dict = self.compute_weight_norm(
            [self.critic1, self.critic2, self.critic_target1, self.critic_target2],
            ["critic1", "critic2", "critic_target1", "critic_target2"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.critic_optimizer.step()

        ### === LOGGING === ###
        loss_dict[f"{self.name}/critic_loss"] = critic_loss.item()
        loss_dict[f"{self.name}/td_error"] = td_error.item()
        loss_dict.update(critic_grad_dict)
        loss_dict.update(critic_norm_dict)

        ### === ACTOR UPDATE === ###
        actor_loss = self.actor_loss(states)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        actor_grad_dict = self.compute_gradient_norm(
            [self.actor],
            ["actor"],
            dir=f"{self.name}",
            device=self.device,
        )
        actor_norm_dict = self.compute_weight_norm(
            [self.actor],
            ["actor"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.actor_optimizer.step()

        loss_dict[f"{self.name}/actor_loss"] = actor_loss.item()
        loss_dict.update(actor_grad_dict)
        loss_dict.update(actor_norm_dict)

        self.steps += 1

        ### === POLYAK AVERAGING === ###
        self._update_target_network(self.critic_target1, self.critic1, self.tau)
        self._update_target_network(self.critic_target2, self.critic2, self.tau)

        # Cleanup
        del states, actions, next_states, rewards, terminals
        self.eval()

        update_time = time.time() - t0

        return loss_dict, update_time

    def actor_loss(
        self,
        states: torch.Tensor,
    ):
        actions, infos = self.actor(states)
        critic_states = torch.cat([states, actions], dim=-1)
        # with torch.no_grad():
        Q1 = self.critic1(critic_states)
        Q2 = self.critic2(critic_states)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.entropy_scaler * infos["logprobs"] - Q).mean()

        return actor_loss

    def critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ):
        with torch.no_grad():
            next_actions, infos = self.actor(next_states)
            critic_next_states = torch.cat([next_states, next_actions], dim=-1)

            target_Q1 = self.critic_target1(critic_next_states)
            target_Q2 = self.critic_target2(critic_next_states)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = target_Q - self.entropy_scaler * infos["logprobs"]
            target_Q = (rewards + (1 - terminals) * self.gamma * target_Q).detach()

        critic_states = torch.cat([states, actions], dim=-1)

        current_Q1 = self.critic1(critic_states)
        current_Q2 = self.critic2(critic_states)

        critic1_loss = F.huber_loss(current_Q1, target_Q)
        critic2_loss = F.huber_loss(current_Q2, target_Q)

        critic_loss = critic1_loss + critic2_loss
        td_error = (target_Q - current_Q1).mean().cpu()

        return critic_loss, td_error
