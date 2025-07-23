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
        entropy_automation: str = "auto_0.2",
        is_discrete: bool = False,
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        """
        SAC Learner for Soft Actor-Critic algorithm.
        Discerete version is based on https://arxiv.org/pdf/1910.07207
        """
        # constants
        self.name = "SAC"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim

        self.entropy_automation = entropy_automation

        if isinstance(entropy_automation, str) and entropy_automation.startswith(
            "auto"
        ):
            init_value = 1.0
            if "_" in entropy_automation:
                init_value = float(entropy_automation.split("_")[1])
                assert init_value > 0, "Entropy scaler must be positive."
            self.entropy_scaler = nn.Parameter(
                init_value * torch.ones(1, dtype=torch.float32, device=device)
            )
            self.entropy_optimizer = torch.optim.Adam(
                [self.entropy_scaler], lr=critic_lr
            )
            self.entropy_target = -actor.action_dim
        else:
            self.entropy_scaler = entropy_scaler
            self.entropy_target = None

        self.gamma = gamma
        self.tau = tau
        self.nupdates = nupdates

        # trainable networks
        self.is_discrete = is_discrete

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
            "dist": metaData["dist"],
        }

    def _update_target_network(self, target: nn.Module, origin: nn.Module, tau: float):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(
                tau * origin_param.data + (1.0 - tau) * target_param.data
            )

    def learn(self, replay_buffer: ReplayBuffer):
        """Performs a single training step using DDPG TD3, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        ### === PREPARE SAMPLES === ###
        loss_dict = {}

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

        ### === ACTOR UPDATE === ###
        actor_loss, infos = self.actor_loss(states)

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

        ### === ENTROPY SCALER UPDATE === ###
        # note that entropy scaler is updated after critic and actor update
        # this ensures that the entropy scaler update does not affect the critic and actor loss
        if self.entropy_target is not None:
            actor_probs = infos["probs"].detach()
            actor_logprobs = infos["logprobs"].detach()
            # Update entropy scaler
            if self.is_discrete:
                entropy_loss = (
                    -torch.log(self.entropy_scaler)
                    * (actor_logprobs + self.entropy_target).detach()
                )
                entropy_loss = (actor_probs * entropy_loss).sum(1).mean()
            else:
                entropy_loss = -(
                    torch.log(self.entropy_scaler)
                    * (actor_logprobs + self.entropy_target).detach()
                ).mean()

            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        ### === LOGGING === ###
        loss_dict[f"{self.name}/critic_loss"] = critic_loss.item()
        loss_dict[f"{self.name}/td_error"] = td_error.item()
        loss_dict[f"{self.name}/actor_loss"] = actor_loss.item()
        loss_dict[f"{self.name}/entropy_loss"] = entropy_loss.item()
        loss_dict.update(critic_grad_dict)
        loss_dict.update(critic_norm_dict)
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

        if self.is_discrete:
            # actor gradient is applied to the actor_probs
            actor_probs = infos["probs"]
            actor_logprobs = infos["logprobs"]

            with torch.no_grad():
                Q1 = self.critic1(states)
                Q2 = self.critic2(states)
                Q = torch.min(Q1, Q2)

            soft_Q = self.entropy_scaler * actor_logprobs - Q
            actor_loss = (actor_probs * soft_Q).sum(dim=1).mean()
        else:
            # actor gradient is applied to the actions and logprobs
            actor_logprobs = infos["logprobs"]

            # compute soft_Q
            critic_states = torch.cat([states, actions], dim=-1)

            Q1 = self.critic1(critic_states)
            Q2 = self.critic2(critic_states)
            Q = torch.min(Q1, Q2)
            # the actor is optimized to maximize the entropy and Q
            soft_Q = self.entropy_scaler * actor_logprobs - Q

            # actor_loss computations
            actor_loss = soft_Q.mean()

        return actor_loss, infos

    def critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ):
        if self.is_discrete:
            with torch.no_grad():
                _, infos = self.actor(next_states)
                next_actor_probs = infos["probs"].detach()
                next_actor_logprobs = infos["logprobs"].detach()

                next_Q1 = self.critic_target1(next_states)
                next_Q2 = self.critic_target2(next_states)
                next_Q = torch.min(next_Q1, next_Q2)
                next_soft_Q = next_Q - self.entropy_scaler * next_actor_logprobs
                next_soft_Q = (next_actor_probs * next_soft_Q).sum(dim=1, keepdim=True)
                target_Q = rewards + (1 - terminals) * self.gamma * next_soft_Q

            current_Q1 = self.critic1(states)
            current_Q2 = self.critic2(states)

            current_Q1 = (actions * current_Q1).sum(dim=1, keepdim=True)
            current_Q2 = (actions * current_Q2).sum(dim=1, keepdim=True)

            critic1_loss = F.huber_loss(current_Q1, target_Q)
            critic2_loss = F.huber_loss(current_Q2, target_Q)

            critic_loss = critic1_loss + critic2_loss
            td_error = (target_Q - current_Q1).mean().cpu()
        else:
            with torch.no_grad():
                next_actions, infos = self.actor(next_states)
                actor_probs = infos["probs"].detach()
                actor_logprobs = infos["logprobs"].detach()

                critic_next_states = torch.cat([next_states, next_actions], dim=-1)

                target_Q1 = self.critic_target1(critic_next_states)
                target_Q2 = self.critic_target2(critic_next_states)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = target_Q - self.entropy_scaler * actor_logprobs
                target_Q = (rewards + (1 - terminals) * self.gamma * target_Q).detach()

            critic_states = torch.cat([states, actions], dim=-1)

            current_Q1 = self.critic1(critic_states)
            current_Q2 = self.critic2(critic_states)

            critic1_loss = F.huber_loss(current_Q1, target_Q)
            critic2_loss = F.huber_loss(current_Q2, target_Q)

            critic_loss = critic1_loss + critic2_loss
            td_error = (target_Q - current_Q1).mean().cpu()

        return critic_loss, td_error
