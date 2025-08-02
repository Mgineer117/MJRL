import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.layers.base import Base
from policy.layers.sac_networks import SAC_Actor, SAC_Critic
from utils.replay_buffer import ReplayBuffer
from utils.rl import estimate_advantages


class SAC_Learner(Base):
    def __init__(
        self,
        actor: SAC_Actor,
        critic1: SAC_Critic,
        critic2: SAC_Critic,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        entropy_scaler: str | float = "auto_0.2",
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

        if isinstance(entropy_scaler, str) and entropy_scaler.startswith("auto"):
            init_value = 1.0
            if "_" in entropy_scaler:
                init_value = float(entropy_scaler.split("_")[1])
                assert init_value > 0, "Entropy scaler must be positive."
            self.log_entropy_scaler = torch.log(
                torch.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam(
                [self.log_entropy_scaler], lr=entropy_lr
            )
            self.entropy_scaler = torch.exp(self.log_entropy_scaler.detach())

            if is_discrete:
                self.entropy_target = -np.log(1.0 / self.action_dim) * 0.98
            else:
                self.entropy_target = -float(actor.action_dim)

        else:
            self.entropy_scaler = torch.tensor(
                float(entropy_scaler), device=self.device
            )
            self.entropy_target = None

        self.gamma = gamma
        self.tau = tau

        # trainable networks
        self.is_discrete = is_discrete

        self.actor = actor

        self.critic1 = critic1
        self.critic2 = critic2

        self.critic_target1 = deepcopy(critic1)
        self.critic_target2 = deepcopy(critic2)

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

    def lr_lambda(self, fraction: float):
        return 1.0 - fraction

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {"probs": metaData["probs"], "logprobs": metaData["logprobs"]}

    def _update_target_network(self, target: nn.Module, origin: nn.Module, tau: float):
        with torch.no_grad():
            for target_param, origin_param in zip(
                target.parameters(), origin.parameters()
            ):
                target_param.data.mul_(1 - tau)
                torch.add(
                    target_param.data,
                    origin_param.data,
                    alpha=tau,
                    out=target_param.data,
                )

    def learn(self, replay_buffer: ReplayBuffer, fraction: float):
        """Performs a single training step using DDPG TD3, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        ### === PREPARE SAMPLES === ###
        loss_dict = {}

        states, actions, next_states, rewards, terminals = replay_buffer.sample()

        actions_pi, infos_pi = self.actor(states)

        ### === ENTROPY SCALER UPDATE === ###
        # note that entropy scaler is updated after critic and actor update
        # this ensures that the entropy scaler update does not affect the critic and actor loss
        entropy_loss = self.entropy_loss(
            probs=infos_pi["probs"],
            logprobs=infos_pi["logprobs"],
        )

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
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=100.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=100.0)
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
        actor_loss = self.actor_loss(states, actions_pi, infos_pi)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
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

        ### === LOGGING === ###
        loss_dict[f"{self.name}/critic_loss"] = critic_loss.item()
        loss_dict[f"{self.name}/td_error"] = td_error.item()
        loss_dict[f"{self.name}/actor_loss"] = actor_loss.item()
        loss_dict[f"{self.name}/entropy_loss"] = entropy_loss.item()
        loss_dict[f"{self.name}/entropy_scaler"] = self.entropy_scaler.item()
        loss_dict.update(critic_grad_dict)
        loss_dict.update(critic_norm_dict)
        loss_dict.update(actor_grad_dict)
        loss_dict.update(actor_norm_dict)

        self.steps += 1

        ### === POLYAK AVERAGING === ###
        with torch.no_grad():
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
        actions: torch.Tensor,
        infos: dict,
    ):
        if self.is_discrete:
            # actor gradient is applied to the actor_probs
            actor_probs = infos["probs"]
            actor_logprobs = infos["logprobs"]

            with torch.no_grad():
                Q1, Q2 = self.critic1(states), self.critic2(states)

            Q = torch.sum(actor_probs * torch.min(Q1, Q2), dim=-1, keepdim=True)
            entropy = torch.sum(actor_probs * actor_logprobs, dim=-1, keepdim=True)
            soft_Q = self.entropy_scaler * entropy - Q
            actor_loss = soft_Q.mean()
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

        return actor_loss

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
                _, infos = self.actor(next_states, deterministic=False)
                next_actor_probs = infos["probs"].detach()
                next_actor_logprobs = infos["logprobs"].detach()

                next_Q1, next_Q2 = self.critic_target1(
                    next_states
                ), self.critic_target2(next_states)
                next_Q = torch.min(next_Q1, next_Q2)
                next_soft_Q = next_Q - self.entropy_scaler * next_actor_logprobs
                next_soft_Q = (next_actor_probs * next_soft_Q).sum(dim=1, keepdim=True)
                target_Q = rewards + (1 - terminals) * self.gamma * next_soft_Q

            current_Q1 = self.critic1(states)
            current_Q2 = self.critic2(states)

            current_Q1 = (actions * current_Q1).sum(dim=1, keepdim=True)
            current_Q2 = (actions * current_Q2).sum(dim=1, keepdim=True)

            critic1_loss = F.mse_loss(current_Q1, target_Q)
            critic2_loss = F.mse_loss(current_Q2, target_Q)

            critic_loss = critic1_loss + critic2_loss
            td_error = (target_Q - current_Q1).mean().cpu()
        else:
            with torch.no_grad():
                next_actions, infos = self.actor(next_states)
                next_actor_logprobs = infos["logprobs"].detach()

                critic_next_states = torch.cat([next_states, next_actions], dim=-1)

                next_Q1 = self.critic_target1(critic_next_states)
                next_Q2 = self.critic_target2(critic_next_states)
                next_Q = torch.min(next_Q1, next_Q2)
                next_soft_Q = next_Q - self.entropy_scaler * next_actor_logprobs
                target_Q = rewards + (1 - terminals) * self.gamma * next_soft_Q

            critic_states = torch.cat([states, actions], dim=-1)

            current_Q1 = self.critic1(critic_states)
            current_Q2 = self.critic2(critic_states)

            critic1_loss = F.mse_loss(current_Q1, target_Q)
            critic2_loss = F.mse_loss(current_Q2, target_Q)

            critic_loss = 0.5 * (critic1_loss + critic2_loss)
            td_error = (target_Q - current_Q1).mean().cpu()

        return critic_loss, td_error

    def entropy_loss(self, probs: torch.Tensor, logprobs: torch.Tensor):
        if self.entropy_target is not None:
            self.entropy_scaler = torch.exp(self.log_entropy_scaler.detach())

            actor_probs = probs.detach()
            actor_logprobs = logprobs.detach()
            # Update entropy scaler
            if self.is_discrete:
                entropy = -(actor_probs * actor_logprobs).sum(dim=-1)
                entropy_loss = (
                    self.log_entropy_scaler * (self.entropy_target - entropy).mean()
                )
                # entropy = torch.sum(actor_probs * actor_logprobs, dim=-1, keepdim=True)
                # entropy_loss = (actor_probs * entropy).sum(-1).mean()
            else:
                entropy_loss = (
                    self.log_entropy_scaler * (actor_logprobs + self.entropy_target)
                ).mean()

            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()

        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        return entropy_loss
