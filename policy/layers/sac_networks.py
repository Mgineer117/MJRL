import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical, MultivariateNormal, Normal

from policy.layers.base import Base
from policy.layers.building_blocks import MLP


class SAC_Actor(Base):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        is_discrete: bool,
        action_space: spaces,
        activation: nn.Module = nn.ReLU(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)  # modify based on Base's signature

        self.state_dim = np.prod(input_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)

        self.is_discrete = is_discrete
        if not self.is_discrete:
            self.action_space = action_space
            assert isinstance(
                self.action_space, spaces.Box
            ), f"The action space must be a Box(): {self.action_space}"

            self.action_high = torch.from_numpy(action_space.high).to(device)
            self.action_low = torch.from_numpy(action_space.low).to(device)

        self.backbone = MLP(
            self.state_dim,
            hidden_dim,
            # self.action_dim,
            activation=activation,
            initialization="actor",
        )

        self.mu = nn.Linear(hidden_dim[-1], self.action_dim)
        if not self.is_discrete:
            self.logstd = nn.Linear(hidden_dim[-1], self.action_dim)

        self.device = device
        self._dummy = torch.tensor(1e-8)
        self.to(self.device).to(self.dtype)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.backbone(state)

        if self.is_discrete:
            logits = self.mu(logits)
            logits = torch.clamp(logits, -20, 20)

            logprobs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            dist = Categorical(probs)

            if deterministic:
                action_idx = torch.argmax(probs, dim=-1)
            else:
                action_idx = dist.sample()

            action = F.one_hot(action_idx, num_classes=self.action_dim).float()
        else:
            mu = self.mu(logits)
            logstd = self.logstd(logits)
            logstd = torch.clamp(logstd, -20, 2)  # Prevent extreme stds
            std = torch.ones_like(mu) * logstd.exp()

            dist = Normal(loc=mu, scale=std)

            action = dist.rsample()

            logprobs = dist.log_prob(action)  # shape: [batch_size, action_dim]
            logprobs = logprobs.sum(dim=1, keepdim=True)  # sum over action dims
            probs = torch.exp(logprobs)

            u = dist.rsample()  # Pre-squash action
            a = torch.tanh(u)  # Squashed action

            # Compute log π(a|s)
            log_prob_u = dist.log_prob(u)  # shape: [batch_size, action_dim]
            log_prob_u = log_prob_u.sum(dim=1, keepdim=True)  # sum over action dims

            # Correction for tanh squashing
            log_det_jacobian = torch.sum(
                torch.log(1 - a**2 + 1e-6), dim=1, keepdim=True
            )

            logprobs = log_prob_u - log_det_jacobian  # final log probability
            probs = torch.exp(logprobs)

            # Unscale to action space if needed
            # action = a
            action = self.unscale_action(a)

        return action, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
        }


class SAC_Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: list,
        is_discrete: bool,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.is_discrete = is_discrete
        if self.is_discrete:
            self.input_dim = np.prod(state_dim)
            self.model = MLP(
                self.input_dim,
                hidden_dim,
                self.action_dim,
                activation=activation,
                initialization="critic",
            )
        else:
            self.input_dim = np.prod(state_dim) + np.prod(action_dim)
            self.model = MLP(
                self.input_dim,
                hidden_dim,
                1,
                activation=activation,
                initialization="critic",
            )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
