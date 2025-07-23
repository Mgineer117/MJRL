import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Normal

from policy.layers.base import Base
from policy.layers.building_blocks import MLP
from gymnasium import spaces


class SAC_Actor(Base):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        action_space: spaces,
        activation: nn.Module = nn.ReLU(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)  # modify based on Base's signature

        self.state_dim = np.prod(input_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)

        self.action_space = action_space
        assert isinstance(
            self.action_space, spaces.Box
        ), f"The action space must be a Box(): {self.action_space}"

        self.action_high = torch.from_numpy(action_space.high).to(device)
        self.action_low = torch.from_numpy(action_space.low).to(device)

        self.is_discrete = False

        self.backbone = MLP(
            self.state_dim,
            hidden_dim,
            # self.action_dim,
            activation=activation,
            initialization="actor",
        )

        self.mu = nn.Linear(hidden_dim[-1], self.action_dim)
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

        mu = self.mu(logits)
        logstd = self.logstd(logits)
        logstd = torch.clamp(logstd, -5, 2)  # Prevent extreme stds
        std = torch.exp(logstd)

        dist = Normal(loc=mu, scale=std)
        u = dist.rsample()  # Pre-squash action
        a = torch.tanh(u)  # Squashed action

        # Unscale to action space if needed
        action = self.unscale_action(a)

        # Compute log π(a|s)
        log_prob_u = dist.log_prob(u)  # shape: [batch_size, action_dim]
        log_prob_u = log_prob_u.sum(dim=1, keepdim=True)  # sum over action dims

        # Correction for tanh squashing
        log_det_jacobian = torch.sum(
            torch.log(1 - torch.tanh(u) ** 2 + 1e-6), dim=1, keepdim=True
        )

        logprobs = log_prob_u - log_det_jacobian  # final log probability
        probs = torch.exp(logprobs)
        # entropy = dist.entropy().sum(1)

        return action, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": self._dummy,
        }


class SAC_Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: list,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.input_dim = np.prod(state_dim) + np.prod(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

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
