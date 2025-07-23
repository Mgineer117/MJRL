import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridworld.core.agent import CtFActions
from policy.layers.base import Base
from utils.rl import estimate_advantages


class LeftPolicy(Base):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_discrete: bool,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "LeftPolicy"
        self.is_discrete = is_discrete
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.idx = CtFActions.left

        #
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        idx_tensor = torch.tensor(self.idx, device=self.device, dtype=torch.long)
        a = (
            F.one_hot(idx_tensor, num_classes=self.action_dim)
            .to(dtype=self.dtype)
            .unsqueeze(0)
        )

        probs = torch.ones(1, device=self.device, dtype=self.dtype)
        logprobs = torch.zeros(1, device=self.device, dtype=self.dtype)
        entropy = torch.zeros(1, device=self.device, dtype=self.dtype)

        return a, {
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def learn(self, batch):
        # no loss dict, no update time
        return {}, 0.0  # No learning


class UpPolicy(Base):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_discrete: bool,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "UpPolicy"
        self.is_discrete = is_discrete
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.idx = CtFActions.up

        #
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        idx_tensor = torch.tensor(self.idx, device=self.device, dtype=torch.long)
        a = (
            F.one_hot(idx_tensor, num_classes=self.action_dim)
            .to(dtype=self.dtype)
            .unsqueeze(0)
        )

        probs = torch.ones(1, device=self.device, dtype=self.dtype)
        logprobs = torch.zeros(1, device=self.device, dtype=self.dtype)
        entropy = torch.zeros(1, device=self.device, dtype=self.dtype)

        return a, {
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def learn(self, batch):
        # no loss dict, no update time
        return {}, 0.0  # No learning


class RightPolicy(Base):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_discrete: bool,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "RightPolicy"
        self.is_discrete = is_discrete
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.idx = CtFActions.right

        #
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        idx_tensor = torch.tensor(self.idx, device=self.device, dtype=torch.long)
        a = (
            F.one_hot(idx_tensor, num_classes=self.action_dim)
            .to(dtype=self.dtype)
            .unsqueeze(0)
        )

        probs = torch.ones(1, device=self.device, dtype=self.dtype)
        logprobs = torch.zeros(1, device=self.device, dtype=self.dtype)
        entropy = torch.zeros(1, device=self.device, dtype=self.dtype)

        return a, {
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def learn(self, batch):
        # no loss dict, no update time
        return {}, 0.0  # No learning


class DownPolicy(Base):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_discrete: bool,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "DownPolicy"
        self.is_discrete = is_discrete
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.idx = CtFActions.down

        #
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        idx_tensor = torch.tensor(self.idx, device=self.device, dtype=torch.long)
        a = (
            F.one_hot(idx_tensor, num_classes=self.action_dim)
            .to(dtype=self.dtype)
            .unsqueeze(0)
        )

        probs = torch.ones(1, device=self.device, dtype=self.dtype)
        logprobs = torch.zeros(1, device=self.device, dtype=self.dtype)
        entropy = torch.zeros(1, device=self.device, dtype=self.dtype)

        return a, {
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def learn(self, batch):
        # no loss dict, no update time
        return {}, 0.0  # No learning


class StayPolicy(Base):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_discrete: bool,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "StayPolicy"
        self.is_discrete = is_discrete
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.idx = CtFActions.stay

        #
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        idx_tensor = torch.tensor(self.idx, device=self.device, dtype=torch.long)
        a = (
            F.one_hot(idx_tensor, num_classes=self.action_dim)
            .to(dtype=self.dtype)
            .unsqueeze(0)
        )

        probs = torch.ones(1, device=self.device, dtype=self.dtype)
        logprobs = torch.zeros(1, device=self.device, dtype=self.dtype)
        entropy = torch.zeros(1, device=self.device, dtype=self.dtype)

        return a, {
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def learn(self, batch):
        # no loss dict, no update time
        return {}, 0.0  # No learning
