import time

import numpy as np
import torch

from policy.ppo_learner import PPO_Learner
from policy.sac_learner import SAC_Learner
from policy.ddpg_learner import DDPG_Learner




class HRL_PPO_Learner(PPO_Learner):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # constants
        self.name = "HRL_PPO"

        #
        self.to(self.dtype).to(self.device)

    def update_options(self, policies):
        """Update the policies with new options."""
        self.policies = policies
        self.trainable_options = [
            i for i, policy in enumerate(self.policies) if hasattr(policy, "actor")
        ]

    def forward(
        self, state: np.ndarray, option_idx: int | None, deterministic: bool = False
    ):
        state = self.preprocess_state(state)
        if option_idx is None:
            logits, metaData = self.actor(state, deterministic=deterministic)
            option_idx = torch.argmax(logits, dim=-1).item()
        else:
            logits = torch.tensor(np.full((1, self.action_dim), np.nan)).to(self.device)
            metaData = {
                "probs": torch.tensor(np.nan).to(self.device),
                "logprobs": torch.tensor(np.nan).to(self.device),
                "dist": torch.tensor(np.nan).to(self.device),
            }

        is_option = True if option_idx in self.trainable_options else False
        if is_option:
            a, _ = self.policies[option_idx].actor(state, deterministic=True)
            value = self.policies[option_idx].critic(state)

            option_termination = True if value.item() < 0 else False
        else:
            a, _ = self.policies[option_idx](state, deterministic=True)
            option_termination = False

        return [option_idx, a], {
            "logits": logits,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "dist": metaData["dist"],
            "is_option": is_option,
            "option_termination": option_termination,
        }

class HRL_DDPG_Learner(DDPG_Learner):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # constants
        self.name = "HRL_DDPG"

        #
        self.to(self.dtype).to(self.device)

    def update_options(self, policies):
        """Update the policies with new options."""
        self.policies = policies
        self.trainable_options = [
            i for i, policy in enumerate(self.policies) if hasattr(policy, "actor")
        ]

    def forward(
        self, state: np.ndarray, option_idx: int | None, deterministic: bool = False
    ):
        state = self.preprocess_state(state)
        if option_idx is None:
            logits, metaData = self.actor(state, deterministic=deterministic)
            option_idx = torch.argmax(logits, dim=-1).item()
        else:
            logits = torch.tensor(np.full((1, self.action_dim), np.nan)).to(self.device)
            metaData = {
                "probs": torch.tensor(np.nan).to(self.device),
                "logprobs": torch.tensor(np.nan).to(self.device),
                "dist": torch.tensor(np.nan).to(self.device),
            }

        is_option = True if option_idx in self.trainable_options else False
        if is_option:
            a, infos = self.policies[option_idx].actor(state, deterministic=True)

            with torch.no_grad():
                value1 = self.policies[option_idx].critic1(state)
                value2 = self.policies[option_idx].critic2(state)
                value = torch.min(value1, value2)  # take the minimum of the two critics

                value = (infos["probs"] * value).sum(dim=-1, keepdim=True)

            option_termination = True if value.item() < 0 else False
        else:
            a, _ = self.policies[option_idx](state, deterministic=True)
            option_termination = False

        return [option_idx, a], {
            "logits": logits,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "dist": metaData["dist"],
            "is_option": is_option,
            "option_termination": option_termination,
        }


class HRL_SAC_Learner(SAC_Learner):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # constants
        self.name = "HRL_SAC"

        #
        self.to(self.dtype).to(self.device)

    def update_options(self, policies):
        """Update the policies with new options."""
        self.policies = policies
        self.trainable_options = [
            i for i, policy in enumerate(self.policies) if hasattr(policy, "actor")
        ]

    def forward(
        self, state: np.ndarray, option_idx: int | None, deterministic: bool = False
    ):
        state = self.preprocess_state(state)
        if option_idx is None:
            logits, metaData = self.actor(state, deterministic=deterministic)
            option_idx = torch.argmax(logits, dim=-1).item()
        else:
            logits = torch.tensor(np.full((1, self.action_dim), np.nan)).to(self.device)
            metaData = {
                "probs": torch.tensor(np.nan).to(self.device),
                "logprobs": torch.tensor(np.nan).to(self.device),
                "dist": torch.tensor(np.nan).to(self.device),
            }

        is_option = True if option_idx in self.trainable_options else False
        if is_option:
            a, infos = self.policies[option_idx].actor(state, deterministic=True)

            with torch.no_grad():
                value1 = self.policies[option_idx].critic1(state)
                value2 = self.policies[option_idx].critic2(state)
                value = torch.min(value1, value2)  # take the minimum of the two critics

                value = (infos["probs"] * value).sum(dim=-1, keepdim=True)

            option_termination = True if value.item() < 0 else False
        else:
            a, _ = self.policies[option_idx](state, deterministic=True)
            option_termination = False

        return [option_idx, a], {
            "logits": logits,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "dist": metaData["dist"],
            "is_option": is_option,
            "option_termination": option_termination,
        }
