import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class Base(nn.Module):
    def __init__(self, device):
        super(Base, self).__init__()

        self.dtype = torch.float32
        self.device = device

        # utils
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss

        self.state_visitation = None

    def print_parameter_devices(self, model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.device}")

    def to_device(self, device):
        self.device = device
        # because actor is coded to be independent nn.Module for decision-making
        if hasattr(self, "actor"):
            self.actor.device = device
        if hasattr(self, "policies"):
            for policy in self.policies:
                if policy is not None:
                    policy.device = device
                    if hasattr(policy, "actor"):
                        policy.actor.device = device
        self.to(device)

    def preprocess_state(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Convert input state to a 2D or 4D torch.Tensor on the correct device and dtype.
        - 2D: (B, D) for vector states
        - 4D: (B, C, H, W) for image states
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        elif not isinstance(state, torch.Tensor):
            raise ValueError("Unsupported state type. Must be a tensor or numpy array.")

        state = state.to(self.device).to(self.dtype)

        # Ensure batch dimension exists
        if state.ndim in [1, 3]:  # (D) or (C, H, W)
            state = state.unsqueeze(0)

        # Final shape control
        if state.ndim == 2:  # (B, D) -> vector input
            return state
        elif state.ndim == 4:  # (B, C, H, W) -> image input
            return state.view(state.size(0), -1)
        else:
            raise ValueError(
                f"Unsupported state shape {state.shape}, expected 2D or 4D."
            )

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        if (
                            param.grad is not None
                        ):  # Only consider parameters that have gradients
                            param_grad_norm = torch.norm(param.grad, p=norm_type)
                            total_norm += param_grad_norm**norm_type
                except:
                    try:
                        param_grad_norm = torch.norm(model.grad, p=norm_type)
                    except:
                        param_grad_norm = torch.tensor(0.0)
                    total_norm += param_grad_norm**norm_type

                total_norm = total_norm ** (1.0 / norm_type)
                grad_dict[dir + "/grad/" + names[i]] = total_norm.item()

        return grad_dict

    def compute_weight_norm(self, models, names, device, dir="None", norm_type=2):
        norm_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        param_norm = torch.norm(param, p=norm_type)
                        total_norm += param_norm**norm_type
                except:
                    param_norm = torch.norm(model, p=norm_type)
                    total_norm += param_norm**norm_type
                total_norm = total_norm ** (1.0 / norm_type)
                norm_dict[dir + "/weight/" + names[i]] = total_norm.item()

        return norm_dict

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict

    def flat_grads(self, grads: tuple):
        """
        Flatten the gradients into a single tensor.
        """
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad
