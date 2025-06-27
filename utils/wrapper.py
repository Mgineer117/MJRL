from io import BytesIO

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        tot_count = self.count + count

        new_mean = self.mean + delta * count / tot_count
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + np.square(delta) * self.count * count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ObsNormWrapper(gym.ObservationWrapper):
    def __init__(self, env, clip_obs=10.0, epsilon=1e-8):
        super().__init__(env)
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        obs_shape = self.observation_space.shape
        self.rms = RunningMeanStd(shape=obs_shape)

    def observation(self, obs):
        self.rms.update(obs[np.newaxis, ...])
        norm_obs = (obs - self.rms.mean) / (np.sqrt(self.rms.var) + self.epsilon)
        return np.clip(norm_obs, -self.clip_obs, self.clip_obs)

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)


class GridWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(GridWrapper, self).__init__(env)

    def step(self, action):
        # Call the original step method
        state, reward, termination, truncation, info = self.env.step(np.argmax(action))

        return state, reward, termination, truncation, info

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)
