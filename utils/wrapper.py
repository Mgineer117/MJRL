import gymnasium as gym
import numpy as np
import torch


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

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        return state, info

    def step(self, action):
        # Call the original step method
        state, reward, termination, truncation, info = self.env.step(action)

        return state, reward, termination, truncation, info

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)

    def get_rewards_heatmap(
        self, extractor, eigenvector_signs: list[int], eigenvectors: list[torch.Tensor]
    ):
        state_representation = self.env.unwrapped.state_representation
        state = self.env.unwrapped.get_grid()
        width, height = self.env.unwrapped.width, self.env.unwrapped.height

        assert state_representation in [
            "vectorized_tensor",
            "tensor",
            "positional",
        ], f"Unsupported state representation: {state_representation}"

        # Environment indices
        empty_idx = 1
        goal_idx = 8
        agent_idx = 10
        obs_idx = 13
        wall_idx = 2

        # Get base state

        agent_pos = np.where(state == agent_idx)
        state[agent_pos] = empty_idx
        grid = state

        mask = (grid != wall_idx) & (grid != goal_idx)

        # Get coordinates where agent can be placed
        valid_coords = np.argwhere(mask)  # shape: [num_valid, 2]
        goal_coords = np.argwhere(grid == goal_idx)[0]

        # Generate a batch of states
        state_batch = []
        for coord in valid_coords:
            if state_representation in ("tensor", "vectorized_tensor"):
                new_grid = grid.copy()
                new_grid[new_grid[..., 0] == agent_idx] = empty_idx
                new_grid[coord[0], coord[1], 0] = agent_idx

                state_batch.append(new_grid)
            elif state_representation == "positional":
                state = np.array([coord[0], coord[1], goal_coords[0], goal_coords[1]])
                state_batch.append(state)

        # Stack the batch: shape = [num_valid, H, W] or [num_valid, H, W, C]
        state_batch = np.stack(state_batch)

        heatmaps = []
        grid_shape = (width, height, 1)
        for n in range(len(eigenvectors)):
            reward_map = np.full(grid_shape, fill_value=0.0)

            with torch.no_grad():
                features = extractor(state_batch)  # .cpu().numpy()

            eigenvector_sign = eigenvector_signs[n]
            eigenvector = eigenvectors[n]
            reward = eigenvector_sign * (features @ eigenvector.unsqueeze(-1))
            reward = reward.cpu().numpy().squeeze()

            for i in range(features.shape[0]):
                if state_representation in ("tensor", "vectorized_tensor"):
                    agent_pos = np.argwhere(state_batch[i] == agent_idx)[0]
                elif state_representation == "positional":
                    agent_pos = [state_batch[i][0], state_batch[i][1]]
                x, y = agent_pos[0], agent_pos[1]

                reward_map[x, y, 0] = reward[i]

            # reward_map = # normalize between -1 to 1
            pos_mask = np.logical_and(mask, (reward_map > 0))
            neg_mask = np.logical_and(mask, (reward_map < 0))

            # Normalize positive values to [0, 1]
            if np.any(pos_mask):
                pos_max, pos_min = (
                    reward_map[pos_mask].max(),
                    reward_map[pos_mask].min(),
                )
                if pos_max != pos_min:
                    reward_map[pos_mask] = (reward_map[pos_mask] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )

            # Normalize negative values to [-1, 0]
            if np.any(neg_mask):
                neg_max, neg_min = (
                    reward_map[neg_mask].max(),
                    reward_map[neg_mask].min(),
                )
                if neg_max != neg_min:
                    reward_map[neg_mask] = (reward_map[neg_mask] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0

            # Set all other entries (walls, empty) to 0
            # print(reward_map[:, :, 0])
            reward_map = self.reward_map_to_rgb(reward_map, mask)

            # set color theme as blue and red (blue = -1 and red = 1)
            # set wall color at value 0 and goal idx as 1
            heatmaps.append(reward_map)

        return heatmaps

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        width, height = self.env.unwrapped.width, self.env.unwrapped.height
        rgb_img = np.zeros((width, height, 3), dtype=np.float32)

        pos_mask = np.logical_and(mask, (reward_map > 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask[:, :, 0], 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask[:, :, 0], 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask[:, :, 0], :] = 0.5

        return rgb_img
