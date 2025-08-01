import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatten_space

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
        state, reward, termination, truncation, info = self.env.step(np.argmax(action))

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


class PointMazeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, maze_map: list, episode_len: int, seed: int):
        super(PointMazeWrapper, self).__init__(env)

        self.maze_map = maze_map
        self.max_steps = episode_len
        self.seed = seed

        self.observation_space = flatten_space(env.observation_space)

    def reset(self, **kwargs):
        observation_dict, info = self.env.reset(**kwargs)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )

        return observation, info

    def step(self, action):
        # Call the original step method
        observation_dict, reward, termination, truncation, info = self.env.step(action)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )
        return observation, reward, termination, truncation, info

    def get_rewards_heatmap(self, extractor: torch.nn.Module, eigenvectors: np.ndarray):
        # Get desired goal (2D)
        state, _ = self.reset(seed=self.seed)
        dg = state[-4:-2]
        del state
        self.close()

        # Maze size
        example_map = self.maze_map
        maze_height = len(example_map)
        maze_width = len(example_map[0])

        # Spatial bounds (MuJoCo centered coordinates)
        cell_size = 1.0
        x_low = -maze_width * cell_size / 2
        x_high = maze_width * cell_size / 2
        y_low = -maze_height * cell_size / 2
        y_high = maze_height * cell_size / 2
        resolution = 80

        x_vals = np.linspace(x_low, x_high, num=resolution)
        y_vals = np.linspace(y_low, y_high, num=resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
        states = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

        self.width = resolution
        self.height = resolution

        # Create wall/goal/agent masks
        wall_mask = np.zeros((resolution, resolution), dtype=bool)
        goal_mask = np.zeros((resolution, resolution), dtype=bool)
        agent_mask = np.zeros((resolution, resolution), dtype=bool)

        for i in range(maze_height):
            for j in range(maze_width):
                val = example_map[i][j]

                # MuJoCo coordinate of cell center
                x_center = (j + 0.5) * cell_size - (maze_width * cell_size / 2)
                y_center = (maze_height * cell_size / 2) - (i + 0.5) * cell_size

                x_start = x_center - cell_size / 2
                x_end = x_center + cell_size / 2
                y_start = y_center - cell_size / 2
                y_end = y_center + cell_size / 2

                region_mask = (
                    (X_grid >= x_start)
                    & (X_grid < x_end)
                    & (Y_grid >= y_start)
                    & (Y_grid < y_end)
                )

                if val == 1:
                    wall_mask |= region_mask
                elif val == "g":
                    goal_mask |= region_mask
                elif val == "r":
                    agent_mask |= region_mask

        valid_mask = ~wall_mask

        # Loop over each eigenvector to generate heatmaps
        images = []
        with torch.no_grad():
            intrinsic_rewards, _ = extractor(states)
        intrinsic_rewards = intrinsic_rewards.cpu().numpy()

        for eigenvector_idx, eigenvector_sign in eigenvectors:
            rewards = eigenvector_sign * intrinsic_rewards[:, eigenvector_idx]

            # Normalize rewards separately for positive and negative
            neg_idx = rewards < 0
            pos_idx = rewards >= 0

            if np.any(pos_idx):
                pos_max, pos_min = rewards[pos_idx].max(), rewards[pos_idx].min()
                if pos_max != pos_min:
                    rewards[pos_idx] = (rewards[pos_idx] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )
                else:
                    rewards[pos_idx] = 1
            if np.any(neg_idx):
                neg_max, neg_min = rewards[neg_idx].max(), rewards[neg_idx].min()
                if neg_max != neg_min:
                    rewards[neg_idx] = (rewards[neg_idx] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0
                else:
                    rewards[neg_idx] = -1

            reward_map = rewards.reshape(resolution, resolution)
            rgb_img = self.reward_map_to_rgb(reward_map, valid_mask)
            images.append(rgb_img)

        return images

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float32)

        pos_mask = np.logical_and(mask, (reward_map >= 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask, 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask, 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask, :] = 0.5

        return rgb_img