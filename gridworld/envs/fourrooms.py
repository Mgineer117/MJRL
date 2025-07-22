import random
from itertools import chain
from typing import (
    Any,
    Final,
    Iterable,
    Literal,
    SupportsFloat,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray

from gridworld.core.agent import Agent, AgentT, GridActions, PolicyAgent
from gridworld.core.constants import *
from gridworld.core.grid import Grid
from gridworld.core.object import Goal, Lava, Obstacle, Wall
from gridworld.core.world import GridWorld
from gridworld.multigrid import MultiGridEnv
from gridworld.policy.ctf.heuristic import (
    HEURISTIC_POLICIES,
    CtfPolicyT,
    RoombaPolicy,
    RwPolicy,
)
from gridworld.typing import Position
from gridworld.utils.window import Window


class FourRooms(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_type: int,
        max_steps: int,
        spawn_agent_random: bool = False,
        num_random_agent: int = 0,
        highlight_visible_cells: bool = False,
        tile_size: int = 10,
        state_representation: str = "positional",
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        render_fps: int = 10,
    ):
        self.state_representation = state_representation
        self.spawn_agent_random = spawn_agent_random
        if grid_type < 0 or grid_type >= 3:
            raise ValueError(
                f"The Fourroom only accepts grid_type of 0 and 1, given {grid_type}"
            )
        else:
            self.grid_type = grid_type

        self.max_steps = max_steps

        self.world = GridWorld
        self.actions_set = GridActions

        see_through_walls: bool = False

        self.agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
                actions=self.actions_set,
                type="agent",
            )
        ]

        # Define positions for goals and agents
        self.goal_positions = [(9, 3), (11, 1)]
        self.agent_positions = [(3, 9), (7, 9)]

        self.num_random_agent = num_random_agent
        self.random_agent_positions = [
            [(6, 5), (6, 14)],
        ]

        for i in range(1, 1 + self.num_random_agent):
            self.agents.append(
                PolicyAgent(
                    RwPolicy(action_set=self.actions_set),
                    self.world,
                    index=i,
                    color="red",
                    bg_color="light_red",
                    actions=self.actions_set,
                    type="obstacle",
                )
            )

        self.grids = {}
        self.grid_imgs = {}
        # Explicit maze structure based on the image
        self.map_structure = [
            "#############",
            "#    #      #",
            "#    #      #",
            "#           #",
            "#    #      #",
            "#    #      #",
            "## ###### ###",
            "#     #     #",
            "#     #     #",
            "#     #     #",
            "#           #",
            "#     #     #",
            "#############",
        ]

        self.width = len(self.map_structure[0])
        self.height = len(self.map_structure)
        self.grid_size = (self.width, self.height)

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
            render_fps=render_fps,
            highlight_visible_cells=highlight_visible_cells,
            tile_size=tile_size,
        )

    def get_grid(self):
        self.reset()
        grid = self.grid.encode()
        self.close()
        return grid

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.state_representation:
            case "positional":
                observation_space = spaces.Box(
                    low=np.array([0, 0, 0, 0], dtype=np.float32),
                    high=np.array(
                        [self.width, self.height, self.width, self.height],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
            case "tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=13,
                    shape=(self.width, self.height, self.world.encode_dim),
                    dtype=np.int64,
                )
            case "vectorized_tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=10,
                    shape=(self.width * self.height * self.world.encode_dim,),
                    dtype=np.int64,
                )
            case _:
                raise ValueError(
                    f"Invalid state representation: {self.state_representation}"
                )

        return observation_space

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for y, row in enumerate(self.map_structure):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                elif cell == " ":
                    self.grid.set(x, y, None)

        # Place the goal
        goal = Goal(self.world, index=4)
        self.put_obj(goal, *self.goal_positions[self.grid_type])
        goal.init_pos, goal.cur_pos = self.goal_positions[self.grid_type]

        # place agent
        if self.spawn_agent_random:
            coords = self.find_obj_coordinates(None)
            agent_positions = random.sample(coords, 1)[0]
        else:
            agent_positions = self.agent_positions[self.grid_type]
        self.place_agent(self.agents[0], pos=agent_positions)

        if len(self.agents) > 1:
            for i, agent in enumerate(self.agents[1:]):
                self.place_agent(
                    agent, pos=self.random_agent_positions[self.grid_type][i]
                )

    def find_obj_coordinates(self, obj) -> tuple[int, int] | None:
        """
        Finds the coordinates (i, j) of the first occurrence of None in the grid.
        Returns None if no None value is found.
        """
        coord_list = []
        for index, value in enumerate(self.grid.grid):
            if value is obj:
                # Calculate the (i, j) coordinates from the 1D index
                i = index % self.width
                j = index // self.width
                coord_list.append((i, j))
        return coord_list

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict = {},
    ):
        super().reset(seed=seed, options=options)

        ### NOTE: NOT MULTIAGENT SETTING
        observations = self.get_obs()
        info = {"success": False}

        return observations, info

    def step(self, action):
        self.step_count += 1

        action = np.argmax(action)

        reward = 0.0
        info = {"success": False}
        done = False

        agent = self.agents[0]

        curr_pos = agent.pos

        move_map = {
            self.actions.left: (0, -1),
            self.actions.right: (0, +1),
            self.actions.up: (-1, 0),
            self.actions.down: (+1, 0),
        }

        assert action in move_map, f"Unknown action {action}"
        delta = move_map[action]
        fwd_pos = tuple(a + b for a, b in zip(curr_pos, delta))
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is not None:
            if agent.type == "agent":
                if fwd_cell.type == "goal":
                    done = True
                    reward += 1.0
                    info["success"] = True
                elif fwd_cell.type == "lava":
                    done = True
                    reward += -1.0
                    info["success"] = False
        elif fwd_cell is None or fwd_cell.can_overlap():
            self.grid.set(*agent.pos, None)
            self.grid.set(*fwd_pos, agent)
            agent.pos = fwd_pos

        self._handle_special_moves(0, reward, fwd_pos, fwd_cell)

        terminated = done
        truncated = self.step_count >= self.max_steps
        state = self.get_obs()

        env_info = {
            "achieved_goal": np.array(
                [self.agents[0].pos[0], self.agents[0].pos[1]], dtype=np.float32
            ),
            "desired_goal": np.array(
                [
                    self.goal_positions[self.grid_type][0],
                    self.goal_positions[self.grid_type][1],
                ],
                dtype=np.float32,
            ),
        }
        info.update(env_info)

        return state, reward, terminated, truncated, info

    def get_obs(
        self,
    ):
        if self.state_representation == "positional":
            obs = np.array(
                [
                    self.agents[0].pos[0],
                    self.agents[0].pos[1],
                    self.goal_positions[self.grid_type][0],
                    self.goal_positions[self.grid_type][1],
                ],
                dtype=np.float32,
            )

        elif self.state_representation == "tensor":
            obs = self.grid.encode()
        elif self.state_representation == "vectorized_tensor":
            obs = self.grid.encode().flatten()
        else:
            raise ValueError(
                f"Unknown state representation {self.state_representation}. "
                "Please use 'positional' or 'tensor'."
            )
        return obs
