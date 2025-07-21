from gymnasium.envs.registration import register

"""
    Just adding environments I frequently use.
    Some details should be addressed such as env state representation and action set (ctf)    
"""


EPI_LENGTH = {"Fourrooms-v0": 100, "CtF-v0": 200}

register(
    id="Fourrooms-v0",
    entry_point="gridworld.envs.fourrooms:FourRooms",
    kwargs={"grid_type": 0, "max_steps": EPI_LENGTH["Fourrooms-v0"]},
    max_episode_steps=EPI_LENGTH["Fourrooms-v0"],
)

register(
    id="CtF-v0",
    entry_point="gridworld.envs.ctf:CtF",
    kwargs={"grid_type": 0, "max_steps": EPI_LENGTH["CtF-v0"]},
    max_episode_steps=EPI_LENGTH["CtF-v0"],
)
