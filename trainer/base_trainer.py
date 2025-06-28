import os
from abc import ABC, abstractmethod

import numpy as np


class BaseTrainer(ABC):
    def __init__(self) -> None:
        pass  # or actual initialization code

    @abstractmethod
    def train(self) -> dict[str, float]:
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def average_discounted_return(self, rewards, terminals, gamma):
        episode_returns = []
        G = 0.0
        for r, done in zip(rewards, terminals):
            G = r + gamma * G
            if done:
                episode_returns.append(G)
                G = 0.0  # Reset for next episode

        if not episode_returns:
            return 0.0
        return sum(episode_returns) / len(episode_returns)

    def discounted_return(self, rewards, gamma):
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        if len(image) > 0:
            tensor = np.stack(image, axis=0)
            video_path = os.path.join(logdir, name)
            self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    @abstractmethod
    def save_model(self, e):
        pass
