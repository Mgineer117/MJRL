import os

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


def main():
    # Create save directory
    log_dir = "sac_hopper_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create training environment
    train_env = make_vec_env("Hopper-v5", n_envs=1, seed=42)
    # train_env = Monitor(train_env)

    # Create evaluation environment
    eval_env = gym.make("Hopper-v5")
    # eval_env = Monitor(eval_env)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=log_dir, name_prefix="sac_hopper_checkpoint"
    )

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.2",  # Automatically tune entropy coefficient
        # ent_coef="auto_0.2",  # Automatically tune entropy coefficient
    )

    # Train
    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_callback, checkpoint_callback],
    )

    # Save final model
    model.save(os.path.join(log_dir, "final_model"))

    print("Training completed and model saved.")


if __name__ == "__main__":
    main()
