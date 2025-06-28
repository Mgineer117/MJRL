import datetime
import os
import random
import uuid

import torch

import wandb
from algorithms import *
from utils.functions import concat_csv_columnwise_and_delete, seed_all, setup_logger
from utils.get_args import get_args
from utils.rl import call_env

# it suppresses the wandb printing when it logs data
os.environ["WANDB_SILENT"] = "true"


def run(args, seed, exp_time):
    # fix seed
    seed_all(seed)

    # get env
    env = call_env(args)
    logger, writer = setup_logger(args, exp_time, seed)

    # algorithm_map.py (or define in same script)
    ALGO_MAP = {
        "ppo": PPO_Algorithm,
        "trpo": TRPO_Algorithm,
        "ddpg": DDPG_Algorithm,
        "psne": PSNE_Algorithm,
        "drndppo": DRND_Algorithm,
        "hrl": HRL,
    }

    # instantiate algorithm
    try:
        algo_cls = ALGO_MAP[args.algo_name.lower()]
        algo = algo_cls(env=env, logger=logger, writer=writer, args=args)
    except KeyError:
        raise NotImplementedError(f"Algorithm '{args.algo_name}' is not implemented.")

    algo.begin_training()

    # ✅ Memory cleanup
    del algo, env, logger, writer  # delete large references
    torch.cuda.empty_cache()  # release unreferenced GPU memory
    wandb.finish()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    args = get_args()

    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(args.seed)
    seeds = [random.randint(1, 100_000) for _ in range(args.num_runs)]
    print(f"      Running ID: {args.unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")

    for seed in seeds:
        args.seed = seed
        run(args, seed, exp_time)

    concat_csv_columnwise_and_delete(folder_path=args.logdir)
