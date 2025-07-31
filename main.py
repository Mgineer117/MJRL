import datetime
import os
import random

import torch

import wandb
from algorithms import *
from utils.functions import (
    add_specific_args,
    concat_csv_columnwise_and_delete,
    seed_all,
    setup_logger,
)
from utils.get_args import get_args
from utils.rl import call_env

# it suppresses the wandb printing when it logs data
os.environ["WANDB_SILENT"] = "true"


def run(args, seed, exp_time):
    # fix seed
    seed_all(seed)

    # use env-specific parameters

    # get env
    env = call_env(args)
    args = add_specific_args(args)
    logger, writer = setup_logger(args, exp_time, seed)

    # algorithm_map.py (or define in same script)
    ALGO_MAP = {
        "ppo": PPO_Algorithm,
        "trpo": TRPO_Algorithm,
        "ddpg": DDPG_Algorithm,
        "sac": SAC_Algorithm,
        "psne": PSNE_Algorithm,
        "drndppo": DRND_Algorithm,
        "hrl_allo": HRL_ALLO,
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

    init_args = get_args()

    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(init_args.seed)
    seeds = [random.randint(1, 100_000) for _ in range(init_args.num_runs)]
    print(f"      Running ID: {init_args.unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")

    for seed in seeds:
        args = get_args(verbose=False)
        args.seed = seed
        args.unique_id = init_args.unique_id
        run(args, seed, exp_time)

    concat_csv_columnwise_and_delete(folder_path=init_args.logdir)
