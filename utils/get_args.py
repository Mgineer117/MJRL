import argparse
import uuid

import torch


def get_args():
    parser = argparse.ArgumentParser(description="")

    # === ENV PARAMETER === #
    parser.add_argument(
        "--env-name", type=str, default="Ant-v5", help="Name of the model."
    )
    parser.add_argument("--algo-name", type=str, default="ppo", help="Disable cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Batch size.")

    # === TRAINING PARAMETER === #
    parser.add_argument(
        "--hl-timesteps", type=int, default=1_000_000, help="Number of training epochs."
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Number of training epochs."
    )
    parser.add_argument(
        "--extractor-epochs",
        type=int,
        default=50000,
        help="Number of training epochs.",
    )
    parser.add_argument("--num-minibatch", type=int, default=4, help="")

    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of samples for training."
    )

    # === ALGORITHMIC PARAMETER === #
    parser.add_argument(
        "--actor-fc-dim", type=list, default=[256, 256], help="Base learning rate."
    )
    parser.add_argument(
        "--critic-fc-dim", type=list, default=[256, 256], help="Base learning rate."
    )
    parser.add_argument(
        "--extractor-lr", type=float, default=1e-3, help="Base learning rate."
    )
    parser.add_argument(
        "--actor-lr", type=float, default=1e-4, help="Base learning rate."
    )
    parser.add_argument(
        "--critic-lr", type=float, default=1e-4, help="Base learning rate."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Base learning rate.")

    parser.add_argument("--K-epochs", type=int, default=5, help="")
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="Upper bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Lower bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--entropy-scaler", type=float, default=1e-3, help="Base learning rate."
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="Base learning rate."
    )
    parser.add_argument(
        "--buffer-size", type=int, default=200_000, help="Base learning rate."
    )
    parser.add_argument(
        "--warmup-samples", type=int, default=1_000, help="Base learning rate."
    )
    parser.add_argument(
        "--action-noise-coeff", type=float, default=0.1, help="Base learning rate."
    )
    parser.add_argument("--tau", type=float, default=0.005, help="Base learning rate.")
    parser.add_argument(
        "--policy-freq", type=int, default=2, help="Base learning rate."
    )

    # === HRL PARAMETER === #
    parser.add_argument(
        "--num-options", type=int, default=None, help="Number of samples for training."
    )

    # === LOGGING PARAMETER === #
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--log-interval", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval-num", type=int, default=10, help="Number of training epochs."
    )

    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Path to a directory for storing the log.",
    )
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Path to a directory for storing the log.",
    )

    parser.add_argument(
        "--gpu-idx", type=int, default=0, help="Number of training epochs."
    )

    args = parser.parse_args()
    args.device = select_device(args.gpu_idx)

    unique_id = str(uuid.uuid4())[:4]
    args.unique_id = unique_id

    return args


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device
