import gymnasium as gym
import torch
import torch.nn as nn

from utils.wrapper import GridWrapper, ObsNormWrapper


def call_env(args, verbose=True):
    """
    Call the environment based on the given name.
    """

    env = gym.make(args.env_name, render_mode="rgb_array")

    args.state_dim = env.observation_space.shape[0]
    args.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    args.action_dim = (
        env.action_space.n if args.is_discrete else env.action_space.shape[0]
    )
    args.episode_len = env.spec.max_episode_steps
    args.batch_size = 20 * args.episode_len
    args.minibatch_size = args.batch_size // args.num_minibatch
    args.nupdates = args.timesteps // args.batch_size

    if args.is_discrete:
        # it argmax the onehot
        env = GridWrapper(env)

    env = ObsNormWrapper(env)

    if verbose:
        print("────────────────────────────")
        print("🧠  Environment Summary")
        print("────────────────────────────")
        print(f"📏  State Dimension  : {args.state_dim}")
        print(f"🎮  Action Dimension : {args.action_dim}")
        print(
            f"🔀  Action Type      : {'Discrete' if args.is_discrete else 'Continuous'}"
        )
        print(f"🔀  Episode Length   : {args.episode_len}")
        print("────────────────────────────\n")

    return env


def estimate_advantages(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate advantages and returns using Generalized Advantage Estimation (GAE),
    while keeping all operations on the original device.

    Args:
        rewards (Tensor): Reward at each timestep, shape [T, 1]
        terminals (Tensor): Binary terminal indicators (1 if done), shape [T, 1]
        values (Tensor): Value function estimates, shape [T, 1]
        gamma (float): Discount factor.
        gae (float): GAE lambda.

    Returns:
        advantages (Tensor): Estimated advantages, shape [T, 1]
        returns (Tensor): Estimated returns (value targets), shape [T, 1]
    """
    device = rewards.device  # Infer device from input tensor

    T = rewards.size(0)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    prev_value = torch.tensor(0.0, device=device)
    prev_advantage = torch.tensor(0.0, device=device)

    for t in reversed(range(T)):
        non_terminal = 1.0 - terminals[t]
        deltas[t] = rewards[t] + gamma * prev_value * non_terminal - values[t]
        advantages[t] = deltas[t] + gamma * gae * prev_advantage * non_terminal

        prev_value = values[t]
        prev_advantage = advantages[t]

    returns = values + advantages
    return advantages, returns


def get_extractor(args):
    from extractor.extractor import ALLO
    from policy.layers.building_blocks import MLP

    # === CREATE FEATURE EXTRACTOR === #
    feature_network = MLP(
        state_dim=len(args.positional_indices),  # discrete position is always 2d
        feature_dim=(10 // 2 + 1),
        encoder_fc_dim=[512, 512, 512],
        activation=nn.Tanh(),
    )

    # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
    extractor = ALLO(
        network=feature_network,
        positional_indices=args.positional_indices,
        extractor_lr=args.extractor_lr,
        epochs=args.extractor_epochs,
        batch_size=1024,
        device=args.device,
    )

    return extractor


def get_vector(env, extractor, args):
    # ALLO does not have explicit eigenvectors.
    # Instead, we make list that contains the eigenvector index and sign
    eigenvectors = [(n // 2 + 1, 2 * (n % 2) - 1) for n in range(args.num_options)]

    heatmaps = env.get_rewards_heatmap(extractor, eigenvectors)

    return eigenvectors, heatmaps


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        num_param = p.numel()
        p.data.copy_(flat_params[pointer : pointer + num_param].view_as(p))
        pointer += num_param


def compute_kl(old_actor, new_policy, obs):
    _, old_infos = old_actor(obs)
    _, new_infos = new_policy(obs)

    kl = torch.distributions.kl_divergence(old_infos["dist"], new_infos["dist"])
    return kl.mean()


def hessian_vector_product(kl_fn, model, damping, v):
    kl = kl_fn()
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    g_v = (flat_grads * v).sum()
    hv = torch.autograd.grad(g_v, model.parameters())
    flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
    return flat_hv + damping * v


def conjugate_gradients(Av_func, b, nsteps=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = Av_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x
