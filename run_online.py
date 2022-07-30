from utils import ReplayBuffer, PrioritizedReplayBuffer, linear_schedule, get_logger
from env_utils import MinAtarEnv
import numpy as np
import pandas as pd
import os
import time
import torch
from tqdm import trange
from models import DQNAgent, DDQNAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ALGOS = {"dqn": DQNAgent, "ddqn": DDQNAgent}


def eval_policy(agent: DQNAgent,
                env_name: str,
                eval_episodes: int = 10):
    eval_env = MinAtarEnv(env_name)
    avg_reward = 0.
    act_counts = np.zeros(eval_env.action_space.n)
    t = 0
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            action = agent.sample_action(obs)
            obs, reward, done, _ = eval_env.step(action)
            act_counts[action] += 1
            avg_reward += reward
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    return avg_reward, act_counts


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="breakout")
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--update_step", type=int, default=4)
    parser.add_argument("--warmup_timesteps", type=int, default=5000)
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--eval_freq", type=int, default=int(1e4))
    parser.add_argument("--ckpt_freq", type=int, default=int(5e4)) 
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--no_per", action="store_true", default=False)
    args = parser.parse_args()
    return args


def run(args):
    print(f"Start to play {args.env_name}")
    start_time = time.time()
    exp_name = f"{args.algo}_s{args.seed}_b{args.batch_size}"
    logger = get_logger(f"logs/online/{args.env_name}/{exp_name}.log")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize MinAtar environment
    env = MinAtarEnv(args.env_name)
    obs_shape = env.observation_space  # (4, 10, 10)
    in_channels = obs_shape[0]         # 4
    act_dim = env.action_space.n       # 6

    # initialize DQN agent
    agent = ALGOS[args.algo](in_channels, act_dim, args, device)

    # initialize the replay buffer
    if args.no_per:
        replay_buffer = ReplayBuffer(obs_shape, args.buffer_size)
    else:
        replay_buffer = PrioritizedReplayBuffer(obs_shape, args.buffer_size)

    # start training
    obs, done = env.reset(), False    # (4, 10, 10)
    res = []
    for t in trange(1, args.total_timesteps+1):
        # epsilon exploration
        epsilon = linear_schedule(start_epsilon=0.5, end_epsilon=0.05,
                                  duration=args.total_timesteps, t=t)

        # sample action
        if t <= args.warmup_timesteps:
            action = np.random.choice(act_dim)
        else:
            if np.random.random() < epsilon:
                action = np.random.choice(act_dim)
            else:
                action = agent.sample_action(obs) 
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, done)
        obs = next_obs

        # update the agent
        if t > args.warmup_timesteps:
            batch = replay_buffer.sample(batch_size=args.batch_size)
            log_info = agent.update(batch)
            if not args.no_per:
                replay_buffer.update_priority(batch.idx, log_info["priority"])

        if t % args.eval_freq == 0:
            eval_reward, act_counts = eval_policy(agent, args.env_name) 
            act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
            logger.info(
                f"Step {t}: reward={eval_reward}, total_time={(time.time()-start_time)/60:.2f}min\n"
                f"\tavg_td_loss: {log_info['avg_td_loss']:.3f}, max_td_loss: {log_info['max_td_loss']:.3f}, "
                f"min_td_loss: {log_info['min_td_loss']:.3f}\n"
                f"\tavg_Q: {log_info['avg_Q']:.3f}, max_Q: {log_info['max_Q']:.3f}, "
                f"min_Q: {log_info['min_Q']:.3f}\n"
                f"\tavg_target_Q: {log_info['avg_target_Q']:.3f}, max_target_Q: {log_info['max_target_Q']:.3f}, "
                f"min_target_Q: {log_info['min_target_Q']:.3f}\n"
                f"\tavg_batch_rewards: {batch.rewards.mean():.3f}, max_batch_rewards: {batch.rewards.max():.3f}, "
                f"min_batch_rewards: {batch.rewards.min():.3f}\n"
                f"\tact_counts: ({act_counts}), epsilon: {epsilon:.3f}\n"
                f"\tavg_priority: {log_info['avg_priority']:.3f}, "
                f"max_priority: {log_info['max_priority']:.3f}, "
                f"min_priority: {log_info['min_priority']:.3f}\n"
            )
            log_info.update({"step": t, "eval_reward": eval_reward})
            res.append(log_info)
            if t % args.ckpt_freq == 0:
                agent.save(f"saved_models/online/{args.env_name}/{args.algo}/{t//args.ckpt_freq}.ckpt")

        if done:
            obs = env.reset()

    # save logs
    df = pd.DataFrame(res).set_index("step")
    df.to_csv(f"logs/online/{args.env_name}/{exp_name}.csv")

    # save replay buffer
    replay_buffer.save(f"datasets/{args.env_name}")


if __name__ == "__main__":
    args = get_args()
    # for env_name in ["breakout", "asterix", "freeway", "space_invaders", "seaquest"]:
    for env_name in ["asterix", "breakout"]:
        args.env_name = env_name
        os.makedirs(f"saved_models/online/{args.env_name}/{args.algo}", exist_ok=True)
        os.makedirs(f"logs/online/{args.env_name}", exist_ok=True)
        os.makedirs(f"datasets", exist_ok=True)
        run(args)
