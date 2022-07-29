from utils import ReplayBuffer, linear_schedule
from env_utils import MinAtarEnv
import numpy as np
import pandas as pd
import os
import torch
from tqdm import trange
from models import BCAgent, CQLAgent, DQNAgent, DQNBCAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ALGOS = {
    "bc": BCAgent,
    "cql": CQLAgent,
    "dqn": DQNAgent,
    "dqnbc": DQNBCAgent,
}


def eval_policy(agent: DQNAgent,
                env_name: str,
                eval_episodes: int = 10):
    eval_env = MinAtarEnv(env_name)
    avg_reward = 0.
    t = 0
    for _ in trange(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            action = agent.sample_action(obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="breakout")
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--update_step", type=int, default=4)
    parser.add_argument("--warmup_timesteps", type=int, default=5000)
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--eval_freq", type=int, default=int(1e4))
    parser.add_argument("--ckpt_freq", type=int, default=int(1e5))
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    args = parser.parse_args()
    return args


def run(args):
    print(f"Start to play {args.env_name}")
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
    replay_buffer = ReplayBuffer(obs_shape, args.buffer_size)
    replay_buffer.load()

    res = []
    # start training
    for t in trange(1, args.total_timesteps+1):
        batch = replay_buffer.sample(batch_size=args.batch_size)
        log_info = agent.update(batch)

        if t % args.eval_freq == 0:
            eval_reward = eval_policy(agent, args.env_name)
            print(f"\nEvaluate agent at step {t}: reward={eval_reward}\n")
            log_info.update({"step": t, "eval_reward": eval_reward})
            res.append(log_info)
            if t % args.ckpt_freq == 0:
                agent.save(f"saved_models/{args.env_name}/{args.algo}/{t//args.ckpt_freq}.ckpt")

    # save logs
    df = pd.DataFrame(res).set_index("step")
    df.to_csv(f"logs/{args.env_name}/{args.algo}/s{args.seed}.csv")

    # save replay buffer
    replay_buffer.save(f"datasets/{args.env_name}")


if __name__ == "__main__":
    args = get_args()
    # for env_name in ["breakout", "asterix", "freeway", "space_invaders"]:
    for env_name in ["breakout"]:
        args.env_name = env_name
        os.makedirs(f"saved_models/{args.env_name}/{args.algo}", exist_ok=True)
        os.makedirs(f"logs/{args.env_name}/{args.algo}", exist_ok=True)
        os.makedirs(f"datasets", exist_ok=True)
        run(args)
