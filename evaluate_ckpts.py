from env_utils import MinAtarEnv
import numpy as np
import pandas as pd
import torch
from models.dqn import DQNAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALGOS = {"dqn": DQNAgent}


def eval_policy(agent: DQNAgent,
                env_name: str,
                eval_episodes: int = 20):
    eval_env = MinAtarEnv(env_name)
    avg_reward = 0.
    act_counts = np.zeros(eval_env.action_space.n)
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            act_counts[action] += 1
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    return avg_reward, act_counts


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="breakout")
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--buffer_size", type=int, default=int(2e6))
    parser.add_argument("--warmup_timesteps", type=int, default=5000)
    parser.add_argument("--total_timesteps", type=int, default=int(2e6))
    parser.add_argument("--eval_freq", type=int, default=int(4e4))
    parser.add_argument("--ckpt_freq", type=int, default=int(2e4))
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


res = []
env_name = "freeway"
args = get_args()
env = MinAtarEnv(env_name)
obs_shape = env.observation_space
in_channels = obs_shape[0]
act_dim = env.action_space.n
agent = DQNAgent(in_channels, act_dim, args.lr, device=device)
for i in range(80, 101):
    ckpt_name = f"saved_models/{env_name}/dqn_{i}.ckpt"
    agent.load(ckpt_name)
    eval_reward = eval_policy(agent, env_name)[0]
    res.append((i, eval_reward))
res_df = pd.DataFrame(res, columns=["i", "eval_reward"])
res_df.to_csv(f"{env_name}.csv")