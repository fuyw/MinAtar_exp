import numpy as np
import pandas as pd
import torch
from models import DQNAgent
from env_utils import MinAtarEnv
device = torch.device("cuda")


env_name = "breakout"


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
    parser.add_argument("--env_name", type=str, default="asterix")
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--batch_size", type=int, default=128)
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
    parser.add_argument("--no_per", action="store_true", default=True)
    parser.add_argument("--loss", type=str, default="huber")
    args = parser.parse_args()
    return args

args = get_args()
env = MinAtarEnv(env_name)
obs_shape = env.observation_space
in_channels = obs_shape[0]
act_dim = env.action_space.n

res = []
agent = DQNAgent(in_channels, act_dim, args, device)
ckpt_dir = "/home/yuwei/MinAtar_exp/saved_models/online/breakout/dqn"
for i in range(19, 21):
    ckpt_name = f"{ckpt_dir}/{i}.ckpt"
    agent.load(ckpt_name)
    eval_reward = eval_policy(agent, env_name)[0]
    res.append((i, eval_reward))
res_df = pd.DataFrame(res, columns=["i", "eval_reward"])
print(res_df)
