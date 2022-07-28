"""DQN agent for MinAtar environments.

Adapted from https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
"""
from typing import Tuple
from utils import ReplayBuffer
from env_utils import MinAtarEnv
import numpy as np
import pandas as pd
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
device = torch.device("mps")


class QNetwork(nn.Module):
    """Tiny CNN network for MinAtar DQN agent."""
    def __init__(self, in_channels, act_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=16,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=act_num),
        )

    def forward(self, x):
        q_values = self.net(x)
        return q_values


class DQNAgent:
    def __init__(self,
                 in_channels: int = 4,
                 act_dim: int = 6,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.05,
                 update_step: int = 4,
                 device: torch.device = torch.device("cpu")):
        self.act_dim = act_dim
        self.qnet = QNetwork(in_channels, act_dim).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.update_step = 0
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.step = 0
        self.update_step = update_step

    def sample_action(self, obs):
        Qs = self.qnet(torch.tensor(obs[None]).to(self.device))
        action = Qs.argmax().item()
        return action

    def loss_fn(self, batch):
        Qs = self.qnet(batch.observations)  # (256, 4, 10, 10)
        Q = torch.gather(Qs, dim=1, index=batch.actions).squeeze()  # (256,)
        with torch.no_grad():
            next_Q = self.target_qnet(batch.next_observations).max(dim=1)[0]  # (256)
        target_Q = batch.rewards + self.gamma * batch.discounts * next_Q
        td_loss = torch.square(Q - target_Q)
        log_info = {
            "avg_Q": Q.mean().item(),
            "max_Q": Q.max().item(),
            "min_Q": Q.min().item(),
            "avg_target_Q": target_Q.mean().item(),
            "max_target_Q": target_Q.max().item(),
            "min_target_Q": target_Q.min().item(),
            "avg_td_loss": td_loss.mean().item(),
            "max_td_loss": td_loss.max().item(),
            "min_td_loss": td_loss.min().item(),
        }
        return td_loss.mean(), log_info

    def update(self, batch):
        self.step += 1
        self.optimizer.zero_grad()
        loss, log_info = self.loss_fn(batch)
        loss.backward()
        self.optimizer.step()
        if self.step % self.update_step == 0:
            for param, target_param in zip(self.qnet.parameters(),
                    self.target_qnet.parameters()):
                target_param.data.copy_(param.data)
        return log_info


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_timesteps", type=int, default=5000)
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--eval_num", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    args = parser.parse_args()
    return args


def run(args):
    print(f"Start to play {args.env_name}")
    np.random.seed(args.seed)

    # initialize MinAtar environment
    env = MinAtarEnv(args.env_name)
    obs_shape = env.observation_space  # (4, 10, 10)
    in_channels = obs_shape[0]         # 4
    act_dim = env.action_space.n       # 6

    # initialize DQN agent
    agent = DQNAgent(in_channels=in_channels,
                     act_dim=act_dim,
                     lr=args.lr,
                     gamma=args.gamma,
                     tau=args.tau,
                     device=device)

    # initialize the replay buffer
    replay_buffer = ReplayBuffer(obs_shape, act_dim, args.buffer_size)

    # start training
    obs, done = env.reset(), False    # (4, 10, 10)
    ep_reward, ep_step, ep_num = 0, 0, 0
    eval_freq = args.total_timesteps // args.eval_num
    res = []
    for t in trange(1, args.total_timesteps+1):
        # warmup
        if t <= args.warmup_timesteps:
            action = np.random.choice(act_dim)
        else:
            action = agent.sample_action(obs)
            # update the agent
            batch = replay_buffer.sample(batch_size=args.batch_size)
            log_info = agent.update(batch)

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, next_obs, reward, done)
        obs = next_obs
        ep_reward += reward
        ep_step += 1

        if t % eval_freq == 0:
            eval_reward = eval_policy(agent, args.env_name)
            print(f"\nEvaluate agent at step {t}: reward={eval_reward}\n")
            log_info.update({"step": t, "eval_reward": eval_reward})
            res.append(log_info)

        if done:
            obs = env.reset()
            ep_num += 1
            if ep_num % 50 == 0:
                print(f"Episode {ep_num}: step={ep_step}, reward={ep_reward}, buffer_size={replay_buffer.size/1000:.1f}K")
                if t > args.warmup_timesteps:
                    print(f"\tavg_td_loss: {log_info['avg_td_loss']:.3f}, max_td_loss: {log_info['max_td_loss']:.3f}, "
                          f"min_td_loss: {log_info['min_td_loss']:.3f}\n"
                          f"\tavg_Q: {log_info['avg_Q']:.3f}, max_Q: {log_info['max_Q']:.3f}, "
                          f"min_Q: {log_info['min_Q']:.3f}\n")
            ep_reward, ep_step = 0, 0
    df = pd.DataFrame(res).set_index("step")
    df.to_csv(f"logs/{args.env_name}/s{args.seed}.csv")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"logs/{args.env_name}", exist_ok=True)
    run(args)

