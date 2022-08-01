import argparse
import random
import time
import gym
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from utils import ReplayBuffer, Experience
from atari_wrappers import wrap_deepmind


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7))
    parser.add_argument("--learning-starts", type=int, default=int(8e4))
    parser.add_argument("--eval-freq", type=int, default=int(1e5))

    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-network-frequency", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-e", type=float, default=1)
    parser.add_argument("--end-e", type=float, default=0.01)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    args = parser.parse_args()
    return args


class QNetwork(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def eval_policy(agent, env):
    obs = env.reset()
    act_counts = np.zeros(env.action_space.n)
    while not env.get_real_done():
        logits = agent(torch.Tensor(obs[None]).to(device))
        action = logits.argmax(1).item()
        act_counts[action] += 1
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    act_counts /= act_counts.sum()
    return np.mean(env.get_eval_rewards()), act_counts, 0


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    env = gym.make(f"BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
    eval_env = gym.make(f"BreakoutNoFrameskip-v4")
    eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NCHW", test=True)
    act_dim = env.action_space.n

    q_network = QNetwork(act_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = copy.deepcopy(q_network)
    start_time = time.time()

    # start training
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    obs = env.reset()   # (84, 84)
    for global_step in trange(1, 1+args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e, args.end_e,
            args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            action = np.random.choice(act_dim)
        else:
            context = replay_buffer.recent_obs()
            context.append(obs)
            context = np.stack(context, axis=0)[None]               # (1, 4, 84, 84)
            logits = q_network(torch.Tensor(context).to(device))    # (1, 4)
            action = torch.argmax(logits, dim=1).item()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))
        obs = next_obs
        if done:
            obs = env.reset()

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            batch = replay_buffer.sample_batch(args.batch_size)
            observations = torch.Tensor(batch.observations).to(device)
            actions = torch.LongTensor(batch.actions).to(device)
            next_observations = torch.Tensor(batch.next_observations).to(device)
            rewards = torch.Tensor(batch.rewards).to(device)
            discounts = torch.Tensor(batch.discounts).to(device)
            with torch.no_grad():
                target_max = target_network(next_observations).max(dim=1)[0]
                td_target = rewards + args.gamma * target_max * discounts
            old_val = q_network(observations).gather(1, actions.reshape(-1, 1)).squeeze()
            loss = F.mse_loss(td_target, old_val)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        if global_step % args.eval_freq == 0:
            eval_reward, act_counts, _ = eval_policy(q_network, eval_env)
            act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
            print(f"Eval at {global_step}: reward = {eval_reward:.1f}\n"
                  f"\tavg_Q: {old_val.mean().item():.2f}, "
                  f"avg_target_Q: {td_target.mean().item():.2f}, "
                  f"avg_loss: {loss.item():.3f}\n"
                  f"\tact_counts: ({act_counts})\n")

        if global_step % (args.total_timesteps // 20) == 0:
            torch.save(q_network.state_dict(), f"ckpts/dqn_{global_step % (args.total_timesteps // 20)}.pth")
