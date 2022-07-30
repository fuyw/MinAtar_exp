import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import gym
import time
import numpy as np
import pandas as pd
from tqdm import trange
from atari_wrappers import wrap_deepmind
from utils import ReplayBuffer, Experience, get_logger, linear_schedule
from models import DQNAgent
IMAGE_SIZE=(84,84)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PongNoFrameskip-v4")
    parser.add_argument("--train_timesteps", type=int, default=int(1e7))
    parser.add_argument("--test_freq", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args
args = get_args()
env = gym.make(args.env)
env = wrap_deepmind(env, dim=IMAGE_SIZE[0], framestack=False, obs_format="NCHW")

act_dim = env.action_space.n

agent = DQNAgent(act_dim=act_dim)

replay_buffer = ReplayBuffer(max_size=int(1e6))
obs, done = env.reset(), False  # (84, 84)
while not done:
    # stack frames
    context = replay_buffer.recent_obs()
    context.append(obs)
    context = np.stack(context, axis=-1)  # (84, 84, 4)

    # sample action
    action = agent.sample_action(agent.state.params, context).item()

    # (84, 84), 0.0, False
    next_obs, reward, done, _ = env.step(action)
    replay_buffer.add(Experience(obs, action, reward, done))

    # update obs
    obs = next_obs


batch = replay_buffer.sample_batch(256)
batch.observations
# grads, log_info = agent.train_step(batch, agent.state, agent.target_params)
