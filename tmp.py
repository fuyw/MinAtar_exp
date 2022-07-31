import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import gym
import time
import numpy as np
from atari_wrappers import wrap_deepmind
from utils import ReplayBuffer, Experience
from models import DQNAgent

IMAGE_SIZE = (84, 84)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PongNoFrameskip-v4")
    parser.add_argument("--train_timesteps", type=int, default=int(1e7))
    parser.add_argument("--test_freq", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def check_network():
    from flax import linen as nn

    class QNetwork(nn.Module):
        act_dim: int

        # initializer =

        def setup(self):
            self.conv1 = nn.Conv(features=32,
                                 kernel_size=(8, 8),
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 strides=(4, 4),
                                 name="conv1")  #, )
            self.conv2 = nn.Conv(
                features=64, kernel_size=(4, 4), strides=(2, 2),
                name="conv2")  #, kernel_init=self.initializer)
            self.conv3 = nn.Conv(
                features=64, kernel_size=(3, 3), strides=(1, 1),
                name="conv3")  #, kernel_init=self.initializer)
            self.fc_layer = nn.Dense(
                features=512, name="fc")  #, kernel_init=self.initializer)
            self.out_layer = nn.Dense(
                features=self.act_dim,
                name="out")  #, kernel_init=self.initializer)

        def __call__(self, observation):
            x = observation.astype(jnp.float32) / 255.  # (84, 84, 4)
            x = nn.relu(self.conv1(x))  # (21, 21, 32)
            x = nn.relu(self.conv2(x))  # (11, 11, 64)
            x = nn.relu(self.conv3(x))  # (11, 11, 64)
            x = x.reshape(len(observation), -1)  # (7744,)
            x = nn.relu(self.fc_layer(x))  # (512,)
            Qs = self.out_layer(x)  # (act_dim,)
            return Qs

    net = QNetwork(6)
    import jax
    import jax.numpy as jnp
    rng = jax.random.PRNGKey(0)
    params = net.init(rng, jnp.ones((1, 84, 84, 4)))


def check_rollout():
    args = get_args()
    env = gym.make(args.env)
    env = wrap_deepmind(env,
                        dim=IMAGE_SIZE[0],
                        framestack=False,
                        obs_format="NCHW")
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
        action = agent.sample_action(agent.state.params, context).item()  # 3

        # (84, 84), 0.0, False
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))

        # update obs
        obs = next_obs

    batch = replay_buffer.sample_batch(256)
    res = agent.train_step(batch, agent.state, agent.target_params)


def check_buffer_ckpt():
    import numpy as np
    # dataset = np.load("datasets/Breakout/dqn_s42_20220731_030132.npz")
    dataset = np.load("datasets/Pong/dqn_s42_20220731_025637.npz")
    print(f"observations.shape = {dataset['observations'].shape}")
    print(f"actions.shape = {dataset['actions'].shape}")
    print(f"rewards.shape = {dataset['rewards'].shape}")
    print(f"dones.shape = {dataset['dones'].shape}")
    print(f"curr_size = {dataset['curr_size']}")  # 2e6
    print(f"curr_pos = {dataset['curr_pos']}")  # 0


def eval_policy(agent, env, eval_episodes=10):
    t1 = time.time()
    avg_reward = 0.
    act_counts = np.zeros(env.action_space.n)
    for _ in range(eval_episodes):
        obs, done = env.reset(), False  # (4, 84, 84)
        while not done:
            action = agent.sample_action(agent.state.params,
                                         np.moveaxis(obs, 0, -1)).item()
            obs, reward, done, _ = env.step(action)
            act_counts[action] += 1
            avg_reward += reward
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    return avg_reward, act_counts, time.time() - t1


def check_ckpts():
    env_name = "Pong"
    ckpt_dir = f"ckpts/online/{env_name}/dqn_s0_20220731_093646"
    env = gym.make(f"{env_name}NoFrameskip-v4")
    env = wrap_deepmind(env, dim=84, obs_format="NCHW", test=True)
    act_dim = env.action_space.n
    agent = DQNAgent(act_dim=act_dim)
    for i in range(32, 33):
        agent.load(ckpt_dir, i)
        eval_reward, act_counts, eval_time = eval_policy(agent, env)
        print(f"ckpt {i}: {eval_reward:.2f}")


def check_atari_utils():
    from atari_utils import create_env
    env = create_env("PongNoFrameskip-v4", stack_num=4, channel_last=False)
    print(env.reset().shape)
    obs, reward, done, _ = env.step(1)
    print(obs.shape)
    print(reward)
    print(done)

