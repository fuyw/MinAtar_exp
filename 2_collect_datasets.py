import os
import time
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import numpy as np
import pandas as pd
from functools import partial

import gym
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import checkpoints, train_state
from tqdm import tqdm

from atari_utils import create_env, create_vec_env
from atari_wrappers import wrap_deepmind
from utils import Experience, ReplayBuffer


###################
# Utils Functions #
###################
def eval_policy(apply_fn, state, env, channel_last=False):
    t1 = time.time()
    obs = env.reset()
    act_counts = np.zeros(env.action_space.n)
    while not env.get_real_done():
        if not channel_last:
            obs = np.moveaxis(obs, 0, -1)
        action = sample(apply_fn, state.params, obs[None])
        act_counts[action] += 1
        obs, _, done, _ = env.step(action.item())
        if done:
            obs = env.reset()
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return np.mean(env.get_eval_rewards()), act_counts, time.time() - t1


def new_eval_policy(apply_fn, state, env, eval_episodes=10):
    t1 = time.time()
    avg_reward = 0.
    act_counts = np.zeros(env.preproc.action_space.n)
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = sample(apply_fn, state.params, obs[None]).item()
            act_counts[action] += 1
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return avg_reward, act_counts, time.time() - t1


def eval_vecv(apply_fn, state, eval_envs, eval_episodes: int = 10):
    """Evaluate with envpool vectorized environments."""
    t1 = time.time()
    act_counts = np.zeros(eval_envs.action_space[0].n)
    n_envs = eval_envs.num_envs

    # record episode reward and length
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    episode_rewards, episode_lengths = [], []
    episode_counts = np.zeros(n_envs, dtype="int")

    # evaluate `target` episodes for each environment
    episode_count_targets = np.array([(eval_episodes + i) // n_envs
                                      for i in range(n_envs)],
                                     dtype="int")

    # start evaluation
    observations = eval_envs.reset()  # (10, 84, 84, 4)
    while (episode_counts < episode_count_targets).any():  # 100_000
        actions = sample(apply_fn, state.params, observations)  # (10,)
        observations, rewards, dones, _ = eval_envs.step(np.asarray(actions))
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            act_counts[actions[i]] += 1
            if episode_counts[i] < episode_count_targets[i]:
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
    avg_reward = np.mean(episode_rewards)
    eval_step = np.sum(episode_lengths)
    return avg_reward, eval_step, time.time() - t1


def create_state(env):
    rng = jax.random.PRNGKey(0)
    q_network = QNetwork(env.preproc.action_space.n)
    params = q_network.init(rng, jnp.ones(shape=(1, 84, 84, 4)))["params"]
    state = train_state.TrainState.create(apply_fn=q_network.apply, params=params,
                                          tx=optax.adam(1e-3))
    return state


def load_states(env, steps=range(1, 5), verbose=False):
    state = create_state(env)
    states = [state]
    for step in steps:
        state = checkpoints.restore_checkpoint("online_ckpts", state, step, prefix="dqn_breakout")
        states.append(state)

    if verbose:
        res = []
        for state in tqdm(states[1:], desc="[Eval ckpts]"):
            eval_rewards, _, eval_time = eval_policy(state.apply_fn, state, env)
            res.append((step, eval_rewards, eval_time))
        res_df = pd.DataFrame(res, columns=["step", "eval_rwards", "eval_time"])
        print(res_df)
    return states


def collect_trajectory(state, env, replay_buffer, rollout_len):
    t1 = time.time()
    total_len = 0
    with tqdm(total=rollout_len) as pbar:
        while total_len < rollout_len:
            obs, done = env.reset(), False
            ep_len = 0
            while not done:
                if np.random.random() < 0.2:
                    action = np.random.choice(env.action_space.n)
                else:
                    context = replay_buffer.recent_obs()
                    context.append(obs)
                    context = np.stack(context, axis=-1)[None]
                    action = sample(state.apply_fn, state.params, context).item()
                next_obs, reward, done, _ = env.step(action)
                replay_buffer.add(Experience(obs, action, reward, done))
                obs = next_obs
                ep_len += 1
            pbar.update(ep_len)
            total_len += ep_len
    print(f"Collected {total_len} trajectories using {(time.time()-t1)/60:.1f}min, "
          f"curr_size: {replay_buffer._curr_size}, "
          f"curr_pos: {replay_buffer._curr_pos}")


#############
# DQN Agent #
#############
class QNetwork(nn.Module):
    act_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1")
        self.conv2 = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2")
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3")
        self.fc_layer = nn.Dense(features=512, name="fc")
        self.out_layer = nn.Dense(features=self.act_dim, name="out")

    def __call__(self, observation):
        x = observation.astype(jnp.float32) / 255.  # (84, 84, 4)
        x = nn.relu(self.conv1(x))                  # (21, 21, 32)
        x = nn.relu(self.conv2(x))                  # (11, 11, 64)
        x = nn.relu(self.conv3(x))                  # (11, 11, 64)
        x = x.reshape(len(observation), -1)         # (7744,)
        x = nn.relu(self.fc_layer(x))               # (512,)
        Qs = self.out_layer(x)                      # (act_dim,)
        return Qs


@partial(jax.jit, static_argnums=0)
def sample(apply_fn, params, observation):
    """sample action s ~ pi(a|s)"""
    logits = apply_fn({"params": params}, observation)
    action = logits.argmax(1)
    return action


########################
# Create env and state #
########################
eval_env = gym.make(f"BreakoutNoFrameskip-v4")
eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NCHW", test=True, test_episodes=3)

new_env = create_env("BreakoutNoFrameskip-v4", stack_num=4, channel_last=True)
states = load_states(new_env, steps=[1, 2, 3, 4, 8], verbose=False)
new_eval_policy(states[-1].apply_fn, states[0], new_env, eval_episodes=10)


eval_envs = create_vec_env("BreakoutNoFrameskip-v4", num_envs=10)


#######################
# Collect transitions #
#######################
total_size = int(2e6)
replay_buffer = ReplayBuffer(max_size=total_size)
rollout_len = total_size // len(states)
env = gym.make(f"BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")

for i in range(1, 5):
    collect_trajectory(states[i], env, replay_buffer, rollout_len)


os.makedirs("datasets", exist_ok=True)
replay_buffer.save("datasets/breakout")
