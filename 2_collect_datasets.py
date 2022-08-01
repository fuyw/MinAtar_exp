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

from atari_wrappers import wrap_deepmind
from utils import Experience, ReplayBuffer


###################
# Utils Functions #
###################
def eval_policy(apply_fn, state, env):
    t1 = time.time()
    obs = env.reset()
    eval_step = 0
    act_counts = np.zeros(env.action_space.n)
    while not env.get_real_done():
        obs = np.moveaxis(obs, 0, -1)  # (4, 84, 84) ==> (84, 84, 4)
        action = sample(apply_fn, state.params, obs[None])
        act_counts[action] += 1
        obs, _, done, _ = env.step(action.item())
        eval_step += 1
        if done:
            obs = env.reset()
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return np.mean(env.get_eval_rewards()), act_counts, eval_step, time.time() - t1


def create_state(env):
    rng = jax.random.PRNGKey(0)
    q_network = QNetwork(env.action_space.n)
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
            eval_rewards, _, eval_step, eval_time = eval_policy(state.apply_fn, state, env)
            res.append((step, eval_rewards, eval_step, eval_time))
        res_df = pd.DataFrame(res, columns=["step", "eval_rwards", "eval_step", "eval_time"])
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
states = load_states(eval_env, steps=[1, 2, 3, 4, 8])


#######################
# Collect transitions #
#######################
total_size = int(1.8e6)
rollout_len = total_size // len(states)
replay_buffer = ReplayBuffer(max_size=total_size)
env = gym.make(f"BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")

for state in states:
    collect_trajectory(state, env, replay_buffer, rollout_len)

os.makedirs("datasets", exist_ok=True)
replay_buffer.save("datasets/breakout")
