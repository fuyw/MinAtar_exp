import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"
import argparse
import random
import time
import gym
import numpy as np
from tqdm import trange
from utils import ReplayBuffer, Experience
from atari_wrappers import wrap_deepmind

from flax import linen as nn
from flax.training import train_state, checkpoints
import jax
import jax.numpy as jnp
import optax
from functools import partial


def run_evaluate_episodes(apply_fn, state, env):
    obs = env.reset()
    act_counts = np.zeros(env.action_space.n)
    while not env.get_real_done():
        action = sample(apply_fn, state.params, np.moveaxis(obs[None], 1, -1))
        act_counts[action] += 1
        obs, _, done, _ = env.step(action.item())
        if done:
            obs = env.reset()
    act_counts /= act_counts.sum()
    return np.mean(env.get_eval_rewards()), act_counts, 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7))
    parser.add_argument("--learning-starts", type=int, default=int(5e4))
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


init_fn = nn.initializers.xavier_uniform()
class QNetwork(nn.Module):
    act_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                             name="conv1", kernel_init=init_fn)
        self.conv2 = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                             name="conv2", kernel_init=init_fn)
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                             name="conv3", kernel_init=init_fn)
        self.fc_layer = nn.Dense(features=512, name="fc", kernel_init=init_fn)
        self.out_layer = nn.Dense(features=self.act_dim, name="out", kernel_init=init_fn)

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
    logits = apply_fn({"params": params}, observation)
    action = logits.argmax(1)
    return action


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    env = gym.make(f"BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
    eval_env = gym.make(f"BreakoutNoFrameskip-v4")
    eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NCHW", test=True)
    act_dim = env.action_space.n
    rng = jax.random.PRNGKey(0)
    q_network = QNetwork(act_dim)
    params = q_network.init(rng, jnp.ones(shape=(1, 84, 84, 4)))["params"]
    lr = optax.linear_schedule(init_value=1., end_value=1e-6, transition_steps=args.total_timesteps)
    state = train_state.TrainState.create(apply_fn=QNetwork.apply, params=params, tx=optax.adam(lr))
    target_params = params
    start_time = time.time()
    ckpt_freq = args.total_timesteps // 20

    @jax.jit
    def update_jit(state, target_params, batch):
        target_max = q_network.apply({"params": target_params}, batch.next_observations).max(-1)
        td_target = batch.rewards + 0.99 * target_max * batch.discounts
        def loss_fn(params):
            old_vals = q_network.apply({"params": params}, batch.observations)
            old_val = jax.vmap(lambda q,a: q[a])(old_vals, batch.actions.reshape(-1, 1)).squeeze()
            loss = (td_target - old_val) ** 2
            return loss.mean()
        grad_fn = jax.value_and_grad(loss_fn)
        _, grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state

    # start training
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    obs = env.reset()   # (84, 84)

    for global_step in trange(1, 1+args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.total_timesteps, global_step)
        if random.random() < epsilon:
            action = np.random.choice(act_dim)
        else:
            context = replay_buffer.recent_obs()
            context.append(obs)
            context = np.stack(context, axis=0)[None]               # (1, 4, 84, 84)
            action = sample(q_network.apply, state.params, np.moveaxis(context, 1, -1)).item()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))
        obs = next_obs
        if done:
            obs = env.reset()

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            batch = replay_buffer.sample_batch(args.batch_size)
            state = update_jit(state, target_params, batch)

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_params = state.params

        if global_step % args.eval_freq == 0:
            eval_reward, act_counts, _ = run_evaluate_episodes(q_network.apply, state, eval_env)
            act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
            print(f"Eval at {global_step}: reward = {eval_reward:.1f}\n"
                  f"\tact_counts: ({act_counts})\n")

        if global_step % ckpt_freq == 0:
            checkpoints.save_checkpoint("ckpts", state, global_step//ckpt_freq, prefix="dqn_breakout", keep=20, overwrite=True)
