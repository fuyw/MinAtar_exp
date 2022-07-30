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

# env params
IMAGE_SIZE = (84, 84)


def eval_policy(agent, env, eval_episodes=10):
    t1 = time.time()
    avg_reward = 0.
    act_counts = np.zeros(env.action_space.n)
    for _ in range(eval_episodes):
        obs, done = env.reset(), False  # (4, 84, 84)
        # while not env.get_real_done():
        while not done:
            action = agent.sample_action(
                agent.state.params, np.moveaxis(obs, 0, -1)).item()
            obs, reward, done, _ = env.step(action)
            act_counts[action] += 1
            avg_reward += reward
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    eval_time = (time.time() - t1)/60
    return avg_reward, act_counts, eval_time


def run(args):
    start_time = time.time()
    exp_name = f"dqn_{args.env_name}_s{args.seed}"
    logger = get_logger(f"logs/{exp_name}.log")

    # create envs
    env = gym.make(args.env_name)
    env = wrap_deepmind(env, dim=IMAGE_SIZE[0], framestack=False, obs_format="NCHW")
    eval_env = gym.make(args.env_name)
    eval_env = wrap_deepmind(eval_env, dim=IMAGE_SIZE[0], obs_format="NCHW", test=True)
    replay_buffer = ReplayBuffer(max_size=int(1e6))
    act_dim = env.action_space.n
    agent = DQNAgent(act_dim=act_dim)

    res = []
    obs = env.reset()
    for t in trange(1, args.total_timesteps+1):
        # greedy epsilon exploration
        epsilon = linear_schedule(start_epsilon=0.5, end_epsilon=0.05,
                                  duration=args.total_timesteps, t=t)

        # sample action
        if t <= args.warmup_timesteps:
            action = np.random.choice(act_dim)
        else:
            if np.random.random() < epsilon:
                action = np.random.choice(act_dim)
            else:
                context = replay_buffer.recent_obs()
                context.append(obs)
                context = np.stack(context, axis=-1)  # (84, 84, 4)
                action = agent.sample_action(agent.state.params, context).item()

        # (84, 84), 0.0, False
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))
        obs = next_obs

        # reset env
        if done:
            obs = env.reset()

        # update the agent
        if t > args.warmup_timesteps:
            batch = replay_buffer.sample_batch(args.batch_size)
            log_info = agent.update(batch)

        # evaluate agent
        if t % args.eval_freq == 0:
            eval_reward, act_counts, eval_time = eval_policy(agent, eval_env)
            act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
            logger.info(
                f"Step {t}: reward={eval_reward}, total_time={(time.time()-start_time)/60:.2f}min, "
                f"eval_time: {eval_time:.2f}min\n"
                f"\tavg_loss: {log_info['loss']:.3f}, max_loss: {log_info['max_loss']:.3f}, "
                f"min_loss: {log_info['min_loss']:.3f}\n"
                f"\tavg_Q: {log_info['Q']:.3f}, max_Q: {log_info['max_Q']:.3f}, "
                f"min_Q: {log_info['min_Q']:.3f}, "
                f"avg_batch_discounts: {batch.discounts.mean():.3f}\n"
                f"\tavg_target_Q: {log_info['target_Q']:.3f}, max_target_Q: {log_info['max_target_Q']:.3f}, "
                f"min_target_Q: {log_info['min_target_Q']:.3f}\n"
                f"\tavg_batch_rewards: {batch.rewards.mean():.3f}, max_batch_rewards: {batch.rewards.max():.3f}, "
                f"min_batch_rewards: {batch.rewards.min():.3f}\n"
                f"\tact_counts: ({act_counts}), epsilon: {epsilon:.3f}\n"
            )
            log_info.update({"step": t, "eval_reward": eval_reward, "eval_time": eval_time})
            res.append(log_info)

        # save agent
        if t >= (0.9*args.total_timesteps) and (t % args.ckpt_freq == 0):
            agent.save("ckpts", t // args.ckpt_freq)

    # save logs
    replay_buffer.save(f"datasets/{exp_name}")
    df = pd.DataFrame(res).set_index("step")
    df.to_csv(f"logs/{exp_name}.csv")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="PongNoFrameskip-v4")
    parser.add_argument("--warmup_timesteps", type=int, default=int(1e4))
    parser.add_argument("--total_timesteps", type=int, default=int(2e6))
    parser.add_argument("--eval_num", type=int, default=100)
    parser.add_argument("--ckpt_num", type=int, default=40)
    parser.add_argument("--buffer_size", type=int, default=int(2e6))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--context_len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    args = get_args()
    args.eval_freq = args.total_timesteps // args.eval_num
    args.ckpt_freq = args.total_timesteps // args.ckpt_num
    run(args)
