import os
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
CONTEXT_LEN = 4

# train an episode
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()  # (84, 84)
    step = 0
    loss_lst = []

    while True:
        step += 1
        context = rpm.recent_obs()  # List of last three (84, 84)
        context.append(obs)
        context = np.stack(context, axis=0)  # (4, 84, 84)

        action = agent.sample(context)
        next_obs, reward, done, _ = env.step(action)
        rpm.add(Experience(obs, action, reward, done))

        # train model
        if (rpm.size() > MEMORY_WARMUP_SIZE) and (step % UPDATE_FREQ == 0):
            # s,a,r,s",done
            (batch_all_obs, batch_action, batch_reward,
             batch_done) = rpm.sample_batch(BATCH_SIZE)
            batch_obs = batch_all_obs[:, :CONTEXT_LEN, :, :]
            batch_next_obs = batch_all_obs[:, 1:, :, :]

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)
            loss_lst.append(train_loss)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward, step, np.mean(loss_lst)


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
                context = np.stack(context, axis=-1)      # (84, 84, 4)
                action = agent.sample_action(
                    agent.state.params, context).item()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))
        obs = next_obs

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
                f"min_Q: {log_info['min_Q']:.3f}\n"
                f"\tavg_target_Q: {log_info['target_Q']:.3f}, max_target_Q: {log_info['max_target_Q']:.3f}, "
                f"min_target_Q: {log_info['min_target_Q']:.3f}\n"
                f"\tavg_batch_rewards: {batch.rewards.mean():.3f}, max_batch_rewards: {batch.rewards.max():.3f}, "
                f"min_batch_rewards: {batch.rewards.min():.3f}\n"
                f"\tact_counts: ({act_counts}), epsilon: {epsilon:.3f}\n"
            )
            log_info.update({"step": t, "eval_reward": eval_reward, "eval_time": eval_time})
            res.append(log_info)

        # save agent
        if t % (args.total_timesteps // 20) == 0:
            agent.save("ckpts", t // (args.total_timesteps // 20))

    # save logs
    df = pd.DataFrame(res).set_index("step")
    df.to_csv(f"logs/{exp_name}.csv")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="PongNoFrameskip-v4")
    parser.add_argument("--warmup_timesteps", type=int, default=int(5e4))
    parser.add_argument("--total_timesteps", type=int, default=int(1e7))
    parser.add_argument("--eval_freq", type=int, default=int(1e5))
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--update_step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("ckpts", exist_ok=True)
    args = get_args()
    run(args)