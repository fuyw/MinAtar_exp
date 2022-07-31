import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import gym
import time
import ml_collections
import numpy as np
import pandas as pd
from tqdm import trange
from atari_wrappers import wrap_deepmind
from utils import ReplayBuffer, get_logger
from models import DQNAgent, BCAgent, CQLAgent, DQNBCAgent


ALGOS = {"bc": BCAgent, "dqn": DQNAgent, "cql": CQLAgent, "dqnbc": DQNBCAgent}


def eval_policy(agent, env, eval_episodes=10):
    t1 = time.time()
    avg_reward = 0.
    act_counts = np.zeros(env.action_space.n)
    for _ in range(eval_episodes):
        obs, done = env.reset(), False  # (4, 84, 84)
        while not done:
            action = agent.sample_action(
                agent.state.params, np.moveaxis(obs, 0, -1)).item()
            obs, reward, done, _ = env.step(action)
            act_counts[action] += 1
            avg_reward += reward
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    return avg_reward, act_counts, time.time() - t1


def train_and_evaluate(config: ml_collections.ConfigDict): 
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f"{config.algo}_s{config.seed}_{timestamp}"
    exp_info = f'# Running experiment for: {exp_name}_{config.env_name} #'
    ckpt_dir = f"{config.ckpt_dir}/offline/{config.env_name}/{exp_name}"
    eval_freq = config.total_timesteps // config.eval_num
    ckpt_freq = config.total_timesteps // config.ckpt_num
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    # initialize logger
    logger = get_logger(f"{config.log_dir}/offline/{config.env_name}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # create envs
    eval_env = gym.make(f"{config.env_name}NoFrameskip-v4")
    eval_env = wrap_deepmind(eval_env, dim=config.image_size[0], obs_format="NCHW", test=True)

    # initialize DQNAgent & Buffer
    act_dim = eval_env.action_space.n
    agent = ALGOS[config.algo](act_dim=act_dim)
    replay_buffer = ReplayBuffer(max_size=config.buffer_size)
    replay_buffer.load(f"datasets/{config.env_name}/dqn.npz")

    # start training
    res = []
    for t in trange(1, config.total_timesteps+1):
        batch = replay_buffer.sample_batch(config.batch_size)
        log_info = agent.update(batch)

        # evaluate agent
        if t % eval_freq == 0:
            eval_reward, act_counts, eval_time = eval_policy(agent, eval_env)
            act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
            logger.info(
                f"Step {t}: reward={eval_reward}, total_time={(time.time()-start_time)/60:.2f}min, "
                f"eval_time: {eval_time:.0f}s\n"
                # f"\tavg_loss: {log_info['avg_loss']:.3f}, max_loss: {log_info['max_loss']:.3f}, "
                # f"min_loss: {log_info['min_loss']:.3f}\n"
                # f"\tavg_Q: {log_info['avg_Q']:.3f}, max_Q: {log_info['max_Q']:.3f}, "
                # f"min_Q: {log_info['min_Q']:.3f}, "
                # f"avg_batch_discounts: {batch.discounts.mean():.3f}\n"
                # f"\tavg_target_Q: {log_info['avg_target_Q']:.3f}, max_target_Q: {log_info['max_target_Q']:.3f}, "
                # f"min_target_Q: {log_info['min_target_Q']:.3f}\n"
                f"\tact_counts: ({act_counts})"
            )
            log_info.update({"step": t, "eval_reward": eval_reward, "eval_time": eval_time})
            res.append(log_info)

        # save agent
        if t >= (0.9*config.total_timesteps) and (t % ckpt_freq == 0):
            agent.save(ckpt_dir, t // ckpt_freq)

    # save logs
    df = pd.DataFrame(res).set_index("step")
    df.to_csv(f"logs/{config.env_name}/offline/{exp_name}.csv")