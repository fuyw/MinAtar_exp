from utils import ReplayBuffer
from env_utils import MinAtarEnv
import numpy as np
import torch
from models import DQNAgent, DDQNAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ALGOS = {"dqn": DQNAgent, "ddqn": DDQNAgent}


def eval_policy(agent: DQNAgent,
                env_name: str,
                eval_episodes: int = 10):
    eval_env = MinAtarEnv(env_name)
    avg_reward = 0.
    act_counts = np.zeros(eval_env.action_space.n)
    t = 0
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            action = agent.sample_action(obs)
            obs, reward, done, _ = eval_env.step(action)
            act_counts[action] += 1
            avg_reward += reward
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    return avg_reward, act_counts


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="breakout")
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--update_step", type=int, default=4)
    parser.add_argument("--warmup_timesteps", type=int, default=5000)
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--eval_freq", type=int, default=int(1e4))
    parser.add_argument("--ckpt_freq", type=int, default=int(5e4)) 
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    args = parser.parse_args()
    return args


args = get_args()
env_name = "asterix"
env = MinAtarEnv(env_name)
obs_shape = env.observation_space  # (4, 10, 10)
in_channels = obs_shape[0]         # 4
act_dim = env.action_space.n       # 6

# initialize DQN agent
agent = ALGOS[args.algo](in_channels, act_dim, args, device)

# initialize the replay buffer

# start training
replay_buffer = ReplayBuffer(obs_shape, args.buffer_size)


for i in range(50):
    ep_reward, ep_len = 0, 0
    obs, done = env.reset(), False    # (4, 10, 10)
    while not done:
        action = agent.sample_action(obs) 
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, done)
        obs = next_obs
        ep_reward += reward
        ep_len += 1
    print(f"#Episode {i+1}: ep_reward={ep_reward}, ep_len={ep_len}")


batch = replay_buffer.sample(batch_size=args.batch_size)
log_info = agent.update(batch)

Qs = agent.qnet(batch.observations)
Q = torch.gather(Qs, dim=1, index=batch.actions).squeeze()
next_Qs = agent.target_qnet(batch.next_observations)
