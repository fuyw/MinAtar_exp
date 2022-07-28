from typing import List
import torch
import collections
import numpy as np
device = torch.device("mps")


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class ReplayBuffer:
    def __init__(self,
                 obs_shape: List[int],
                 act_dim: int,
                 max_size: int = int(5e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.observations = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, 1), dtype=np.int32)
        self.next_observations = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)

    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray,
            reward: float,
            done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=torch.Tensor(self.observations[idx]).to(device),
                      actions=torch.LongTensor(self.actions[idx]).to(device),
                      rewards=torch.Tensor(self.rewards[idx]).to(device),
                      discounts=torch.Tensor(self.discounts[idx]).to(device),
                      next_observations=torch.Tensor(
                          self.next_observations[idx]).to(device))
        return batch

