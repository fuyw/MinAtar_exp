from typing import List
import logging
import torch
import collections
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])
PerBatch = collections.namedtuple(
    "PerBatch", ["observations", "actions", "rewards", "discounts", "next_observations",
                 "idx", "weights"])


# Buffer
class ReplayBuffer:
    def __init__(self,
                 obs_shape: List[int],
                 max_size: int = int(5e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.observations = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, 1), dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)

    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size-1, size=batch_size)
        batch = Batch(observations=torch.Tensor(self.observations[idx]).to(device),
                      actions=torch.LongTensor(self.actions[idx]).to(device),
                      rewards=torch.Tensor(self.rewards[idx]).to(device),
                      discounts=torch.Tensor(self.discounts[idx]).to(device),
                      next_observations=torch.Tensor(self.observations[idx+1]).to(device))
        return batch

    def save(self, fname: str):
        np.savez(fname,
                 observations=self.observations[:self.size],
                 actions=self.actions[:self.size],
                 rewards=self.rewards[:self.size],
                 discounts=self.discounts[:self.size])



# PER
class SumTree:
    """A sum tree data structure for storing replay priorities.

    A sum tree is a complete binary tree whose leaves contain values called
    priorities. Internal nodes maintain the sum of the priorities of all leaf
    nodes in their subtree.

    For max_size = 4, the tree may look like this:

                       +---+
                       |2.5|
                       +-+-+
                         |
                 +-------+--------+
                 |                |
               +-+-+            +-+-+
               |1.5|            |1.0|
               +-+-+            +-+-+
                 |                |
            +----+----+      +----+----+
            |         |      |         |
          +-+-+     +-+-+  +-+-+     +-+-+
          |0.5|     |1.0|  |0.5|     |0.5|
          +---+     +---+  +---+     +---+

    This is stored in a list of numpy arrays:
    self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5]  ]

    For conciseness, we allocates arrays as powers of two, and pad the excess
    elements with zero values.

    This is similar to the usual array-based representation of a complete
    binary tree, but it is little more user-friendly.

    """

    def __init__(self, max_size):
        """Create the sum tree data structure.

        Args:
            max_size: int, the maximum number of elements that can be stored
                in this data structure. 
        """

        self.levels = [np.zeros(1)]
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        """Sample an element from the sum tree.

        Each element has probability p_i / sum_j p_j of being picked, where
        p_i is the (positive) value associated with the node i.

        Args:
            batch_size: int, number of samples in each batch.
        """
        # with stratified sampling
        # bounds = np.linspace(0., self.levels[0][0], batch_size + 1)
        # assert len(bounds) == batch_size + 1
        # segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
        # value = [np.random.uniform(x[0], x[1]) for x in segments]

        # without stratified sampling
        value = np.random.uniform(0, self.levels[0][0], size=batch_size)
        ind = np.zeros(batch_size, dtype=int)
        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]
            is_greater = np.greater(value, left_sum)

            # If value > left_sum -> go right (+1), else go left (+0)
            ind += is_greater

            # IF we go right, we only need to consider the values in the right
            # tree so we substract the sum of values in the left tree
            value -= left_sum * is_greater
        return ind

    def set(self, ind, new_priority):
        """Set the value of a leaf node and update internal nodes accordingly.

        This operation takes O(log(max_size)).
        Args:
            ind: int, the index of the leaf node to be updated.
            new_priority: float, the value which we assign to the node.
        """
        priority_diff = new_priority - self.levels[-1][ind]

        for nodes in reversed(self.levels):
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(self, ind, new_priority):
        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2


class PrioritizedReplayBuffer:
    def __init__(self,
                 obs_shape: List[int],
                 max_size: int = int(1e6),
                 alpha: float = 0.6,
                 beta: float = 0.4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.observations = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, 1), dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)

        self.tree = SumTree(max_size)
        self.max_priority = 1.0
        self.beta = beta
        self.delta = (1 - beta) / 1e6  # anneal the IS weight with 1e6 steps
        self.alpha = alpha  # add priority ** alpha to the sum_tree

    # add transition with default max_priroity
    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.tree.set(self.ptr, self.max_priority)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = self.tree.sample(batch_size)
        idx %= self.size

        # replace `last idx` with the `second last idx` for next_observation
        last_idx_mask = (idx == self.size)
        idx = idx * (1 - last_idx_mask) + last_idx_mask * (self.size - 1)

        # importance sampling
        weights = self.tree.levels[-1][idx] ** -self.beta
        weights /= weights.max()

        # Hack: 0.4 + 6e-7 * 1e6 = 1. Only used by PER.
        self.beta = min(self.beta + self.delta, 1)
        batch = PerBatch(observations=torch.Tensor(self.observations[idx]).to(device),
                          actions=torch.LongTensor(self.actions[idx]).to(device),
                          rewards=torch.Tensor(self.rewards[idx]).to(device),
                          discounts=torch.Tensor(self.discounts[idx]).to(device),
                          next_observations=torch.Tensor(self.observations[idx+1]).to(device),
                          idx=idx,
                          weights=torch.Tensor(weights).to(device))
        return batch

    def update_priority(self, ind, priority):
        # update max priority
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)

    def save(self, fname: str):
        np.savez(fname,
                 observations=self.observations[:self.size],
                 actions=self.actions[:self.size],
                 rewards=self.rewards[:self.size],
                 discounts=self.discounts[:self.size])

# Exploration linear decay
def linear_schedule(start_epsilon: float, end_epsilon: float, duration: int, t: int):
    slope = (end_epsilon - start_epsilon) / duration
    return max(slope * t + start_epsilon, end_epsilon)


# Logger
def get_logger(fname):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger
