from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import functools
import jax
import jax.numpy as jnp
import optax
from utils import target_update, Batch


class QNetwork(nn.Module):
    act_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                             name="conv1", dtype=jnp.float32) 
        self.conv2 = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                             name="conv2", dtype=jnp.float32)
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                             name="conv3", dtype=jnp.float32)
        self.fc_layer = nn.Dense(features=512, name="fc", dtype=jnp.float32)
        self.out_layer = nn.Dense(features=self.act_dim, name="out", dtype=jnp.float32)

    def __call__(self, observation):
        x = observation.astype(jnp.float32) / 255.  # (84, 84, 4)
        x = nn.relu(self.conv1(x))                  # (21, 21, 32)
        x = nn.relu(self.conv2(x))                  # (11, 11, 64)
        x = nn.relu(self.conv3(x))                  # (11, 11, 64)
        x = x.flatten()                             # (7744,)
        x = nn.relu(self.fc_layer(x))               # (512,)
        Qs = self.out_layer(x)                        # (act_dim,)
        return Qs


class DQNAgent:
    def __init__(self,
                 obs_shape: Tuple[int] = (84, 84, 4),
                 act_dim: int = 6,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 seed: int = 42,
                 target_update_freq: int = 2500):

        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.target_update_freq = target_update_freq

        rng = jax.random.PRNGKey(seed)
        self.qnet = QNetwork(act_dim)
        dummy_obs = jnp.ones(obs_shape)
        params = self.qnet.init(rng, dummy_obs)["params"]
        self.target_params = params
        self.state = train_state.TrainState.create(
            apply_fn=QNetwork.apply, params=params, tx=optax.adam(lr))
        self.cnt = 0

    @functools.partial(jax.jit, static_argnames=("self"))
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray):
        Qs = self.qnet.apply({"params": params}, observation) 
        action = Qs.argmax()
        return action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   state: train_state.TrainState,
                   target_params: FrozenDict):
        def loss_fn(params, observation, action, reward, next_observation, discount):
            Q = self.qnet.apply({"params": params}, observation)[action]
            next_Q = self.qnet.apply({"params": target_params}, next_observation).max()
            target_Q = reward + self.gamma * discount * next_Q
            loss = (Q - target_Q) ** 2
            return loss, {"loss": loss, "Q": Q, "target_Q": target_Q}
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0), has_aux=True),
                           in_axes=(None, 0, 0, 0, 0, 0))
        (_, log_info), grads = grad_fn(state.params,
                                       batch.observations,
                                       batch.actions,
                                       batch.rewards,
                                       batch.next_observations,
                                       batch.discounts)
        extra_info = {
            "max_loss": log_info["loss"].max(), "min_loss": log_info["loss"].min(),
            "max_Q": log_info["Q"].max(), "min_Q": log_info["Q"].min(),
            "max_target_Q": log_info["target_Q"].max(), "min_target_Q": log_info["target_Q"].min(),
        }
        grads = jax.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_info)
        new_state = state.apply_gradients(grads=grads)
        return new_state, log_info

    def update(self, batch: Batch):
        self.state, log_info = self.train_step(batch, self.state, self.target_params)
        self.target_params = target_update(self.state.params, self.target_params, self.tau)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.state, cnt, prefix="dqn", keep=20, overwrite=True)