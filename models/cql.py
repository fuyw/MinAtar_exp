from typing import Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import functools
import jax
import jax.numpy as jnp
import optax
from utils import Batch


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
        x = x.reshape(len(observation), -1)         # (7744,)
        x = nn.relu(self.fc_layer(x))               # (512,)
        Qs = self.out_layer(x)                      # (act_dim,)
        return Qs


class CQLAgent:
    def __init__(self,
                 obs_shape: Tuple[int] = (1, 84, 84, 4),
                 act_dim: int = 6,
                 tau: float = 0.005,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 cql_alpha: float = 5.0,
                 seed: int = 42):
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha

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
        Qs = self.qnet.apply({"params": params}, observation[None]) 
        action = Qs.argmax()
        return action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   state: train_state.TrainState,
                   target_params: FrozenDict):
        def loss_fn(params):
            Qs = self.qnet.apply({"params": params}, batch.observations)
            Q = jax.vmap(lambda q,a: q[a])(Qs, batch.actions)
            next_Q = self.qnet.apply({"params": target_params}, batch.next_observations).max(-1)
            target_Q = batch.rewards + self.gamma * batch.discounts * next_Q
            cql_loss = (jax.scipy.special.logsumexp(Qs, axis=1) - Q) * self.cql_alpha
            td_loss = (Q - target_Q) ** 2
            log_info = {
                "avg_loss": td_loss.mean(),
                "max_loss": td_loss.max(),
                "min_loss": td_loss.min(),
                "avg_cql_loss": cql_loss.mean(),
                "max_cql_loss": cql_loss.max(),
                "min_cql_loss": cql_loss.min(),
                "avg_Q": Q.mean(),
                "max_Q": Q.max(),
                "min_Q": Q.min(),
                "avg_target_Q": target_Q.mean(),
                "max_target_Q": target_Q.max(),
                "min_target_Q": target_Q.min(),
            }
            total_loss = cql_loss.mean() + td_loss.mean()
            return total_loss, log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, log_info

    def update(self, batch: Batch):
        self.cnt += 1
        self.state, log_info = self.train_step(batch, self.state, self.target_params)
        if self.cnt % 2500 == 0:
            self.target_params = self.state.params
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.state, cnt, prefix="cql_", keep=20,
                                    overwrite=True)