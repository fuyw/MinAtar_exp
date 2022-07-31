from typing import Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import functools
import jax
import jax.numpy as jnp
import optax
from utils import Batch


class BCNetwork(nn.Module):
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
        logits = self.out_layer(x)                  # (act_dim,)
        log_probs = jax.nn.log_softmax(logits)      # (act_dim,)
        return log_probs


class BCAgent:
    def __init__(self,
                 obs_shape: Tuple[int] = (1, 84, 84, 4),
                 act_dim: int = 6,
                 lr: float = 3e-4,
                 seed: int = 42):

        self.obs_shape = obs_shape
        self.act_dim = act_dim

        rng = jax.random.PRNGKey(seed)
        self.net = BCNetwork(act_dim)
        dummy_obs = jnp.ones(obs_shape)
        params = self.net.init(rng, dummy_obs)["params"]
        self.target_params = params
        self.state = train_state.TrainState.create(
            apply_fn=BCNetwork.apply, params=params, tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray):
        log_probs = self.net.apply({"params": params}, observation[None]) 
        action = log_probs.argmax()
        return action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   state: train_state.TrainState):
        def loss_fn(params):
            log_probs = self.net.apply({"params": params}, batch.observations)
            log_prob = jax.vmap(lambda lp, a: lp[a])(log_probs, batch.actions)
            loss = -log_prob.mean()
            log_info = {
                "avg_logp": log_prob.mean(),
                "max_logp": log_prob.max(),
                "min_logp": log_prob.min(),
            }
            return loss, log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, log_info

    def update(self, batch: Batch):
        self.state, log_info = self.train_step(batch, self.state)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.state, cnt, prefix="bc_", keep=20,
                                    overwrite=True)
