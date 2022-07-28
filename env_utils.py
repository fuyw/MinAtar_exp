"""MinAtar environment made compatible for Dopamine."""
import minatar
import numpy as np

#gin.constant('minatar_env.ASTERIX_SHAPE', (10, 10, 4))
#gin.constant('minatar_env.BREAKOUT_SHAPE', (10, 10, 4))
#gin.constant('minatar_env.FREEWAY_SHAPE', (10, 10, 7))
#gin.constant('minatar_env.SEAQUEST_SHAPE', (10, 10, 10)
#gin.constant('minatar_env.SPACE_INVADERS_SHAPE', (10, 10, 6))


class MinAtarEnv:
    def __init__(self, game_name):
        self.env = minatar.Environment(env_name=game_name)
        self.env.n = self.env.num_actions()
        self.game_over = False

    @property
    def observation_space(self):
        obs_shape = self.env.state_shape()
        obs_shape = (obs_shape[-1], *obs_shape[:-1])
        return obs_shape

    @property
    def action_space(self):
        return self.env  # Only used for the `n` parameter.

    @property
    def reward_range(self):
        pass  # Unused

    @property
    def metadata(self):
        pass  # Unused

    def reset(self):
        self.game_over = False
        self.env.reset()
        obs = np.moveaxis(self.env.state(), 2, 0)
        return obs.astype(np.float32)  # (4, 10, 10)

    def step(self, action):
        r, terminal = self.env.act(action)
        self.game_over = terminal
        next_obs = self.env.state()  # (10, 10, 4)
        next_obs = np.moveaxis(next_obs, 2, 0)  # (4, 10, 10)
        return next_obs.astype(np.float32), r, terminal, None
