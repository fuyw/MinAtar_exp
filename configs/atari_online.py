import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Atari game
    config.env_name = "Breakout"
    config.contex_len = 4
    config.image_size = (84, 84)

    # General setting
    config.warmup_timesteps = int(1e4)
    config.total_timesteps = int(2e6)
    config.buffer_size = int(2e6)

    # Training parameters
    config.lr = 3e-4
    config.seed = 42
    config.tau = 0.005
    config.gamma = 0.99
    config.batch_size = 32

    # Logging
    config.ckpt_num = 40
    config.eval_num = 100

    # Dirs
    config.log_dir = "logs"
    config.ckpt_dir = "ckpts"
    config.dataset_dir = "datasets"
    return config