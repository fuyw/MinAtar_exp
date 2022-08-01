import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Atari game
    config.env_name = "Breakout"
    config.contex_len = 4
    config.image_size = (84, 84)

    # General setting
    config.warmup_timesteps = int(2e4)
    config.total_timesteps = int(1e7)
    config.buffer_size = int(1e6)

    # config.warmup_timesteps = int(1e3)
    # config.total_timesteps = int(1e5)
    # config.buffer_size = int(1e6)

    # Training parameters
    config.lr = 3e-4  #6.25e-5
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.batch_size = 32
    config.update_freq = 4

    # Logging
    config.ckpt_num = 2
    config.eval_num = 100

    # Dirs
    config.log_dir = "logs"
    config.ckpt_dir = "ckpts"
    config.dataset_dir = "datasets"
    return config
