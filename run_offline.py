from absl import app, flags
from ml_collections import config_flags
import os
import train_offline

config_flags.DEFINE_config_file("config", default="configs/atari_offline.py")
FLAGS = flags.FLAGS


def main(argv):
    config = FLAGS.config
    os.makedirs(f"{config.log_dir}/offline/{config.env_name}", exist_ok=True)
    os.makedirs(f"{config.ckpt_dir}/offline/{config.env_name}", exist_ok=True)
    train_offline.train_and_evaluate(config)


if __name__ == '__main__':
    app.run(main)
