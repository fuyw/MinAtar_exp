from absl import app, flags
from ml_collections import config_flags
import os
import train_online

config_flags.DEFINE_config_file("config", default="configs/atari_online.py")
FLAGS = flags.FLAGS

import sys
FLAGS(sys.argv)
def main(argv):
    config = FLAGS.config
    os.makedirs(f"{config.log_dir}/online/{config.env_name}", exist_ok=True)
    os.makedirs(f"{config.ckpt_dir}/online/{config.env_name}", exist_ok=True)
    os.makedirs(f"{config.dataset_dir}/{config.env_name}", exist_ok=True)
    train_online.train_and_evaluate(config)


if __name__ == '__main__':
    app.run(main)
