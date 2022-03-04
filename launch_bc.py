# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import hydra
from omegaconf import OmegaConf
from train_bc import train


@hydra.main(config_name="main_bc", config_path="configs")
def main(cfg=None):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == "__main__":
    main()