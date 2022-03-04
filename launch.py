# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import hydra
from omegaconf import OmegaConf
from train import launch


@hydra.main(config_name="main", config_path="configs")
def main(cfg=None):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    launch(cfg)


if __name__ == "__main__":
    main()