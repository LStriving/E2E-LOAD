#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import pdb

import sys
sys.path.append("/mnt/cephfs/home/liyirui/project/E2E-LOAD")
sys.path.append("/mnt/cephfs/home/liyirui/project/slowfast")

"""Wrapper to train and test a video classification model."""
from src.config.defaults import assert_and_infer_cfg
from src.utils.misc import launch_job
from src.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

        # Perform multi-clip testing.
        if cfg.TEST.ENABLE:
            if cfg.TEST.NUM_ENSEMBLE_VIEWS == -1:
                num_view_list = [1, 3, 5, 7, 10]  # ? 这个是啥?
                for num_view in num_view_list:
                    cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)
            else:
                launch_job(cfg=cfg, init_method=args.init_method, func=test)

        # Perform model visualization.
        if cfg.TENSORBOARD.ENABLE and (
            cfg.TENSORBOARD.MODEL_VIS.ENABLE
            or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
        ):
            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

        # Run demo.
        if cfg.DEMO.ENABLE:
            demo(cfg)


if __name__ == "__main__":
    main()
