"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import h5py


def check_dataset(datasets):
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    if 'iu_xray' not in datasets:
        print("iu_xray not found in datasets")
        return
    iu_xray_datasets = datasets['iu_xray']
    splits = ['train', 'eval']
    for split in splits:
        if split not in iu_xray_datasets:
            print(f"Split {split} not found in datasets")
            continue
            
        print(f"\n=== Check {split} dataset ===")
        dataset = iu_xray_datasets[split]
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # get one batch
        batch = next(iter(loader))
        
        # show image
        plt.figure(figsize=(15, 10))
        for i in range(min(2, len(batch['image_0']))):
            # show the first image
            plt.subplot(2, 2, i*2 + 1)
            img_0 = batch['image_0'][i].permute(1, 2, 0).cpu().numpy()
            img_0 = (img_0 * 0.5 + 0.5).clip(0, 1)
            plt.imshow(img_0, cmap='gray')
            plt.title(f"Sample {i+1} - Frontal")
            
            # show the second image
            plt.subplot(2, 2, i*2 + 2)
            img_1 = batch['image_1'][i].permute(1, 2, 0).cpu().numpy()
            img_1 = (img_1 * 0.5 + 0.5).clip(0, 1)
            plt.imshow(img_1, cmap='gray')
            plt.title(f"Sample {i+1} - Lateral")
            
            # print report
            print(f"\nReport {i+1}: {batch['text_input'][i]}")
            print(f"Study ID: {batch['study_id'][i]}")
            
        plt.tight_layout()
        plt.show()
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"\nBatch strcuture:")
        for k, v in batch.items():
            if isinstance(v, (list, tuple)):
                print(f"{k}: {len(v)} items")
            else:
                print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())
    import os
    from pathlib import Path

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    print("Config contents:", vars(cfg))
    
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    # print(model)


    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    
    runner.train()
    
    
if __name__ == "__main__":
    main()

