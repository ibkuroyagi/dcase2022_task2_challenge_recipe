#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Train Anomaly Sound Detection model."""

import argparse
import logging
import os
import sys
import matplotlib
import torch
import yaml

from torch.utils.data import DataLoader

import asd_tools
import asd_tools.models
import asd_tools.losses
import asd_tools.optimizers
import asd_tools.schedulers
from asd_tools.datasets import WaveASDDataset
from asd_tools.datasets import WaveCollator
from asd_tools.datasets import DomainClassifierBatchSampler
from asd_tools.trainer import DomainClassifierTrainer
from asd_tools.utils import count_params
from asd_tools.utils import seed_everything

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train outlier exposure model (See detail in asd_tools/bin/train.py)."
    )
    parser.add_argument(
        "--pos_machine", type=str, required=True, help="Name of positive machine."
    )
    parser.add_argument(
        "--train_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--train_neg_machine_scps",
        default=[],
        type=str,
        nargs="+",
        help="list of root directories of positive train datasets.",
    )
    parser.add_argument(
        "--valid_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--valid_neg_machine_scps",
        default=[],
        type=str,
        nargs="+",
        help="list of root directories of positive train datasets.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    seed_everything(seed=config["seed"])
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    # get dataset
    train_dataset = WaveASDDataset(
        pos_machine_scp=args.train_pos_machine_scp,
        neg_machine_scps=args.train_neg_machine_scps,
        allow_cache=config.get("allow_cache", False),
        use_target=config["use_target"],
        augmentation_params=config.get("augmentation_params", {}),
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"pos_source = {len(train_dataset.pos_source_files)}.")
    logging.info(f"pos_target = {len(train_dataset.pos_target_files)}.")
    logging.info(f"neg = {len(train_dataset.neg_files)}.")
    valid_dataset = WaveASDDataset(
        pos_machine_scp=args.valid_pos_machine_scp,
        neg_machine_scps=args.valid_neg_machine_scps,
        allow_cache=True,
        use_target=config["use_target"],
        augmentation_params=config.get("augmentation_params", {}),
    )
    logging.info(f"The number of validation files = {len(valid_dataset)}.")
    train_balanced_batch_sampler = DomainClassifierBatchSampler(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    valid_balanced_batch_sampler = DomainClassifierBatchSampler(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    train_collator = WaveCollator(
        sf=config["sf"],
        sec=config["sec"],
        pos_machine=args.pos_machine,
        shuffle=True,
        use_target=config["use_target"],
        use_is_normal=False,
    )
    valid_collator = WaveCollator(
        sf=config["sf"],
        sec=config["sec"],
        pos_machine=args.pos_machine,
        shuffle=False,
        use_target=config["use_target"],
        use_is_normal=False,
    )
    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_sampler=train_balanced_batch_sampler,
            collate_fn=train_collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_sampler=valid_balanced_batch_sampler,
            collate_fn=valid_collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models and optimizers
    model_class = getattr(asd_tools.models, config["model_type"])
    model = model_class(**config["model_params"]).to(device)
    logging.info(model)
    params_cnt = count_params(model)
    logging.info(f"Size of model is {params_cnt}.")
    criterion = {
        "machine_loss": getattr(asd_tools.losses, config["machine_loss_type"],)(
            **config["machine_loss_params"]
        ).to(device),
        "section_loss": getattr(asd_tools.losses, config["section_loss_type"],)(
            **config["section_loss_params"]
        ).to(device),
    }
    optimizer_class = getattr(
        asd_tools.optimizers,
        config["optimizer_type"],
    )
    params_list = [{"params": model.parameters()}]
    metric_fc = None
    if config.get("metric_fc_type", None) is not None:
        metric_fc_class = getattr(
            asd_tools.losses,
            config["metric_fc_type"],
        )
        metric_fc = metric_fc_class(**config["metric_fc_params"]).to(device)
        params_list.append({"params": metric_fc.parameters()})
    optimizer = optimizer_class(params_list, **config["optimizer_params"])
    scheduler = None
    if config.get("scheduler_type", None) is not None:
        scheduler_class = getattr(
            asd_tools.schedulers,
            config["scheduler_type"],
        )
        if config["scheduler_type"] == "OneCycleLR":
            config["scheduler_params"]["epochs"] = config["train_max_epochs"]
            config["scheduler_params"]["steps_per_epoch"] = (
                len(train_dataset.pos_source_files) // (config["batch_size"] // 2) + 1
            )
        scheduler = scheduler_class(optimizer=optimizer, **config["scheduler_params"])

    # define trainer
    trainer = DomainClassifierTrainer(
        steps=1,
        epochs=1,
        data_loader=data_loader,
        model=model.to(device),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        train=True,
        metric_fc=metric_fc,
    )

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
