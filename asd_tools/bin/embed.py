#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Extract embedding vectors."""

import argparse
import logging
import os
import random
import sys
import numpy as np
import matplotlib
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader

import asd_tools
import asd_tools.losses
import asd_tools.models
from asd_tools.datasets import WaveASDDataset
from asd_tools.datasets import WaveEvalCollator
from asd_tools.utils import read_hdf5
from asd_tools.utils import seed_everything

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def random_sampled_target(target_batch, source_batch, sf=16000, sec=2.0):
    """Random sampling target data to mixup during inference.

    Args:
        target_batch (tensor of list of dict): {0:[(sr*sec,),..,(sr*sec,)],..}]
        source_batch (dict): batch
        sf (int, optional): sampling rate. Defaults to 16000.
        sec (float, optional): inference segments second. Defaults to 2.0.
    """
    mixup_batch = []
    for section in source_batch["section"]:
        full_seg = random.choice(target_batch[section])
        l_seg = int(sf * sec)
        start_idx = random.randint(0, len(full_seg) - l_seg)
        mixup_batch.append(full_seg[start_idx : start_idx + l_seg])
    return torch.stack(mixup_batch)


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train outlier exposure model (See detail in asd_tools/bin/train.py)."
    )
    parser.add_argument(
        "--valid_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--eval_pos_machine_scp",
        type=str,
        required=True,
        help="directory to evaluation machine.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--use_10sec",
        type=str,
        default="false",
        help="If true, use 10 second wav input.",
    )
    parser.add_argument(
        "--tail_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--statistic_path",
        type=str,
        default="",
        help="Statistic info of positive data in json.",
    )
    parser.add_argument(
        "--checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="checkpoint file path to resume training. (default=[])",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
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

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    if args.use_10sec == "true":
        config["sec"] = 10
        config["n_split"] = 1
    seed_everything(seed=config["seed"])
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    use_domain_classification = config["model_params"].get("use_domain_head", False)
    # get dataset
    valid_dataset = WaveASDDataset(
        pos_machine_scp=args.valid_pos_machine_scp,
        neg_machine_scps=[],
        allow_cache=True,
        use_target=config["use_target"],
        statistic_path=args.statistic_path,
        in_sample_norm=config.get("in_sample_norm", False),
    )
    logging.info(f"The number of validation files = {len(valid_dataset)}.")
    logging.info(f"pos_source = {len(valid_dataset.pos_source_files)}.")
    logging.info(f"pos_target = {len(valid_dataset.pos_target_files)}.")
    if "target" in args.tail_name:
        target_batch = {i: [] for i in range(6)}  # {section: [n, sr*sec]}
        for ta_file in valid_dataset.pos_target_files:
            section = int(ta_file.split("/")[-1].split("_")[1])
            target_batch[section].append(
                torch.tensor(read_hdf5(ta_file, "wave"), dtype=torch.float)
            )

    collator = WaveEvalCollator(
        sf=config["sf"],
        sec=config["sec"],
        n_split=config["n_split"],
        use_domain=False,
        is_label=True,
        is_dcase2022=True,
    )
    loader_dict = {
        "valid": DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            collate_fn=collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
    }
    if os.path.isfile(args.eval_pos_machine_scp):
        eval_dataset = WaveASDDataset(
            pos_machine_scp=args.eval_pos_machine_scp,
            neg_machine_scps=[],
            allow_cache=True,
            use_target=config["use_target"],
            statistic_path=args.statistic_path,
            in_sample_norm=config.get("in_sample_norm", False),
        )
        logging.info(f"The number of evaluation files = {len(eval_dataset)}.")
        loader_dict["eval"] = DataLoader(
            eval_dataset,
            batch_size=config["batch_size"],
            collate_fn=collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

    metric_fc = None
    if config.get("metric_fc_type", None) is not None:
        metric_fc_class = getattr(
            asd_tools.losses,
            config["metric_fc_type"],
        )
        metric_fc = metric_fc_class(**config["metric_fc_params"])
    for checkpoint in args.checkpoints:
        model_class = getattr(asd_tools.models, config["model_type"])
        model = model_class(**config["model_params"])
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        logging.info(f"Successfully loaded {checkpoint}.")
        logging.info(
            f"Steps:{state_dict['steps']}, "
            f"Epochs:{state_dict['epochs']}, "
            f"BEST loss:{state_dict['best_loss']}"
        )
        if "metric_fc" in state_dict.keys():
            metric_fc.weight = torch.nn.Parameter(state_dict["metric_fc"])
            metric_fc = metric_fc.to(device)
            metric_fc.eval()
            logging.info(
                f"Successfully load weights of metric_fc:{metric_fc.weight.shape}."
            )
        for mode, loader in loader_dict.items():
            pred_machine = np.empty((0, 1))
            pred_section = np.empty((0, 6))
            embed = np.empty((0, config["model_params"]["embedding_size"]))
            path_list = np.empty((0, 1))
            split_list = np.empty((0, 1))
            section_list = np.empty((0, 1))
            is_normal_list = np.empty((0, 1))
            if use_domain_classification:
                is_target_pred = np.empty((0, 1))
            for batch in loader:
                for i in range(config["n_split"]):
                    with torch.no_grad():
                        if ("target" in args.tail_name) and (mode == "dev"):
                            lam = torch.tensor(
                                np.random.beta(0.2, 0.2, len(batch)),
                                dtype=torch.float32,
                            )
                            batch[f"X{i}"] = lam * batch[f"X{i}"] + (
                                1 - lam
                            ) * random_sampled_target(
                                target_batch, batch, sf=config["sf"], sec=config["sec"]
                            )
                            logging.info("Mixuped by target domain data.")
                        y_ = model(batch[f"X{i}"].to(device))
                        if metric_fc is not None:
                            y_["section"] = metric_fc(
                                y_["embedding"],
                                torch.tensor(batch["section"]).to(device),
                            )
                    pred_machine = np.concatenate(
                        [pred_machine, y_["machine"].cpu().numpy()], axis=0
                    )
                    pred_section = np.concatenate(
                        [pred_section, y_["section"].cpu().numpy()], axis=0
                    )
                    embed = np.concatenate(
                        [embed, y_["embedding"].cpu().numpy()], axis=0
                    )
                    if use_domain_classification:
                        is_target_pred = np.concatenate(
                            [is_target_pred, y_["domain"].cpu().numpy()], axis=0
                        )
                    path_list = np.concatenate([path_list, batch["path"][:, None]])
                    split_list = np.concatenate(
                        [split_list, np.ones((len(y_["embedding"]), 1)) * i]
                    )
                    section_list = np.concatenate(
                        [section_list, batch["section"][:, None]]
                    )
                    is_normal_list = np.concatenate(
                        [is_normal_list, batch["is_normal"][:, None]]
                    )

            embed_cols = [
                f"e{i}" for i in range(config["model_params"]["embedding_size"])
            ]
            pred_section_cols = [f"pred_section{i}" for i in range(6)]
            columns = (
                [
                    "path",
                    "split",
                    "section",
                    "is_normal",
                    "is_target",
                    "pred_machine",
                    "embed_norm",
                ]
                + pred_section_cols
                + embed_cols
            )

            logging.info(
                f"path_list:{path_list.shape}, split_list:{split_list.shape}, "
                f"section_list:{section_list.shape}, "
                f"is_normal_list:{is_normal_list.shape}, pred_machine:{pred_machine.shape}, "
                f"pred_section:{pred_section.shape}, embed:{embed.shape}"
            )
            df = pd.DataFrame(embed, columns=embed_cols)
            df["path"] = path_list
            df["split"] = split_list.astype(int)
            df["section"] = section_list.astype(int)
            df["is_normal"] = is_normal_list.astype(int)
            df["is_target"] = df["path"].map(lambda x: int("target" in x))
            df["pred_machine"] = pred_machine
            df["embed_norm"] = (
                np.sqrt(np.power(embed, 2).sum(1))
                / config["model_params"]["embedding_size"]
            )
            df[pred_section_cols] = pred_section
            if use_domain_classification:
                columns.append("is_target_pred")
                df["is_target_pred"] = is_target_pred
                logging.info(f"is_target_pred:{is_target_pred.shape}")
            df = df[columns]
            csv_path = checkpoint.replace(".pkl", f"{args.tail_name}_{mode}.csv")
            if args.use_10sec == "true":
                csv_path = checkpoint.replace(
                    ".pkl", f"{args.tail_name}_10sec_{mode}.csv"
                )
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved at {csv_path}")


if __name__ == "__main__":
    main()
