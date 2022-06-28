# -*- coding: utf-8 -*-

# Copyright 2022 Kuroyanagi Ibuki
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import fnmatch
import logging
import os
import random
import sys

import h5py
import numpy as np
import torch
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in {hdf5_name}. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mixup_apply_rate(max_step=8000, step=0, max_rate=1.0, min_rate=0.0, mode="const"):
    """Mixup apply rate.

    Args:
        max_step (int, optional): Defaults to 8000.
        step (int, optional): Defaults to 0.
        max_rate (float, optional): Defaults to 1.0.
        min_rate (float, optional): Defaults to 0.0.
        mode (str, optional): Defaults to "const".
    """
    if mode == "const":
        return max(min(max_rate, 1.0), 0.0)
    elif mode == "cos":
        tmp = np.cos(np.pi / 2 * step / max_step)
        p = tmp * (max_rate - min_rate) + min_rate
        return p
    elif mode == "sin":
        tmp = np.sin(np.pi * step / max_step)
        p = tmp * (max_rate - min_rate) + min_rate
        return p


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def zscore(pred):
    return (pred - pred.mean()) / pred.std()


def hauc(y_true, y_pred, sections, domains=None, mode="all"):
    score_list = []
    for i in range(3):
        if mode == "all":
            idx = sections == i
        else:
            idx = (sections == i) & (domains == mode)
        score_list.append(roc_auc_score(y_true[idx], y_pred[idx]))
        score_list.append(roc_auc_score(y_true[idx], y_pred[idx], max_fpr=0.1))
    return hmean(score_list), np.array(score_list).std()
