import json
import logging
import numpy as np
from multiprocessing import Manager
from torch.utils.data import Dataset
from asd_tools import datasets
from asd_tools.utils import read_hdf5


class WaveASDDataset(Dataset):
    """Wave dataset."""

    def __init__(
        self,
        pos_machine_scp="",
        neg_machine_scps=[],
        use_target=False,
        allow_cache=False,
        augmentation_params={},
        statistic_path="",
        in_sample_norm=False,
    ):
        """Initialize dataset. Caution: if in_sample_norm=True, override statistic_path."""
        self.pos_source_files, self.pos_target_files, self.neg_files = [], [], []
        with open(pos_machine_scp, "r") as f:
            pos_files = [s.strip() for s in f.readlines()]
        for pos_file in pos_files:
            if "source" in pos_file:
                self.pos_source_files.append(pos_file)
            elif "target" in pos_file:
                self.pos_target_files.append(pos_file)
        self.pos_source_files.sort()
        self.pos_target_files.sort()
        for neg_machine_scp in neg_machine_scps:
            with open(neg_machine_scp, "r") as f:
                neg_files = [s.strip() for s in f.readlines()]
            self.neg_files += neg_files
        self.neg_files.sort()
        self.wav_files = self.pos_source_files + self.pos_target_files + self.neg_files
        self.use_target = use_target
        self.augmentation_params = augmentation_params
        self.transform = None
        if len(augmentation_params) != 0:
            compose_list = []
            for key in self.augmentation_params.keys():
                aug_class = getattr(datasets, key)
                compose_list.append(aug_class(**self.augmentation_params[key]))
                logging.debug(f"{key}")
            self.transform = datasets.Compose(compose_list)
        # statistic
        self.statistic = None
        self.in_sample_norm = in_sample_norm
        if in_sample_norm:
            logging.info(
                "Data is normalized in each sample. Don't use statistic feature."
            )
        else:
            if len(statistic_path) > 0:
                with open(statistic_path, "r") as f:
                    self.statistic = json.load(f)
                logging.info(
                    f"{statistic_path} mean: {self.statistic['mean']:.4f},"
                    f" std: {self.statistic['std']:.4f}"
                )
        # for cache
        self.caches_size = len(self.pos_source_files) + len(self.pos_target_files)
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE: Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(self.caches_size)]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            items: Dict
                wave: (ndarray) Wave (T, ).
                machine: (str) Name of machine.
                domain: (str) Name of domain.
                section: (int) Number of machine id.
        """
        if self.allow_cache and (self.caches_size > idx):
            # logging.info(f"self.caches[{idx}]:{self.caches[idx]}")
            if len(self.caches[idx]) != 0:
                if self.transform is None:
                    return self.caches[idx]
                else:
                    self.caches[idx]["wave"] = self.transform(
                        self.caches[idx]["origin_wave"]
                    )
                    return self.caches[idx]
        path = self.wav_files[idx]
        items = {"path": path}
        items["wave"] = read_hdf5(path, "wave")
        if self.statistic is not None:
            items["wave"] -= self.statistic["mean"]
            items["wave"] /= self.statistic["std"]
        if self.in_sample_norm:
            items["wave"] -= items["wave"].mean()
            items["wave"] /= items["wave"].std()
        items["machine"] = path.split("/")[-3]
        items["section"] = int(path.split("/")[-1].split("_")[1])
        items["is_normal"] = int(path.split("/")[-1].split("_")[4] == "normal")
        if self.use_target:
            items["domain"] = path.split("/")[-1].split("_")[2]
        if self.transform is not None:
            items["origin_wave"] = items["wave"].copy()
            items["wave"] = self.transform(items["origin_wave"])
        if self.allow_cache and (self.caches_size > idx):
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length."""
        return len(self.wav_files)


class OutlierWaveASDDataset(Dataset):
    """Outlier Wave dataset."""

    def __init__(
        self,
        pos_machine_scp="",
        outlier_scp="",
        neg_machine_scps=[],
        use_target=False,
        allow_cache=False,
        augmentation_params={},
        statistic_path="",
        in_sample_norm=False,
    ):
        """Initialize dataset."""
        self.pos_source_files, self.pos_target_files, self.neg_files = [], [], []
        with open(pos_machine_scp, "r") as f:
            pos_files = [s.strip() for s in f.readlines()]
        for pos_file in pos_files:
            if "source" in pos_file:
                self.pos_source_files.append(pos_file)
            elif "target" in pos_file:
                self.pos_target_files.append(pos_file)
        self.pos_source_files.sort()
        self.pos_target_files.sort()
        for neg_machine_scp in neg_machine_scps:
            with open(neg_machine_scp, "r") as f:
                neg_files = [s.strip() for s in f.readlines()]
            self.neg_files += neg_files
        if len(outlier_scp) == 0:
            self.outlier_files = []
        else:
            with open(outlier_scp, "r") as f:
                self.outlier_files = [s.strip() for s in f.readlines()]
        self.neg_files.sort()
        self.wav_files = (
            self.pos_source_files
            + self.pos_target_files
            + self.neg_files
            + self.outlier_files
        )
        self.outlier_size = (
            len(self.pos_source_files)
            + len(self.pos_target_files)
            + len(self.neg_files)
        )
        self.caches_size = (
            len(self.pos_source_files)
            + len(self.pos_target_files)
            # + len(self.neg_files)
        )
        self.use_target = use_target
        self.augmentation_params = augmentation_params
        self.transform = None
        self.rng = np.random.default_rng()
        if len(augmentation_params) != 0:
            compose_list = []
            for key in self.augmentation_params.keys():
                aug_class = getattr(datasets, key)
                compose_list.append(aug_class(**self.augmentation_params[key]))
                logging.debug(f"{key}")
            self.transform = datasets.Compose(compose_list)
        # statistic
        self.statistic = None
        self.in_sample_norm = in_sample_norm
        if in_sample_norm:
            logging.info("Data is normalized in sample. Not using statistic feature.")
        else:
            if len(statistic_path) > 0:
                with open(statistic_path, "r") as f:
                    self.statistic = json.load(f)
                logging.info(
                    f"{statistic_path} mean: {self.statistic['mean']:.4f},"
                    f" std: {self.statistic['std']:.4f}"
                )
        # for cache
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(ibuki): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(self.caches_size)]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            items: Dict
                wave: (ndarray) Wave (T, ).
                machine: (str) Name of machine.
                domain: (str) Name of domain.
                section: (int) Number of machine id.
        """
        if self.allow_cache and (self.caches_size > idx):
            # logging.info(f"self.caches[{idx}]:{self.caches[idx]}")
            if len(self.caches[idx]) != 0:
                if self.transform is None:
                    return self.caches[idx]
                else:
                    self.caches[idx]["wave"] = self.transform(
                        self.caches[idx]["origin_wave"]
                    )
                    return self.caches[idx]
        path = self.wav_files[idx]
        items = {"path": path}
        if self.outlier_size > idx:
            items["wave"] = read_hdf5(path, "wave")
            if self.statistic is not None:
                items["wave"] -= self.statistic["mean"]
                items["wave"] /= self.statistic["std"]
            if self.in_sample_norm:
                items["wave"] -= items["wave"].mean()
                items["wave"] /= items["wave"].std()
            items["machine"] = path.split("/")[-3]
            items["section"] = int(path.split("/")[-1].split("_")[1])
            items["is_normal"] = int(path.split("/")[-1].split("_")[4] == "normal")
            if self.use_target:
                items["domain"] = path.split("/")[-1].split("_")[2]
            if self.transform is not None:
                items["origin_wave"] = items["wave"].copy()
                items["wave"] = self.transform(items["origin_wave"])
            if self.allow_cache and (self.caches_size > idx):
                self.caches[idx] = items
        else:
            items["wave"] = read_hdf5(path, f"wave{self.rng.integers(5)}")
            if self.statistic is not None:
                items["wave"] -= self.statistic["mean"]
                items["wave"] /= self.statistic["std"]
            if self.in_sample_norm:
                items["wave"] -= items["wave"].mean()
                if items["wave"].std() == 0:
                    items["wave"] += np.random.randn(len(items["wave"]))
                items["wave"] /= items["wave"].std()
            items["machine"] = "outlier"
            items["section"] = 0
            items["is_normal"] = 0
            if self.use_target:
                items["domain"] = "source"
        return items

    def __len__(self):
        """Return dataset length."""
        return len(self.wav_files)
