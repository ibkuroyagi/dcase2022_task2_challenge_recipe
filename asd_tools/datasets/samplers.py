import random
import numpy as np
from torch.utils.data.sampler import BatchSampler


class OECBalancedBatchSampler(BatchSampler):
    """BatchSampler - positive:negative = 1 : 1.

    Returns batches of size n_classes * n_samples
    """

    def __init__(
        self, dataset, batch_size=64, shuffle=False, drop_last=False, n_target=1
    ):
        """Batch Sampler.

        Args:
            dataset (dataset): dataset for ASD
            batch_size (int, optional): batch size. Defaults to 64.
            shuffle (bool, optional): shuffle. Defaults to False.
            drop_last (bool, optional): drop last. Defaults to False.
            n_target (int, optional): The number of target sample. Defaults to 1.
        """
        self.n_pos_source = len(dataset.pos_source_files)
        self.pos_source_idx = np.arange(self.n_pos_source)
        self.n_pos_target = len(dataset.pos_target_files)
        self.pos_target_idx = np.arange(
            self.n_pos_source, self.n_pos_source + self.n_pos_target
        )
        self.n_neg = len(dataset.neg_files)
        self.neg_idx = np.arange(
            self.n_pos_source + self.n_pos_target,
            self.n_pos_source + self.n_pos_target + self.n_neg,
        )
        self.used_idx_cnt = {"pos_source": 0, "pos_target": 0, "neg": 0}
        self.count = 0
        self.batch_size = batch_size
        self.n_samples = batch_size // 2
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_target = min(n_target, self.n_pos_target)

    def __iter__(self):
        self.count = 0
        if self.shuffle:
            np.random.shuffle(self.pos_source_idx)
            np.random.shuffle(self.pos_target_idx)
            np.random.shuffle(self.neg_idx)
        while self.count + self.n_samples < self.n_pos_source:
            indices = []
            indices.extend(
                self.pos_source_idx[
                    self.used_idx_cnt["pos_source"] : self.used_idx_cnt["pos_source"]
                    + self.n_samples
                    - self.n_target
                ]
            )
            self.used_idx_cnt["pos_source"] += self.n_samples - self.n_target
            if self.n_target > 0:
                indices.extend(np.random.choice(self.pos_target_idx, self.n_target))
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg"] : self.used_idx_cnt["neg"] + self.n_samples
                ]
            )
            self.used_idx_cnt["neg"] += self.n_samples
            if self.shuffle:
                random.shuffle(indices)
            yield indices
            self.count += self.n_samples - self.n_target
        if not self.drop_last:
            indices = []
            indices.extend(self.pos_source_idx[self.used_idx_cnt["pos_source"] :])
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg"] : self.used_idx_cnt["neg"] + self.n_samples
                ]
            )
            yield indices
        if self.used_idx_cnt["pos_source"] + self.n_samples > self.n_pos_source:
            self.used_idx_cnt["pos_source"] = 0
            self.used_idx_cnt["neg"] = 0

    def __len__(self):
        return self.n_pos_source // self.n_samples


class OutlierBalancedBatchSampler(BatchSampler):
    """BatchSampler = positive:negative+outlier."""

    def __init__(
        self,
        dataset,
        n_pos=32,
        n_neg=32,
        shuffle=False,
        drop_last=False,
        n_target=1,
    ):
        """Batch Sampler.

        Args:
            dataset (dataset): dataset for ASD
            n_pos (int, optional): The number of positive sample in the mini-batch. Defaults to 64.
            n_neg (int, optional): The number of negative sample in the mini-batch. Defaults to 64.
            shuffle (bool, optional): shuffle. Defaults to False.
            drop_last (bool, optional): drop last. Defaults to False.
            n_target (int, optional): The number of target sample. Defaults to 1.
        """
        self.n_pos_source = len(dataset.pos_source_files)
        self.pos_source_idx = np.arange(self.n_pos_source)
        self.n_pos_target = len(dataset.pos_target_files)
        self.pos_target_idx = np.arange(
            self.n_pos_source, self.n_pos_source + self.n_pos_target
        )
        self.n_neg = len(dataset.neg_files) + len(dataset.outlier_files)
        self.neg_idx = np.arange(
            self.n_pos_source + self.n_pos_target,
            self.n_pos_source + self.n_pos_target + self.n_neg,
        )
        self.used_idx_cnt = {"pos_source": 0, "pos_target": 0, "neg": 0}
        self.count = 0
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_target = min(n_target, self.n_pos_target)

    def __iter__(self):
        self.count = 0
        if self.shuffle:
            np.random.shuffle(self.pos_source_idx)
            np.random.shuffle(self.pos_target_idx)
            np.random.shuffle(self.neg_idx)
        while self.count + self.n_pos < self.n_pos_source:
            indices = []
            indices.extend(
                self.pos_source_idx[
                    self.used_idx_cnt["pos_source"] : self.used_idx_cnt["pos_source"]
                    + self.n_pos
                    - self.n_target
                ]
            )
            self.used_idx_cnt["pos_source"] += self.n_pos - self.n_target
            if self.n_target > 0:
                indices.extend(np.random.choice(self.pos_target_idx, self.n_target))
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg"] : self.used_idx_cnt["neg"] + self.n_neg
                ]
            )
            self.used_idx_cnt["neg"] += self.n_neg
            if self.shuffle:
                random.shuffle(indices)
            yield indices
            self.count += self.n_pos - self.n_target
        if not self.drop_last:
            indices = []
            indices.extend(self.pos_source_idx[self.used_idx_cnt["pos_source"] :])
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg"] : self.used_idx_cnt["neg"] + self.n_neg
                ]
            )
            yield indices
        if self.used_idx_cnt["pos_source"] + self.n_pos > self.n_pos_source:
            self.used_idx_cnt["pos_source"] = 0
            self.used_idx_cnt["neg"] = 0

    def __len__(self):
        return self.n_pos_source // self.n_pos


class DomainClassifierBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size=64, shuffle=False, drop_last=False):
        """Batch Sampler.

        Args:
            dataset (dataset): dataset for ASD
            batch_size (int, optional): batch size. Defaults to 64.
            shuffle (bool, optional): shuffle. Defaults to False.
            drop_last (bool, optional): drop last. Defaults to False.
        """
        self.n_pos_so = len(dataset.pos_source_files)
        self.pos_so_idx = np.arange(self.n_pos_so)
        self.n_pos_ta = len(dataset.pos_target_files)
        self.pos_ta_idx = np.arange(self.n_pos_so, self.n_pos_so + self.n_pos_ta)
        self.n_neg = len(dataset.neg_files)
        self.neg_idx = np.arange(
            self.n_pos_so + self.n_pos_ta,
            self.n_pos_so + self.n_pos_ta + self.n_neg,
        )
        self.all_so_idx = np.concatenate([self.pos_so_idx, self.neg_idx])
        self.used_idx_cnt = {"pos_so": 0, "neg_so": 0, "all_so": 0}
        self.count = 0
        self.batch_size = batch_size
        self.n_samples = batch_size // 2
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        self.count = 0
        if self.shuffle:
            np.random.shuffle(self.pos_so_idx)
            np.random.shuffle(self.all_so_idx)
            np.random.shuffle(self.neg_idx)
        while self.count + self.n_samples < self.n_pos_so:
            indices = []
            indices.extend(
                self.pos_so_idx[
                    self.used_idx_cnt["pos_so"] : self.used_idx_cnt["pos_so"]
                    + self.n_samples
                ]
            )
            self.used_idx_cnt["pos_so"] += self.n_samples
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg_so"] : self.used_idx_cnt["neg_so"]
                    + self.n_samples
                ]
            )
            self.used_idx_cnt["neg_so"] += self.n_samples
            indices.extend(
                self.all_so_idx[
                    self.used_idx_cnt["all_so"] : self.used_idx_cnt["all_so"]
                    + self.batch_size
                ]
            )
            self.used_idx_cnt["all_so"] += self.batch_size
            indices.extend(np.random.choice(self.pos_ta_idx, self.batch_size))
            yield indices
            self.count += self.n_samples
        if not self.drop_last:
            indices = []
            indices.extend(self.pos_so_idx[self.used_idx_cnt["pos_so"] :])
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg_so"] : self.used_idx_cnt["neg_so"]
                    + self.n_samples
                ]
            )
            indices.extend(
                self.all_so_idx[
                    self.used_idx_cnt["all_so"] : self.used_idx_cnt["all_so"]
                    + self.batch_size
                ]
            )
            indices.extend(np.random.choice(self.pos_ta_idx, self.batch_size))
            yield indices
        if self.used_idx_cnt["pos_so"] + self.n_samples > self.n_pos_so:
            self.used_idx_cnt["pos_so"] = 0
            self.used_idx_cnt["neg_so"] = 0
            self.used_idx_cnt["all_so"] = 0

    def __len__(self):
        return self.n_pos_so // self.n_samples
