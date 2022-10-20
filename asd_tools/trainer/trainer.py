import logging
import os

from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from torchaudio.functional import pitch_shift
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter

from asd_tools.utils import mixup_for_asd
from asd_tools.utils import mixup_apply_rate


class MetricOECTrainer(object):
    """Customized trainer module for OEC training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
        train=False,
        metric_fc=None,
    ):
        """Initialize trainer."""
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.train = train
        self.metric_fc = metric_fc
        if train:
            self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.epoch_valid_pred_machine = np.empty((0, 1))
        self.epoch_valid_y_machine = np.empty((0, 1))
        self.epoch_valid_y_section = np.empty(0)
        self.epoch_valid_pred_section = np.empty((0, 6))
        self.total_train_loss = defaultdict(float)
        self.total_valid_loss = defaultdict(float)
        self.best_loss = 99999
        self.steps_per_epoch = 99999
        self.forward_count = 0

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.epochs, total=self.config["train_max_epochs"], desc="[train]"
        )
        while True:
            self._train_epoch()
            if self.epochs % self.config["log_interval_epochs"] == 0:
                self._valid_epoch()
            self._check_save_interval()
            # check whether training is finished
            self._check_train_finish()
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path, save_model_only=True):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            save_model_only (bool): Whether to save model parameters only.
        """
        state_dict = {
            "steps": self.steps,
            "epochs": self.epochs,
            "best_loss": self.best_loss,
        }
        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()
        if not save_model_only:
            state_dict["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state_dict["scheduler"] = self.scheduler.state_dict()
        if self.metric_fc is not None:
            state_dict["metric_fc"] = self.metric_fc.weight.cpu()
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state_dict, checkpoint_path)
        self.last_checkpoint = checkpoint_path

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.best_loss = state_dict.get("best_loss", 99999)
            logging.info(
                f"Steps:{self.steps}, Epochs:{self.epochs}, BEST loss:{self.best_loss}"
            )
            if (self.optimizer is not None) and (
                state_dict.get("optimizer", None) is not None
            ):
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if (self.scheduler is not None) and (
                state_dict.get("scheduler", None) is not None
            ):
                self.scheduler.load_state_dict(state_dict["scheduler"])
            if self.metric_fc is not None:
                tmp = self.metric_fc.weight
                self.metric_fc.weight = torch.nn.Parameter(state_dict["metric_fc"])
                self.metric_fc = self.metric_fc.to(self.device)
                if (tmp - self.metric_fc.weight).sum() != 0:
                    logging.info("Successfully load weights of metric_fc.")

    def _train_step(self, batch):
        """Train model one step."""
        machine = batch["machine"].to(self.device)
        section_idx = machine.bool()
        machine = machine.unsqueeze(1)
        section = batch["section"].to(self.device)
        if self.config["section_loss_type"] == "BCEWithLogitsLoss":
            section = torch.nn.functional.one_hot(section, num_classes=6).float()
        wave = batch["wave"].to(self.device)
        if self.config.get("PitchShift") is not None:
            if np.random.rand() < self.config.get("apply_rate", 1.0):
                n_steps = np.random.normal(loc=0, scale=2, size=1)[0]
                wave = pitch_shift(wave, n_steps=n_steps, **self.config["PitchShift"])
        if self.config.get("mixup_alpha", 0) > 0:
            if np.random.rand() < mixup_apply_rate(
                max_step=self.config["train_max_epochs"] * self.steps_per_epoch,
                step=self.steps,
                **self.config["mixup_scheduler"],
            ):
                wave, machine, section, section_idx = mixup_for_asd(
                    wave,
                    machine,
                    section,
                    mix_section=self.config["section_loss_type"] == "BCEWithLogitsLoss",
                    alpha=self.config["mixup_alpha"],
                )
        y_ = self.model(wave)
        machine_loss = (
            self.criterion["machine_loss"](y_["machine"], machine)
            / self.config["accum_grads"]
        )
        section_pred = y_["section"][section_idx]
        if self.metric_fc is not None:
            section_pred = self.metric_fc(
                y_["embedding"][section_idx], section[section_idx]
            )
        section_loss = (
            self.criterion["section_loss"](section_pred, section[section_idx])
            / self.config["accum_grads"]
        )
        loss = (
            self.config.get("machine_loss_lambda", 1) * machine_loss
            + self.config["section_loss_lambda"] * section_loss
        )

        logging.debug(f"backward:{loss.item():.4f}")
        loss.backward()

        self.forward_count += 1
        if self.forward_count == self.config["accum_grads"]:
            self.total_train_loss["train/machine_loss"] += machine_loss.item()
            self.total_train_loss["train/section_loss"] += section_loss.item()
            self.total_train_loss["train/loss"] += loss.item()

            # update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.forward_count = 0

            # update scheduler step
            if self.scheduler is not None:
                if self.config["scheduler_type"] != "CosineLRScheduler":
                    self.scheduler.step()
            # update counts
            self.steps += 1

    def _train_epoch(self):
        """Train model one epoch."""
        self.model.train()
        if self.metric_fc is not None:
            self.metric_fc.train()
        for steps_per_epoch, batch in enumerate(self.data_loader["train"]):
            # train one step
            self._train_step(batch)

            # check whether training is finished
            if self.finish_train:
                return
        # log
        self.steps_per_epoch = steps_per_epoch + 1
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({steps_per_epoch} steps per epoch)."
        )
        self._write_to_tensorboard(
            self.total_train_loss, steps_per_epoch=steps_per_epoch + 1
        )
        self.tqdm.update(1)
        self.epochs += 1
        if self.config.get("scheduler_type", None) == "CosineLRScheduler":
            self.scheduler.step(self.epochs)
        self.total_train_loss = defaultdict(float)

    def _valid_step(self, batch):
        """Validate model one step."""
        machine = batch["machine"].to(self.device)
        section_idx = batch["machine"].bool()
        section = batch["section"].to(self.device)[section_idx]
        if self.config["section_loss_type"] == "BCEWithLogitsLoss":
            section = torch.nn.functional.one_hot(section, num_classes=6).float()
        with torch.no_grad():
            y_ = self.model(batch["wave"].to(self.device))
            machine_loss = (
                self.criterion["machine_loss"](y_["machine"], machine.unsqueeze(1))
                / self.config["accum_grads"]
            )
            section_pred = y_["section"][section_idx]
            if self.metric_fc is not None:
                section_pred = self.metric_fc(y_["embedding"][section_idx], section)
            section_loss = (
                self.criterion["section_loss"](section_pred, section)
                / self.config["accum_grads"]
            )
            loss = (
                self.config.get("machine_loss_lambda", 1) * machine_loss
                + self.config["section_loss_lambda"] * section_loss
            )
            self.forward_count += 1
            if self.forward_count == self.config["accum_grads"]:
                self.total_valid_loss["valid/machine_loss"] += machine_loss.item()
                self.total_valid_loss["valid/section_loss"] += section_loss.item()
                self.total_valid_loss["valid/loss"] += loss.item()
                self.forward_count = 0
        self.epoch_valid_pred_machine = np.concatenate(
            [self.epoch_valid_pred_machine, y_["machine"].cpu().numpy()]
        )
        self.epoch_valid_pred_section = np.concatenate(
            [self.epoch_valid_pred_section, section_pred.cpu().numpy()]
        )
        self.epoch_valid_y_machine = np.concatenate(
            [self.epoch_valid_y_machine, machine.cpu().numpy()[:, np.newaxis]]
        )
        if self.config["section_loss_type"] == "BCEWithLogitsLoss":
            section = torch.argmax(section, dim=1)
        self.epoch_valid_y_section = np.concatenate(
            [self.epoch_valid_y_section, section.cpu().numpy()]
        )

    def _valid_epoch(self):
        """Validate model one epoch."""
        self.model.eval()
        if self.metric_fc is not None:
            self.metric_fc.eval()
        for steps_per_epoch, batch in enumerate(self.data_loader["valid"]):
            self._valid_step(batch)
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch validation "
            f"({steps_per_epoch} steps per epoch)."
        )
        self._write_to_tensorboard(
            self.total_valid_loss, steps_per_epoch=steps_per_epoch + 1
        )
        machine_auc = roc_auc_score(
            self.epoch_valid_y_machine, self.epoch_valid_pred_machine
        )
        section_micro_f1 = f1_score(
            self.epoch_valid_y_section,
            np.argmax(self.epoch_valid_pred_section, 1),
            average="micro",
        )
        self._write_to_tensorboard(
            {
                "valid/machine_auc": machine_auc,
                "valid/section_micro_f1": section_micro_f1,
            }
        )
        self.total_valid_loss["valid/loss"] /= steps_per_epoch + 1
        if self.best_loss > self.total_valid_loss["valid/loss"]:
            self.best_loss = self.total_valid_loss["valid/loss"]
            logging.info(f"BEST Loss is updated: {self.best_loss:.5f}")
            self.save_checkpoint(
                os.path.join(self.config["outdir"], "best_loss", "best_loss.pkl"),
                save_model_only=False,
            )
        self.epoch_valid_pred_machine = np.empty((0, 1))
        self.epoch_valid_y_machine = np.empty((0, 1))
        self.epoch_valid_y_section = np.empty(0)
        self.epoch_valid_pred_section = np.empty((0, 6))
        self.total_valid_loss = defaultdict(float)

    def _write_to_tensorboard(self, loss, steps_per_epoch=1):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value / steps_per_epoch, self.epochs)
            logging.info(
                f"(Epochs: {self.epochs}) {key} = {value / steps_per_epoch:.5f}."
            )

    def _check_save_interval(self):
        if (self.epochs % self.config["save_interval_epochs"] == 0) and (
            self.epochs != 0
        ):
            self.save_checkpoint(
                os.path.join(
                    self.config["outdir"],
                    f"checkpoint-{self.epochs}epochs",
                    f"checkpoint-{self.epochs}epochs.pkl",
                ),
                save_model_only=False,
            )
            logging.info(f"Successfully saved checkpoint @ {self.epochs} epochs.")

    def _check_train_finish(self):
        if self.epochs >= self.config["train_max_epochs"]:
            self.finish_train = True
