import logging
import math
import torch
import numpy as np


def mixup_for_asd(
    X: torch.tensor,
    Y: torch.tensor,
    section: torch.tensor,
    mix_section=False,
    alpha=0.2,
    mode=None,
):
    """MixUp for ASD."""
    with torch.no_grad():
        batch_size = X.size(0)
        x_size = len(X.shape)
        lam = torch.tensor(
            np.random.beta(alpha, alpha, batch_size), dtype=torch.float32
        ).to(X.device)[:, None]
        perm = torch.randperm(batch_size).to(X.device)
        if x_size == 3:
            mixed_X = lam[:, None] * X + (1 - lam[:, None]) * X[perm]
        elif x_size == 2:
            mixed_X = lam * X + (1 - lam) * X[perm]
        logging.debug(
            f"lam:{lam.shape}, prem:{perm.shape}, Y:{Y.shape}, Y[perm]:{Y[perm].shape}, "
            f"lam*Y:{(lam*Y).shape}, (1 - lam) * Y[perm]:{((1 - lam) * Y[perm]).shape}"
        )
        mixed_Y = lam * Y + (1 - lam) * Y[perm]
        mixed_section = (
            lam * section + (1 - lam) * section[perm] if mix_section else section
        )
    section_idx = (
        (0 < mixed_Y).squeeze(1)
        if mix_section
        else ((0 < mixed_Y) & (mixed_Y < 1)).squeeze(1)
    )
    if mode == "hard":
        mixed_Y = (mixed_Y > 0.0).float()
        mixed_section = (mixed_section > 0.0).float()
    elif mode == "soft":
        mixed_Y = (~(mixed_Y < 1.0)).float()
        mixed_section = (~(mixed_section < 1.0)).float()
    return mixed_X, mixed_Y, mixed_section, section_idx


def mixup_for_outlier(
    X: torch.tensor,
    Y: torch.tensor,
    section: torch.tensor,
    alpha=0.2,
    mode="vanilla",
):
    """MixUp for ASD."""
    batch_size = X.size(0)
    lam = torch.tensor(
        np.random.beta(alpha, alpha, batch_size), dtype=torch.float32
    ).to(X.device)[:, None]
    perm = torch.randperm(batch_size).to(X.device)
    mixed_X = lam * X + (1 - lam) * X[perm]
    logging.debug(
        f"lam:{lam.shape}, prem:{perm.shape}, Y:{Y.shape}, Y[perm]:{Y[perm].shape}, "
        f"lam*Y:{(lam*Y).shape}, (1 - lam) * Y[perm]:{((1 - lam) * Y[perm]).shape}"
    )
    mixed_Y = lam * Y + (1 - lam) * Y[perm]
    mixed_section = lam * section + (1 - lam) * section[perm]
    section_idx = (0 < mixed_Y).squeeze(1)

    return mixed_X, mixed_Y, mixed_section, section_idx


def mixup_for_domain_classifier_target(
    ta_wave: torch.tensor, ta_machine: torch.tensor, ta_section: torch.tensor, alpha=0.2
):
    batch_size = len(ta_wave) // 2
    lam = torch.tensor(
        np.random.beta(alpha, alpha, batch_size // 2), dtype=torch.float32
    )
    lam = torch.cat([lam, torch.zeros(batch_size - len(lam))]).to(ta_wave.device)[
        :, None
    ]
    ta_wave = lam * ta_wave[:batch_size] + (1 - lam) * ta_wave[-batch_size:]
    ta_machine = lam * ta_machine[:batch_size] + (1 - lam) * ta_machine[-batch_size:]
    ta_section = lam * ta_section[:batch_size] + (1 - lam) * ta_section[-batch_size:]
    ta_section_idx = (0 < ta_machine).squeeze(1)
    return ta_wave, ta_machine, ta_section, ta_section_idx


def schedule_cos_phases(max_step, step):
    return 0.5 * (1.0 - math.cos(min(math.pi, 2 * math.pi * step / max_step)))


def count_params(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params
