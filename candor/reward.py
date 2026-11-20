"""CANDOR reward terms, called by the patched SOLACE trainer (patches/solace-3c9b4e4.patch)."""
import math

import torch


def log_ratio(mse_cond, mse_uncond):
    """Per-step D: log MSE under the empty prompt minus log MSE under the prompt, both from the same probe noise."""
    return torch.log(mse_uncond + 1e-6) - torch.log(mse_cond + 1e-6)


def candor_scalar(d_per_step, wt, scale=1.0):
    """Reward per sample [B]: per-step z-score of D across the batch, timestep weight wt, mean over the steps."""
    d_norm = (d_per_step - d_per_step.mean(dim=0, keepdim=True)) / d_per_step.std(dim=0, keepdim=True).clamp_min(1e-6)
    return scale * (d_norm * wt).mean(dim=1)


def suffix_start(num_steps, rho=None):
    """First trained sampler step: the last ceil(rho * num_steps) steps, or the latter half if rho is None."""
    return num_steps - math.ceil(rho * num_steps) if rho else int(num_steps * 0.5)

