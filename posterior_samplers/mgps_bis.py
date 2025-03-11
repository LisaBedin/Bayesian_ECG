import torch
from typing import Callable
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.diffusion_utils import sample_bridge_kernel
from posterior_samplers.dcps import normalized_grad_step
from posterior_samplers.diffusion_utils import (
    ddim_step,
    bridge_kernel_statistics,
)

from tqdm import tqdm
from functools import partial
import numpy as np


def get_scheduler(t, hi, lo, t_hat, T, tau):
    gamma = -np.log(1 - tau) / (T - t_hat)
    return hi * (1 - np.exp(-gamma * t)) + lo


def _elbo(
    vmean: torch.Tensor,
    vlogstd: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    t: int,
    t_mid: int,
    labels: torch.Tensor = None
):
    acp_t, acp_tmid = (
        epsilon_net.alphas_cumprod[t],
        epsilon_net.alphas_cumprod[t_mid],
    )
    ratio_acp = acp_t / acp_tmid

    x_tmid = vmean + vlogstd.exp() * torch.randn_like(vmean)
    e_tmid = epsilon_net.predict_x0(x_tmid, t_mid, labels)

    with torch.no_grad():
        score_tmid = (-x_tmid + acp_tmid.sqrt() * e_tmid) / (1 - acp_tmid)

    log_pot_val = log_pot(x_tmid) if t_mid == 1 else log_pot(e_tmid)
    target_shape = x_t.shape  # (log_pot_val.shape[0], -1, *x_t.shape[1:])
    axes = tuple(np.arange(1, len(x_t.shape)).astype(int))
    log_fwd = (
        -0.5
        * (
            (x_t.reshape(target_shape) - ratio_acp.sqrt() * vmean.reshape(target_shape)) ** 2 + ratio_acp * (2 * vlogstd.reshape(target_shape)).exp()
        ).sum(dim=axes)
        / (1 - ratio_acp)
    )
    kl_div = -log_pot_val - vlogstd.reshape(target_shape).sum(dim=axes) - log_fwd - (x_tmid * score_tmid).reshape(target_shape).sum(dim=axes)

    return kl_div


def _rb_elbo(
    vmean: torch.Tensor,
    vlogstd: torch.Tensor,
    mean_prior: torch.Tensor,
    logstd_prior: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    t: int,
    labels: torch.Tensor = None
):
    """
    'Rao-Blackwellized' elbo using the Gaussian approximation of the backward transition
    """
    kl_prior = kl_mvn(vmean, vlogstd, mean_prior, logstd_prior)
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)

    if t > 0:
        pred_x0 = epsilon_net.predict_x0(vsample, t, labels)
        int_log_pot_est = -log_pot(pred_x0)
    else:
        int_log_pot_est = -log_pot(vsample)

    return int_log_pot_est + kl_prior


def mgps_vi_tmid_step(
    i: int,
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    optimizer: "str",
    lr: float,
    gradient_steps_fn: Callable[[int], float],
    threshold: float,
    t: int,
    t_prev: int,
    tmid_fn: int,
    labels: torch.Tensor = None
):
    n_gradient_steps = gradient_steps_fn(i)
    t_mid = tmid_fn(t)
    t_mid = t_mid if t_mid <= t_prev else t_prev

    t1 = 0 if t == epsilon_net.timesteps[-1] else 1
    init_mean_tmid, std_prior = bridge_kernel_statistics(
        x_t, pred_x0, epsilon_net, t, t_mid, t1
    )
    logstd_tmid = torch.tensor(std_prior).log() * torch.ones_like(x_t)

    vmean, vlogstd = (
        init_mean_tmid.requires_grad_(),
        logstd_tmid.clone().requires_grad_(),
    )

    if t_prev >= threshold:
        kl_fn = partial(
            _elbo, epsilon_net=epsilon_net, log_pot=log_pot, x_t=x_t, t=t, t_mid=t_mid, labels=labels
        )

    else:
        pred_x0t = epsilon_net.predict_x0(x_t, t, labels)
        mean_tmid, std_tmid = bridge_kernel_statistics(
            x_t, pred_x0t, epsilon_net, t, t_mid, 0
        )
        logstd_tmid = std_tmid.log()

        kl_fn = partial(
            _rb_elbo,
            mean_prior=mean_tmid,
            logstd_prior=logstd_tmid,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            t=t_mid,
            labels=labels
        )

    optim = torch.optim.Adam(params=[vmean, vlogstd], lr=lr)

    if optimizer == "sgd":
        for _ in range(n_gradient_steps):
            vmean.requires_grad_(), vlogstd.requires_grad_()
            kl_div = kl_fn(vmean, vlogstd)
            mean_grad, logstd_grad = torch.autograd.grad(kl_div, (vmean, vlogstd))

            vmean = normalized_grad_step(vmean, mean_grad, lr=lr)
            vlogstd = normalized_grad_step(vlogstd, logstd_grad, lr=lr)

    if optimizer == "adam":
        for _ in range(n_gradient_steps):
            optim.zero_grad()
            kl_div = kl_fn(vmean, vlogstd)
            kl_div.sum().backward()
            optim.step()

    vmean.detach_(), vlogstd.detach_()
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)

    return vsample


def mgps_vi_t0_step(
    i: int,
    x_tmid: torch.Tensor,
    vlogstd: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    optimizer: "str",
    lr: float,
    gradient_steps_fn: Callable[[int], float],
    t_mid: int,
    labels: torch.Tensor = None
):
    n_gradient_steps = gradient_steps_fn(i)

    mean_t1, std_t1 = bridge_kernel_statistics(
        x_ell=x_tmid,
        x_s=epsilon_net.predict_x0(x_tmid, t_mid, labels),
        epsilon_net=epsilon_net,
        ell=t_mid,
        t=1,
        s=0,
    )
    logstd_t1 = torch.tensor(std_t1).log() * torch.ones_like(x_tmid)

    vmean, vlogstd = (
        mean_t1.clone().requires_grad_(),
        logstd_t1.clone().requires_grad_(),
    )

    kl_fn = partial(
        _elbo, epsilon_net=epsilon_net, log_pot=log_pot, x_t=x_tmid, t=t_mid, t_mid=1, labels=labels
    )

    optim = torch.optim.Adam(params=[vmean, vlogstd], lr=lr)

    if optimizer == "sgd":
        for _ in range(n_gradient_steps):
            vmean.requires_grad_(), vlogstd.requires_grad_()
            kl_div = kl_fn(vmean, vlogstd)
            mean_grad, logstd_grad = torch.autograd.grad(kl_div, (vmean, vlogstd))

            vmean = normalized_grad_step(vmean, mean_grad, lr=lr)
            vlogstd = normalized_grad_step(vlogstd, logstd_grad, lr=lr)

    if optimizer == "adam":
        for _ in range(n_gradient_steps):
            optim.zero_grad()
            kl_div = kl_fn(vmean, vlogstd)
            kl_div.sum().backward()
            optim.step()

    vmean.detach_(), vlogstd.detach_()
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)
    return vsample, vlogstd


def mgps_vi_step(
    i: int,
    x_t: torch.Tensor,
    vlogstd: torch.Tensor,
    pred_x0: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    optimizer: "str",
    lr: float,
    lr_t1: Callable[[int], float],
    gradient_steps_fn: Callable[[int], float],
    threshold: float,
    t: int,
    t_prev: int,
    tmid_fn: int,
    labels: torch.Tensor = None
):
    t_mid = tmid_fn(t)
    vsample_tmid = mgps_vi_tmid_step(
        i=i,
        x_t=x_t,
        pred_x0=pred_x0,
        epsilon_net=epsilon_net,
        log_pot=log_pot,
        optimizer=optimizer,
        gradient_steps_fn=gradient_steps_fn,
        lr=lr,
        threshold=threshold,
        t=t,
        t_prev=t_prev,
        tmid_fn=tmid_fn,
        labels=labels
    )

    vsample_t1, vlogstd = mgps_vi_t0_step(
        i=i,
        x_tmid=vsample_tmid,
        vlogstd=vlogstd,
        epsilon_net=epsilon_net,
        log_pot=log_pot,
        optimizer=optimizer,
        gradient_steps_fn=gradient_steps_fn,
        lr=lr_t1(i),
        t_mid=t_mid,
        labels=labels
    )

    x_tprev = sample_bridge_kernel(
        x_ell=x_t, x_s=vsample_t1, epsilon_net=epsilon_net, ell=t, t=t_prev, s=1
    )

    return x_tprev, vsample_t1, vlogstd


def mgps_half(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], float],
    alpha: float,
    gradient_steps_fn: dict,
    optimizer: str,
    lr: float,
    threshold: float,
    ddim_init: bool = False,
    ddim_init_steps: int = 10,
    display_im: bool = True,
    display_fn: Callable = None,
    display_freq: int = 50,
    labels: torch.Tensor = None,
    ground_truth: torch.Tensor = None,
) -> torch.Tensor:

    tmid_fn = lambda t: int(alpha * t)
    n_steps = len(epsilon_net.timesteps)

    def grad_steps_fn(i):
        for rule in gradient_steps_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    lr_t1 = lambda t: get_scheduler(
        t,
        hi=lr,
        lo=1e-4,
        t_hat=0.7 * epsilon_net.timesteps[-1].item(),
        T=epsilon_net.timesteps[-1].item(),
        tau=0.70,
    )

    t = epsilon_net.timesteps[-1]
    t_mid = tmid_fn(t)
    _, std = bridge_kernel_statistics(
        initial_noise, initial_noise, epsilon_net, t, t_mid, 0
    )

    if ddim_init:
        x_tmid = ddim_initialization(
            initial_noise, t, t_mid, ddim_init_steps, epsilon_net, labels=labels
        )
        pred_x0 = epsilon_net.predict_x0(x_tmid, t_mid, labels)
    else:
        pred_x0 = epsilon_net.predict_x0(initial_noise, t, labels)

    x_tprev = initial_noise
    n_samples = x_tprev.shape[0]
    vlogstd = torch.tensor(1e-6).log() * torch.ones_like(x_tprev)

    for i in tqdm(range(n_steps - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        x_tprev, pred_x0, vlogstd = mgps_vi_step(
            i=i,
            x_t=x_tprev,
            vlogstd=vlogstd,
            pred_x0=pred_x0,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            optimizer=optimizer,
            gradient_steps_fn=grad_steps_fn,
            lr=lr,
            lr_t1=lr_t1,
            threshold=threshold,
            t=t,
            t_prev=t_prev,
            tmid_fn=tmid_fn,
            labels=labels
        )

        if i % display_freq == 0:
            if display_im:
                print(f"{i} / {n_steps}")
                j = 0
                # for j in range(n_samples):
                img = epsilon_net.predict_x0(x_tprev[[j]], t_prev, labels)
                display_fn(img, gt=ground_truth[j])

    return epsilon_net.predict_x0(x_tprev, epsilon_net.timesteps[1], labels)


def kl_mvn(
    v_mean: torch.Tensor,
    v_logstd: torch.Tensor,
    mean: torch.Tensor,
    logstd: torch.Tensor,
):
    # NOTE `logstd` must be of shape (1,) and `v_logstd` of shape v_mea
    assert v_mean.shape == v_logstd.shape
    bs = v_mean.shape[0]
    axes = tuple(np.arange(1, len(v_mean.shape)).astype(int))
    return 0.5 * (
        -2 * v_logstd.sum(dim=axes)
        + (torch.norm((v_mean - mean).reshape(bs, -1), dim=-1) ** 2.0 + (2.0 * v_logstd).exp().sum(dim=axes))
        / (2.0 * logstd).exp()
    )


def ddim_initialization(sample, start, end, n_steps, epsilon_net, labels=None) -> torch.Tensor:
    """start > end"""
    interm_timesteps = torch.linspace(end, start, n_steps).int().unique()
    return partial_ddim(sample, epsilon_net, timesteps=interm_timesteps, labels=labels)


def partial_ddim(
    initial_sample: torch.Tensor,
    epsilon_net: EpsilonNet,
    eta: float = 1.0,
    timesteps: torch.Tensor = None,
    labels: torch.Tensor = None
) -> torch.Tensor:
    """Partial sub-sampling with DDIM sampler on specified ``timesteps``."""
    sample = initial_sample

    if timesteps is None:
        timesteps = epsilon_net.timesteps

    for i in range(len(timesteps) - 1, 1, -1):
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=timesteps[i],
            t_prev=timesteps[i - 1],
            eta=eta,
            labels=labels
        )
    return sample


def ddim_coefficients(t: int, t_prev: int, epsilon_net: EpsilonNet):
    alphas_cumprod = epsilon_net.alphas_cumprod

    acp_t_prev = alphas_cumprod[t_prev]
    acp_t_t_prev = alphas_cumprod[t] / alphas_cumprod[t_prev]
    acp_t = alphas_cumprod[t]

    std = ((1 - acp_t_t_prev) * (1 - acp_t_prev) / (1 - acp_t)) ** 0.5
    coeff_x_t = ((1 - acp_t_prev - std**2) / (1 - acp_t)) ** 0.5
    coeff_x_0 = (acp_t_prev**0.5) - coeff_x_t * (acp_t**0.5)

    return coeff_x_0, std