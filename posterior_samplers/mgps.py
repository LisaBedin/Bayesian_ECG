import torch

from omegaconf import OmegaConf
from posterior_samplers.utils import display
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.diffusion_utils import sample_bridge_kernel
from posterior_samplers.dcps import normalized_grad_step
from posterior_samplers.diffusion_utils import (
    ddim_step,
    bridge_kernel_statistics,
)
from torch.distributions import Distribution
from dataclasses import dataclass
from typing import Callable
from tqdm import tqdm
from functools import partial


@dataclass
class Debugger:
    posterior_samples: torch.Tensor
    prior: Distribution
    display_im: bool = False
    display_freq: bool = None


@dataclass
class VI:
    gradient_steps: int
    optimizer: str
    lr: float
    threshold: float
    sampler: str = "vi"


@dataclass
class VDPScfg:
    tmid_fn: Callable[[int], int]
    ddim_init_steps: int
    init_logstd: float


def load_vdps_sampler(cfg):

    lr, optimizer, alpha, ddim_init_steps, n_steps, threshold = (
        cfg.lr,
        cfg.optimizer,
        cfg.alpha,
        cfg.ddim_init_steps,
        cfg.n_steps,
        cfg.threshold,
    )
    tmid_fn = lambda t: int(alpha * t)

    def grad_steps_fn(i):
        for rule in cfg.grad_steps_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    vi_cfg = VI(
        gradient_steps=grad_steps_fn,
        optimizer=optimizer,
        lr=lambda i: lr,
        threshold=threshold,
    )
    vdps_fn = partial(
        vdps_half,
        tmid_fn=tmid_fn,
        sampler_args=vi_cfg,
        ddim_init_steps=ddim_init_steps,
    )

    return vdps_fn


def _elbo(
    vmean: torch.Tensor,
    vlogstd: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    t: int,
    t_mid: int,
    labels: torch.Tensor = None,
):
    acp_t, acp_tmid = epsilon_net.alphas_cumprod[t], epsilon_net.alphas_cumprod[t_mid]
    ratio_acp = acp_t / acp_tmid

    x_tmid = vmean + vlogstd.exp() * torch.randn_like(vmean)
    e_tmid = epsilon_net.predict_x0(x_tmid, t_mid, labels)

    with torch.no_grad():
        score_tmid = (-x_tmid + acp_tmid.sqrt() * e_tmid) / (1 - acp_tmid)

    log_fwd = (
        -0.5
        * (
            (x_t - ratio_acp.sqrt() * vmean) ** 2 + ratio_acp * (2 * vlogstd).exp()
        ).sum()
        / (1 - ratio_acp)
    )
    kl_div = -log_pot(e_tmid) - vlogstd.sum() - log_fwd - (x_tmid * score_tmid).sum()

    return kl_div


def _rb_elbo(
    vmean: torch.Tensor,
    vlogstd: torch.Tensor,
    mean_prior: torch.Tensor,
    logstd_prior: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    t: int,
    labels: torch.Tensor = None,
):
    """
    'Rao-Blackwellized' elbo using the Gaussian approximation of the backward transition
    """
    kl_prior = kl_mvn(vmean, vlogstd, mean_prior, logstd_prior)
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)
    pred_x0 = epsilon_net.predict_x0(vsample, t, labels)
    int_log_pot_est = -log_pot(pred_x0)

    return int_log_pot_est + kl_prior


def vdps_vi_step(
    i: int,
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    vlogstd: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    sampler_args: VI,
    t: int,
    t_prev: int,
    tmid_fn: int,
    labels: torch.Tensor = None,
):

    optimizer, lr_fn, gradient_steps_fn = (
        sampler_args.optimizer,
        sampler_args.lr,
        sampler_args.gradient_steps,
    )

    n_gradient_steps = gradient_steps_fn(i)
    lr = lr_fn(i)

    t_mid = tmid_fn(t)
    t_mid = t_mid if t_mid <= t_prev else t_prev

    init_mean_tmid, std_prior = bridge_kernel_statistics(
        x_t, pred_x0, epsilon_net, t, t_mid, 0
    )

    logstd_tmid = torch.tensor(std_prior).log() * torch.ones_like(x_t)

    vmean, vlogstd = (
        init_mean_tmid.requires_grad_(),
        logstd_tmid.clone().requires_grad_(),
    )

    if t_prev >= sampler_args.threshold:
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
            kl_div.backward()
            optim.step()

    vmean.detach_(), vlogstd.detach_()
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)

    pred_x0_tmid = epsilon_net.predict_x0(vsample, t_mid, labels)

    x_tprev = sample_bridge_kernel(
        x_ell=x_t, x_s=vsample, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_mid
    )

    return x_tprev, pred_x0_tmid, vlogstd


def vdps_half(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    tmid_fn: Callable[[int], int],
    sampler_args: VI,
    ddim_init_steps: int = 10,
    display_freq: int = 10,
    display_fn: Callable = display,
    ground_truth: torch.Tensor = None,
    labels: torch.Tensor = None,
) -> torch.Tensor:

    x_tprev = initial_noise
    n_samples = x_tprev.shape[0]

    t = epsilon_net.timesteps[-1]
    t_mid = tmid_fn(t)
    x_tmid = ddim_init(initial_noise, t, t_mid, ddim_init_steps, epsilon_net, labels=labels);
    pred_x0 = epsilon_net.predict_x0(x_tmid, t_mid, labels)
    _, std = bridge_kernel_statistics(
        initial_noise, initial_noise, epsilon_net, t, t_mid, 0
    )
    vlogstd = std.log() * torch.ones_like(initial_noise)

    for i in tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        x_tprev, pred_x0, vlogstd = vdps_vi_step(
            i=i,
            x_t=x_tprev,
            pred_x0=pred_x0,
            vlogstd=vlogstd,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            sampler_args=sampler_args,
            t=t,
            t_prev=t_prev,
            tmid_fn=tmid_fn,
            labels=labels
        )
        # print(x_tprev.mean())
        if i % display_freq == 0:
            print(f"{i} / {len(epsilon_net.timesteps)}")
            # for j in range(n_samples):
            j = 0
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

    return 0.5 * (
        -2 * v_logstd.sum()
        + (torch.norm(v_mean - mean) ** 2.0 + (2.0 * v_logstd).exp().sum())
        / (2.0 * logstd).exp()
    )


def ddim_init(sample, start, end, n_steps, epsilon_net, labels=None):
    """start > end"""
    interm_timesteps = torch.linspace(end, start, n_steps).int().unique()
    return partial_ddim(sample, epsilon_net, timesteps=interm_timesteps, labels=labels)


def partial_ddim(
    initial_sample: torch.Tensor,
    epsilon_net: EpsilonNet,
    eta: float = 1.0,
    timesteps: torch.Tensor = None,
    labels: torch.Tensor = None,
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
