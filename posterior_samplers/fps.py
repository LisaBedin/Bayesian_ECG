import math
from tqdm import trange

import torch
from torch.distributions import Categorical
from ddrm.functions.svd_replacement import H_functions

from posterior_samplers.diffusion_utils import bridge_kernel_statistics
from posterior_samplers.diffusion_utils import EpsilonNetSVD


def fps(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNetSVD,
    H_funcs: H_functions,
    y: torch.Tensor,
    std_y: float,
) -> torch.Tensor:
    """FPS algorithm as described in [1].

    The implementation follows primally Appendix, Algorithm 2.

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    epsilon_net: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

        H_funcs :
        Inverse problem operator.

    y : Tensor
        The observation.

    std_y : float
        The standard deviation.

    References
    ----------
    .. [1] Dou, Zehao, and Yang Song.
        "Diffusion posterior sampling for linear inverse problem solving: A filtering perspective."
        The Twelfth International Conference on Learning Representations. 2024.
    """
    y = H_funcs.Ut(y)
    sing_vals = H_funcs.singulars()

    y_tprev = torch.randn_like(y)
    samples_tprev = initial_noise.reshape(initial_noise.shape[0], -1)

    for i in trange(len(epsilon_net.timesteps) - 1, 1, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        samples_tprev, y_tprev = _fps_step(
            samples_tprev, t, t_prev, epsilon_net, y, y_tprev, sing_vals, std_y
        )

        # # XXX uncomment to view evolution of reconstruction
        # if i % 20 == 0:
        #     e_t = epsilon_net.predict_x0(samples_tprev[[0]], t_prev)
        #     img = H_funcs.V(e_t).reshape(*initial_noise.shape[1:])
        #     display(img)

    # last denoising step
    samples_tprev = epsilon_net.predict_x0(samples_tprev, epsilon_net.timesteps[1])

    # map back to original pixel space
    samples_tprev = H_funcs.V(samples_tprev).reshape(*initial_noise.shape)

    return samples_tprev


def _fps_step(samples, t, t_prev, epsilon_net, y, y_t, singulars, std_y):
    # refer to Appendix: Algorithm 2 and Proposition B.3

    dimy = singulars.shape[0]
    n_samples = samples.shape[0]
    rem_dim = math.prod(samples.shape[1:]) - dimy

    # sample observation as in in Algo 2 (step 5)
    mean_y, std_y = bridge_kernel_statistics(y_t, y, epsilon_net, t, t_prev, 0, eta=1.0)
    y_tprev = mean_y + std_y * singulars * torch.randn_like(mean_y)

    xhat_0 = epsilon_net.predict_x0(samples, t)
    mean_tprev, std_tprev = bridge_kernel_statistics(
        samples, xhat_0, epsilon_net, t, t_prev, s=0, eta=1.0
    )

    var_tprev = std_tprev**2
    var_y = epsilon_net.alphas_cumprod[t_prev] * std_y**2

    # compute log weights weights for SMC resampling (eq. 13)
    log_w = (
        -0.5
        * (y_tprev.flatten() - singulars * mean_tprev[:, :dimy]) ** 2
        / (var_y + (singulars * std_tprev) ** 2)
    ).sum(dim=-1)
    cat = Categorical(logits=log_w)
    idxs = cat.sample((n_samples,))

    samples_tprev = mean_tprev[idxs, :]
    cov_tprev_top = var_tprev * var_y / (var_y + var_tprev * singulars**2)
    mean_tprev_top = cov_tprev_top * (
        (singulars * y_tprev / var_y) + (samples_tprev[:, :dimy] / var_tprev)
    )

    samples_tprev[:, :dimy] = mean_tprev_top + cov_tprev_top.sqrt() * torch.randn_like(
        mean_tprev_top
    )
    samples_tprev[:, dimy:] += std_tprev * torch.randn((n_samples, rem_dim))

    return samples_tprev, y_tprev
