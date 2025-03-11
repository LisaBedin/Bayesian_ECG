from tqdm import trange

import torch
from ddrm.functions.svd_replacement import H_functions
from posterior_samplers.diffusion_utils import EpsilonNet


def sda(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    H_funcs: H_functions,
    y: torch.Tensor,
    sigma: float,
    n_correction_steps: int = 1,
    tau: float = 1.0,
    gamma=1e-2,
) -> torch.Tensor:
    """SDA algorithm as described in [1].

    This implement the algorithm 3 and 4 in Appendix C.

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

    sigma : float
        The standard deviation.

    n_correction_steps : int, default = 1
        The number of Langevin steps to be perform.

    tau : float, default=1
        The intensity of the Langevin stepsize.

    gamma : float or Tensor
        The diagonal approximation the second term in the covariance
        (Algorithm 3 step 4).

    References
    ----------
    .. [1] Rozet, François, and Gilles Louppe. "Score-based data assimilation."
        Advances in Neural Information Processing Systems 36 (2023): 40521-40541.
    """
    acp = epsilon_net.alphas_cumprod
    timesteps = epsilon_net.timesteps

    x_t = torch.randn_like(initial_noise)
    for it, i in enumerate(trange(len(timesteps) - 1, 1, -1), start=1):
        t, t_prev = timesteps[i], timesteps[i - 1]
        acp_t, acp_t_prev = acp[t], acp[t_prev]

        score_y = _get_conditional_score(x_t, t, epsilon_net, y, H_funcs, sigma, gamma)

        sigma_t_square = 1 - acp_t
        ratio_mu = acp_t_prev.sqrt() / acp_t.sqrt()
        ratio_sigma = (1 - acp_t_prev).sqrt() / (1 - acp_t).sqrt()

        x_t = ratio_mu * x_t + (ratio_mu - ratio_sigma) * sigma_t_square * score_y

        # Langevin corrections
        for _ in range(n_correction_steps):
            # NOTE here it should be t_prev as we already performed the update
            score_y = _get_conditional_score(
                x_t, t_prev, epsilon_net, y, H_funcs, sigma, gamma
            )

            dim = [k for k in range(1, x_t.ndim)]
            delta = tau / (score_y**2).mean(dim=dim, keepdim=True)

            noise = torch.randn_like(x_t)
            x_t = x_t + delta * score_y + (2 * delta).sqrt() * noise

        # # XXX uncomment to view evolution of reconstruction
        # if it % 30 == 0:
        #     img = epsilon_net.predict_x0(x_t[[0]], t_prev)
        #     display(img)

    return x_t


def _get_conditional_score(
    x_t: torch.Tensor,
    t: int,
    epsilon_net: EpsilonNet,
    y: torch.Tensor,
    H_funcs: H_functions,
    sigma: float,
    gamma: float = 1e-2,
):
    # compute grad log of
    #   p = N(
    #          mean = H x_0t(x_t),
    #          var  = sigma_y**2 * I + var_0t * gamma
    # )
    # provided the SVD of H = U Sigma V^T
    # To do so, compute
    #  - 0.5 * || (y - mean) / sqrt(var) ||^2
    # and then back-propagate
    acp_t = epsilon_net.alphas_cumprod[t]
    var_0t = (1 - acp_t) / acp_t
    singulars = H_funcs.singulars()

    x_t.requires_grad_()
    score_t = epsilon_net.score(x_t, t)
    x_0t = (x_t + (1 - acp_t) * score_t) / acp_t.sqrt()

    diff = y - H_funcs.H(x_0t)

    diff[:, : len(singulars)] /= (sigma**2 + var_0t * gamma).sqrt()
    diff[:, len(singulars) :] /= sigma

    err = -0.5 * (diff**2).sum()
    err.backward()
    grad = x_t.grad

    # to avoid recording/accumulating grad in later iteration
    x_t.requires_grad_(False)

    with torch.no_grad():
        score_y = score_t + grad

    return score_y
