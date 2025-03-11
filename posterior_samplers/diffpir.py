from tqdm import trange
import torch
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.svd_replacement import H_functions


def diffpir(
    initial_noise: torch.Tensor,
    labels: torch.Tensor,
    epsilon_net: EpsilonNet,
    H_funcs: H_functions,
    y: torch.Tensor,
    sigma: float,
    lmbd: float,
    zeta: float,
    n_reps: int = 1,
) -> torch.Tensor:
    """DiffPIR algorithm as described in [1].

    Refer to Table 3 for setting the hyperparameters ``lmbd`` and ``zeta``.

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

    lmbd : float
        Hyperparameter.

    zeta : float
        Hyperparameter.

    n_reps : int, default = 1
        The number of times to repeat a diffusion step.

    References
    ----------
    .. [1] Zhu, Yuanzhi, et al. "Denoising diffusion models for plug-and-play image restoration."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """
    epsilon_net.requires_grad_(False)
    acp = epsilon_net.alphas_cumprod
    timesteps = epsilon_net.timesteps

    n_samples = initial_noise.shape[0]
    zeta = torch.tensor(zeta, dtype=initial_noise.dtype)

    x_t = torch.randn_like(initial_noise)
    for it, i in enumerate(trange(len(timesteps) - 1, 1, -1), start=1):
        t, t_prev = timesteps[i], timesteps[i - 1]
        acp_t, acp_t_prev = acp[t], acp[t_prev]

        for _ in range(n_reps):
            # --- step 1
            x_0t: torch.Tensor = epsilon_net.predict_x0(x_t, t, labels)

            # --- step 2
            # solve data fitting problem
            rho_t = sigma**2 * lmbd / ((1 - acp_t) / acp_t)

            x_0t = _argmin_quadratic_problem(
                A=H_funcs,
                gamma=rho_t,
                b=rho_t * x_0t.view(n_samples, -1) + H_funcs.Ht(y),
            )
            x_0t = x_0t.view(*initial_noise.shape)

            # --- step 3
            e_t_y = (x_t - acp_t.sqrt() * x_0t) / (1 - acp_t).sqrt()
            noise = (1 - zeta).sqrt() * e_t_y + zeta.sqrt() * torch.randn_like(x_t)

            x_t = acp_t_prev.sqrt() * x_0t + (1 - acp_t_prev).sqrt() * noise

        # # XXX uncomment to view evolution of images
        # if it % 50 == 0:
        #     img = epsilon_net.predict_x0(x_t[[0]], t_prev.long())
        #     display(img)

    return x_t


def _argmin_quadratic_problem(
    A: H_functions, gamma: float, b: torch.Tensor
) -> torch.Tensor:
    """Solve for x the problem ``(gamma * I + A.T @ A) x = b``"""
    singulars = A.singulars()

    out = A.Vt(b)
    out[:, : len(singulars)] /= gamma + singulars**2
    out[:, len(singulars) :] /= gamma
    out = A.V(out)

    return out
