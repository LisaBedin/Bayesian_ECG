from tqdm import trange

import torch
from posterior_samplers.svd_replacement import H_functions
from posterior_samplers.diffusion_utils import ddim_step, EpsilonNetSVD


def repaint_svd(
    initial_noise: torch.Tensor,
    H_funcs: H_functions,
    y: torch.Tensor,
    labels: torch.Tensor,
    epsilon_net: EpsilonNetSVD,
    n_reps: int = 2,
    eta: float = 1.0,
) -> torch.Tensor:
    """Generalized RePaint algorithm for noiseless linear inverse problems.

    This is a modified version of the original algorithm [1].

    The algorithm operates on the orthonormal basis defined by the SVD.
    Pass in the wrapper ``EpsilonNetSVD`` around ``EpsilonNet``.

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    epsilon_net_svd: Instance of EpsilonNetSVD
        Noise predictor coming from a diffusion model.

    H_funcs :
        Inverse problem operator.

    y : Tensor
        The observation.

    n_reps : int, default = 2
        The number of repetition of each diffusion steps.

    References
    ----------
    .. [1] Lugmayr, Andreas, et al. "Repaint: Inpainting using denoising diffusion probabilistic models."
        Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
    """
    # ensure y has the right shape (1, -1)
    y = y.reshape(1, -1)
    Ut_y, diag = H_funcs.Ut(y), H_funcs.singulars()

    timesteps = epsilon_net.timesteps
    alphas_cumprod = epsilon_net.alphas_cumprod

    sample = initial_noise.reshape(initial_noise.shape[0], -1)
    for i in trange(len(timesteps) - 1, 1, -1):
        t, t_prev = timesteps[i], timesteps[i - 1]

        for r in range(n_reps):

            acp_t, acp_tprev = alphas_cumprod[t], alphas_cumprod[t_prev]

            sample = ddim_step(
                x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, labels=labels
            )

            # replace with known values
            # NOTE don't forget rescale by the diag
            noise = torch.randn_like(Ut_y)
            sample[:, : diag.shape[0]] = (
                acp_tprev.sqrt() * Ut_y / diag + (1 - acp_tprev).sqrt() * noise
            )

            # Don't get back to x_t in the last rep
            if r == n_reps - 1:
                continue

            a_t = acp_t / acp_tprev
            noise = torch.randn_like(sample)
            sample = a_t.sqrt() * sample + (1 - a_t).sqrt() * noise

    # last diffusion step
    # sample = epsilon_net.predict_x0(sample, timesteps[1])
    sample = epsilon_net.predict_x0(sample, timesteps[1], labels)

    # map back to original pixel space
    sample = H_funcs.V(sample).reshape(initial_noise.shape)

    return sample
