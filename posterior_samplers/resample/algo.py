import torch
from functools import partial

from ldm.models.diffusion.ddpm import LatentDiffusion
from ddrm.functions.svd_replacement import H_functions
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.resample.utils import DDIMSampler, get_conditioning_method


def resample(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    H_funcs: H_functions,
    y: torch.Tensor,
    noise_type: str = "gaussian",
    eta: float = 1.0,
) -> torch.Tensor:
    """Wrapper around Resample algorithm [1].

    Source of the implementation https://github.com/soominkwon/resample

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

    noise_type : str
        Either "gaussian" or "poisson".

    eta : float
        The coefficient that multiplies DDIM variance.

    References
    ----------
    .. [1] Song, Bowen, et al. "Solving inverse problems with latent diffusion
        models via hard data consistency." arXiv preprint arXiv:2307.08123 (2023).
    """
    # check model abides by LDM
    latent_model = getattr(epsilon_net.net, "net", None)
    if not isinstance(latent_model, LatentDiffusion):
        raise ValueError("ReSample algorithm is only compatible with `LatentDiffusion`")

    # BUG currently ReSample doesn't throws an index out of bound when
    # the number diffusion steps is different than 500
    # c.f. https://github.com/soominkwon/resample/issues/5
    if len(epsilon_net.timesteps) != 500:
        raise ValueError(
            "ReSample algorithm supports only 500 diffusion steps.\n"
            "Change Diffusion steps to 500."
        )

    # init
    n_samples = initial_noise.shape[0]
    sampler = DDIMSampler(epsilon_net)
    measurement_cond_fn = get_conditioning_method(
        "ps", epsilon_net.net.net, H_funcs.H, noiser=noise_type
    ).conditioning

    # Instantiate sampler
    sample_fn = partial(
        sampler.posterior_sampler,
        measurement_cond_fn=measurement_cond_fn,
        operator_fn=H_funcs.H,
        S=len(epsilon_net.timesteps),
        cond_method="resample",
        conditioning=None,
        ddim_use_original_steps=True,
        batch_size=n_samples,
        shape=[3, 64, 64],
        verbose=False,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        eta=eta,
    )

    # solve problem
    samples_ddim, _ = sample_fn(measurement=y)

    return samples_ddim
