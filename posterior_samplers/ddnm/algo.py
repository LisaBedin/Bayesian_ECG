import torch
from tqdm import tqdm

from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.ddnm.utils import get_special_methods

class_num = 951


@torch.no_grad()
def ddnm(
    x: torch.Tensor,
    labels: torch.Tensor,
    model: EpsilonNet,
    A_funcs,
    y: torch.Tensor,
    eta: float = 0.85,
    config=None,
) -> torch.Tensor:
    """DDNM algorithm without noise as described in [1].

    Parameters
    ----------
    x : Tensor
        initial noise

    model: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    A_funcs :
        Inverse problem operator.

    y : Tensor
        The observation.

    eta : float
        Hyperparameter, default to 0.85 as in
        https://github.com/wyhuai/DDNM/blob/main/README.md#quick-start

    config : Dict
        configuration.

    References
    ----------
    .. [1] Wang, Yinhuai, Jiwen Yu, and Jian Zhang.
        "Zero-shot image restoration using denoising diffusion null-space model."
        arXiv preprint arXiv:2212.00490 (2022).
    """
    device = x.device

    # deduce b from alphas_cumprod
    # b ---> betas in DDPM
    # c.f. https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/guided_diffusion/diffusion.py#L588
    b = 1 - model.alphas_cumprod[1:] / model.alphas_cumprod[:-1]

    # use default configs of DDNM as in
    # https://github.com/wyhuai/DDNM/blob/main/configs/celeba_hq.yml
    if config is None:
        config = {
            "diffusion": {
                "num_diffusion_timesteps": len(model.alphas_cumprod),
            },
            "time_travel": {
                "T_sampling": len(model.timesteps),
                "travel_length": 1,
                "travel_repeat": 1,
            },
        }

    # setup iteration variables
    skip = (
        config["diffusion"]["num_diffusion_timesteps"]
        // config["time_travel"]["T_sampling"]
    )
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = get_schedule_jump(
        config["time_travel"]["T_sampling"],
        config["time_travel"]["travel_length"],
        config["time_travel"]["travel_repeat"],
    )
    time_pairs = list(zip(times[:-1], times[1:]))

    # reverse diffusion sampling
    for it, (i, j) in enumerate(tqdm(time_pairs), start=1):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        if j < i:  # normal sampling
            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)

            try:
                et = model(xt, t, labels)
            except RuntimeError:
                et = model(xt, t[0], labels)

            # NOTE we don't currently account for unconditional sampling
            # to add support of it see
            # https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/functions/svd_ddnm.py#L46-L52

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            x0_t_hat = x0_t - A_funcs.H_pinv(
                A_funcs.H(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())

            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta**2) ** 0.5)
            xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

            x0_preds.append(x0_t.to("cpu"))
            xs.append(xt_next.to("cpu"))

            # XXX uncomment to view evolution of images
            # if it % 50 == 0:
            #     img = model.predict_x0(xt_next[[0]], next_t.long()[0])
            #     display(img)

        else:  # time-travel back
            next_t = (torch.ones(n) * j).to(x.device)
            at_next = compute_alpha(b, next_t.long())
            x0_t = x0_preds[-1].to(device)

            xt_next = (
                at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
            )

            xs.append(xt_next.to("cpu"))

    return xt_next


@torch.no_grad()
def ddnm_plus(
    x: torch.Tensor,
    labels: torch.Tensor,
    model: EpsilonNet,
    type_problem: str,
    A_funcs,
    y,
    sigma_y: float,
    eta: float = 0.85,
    config=None,
) -> torch.Tensor:
    """DDNM algorithm with noise as described in [1].

    Parameters
    ----------
    x : Tensor
        initial noise

    model: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    type_problem : "sr" or "inpainting"
        "sr" include all Super Resolution problem "sr4" or "sr16".
        "inpainting" include completion problems, "outpainting_half, ...

    A_funcs :
        Inverse problem operator.

    y : Tensor
        The observation.

    sigma_y : float
        The standard deviation of the problem.

    eta : float
        Hyperparameter, default to 0.85 as in
        https://github.com/wyhuai/DDNM/blob/main/README.md#quick-start

    config : Dict
        configuration.

    References
    ----------
    .. [1] Wang, Yinhuai, Jiwen Yu, and Jian Zhang.
        "Zero-shot image restoration using denoising diffusion null-space model."
        arXiv preprint arXiv:2212.00490 (2022).
    """
    device = x.device

    # make SVD operator compatible with DDNM
    Lambda_func, Lambda_noise_func = get_special_methods(A_funcs, type_problem)

    # deduce b from alphas_cumprod
    # b ---> betas in DDPM
    # c.f. https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/guided_diffusion/diffusion.py#L588
    b = 1 - model.alphas_cumprod[1:] / model.alphas_cumprod[:-1]

    # use default configs of DDNM as in
    # https://github.com/wyhuai/DDNM/blob/main/configs/celeba_hq.yml
    if config is None:
        config = {
            "diffusion": {
                "num_diffusion_timesteps": len(model.alphas_cumprod),
            },
            "time_travel": {
                "T_sampling": len(model.timesteps),
                "travel_length": 1,
                "travel_repeat": 1,
            },
        }

    # setup iteration variables
    skip = (
        config["diffusion"]["num_diffusion_timesteps"]
        // config["time_travel"]["T_sampling"]
    )
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = get_schedule_jump(
        config["time_travel"]["T_sampling"],
        config["time_travel"]["travel_length"],
        config["time_travel"]["travel_repeat"],
    )
    time_pairs = list(zip(times[:-1], times[1:]))

    # reverse diffusion sampling
    for i, j in tqdm(time_pairs):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        if j < i:  # normal sampling
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)

            et = model(xt, t, labels)

            # NOTE we don't currently account for unconditional sampling
            # to add support of it see
            # https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/functions/svd_ddnm.py#L46-L52

            # Eq. 12
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            sigma_t = (1 - at_next).sqrt()[0, 0, 0]

            # Eq. 17
            x0_t_hat = x0_t - Lambda_func(
                A_funcs.H_pinv(
                    A_funcs.H(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, 0, 0],
                sigma_y,
                sigma_t,
                eta,
            ).reshape(*x0_t.size())

            # Eq. 51
            xt_next = at_next.sqrt() * x0_t_hat + Lambda_noise_func(
                torch.randn_like(x0_t).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, 0, 0],
                sigma_y,
                sigma_t,
                eta,
                et.reshape(et.size(0), -1),
            ).reshape(*x0_t.size())

            x0_preds.append(x0_t.to("cpu"))
            xs.append(xt_next.to("cpu"))
        else:  # time-travel back
            next_t = (torch.ones(n) * j).to(x.device)
            at_next = compute_alpha(b, next_t.long())
            x0_t = x0_preds[-1].to(device)

            xt_next = (
                at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
            )

            xs.append(xt_next.to("cpu"))

    return xt_next


# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts


def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)
