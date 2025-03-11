import torch
import tqdm
import numpy as np

from posterior_samplers.utils import fwd_mixture

from torch.func import grad
from torch.distributions import Distribution


class UNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        return self.unet(x, torch.tensor([t]))[:, :3]


class LDM(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t):
        return self.net.model(x, torch.tensor([t]))

    def decode(self, z):
        if hasattr(self.net, "decode_first_stage"):
            return self.net.decode_first_stage(z)
        else:
            raise NotImplementedError

    def differentiable_decode(self, z):
        if hasattr(self.net, "differentiable_decode_first_stage"):
            return self.net.differentiable_decode_first_stage(z)
        else:
            raise NotImplementedError


class EpsilonNetGM(torch.nn.Module):

    def __init__(self, means, weights, alphas_cumprod, cov=None):
        super().__init__()
        self.means = means
        self.weights = weights
        self.covs = cov
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x, t):
        # if len(t) == 1 or t.dim() == 0:
        #     acp_t = self.alphas_cumprod[t.to(int)]
        # else:
        #     acp_t = self.alphas_cumprod[t.to(int)][0]
        acp_t = self.alphas_cumprod[t.to(int)]
        grad_logprob = grad(
            lambda x: fwd_mixture(
                self.means, self.weights, self.alphas_cumprod, t, self.covs
            )
            .log_prob(x)
            .sum()
        )
        return -((1 - acp_t) ** 0.5) * grad_logprob(x)


class EpsilonNetMCGD(torch.nn.Module):

    def __init__(self, H_funcs, unet, dim):
        super().__init__()
        self.unet = unet
        self.H_funcs = H_funcs
        self.dim = dim

    def forward(self, x, t):
        x_normal_basis = self.H_funcs.V(x).reshape(-1, *self.dim)
        # .repeat(x.shape[0]).to(x.device)
        t_emb = torch.tensor(t).to(x.device)
        eps = self.unet(x_normal_basis, t_emb)
        eps_svd_basis = self.H_funcs.Vt(eps)
        return eps_svd_basis


class EpsilonNet(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps, target_shape):
        super().__init__()
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps
        self.target_shape = target_shape

    def forward(self, x, t, labels=None):
        try:
            return self.net((x.reshape(-1, *self.target_shape), torch.tensor(t).unsqueeze(0).unsqueeze(0).to(x.device)), labels).reshape(x.shape)
        except RuntimeError:
            return self.net((x.reshape(-1, *self.target_shape), t.unsqueeze(1)), labels).reshape(x.shape)

    def predict_x0(self, x, t, labels=None):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, t, labels)) / (acp_t**0.5)

    def score(self, x, t):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - acp_t) ** 0.5

    def decode(self, z):
        return self.net.decode(z)

    def differentiable_decode(self, z):
        return self.net.differentiable_decode(z)


class EpsilonNetSVD(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        shape = (x.shape[0], 3, int(np.sqrt((x.shape[-1] // 3))), -1)
        print('shape', shape)
        x = self.H_func.V(x.to(self.device)).reshape(shape)
        return self.H_func.Vt(self.net(x, t))


class EpsilonNetSVDTimeSeries(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, target_shape, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps, target_shape)
        self.net = net
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.H_func = H_func
        self.timesteps = timesteps.to(device)
        self.device = device
        self.target_shape = target_shape

    def forward(self, x, t, labels=None):
        x = self.H_func.V(x.to(self.device)).reshape(self.target_shape)
        return self.H_func.Vt(self.net((x, t.unsqueeze(0).unsqueeze(0)), labels))

class EpsilonNetSVDTimeSeries_old(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, shape, device="cuda"):
        super().__init__()
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device
        self.shape = shape

    def predict_x0(self, x, t, labels=None):
        try:
            acp_t = self.alphas_cumprod[t.int()] / self.alphas_cumprod[self.timesteps[0].int()]
            return (x.reshape(self.shape) - (1 - acp_t) ** 0.5 * self.forward(x.reshape(self.shape), t, labels)) / (acp_t**0.5)
        except:
            print('ok')

    def score(self, x, t):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - acp_t) ** 0.5

    def decode(self, z):
        return self.net.decode(z)

    def differentiable_decode(self, z):
        return self.net.differentiable_decode(z)

    def forward(self, x, t, labels=None):
        x = self.H_func.V(x.to(self.device)).reshape(self.shape)
        net_pred = self.net((x, t.unsqueeze(0).unsqueeze(0).to(x.device)), labels)

        return self.H_func.Vt(net_pred).reshape(self.shape)


class EpsilonNetSVDGM(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        x = self.H_func.V(x.to(self.device))
        return self.H_func.Vt(self.net(x, t))


def load_gmm_epsilon_net(prior: Distribution, dim: int, n_steps: int):
    timesteps = torch.linspace(0, 999, n_steps).long()
    alphas_cumprod = torch.linspace(0.9999, 0.98, 1000)
    alphas_cumprod = torch.cumprod(alphas_cumprod, 0).clip(1e-10, 1)
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

    means, covs, weights = (
        prior.component_distribution.mean,
        prior.component_distribution.covariance_matrix,
        prior.mixture_distribution.probs,
    )

    epsilon_net = EpsilonNet(
        net=EpsilonNetGM(means, weights, alphas_cumprod, covs),
        alphas_cumprod=alphas_cumprod,
        timesteps=timesteps,
    )

    return epsilon_net


def bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    """s < t < ell"""
    alpha_cum_s_to_t = epsilon_net.alphas_cumprod[t] / epsilon_net.alphas_cumprod[s]
    alpha_cum_t_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[t]
    alpha_cum_s_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[s]
    std = (
        eta
        * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
        ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)
    try:
        return coeff_xell * x_ell.reshape(x_s.shape) + coeff_xs * x_s, std
    except:
        print('debug')


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return mean + std * torch.randn_like(mean)


def ddim_statistics(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
    labels: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t, labels)
    return bridge_kernel_statistics(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim_step(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
    labels: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t, labels)
    return sample_bridge_kernel(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim(
    initial_noise_sample: torch.Tensor, epsilon_net: EpsilonNet, eta: float = 1.0, labels: torch.Tensor = None,
) -> torch.Tensor:
    """
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    """
    sample = initial_noise_sample
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=t,
            t_prev=t_prev,
            eta=eta,
        )
    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1], labels)

    return epsilon_net.decode(sample) if hasattr(epsilon_net.net, "decode") else sample
    # if hasattr(epsilon_net, "decode"):
    #     return epsilon_net.decode(sample)
    # else:
    #     return sample
