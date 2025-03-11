from posterior_samplers.diffusion_utils import ddim_step, EpsilonNetMCGD
#from ddrm.functions.denoising import efficient_generalized_steps

import torch
from tqdm import tqdm

import numpy as np



class EpsilonNetDDRM(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, labels=None):
        t = t.to(int)
        return self.unet((x, t), labels)


def ddrm(initial_noise, unet, labels, inverse_problem, timesteps, alphas_cumprod, device):
    obs, A, std = inverse_problem
    ddrm_timesteps = timesteps.clone()
    ddrm_timesteps[-1] = ddrm_timesteps[-1] - 1
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    ddrm_samples = efficient_generalized_steps(
        x=initial_noise,
        b=betas,
        seq=ddrm_timesteps.cpu(),
        model=EpsilonNetDDRM(unet=unet),
        y_0=obs[None, ...].to(device),
        H_funcs=A,
        sigma_0=std,
        etaB=1.0,
        etaA=0.85,
        etaC=1.0,
        device=device,
        classes=labels,
        cls_fn=None,
    )
    return ddrm_samples[0][-1]


def pgdm_svd(initial_noise, labels, epsilon_net, obs, H_func, std_obs, eta=1.0):
    """
    obs = D^{-1} U^T y
    """
    Ut_y, diag = H_func.Ut(obs), H_func.singulars()

    def pot_fn(x, t):
        rsq_t = 1 - epsilon_net.alphas_cumprod[t]
        diag_cov = diag**2 + (std_obs**2 / rsq_t)
        return (
            -0.5
            * torch.norm((Ut_y - diag * x[:, : diag.shape[0]]) / diag_cov.sqrt()) ** 2.0
        )

    sample = initial_noise.reshape(initial_noise.shape[0], -1)
    for i in tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = sample.requires_grad_()
        xhat_0 = epsilon_net.predict_x0(sample, t, labels)
        # acp_t, acp_tprev = (
        #     torch.tensor([epsilon_net.alphas_cumprod[t]]),
        #     torch.tensor([epsilon_net.alphas_cumprod[t_prev]]),
        # )
        acp_t, acp_tprev = epsilon_net.alphas_cumprod[t], epsilon_net.alphas_cumprod[t_prev]
        # grad_pot = grad_pot_fn(sample, t)
        grad_pot = pot_fn(xhat_0, t)
        grad_pot = torch.autograd.grad(grad_pot, sample)[0]
        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=xhat_0
        ).detach()
        sample += acp_tprev.sqrt() * acp_t.sqrt() * grad_pot

    return (
        H_func.V(epsilon_net.predict_x0(sample, epsilon_net.timesteps[1], labels))
        .reshape(initial_noise.shape)
        .detach()
    )


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def efficient_generalized_steps_w_grad(
    x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None
):
    # setup vectors used in the algorithm
    singulars = H_funcs.singulars()
    Sigma = torch.zeros(np.prod(x.shape[1:]), device=x.device)
    Sigma[: singulars.shape[0]] = singulars
    U_t_y = H_funcs.Ut(y_0)
    Sig_inv_U_t_y = U_t_y / singulars[: U_t_y.shape[-1]]

    # initialize x_T as given in the paper
    largest_alphas = compute_alpha(
        b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long()
    )
    largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
    large_singulars_index = torch.where(
        singulars * largest_sigmas[0, 0, 0, 0] > sigma_0
    )
    inv_singulars_and_zero = torch.zeros(np.prod(x.shape[1:])).to(
        singulars.device
    )
    inv_singulars_and_zero[large_singulars_index] = (
        sigma_0 / singulars[large_singulars_index]
    )
    inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

    # implement p(x_T | x_0, y) as given in the paper
    # if eigenvalue is too small, we just treat it as zero (only for init)
    init_y = torch.zeros(x.shape[0], np.prod(x.shape[1:])).to(x.device)
    init_y[:, large_singulars_index[0]] = U_t_y[
        :, large_singulars_index[0]
    ] / singulars[large_singulars_index].view(1, -1)
    init_y = init_y.view(*x.size())
    remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero**2
    remaining_s = (
        remaining_s.view(x.shape)  # x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        .clamp_min(0.0)
        .sqrt()
    )
    init_y = init_y + remaining_s * x
    init_y = init_y / largest_sigmas

    # setup iteration variables
    x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        # xt = xs[-1].to('cpu')
        xt = xs[-1]
        if cls_fn == None:
            if classes is not None:
                et = model(xt, t, classes)
            else:
                et = model(xt, t)
        else:
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # variational inference conditioned on y
        sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
        sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
        xt_mod = xt / at.sqrt()[0, 0, 0, 0]
        V_t_x = H_funcs.Vt(xt_mod)
        SVt_x = (V_t_x * Sigma)[:, : U_t_y.shape[1]]
        V_t_x0 = H_funcs.Vt(x0_t)
        SVt_x0 = (V_t_x0 * Sigma)[:, : U_t_y.shape[1]]

        falses = torch.zeros(
            V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device
        )
        cond_before_lite = singulars * sigma_next > sigma_0
        cond_after_lite = singulars * sigma_next < sigma_0
        cond_before = torch.hstack((cond_before_lite, falses))
        cond_after = torch.hstack((cond_after_lite, falses))

        std_nextC = sigma_next * etaC
        sigma_tilde_nextC = torch.sqrt(sigma_next**2 - std_nextC**2)

        std_nextA = sigma_next * etaA
        sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)

        diff_sigma_t_nextB = torch.sqrt(
            sigma_next**2 - sigma_0**2 / singulars[cond_before_lite] ** 2 * (etaB**2)
        )

        # missing pixels
        Vt_xt_mod_next = (
            V_t_x0
            + sigma_tilde_nextC * H_funcs.Vt(et)
            + std_nextC * torch.randn_like(V_t_x0)
        )

        # less noisy than y (after)
        Vt_xt_mod_next[:, cond_after] = (
            V_t_x0[:, cond_after]
            + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite]
            + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
        )

        # noisier than y (before)
        Vt_xt_mod_next[:, cond_before] = (
            Sig_inv_U_t_y[:, cond_before_lite] * etaB
            + (1 - etaB) * V_t_x0[:, cond_before]
            + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite]
        )

        # aggregate all 3 cases and give next prediction
        xt_mod_next = H_funcs.V(Vt_xt_mod_next)
        xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

        x0_preds.append(x0_t)
        xs.append(xt_next)

    return xs, x0_preds


def efficient_generalized_steps(
    x,
    seq,
    model,
    b,
    H_funcs,
    y_0,
    sigma_0,
    etaB,
    etaA,
    etaC,
    device,
    cls_fn=None,
    classes=None,
):
    with torch.no_grad():
        # setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(np.prod(x.shape[1:]), device=x.device)
        Sigma[: singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[: U_t_y.shape[-1]]

        # initialize x_T as given in the paper
        largest_alphas = compute_alpha(
            b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long()
        )
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(
            singulars * largest_sigmas[0, 0, 0, 0] > sigma_0
        )
        inv_singulars_and_zero = torch.zeros(np.prod(x.shape[1:])).to(
            singulars.device
        )
        inv_singulars_and_zero[large_singulars_index] = (
            sigma_0 / singulars[large_singulars_index]
        )
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init)
        init_y = torch.zeros(x.shape[0], np.prod(x.shape[1:])).to(
            x.device
        )
        init_y[:, large_singulars_index[0]] = U_t_y[
            :, large_singulars_index[0]
        ] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero**2
        remaining_s = (
            remaining_s.view(x.shape)
            .clamp_min(0.0)
            .sqrt()
        )
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas

        # setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        # iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)[0]
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # variational inference conditioned on y
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, : U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, : U_t_y.shape[1]]

            falses = torch.zeros(
                V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device
            )
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next**2 - std_nextC**2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)

            diff_sigma_t_nextB = torch.sqrt(
                sigma_next**2
                - sigma_0**2 / singulars[cond_before_lite] ** 2 * (etaB**2)
            )

            # missing pixels
            Vt_xt_mod_next = (
                V_t_x0
                + sigma_tilde_nextC * H_funcs.Vt(et)
                + std_nextC * torch.randn_like(V_t_x0)
            )

            # less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = (
                V_t_x0[:, cond_after]
                + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite]
                + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            )

            # noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = (
                Sig_inv_U_t_y[:, cond_before_lite] * etaB
                + (1 - etaB) * V_t_x0[:, cond_before]
                + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite]
            )

            # aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t)
            xs.append(xt_next)

    return xs, x0_preds