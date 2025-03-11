import os
from dataclasses import dataclass
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
from jaxopt import ProximalGradient
from jaxopt.prox import prox_lasso, prox_ridge, prox_elastic_net
from posterior_samplers.utils import display_time_series
from posterior_samplers.diffusion_utils import EpsilonNet
import jax.numpy as jnp
from posterior_samplers.mgps_bis import vdps_half
from posterior_samplers.mask_generator import get_mask_bm, get_mask_rm, get_mask_mnr
import wfdb
import torch
from scipy.signal import resample, resample_poly
import math
import neurokit2 as nk

from diffusion_prior.dataloaders import dataloader
from diffusion_prior.utils import calc_diffusion_hyperparams, local_directory, print_size  # 
from diffusion_prior.models import construct_model
from posterior_samplers.mgps import load_vdps_sampler
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import resample_poly

def cos_sin_proj(X, w):
    J = int(X.shape[0] // 2)
    return w[..., :J]@X[:J] + w[..., J:]@X[J:]


def least_squares(X, y, w):
    residuals = torch.nansum((cos_sin_proj(X, w)[None] - y)**2, axis=-1) #  / jnp.nansum(y**2, axis=-1)
    return torch.nanmean(residuals)


def least_squares_jax(w, data):
    X, y = data
    residuals = jnp.nansum((cos_sin_proj(X, w)[None] - y)**2, axis=-1) #  / jnp.nansum(y**2, axis=-1)
    return jnp.nanmean(residuals)



def prepare_signal(signal):
    newFs, oldFs = 100, 360
    L = math.ceil(len(signal) * newFs / oldFs)
    normBeat = list(reversed(signal)) + list(signal) + list(reversed(signal))

    # resample beat by beat and saving it
    res = resample_poly(normBeat, newFs, oldFs)
    res = res[L - 1:2 * L - 1]
    return res


def get_12leads(X_test):
    X_test[:, 2] = X_test[:, 1] - X_test[:, 0]
    aVR = -(X_test[:, 0] + X_test[:, 1]) / 2
    aVL = (X_test[:, 0] - X_test[:, 2]) / 2
    aVF = (X_test[:, 1] + X_test[:, 2]) / 2
    augm_leads = np.stack([aVR, aVL, aVF], axis=1)
    X_test = np.concatenate([X_test[:, :3], augm_leads, X_test[:, 3:]], axis=1)
    return X_test


def get_12leads_mask(X_test):
    aVR = np.logical_or(X_test[:, 0], X_test[:, 1])
    aVL = np.logical_or(X_test[:, 0], X_test[:, 2])
    aVF = np.logical_or(X_test[:, 1], X_test[:, 2])
    augm_leads = np.stack([aVR, aVL, aVF], axis=1)
    X_test = np.concatenate([X_test[:, :3], augm_leads, X_test[:, 3:]], axis=1)
    return X_test


def plot_12leads(ecg, ecg_pred, noisy_ecg, mask, plot_T, offset_T,
                 ecg_color='darkorange', pred_color='darkblue', noise_color='darkred',
                 ecg_pred25=None, ecg_pred75=None, ticks=True, fs=11, title='', top_=1):
    max_H = 2.1   # 3.1  # cm = mV
    offset_H = 1  # 1.5
    margin_H = 0.
    margin_W = 0.2

    # T = ecg.shape[1]*4*2.5*0.001  # (*4 ms * 25 mm)
    T = plot_T*10*2.5*0.001  # (*10 ms * 25 mm)
    times = np.linspace(0, T, plot_T)

    H = max_H*4 + margin_H*5  # in cm
    W = T*4 + margin_W*5  # in cm
    # W = T*3 + margin_W*4

    fig, ax = plt.subplots(figsize=(W/2.54, H/2.54))  # 1/2.54  # centimeters in inches
    if len(title) == 0:
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    else:
        fig.suptitle(title, fontsize=fs)
        fig.subplots_adjust(top=top_, bottom=0, left=0, right=1)

    # fpath = Path(mpl.get_data_path(), '/mnt/data/lisa/times.ttf')
    # Couleurs des grilles
    color_major, color_minor, color_line = (1, 0, 0), (1, 0.7, 0.7), (0, 0, 0.7)

    # Configuration des ticks majeurs et mineurs
    if ticks:
        ax.set_xticks(np.arange(0, W/2.54, 1/2.54))  # Ticks majeurs vertical
        ax.set_yticks(np.arange(0, H/2.54, 1/2.54))  # Ticks majeurs horizontal
        ax.minorticks_on()  # Activer les ticks mineurs
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 5 ticks mineurs par tick majeur
        ax.grid(which='major', linestyle='-', linewidth=0.4, color='k', alpha=0.3)
        ax.grid(which='minor', linestyle='-', linewidth=0.4, color='k', alpha=0.1)



    ax.set_xticklabels([])  # Supprimer les labels des ticks sur l'axe X
    ax.set_yticklabels([])  # Supprimer les labels des ticks sur l'axe Y
    ax.set_ylim(0, H/2.54)
    ax.set_xlim(0, W/2.54)
    ind_W = [1]*3 + [2]*3 + [3]*3 + [4]*3
    # ind_W = [1]*3 + [2]*3 + [3]*3
    ind_H = np.concatenate([np.arange(4, 1, -1)]*4)
    # ind_H = np.concatenate([np.arange(3, 0, -1)]*3)

    for mask_l, lW, lH in zip(mask[:, offset_T:offset_T+plot_T], ind_W, ind_H):
        ax.fill_between(((lW-1)*T+lW*margin_W + times)[mask_l] / 2.54,
                        (max_H*(lH-1)+margin_W/2) / 2.54,
                        (lH*max_H-margin_W/2)/2.54,
                        alpha=0.2, color='gray', rasterized=True)
    for ecg_l, lW, lH in zip(noisy_ecg[:, offset_T:offset_T+plot_T], ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c=noise_color, alpha=1., lw=1., ls='-', #'-.'
                rasterized=True)
    for ecg_l, lW, lH in zip(ecg[:, offset_T:offset_T+plot_T], ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c=ecg_color, alpha=1., lw=1.,
                rasterized=True)
    if ecg_pred25 is not None:
        for ecg_q25, ecg_q75, lW, lH in zip(ecg_pred25[:, offset_T:offset_T+plot_T],
                                            ecg_pred75[:, offset_T:offset_T+plot_T],
                                            ind_W, ind_H):
            ax.fill_between(((lW-1)*T+lW*margin_W + times) / 2.54,
                            (ecg_q25 + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                            (ecg_q75 + offset_H + max_H * (lH - 1) + lH * margin_H) / 2.54,
                    color='green', alpha=.4, lw=0,
                    rasterized=True)

    lead_names = ['I', 'II', 'III'] + ['aVR', 'aVF', 'aVL'] + [f'V{k}' for k in range(7)]
    for ecg_l, lW, lH, ann in zip(ecg_pred[:, offset_T:offset_T+plot_T], ind_W, ind_H, lead_names):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c=pred_color, alpha=1, lw=1.,
                rasterized=True)
        ax.annotate(ann, (((lW-1)*T+margin_W*lW) / 2.54, (1.5*offset_H + max_H*(lH-1)) / 2.54), fontsize=fs)

    # ====== plot lead II =====
    ecgII = ecg[1, offset_T:]
    ecg_predII = ecg_pred[1, offset_T:]
    ecg_noisyII = noisy_ecg[1, offset_T:]
    if ecg_pred25 is not None:
        ecg_pred25II, ecg_pred75II = ecg_pred25[1, offset_T:], ecg_pred75[1, offset_T:]
    T = ecgII.shape[0] * 10 * 2.5 * 0.001  # (*10 ms * 25 mm)
    times = np.linspace(0, T, ecgII.shape[0])
    kept_inds = times < (W - 2*margin_W)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecg_noisyII[kept_inds] + offset_H) / 2.54,
            c=noise_color, alpha=1., lw=1., ls='-',  # '-.'
            rasterized=True)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecgII[kept_inds] + offset_H) / 2.54,
            c=ecg_color, alpha=1., lw=1.,
            rasterized=True)
    if ecg_pred25 is not None:
        ax.fill_between((margin_W + times[kept_inds]) / 2.54, (offset_H + ecg_pred25II[kept_inds]) / 2.54,
                        (offset_H + ecg_pred75II[kept_inds]) / 2.54,
                color='green', alpha=.4, lw=0.,
                rasterized=True)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecg_predII[kept_inds] + offset_H) / 2.54,
            c=pred_color, alpha=1., lw=1.,
            rasterized=True)
    ax.annotate('II', (margin_W * 2 / 2.54, (1.5 * offset_H) / 2.54),
                fontsize=fs)

    return fig


@hydra.main(version_base=None, config_path="sashimi/configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys
    cfg.model.unconditional = True
    torch.manual_seed(cfg.algo.seed)
    torch.cuda.manual_seed(cfg.algo.seed)

    num_gpus = torch.cuda.device_count()
    cfg.dataset.training_class = 'test'
    testloader = dataloader(cfg.dataset, batch_size=cfg.train.batch_size_per_gpu, num_gpus=num_gpus, unconditional=cfg.model.unconditional, shuffle=False)

    local_path, output_directory = local_directory(None, cfg.train.results_path,
                                                   cfg.model, cfg.diffusion,
                                                   cfg.dataset,
                                                   'waveforms')

    # ==== loading checkpoints ==== #
    sashimi_net = construct_model(cfg.model).cuda()
    print_size(sashimi_net)
    sashimi_net.eval()
    ckpt_path = os.path.join(cfg.train.results_path, local_path, 'checkpoint', 'checkpoint.pkl')  # '{}.pkl'.format(ckpt_iter))
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    sashimi_net.load_state_dict(checkpoint['model_state_dict'])
    sashimi_net.requires_grad_(False)
    sashimi_net.eval()
    # sashimi_net = ModelWrapper(sashimi_net)

    #intermed = int(cfg.diffusion.T-20-1)  # 3*cfg.diffusion.T/cfg.algo.n_steps)-1
    #timesteps = torch.linspace(0, intermed, cfg.algo.n_steps).long()
    #timesteps = torch.cat([timesteps, torch.arange(intermed, cfg.diffusion.T).long()])
    timesteps = torch.linspace(0, cfg.diffusion.T-1, cfg.algo.n_steps).long()
    cfg.algo.n_steps = timesteps.shape[0]
    _dh = calc_diffusion_hyperparams(**cfg.diffusion, fast=False)
    T, Alpha, alphas_cumprod, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]).to(alphas_cumprod.device), alphas_cumprod])
    # obs_std = 0.05  # 01 # 5  # cfg.algo.init_std

    epsilon_net = EpsilonNet(sashimi_net, alphas_cumprod, timesteps, (9, 1024))

    # dataset = PhysionetECG(**cfg.dataset)
    # default_grad_steps = cfg.algo.parameters.gradient_steps_fn['conditions'][-1]['return']
    results_path = os.path.join(output_directory, f'{cfg.dataset.name}_denoising')  # _K{cfg.algo.nsteps}_grad{default_grad_steps}')
    os.makedirs(results_path, exist_ok=True)
    all_samples, all_labels, all_x, all_masks = [], [], [], []

    # for x_orig, labels in tqdm(testloader, total=len(testloader)):
    #     x_orig = x_orig[:1]
    #     bs = x_orig.shape[0]
    #     labels = labels.to(torch.float32)[:bs]
    #     break
    lab = 'AF'  # 'RBBB'  # [0, 140, 28, 82, 42]  # 128, 408, 198]  #, 408, 198]
    lab_sample_dic = {'NSR': 0, 'RBBB': 140, 'LBBB': 28, 'SB': 82, 'AF': 42}
    lab_id_dic = {'NSR': -1, 'RBBB':12, 'LBBB': 11, 'SB': 59, 'AF': 4}
    #    plot_ids = [0, 140, 28, 82, 42]  # 128, 408, 198]  #, 408, 198]
    #    # sample_num = [0, 0, 0, 0, 0]  # 6, 5]  #, 4, 1]
    #    plot_labs = ['NSR', 'RBBB', 'LBBB', 'SB', 'AF']
    #    lab_ids = [-1, 12, 11, 59, 4]
    x_orig, labels  = testloader.dataset[lab_sample_dic[lab]]
    x_orig = x_orig.unsqueeze(0)
    labels = labels.unsqueeze(0)

    # ==== corrupt x_orig ====
    noise_type = 'em'
    noise_path = '/mnt/data/lisa/physionet.org/mit-bih-noise-stress-test-database-1.0.0/'+noise_type
    record = wfdb.rdrecord(noise_path).__dict__
    noise = record['p_signal']
    f_s_prev = record['fs']
    new_s = int(round(noise.shape[0] / f_s_prev * 100))
    noise_stress = resample(noise, num=new_s)
    bs = 1

    lead = np.arange(9).astype(int)
    start_ind = np.random.choice(a=len(noise_stress) - cfg.dataset.segment_length,
                             size=(lead.shape[0],), replace=False)
    noise_ecg = np.stack([noise_stress[start_ind[b]:start_ind[b]+cfg.dataset.segment_length, 1].T for b in range(lead.shape[0])], axis=0)
    ecg_max = np.max(x_orig.cpu().numpy(), axis=-1) - np.min(x_orig.cpu().numpy(), axis=-1)
    noise_max_value = np.max(noise_ecg, axis=-1) - np.min(noise_ecg, axis=-1)
    Ase = noise_max_value / ecg_max[:, lead]
    noisy_obs = torch.nan * torch.ones_like(x_orig)
    noise_power = 2.
    noisy_obs[:, lead] = x_orig[0, lead] + torch.tensor(noise_power*noise_ecg[None, lead] / Ase[:, lead, None]).to(torch.float32).to(x_orig.device)
    # noisy_obs[:, lead] = x_orig[:, lead] + torch.tensor(noise_ecg[:, 0]).to(x_orig.device)

    display_time_series(noisy_obs[0], gt=x_orig[0])
    ecg, ecg_noisy = x_orig[0, :, :1000].T.numpy(), noisy_obs[0, :, :1000].cpu().T.numpy()
    mask = np.zeros(ecg.shape).astype(bool)
    mask = get_12leads_mask(mask)
    ecg, ecg_pred, ecg_noisy = get_12leads(ecg), get_12leads(ecg), get_12leads(ecg_noisy)
    fig = plot_12leads(ecg.T, ecg_pred.T, ecg_noisy.T,
                 mask.T, offset_T=150, plot_T=200,
                       ecg_color='green', pred_color='blue', noise_color='red',
                       ecg_pred25=None, # np.percentile(samples12, 5, axis=0),
                       ecg_pred75=None, # np.percentile(samples12, 95, axis=0),
                       ticks=False, fs=16,
                       title=f'{lab}', top_=0.9)
    plt.show()
    #  === noise model === #
    # f_c = 0.7 # BW
    f_s = 100 # sampling frequency
    # 2 * jnp.pi ??

    all_RR = []
    for k in range(9):
        signals, info = nk.ecg_process(noisy_obs[0, 4, :1000], sampling_rate=100)
        RR = (info['ECG_R_Peaks'][1:]-info['ECG_R_Peaks'][:-1]).mean()
        all_RR.append(RR)
    RR = np.median(all_RR)
    cfg.denoising.f_c_max = 50  # max(20, int(round(100-10/11*round(RR))))
    print('f_c_max', cfg.denoising.f_c_max)# 20  #40  # 10
    J = 512 # int(cfg.denoising.f_c_max*10) # 0 # int(cfg.denoising.f_c_max / 10 * 10000)  # 20000 # : which parameter ? 1000 cfg.denoising.J # number of harmonics
    phi = np.concatenate([np.cos(np.arange(1024)[:, None] / f_s * (np.arange(J)[None] / J *
                              (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                              +cfg.denoising.f_c_min)),
                           np.sin(np.arange(1024)[:, None] / f_s * np.arange(J)[None] / J * (
                                   (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                                   + cfg.denoising.f_c_min)
                                   )],
                          axis=1).T
    f_c = (cfg.denoising.f_c_min + cfg.denoising.f_c_max) / 2.

    eta_param = np.zeros((bs, 9, phi.shape[0]))
    mask_phi = eta_param == 0
    #obs_std = 2.*torch.tensor(np.std(x_orig[0].cpu().numpy(), axis=1)[None, :, None]).cuda()  # .5 # we start with a large variance to allow to generate a signal far away from the observation
    obs_std = torch.tensor(np.std(noisy_obs[0].cpu().numpy(), axis=1)[None, :, None]).cuda()  # .5 # we start with a large variance to allow to generate a signal far away from the observation

    cfg.algo.nsamples = 20
    reg = 1  # 1.  # we start with small regularization to allow to remove a lot of noise even if we end-up with a flat signal
    solver = prox_lasso
    print(obs_std)
    for k in range(6):
        if k == 5:
            obs = samples # .mean(dim=0).unsqueeze(0)
        else:
            obs = (noisy_obs-cos_sin_proj(phi, eta_param)).cuda()
        # obs_std = new_obs_std
        def log_pot(x):
            b = obs.shape[0]
            diff = (obs - x) / obs_std  # obs.reshape(b, -1) - x.reshape(cfg.algo.nsamples, -1)
            diff = diff[~torch.isnan(diff)]
            return -0.5 * torch.norm(diff) ** 2 # / obs_std ** 2

        shape = obs.shape[1:]  # (9, 1024)

        initial_noise = torch.randn(cfg.algo.nsamples, *shape).cuda()

        vdps_fn = load_vdps_sampler(cfg.algo)
        samples = vdps_fn(
            initial_noise=initial_noise,
            labels=labels,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            display_freq=300,  # 10,
            display_fn=display_time_series,
            ground_truth=x_orig.cpu()
        )  # 10 minutes for 1 sample 1 observation...
        display_time_series(samples[0], gt=noisy_obs[0])
        display_time_series(samples[0], gt=x_orig[0])

        new_eta_param = []
        phi_estim = jnp.array(phi)
        for l in range(samples.shape[1]):
            y = jnp.array(noisy_obs[:, l] - samples[:, l].detach().cpu())
            # y = noisy_obs[:, l].to(samples.device) - samples[:, l]
            pg = ProximalGradient(fun=least_squares_jax, prox=solver, tol=0.0001, maxiter=5, maxls=100, decrease_factor=0.5)
            # reg = 10.
            eta_l = jnp.array(eta_param[0, l]*mask_phi[0, l])
            pg_sol = pg.run(eta_l, hyperparams_prox=reg, data=(phi_estim*mask_phi[:, l].T, y)).params
            new_eta_param.append(pg_sol)
        new_eta_param = np.stack(new_eta_param, axis=0)
        eta_param, prev_eta = new_eta_param[None], eta_param
        print(eta_param[0])
        display_time_series(noisy_obs[0] - cos_sin_proj(phi, eta_param), gt=x_orig[0])
        #if k > 2:
        #   reg = 5.
        if k >= 0:
            obs_std = np.round(
                np.nanstd(noisy_obs - samples.detach().cpu() - cos_sin_proj(phi, eta_param[0])[None],
                           axis=(0, 2)),
                3)
            #if k == 0:
            #    obs_std /= 2.# EM
            obs_std[np.isnan(obs_std)] = obs_std[~np.isnan(obs_std)].mean()   # if i don't put that, i have NaN's values
            print(obs_std)
            obs_std = torch.tensor(obs_std[None, :, None]).max().cuda()
            #reg = 0.1
        if k == 1:
            print('switching to ridge regression')
            solver = prox_ridge
            harmonics_th = 1e-3
            mask_phi = np.absolute(eta_param) > harmonics_th
            #if k >=5:
            #reg = 5.

    print('ok')
    ecg, ecg_pred, ecg_noisy = x_orig[0, :, :1000].T.numpy(), samples.mean(axis=0).cpu().T.numpy()[
                                                              :1000], noisy_obs[0, :, :1000].cpu().T.numpy()
    mask = np.zeros(ecg.shape).astype(bool)
    #mask[:, lead] = True
    mask = get_12leads_mask(mask)
    ecg, ecg_pred, ecg_noisy = get_12leads(ecg), get_12leads(ecg_pred), get_12leads(ecg_noisy)
    samples12 = get_12leads(samples.cpu().numpy())[..., :1000]
    if noise_type == 'em':
        noise_name = 'Electrode motion'
    else: # "bw"
        noise_name = 'Baseline wander'
    fig = plot_12leads(ecg.T, ecg_pred.T, ecg_noisy.T,
                 mask.T, offset_T=150, plot_T=200,
                       ecg_color='green', pred_color='blue', noise_color='red',
                       ecg_pred25=None, # np.percentile(samples12, 5, axis=0),
                       ecg_pred75=None, # np.percentile(samples12, 95, axis=0),
                       ticks=False, fs=16,
                       title=f'{lab}', top_=0.9)
    plt.show()
    fig.savefig(f'/mnt/data/lisa/papier_figures_final/{noise_type.upper()}_{lab}_good.pdf')
    fig.savefig(f'/mnt/data/lisa/papier_figures_final/{noise_type.upper()}_{lab}_good.png')

    np.savez(os.path.join(results_path, f'{noise_type.upper()}_{lab}_sample.npz'),
             generated=samples.cpu(),
             x_orig=x_orig[0].cpu(),
             mask_phi=mask_phi.cpu(),
             noisy_obs=noisy_obs[0].cpu(),
             ecg=ecg.T, ecg_pred=ecg_pred.T, ecg_noisy=ecg_noisy.T
             )
    print('ok')


if __name__ == '__main__':
    main()

