import os
from dataclasses import dataclass
os.environ['OMP_NUM_THREADS'] = '10'
import sys
sys.path.append('.')
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from posterior_samplers.utils import display_time_series, fix_seed
from posterior_samplers.diffusion_utils import EpsilonNet, EpsilonNetSVDTimeSeries
from posterior_samplers.svd_replacement import InpaintingTimeSeries
from posterior_samplers.cond_sampling_algos import pgdm_svd
from posterior_samplers.ddnm import ddnm, ddnm_plus
from posterior_samplers.mask_generator import get_mask_bm, get_mask_rm, get_mask_mnr

import torch
from torch.func import vmap
from plot_scripts.plot_imputation import plot_12leads
from diffusion_prior.dataloaders import dataloader
from diffusion_prior.utils import calc_diffusion_hyperparams, local_directory, print_size  # 
from diffusion_prior.models import construct_model
import matplotlib.pyplot as plt
import jax.numpy as jnp
import math
import wfdb
from scipy.signal import resample
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

    fix_seed(cfg.algo.seed)
    print(f'Fixed seed {cfg.algo.seed}')

    num_gpus = torch.cuda.device_count()
    cfg.dataset.training_class = 'test'
    cfg.train.batch_size_per_gpu = 1
    #cfg.dataset.label_names = ['AF']
    testloader = dataloader(cfg.dataset,
                            batch_size=cfg.train.batch_size_per_gpu,
                            num_gpus=num_gpus,
                            unconditional=cfg.model.unconditional,
                            shuffle=False)

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

    timesteps = torch.linspace(0, cfg.diffusion.T-1, cfg.algo.nsteps).long()
    _dh = calc_diffusion_hyperparams(**cfg.diffusion, fast=False)
    T, Alpha, alphas_cumprod, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]).to(alphas_cumprod.device), alphas_cumprod])
    obs_std = cfg.algo.init_std

    results_path = os.path.join(
        output_directory,
        f'{cfg.dataset.name}_denoising_{cfg.denoising.noise_type}_step{cfg.denoising.EM_step}_lasso{cfg.denoising.alpha_lasso}_ridge{cfg.denoising.alpha_ridge}')  # _K{cfg.algo.nsteps}_grad{default_grad_steps}')
    os.makedirs(results_path, exist_ok=True)
    all_samples, all_labels, all_x, all_noisy = [], [], [], []
    for x_orig, labels in tqdm(testloader, total=len(testloader)):
        x_orig = x_orig.cuda()  # [:2]
        labels = labels.cuda().to(torch.float32) # [:2]
        cfg.dataset.segment_length = x_orig.shape[2]

        # ======== create the Fourier parameters ========= #
        # Paramètres
        n_channels = 9
        segment_length = 256
        n_frequencies = int(segment_length/2)  # Nombre de fréquences à utiliser (jusqu'à la fréquence de Nyquist)
        sampling_rate = 100  # Fréquence d'échantillonnage en Hz
        overlap = 128
        # Calcul de la base de Fourier pour les segments
        t_four = np.linspace(0, (segment_length - 1) / sampling_rate, segment_length)
        frequencies = np.linspace(0, sampling_rate / 2, n_frequencies)
        fourier_basis = np.zeros((segment_length, n_frequencies * 2))

        for i, freq in enumerate(frequencies):
            fourier_basis[:, i] = np.cos(2 * np.pi * freq * t_four)
            fourier_basis[:, i + n_frequencies] = np.sin(2 * np.pi * freq * t_four)

        # Découpage de l'ECG en segments
        def segment_ecg(ecg):
            segments = []
            start = 0
            while start < ecg.shape[2]:
                end = min(start + segment_length, ecg.shape[2])
                segments.append(ecg[:, :, start:end])
                start += segment_length - overlap
            return segments

        # t_four = np.linspace(0, (temporal_length - 1) / sampling_rate, temporal_length)
        # frequencies = np.linspace(0, sampling_rate / 2, n_frequencies)
        # fourier_basis = np.zeros((temporal_length, n_frequencies * 2))
        #
        # for i, freq in enumerate(frequencies):
        #     fourier_basis[:, i] = np.cos(2 * np.pi * freq * t_four)
        #     fourier_basis[:, i + n_frequencies] = np.sin(2 * np.pi * freq * t_four)

        # ======== getting the observation =========== #
        # if cfg.algo.missingness_type == 'rm':
        #     mask = np.stack([get_mask_rm(x_orig[0].T, k=int(cfg.dataset.segment_length*cfg.algo.missingness/100)).T.to(bool) for _ in range(len(x_orig))])
        # if cfg.algo.missingness_type == 'mnr':
        #     mask = np.stack([get_mask_mnr(x_orig[0].T, k=int(cfg.dataset.segment_length * cfg.algo.missingness / 100)).T.to(bool) for _ in range(len(x_orig))])
        # if cfg.algo.missingness_type == 'bm':
        #     mask = np.stack([get_mask_bm(x_orig[0].T, k=int(cfg.dataset.segment_length * cfg.algo.missingness / 100)).T.to(bool) for _ in range(len(x_orig))])
        # if cfg.algo.missingness_type == 'lead':
        #     mask = np.zeros(x_orig.shape)
        #     mask[:, :cfg.algo.missingness] = 1
        #     mask = mask.astype(bool)
        # obs = torch.ones_like(x_orig)*torch.nan  # [mask].view(x_orig.shape[0], -1)
        # obs[mask] = x_orig[mask]
        mask = np.zeros(x_orig.shape).astype(bool)

        # ==== using H function ==== #
        leads, time_points = np.meshgrid(
            np.arange(9),
            np.arange(cfg.dataset.segment_length),
        )
        #missing_indices = (leads[~mask[0].T] * cfg.dataset.segment_length + time_points[~mask[0].T]).flatten()
        missing_indices = torch.Tensor([]).to(torch.long)

        H_func = InpaintingTimeSeries(9, cfg.dataset.segment_length, missing_indices, torch.device('cuda'))
        cfg.algo.nsamples = 10  # obs.shape[0]

        # ==== corrupt x_orig ====
        noise_type = cfg.denoising.noise_type # 'em'
        noise_path = '/mnt/data/lisa/physionet.org/mit-bih-noise-stress-test-database-1.0.0/' + noise_type
        record = wfdb.rdrecord(noise_path).__dict__
        noise = record['p_signal']
        f_s_prev = record['fs']
        new_s = int(round(noise.shape[0] / f_s_prev * 100))
        noise_stress = resample(noise, num=new_s)
        bs = 1

        lead = np.arange(9).astype(int)
        start_ind = np.random.choice(a=len(noise_stress) - cfg.dataset.segment_length,
                                     size=(lead.shape[0],), replace=False)
        noise_ecg = np.stack(
            [noise_stress[start_ind[b]:start_ind[b] + cfg.dataset.segment_length, 1].T for b in range(lead.shape[0])],
            axis=0)
        ecg_max = np.max(x_orig.cpu().numpy(), axis=-1) - np.min(x_orig.cpu().numpy(), axis=-1)
        noise_max_value = np.max(noise_ecg, axis=-1) - np.min(noise_ecg, axis=-1)
        Ase = noise_max_value / ecg_max[:, lead]
        noisy_obs = torch.nan * torch.ones_like(x_orig)
        noise_power = 2.
        noisy_obs[:, lead] = x_orig[0, lead] + torch.tensor(
            noise_power * noise_ecg[None, lead] / Ase[:, lead, None]).to(torch.float32).to(x_orig.device)
        ecg, ecg_noisy = x_orig[0, :, :1000].cpu().T.numpy(), noisy_obs[0, :, :1000].cpu().T.numpy()
        mask12 = get_12leads_mask(mask)
        ecg, ecg_pred, ecg_noisy = get_12leads(ecg), get_12leads(ecg), get_12leads(ecg_noisy)
        # fig = plot_12leads(ecg.T, ecg_pred.T, ecg_noisy.T,
        #                    mask12[0], offset_T=150, plot_T=200,
        #                    ecg_color='green', pred_color='blue', noise_color='red',
        #                    ecg_pred25=None,  # np.percentile(samples12, 5, axis=0),
        #                    ecg_pred75=None,  # np.percentile(samples12, 95, axis=0),
        #                    ticks=False, fs=16,
        #                    title=f'{cfg.dataset.label_names[0]}',
        #                    top_=0.9)
        # plt.show()

        shape = x_orig.shape[1:]  # (9, 1024)
        initial_noise = torch.randn((cfg.algo.nsamples, *shape)).cuda()
        epsilon_net = EpsilonNet(sashimi_net, alphas_cumprod, timesteps, x_orig.shape[1:])
        obs = H_func.H(noisy_obs[0:1]).reshape(1, -1)
        obs_std = np.std(noisy_obs[0].cpu().numpy(), axis=1).mean()  #/2
        for EM_step in range(cfg.denoising.EM_step):
            obs += torch.randn_like(obs)*obs_std
            if EM_step == 0:
                samples = torch.zeros_like(initial_noise)
            else:
                samples = ddnm_plus(
                    initial_noise,
                    labels,
                    epsilon_net,
                    'inpainting',
                    H_func,
                    obs,
                    sigma_y=obs_std,
                )  # 10 x 9 x 1024
            ecg_pred = get_12leads(samples.mean(dim=0).T[:1000].cpu())
            fig = plot_12leads(ecg.T, ecg_pred.T, ecg_noisy.T,
                               mask12[0], offset_T=150, plot_T=200,
                               ecg_color='green', pred_color='blue', noise_color='red',
                               ecg_pred25=None,  # np.percentile(samples12, 5, axis=0),
                               ecg_pred75=None,  # np.percentile(samples12, 95, axis=0),
                               ticks=False, fs=16,
                               title=f'{cfg.dataset.label_names[0]}',
                               top_=0.9)
            plt.show()

            # ===== Linear regression of fourier coeffs ==== #
            # Initialiser les coefficients de Fourier
            # fourier_coefficients = np.zeros((n_channels, n_frequencies * 2))
            # mean_diff = -samples.cpu().numpy().mean(axis=0)+noisy_obs[0].cpu().numpy()
            # # Utiliser Ridge de scikit-learn pour chaque canal
            # for channel in range(n_channels):
            #     if EM_step == 0:
            #         ridge = Lasso(alpha=cfg.denoising.alpha_lasso) # 0.01)
            #     else:
            #         ridge = Ridge(alpha=cfg.denoising.alpha_ridge)  #, l1_ratio=0.001)
            #     ridge.fit(fourier_basis, mean_diff[channel])
            #     fourier_coefficients[channel, :] = ridge.coef_
            # cleaning_ECG = noisy_obs.cpu().numpy()[0]-fourier_coefficients@fourier_basis.T
            # Découpage de samples et obs en segments
            sample_segments = segment_ecg(samples)
            obs_segments = segment_ecg(noisy_obs)

            # Régression Ridge locale sur chaque segment
            cleaned_segments = []
            for sample_segment, obs_segment in zip(sample_segments, obs_segments):
                fourier_coefficients = np.zeros((n_channels, n_frequencies * 2))
                seg_L = obs_segment.shape[-1]
                mean_diff = -sample_segment.cpu().numpy().mean(axis=0) + obs_segment[0].cpu().numpy()
                for channel in range(n_channels):
                    if EM_step == 0:
                        ridge = Lasso(alpha=cfg.denoising.alpha_lasso)
                    else:
                        ridge = Ridge(alpha=0.1)  # cfg.denoising.alpha_ridge)
                    ridge.fit(fourier_basis[:seg_L], mean_diff[channel])
                    fourier_coefficients[channel, :] = ridge.coef_
                cleaning_ECG = obs_segment.cpu().numpy()[0] - fourier_coefficients @ fourier_basis[:seg_L].T
                cleaned_segments.append(cleaning_ECG)
            # Reassemblage des segments
            cleaning_ECG_all = np.zeros_like(x_orig[0].cpu().numpy())
            count = np.zeros_like(x_orig[0].cpu().numpy())
            start = 0
            for i, segment in enumerate(cleaned_segments):
                end = min(start + segment_length, x_orig.shape[2])
                cleaning_ECG_all[:, start:end] += segment[:, :end - start]
                count[:, start:end] += 1
                start += segment_length - overlap
            cleaning_ECG = cleaning_ECG_all / count
            # Conversion en tensor et retour sur le GPU
            cleaning_ECG12 = get_12leads(cleaning_ECG.T[:1000])

            fig = plot_12leads(ecg.T, cleaning_ECG12.T, ecg_noisy.T,
                               mask12[0], offset_T=150, plot_T=200,
                               ecg_color='green', pred_color='blue', noise_color='red',
                               ecg_pred25=None,  # np.percentile(samples12, 5, axis=0),
                               ecg_pred75=None,  # np.percentile(samples12, 95, axis=0),
                               ticks=False, fs=16,
                               title=f'{cfg.dataset.label_names[0]}',
                               top_=0.9)
            plt.show()
            obs = H_func.H(torch.from_numpy(cleaning_ECG).cuda().unsqueeze(0)).reshape(1, -1)
            obs_std = np.std(cleaning_ECG, axis=1).mean()#/2.  # .5 # we start with a large variance to allow to generate a signal far away from the observation

        all_samples.append(samples.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_x.append(x_orig.detach().cpu())
        all_noisy.append(noisy_obs.detach().cpu())
        lab_suffix = '_'.join(cfg.dataset.label_names)
        np.savez(os.path.join(results_path, f'Denoising_{lab_suffix}_seed{cfg.algo.seed}.npz'),
                 generated=np.stack(all_samples),
                 label=np.concatenate(all_labels),
                 real=np.concatenate(all_x),
                 noisy=np.concatenate(all_noisy),
                 #mask=np.concatenate(all_masks)
                 )
        print('ok')


if __name__ == '__main__':
    main()

