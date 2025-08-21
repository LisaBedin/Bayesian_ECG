import os
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm

from posterior_samplers.utils import display_time_series
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.mgps_bis import mgps_half
from posterior_samplers.mask_generator import get_mask_bm, get_mask_rm, get_mask_mnr

import torch

from diffusion_prior.dataloaders import dataloader

from diffusion_prior.utils import calc_diffusion_hyperparams, local_directory, print_size  # 
from diffusion_prior.models import construct_model
from posterior_samplers.mgps import load_vdps_sampler
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def plot_12leads(ecg, ecg_pred, ecg_pred25, ecg_pred75, mask, plot_T, offset_T):
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
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

    # fpath = Path(mpl.get_data_path(), '/mnt/data/lisa/times.ttf')
    # Couleurs des grilles
    color_major, color_minor, color_line = (1, 0, 0), (1, 0.7, 0.7), (0, 0, 0.7)

    # Configuration des ticks majeurs et mineurs
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

    for ecg_q25, ecg_q75, lW, lH in zip(ecg_pred25[:, offset_T:offset_T+plot_T],
                                        ecg_pred75[:, offset_T:offset_T+plot_T],
                                        ind_W, ind_H):
        ax.fill_between(((lW-1)*T+lW*margin_W + times) / 2.54,
                        (ecg_q25 + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                        (ecg_q75 + offset_H + max_H * (lH - 1) + lH * margin_H) / 2.54,
                color='green', alpha=.4, lw=0,
                rasterized=True)

    for ecg_l, lW, lH in zip(ecg[:, offset_T:offset_T+plot_T], ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='royalblue', alpha=1., lw=1.,
                rasterized=True)

    lead_names = ['I', 'II', 'III'] + ['aVR', 'aVF', 'aVL'] + [f'V{k}' for k in range(7)]
    for ecg_l, lW, lH, ann in zip(ecg_pred[:, offset_T:offset_T+plot_T], ind_W, ind_H, lead_names):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='darkorange', alpha=1, lw=1.,
                rasterized=True)
        ax.annotate(ann, (((lW-1)*T+lW*margin_W*2) / 2.54, (1.5*offset_H + max_H*(lH-1)) / 2.54), fontsize=11)

    # ====== plot lead II =====
    ecgII = ecg[1]
    ecg_predII = ecg_pred[1]
    T = ecgII.shape[0] * 10 * 2.5 * 0.001  # (*10 ms * 25 mm)
    times = np.linspace(0, T, ecgII.shape[0])
    kept_inds = times < (W - 2*margin_W)
    ax.fill_between((margin_W + times[kept_inds]) / 2.54, (offset_H + ecg_pred25[1, kept_inds]) / 2.54,
                    (offset_H - ecg_pred25[1, kept_inds]) / 2.54,
            color='green', alpha=.4, lw=0.,
            rasterized=True)

    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecgII[kept_inds] + offset_H) / 2.54,
            c='royalblue', alpha=1., lw=1.,
            rasterized=True)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecg_predII[kept_inds] + offset_H) / 2.54,
            c='darkorange', alpha=1., lw=1.,
            rasterized=True)
    ax.annotate('II', (margin_W * 2 / 2.54, (1.5 * offset_H) / 2.54),
                fontsize=11)

    return fig


def get_12leads(X_test):
    X_test[:, 2] = X_test[:, 1] - X_test[:, 0]
    aVR = -(X_test[:, 0] + X_test[:, 1]) / 2
    aVL = (X_test[:, 0] - X_test[:, 2]) / 2
    aVF = (X_test[:, 1] + X_test[:, 2]) / 2
    augm_leads = np.stack([aVR, aVL, aVF], axis=1)
    X_test = np.concatenate([X_test[:, :3], augm_leads, X_test[:, 3:]], axis=1)
    return X_test


@hydra.main(version_base=None, config_path="diffusion_prior/configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    torch.manual_seed(cfg.algo.seed)
    torch.cuda.manual_seed(cfg.algo.seed)

    num_gpus = torch.cuda.device_count()
    cfg.dataset.training_class = 'test'
    cfg.train.batch_size_per_gpu = 1
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
    obs_std = cfg.algo.init_std
    # dataset = PhysionetECG(**cfg.dataset)
    # default_grad_steps = cfg.algo.parameters.gradient_steps_fn['conditions'][-1]['return']
    results_path = os.path.join(output_directory, f'{cfg.dataset.name}_{cfg.algo.missingness_type}{cfg.algo.missingness}')  # _K{cfg.algo.nsteps}_grad{default_grad_steps}')
    os.makedirs(results_path, exist_ok=True)
    all_samples, all_labels, all_x, all_masks = [], [], [], []
    # cfg.algo.missingness_type = 'lead'
    #cfg.algo.missingness = 2
    id_ = 0
    for i, (x_orig, labels) in tqdm(enumerate(testloader), total=len(testloader)):
        # if i >= 1421:
        #    if i == 1421:
        #    print(f'# =========== start at label {i} ============== #')
        # if id_ == 140:
        #     print('RBBB')
        #     break
        id_ += 1
        x_orig = x_orig.cuda()  # [:2]
        labels = labels.cuda().to(torch.float32) # [:2]
        # break

        # ======== getting the observation =========== #
        if cfg.algo.missingness_type == 'rm':
            mask = np.stack([get_mask_rm(x_orig[0].T, k=int(cfg.dataset.segment_length*cfg.algo.missingness/100)).T.to(bool) for _ in tqdm(range(len(x_orig)))])
        if cfg.algo.missingness_type == 'mnr':
            mask = np.stack([get_mask_mnr(x_orig[0].T, k=int(cfg.dataset.segment_length * cfg.algo.missingness / 100)).T.to(bool) for _ in tqdm(range(len(x_orig)))])
        if cfg.algo.missingness_type == 'bm':
            mask = np.stack([get_mask_bm(x_orig[0].T, k=int(cfg.dataset.segment_length * cfg.algo.missingness / 100)).T.to(bool) for _ in tqdm(range(len(x_orig)))])
        if cfg.algo.missingness_type == 'lead':
            mask = np.zeros(x_orig.shape)
            mask[:, :cfg.algo.missingness] = 1
            mask = mask.astype(bool)
        if cfg.algo.missingness_type == 'random_lead':
            mask = np.zeros(x_orig.shape)
            chosen_leads = np.random.choice(9, size=(cfg.algo.missingness,), replace=False)
            mask[:, chosen_leads] = 1
            mask = mask.astype(bool)
        if cfg.algo.missingness_type == 'ML_BL':
            mask = np.zeros(x_orig.shape)
            mask[:, np.array([0, 2]), :256] = 1
            mask[:, np.array([1, 2]), -24-256:] = 1
            mask = mask.astype(bool)
        obs = torch.ones_like(x_orig)*torch.nan  # [mask].view(x_orig.shape[0], -1)
        obs[mask] = x_orig[mask]
        # obs += torch.randn_like(obs)*obs_std
        # obs_img = x_orig[0, :3]

        # display_time_series(obs_img.detach().cpu(), gt=x_orig[0, :3].detach().cpu())
        # cfg.algo.n_samples = 10  # obs.shape[0]
        def log_pot(x):
            b = obs.shape[0]
            diff = obs.reshape(b, -1) - x.reshape(cfg.algo.n_samples, -1)
            diff = diff[~torch.isnan(diff)]
            return -0.5 * torch.norm(diff) ** 2 / obs_std ** 2

        shape = x_orig.shape[1:]  # (9, 1024)
        epsilon_net = EpsilonNet(sashimi_net, alphas_cumprod, timesteps, shape)

        initial_noise = torch.randn(cfg.algo.n_samples, *shape).cuda()

        mgps_fn = load_vdps_sampler(cfg.algo)

        samples = mgps_fn(
            initial_noise=initial_noise,
            labels=labels,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            display_freq=300,  # 10,
            display_fn=display_time_series,
            ground_truth=x_orig.cpu()
        )  # 10 minutes for 1 sample 1 observation...
        import pdb
        pdb.set_trace()

        # ecg_real, ecg_fake = get_12leads(x_orig.detach().cpu())[0], get_12leads(samples.detach().cpu())
        # mask_plot = np.zeros(ecg_real.shape).astype(bool)
        # mask_plot[0] = True
        # offset = 128
        # plot_12leads(ecg_real[:, offset:offset+200], ecg_fake[..., offset:offset+200].mean(axis=0), ecg_fake[..., offset:offset+200].std(axis=0),
        #              ecg_real[1, offset:], ecg_fake[:, 1, offset:].mean(axis=0), ecg_fake[:, 1, offset:].std(axis=0),
        #              )
        # plot_12leads(ecg_real, ecg_fake[0], np.percentile(ecg_fake, 5, axis=0), np.percentile(ecg_fake, 95, axis=0),
        #              mask[0], offset_T=offset, plot_T=200)
        # plt.show()

        # pred = samples.mean(dim=0).cpu().numpy()
        # real = x_orig.cpu()[0].numpy()
        # print('MAE', np.absolute(pred - real)[3:].sum(axis=0).mean())
        # display_time_series(pred, gt=real)
        # display_time_series(samples[-1].cpu(), gt=x_orig.cpu()[-1])
        # ecg_fake = get_12leads(samples.detach().cpu().numpy())[..., :1000]
        # ecg_real = get_12leads(x_orig.cpu().numpy())[0, :, :1000]
        all_samples.append(samples.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_x.append(x_orig.detach().cpu())
        all_masks.append(mask)
        np.savez(os.path.join(results_path, f'mgps_seed{cfg.algo.seed}_50diff.npz'),
                 generated=np.stack(all_samples),
                 label=np.concatenate(all_labels),
                 real=np.concatenate(all_x),
                 mask=np.concatenate(all_masks))
    print('ok')


if __name__ == '__main__':
    main()

