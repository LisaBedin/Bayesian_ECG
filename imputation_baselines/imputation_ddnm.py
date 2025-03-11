import os
from dataclasses import dataclass
import sys
sys.path.append('.')
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm

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

@hydra.main(version_base=None, config_path="../sashimi/configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    fix_seed(cfg.algo.seed)
    print(f'Fixed seed {cfg.algo.seed}')

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

    timesteps = torch.linspace(0, cfg.diffusion.T-1, cfg.algo.nsteps).long()
    _dh = calc_diffusion_hyperparams(**cfg.diffusion, fast=False)
    T, Alpha, alphas_cumprod, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]).to(alphas_cumprod.device), alphas_cumprod])
    obs_std = cfg.algo.init_std

    # dataset = PhysionetECG(**cfg.dataset)
    # default_grad_steps = cfg.algo.parameters.gradient_steps_fn['conditions'][-1]['return']
    results_path = os.path.join(output_directory, f'{cfg.dataset.name}_{cfg.algo.missingness_type}{cfg.algo.missingness}')  # _K{cfg.algo.nsteps}_grad{default_grad_steps}')
    os.makedirs(results_path, exist_ok=True)
    all_samples, all_labels, all_x, all_masks = [], [], [], []
    # all_x_orig = np.load('/mnt/data/lisa/physionet.org/ptbxl_preprocessed/test_ptbxl_1000.npy')
    # all_y_orig = np.load('/mnt/data/lisa/physionet.org/ptbxl_preprocessed/1000_test_labels.npy')
    for x_orig, labels in tqdm(testloader, total=len(testloader)):
        # for x_orig, labels in zip(all_x_orig, all_y_orig):
        # start_i = np.random.randint(1000-256)
        # x_orig, labels = torch.tensor(x_orig[None, :, start_i:start_i+256]), torch.tensor(labels[None])
        # x_orig = torch.cat([x_orig[:, :3], x_orig[:, 6:]], dim=1)
        # x_orig = torch.zeros((*x_orig_tmp.shape[:-1], 1024))
        x_orig = x_orig.cuda()  # [:2]
        labels = labels.cuda().to(torch.float32) # [:2]
        cfg.dataset.segment_length = x_orig.shape[2]
        # break
        # epsilon_net = EpsilonNet(sashimi_net, alphas_cumprod, timesteps, x_orig.shape[1:])
        # shape = (-1, *x_orig.shape[1:])

        # ======== getting the observation =========== #
        if cfg.algo.missingness_type == 'rm':
            mask = np.stack([get_mask_rm(x_orig[0].T, k=int(cfg.dataset.segment_length*cfg.algo.missingness/100)).T.to(bool) for _ in range(len(x_orig))])
        if cfg.algo.missingness_type == 'mnr':
            mask = np.stack([get_mask_mnr(x_orig[0].T, k=int(cfg.dataset.segment_length * cfg.algo.missingness / 100)).T.to(bool) for _ in range(len(x_orig))])
        if cfg.algo.missingness_type == 'bm':
            mask = np.stack([get_mask_bm(x_orig[0].T, k=int(cfg.dataset.segment_length * cfg.algo.missingness / 100)).T.to(bool) for _ in range(len(x_orig))])
        if cfg.algo.missingness_type == 'lead':
            mask = np.zeros(x_orig.shape)
            mask[:, :cfg.algo.missingness] = 1
            mask = mask.astype(bool)
        # obs = torch.ones_like(x_orig)*torch.nan  # [mask].view(x_orig.shape[0], -1)
        # obs[mask] = x_orig[mask]

        # ==== using H function ==== #
        leads, time_points = np.meshgrid(
            np.arange(9),
            np.arange(cfg.dataset.segment_length),
        )
        missing_indices = (leads[~mask[0].T] * cfg.dataset.segment_length + time_points[~mask[0].T]).flatten()

        H_func = InpaintingTimeSeries(9, cfg.dataset.segment_length, missing_indices, torch.device('cuda'))
        # obs_tmp = H_func.H(x_orig[0:1]).reshape(-1, 9)
        # obs_bis = torch.zeros((9, 256)).cuda()
        # obs_bis[mask[0]] = obs_tmp.flatten()
        # display_time_series(obs[:1], gt=obs_bis.cpu())
        # recon = H_func.H_pinv(H_func.H(x_orig[0:1])).reshape(9, -1)
        # display_time_series(obs[:1], gt=recon.cpu())

        # display_time_series(obs_img.detach().cpu(), gt=x_orig[0, :3].detach().cpu())
        cfg.algo.nsamples = 10  # obs.shape[0]

        shape = x_orig.shape[1:]  # (9, 1024)
        initial_noise = torch.randn((cfg.algo.nsamples, *shape)).cuda()
        epsilon_net_svd = EpsilonNetSVDTimeSeries(sashimi_net, alphas_cumprod, timesteps, H_func, (-1, *shape), torch.device('cuda:0'))
        epsilon_net = EpsilonNet(sashimi_net, alphas_cumprod, timesteps, x_orig.shape[1:])
        obs = H_func.H(x_orig[0:1]).reshape(1, -1)
        # obs += torch.randn_like(obs)*obs_std
        samples = ddnm_plus(
            initial_noise,
            labels,
            epsilon_net,
            'inpainting',
            H_func,
            obs,
            sigma_y=obs_std,
        )

        ecg_real, ecg_fake = x_orig[0].cpu(), samples.detach().cpu()
        ecg_pred25, ecg_pred75 = np.percentile(ecg_fake, 5, axis=0), np.percentile(ecg_fake, 95, axis=0)
        fig = plot_12leads(ecg_real, ecg_fake[0],
                     ecg_pred25, ecg_pred75, mask[0]
                     )
        plt.show()

        # pred = samples.mean(dim=0).cpu().numpy()
        # real = x_orig.cpu()[0].numpy()
        # print('MAE', np.absolute(pred - real)[3:].sum(axis=0).mean())
        # display_time_series(pred, gt=real)
        # display_time_series(samples[-1].cpu(), gt=x_orig.cpu()[-1])
        all_samples.append(samples.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_x.append(x_orig.detach().cpu())
        all_masks.append(mask)
        np.savez(os.path.join(results_path, f'DDNM_seed{cfg.algo.seed}_GOOD.npz'),
                 generated=np.stack(all_samples),
                 label=np.concatenate(all_labels),
                 real=np.concatenate(all_x),
                 mask=np.concatenate(all_masks))
        print('ok')


if __name__ == '__main__':
    main()

