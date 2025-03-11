import os
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf


from posterior_samplers.utils import display_time_series
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.mgps import load_vdps_sampler

import torch

from diffusion_prior.dataloaders import dataloader
from diffusion_prior.utils import calc_diffusion_hyperparams, local_directory, print_size  # 
from diffusion_prior.models import construct_model


@hydra.main(version_base=None, config_path="sashimi/configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

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
    # sashimi_net = ModelWrapper(sashimi_net)

    timesteps = torch.linspace(0, cfg.diffusion.T-1, cfg.algo.n_steps).long()
    _dh = calc_diffusion_hyperparams(**cfg.diffusion, fast=False)
    T, Alpha, alphas_cumprod, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]).to(alphas_cumprod.device), alphas_cumprod])
    obs_std = cfg.algo.init_std

    epsilon_net = EpsilonNet(sashimi_net, alphas_cumprod, timesteps)

    # dataset = PhysionetECG(**cfg.dataset)
    for x_orig, labels in testloader:
        x_orig = x_orig.cuda()[:1]
        labels = labels.cuda()[:1].to(torch.float32)
        break
    n_lead = 3
    obs = x_orig[:, :n_lead].view(x_orig.shape[0], -1)
    obs_img = x_orig[:1, :n_lead]
    display_time_series(obs_img.detach().cpu(), gt=x_orig[0, :n_lead].detach().cpu())

    def log_pot(x):
        b = obs.shape[0]
        diff = obs.reshape(b, -1) - x[:, :n_lead].reshape(1, -1)
        return -0.5 * torch.norm(diff) ** 2 / obs_std ** 2

    shape = x_orig.shape[1:]  # (9, 1024)

    initial_noise = torch.randn(cfg.algo.n_samples, *shape).cuda()

    vdps_fn = load_vdps_sampler(cfg.algo)

    samples = vdps_fn(
        initial_noise=initial_noise,
        labels=labels,
        epsilon_net=epsilon_net,
        log_pot=log_pot,
        display_freq=300,
        display_fn=display_time_series,
        ground_truth=x_orig.cpu()[0]
    )  # 10 minutes for 1 sample 1 observation...

    display_time_series(samples[0].cpu(), gt=x_orig.cpu()[0])
    print('ok')

if __name__ == '__main__':
    main()

