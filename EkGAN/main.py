import os
from tqdm import tqdm
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from EkGAN.models import InferenceGenerator, LabelGenerator, Discriminator
from EkGAN.training  import train_step, eval_step, plot_ecg
from torch.optim.lr_scheduler import StepLR
from diffusion_prior.dataloaders import PtbXLDataset

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset


def main():
    wandb.init(
        mode='online',  # Pass in 'wandb.mode=online' to turn on wandb logging
        project='EkGAN',
        entity='phdlisa',
        id='null',  # Set to string to resume logging from run
        job_type='training'        #config=OmegaConf.to_container(cfg, resolve=True)
    )
    path_to_save = '/mnt/data/lisa/ecg_results/EkGAN'
    train_set = PtbXLDataset('/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3',
                             training_class='train',
                             all_limb=False, segment_length=1024,
                             sampling_rate=100,
                            label_names=None,
                            noise=None)

    val_set = PtbXLDataset('/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3',
                             training_class='val',
                             all_limb=False, segment_length=1024,
                             sampling_rate=100,
                            label_names=None,
                            noise=None)

    train_loader = DataLoader(train_set, batch_size=128,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, drop_last=True, num_workers=0)

    inference_generator = InferenceGenerator()
    discriminator = Discriminator()
    label_generator = LabelGenerator()

    #inference_generator.apply(weights_init)
    #discriminator.apply(weights_init)
    #label_generator.apply(weights_init)

    inference_generator = inference_generator.cuda()
    discriminator = discriminator.cuda()
    label_generator = label_generator.cuda()

    lr, beta1, beta2 = 0.0001, 0.5, 0.999
    ig_optimizer = optim.Adam(inference_generator.parameters(), lr=lr, betas=(beta1, beta2))
    ig_scheduler = StepLR(ig_optimizer, step_size=1, gamma=0.95)
    lr, beta1, beta2 = 0.0001, 0.5, 0.999
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    disc_scheduler = StepLR(disc_optimizer, step_size=1, gamma=0.95)
    lr, beta1, beta2 = 0.0001, 0.5, 0.999
    lg_optimizer = optim.Adam(label_generator.parameters(), lr=lr, betas=(beta1, beta2))
    lg_scheduler = StepLR(lg_optimizer, step_size=1, gamma=0.95)

    n_epochs = 100
    lambda_, alpha = 50, 1
    best_epoch, best_loss = 0, 100000000
    for epoch in range(n_epochs):
        train_metrics = {'train/ig_loss': 0,
                       'train/ig_adv_loss': 0,
                       'train/ig_l1_loss': 0,
                       'train/lg_l1_loss': 0,
                       'train/vector_loss': 0}
        inference_generator.train()
        label_generator.train()
        discriminator.train()
        for i, (X, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            #X = nn.Upsample(256, mode='linear')(torch.swapaxes(X.to(torch.float32), 1, 2))

            # ==== prepare input ==== #
            n_leads = X.shape[1]
            lead_factor = int(n_leads)
            input_image = X[:, :1].repeat(1, lead_factor, 1)
            top_pad = int((16 -input_image.shape[1]) //2)
            bottom_pad = 16 - (top_pad + input_image.shape[1])
            input_image = torch.swapaxes(nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(input_image, 1, 2)), 1, 2)

            # ==== prepare target ==== #
            top_pad = int((16 -X.shape[1]) //2)
            bottom_pad = 16 - (top_pad + X.shape[1])
            target = torch.swapaxes(nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(X, 1, 2)), 1, 2)

            metrics = train_step(input_image.unsqueeze(1).cuda(),
                                 target.unsqueeze(1).cuda(),
                                 inference_generator,
                                 discriminator, ig_optimizer,
                                 disc_optimizer, label_generator,
                                 lg_optimizer, lambda_, alpha)
            train_metrics = {k: train_metrics[k] + metrics[k] for k in train_metrics.keys()}
        train_metrics = {k: train_metrics[k]/len(train_loader) for k in train_metrics.keys()}
        learning_rate = disc_optimizer.param_groups[0]['lr']
        train_metrics['train/lr'] = learning_rate
        wandb.log(train_metrics, step=epoch)

        if epoch >= 150:
            ig_scheduler.step()
            disc_scheduler.step()
            lg_scheduler.step()
        with torch.no_grad():
            inference_generator.eval()
            label_generator.eval()
            discriminator.eval()
            val_metrics = {'val/ig_loss': 0,
                           'val/ig_adv_loss': 0,
                           'val/ig_l1_loss': 0,
                           'val/lg_l1_loss': 0,
                           'val/vector_loss': 0}
            for i, (X, feats) in tqdm(enumerate(val_loader), total=len(val_loader)):
                X = nn.Upsample(256, mode='linear')(torch.swapaxes(X.to(torch.float32), 1, 2))

                # ==== prepare input ==== #
                n_leads = X.shape[1]
                lead_factor = int(n_leads)
                input_image = X[:, :1].repeat(1, lead_factor, 1)
                top_pad = int((16 - input_image.shape[1]) // 2)
                bottom_pad = 16 - (top_pad + input_image.shape[1])
                input_image = torch.swapaxes(
                    nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(input_image, 1, 2)), 1, 2)

                # ==== prepare target ==== #
                top_pad = int((16 - X.shape[1]) // 2)
                bottom_pad = 16 - (top_pad + X.shape[1])
                target = torch.swapaxes(nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(X, 1, 2)), 1, 2)

                metrics, ig_output = eval_step(input_image.cuda().unsqueeze(1), target.cuda().unsqueeze(1), inference_generator,
                                               discriminator, label_generator, lambda_, alpha)
                val_metrics = {k: val_metrics[k] + metrics[k] for k in val_metrics.keys()}
            val_metrics = {k: val_metrics[k] / len(val_loader) for k in val_metrics.keys()}
            wandb.log(val_metrics, step=epoch)

            # === visualize reconstruction === #
            for j, (corrupted_track, real_track, pred_track) in enumerate(
                    zip(input_image.numpy(), target.numpy(), ig_output.detach().cpu().numpy())):
                if j == 5:
                    break
                fig, ax = plt.subplots(1, 1, figsize=(5, 8))
                fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
                for color, ecg in zip(('red', 'blue', 'orange'), (corrupted_track, real_track, pred_track)):
                    for i, track in enumerate(ecg):
                        ax.plot(track - i, color=color, alpha=.7)
                wandb.log({f"reconstruction/{j}": wandb.Image(fig)}, step=epoch)
                plt.close(fig)

            # === saving model === #
            if val_metrics['val/ig_loss'] < best_loss:
                best_loss = val_metrics['val/ig_loss']
                best_epoch = epoch
                torch.save(inference_generator, os.path.join(path_to_save, "best_inference_generator.pth"))
                torch.save(label_generator, os.path.join(path_to_save, "best_label_generator.pth"))
                torch.save(discriminator, os.path.join(path_to_save, "best_discriminator.pth"))

if __name__ == '__main__':
    main()