import os
import time

from functools import partial
import multiprocessing as mp

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scipy.io.wavfile import write as wavwrite

from dataloaders import dataloader
from models import construct_model
from utils import print_size, calc_diffusion_hyperparams, local_directory, smooth_ckpt
from distributed_util import init_distributed

def sampling(net, size, diffusion_hyperparams, label=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    # Alpha = Alpha[::10]
    # Alpha_bar = Alpha_bar[::10]
    # Sigma = Sigma[::10]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = torch.normal(0, 1, size=size).cuda()
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, diffusion_steps,), label=label)  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=size).cuda()  # add the variance term to x_{t-1}
    return x


@torch.no_grad()
def generate(
        rank,
        results_path,
        diffusion_cfg,
        model_cfg,
        dataset_cfg,
        net=None,
        ckpt_iter="max",
        n_samples=1, # Samples per GPU
        name=None,
        label=None,
        batch_size=None,
        ckpt_smooth=None,
        batch_id=0,
        gt=None,
        save=False,
        # mel_path=None, mel_name=None,
        dataloader=None,
    ):
    """
    Generate audio based on ground truth mel spectrogram

    Parameters:
    output_directory (str):         checkpoint path
    n_samples (int):                number of samples to generate, default is 4
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    mel_path, mel_name (str):       condition on spectrogram "{mel_path}/{mel_name}.wav.pt"
    # dataloader:                     condition on spectrograms provided by dataloader
    """

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    local_path, output_directory = local_directory(name, results_path,
                                                   model_cfg, diffusion_cfg,
                                                   dataset_cfg, 'waveforms')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters

    if net is None:
        # predefine model
        net = construct_model(model_cfg).cuda()
        print_size(net)
        net.eval()

        # load checkpoint
        print('ckpt_iter', ckpt_iter)
        ckpt_path = os.path.join(results_path, local_path, 'checkpoint')
        # if ckpt_iter == 'max':
        #     ckpt_iter = find_max_epoch(ckpt_path)
        # ckpt_iter = int(ckpt_iter)

        if ckpt_smooth is None:
            try:
                # model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
                model_path = os.path.join(ckpt_path, 'checkpoint.pkl')
                checkpoint = torch.load(model_path, map_location='cpu')
                net.load_state_dict(checkpoint['model_state_dict'])
                # print('Successfully loaded model at iteration {}'.format(ckpt_iter))
            except:
                raise Exception('No valid model found')
        else:
            state_dict = smooth_ckpt(ckpt_path, ckpt_smooth, ckpt_iter, alpha=None)
            net.load_state_dict(state_dict)

    # Add checkpoint number to output directory
    output_directory = os.path.join(output_directory, 'generated_samples')
    if rank == 0:
        os.makedirs(output_directory, mode=0o775, exist_ok=True)
        print("saving to output directory", output_directory)

    if batch_size is None:
        batch_size = n_samples
    assert n_samples % batch_size == 0

    # if mel_path is not None and mel_name is not None:
    #     # use ground truth mel spec
    #     try:
    #         ground_truth_mel_name = os.path.join(mel_path, '{}.wav.pt'.format(mel_name))
    #         ground_truth_mel_spectrogram = torch.load(ground_truth_mel_name).unsqueeze(0).cuda()
    #     except:
    #         raise Exception('No ground truth mel spectrogram found')
    #     audio_length = ground_truth_mel_spectrogram.shape[-1] * dataset_cfg["hop_length"]
    # if mel_name is not None:
    #     if mel_path is not None: # pre-generated spectrogram
    #         # use ground truth mel spec
    #         try:
    #             ground_truth_mel_name = os.path.join(mel_path, '{}.wav.pt'.format(mel_name))
    #             ground_truth_mel_spectrogram = torch.load(ground_truth_mel_name).unsqueeze(0).cuda()
    #         except:
    #             raise Exception('No ground truth mel spectrogram found')
    #     else:
    #         import dataloaders.mel2samp as mel2samp
    #         dataset_name = dataset_cfg.pop("_name_")
    #         _mel = mel2samp.Mel2Samp(**dataset_cfg)
    #         dataset_cfg["_name_"] = dataset_name # Restore
    #         filepath = f"{dataset_cfg.data_path}/{mel_name}.wav"
    #         audio, sr = mel2samp.load_wav_to_torch(filepath)
    #         melspectrogram = _mel.get_mel(audio)
    #         # filename = os.path.basename(filepath)
    #         # new_filepath = cfg.output_dir + '/' + filename + '.pt'
    #         # print(new_filepath)
    #         # torch.save(melspectrogram, new_filepath)
    #         ground_truth_mel_spectrogram = melspectrogram.unsqueeze(0).cuda()
    #     audio_length = ground_truth_mel_spectrogram.shape[-1] * dataset_cfg["hop_length"]
    # else:
    # predefine audio shape
    audio_length = dataset_cfg["segment_length"]  # 16000
    # ground_truth_mel_spectrogram = None
    print(f'begin generating audio of length {audio_length} | {n_samples} samples with batch size {batch_size}')

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_audio = []

    with torch.no_grad():
        for _ in range(n_samples // batch_size):
            duration = time.time()
            _audio = sampling(
                net,
                (batch_size, model_cfg.in_channels, audio_length),
                diffusion_hyperparams,
                label=label,
            )
            duration = time.time() - duration
            generated_audio.append(_audio)
        generated_audio = torch.cat(generated_audio, dim=0)

    end.record()
    torch.cuda.synchronize()
    print('generated {} samples shape {} at iteration {} in {} seconds'.format(n_samples,
        generated_audio.shape,
        ckpt_iter,
        int(start.elapsed_time(end)/1000)))

    # save audio to .wav
    if 'physionet' not in dataset_cfg._name_ and 'ptbxl' not in dataset_cfg._name_:
        for i in range(n_samples):
            outfile = '{}k_{}.wav'.format(ckpt_iter // 1000, n_samples*rank + i)
            wavwrite(os.path.join(output_directory, outfile),
                        dataset_cfg["sampling_rate"],
                        generated_audio[i].squeeze().cpu().numpy())

            # save audio to tensorboard
            # tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))
            # tb.add_audio(tag=outfile, snd_tensor=generated_audio[i], sample_rate=dataset_cfg["sampling_rate"])
            # tb.close()

        print('saved generated samples at iteration %s' % ckpt_iter)
    if save:
        np.savez(os.path.join(output_directory, f'{dataset_cfg._name_}_{dataset_cfg.training_class}{batch_id}.npz'), ecg=generated_audio.detach().cpu().numpy(),
                 gt=gt.cpu().numpy(), label=label.detach().cpu().numpy())
    return generated_audio


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # load training data
    num_gpus = torch.cuda.device_count()
    # cfg.dataset.training_class = 'test'
    cfg.train.batch_size_per_gpu = int(10*cfg.train.batch_size_per_gpu)
    print(cfg.train.batch_size_per_gpu)
    testloader = dataloader(cfg.dataset, batch_size=cfg.train.batch_size_per_gpu, num_gpus=num_gpus, unconditional=cfg.model.unconditional, shuffle=False)
    print('Data loaded')
    # cfg.generate.n_samples = cfg.train.batch_size_per_gpu
    for batch_id, data in tqdm(enumerate(testloader), total=len(testloader)):
        if cfg.model.unconditional:
            audio = data[0]
            label = None
        elif 'ptbxl' in cfg.dataset._name_:
            audio, label = data
            label = label.to(torch.float32).cuda()
        cfg.generate.n_samples = audio.shape[0]
        generate_fn = partial(
            generate,
            results_path=cfg.train.results_path,
            diffusion_cfg=cfg.diffusion,
            model_cfg=cfg.model,
            label=label,
            gt=audio,
            dataset_cfg=cfg.dataset,
            save=True,
            batch_id=batch_id,
            **cfg.generate,
        )
        group_name = time.strftime("%Y%m%d-%H%M%S")
        if num_gpus <= 1:
            generate_fn(0)
        else:
            dist_cfg = cfg.pop("distributed")
            mp.set_start_method("spawn")
            processes = []
            for i in range(num_gpus):
                init_distributed(i, num_gpus, group_name, **dist_cfg)
                p = mp.Process(target=generate_fn, args=(i,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()


if __name__ == "__main__":
    main()
