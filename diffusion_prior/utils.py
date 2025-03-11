import os
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('sashimi')
import glob
import torch.nn as nn
import numpy as np
import torch

from diffusion_prior.models import model_identifier


class ModelWrapper(nn.Module):
    def __init__(self, model, in_channels=9):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.in_channels = in_channels

    def forward(self, x, time_step):
        '''

        Args:
            x: flatten batch tensor of shape (B, C*seq_length)
            time_step: time_step for the backward diffusion, tensor of shape T

        Returns:

        '''
        x = x.view(x.size(0), self.in_channels, -1)
        return self.model((x, time_step.unsqueeze(0).unsqueeze(0)), mel_spec=None)  # .view(x.size(0), -1)


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def rescale(x):
    """
    Rescale a tensor to 0-1
    """

    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch_old(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path

    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch

def find_max_epoch(path):
    file_p = glob.glob(os.path.join(path, '*.txt'))
    if len(file_p) == 0:
        return -1
    file_p = file_p[0]
    with open(file_p, 'r') as file:
        line = file.readline().strip()
        last_iter = int(line)
    return last_iter

def smooth_ckpt(path, min_ckpt, max_ckpt, alpha=None):
    print(f"finding checkpoints in ({min_ckpt}, {max_ckpt}] in {path}")
    files = os.listdir(path)
    ckpts = []
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            print(f)
            try:
                it = int(f[:-4])
                if min_ckpt < it and it <= max_ckpt:
                    ckpts.append(it)
            except:
                continue
    ckpts = sorted(ckpts)
    print("found ckpts", ckpts)
    state_dict = None
    for n, it in enumerate(ckpts):
        model_path = os.path.join(path, '{}.pkl'.format(it))
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # net.load_state_dict(checkpoint['model_state_dict'])
            state_dict = smooth_dict(state_dict, checkpoint['model_state_dict'], n, alpha=alpha)
            print('Successfully loaded model at iteration {}'.format(it))
        except:
            raise Exception(f'No valid model found at iteration {it}, path {model_path}')
    return state_dict


def print_size(net, verbose=False):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        # module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        module_parameters = list(filter(lambda p: p[1].requires_grad, net.named_parameters()))

        if verbose:
            for n, p in module_parameters:
                print(n, p.numel())

        params = sum([np.prod(p.size()) for n, p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


@torch.no_grad()
def get_test_predictions(
        results_path,
        diffusion_cfg,
        model_cfg,
        dataset_cfg,
        name=None):
    local_path, output_directory = local_directory(name, results_path, model_cfg, diffusion_cfg, dataset_cfg, 'waveforms')
    output_directory = os.path.join(output_directory, 'generated_samples')
    all_files = glob.glob(os.path.join(output_directory, f'{dataset_cfg._name_}_{dataset_cfg.training_class}*.npz'))
    labels, ecg_real, ecg_gen = [], [], []
    for file_p in all_files:
        npz = np.load(file_p)
        labels.append(npz['label'])
        ecg_real.append(npz['gt'])
        ecg_gen.append(npz['ecg'])
    return np.concatenate(labels), np.concatenate(ecg_real), np.concatenate(ecg_gen), output_directory


def prepare_downstream_data(ecg, labels, segment_length=1024, sampling_rate=100):
    ecg = np.swapaxes(ecg[:, :, :min(int(sampling_rate * 10), segment_length)], 1,2)
    aVR = -(ecg[:, :, 0] + ecg[:, :, 1]) / 2
    aVL = (ecg[:, :, 0] - ecg[:, :, 2]) / 2
    aVF = (ecg[:, :, 1] + ecg[:, :, 2]) / 2
    augm_leads = np.stack([aVR, aVL, aVF], axis=-1)
    ecg = np.concatenate([ecg[:, :, :3], augm_leads, ecg[:, :, 3:]], axis=-1)
    return ecg, labels

def local_directory(name, results_path, model_cfg, diffusion_cfg, dataset_cfg, output_directory):
    # tensorboard_directory = train_cfg['tensorboard_directory']
    # ckpt_path = output_directory # train_cfg['output_directory']

    # generate experiment (local) path
    model_name = model_identifier(model_cfg)
    diffusion_name = f"_T{diffusion_cfg['T']}_betaT{diffusion_cfg['beta_T']}"
    # if model_cfg["unconditional"]:
    #     data_name = ""
    # else:
    # if dataset_cfg.name == 'physionet_centered':
    #     data_name = '_centered'
    # else:
    dataset_name = 'ptbxl' if dataset_cfg.name == 'georgia' or dataset_cfg.name == 'codetest' else dataset_cfg.name
    data_name = f"_{dataset_name}{dataset_cfg.sampling_rate}_L{dataset_cfg['segment_length']}"  # _hop{dataset_cfg['hop_length']}"
    # if 'arrhythmia' in dataset_cfg.name or 'arrhythmia' in dataset_cfg.database_path:
    #     data_name += '_arrhythmia'
    local_path = model_name + diffusion_name + data_name #+ f"_{'12L' if dataset_cfg['all_limb'] else '9L'}"
    if not model_cfg['unconditional']:
        local_path += '_cond'
    if dataset_cfg['all_limb']:
        local_path += '_allLimbs'
    if model_cfg.scheduler:
        local_path += '_sched'

    if not (name is None or name == ""):
        local_path = name + "_" + local_path

    # Get shared output_directory ready
    output_directory = os.path.join(results_path, local_path, output_directory)
    os.makedirs(output_directory, mode=0o775, exist_ok=True)
    print("output directory", output_directory, flush=True)
    return local_path, output_directory


# Utilities for diffusion models

def calc_diffusion_hyperparams(T, beta_0, beta_T, beta=None, fast=False):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    if fast and beta is not None:
        Beta = torch.tensor(beta)
        T = len(beta)
    else:
        Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta.cuda(), Alpha.cuda(), Alpha_bar.cuda(), Sigma
    return _dh


""" Experimental feature for checkpoint smoothing. Didn't seem to help in brief tests """
def smooth_dict(d, d0, n=None, alpha=None):
    """ Smooth with arithmetic average (if n not None) or geometric average (if alpha not None) """
    assert int(n is None) + int(alpha is None) == 1
    if d is None:
        assert n is None or n == 0 # must be first iteration
        return d0

    if n is not None:
        avg_fn = lambda x, y: (x * n + y) / (n+1)
    else:
        avg_fn = lambda x, y: alpha * x + (1. - alpha) * y
    return _bin_op_dict(d, d0, avg_fn)

def _bin_op_dict(d0, d1, op):
    """ Apply binary operator recursively to two dictionaries with matching keys """
    if isinstance(d0, dict) and isinstance(d1, dict):
        assert d0.keys() == d1.keys(), "Dictionaries must hvae matching keys"
        return {
            k: _bin_op_dict(d0[k], d1[k], op) for k in d0.keys()
        }
    elif not isinstance(d0, dict) and not isinstance(d1, dict):
        return op(d0, d1)
    else: raise Exception("Dictionaries must match keys")

