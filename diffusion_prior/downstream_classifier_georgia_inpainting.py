# inspired from https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/Finetuning-Example.ipynb
import os
from sklearn.metrics import roc_auc_score, auc, f1_score, precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
import hydra
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from benchmarks.ptbxl_strodoff.fastai_model import fastai_model
from utils import get_test_predictions, prepare_downstream_data
from diffusion_prior.dataloaders import dataloader
from diffusion_prior.utils import calc_diffusion_hyperparams, local_directory, print_size  # 
from diffusion_prior.models import construct_model
from posterior_samplers.mgps import load_vdps_sampler


def evaluation_scores(y_true, y_pred):
    auc_scores = []  # auc one-vs-the rest
    # y_val_pred = model.predict(X_test_gen)
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores.append(auc)
        except:  # current class not present in the test set
            ()
    return np.mean(auc_scores)



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys
    cfg.dataset.all_limb = True
    cfg.dataset.training_class = 'train'
    num_gpus = torch.cuda.device_count()
    trainloader = dataloader(cfg.dataset, batch_size=cfg.train.batch_size_per_gpu, num_gpus=num_gpus, unconditional=cfg.model.unconditional, shuffle=False)
    cfg.dataset.training_class = 'test'
    testloader = dataloader(cfg.dataset, batch_size=cfg.train.batch_size_per_gpu, num_gpus=num_gpus, unconditional=cfg.model.unconditional, shuffle=False)

    local_path, output_directory = local_directory(None, cfg.train.results_path,
                                                   cfg.model, cfg.diffusion,
                                                   cfg.dataset,
                                                   'waveforms')


    all_limbs = '12D'*cfg.dataset.all_limb + ''
    output_directory = os.path.join(output_directory, f'{cfg.dataset.name}_downstream_inpainting{all_limbs}')
    os.makedirs(output_directory, exist_ok=True)

    X_train, y_train = [], []
    for i, (data, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
        X_train.append(data.numpy())
        y_train.append(labels.numpy())
    X_train = np.concatenate(X_train)
    X_train = np.swapaxes(X_train, 1, 2)[:, :1000]
    y_train = np.concatenate(y_train)

    X_test, y_test = [], []
    for i, (data, labels) in tqdm(enumerate(testloader), total=len(testloader)):
        X_test.append(data.numpy())
        y_test.append(labels.numpy())
    X_test = np.concatenate(X_test)
    X_test = np.swapaxes(X_test, 1, 2)[:, :1000]
    y_test = np.concatenate(y_test)

    mean_train, std_train = X_train.mean(), X_train.std()
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - mean_train) / std_train

    model = fastai_model(
        'fastai_xresnet1d50',
        8,
        cfg.dataset.sampling_rate,
        outputfolder=output_directory,
        input_shape=X_train.shape[1:],  # [1000, 12],
        pretrainedfolder=None,  # os.path.join(cfg.evaluate.benchmark_folder, 'ptbxl_strodoff/fastai_xresnet1d101') if ,
        n_classes_pretrained=8,
        pretrained=False,
        epochs_finetuning=0,
        aggregate_fn='mean',
        bs=64, # 64,
        epochs=50,
        lr=1e-3,
        wd=1e-3,  # 1e-3,
    )

    model.fit(X_train, y_train, X_test, y_test)

    # == load best model == #
    model = fastai_model(
        'fastai_xresnet1d50',
        8,
        cfg.dataset.sampling_rate,
        outputfolder=output_directory,
        input_shape=X_train.shape[1:],  # [1000, 12],
        pretrainedfolder=os.path.join(output_directory, 'models/fastai_xresnet1d50.pth'),
        n_classes_pretrained=8,
        pretrained=True,
        epochs_finetuning=0,
        aggregate_fn='mean',
        bs=64,
        epochs=0,
        lr=1e-3,
        wd=1e-3,
    )

    y_val_pred = model.predict(X_test)
    auc_real = evaluation_scores(y_test, y_val_pred)
    print('test auc', auc_real)


if __name__ == '__main__':
    main()
