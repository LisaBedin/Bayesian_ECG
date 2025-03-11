import os
import time
import sys
sys.path.append('..')

from functools import partial
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors

import numpy as np
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

from dataloaders import dataloader
from models import construct_model
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory, smooth_ckpt
from utils import get_test_predictions

from benchmarks.ptbxl_strodoff.fastai_model import fastai_model


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys
    cfg.dataset.training_class = 'test'
    labels, ecg_real, ecg_gen, output_directory = get_test_predictions(
        results_path=cfg.train.results_path,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset)
    print('ok')

    # === NN evaluation === #
    X_fake = ecg_gen.reshape(ecg_real.shape[0], -1)
    X_real = ecg_real.reshape(ecg_gen.shape[0], -1)
    n_data = len(X_fake)
    X_combined = np.vstack((X_real, X_fake))
    y_combined = np.array([0] * n_data + [1] * n_data)

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_combined)
    distances, indices = nbrs.kneighbors(X_combined)
    y_pred = y_combined[indices[:, 1]]
    accuracy = np.mean(y_pred == y_combined)
    print(f"Accuracy of 1-NN classifier: {accuracy:.4f}")

    # === NN evaluation on RBBB === #
    inds = labels[:, 11].astype(bool)
    X_fake = ecg_gen[inds].reshape(inds.sum(), -1)
    X_real = ecg_real[inds].reshape(inds.sum(), -1)
    n_data = len(X_fake)
    X_combined = np.vstack((X_real, X_fake))
    y_combined = np.array([0] * n_data + [1] * n_data)

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_combined)
    distances, indices = nbrs.kneighbors(X_combined)
    y_pred = y_combined[indices[:, 1]]
    accuracy = np.mean(y_pred == y_combined)
    print(f"Accuracy of 1-NN classifier: {accuracy:.4f}")


if __name__ == '__main__':
    main()
