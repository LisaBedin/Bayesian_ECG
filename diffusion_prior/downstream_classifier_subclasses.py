# inspired from https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/Finetuning-Example.ipynb
import os

#from fastai.metrics import auc_roc_score
from sklearn.metrics import roc_auc_score, auc, f1_score, precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import torch
sys.path.append('.')

from benchmarks.ptbxl_strodoff.fastai_model import fastai_model
from utils import get_test_predictions, prepare_downstream_data
from diffusion_prior.dataloaders import dataloader


def find_optimal_threshold(y_true, y_pred):
    n_classes = y_true.shape[1]
    optimal_thresholds = np.zeros(n_classes)

    for i in range(n_classes):
        # Calculer la courbe ROC
        #fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])

        # Calculer la courbe Precision-Recall
        precision, recall, thresholds_pr = precision_recall_curve(y_true[:, i], y_pred[:, i])

        # Calculer le score F1 pour chaque seuil
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # Trouver le seuil qui maximise le score F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_pr[optimal_idx]

        optimal_thresholds[i] = optimal_threshold

    return optimal_thresholds


def evaluation_scores(y_true, y_pred, score_fn=roc_auc_score):
    auc_scores = []  # auc one-vs-the rest
    # y_val_pred = model.predict(X_test_gen)
    for i in range(y_true.shape[1]):
        try:
            auc = score_fn(y_true[:, i], y_pred[:, i])
            auc_scores.append(auc)
        except:  # current class not present in the test set
            ()
    return np.mean(auc_scores)



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys
    cfg.model.unconditional = False
    cfg.dataset.training_class = 'train'
    cfg.train.batch_size_per_gpu = 1
    lead1 = True # ATTENTION
    cfg.dataset.label_names = ['NSR', 'RBBB', 'LBBB', 'AF']
    trainloader = dataloader(cfg.dataset,
                            batch_size=cfg.train.batch_size_per_gpu,
                            num_gpus=1,
                            unconditional=cfg.model.unconditional,
                            shuffle=False)
    lab_tmp = trainloader.dataset.labels.numpy()
    X_train = np.swapaxes(trainloader.dataset.data.numpy(), 1, 2)
    if lead1:
        X_train = X_train[:, :, :1]

    y_train = np.stack([
        lab_tmp[:, 46] * lab_tmp[:, 61],
        np.maximum(lab_tmp[:, 12], lab_tmp[:, 11]),
        lab_tmp[:, 4]
    ], axis=1)
    cfg.dataset.training_class = 'val'

    valloader = dataloader(cfg.dataset,
                      batch_size=cfg.train.batch_size_per_gpu,
                      num_gpus=1,
                      unconditional=cfg.model.unconditional,
                      shuffle=False)
    lab_tmp = valloader.dataset.labels.numpy()
    y_val = np.stack([
        lab_tmp[:, 46] * lab_tmp[:, 61],
        np.maximum(lab_tmp[:, 12],lab_tmp[:, 11]),
        lab_tmp[:, 4]
    ], axis=1)
    X_val = np.swapaxes(valloader.dataset.data.numpy(), 1, 2)
    if lead1:
        X_val = X_val[:, :, :1]
    cfg.dataset.training_class = 'test'

    testloader = dataloader(cfg.dataset,
                      batch_size=cfg.train.batch_size_per_gpu,
                      num_gpus=1,
                      unconditional=cfg.model.unconditional,
                      shuffle=False)
    lab_tmp = testloader.dataset.labels.numpy()
    y_test = np.stack([
        lab_tmp[:, 46] * lab_tmp[:, 61],
        np.maximum(lab_tmp[:, 12], lab_tmp[:, 11]),
        lab_tmp[:, 4]
    ], axis=1)
    X_test = np.swapaxes(testloader.dataset.data.numpy(), 1, 2)
    if lead1:
        X_test = X_test[:, :, :1]

    output_directory = os.path.join(cfg.train.results_path, 'ptbxlStrodoff_3classes/'+ ('maxn_agg_I' if lead1 else 'max_agg'))
    os.makedirs(output_directory, exist_ok=True)

    mean_train, std_train = X_train.mean(), X_train.std()
    print('###### mean and std of train data ######')
    print(f'mean: {mean_train}, std: {std_train}')
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_val = (X_val - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    model = fastai_model(
        'fastai_xresnet1d101',
        3,
        cfg.dataset.sampling_rate,
        outputfolder=output_directory,
        input_shape=X_train.shape[1:],  # [1000, 12],
        n_classes_pretrained=3,  # 71
        pretrained=False,
        #pretrained=True,
        # early_stopping="valid_loss",
        epochs_finetuning=0,  #50,
        aggregate_fn='max',
        bs=128, # 64, # 64,
        epochs=100,
        lr=1e-3,
        wd=1e-3,  # 1e-3,
    )

    # == training the model from real data == #
    model.fit(X_train, y_train, X_val, y_val)

    # == load best model == #
    model = fastai_model(
        'fastai_xresnet1d101',
        3,
        cfg.dataset.sampling_rate,
        outputfolder=output_directory,
        input_shape=X_train.shape[1:],  # [1000, 12],
        pretrainedfolder=output_directory,
        n_classes_pretrained=3,
        pretrained=True,
        epochs_finetuning=1,
        aggregate_fn='max',
        bs=128,
        epochs=0,
        lr=1e-3,
        wd=1e-3,
    )

    print(f'=========== validation set ============')
    y_val_pred = model.predict(X_val)
    #auc_val = evaluation_scores(y_val, y_val_pred, score_fn=average_precision_score)
    #print(f'avgPrecision val = {auc_val:.4f}')
    total_score = 0
    for k, lab in enumerate(['NSR', 'RBBB', 'AF']):
        score = average_precision_score(y_val[:, k], y_val_pred[:, k])
        print(f'avgPrecision {lab} = {score:.4f}')
        total_score += score
    total_score /= 3
    print(f'avgPrecision val = {total_score:.4f}')
    '''
    y_test_pred = model.predict(X_test)
    #auc_val = evaluation_scores(y_val, y_val_pred, score_fn=average_precision_score)
    #print(f'avgPrecision val = {auc_val:.4f}')
    total_score = 0
    opt_th = find_optimal_threshold(y_val, y_val_pred)

    for k, lab in enumerate(['NSR', 'LBBB', 'RBBB', 'AF']):
        score = f1_score(y_test[:, k], y_test_pred[:, k]>opt_th[k])
        print(f'F1 {lab} = {score:.4f}')
        total_score += score
    total_score /= 4.
    print(f'F1 test = {total_score:.4f}')
    total_score = 0
    for k, lab in enumerate(['NSR', 'LBBB', 'RBBB', 'AF']):
        score = average_precision_score(y_test[:, k], y_test_pred[:, k]>opt_th[k])
        print(f'auc {lab} = {score:.4f}')
        total_score += score
    total_score /= 4.
    print(f'avgPrecision test = {total_score:.4f}')

    for k, lab in zip([0, 1, 3], ['NSR', 'BBB', 'AF']):
        if lab == 'BBB':
            tmp_lab = np.maximum(y_test[:, 1], y_test[:, 2])
            tmp_pred = y_test_pred[:, 1:3].max(axis=1)
            score = average_precision_score(tmp_lab, tmp_pred)
        else:
            score = average_precision_score(y_test[:, k], y_test_pred[:, k])
        print(f'avgPrecision pred {lab} = {score:.4f}')

    total_score = 0
    for k, lab in enumerate(['NSR', 'LBBB', 'RBBB', 'AF']):
        score = roc_auc_score(y_test[:, k], y_test_pred[:, k]>opt_th[k])
        print(f'auc {lab} = {score:.4f}')
        total_score += score
    total_score /= 4.
    print(f'auc test = {total_score:.4f}')

    total_score = 0
    opt_th = find_optimal_threshold(y_val, y_val_pred)
    for k, lab in enumerate(['NSR', 'LBBB', 'RBBB', 'AF']):
        score = f1_score(y_val[:, k], y_val_pred[:, k]>0.5) # opt_th[k])
        print(f'F1 {lab} = {score:.4f}')
        total_score += score
    total_score /= 4.
    print(f'F1 val = {total_score:.4f}')
    # df = pd.DataFrame({'data': ['all', 'RBBB'], 'real': [auc_real, RBBB_real], 'gen': [auc_gen, RBBB_gen]})
    # print(df)
    # df.to_csv(os.path.join(output_directory, 'auc.csv'))
    '''


if __name__ == '__main__':
    main()
