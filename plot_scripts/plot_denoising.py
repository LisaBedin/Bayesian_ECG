import os
import hydra
from keras.src.metrics import F1Score
from omegaconf import DictConfig
import sys
sys.path.append('.')
sys.path.append('sashimi')
import numpy as np
from tqdm import tqdm
from ecg_ribeiro.resnet import ResNet1d
import json
import torch
from sklearn.metrics import recall_score, precision_score, f1_score
from posterior_samplers.utils import display_time_series
from diffusion_prior.utils import local_directory
from scipy import stats
from diffusion_prior.benchmarks.ptbxl_strodoff.fastai_model import fastai_model
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, balanced_accuracy_score
import torch.nn as nn
from keras.src.saving.saving_api import load_model as keras_load_model
import h5py
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import resample_poly
from plot_scripts.SoftDTW_functions import SoftDTW


#plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 14})


'''
plt.rcParams["font.serif"] = "cm"  # Use Computer Modern (LaTeX default font)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
colors = ["blue", "red"]
'''

def MAE(y, y_hat):
    '''
    Args:
        y: B x T x L
        y_hat:  B x T x L
        mask:  B x T x L

    Returns: mae:  B
    '''
    mae = [np.absolute((y[k]-y_hat[k])).mean() for k in range(y.shape[0])]
    return np.array(mae)


def MaxAE(y, y_hat):
    '''
    Args:
        y: B x T x L
        y_hat:  B x T x L
        mask:  B x T x L

    Returns: mae:  B
    '''
    mae = [np.absolute((y[k]-y_hat[k])).max(axis=-1).mean() for k in range(y.shape[0])]
    return np.array(mae)

def RMSE(y, y_hat):
    rmse = [np.sqrt(((y[k]-y_hat[k])**2).mean()) for k in range(y.shape[0])]
    return np.array(rmse)


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)


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


def evaluation_scores_old(y_true, y_pred):
    if y_true.shape[1] == 8:
        y_true_bis = np.ones_like(y_pred)
        y_true_bis[:, 11] = y_true[:, 6]
        y_true_bis[:, 12] = y_true[:, 7]
        y_true_bis[:, 4] = y_true[:, 0]
        y_true_bis[:, 67] = y_true[:, 1]
        y_true_bis[:, 54] = y_true[:, 3]
        y_true_bis[:, 58] = y_true[:, 5]
        y_true = y_true_bis
    auc_score_dic = {}
    labels_dics = {'LBBB': 11, 'RBBB': 12, 'AF': 4, 'TAb': 67, 'PVC': 54, 'SA': 58}
    for lab, i in labels_dics.items():
        try:
            f1_s = f1_score(y_true[:, i], y_pred[:, i])
            auc_score_dic[lab] = f1_s
        except:  # current class not present in the test set
            ()
    return auc_score_dic


def evaluation_report(y_true, y_pred):
    lab_lst = ['AF', 'TAb', 'QAb', 'VPB', 'LAD', 'SA', 'LBBB', 'RBBB']
    recall_dic, specific_dic = {}, {}
    for i, lab in enumerate(lab_lst):
        recall = recall_score(y_true[:, i], y_pred[:, i], pos_label=1)
        specificity = recall_score(y_true[:, i], y_pred[:, i], pos_label=0)
        recall_dic[lab] = recall
        specific_dic[lab] = specificity
    return recall_dic, specific_dic


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


def prepare_for_ribeiro(signal):
    newFs, oldFs = 400, 100
    L = math.ceil(len(signal) * newFs / oldFs)
    normBeat = list(reversed(signal)) + list(signal) + list(reversed(signal))

    # resample beat by beat and saving it
    res = resample_poly(np.array(normBeat), newFs, oldFs, axis=0)
    res = res[L - 1:2 * L - 1]
    res_padded = np.zeros((4096, res.shape[1]))
    res_padded[48:-48] = res
    return res_padded


def plot_12leads_old(ecg, ecg_pred, ecg_std, ecgII, ecg_predII, ecg_predII_std):
    max_H = 2.1   # 3.1  # cm = mV
    offset_H = 1  # 1.5
    margin_H = 0.
    margin_W = 0.2

    # T = ecg.shape[1]*4*2.5*0.001  # (*4 ms * 25 mm)
    T = ecg.shape[1]*10*2.5*0.001  # (*10 ms * 25 mm)
    times = np.linspace(0, T, ecg.shape[1])

    H = max_H*4 + margin_H*5  # in cm
    W = T*4 + margin_W*5  # in cm

    fig, ax = plt.subplots(figsize=(W/2.54, H/2.54))  # 1/2.54  # centimeters in inches
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)


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
    ind_H = np.concatenate([np.arange(4, 1, -1)]*4)

    lW, lH = ind_W[0], ind_H[0]
    ax.fill_between(((lW-1)*T+lW*margin_W + times) / 2.54, max_H*(lH-1) / 2.54,(lH*max_H-margin_W)/2.54, alpha=0.1, color='red', rasterized=True)

    for ecg_l, lW, lH in zip(ecg, ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='red', alpha=.5, lw=1.,
                rasterized=True)
    for ecg_l, lW, lH in zip(ecg_pred, ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='blue', alpha=.5, lw=1.,
                rasterized=True)
    for ecg_l, ecg_st, lW, lH in zip(ecg_pred, ecg_std, ind_W, ind_H):
        ax.fill_between(((lW-1)*T+lW*margin_W + times) / 2.54,
                        (ecg_l + ecg_st + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                        (ecg_l - ecg_st + offset_H + max_H * (lH - 1) + lH * margin_H) / 2.54,
                color='blue', alpha=.5, # lw=2.,
                rasterized=True)

    T = ecgII.shape[0] * 10 * 2.5 * 0.001  # (*10 ms * 25 mm)
    times = np.linspace(0, T, ecgII.shape[0])
    kept_inds = times < (W - 2*margin_W)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecgII[kept_inds] + offset_H) / 2.54,
            c='red', alpha=.5, lw=1.,
            rasterized=True)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecg_predII[kept_inds] + offset_H) / 2.54,
            c='blue', alpha=.5, lw=1.,
            rasterized=True)
    ax.fill_between((margin_W + times[kept_inds]) / 2.54, (ecg_predII[kept_inds] + offset_H + ecg_predII_std[kept_inds]) / 2.54,
                    (ecg_predII[kept_inds] + offset_H - ecg_predII_std[kept_inds]) / 2.54,
            color='blue', alpha=.5,   # lw=1.,
            rasterized=True)


def plot_12leads(ecg, ecg_pred, ecg_pred25, ecg_pred75, mask, plot_T, offset_T, ticks=True, fs=11, title='', top_=1):
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
        fig.suptitle(title)
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
    for ecg_l, lW, lH in zip(ecg[:, offset_T:offset_T+plot_T], ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='royalblue', alpha=1., lw=1.,
                rasterized=True)
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
                c='darkorange', alpha=1, lw=1.,
                rasterized=True)
        ax.annotate(ann, (((lW-1)*T+margin_W*lW) / 2.54, (1.5*offset_H + max_H*(lH-1)) / 2.54), fontsize=fs)

    # ====== plot lead II =====
    ecgII = ecg[1, offset_T:]
    ecg_predII = ecg_pred[1, offset_T:]
    ecg_pred25II, ecg_pred75II = ecg_pred25[1, offset_T:], ecg_pred75[1, offset_T:]
    T = ecgII.shape[0] * 10 * 2.5 * 0.001  # (*10 ms * 25 mm)
    times = np.linspace(0, T, ecgII.shape[0])
    kept_inds = times < (W - 2*margin_W)

    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecgII[kept_inds] + offset_H) / 2.54,
            c='royalblue', alpha=1., lw=1.,
            rasterized=True)
    ax.fill_between((margin_W + times[kept_inds]) / 2.54, (offset_H + ecg_pred25II[kept_inds]) / 2.54,
                    (offset_H + ecg_pred75II[kept_inds]) / 2.54,
            color='green', alpha=.4, lw=0.,
            rasterized=True)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecg_predII[kept_inds] + offset_H) / 2.54,
            c='darkorange', alpha=1., lw=1.,
            rasterized=True)
    ax.annotate('II', (margin_W * 2 / 2.54, (1.5 * offset_H) / 2.54),
                fontsize=fs)

    return fig

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

def get_3labels(labs):
    return np.stack([
        labs[:, 46] * labs[:, 61],  # NSR
        np.maximum(labs[:, 12],labs[:, 11]), # RBBB
        labs[:, 4] # AF
       # labs[:, 59],
    ], axis=1)


def main():
    output_directory = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L1024/waveforms'
    # dataset = PhysionetECG(**cfg.dataset)
    dataset = 'ptbxl'  #  'ptbxl'  #  'codetest'  # ptbxl

    model_path = '/mnt/data/lisa/ecg_results/sashimi/ptbxlStrodoff_3classes/max_agg'
    model = fastai_model(
        'fastai_xresnet1d101',
        3,
        100,
        outputfolder=model_path,
        input_shape=[1000, 12],
        pretrainedfolder=model_path,
        n_classes_pretrained=3,
        pretrained=True,
        epochs_finetuning=1,
        aggregate_fn='max',
        bs=64,
        epochs=0,
        lr=1e-3,
        wd=1e-3,
    )

    mean_train, std_train = (3.4904378480860032e-06, 0.2081562578678131)
    em_path = os.path.join(output_directory, 'GOOD_denoising/ptbxl_denoising_em_step5_lasso0.1_ridge1')  # ptbxl_denoising_em_step10_lasso0.01_ridge0.01')
    bw_path = os.path.join(output_directory, 'GOOD_denoising/ptbxl_denoising_bw_step5_lasso0.1_ridge1')

    # ===== get the threshold from the validation set ====== #
    opt_th = np.array([0.5]*4) # 0.610611, 0.822932, 0.206047, 0.428946])

    all_ecg_real, all_ecg_em, all_ecg_bw = [], [], []
    preds_em, preds_bw, preds_real = [], [], []
    lab_em, lab_bw, lab_real = [], [], []
    for lab in ['NSR', 'LBBB', 'RBBB', 'AF']:
        npz_em = np.load(os.path.join(em_path, f'Denoising_{lab}_seed0.npz'))
        npz_bw = np.load(os.path.join(bw_path, f'Denoising_{lab}_seed0.npz'))
        ecg_real = get_12leads(npz_em['real'][:, :, :1000])
        ecg_em = get_12leads(npz_em['generated'].reshape(-1, 9, 1024)[:, :, :1000])
        ecg_bw = get_12leads(npz_bw['generated'].reshape(-1, 9, 1024)[:, :, :1000])
        all_ecg_real.append(ecg_real)
        all_ecg_em.append(ecg_em.reshape(-1, 10, 12, 1000))
        all_ecg_bw.append(ecg_bw.reshape(-1, 10, 12, 1000))
        pred_real = model.predict(np.swapaxes((ecg_real-mean_train)/std_train, 1, 2))
        pred_em = model.predict(np.swapaxes((ecg_em-mean_train)/std_train, 1, 2))
        pred_bw = model.predict(np.swapaxes((ecg_bw-mean_train)/std_train, 1, 2))
        #pred_real = pred_real[:, kept_inds]
        pred_em = pred_em.reshape(-1, 10, 3)#[:, :, kept_inds]
        pred_bw = pred_bw.reshape(-1, 10, 3)#[:, :, kept_inds]
        pred_em = pred_em.mean(axis=1)
        pred_bw = pred_bw.mean(axis=1)
        lab_em.append(npz_em['label'])#[:, kept_inds])labs
        lab_bw.append(npz_bw['label'])#[:, kept_inds])
        lab_real.append(npz_em['label'])#[:, kept_inds])
        # preds_em[:, 1] = np.minimum(pred_em[:, 1]+preds_em[:, 2], 1)
        # preds_real[:, 1] = np.minimum(pred_real[:, 1]+preds_real[:, 2], 1)
        # preds_bw[:, 1] = np.minimum(pred_bw[:, 1]+pred_bw[:, 2], 1)
        # preds_em[:, 3] = preds_em[:, 4]
        # preds_real[:, 3] = preds_real[:, 4]
        # preds_bw[:, 3] = preds_bw[:, 4]
        preds_em.append(pred_em)
        preds_bw.append(pred_bw)
        preds_real.append(pred_real)
    lab_real = get_3labels(np.concatenate(lab_real))
    lab_em = get_3labels(np.concatenate(lab_em))
    lab_bw = get_3labels(np.concatenate(lab_bw))
    preds_real = np.concatenate(preds_real)
    preds_em = np.concatenate(preds_em)
    preds_bw = np.concatenate(preds_bw)
    all_ecg_real = np.concatenate(all_ecg_real)
    all_ecg_em = np.concatenate(all_ecg_em)
    all_ecg_bw = np.concatenate(all_ecg_bw)

    mae_per_test = np.stack([MAE(all_ecg_real, all_ecg_em[:, k]) for k in range(all_ecg_em.shape[1])], axis=1).mean(
        axis=1)
    rmse_per_test = np.stack([RMSE(all_ecg_real, all_ecg_em[:, k]) for k in range(all_ecg_em.shape[1])], axis=1).mean(
        axis=1)
    CI_mae = stats.t.interval(0.95, len(mae_per_test) - 1, loc=mae_per_test.mean(), scale=stats.sem(mae_per_test))
    CI_rmse = stats.t.interval(0.95, len(rmse_per_test) - 1, loc=rmse_per_test.mean(), scale=stats.sem(rmse_per_test))
    print('MAE', mae_per_test.mean(), '+/-', abs(CI_mae[1] - CI_mae[0]) / 2.)
    print('RMSE', rmse_per_test.mean(), '+/-', abs(CI_rmse[1] - CI_rmse[0]) / 2.)
    #cos_per_test = cosine_similarity(all_ecg_real, all_ecg_em.mean(axis=1))
    #CI_cos = stats.t.interval(0.95, len(cos_per_test) - 1, loc=cos_per_test.mean(), scale=stats.sem(cos_per_test))
    #print('Cos.', cos_per_test.mean(), '+/-', abs(CI_cos[1] - CI_cos[0]) / 2.)
    y_pred_tmp = all_ecg_em.mean(axis=1)
    sdtw = SoftDTW(use_cuda=False, gamma=0.1)
    dtw_per_test = sdtw(torch.tensor(all_ecg_real), torch.tensor(y_pred_tmp)).numpy() / 1024
    CI_dtw = stats.t.interval(0.95, len(dtw_per_test) - 1, loc=dtw_per_test.mean(), scale=stats.sem(dtw_per_test))
    print('DTW', dtw_per_test.mean(), '+/-', abs(CI_dtw[1] - CI_dtw[0]) / 2.)

    mae_per_test = np.stack([MAE(all_ecg_real, all_ecg_bw[:, k]) for k in range(all_ecg_bw.shape[1])], axis=1).mean(
        axis=1)
    rmse_per_test = np.stack([RMSE(all_ecg_real, all_ecg_bw[:, k]) for k in range(all_ecg_bw.shape[1])], axis=1).mean(
        axis=1)
    CI_mae = stats.t.interval(0.95, len(mae_per_test) - 1, loc=mae_per_test.mean(), scale=stats.sem(mae_per_test))
    CI_rmse = stats.t.interval(0.95, len(rmse_per_test) - 1, loc=rmse_per_test.mean(), scale=stats.sem(rmse_per_test))
    print('MAE', mae_per_test.mean(), '+/-', abs(CI_mae[1] - CI_mae[0]) / 2.)
    print('RMSE', rmse_per_test.mean(), '+/-', abs(CI_rmse[1] - CI_rmse[0]) / 2.)
    #cos_per_test = cosine_similarity(all_ecg_real, all_ecg_em.mean(axis=1))
    #CI_cos = stats.t.interval(0.95, len(cos_per_test) - 1, loc=cos_per_test.mean(), scale=stats.sem(cos_per_test))
    #print('Cos.', cos_per_test.mean(), '+/-', abs(CI_cos[1] - CI_cos[0]) / 2.)
    y_pred_tmp = all_ecg_bw.mean(axis=1)
    sdtw = SoftDTW(use_cuda=False, gamma=0.1)
    dtw_per_test = sdtw(torch.tensor(all_ecg_real), torch.tensor(y_pred_tmp)).numpy() / 1024
    CI_dtw = stats.t.interval(0.95, len(dtw_per_test) - 1, loc=dtw_per_test.mean(), scale=stats.sem(dtw_per_test))
    print('DTW', dtw_per_test.mean(), '+/-', abs(CI_dtw[1] - CI_dtw[0]) / 2.)

    pred_dic = {'clean': (lab_real, preds_real), 'em': (lab_em, preds_em), 'bw': (lab_bw, preds_bw)}
    for noise_type, (labs, preds) in pred_dic.items():
        print(f'####### noise type {noise_type}. ######## ')
        total_score = 0
        for k, lab in enumerate(['NSR', 'BBB','AF']):
            score = average_precision_score(labs[:, k], preds[:, k])
            print(f'avgPrecision {lab} = {score:.4f}')
            total_score += score
        total_score /= 3
        print(f'## avgPrecision total = {total_score:.4f} ##')
        print(' ')
        #auc_val = evaluation_scores(y_val, y_val_pred, score_fn=auc_roc_score)
        total_score = 0
        for k, lab in enumerate(['NSR', 'LBBB', 'RBBB','AF']):
            score = average_precision_score(labs[:, k], preds[:, k])
            print(f'auc {lab} = {score:.4f}')
            total_score += score
        total_score /= 4.
        print(f'## auc total = {total_score:.4f} ##')
        print(' ')
        opt_th = find_optimal_threshold(labs, preds)
        print(noise_type, opt_th)
        #opt_th = np.array([0.999531, 0.999464, 0.999384, 0.997203])
        #opt_th = [0.5]*4
        total_score = 0
        for k, lab in enumerate(['NSR', 'LBBB', 'RBBB','AF']):
            score = f1_score(labs[:, k], preds[:, k]>opt_th[k])
            total_score += score
            print(f'F1 {lab} = {score:.4f}')
        total_score /= 4.
        print(f'## F1 total = {total_score:.4f} ##')
        print(' ')
        print(' ')

    th = 0.1
    for lab_id, lab_name in zip([46, 61, 12, 11, 59, 4],
                                ['SR', 'Normal', 'RBBB', 'LBBB', 'SB', 'AF']):
        th = opt_th_real[lab_id]
        print(lab_name, th)
        real_score = f1_score(lab_real[:, lab_id].astype(int),
                                             (preds_real[:, lab_id] > th).astype(int))
        bw_score = f1_score(lab_bw[:, lab_id].astype(int),
                                           (preds_bw[:, lab_id] > th).astype(int))
        em_score = f1_score(lab_em[:, lab_id].astype(int),
                                           (preds_em[:, lab_id] > th).astype(int))
        print(f'{lab_name}: real={real_score:.3f}, bw={bw_score:.3f}, em_score={em_score:.3f}')
    print(' ')

    results_path = os.path.join(output_directory, f'{dataset}_lead1')
    prefix_lst = ['VDPS_all_GOOD', 'VDPS_seed0_50diff', 'DPS_seed0', 'PGDM_seed0', 'DDNM_seed0_GOOD', 'diffpir_seed0_GOOD', 'reddiff_seed0']
    title_lst = ['MGPS$_{300}$', 'MGPS$_{50}$', 'DPS', 'PGDM', 'DDNM', 'diffpir', 'reddiff']
    for prefix, title_plt in zip(prefix_lst[1:], title_lst[1:]):  # VDPS_all_GOOD
        print(f'# ================== {prefix} ============== # ')
        npz = np.load(os.path.join(results_path, f'{prefix}.npz')) # DDPM_seed0_GOOD.npz'))  # reddiff_seed0.npz'))
        print('ok')
        T = 1000 # int(min(10*cfg.dataset.sampling_rate, cfg.dataset.segment_length))
        n_data = 2198
        labels = npz['label'][:n_data]
        y = get_12leads(npz['real'][:n_data, :, :T])
        y_hat = npz['generated']
        if len(y_hat.shape) == 5:
            y_hat = np.concatenate(y_hat)
        print(y_hat.shape)

        y_hat = y_hat[:n_data, :, :, :T]
        print(y_hat.shape)
        mask = get_12leads_mask(npz['mask'][:n_data, :, :T])
        mae_per_test = np.stack([MAE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
        rmse_per_test = np.stack([RMSE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
        CI_mae = stats.t.interval(0.95, len(mae_per_test)-1, loc=mae_per_test.mean(), scale=stats.sem(mae_per_test))
        CI_rmse = stats.t.interval(0.95, len(rmse_per_test)-1, loc=rmse_per_test.mean(), scale=stats.sem(rmse_per_test))
        print('MAE', mae_per_test.mean(), '+/-', abs(CI_mae[1]-CI_mae[0])/2.)
        print('RMSE', rmse_per_test.mean(), '+/-', abs(CI_rmse[1]-CI_rmse[0])/2.)

        # === Downstream classification from ptbxl === #
        mean_train = 2.9521857e-06  # For trainin gon ptbxl
        std_train = 0.2095087

        pred_scores = []
        for k in range(y_hat.shape[1]):
            X_test_gen = np.swapaxes(get_12leads(y_hat[:, k]), 1, 2)
            pred_scores.append(model.predict((X_test_gen-mean_train) / std_train))
            # auc = evaluation_scores(labels, pred_labs)
        pred_scores_r = model.predict(np.swapaxes((y-mean_train) / std_train, 1, 2))
        # opt_prec, opt_recall, opt_th = get_optimal_precision_recall(labels.astype(int), pred_scores_r)

        pred_scores = np.stack(pred_scores, axis=1)
        # for lab_id, lab_name in zip([0, 12, 11, 59, 4, 47, 67, 54, 58], ['1AVB', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'TAB', 'PVC', 'SA']):
        for lab_id, lab_name in zip([12, 11, 59, 4],
                                        ['RBBB', 'LBBB', 'SB', 'AF']):
            pred_score = balanced_accuracy_score(labels[:, lab_id].astype(int), (pred_scores.mean(axis=1)[:, lab_id] > opt_th[lab_id]).astype(int))
            real_score = balanced_accuracy_score(labels[:, lab_id].astype(int),
                                             (pred_scores_r[:, lab_id] > opt_th[lab_id]).astype(int))
            print(f'{lab_name}: real={real_score:.3f}, pred={pred_score:.3f}')
        print(' ')

        plot_ids = [0, 140, 28, 82, 42]  # 128, 408, 198]  #, 408, 198]
        # sample_num = [0, 0, 0, 0, 0]  # 6, 5]  #, 4, 1]
        plot_labs = ['NSR', 'RBBB', 'LBBB', 'SB', 'AF']
        lab_ids = [-1, 12, 11, 59, 4]
        # RBBB [12 for ptbxl], 140, 318, 26  # id_ = 15 for codetest
        # LBBB [11 for ptbxl], 26
        # AF [4 for ptbxl], 42

        for id_, lab, lab_id_ in zip(plot_ids, plot_labs, lab_ids):
            if lab == 'NSR':
                s = 0
            else:
                s = pred_scores[id_, :, lab_id_].argmax()
                print(f'{lab}: {pred_scores[id_, :, lab_id_].max():.3f}')
            ecg_real, ecg_fake = y[id_], get_12leads(y_hat[id_])
            mask_plot = mask[id_]
            offset = 128
            mask_plot[3:6] = False

            fig = plot_12leads(ecg_real, ecg_fake[s],
                               np.percentile(ecg_fake, 5, axis=0),
                               np.percentile(ecg_fake, 95, axis=0),
                         mask_plot, offset_T=offset, plot_T=200, ticks=False, fs=16, title=f'{title_plt} ({lab})', top_=0.9)
            plt.show()
            fig.savefig(f'/mnt/data/lisa/papier_figures_new/lead1_{dataset}{lab}_{prefix}.pdf')
            fig.savefig(f'/mnt/data/lisa/papier_figures_new/lead1_{dataset}{lab}_{prefix}.png')

        with torch.no_grad():
            display_time_series(torch.tensor(get_12leads(y_hat[0, 0].T).T), gt=torch.tensor(y[0]))

if __name__ == '__main__':
    main()

