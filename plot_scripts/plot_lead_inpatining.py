import os
import hydra
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
from fastdtw import fastdtw
from plot_scripts.SoftDTW_functions import SoftDTW

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


#plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 14})


'''
plt.rcParams["font.serif"] = "cm"  # Use Computer Modern (LaTeX default font)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
colors = ["blue", "red"]
'''

def MAE(y, y_hat, mask):
    '''
    Args:
        y: B x T x L
        y_hat:  B x T x L
        mask:  B x T x L

    Returns: mae:  B
    '''
    mae = [np.absolute((y[k, ~mask[k]]-y_hat[k, ~mask[k]])).mean() for k in range(y.shape[0])]
    return np.array(mae)


def MaxAE(y, y_hat, mask):
    '''
    Args:
        y: B x T x L
        y_hat:  B x T x L
        mask:  B x T x L

    Returns: mae:  B
    '''
    mae = [np.absolute((y[k, ~mask[k]]-y_hat[k, ~mask[k]])).max(axis=-1).mean() for k in range(y.shape[0])]
    return np.array(mae)

def RMSE(y, y_hat, mask):
    rmse = [np.sqrt(((y[k, ~mask[k]]-y_hat[k, ~mask[k]])**2).mean()) for k in range(y.shape[0])]
    return np.array(rmse)


def cosine_similarity(y, y_hat, mask):
    cos_sim = []
    for k in range(y.shape[0]):
        y_true = y[k, ~mask[k]]
        y_pred = y_hat[k, ~mask[k]]
        dot_product = np.dot(y_true, y_pred)
        norm_y_true = np.linalg.norm(y_true)
        norm_y_pred = np.linalg.norm(y_pred)
        if norm_y_true == 0 or norm_y_pred == 0:
            cos_sim.append(0)  # Similarité cosinus est 0 si l'un des vecteurs est nul
        else:
            cos_sim.append(dot_product / (norm_y_true * norm_y_pred))
    return np.array(cos_sim)


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
                c='green', alpha=1., lw=1.,
                rasterized=True)
    for ecg_q25, ecg_q75, lW, lH in zip(ecg_pred25[:, offset_T:offset_T+plot_T],
                                        ecg_pred75[:, offset_T:offset_T+plot_T],
                                        ind_W, ind_H):
        ax.fill_between(((lW-1)*T+lW*margin_W + times) / 2.54,
                        (ecg_q25 + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                        (ecg_q75 + offset_H + max_H * (lH - 1) + lH * margin_H) / 2.54,
                color='gray', alpha=.4, lw=0,
                rasterized=True)

    lead_names = ['I', 'II', 'III'] + ['aVR', 'aVF', 'aVL'] + [f'V{k}' for k in range(7)]
    for ecg_l, lW, lH, ann in zip(ecg_pred[:, offset_T:offset_T+plot_T], ind_W, ind_H, lead_names):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='blue', alpha=1, lw=1.,
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
            c='green', alpha=1., lw=1.,
            rasterized=True)
    ax.fill_between((margin_W + times[kept_inds]) / 2.54, (offset_H + ecg_pred25II[kept_inds]) / 2.54,
                    (offset_H + ecg_pred75II[kept_inds]) / 2.54,
            color='gray', alpha=.4, lw=0.,
            rasterized=True)
    ax.plot((margin_W + times[kept_inds]) / 2.54, (ecg_predII[kept_inds] + offset_H) / 2.54,
            c='blue', alpha=1., lw=1.,
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
        #np.minimum(labs[:, 12]+labs[:, 11], 1), #LBBB or RBBB
        np.maximum(labs[:, 12], labs[:, 11]),
        labs[:, 4] # AF
       # labs[:, 59],
    ], axis=1)

def get_4labels(labs):
    return np.stack([
        labs[:, 46] * labs[:, 61],  # NSR
        #np.minimum(labs[:, 12]+labs[:, 11], 1), #LBBB or RBBB
        labs[:, 12],
        labs[:, 11],
        labs[:, 4] # AF
       # labs[:, 59],
    ], axis=1)

def dtw_similarity_old(y, y_hat, mask):
    dtw_scores = []
    for k in range(y.shape[0]):
        channel_dtw_scores = []
        for channel in range(y.shape[1]):
            y_true = y[k, channel, ~mask[k, channel]]
            y_pred = y_hat[k, channel, ~mask[k, channel]]
            distance, _ = fastdtw(y_true, y_pred)
            channel_dtw_scores.append(distance)
        dtw_scores.append(np.mean(channel_dtw_scores))
    return np.array(dtw_scores)

def dtw_similarity(y, y_hat, sdtw):
    dtw_scores = []
    for k in range(y.shape[0]):
        channel_dtw_scores = []
        for k in range(y.shape[1]):
            distance = sdtw(y, y_hat[k, :, :])
            channel_dtw_scores.append(distance)
        dtw_scores.append(np.mean(channel_dtw_scores))
    return np.array(dtw_scores)



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
        bs=128,
        epochs=0,
        lr=1e-3,
        wd=1e-3,
    )
    mean_train, std_train = (3.4904378480860032e-06, 0.2081562578678131)
    sdtw = SoftDTW(use_cuda=False, gamma=0.1)

    results_path = os.path.join(output_directory, f'{dataset}_lead1')
    prefix_lst = ['VDPS_all_GOOD', 'VDPS_seed0_50diff', 'DPS_seed0', 'PGDM_seed0', 'DDNM_seed0_GOOD', 'diffpir_seed0_GOOD', 'reddiff_seed0']
    title_lst = ['MGPS$_{300}$', 'MGPS$_{50}$', 'DPS', 'PGDM', 'DDNM', 'diffpir', 'reddiff']
    for prefix, title_plt in zip(prefix_lst[1:], title_lst[1:]):  # VDPS_all_GOOD
        print(f'# ================== {prefix} ============== # ')
        npz = np.load(os.path.join(results_path, f'{prefix}.npz')) # DDPM_seed0_GOOD.npz'))  # reddiff_seed0.npz'))
        print('ok')
        T = 1000 # int(min(10*cfg.dataset.sampling_rate, cfg.dataset.segment_length))
        labels = get_3labels(npz['label'])
        inds = np.where(labels.sum(axis=-1) > 0)[0]
        labels = labels[inds]
        y = get_12leads(npz['real'][inds, :, :T])
        y_hat = npz['generated'][inds]
        if len(y_hat.shape) == 5:
            y_hat = np.concatenate(y_hat)
        print(y_hat.shape)

        y_hat = y_hat[:, :, :, :T]
        print(y_hat.shape)
        mask = get_12leads_mask(npz['mask'][inds, :, :T])
        mae_per_test = np.stack([MAE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
        rmse_per_test = np.stack([RMSE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
        CI_mae = stats.t.interval(0.95, len(mae_per_test)-1, loc=mae_per_test.mean(), scale=stats.sem(mae_per_test))
        CI_rmse = stats.t.interval(0.95, len(rmse_per_test)-1, loc=rmse_per_test.mean(), scale=stats.sem(rmse_per_test))
        print('MAE', mae_per_test.mean(), '+/-', abs(CI_mae[1]-CI_mae[0])/2.)
        print('RMSE', rmse_per_test.mean(), '+/-', abs(CI_rmse[1]-CI_rmse[0])/2.)
        #cos_per_test = np.stack([cosine_similarity(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
        cos_per_test = cosine_similarity(y, get_12leads(np.swapaxes(y_hat, 1, 2)).mean(axis=2), mask)
        CI_cos = stats.t.interval(0.95, len(cos_per_test)-1, loc=cos_per_test.mean(), scale=stats.sem(cos_per_test))
        print('Cos.', cos_per_test.mean(), '+/-', abs(CI_cos[1]-CI_cos[0])/2.)
        y_pred_tmp = get_12leads(np.swapaxes(y_hat, 1, 2)).mean(axis=2)
        #y_pred_tmp /= np.absolute(y_pred_tmp).max(axis=1)[:, np.newaxis]
        #y_tmp = y / np.absolute(y).max(axis=1)[:, np.newaxis]
        #dtw_per_test = dtw_similarity(y_tmp, y_pred_tmp, mask)
        #dtw_per_test = np.stack([dtw_similarity(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
        #dtw_similarity(y, y_hat, sdtw)
        #dtw_per_test = dtw_similarity(y[:, :, None], y_pred_tmp[:, :, None], sdtw)
        dtw_per_test = sdtw(torch.tensor(y[:, 1:]), torch.tensor(y_pred_tmp[:, 1:])).numpy() / 1024
        CI_dtw = stats.t.interval(0.95, len(dtw_per_test)-1, loc=dtw_per_test.mean(), scale=stats.sem(dtw_per_test))
        print('DTW', dtw_per_test.mean(), '+/-', abs(CI_dtw[1]-CI_dtw[0])/2.)

        # === filter NSR, RBBB, LBBB, SB, AF === #

        # === Downstream classification from ptbxl === #
        pred_scores = []
        for k in range(y_hat.shape[1]):
            X_test_gen = np.swapaxes(get_12leads(y_hat[:, k]), 1, 2)
            pred_scores.append(model.predict((X_test_gen-mean_train) / std_train))
            # auc = evaluation_scores(labels, pred_labs)
        pred_scores_r = model.predict(np.swapaxes((y-mean_train) / std_train, 1, 2))
        # opt_prec, opt_recall, opt_th = get_optimal_precision_recall(labels.astype(int), pred_scores_r)

        pred_scores = np.stack(pred_scores, axis=1)

        for k, lab in zip([0, 1, 2], ['NSR', 'BBB', 'AF']):
            #if lab == 'BBB':
            #    tmp_lab = np.maximum(labels[:, 1], labels[:, 2])
            #    tmp_pred = np.median(pred_scores, axis=1)[:, 1:3].max(axis=1)
            #    score = average_precision_score(tmp_lab, tmp_pred)
            #else:
            score = average_precision_score(labels[:, k], np.median(pred_scores, axis=1)[:, k])
            print(f'avgPrecision pred {lab} = {score:.4f}')


        for k, lab in enumerate(['NSR', 'BBB', 'AF']):
            score = average_precision_score(labels[:, k], pred_scores_r[:, k])
            print(f"avgPrecision full {lab} = {score:.4f}")


        plot_ids = [0, 75, 16, 25]  # 128, 408, 198]  #s, 408, 198]
        # sample_num = [0, 0, 0, 0, 0]  # 6, 5]  #, 4, 1]
        plot_labs = ['NSR', 'RBBB', 'LBBB', 'AF']
        lab_ids = [0, 1, 2, 3]
        # RBBB [12 for ptbxl], 140, 318, 26  # id_ = 15 for codetest
        # LBBB [11 for ptbxl], 26
        # AF [4 for ptbxl], 42
        #for k, lab in enumerate(['NSR', 'LBBB', 'RBBB', 'AF']):
        #    print(lab, np.where(labels[:, k] == 1)[0][0:10])

        s_list = [0, 5, 9, 5]

        for id_, lab, lab_id_ in zip(plot_ids, plot_labs, lab_ids):
            #s = pred_scores[id_, :, lab_id_].argmax()
            s = s_list[lab_id_]
            print(f'{lab}: {pred_scores[id_, :, lab_id_].mean():.3f}')
            ecg_real, ecg_fake = y[id_], get_12leads(y_hat[id_])
            mask_plot = mask[id_]
            offset = 128
            mask_plot[3:6] = False

            fig = plot_12leads(ecg_real, ecg_fake[s],
                               np.percentile(ecg_fake, 5, axis=0),
                               np.percentile(ecg_fake, 95, axis=0),
                         mask_plot, offset_T=offset, plot_T=200, ticks=False, fs=16, title=f'{title_plt} ({lab})', top_=0.9)
            plt.show()
            fig.savefig(f'/mnt/data/lisa/papier_philTrans/lead1_{dataset}{lab}_{prefix}.pdf')
            fig.savefig(f'/mnt/data/lisa/papier_philTrans/lead1_{dataset}{lab}_{prefix}.png')

        with torch.no_grad():
            display_time_series(torch.tensor(get_12leads(y_hat[0, 0].T).T), gt=torch.tensor(y[0]))

if __name__ == '__main__':
    main()

