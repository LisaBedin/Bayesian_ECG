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
from diffusion_prior.benchmarks.ptbxl_strodoff.fastai_model import fastai_model
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support, recall_score
import torch.nn as nn
from keras.src.saving.saving_api import load_model as keras_load_model
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import resample_poly


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


def plot_12leads(ecg, ecg_pred, ecg_pred25, ecg_pred75, mask, plot_T, offset_T):
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
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

    # fpath = Path(mpl.get_data_path(), '/mnt/data/lisa/times.ttf')
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
        ax.annotate(ann, (((lW-1)*T+lW*margin_W*2) / 2.54, (1.5*offset_H + max_H*(lH-1)) / 2.54), fontsize=11)

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
                fontsize=11)

    return fig




def main():
    output_directory = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L256/waveforms/QTd_denoising'
    # dataset = PhysionetECG(**cfg.dataset)
    dataset = 'QTd'  # 'ptbxl'  #  'codetest'  # ptbxl
    noise_type = 'em'
    prefix = '2l' # '1l
    # npz = np.load(os.path.join(output_directory, f'denoising_{noise_type}_{dataset}_{prefix}_seed0.npz'))
    npz = np.load(os.path.join(output_directory, 'sel123_em_2l_seed0.npz'))
    print('ok')
    T = 1024 # int(min(10*cfg.dataset.sampling_rate, cfg.dataset.segment_length))
    labels = npz['label']
    y = get_12leads(npz['real'])
    y_hat = np.concatenate(npz['generated'], axis=-1)  # [..., :T]
    mae_per_test = np.stack([MAE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
    rmse_per_test = np.stack([RMSE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
    CI_mae = stats.t.interval(0.95, len(mae_per_test)-1, loc=mae_per_test.mean(), scale=stats.sem(mae_per_test))
    CI_rmse = stats.t.interval(0.95, len(rmse_per_test)-1, loc=rmse_per_test.mean(), scale=stats.sem(rmse_per_test))
    print('MAE', mae_per_test.mean(), '+/-', abs(CI_mae[1]-CI_mae[0])/2.)
    print('RMSE', rmse_per_test.mean(), '+/-', abs(CI_rmse[1]-CI_rmse[0])/2.)

    plot_ids = [0, 128]  #, 408, 198]
    sample_num = [6, 5]  #, 4, 1]
    plot_labs = ['NSR', 'RBBB', 'AF', 'TAb']
    # RBBB [12 for ptbxl], 140, 318, 26  # id_ = 15 for codetest
    # LBBB [11 for ptbxl], 26
    # AF [4 for ptbxl], 42
    id_ = 0
    s = 0
    for id_, s, lab in zip(plot_ids, sample_num, plot_labs):
        ecg_real, ecg_fake = y[id_], get_12leads(y_hat[id_])
        mask_plot = mask[id_]
        offset = 128
        mask_plot[3:6] = False

        fig = plot_12leads(ecg_real, ecg_fake[s],
                           np.percentile(ecg_fake, 5, axis=0),
                           np.percentile(ecg_fake, 95, axis=0),
                     mask_plot, offset_T=offset, plot_T=200)
        # plt.show()
        fig.savefig(f'/mnt/data/lisa/papier_figures/lead1_{dataset}{lab}.pdf')
        fig.savefig(f'/mnt/data/lisa/papier_figures/lead1_{dataset}{lab}.png')
        plt.show()

    with torch.no_grad():
        display_time_series(torch.tensor(get_12leads(y_hat[0, 0].T).T), gt=torch.tensor(y[0]))

    path_to_model = '/mnt/Reseau/Signal/lisa/ECG_inverse_problems_benchmark_data/inpainting_anomaly_detection/model_evals/model_ecg_diagnosis/model.hdf5'
    model = keras_load_model(path_to_model, compile=False)
    x_test = np.array([np.array([get_12leads(prepare_for_ribeiro(y_hat[k, l].T)) for l in range(10)]) for k in range(y_hat.shape[0])])
    pred_gen = np.stack([model.predict(x_test[k], verbose=False) for k in tqdm(range(x_test.shape[0]))])
    pred_real = model.predict(np.stack([prepare_for_ribeiro(y[k].T) for k in range(y.shape[0])]))

    if dataset == 'ptbxl':
        lab_names = ['RBBB', 'LBBB', 'AF']
        pred_gen = pred_gen[:, :, np.array([1, 2, 4]).astype(int)]
        pred_real = pred_real[:, np.array([1, 2, 4]).astype(int)]
        real_labs = labels[:, np.array([12, 11, 4]).astype(int)]  # RBBB, LBBB, AF
    else:
        lab_names = ['1AVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
        # 1st degree AV block(1dAVb), right bundle branch block (RBBB), left bundle branch block (LBBB), sinus bradycardia (SB), atrial fibrillation (AF), sinus tachycardia (ST)
        real_labs = labels  # [:, np.array([1, 2, 4]).astype(int)]
    opt_prec, opt_recall, opt_th = get_optimal_precision_recall(real_labs, pred_real)

    # RBBB_gen = [f1_score(real_labs[:, 0].astype(int), (pred_gen[:, k, 0] > opt_th[0]).astype(int)) for k in range(pred_gen.shape[1])]
    # LBBB_gen = [f1_score(real_labs[:, 1].astype(int), (pred_gen[:, k, 1] > opt_th[1]).astype(int)) for k in range(pred_gen.shape[1])]
    # AF_gen = [f1_score(real_labs[:, 2].astype(int), (pred_gen[:, k, 2] > opt_th[2]).astype(int))for k in range(pred_gen.shape[1])]

    gen_scores = [f1_score(real_labs[:, k], (pred_gen[:, :, k].mean(axis=1) > opt_th[k]).astype(int)) for k in range(real_labs.shape[1])]
    real_scores = [f1_score(real_labs[:, k], (pred_real[:, k] > opt_th[k]).astype(int)) for k in range(real_labs.shape[1])]
    for k in range(real_labs.shape[1]):
        print(f'{lab_names[k]}: real={real_scores[k]:.3f}, pred={gen_scores[k]:.3f}')
    # RBBB_gen = [f1_score(real_labs[:, 0].astype(int), (pred_gen[:, :, 0].mean(axis=1) > opt_th[0]).astype(int))]
    # LBBB_gen = [f1_score(real_labs[:, 1].astype(int), (pred_gen[:, :, 1].mean(axis=1) > opt_th[1]).astype(int))]
    # AF_gen = [f1_score(real_labs[:, 2].astype(int), (pred_gen[:, :, 2].mean(axis=1) > opt_th[2]).astype(int))]

    # RBBB_real = f1_score(real_labs[:, 0].astype(int), (pred_real[:, 0] > opt_th[0]).astype(int))
    # LBBB_real = f1_score(real_labs[:, 1].astype(int), (pred_real[:, 1] > opt_th[1]).astype(int))
    # AF_real = f1_score(real_labs[:, 2].astype(int), (pred_real[:, 2] > opt_th[2]).astype(int))
    # print(f'real: LBBB={LBBB_real:.4f} RBBB={RBBB_real:.4f} AF={AF_real:.4f}')
    # print(f'pred: LBBB={np.mean(LBBB_gen):.4f} RBBB={np.mean(RBBB_gen):.4f} AF={np.mean(AF_gen):.4f}')
    # print('ok')


    # === Downstream classification from ptbxl === #
    mean_train = 2.9521857e-06  # For trainin gon ptbxl
    std_train = 0.2095087
    #mean_train = 2.0920837e-05 # for georgia 9D
    # std_train = 0.20492806
    model_path = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L1024_cond/waveforms/generated_samples/ptbxlStrodoff_real'
    model = fastai_model(
        'fastai_xresnet1d50',
        71,
        100,
        outputfolder=model_path,
        input_shape=[1000, 12],
        pretrainedfolder=os.path.join(model_path, 'models/fastai_xresnet1d50.pth'),
        n_classes_pretrained=71,
        pretrained=True,
        epochs_finetuning=0,
        aggregate_fn='mean',
        bs=64,
        epochs=0,
        lr=1e-3,
        wd=1e-3,
    )

    pred_scores = []
    for k in range(y_hat.shape[1]):
        X_test_gen = np.swapaxes(get_12leads(y_hat[:, k]), 1, 2)
        pred_scores.append(model.predict((X_test_gen-mean_train) / std_train))
        # auc = evaluation_scores(labels, pred_labs)
    pred_scores_r = model.predict(np.swapaxes((y-mean_train) / std_train, 1, 2))
    opt_prec, opt_recall, opt_th = get_optimal_precision_recall(labels.astype(int), pred_scores_r)

    RBBB_gen = [f1_score(labels[:, 12].astype(int), (pred_labs[:, 12] > opt_th[12]).astype(int)) for pred_labs in pred_scores]
    LBBB_gen = [f1_score(labels[:, 11].astype(int), (pred_labs[:, 11] > opt_th[12]).astype(int)) for pred_labs
                in pred_scores]
    AF_gen = [f1_score(labels[:, 4].astype(int), (pred_labs[:, 4] > 0.5).astype(int)) for pred_labs in pred_scores]

    RBBB_real = f1_score(labels[:, 12].astype(int), (pred_scores_r[:, 12] > opt_th[12]).astype(int))
    LBBB_real = f1_score(labels[:, 11].astype(int), (pred_scores_r[:, 11] > opt_th[12]).astype(int))
    AF_real = f1_score(labels[:, 4].astype(int), (pred_scores_r[:, 4] > opt_th[12]).astype(int))

    print(f'real: LBBB={LBBB_real:.4f} RBBB={RBBB_real:.4f} AF={AF_real:.4f}')
    print(f'pred: LBBB={np.mean(LBBB_gen):.4f} RBBB={np.mean(RBBB_gen):.4f} AF={np.mean(AF_gen):.4f}')
    print('ok')


if __name__ == '__main__':
    main()

