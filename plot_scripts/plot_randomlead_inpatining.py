import os
import hydra
from omegaconf import DictConfig
import sys
sys.path.append('.')
sys.path.append('sashimi')
import numpy as np
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


def main():
    output_directory = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L1024/waveforms'
    # dataset = PhysionetECG(**cfg.dataset)
    results_path = os.path.join(output_directory, 'georgia_random_lead2')
    results_path = os.path.join(output_directory, 'ptbxl_lead1')
    npz = np.load(os.path.join(results_path, 'seed0_GOOD.npz'))
    print('ok')
    T = 1000 # int(min(10*cfg.dataset.sampling_rate, cfg.dataset.segment_length))
    y = get_12leads(npz['real'][..., :T])
    y_hat = npz['generated'][..., :T]
    y_test = npz['label']
    mask = get_12leads_mask(npz['mask'][..., :T])
    mae_per_test = np.stack([MAE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
    rmse_per_test = np.stack([RMSE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)
    CI_mae = stats.t.interval(0.95, len(mae_per_test)-1, loc=mae_per_test.mean(), scale=stats.sem(mae_per_test))
    CI_rmse = stats.t.interval(0.95, len(rmse_per_test)-1, loc=rmse_per_test.mean(), scale=stats.sem(rmse_per_test))
    print('MAE', mae_per_test.mean(), '+/-', abs(CI_mae[1]-CI_mae[0])/2.)
    print('RMSE', rmse_per_test.mean(), '+/-', abs(CI_rmse[1]-CI_rmse[0])/2.)
    with torch.no_grad():
        display_time_series(torch.tensor(y_hat[0]), gt=torch.tensor(y[0]))

    # === Downstream classification from ptbxl === #
    # mean_train = 2.9521857e-06 # For trainin gon ptbxl
    # std_train = 0.2095087
    #mean_train = 2.0920837e-05 # for georgia 9D
    # std_train = 0.20492806

    # (1.5690594216135936e-05, 0.17747307685423994) # georgai 12D
    X_test = torch.Tensor(get_12leads(npz['generated'][:, 0])).cuda()

    X_test_r = torch.Tensor(get_12leads(npz['real'])).cuda()
    # X_test = (X_test - mean_train) / std_train
    mdl = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L1024_allLimbs/waveforms'
    for i, lab in zip(np.arange(8), ['AF', 'TAb', 'QAb', 'VPB', 'LAD', 'SA', 'LBBB', 'RBBB']):
        cktp_path = os.path.join(mdl, f'georgia_ribeiro_BCEweight_{lab}_lr0.001_bs128')
        ckpt = torch.load(os.path.join(cktp_path, 'model.pth'))  # , map_location=lambda storage, loc: storage)
        # Get config
        config = os.path.join(cktp_path, 'args.json')
        with open(config, 'r') as f:
            config_dict = json.load(f)
        # Get model
        N_LEADS = 12
        model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                         blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                         n_classes=1,
                         kernel_size=config_dict['kernel_size'],
                         dropout_rate=config_dict['dropout_rate'])
        # load model checkpoint
        model.load_state_dict(ckpt["model"])
        model = model.cuda()
        model.eval()

        with torch.no_grad():
            y_scores = nn.Sigmoid()(model(X_test)).detach().cpu().numpy()
            y_scores_r = nn.Sigmoid()(model(X_test_r)).detach().cpu().numpy()

        _, _, opt_th = get_optimal_precision_recall(y_test[:, i:i+1], y_scores_r[:, None])
        opt_th = 0.5  #  opt_th[0]
        y_pred = (y_scores > opt_th).astype(int)
        y_pred_r = (y_scores_r > opt_th).astype(int)

        recall = recall_score(y_test[:, i], y_pred, pos_label=1)
        specificity = recall_score(y_test[:, i], y_pred, pos_label=0)
        #f1_s = f1_score(y_test[:, i], y_pred, average='micro')  # , pos_label=1)

        recall_r = recall_score(y_test[:, i], y_pred_r, pos_label=1)
        specificity_r = recall_score(y_test[:, i], y_pred_r, pos_label=0)
        #f1_s_r = f1_score(y_test[:, i], y_pred_r, average='micro')  # , pos_label=1)

        print(f'real {lab} gmean={np.sqrt(recall_r*specificity_r):.2f}, recall={recall_r:.2f}, specificity={specificity_r:.2f}')  # , f1_s={f1_s_r:.2f}')
        print(f'gen {lab} gmean={np.sqrt(recall*specificity):.2f},recall={recall:.2f}, specificity={specificity:.2f}')  # , f1_s={f1_s:.2f}')
        print('')
    print('ok')


if __name__ == '__main__':
    main()

