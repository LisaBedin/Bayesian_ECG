import os
import numpy as np
import torch
from posterior_samplers.utils import display_time_series
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def MAE(y, y_hat, mask): # mean absolute deviation
    '''
    Args:
        y: B x T x L
        y_hat:  B x T x L
        mask:  B x T x L

    Returns: mae:  B
    '''
    mae = [np.absolute((y[k, ~mask[k]]-y_hat[k, ~mask[k]])).mean() for k in range(y.shape[0])]
    return np.array(mae)


def RMSE(y, y_hat, mask):
    rmse = [np.sqrt(((y[k, ~mask[k]]-y_hat[k, ~mask[k]])**2).mean()) for k in range(y.shape[0])]
    return np.array(rmse)


# def SSD(y, y_hat, mask):
#     ssd = [((y[k, ~mask[k]]-y_hat[k, ~mask[k]])**2).sum(axis=-1).mean() for k in range(y.shape[0])]
#     return np.array(ssd)


def MAD(y, y_hat, mask):  # maximum aboslute deviation averaged over leads
    mad = []
    for l in range(y.shape[1]):
        if (1-mask[:, l]).sum() > 0:
            mad.append(np.array([(y[k, l, ~mask[k, l]]-y_hat[k, l, ~mask[k, l]]).max() for k in range(y.shape[0])]))
    mad = np.stack(mad, axis=1).mean(axis=1)
    return mad


def PRD_old(y, y_hat, mask):
    num, denom = [], []
    for l in range(y.shape[1]):
        if (1 - mask[:, l]).sum() > 0:
            num.append(np.array([((y[k, l, ~mask[k, l]]-y_hat[k, l, ~mask[k, l]])**2).sum() for k in range(y.shape[0])]))
            denom.append(np.array([((y[k, l, ~mask[k, l]].mean()-y[k, l, ~mask[k, l]])**2).sum() for k in range(y.shape[0])]))
    num = np.stack(num, axis=1)
    denom = np.stack(denom, axis=1)
    return np.mean(np.sqrt(num/denom)*100, axis=1)


def PRD(y, y_hat, mask):
    prd = [np.sqrt(np.nanmean((y[k, ~mask[k]]-y_hat[k, ~mask[k]])**2/(0.5*np.absolute(y[k, ~mask[k]])+0.5*np.absolute(y_hat[k, ~mask[k]])))) for k in range(y.shape[0])]
    return np.array(prd)


def CosSimBis(y, y_hat, mask):
    dotprod, y_norm, y_hat_norm = [], [], []
    for l in range(y.shape[1]):
        if (1 - mask[:, l]).sum() > 0:
            dotprod.append(np.array([(y[k, l, ~mask[k, l]]*y_hat[k, l, ~mask[k, l]]).sum() for k in range(y.shape[0])]))
            y_norm.append(np.array([np.sqrt(np.sum(y[k, l, ~mask[k, l]]**2)) for k in range(y.shape[0])]))
            y_hat_norm.append(np.array([np.sqrt(np.sum(y_hat[k, l, ~mask[k, l]]**2)) for k in range(y.shape[0])]))
    y_hat_norm = np.stack(y_hat_norm, axis=1)
    y_norm = np.stack(y_norm, axis=1)
    dotprod = np.stack(dotprod, axis=1)
    cosim = dotprod/(y_norm*y_hat_norm)  # ).mean(axis=-1)
    return cosim


def CosSim(y, y_hat, mask):
    y_norm = np.sqrt((y**2).sum(axis=(1, 2)))
    y_hat_norm = np.sqrt((y_hat**2).sum(axis=(1, 2)))
    dotprod = (y*y_hat).sum(axis=(1, 2))
    cosim = dotprod/(y_norm*y_hat_norm)  # ).mean(axis=-1)
    return cosim



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


def plot_12leads(ecg, ecg_pred, ecg_pred25, ecg_pred75, mask, ticks=False):
    max_H = 2.1   # 3.1  # cm = mV
    offset_H = 1  # 1.5
    margin_H = 0.
    margin_W = 0.2

    # T = ecg.shape[1]*4*2.5*0.001  # (*4 ms * 25 mm)
    T = ecg.shape[1]*10*2.5*0.001  # (*10 ms * 25 mm)
    times = np.linspace(0, T, ecg.shape[1])

    H = max_H*3 + margin_H*4  # in cm
    # W = T*4 + margin_W*5  # in cm
    W = T*3 + margin_W*4

    fig, ax = plt.subplots(figsize=(W/2.54, H/2.54))  # 1/2.54  # centimeters in inches
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

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
    # ind_W = [1]*3 + [2]*3 + [3]*3 + [4]*3
    ind_W = [1]*3 + [2]*3 + [3]*3
    # ind_H = np.concatenate([np.arange(3, 0, -1)]*4)
    ind_H = np.concatenate([np.arange(3, 0, -1)]*3)

    for mask_l, lW, lH in zip(mask, ind_W, ind_H):
        ax.fill_between(((lW-1)*T+lW*margin_W + times)[mask_l] / 2.54,
                        (max_H*(lH-1)+margin_W/2) / 2.54,
                        (lH*max_H-margin_W/2)/2.54,
                        alpha=0.2, color='gray', rasterized=True)

    for ecg_q25, ecg_q75, lW, lH in zip(ecg_pred25, ecg_pred75, ind_W, ind_H):
        ax.fill_between(((lW-1)*T+lW*margin_W + times) / 2.54,
                        (ecg_q25 + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                        (ecg_q75 + offset_H + max_H * (lH - 1) + lH * margin_H) / 2.54,
                color='green', alpha=.4, lw=0,
                rasterized=True)

    for ecg_l, lW, lH in zip(ecg, ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='royalblue', alpha=1., lw=1.,
                rasterized=True)
    for ecg_l, lW, lH, ann in zip(ecg_pred, ind_W, ind_H, ['I', 'II', 'III'] + [f'V{k}' for k in range(7)]):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c='darkorange', alpha=1, lw=1.,
                rasterized=True)
        ax.annotate(ann, (((lW-1)*T+lW*margin_W*2) / 2.54, (1.5*offset_H + max_H*(lH-1)) / 2.54), fontsize=16)
    return fig


def main():
    output_directory = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L256_cond/waveforms'
    # output_directory = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L1024/waveforms'

    for prefix in ['VDPS_seed0_vmaped_GOOD', 'DPS_seed0', 'PGDM_seed0', 'DDNM_seed0', 'diffpir_seed0', 'reddiff_seed0_1000', 'repaint_seed0']:
        print(f'# ================== {prefix} ============== # ')
        # for missingness_type in ['bm']:  # , 'mnr']:
        missingness_type = 'bm'  # 'bm'
        missingness = 50
        results_path = os.path.join(output_directory, f'ptbxl_{missingness_type}{missingness}')
        npz = np.load(os.path.join(results_path, f'{prefix}.npz')) #  'DPS_seed0.npz'))  # 'VDPS_seed0_vmaped_GOOD.npz'))  # reddiff_seed0.npz'))  # 'VDPS_seed0_vmaped.npz'))  #'seed0_GOOD.npz'))
        print('ok')
        n_data = 1728
        y9leads = npz['real']
        Tmax = y9leads.shape[-1]
        print(y9leads.shape)
        y9leads = y9leads[:n_data, :, :Tmax]
        mask9leads = npz['mask'][:n_data, :, :Tmax]
        y_hat = npz['generated']
        if len(y_hat.shape) == 5:
            y_hat = np.concatenate(y_hat)
        y_hat = y_hat[:n_data, :, :, :Tmax]
        y = get_12leads(y9leads)
        mask = get_12leads_mask(mask9leads)
        mae_per_test = np.stack([MAE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)  # mean absolute error
        rmse_per_test = np.stack([RMSE(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)  # Root Mean Squared Error
        mad_per_test = np.stack([MAD(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)  # max absolute dev
        prd_per_test = np.stack([PRD(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)  # Percentage root mean square diff
        cosim_per_test = np.stack([CosSim(y, get_12leads(y_hat[:, k]), mask) for k in range(y_hat.shape[1])], axis=1).mean(axis=1)  # Cosine Similarity

        CI_mae = stats.t.interval(0.95, len(mae_per_test)-1, loc=mae_per_test.mean(), scale=stats.sem(mae_per_test))
        CI_rmse = stats.t.interval(0.95, len(rmse_per_test)-1, loc=rmse_per_test.mean(), scale=stats.sem(rmse_per_test))
        CI_mad = stats.t.interval(0.95, len(mad_per_test)-1, loc=mad_per_test.mean(), scale=stats.sem(mad_per_test))
        CI_prd = stats.t.interval(0.95, len(prd_per_test)-1, loc=prd_per_test.mean(), scale=stats.sem(prd_per_test))
        CI_cosim = stats.t.interval(0.95, len(cosim_per_test)-1, loc=cosim_per_test.mean(), scale=stats.sem(cosim_per_test))
        print('MAE', mae_per_test.mean(), '+/-', abs(CI_mae[1]-CI_mae[0])/2.)
        print('RMSE', rmse_per_test.mean(), '+/-', abs(CI_rmse[1]-CI_rmse[0])/2.)
        print('MAD', mad_per_test.mean(), '+/-', abs(CI_mad[1]-CI_mad[0])/2.)
        print('PRD', prd_per_test.mean(), '+/-', abs(CI_prd[1]-CI_prd[0])/2.)
        print('cosim', cosim_per_test.mean(), '+/-', abs(CI_cosim[1]-CI_cosim[0])/2.)
        id_ = 0
        offset = 0
        T = 256
        # ecg_real, ecg_fake = y[id_], get_12leads(y_hat[id_])
        ecg_real, ecg_fake = y9leads[id_][:, offset:offset + T], y_hat[id_][..., offset:offset + T]
        ecg_pred25, ecg_pred75 = np.percentile(ecg_fake, 5, axis=0), np.percentile(ecg_fake, 95, axis=0)
        fig = plot_12leads(ecg_real, ecg_fake[0],
                     ecg_pred25, ecg_pred75, mask9leads[id_][:, offset:offset + T]
                     )
        plt.show()
        fig.savefig(f'/mnt/data/lisa/papier_figures_new/IMPUTATION_{missingness_type}_{prefix}.pdf')
        fig.savefig(f'/mnt/data/lisa/papier_figures_new/IMPUTATION_{missingness_type}_{prefix}.png')
        plt.show()
        with torch.no_grad():
            display_time_series(torch.tensor(get_12leads(y_hat[1, 0].T).T), gt=torch.tensor(y[1]))
        print(' ')


if __name__ == '__main__':
    main()

