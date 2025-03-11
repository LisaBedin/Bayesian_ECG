from DeScoD.Data_Preparation.data_preparation import Data_Preparation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from DeScoD.main_model import DDPM
from DeScoD.denoising_model_small import ConditionalModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y))
    D = np.sum(np.square(y_pred - np.mean(y)))

    PRD = np.sqrt(N/D) * 100

    return PRD


if __name__ == "__main__": 
    shot_sg = [1, 10]
    device = 'cuda:0'
    
    nts = ['em', 'bw']  # [1,2]

    leads_dic = {
        'sel123': np.array([1, 7]).astype(int),  # ['MLII', 'V5']
        'sel233': np.array([1, 3]).astype(int),  # ['MLII', 'V1']
        'sele0106': np.array([2, 5]).astype(int),   # ['D3', 'V3']
        'sele0121': np.array([6, 2]).astype(int),   # ['V4', 'D3']
    }

    for shots in shot_sg:

        for n_type in nts:
            mse_total = []
            mad_total = []
            prd_total = []
            cos_sim_total = []
            n_level = []

            path = "config/base.yaml"
            with open(path, "r") as f:
                config = yaml.safe_load(f)    
            
            base_model = ConditionalModel(config['train']['feats']).to('cuda:0')
            model = DDPM(base_model, config, 'cuda:0')
            foldername = "./check_points/noise_type_" + str(2) + "/"
            output_path = foldername + "/model.pth"
            
            model.load_state_dict(torch.load(output_path))
            model.eval()
            
            [_, _, X_test, y_test, padding_test] = Data_Preparation(noise_type=n_type)
            input_signals, signal_name_lst, target_signals, pred_signals = [], [], [], []
            for signal_name in X_test.keys():
                noisy_batch = torch.Tensor(np.swapaxes(X_test[signal_name], 1, 2)).cuda()
                clean_batch = np.swapaxes(y_test[signal_name], 1, 2)
                pad_lst = padding_test[signal_name]
                if shots > 1:
                    output = 0
                    for i in range(shots):
                        output+=model.denoising(noisy_batch[:, :1])
                    output /= shots
                else:
                    output = model.denoising(noisy_batch[:, :1])
                output = np.concatenate([output[k, 0, 16:16+pad_lst[k]].detach().cpu() for k in range(len(output))])
                clean_batch = np.concatenate([clean_batch[k, :, 16:16+pad_lst[k]] for k in range(len(clean_batch))], axis=-1)
                noisy_batch = np.concatenate([noisy_batch[k, :, 16:16+pad_lst[k]].detach().cpu() for k in range(len(noisy_batch))], axis=-1)
                output = output[:int(1024*360/100)]
                clean_batch = clean_batch[:, :int(1024*360/100)]
                noisy_batch = noisy_batch[:, :int(1024*360/100)]
                mean_clean = clean_batch.mean(axis=1)[:, None]
                output -= mean_clean[0]
                clean_batch -= mean_clean
                noisy_batch -= mean_clean
                input_signals.append(noisy_batch)
                signal_name_lst.append(signal_name)
                target_signals.append(clean_batch)
                pred_signals.append(output)
                MSE_score = np.mean((output - clean_batch[0]) ** 2)
                MAD_score = np.max(np.abs(output - clean_batch[0]))
                PRD_score = PRD(clean_batch[0], output)
                COS_score = cosine_similarity(clean_batch[:1, :], output[None, :])[0][0]

                print(f'{signal_name}: MSE={MSE_score:.4f}, MAD={MAD_score:.4f}, PRD={PRD_score:.4f}, COS={COS_score:.4f}')
                mse_total.append(MSE_score)
                mad_total.append(MAD_score)
                prd_total.append(PRD_score)
                cos_sim_total.append(COS_score)
            print(' ')
            CI_mse = stats.t.interval(0.95, len(mse_total) - 1, loc=np.mean(mse_total),
                                      scale=stats.sem(mse_total))
            print(f'MSE={np.mean(mse_total):.4f}+/-{(CI_mse[1]-CI_mse[0]) / 2.:.4f}')

            CI_mad = stats.t.interval(0.95, len(mad_total) - 1, loc=np.mean(mad_total),
                                      scale=stats.sem(mad_total))
            print(f'MAD={np.mean(mad_total):.4f}+/-{(CI_mad[1]-CI_mad[0]) / 2.:.4f}')

            CI_prd = stats.t.interval(0.95, len(prd_total) - 1, loc=np.mean(prd_total),
                                      scale=stats.sem(prd_total))
            print(f'PRD={np.mean(prd_total):.4f}+/-{(CI_prd[1]-CI_prd[0]) / 2.:.4f}')

            CI_cos = stats.t.interval(0.95, len(cos_sim_total) - 1, loc=np.mean(cos_sim_total),
                                      scale=stats.sem(cos_sim_total))
            print(f'COS_SIM={np.mean(cos_sim_total):.4f}+/-{(CI_cos[1]-CI_cos[0]) / 2.:.4f}')
            np.savez(f'/mnt/data/lisa/ecg_results/denoising/{n_type}_DeScoD{shots}.npz',
                     noisy=input_signals,
                     signal_name=signal_name_lst,
                     target=target_signals,
                     prediction=pred_signals
                     )
