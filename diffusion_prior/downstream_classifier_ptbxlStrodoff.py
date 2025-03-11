# inspired from https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/Finetuning-Example.ipynb
import os
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('.')

from benchmarks.ptbxl_strodoff.fastai_model import fastai_model
from utils import get_test_predictions, prepare_downstream_data


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
    cfg.model.unconditional = False
    cfg.dataset.training_class = 'train'
    # cfg.dataset.sampling_rate = 125
    labels, ecg_real, ecg_gen, output_directory = get_test_predictions(
        results_path=cfg.train.results_path,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset)
    cfg.dataset.training_class = 'test'
    labels_test, ecg_real_test, ecg_gen_test, output_directory = get_test_predictions(
        results_path=cfg.train.results_path,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset)

    output_directory = os.path.join(output_directory, f'ptbxlStrodoff_{cfg.evaluate.train_downstream}')
    os.makedirs(output_directory, exist_ok=True)

    if cfg.evaluate.train_downstream == 'gen':
        X_train, y_train = prepare_downstream_data(ecg_gen, labels,
                                                   segment_length=cfg.dataset.segment_length,
                                                   sampling_rate=cfg.dataset.sampling_rate)
    else:
        X_train, y_train = prepare_downstream_data(ecg_real, labels,
                                                   segment_length=cfg.dataset.segment_length,
                                                   sampling_rate=cfg.dataset.sampling_rate)
    mean_train, std_train = X_train.mean(), X_train.std()
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test_gen, y_test_gen = prepare_downstream_data(ecg_gen_test, labels_test,
                                                     segment_length=cfg.dataset.segment_length,
                                                     sampling_rate=cfg.dataset.sampling_rate)
    X_test_gen = (X_test_gen - mean_train) / std_train
    X_test_real, y_test_real = prepare_downstream_data(ecg_real_test, labels_test,
                                                       segment_length=cfg.dataset.segment_length,
                                                       sampling_rate=cfg.dataset.sampling_rate)
    X_test_real = (X_test_real - mean_train) / std_train

    # lab_counts = y_test_gen.sum(axis=0)
    # X_gen_disease = [X_test_gen[k] for k in range(X_test_gen.shape[0]) if y_test_gen[k][(lab_counts < 100000)*(lab_counts>900)].sum() > 0]
    # X_real_disease = [X_test_real[k] for k in range(X_test_gen.shape[0]) if y_test_gen[k][(lab_counts < 100000)*(lab_counts>900)].sum() > 0]
    # gen_plot = X_gen_disease[0]
    # gen_plot /= np.absolute(gen_plot).max(axis=0)[np.newaxis]
    # real_plot = X_real_disease[0]
    # real_plot /= np.absolute(real_plot).max(axis=0)[np.newaxis]
    # for k in range(10):
    #     gen_plot = X_test_gen[k]
    #     gen_plot /= np.absolute(gen_plot).max(axis=0)[np.newaxis]
    #     real_plot = X_test_real[k]
    #     real_plot /= np.absolute(real_plot).max(axis=0)[np.newaxis]
    #     display_time_series(torch.tensor(gen_plot[:512]).T, gt=None)
    #     display_time_series(torch.tensor(real_plot[:1].T), gt=torch.tensor(real_plot[:512]).T)
    # # strodoff model on 71 classes
    os.makedirs(os.path.join(output_directory, 'fastai_xresnet1d101/models'), exist_ok=True)
    model = fastai_model(
        'fastai_xresnet1d101',
        71,
        cfg.dataset.sampling_rate,
        outputfolder=os.path.join(output_directory, 'fastai_xresnet1d101/models'),
        input_shape=X_train.shape[1:],  # [1000, 12],
        #pretrainedfolder=os.path.join(cfg.evaluate.benchmark_folder, 'ptbxl_strodoff/fastai_xresnet1d101'),
        pretrainedfolder=os.path.join(cfg.evaluate.benchmark_folder, 'ptbxl_strodoff/fastai_xresnet1d101'),
        #pretrainedfolder=os.path.join(output_directory, 'models/fastai_xresnet1d50'),
        n_classes_pretrained=71,
        # pretrained=False,
        pretrained=True,
        # early_stopping="valid_loss",
        epochs_finetuning=50,  #50,
        aggregate_fn='mean',
        bs=64, # 64, # 64,
        epochs=50,
        lr=1e-3,
        wd=1e-3,  # 1e-3,
    )

    # == training the model from real data == #
    if cfg.evaluate.train_downstream == 'gen':
        model.fit(X_train, y_train, X_test_gen, y_test_gen)
    else:
        model.fit(X_train, y_train, X_test_real, y_test_real)

    # == load best model == #
    model = fastai_model(
        'fastai_xresnet1d101',
        71,
        cfg.dataset.sampling_rate,
        outputfolder=output_directory,
        input_shape=X_train.shape[1:],  # [1000, 12],
        pretrainedfolder=os.path.join(output_directory, 'models/fastai_xresnet1d101'),
        n_classes_pretrained=71,
        pretrained=True,
        epochs_finetuning=1,
        aggregate_fn='mean',
        bs=64,
        epochs=0,
        lr=1e-3,
        wd=1e-3,
    )

    print(f'=========== train on {cfg.evaluate.train_downstream}, evaluated on real samples ============')
    y_val_pred = model.predict(X_test_real)
    auc_real = evaluation_scores(y_test_real, y_val_pred)

    # precision, recall, thresholds = precision_recall_curve(y_test_real[:, 11], y_val_pred[:, 11], pos_label=1)
    RBBB_real = average_precision_score(y_test_real[:, 12], y_val_pred[:, 12])  # auc(precision, recall)  # roc_auc(y_test_real[:, 11], y_val_pred[:, 11])
    print('CRBBB auc real',  RBBB_real)
    print(f'=========== train on {cfg.evaluate.train_downstream}, evaluated on generated samples ============')
    y_val_pred = model.predict(X_test_gen)
    auc_gen = evaluation_scores(y_test_gen, y_val_pred)
    RBBB_gen = average_precision_score(y_test_real[:, 12], y_val_pred[:, 12])  #  auc(y_test_real[:, 11], y_val_pred[:, 11])

    print('CRBBB auc', RBBB_gen)
    print('auc real',  auc_real)
    print('auc gen',  auc_gen)

    df = pd.DataFrame({'data': ['all', 'RBBB'], 'real': [auc_real, RBBB_real], 'gen': [auc_gen, RBBB_gen]})
    print(df)
    df.to_csv(os.path.join(output_directory, 'auc.csv'))


if __name__ == '__main__':
    main()
