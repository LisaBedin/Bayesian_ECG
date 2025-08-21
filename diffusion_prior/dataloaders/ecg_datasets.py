## Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py

import os
import time
import timeit
import glob

from pandas.core._numba.kernels import mean_
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from scipy.signal import resample_poly
import wfdb
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import Tensor
from typing import List
from jax import random, jit
from sqlalchemy import create_engine, text
from nnresample import resample as nnresample_fn
import h5py
import math
#from benchmarks.ptbxl_strodoff.fastai_model import fastai_model


def read_memmap_from_position(filename,
                              shape,
                              beat_index):
    fp = np.memmap(filename=filename,
                   mode='r',
                   dtype='float16',
                   shape=shape,
                   offset=0)
    data = np.asarray(fp[:, beat_index])
    return data

class DownstreamEval(Dataset):
    def __init__(self, ecg, labels, segment_length=1024, sampling_rate=100, mean_ecg=None, std_ecg=None):
        self.ecg = ecg[:, :, :min(int(sampling_rate*10), segment_length)]
        if mean_ecg is not None and std_ecg is not None:
            self.mean_ecg = mean_ecg
            self.std_ecg = std_ecg
        else:
            self.mean_ecg = self.ecg.mean()
            self.std_ecg = self.ecg.std()
        self.ecg = (self.ecg - self.mean_ecg) / self.std_ecg
        self.labels = labels
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.ecg_shape = [ecg.shape[-1], 12]

    def __len__(self):
        return self.ecg.shape[0]

    def __getitem__(self, idx):
        ecg = self.ecg[idx].T  # T x 9
        aVR = -(ecg[:, 0] + ecg[:, 1]) / 2
        aVL = (ecg[:, 0] - ecg[:, 2]) / 2
        aVF = (ecg[:, 1] + ecg[:, 2]) / 2
        augm_leads = np.stack([aVR, aVL, aVF], axis=-1)
        ecg = np.concatenate([ecg, augm_leads], axis=-1)
        return ecg, self.labels[idx]


class PtbXLDataset(Dataset):
    def __init__(self, database_path, training_class='train',
                 all_limb=False, segment_length=1024, sampling_rate=100,
                 label_names=None,
                 noise=None,
                 # label_names=['AF', 'TAb', 'QAb', 'VPB', 'LAD', 'SA'],
                 **kwargs):
        self.label_names = label_names
        self.training_class = training_class
        data_path = os.path.join(database_path, f'processed_{sampling_rate}Hz/{training_class}')
        self.data_path = data_path
        self.labels = torch.tensor(np.load(os.path.join(data_path, 'labels.npy')))
        self.header = pd.read_csv(os.path.join(data_path, 'headers.csv'))
        self.lab_id_dic = {'NSR': [46, 61], 'RBBB': 12, 'LBBB': 11, 'SB': 59, 'AF': 4}
        if self.label_names is not None:
            kept_inds = [self.lab_id_dic[k] for k in self.label_names if k != 'NSR']
            if 'NSR' in self.label_names:
                all_inds = [np.unique(np.where(self.labels[:, 46]*self.labels[:, 61])[0])]
            else:
                all_inds = []
            for ind in kept_inds:
                all_inds.append(np.where(self.labels[:, ind])[0])
            all_inds = np.unique(np.concatenate(all_inds))
            #self.all_inds = all_inds
            self.labels = self.labels[all_inds]
            self.header = self.header.iloc[all_inds]
        self.noise = noise
        all_files = [os.path.join(data_path, filename_hr.split('/')[-1]+'.npz') for filename_hr in self.header['filename_hr'].tolist()]
        if noise is not None:
            n_data, n_channels = len(all_files), 12
            self.all_start_inds = np.random.randint(low=0, high=len(self.noise) -segment_length,
                                         size=(n_data, n_channels))
            self.rnd_train = np.random.randint(low=20, high=200, size=n_data) / 100

        self.data = torch.zeros((len(all_files), 12, int(sampling_rate*10)))
        for i, file_p in tqdm(enumerate(all_files), total=len(all_files)):
            with np.load(file_p) as npz:
                self.data[i] = torch.tensor(npz['ecg'])

        self.segment_length = segment_length
        self.all_limb = all_limb

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        ecg = self.data[idx]
        if self.noise is not None:
            start_ind = self.all_start_inds[idx]
            noise_ecg = torch.stack(
                [torch.tensor(self.noise).to(ecg.dtype)[start_ind[b]:start_ind[b] + ecg.shape[1]] for b in
                 range(ecg.shape[0])],
                dim=0)
            ecg_max = np.max(ecg.numpy(), axis=-1) - np.min(ecg.numpy(), axis=-1)
            noise_max_value = np.max(noise_ecg.numpy(), axis=-1) - np.min(noise_ecg.numpy(), axis=-1)
            Ase = torch.tensor(noise_max_value / ecg_max).to(ecg.dtype)
            noise_power = self.rnd_train[idx]
            noisy_ecg = ecg + noise_power * noise_ecg / Ase.unsqueeze(-1)
        T = ecg.shape[1]
        if T > self.segment_length:
            start = torch.randint(0, T-self.segment_length, size=(1,))[0]
            end = start + self.segment_length
            ecg = ecg[:, start:end]
            if self.noise is not None:
                noisy_ecg = noisy_ecg[:, start:end]
        if T < self.segment_length:  # padd with zeros
            pad = self.segment_length - T
            ecg = torch.cat([ecg, torch.zeros((ecg.shape[0], pad))], dim=1)
            if self.noise is not None:
                noisy_ecg = torch.cat([noisy_ecg, torch.zeros((noisy_ecg.shape[0], pad))], dim=1)
        if not self.all_limb:
            ecg = torch.cat([ecg[:3], ecg[6:]]).to(torch.float32)
            if self.noise is not None:
                noisy_ecg = torch.cat([noisy_ecg[:3], noisy_ecg[6:]]).to(torch.float32)
        if self.noise is not None:
            return ecg, noisy_ecg, self.labels[idx]
        return ecg, self.labels[idx]


def process_file(file_p):
    with np.load(file_p) as npz:
        ecg = npz['ecg']
    return torch.tensor(ecg)


class LargeDataset(Dataset):
    def __init__(self, ptbxl_path, mimic_path, model_path, training_class='train',
                 all_limb=False, segment_length=1024, sampling_rate=100,
                 **kwargs):
        self.training_class = training_class
        self.segment_length = segment_length
        self.all_limb = all_limb

        # == PTB-XL == #
        data_path = os.path.join(ptbxl_path, f'processed_{sampling_rate}Hz/{training_class}')
        self.label_ptb = torch.tensor(np.load(os.path.join(data_path, 'labels.npy')))
        self.header_ptb = pd.read_csv(os.path.join(data_path, 'headers.csv'))

        all_files = [os.path.join(data_path, filename_hr.split('/')[-1]+'.npz') for filename_hr in self.header_ptb['filename_hr'].tolist()]
        self.data_ptb = torch.zeros((len(all_files), 12, int(sampling_rate*10)))
        for i, file_p in tqdm(enumerate(all_files), total=len(all_files)):  # 30 seconds
            with np.load(file_p) as npz:
                self.data_ptb[i] = torch.tensor(npz['ecg'])

        # == pseudo-labeler == #
        model = fastai_model(
            'fastai_xresnet1d101',
            71,
            sampling_rate,
            outputfolder=model_path,
            input_shape=(int(sampling_rate*10), 12),  # [1000, 12],
            pretrainedfolder=os.path.join(model_path, 'models/fastai_xresnet1d101.pth'),
            n_classes_pretrained=71,
            pretrained=True,
            epochs_finetuning=0,
            aggregate_fn='max',
            bs=128,
            epochs=0,
            lr=1e-3,
            wd=1e-3,
        )
        # load the mean / std of the train set ?
        with open(os.path.join(model_path, 'mean.txt'), 'r') as file:
            line = file.readline().strip()
            mean_train = float(line)
        with open(os.path.join(model_path, 'std.txt'), 'r') as file:
            line = file.readline().strip()
            std_train = float(line)
        # == MIMIC-IV == #
        data_path = os.path.join(mimic_path, f'processed_{sampling_rate}Hz/{training_class}')
        all_files = glob.glob(os.path.join(data_path, f'*.npz'))[:10]

        # num_workers = 10 # min(cpu_count(), len(all_files))
        # print('num workers', num_workers)
        # with Pool(num_workers) as pool:
        #     self.data_mimic = list(tqdm(pool.map(process_file, tqdm(all_files, total=len(all_files)))))
        #
        # self.data_mimic = torch.cat(self.data_mimic)
        self.data_mimic = torch.zeros((int(len(all_files)*1000), 12, int(sampling_rate*10)))
        self.label_mimic = torch.zeros((int(len(all_files)*1000), self.label_ptb.shape[1]))
        count = 0
        for i, file_p in tqdm(enumerate(all_files), total=len(all_files)):
            with np.load(file_p) as npz: # 3 minutes roughly
                ecg = npz['ecg']
                # filter ecg with NaN values (ecg that were too noisy to be denoised during the preprocessing step)
                inds = np.prod(1-np.isnan(ecg[:, :, 0]).astype(int), axis=1).astype(bool)
                ecg = ecg[inds]
                import pdb
                pdb.set_trace()
                with torch.no_grad():
                    ecg_input = (np.swapaxes(ecg, 1, 2)-mean_train)/std_train
                    labs = model.predict(ecg_input)
                    # transform labs into 0 1 values. What is the threshold ?
                self.label_mimic[count:count+len(ecg)] = labs
                self.data_mimic[count:count+len(ecg)] = torch.tensor(ecg)
                count += len(ecg)
        self.data_mimic = self.data_mimic[:count]
        self.label_mimic = self.label_mimic[:count]
        # ======== 44G required ======= #


    def __len__(self):
        return max(self.data_ptb.shape[0], self.data_mimic.shape[0])

    def __getitem__(self, idx):
        ptb_data = bool(torch.randint(low=0, high=2, size=(1,)).item())
        if ptb_data:
            ecg = self.data_ptb[idx%self.data_ptb.shape[0]]
        else:
            ecg = self.data_mimic[idx%self.data_mimic.shape[0]]
        T = ecg.shape[1]
        if T > self.segment_length:
            start = torch.randint(0, T-self.segment_length, size=(1,))[0]
            end = start + self.segment_length
            ecg = ecg[:, start:end]
        if T < self.segment_length:  # padd with zeros
            pad = self.segment_length - T
            ecg = torch.cat([ecg, torch.zeros((ecg.shape[0], pad))], dim=1)
        if ptb_data:
            if self.all_limb:
                return ecg, self.label_ptb[idx]
            return torch.cat([ecg[:3], ecg[6:]]).to(torch.float32), self.label_ptb[idx%self.data_ptb.shape[0]]
        return torch.cat([ecg[:3], ecg[6:]]), torch.zeros_like(self.label_ptb[idx%self.data_ptb.shape[0]])


class ProbabilisticSampler(Sampler):
    def __init__(self, len_ptb, len_mimic, p_ptb=0.1, p_mimic=0.9, batch_size=32):
        self.len_ptb = len_ptb
        self.len_mimic = len_mimic
        self.p_ptb = p_ptb
        self.p_mimic = p_mimic
        self.batch_size = batch_size
        self.length = min(len_ptb, len_mimic)

    def __iter__(self):
        indices = []
        for _ in range(self.length):
            if random.random() < self.p1:
                indices.append(random.randint(0, self.len_ptb - 1))
            else:
                indices.append(random.randint(0, self.len_mimic - 1) + self.len_ptb)
        return iter(indices)

    def __len__(self):
        return self.length



class CodeTrainDataset(Dataset):
    def __init__(self, database_path, all_limb=False, segment_length=1024, sampling_rate=100, **kwargs):
        self.df = pd.read_csv(os.path.join(database_path, 'exams.csv'))
        all_h5py = glob.glob(os.path.join(database_path, 'exams*.hdf5'))
        self.ecg, self.ids = [], []
        for file_p in all_h5py:
            with h5py.File(file_p, 'r') as f_:
                self.ecg.append(np.array(f_['tracings']))
                self.ids.append(np.array(f_['exam_id']))
        self.ecg = np.concatenate(self.ecg, axis=0)[:, 48:4096-48]
        self.ids = np.concatenate(self.ids, axis=0).astype(int)
        self.labels = np.concatenate([self.get_lab(id_) for id_ in self.ids])

        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.all_limb = all_limb

    def __len__(self):
        return self.ecg.shape[0]

    def get_lab(self, id_):
        return np.array(self.df[self.df['exam_id'] == id_][['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']]).astype(int)


    def resamp(self, signal):
        newFs, oldFs = self.sampling_rate, 400
        L = math.ceil(len(signal) * newFs / oldFs)
        normBeat = list(reversed(signal)) + list(signal) + list(reversed(signal))

        # resample beat by beat and saving it
        res = resample_poly(np.array(normBeat), newFs, oldFs)  # , axis=0)
        res = res[L - 1:2 * L - 1]
        return res

    def __getitem__(self, idx):
        # ecg = torch.Tensor(nnresample_fn(self.ecg[idx].T, self.sampling_rate, 400, axis=1))
        ecg = torch.Tensor(self.resamp(self.ecg[idx]).T)
        ecg_final = torch.zeros((ecg.shape[0], self.segment_length))
        ecg_final[:, :ecg.shape[1]] = ecg
        labels = torch.Tensor(self.labels[idx].tolist())
        if self.all_limb:
            return ecg_final, labels
        return torch.cat([ecg_final[:3], ecg_final[6:]]), labels



class CodeTestDataset(Dataset):
    def __init__(self, database_path, all_limb=False, segment_length=1024, sampling_rate=100, **kwargs):
        self.df = pd.read_csv(os.path.join(database_path, 'annotations/gold_standard.csv'))
        f = h5py.File(os.path.join(database_path, 'ecg_tracings.hdf5'), 'r')
        self.ecg = np.array(f['tracings'])[:, 48:4096-48] # / 10
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.all_limb = all_limb

    def __len__(self):
        return self.ecg.shape[0]

    def resamp(self, signal):
        newFs, oldFs = self.sampling_rate, 400
        L = math.ceil(len(signal) * newFs / oldFs)
        normBeat = list(reversed(signal)) + list(signal) + list(reversed(signal))

        # resample beat by beat and saving it
        res = resample_poly(np.array(normBeat), newFs, oldFs)  # , axis=0)
        res = res[L - 1:2 * L - 1]
        return res

    def __getitem__(self, idx):
        # ecg = torch.Tensor(nnresample_fn(self.ecg[idx].T, self.sampling_rate, 400, axis=1))
        ecg = torch.Tensor(self.resamp(self.ecg[idx]).T)
        ecg_final = torch.zeros((ecg.shape[0], self.segment_length))
        ecg_final[:, :ecg.shape[1]] = ecg
        labels = torch.Tensor(self.df.iloc[idx].tolist())
        if self.all_limb:
            return ecg_final, labels
        return torch.cat([ecg_final[:3], ecg_final[6:]]), labels


class GeorgiaDataset(Dataset):
    def __init__(self, database_path, training_class='train',
                 all_limb=False, segment_length=1024, sampling_rate=100, label_names=['AF', 'TAb', 'QAb', 'VPB', 'LAD', 'SA'],  # ['IVAB', 'RBBB', 'LBBB', 'SB', 'AF', 'STach'],
                 return_weights=False, negative_class=False, **kwargs):
        self.data_path = os.path.join(database_path, f'georgia_{sampling_rate}Hz')
        engine = create_engine(f'sqlite:///{self.data_path}/database.db')
        self.training_class = training_class
        self.return_weights = return_weights

        with engine.connect() as conn:
            if 'train' in training_class:
                training_class = 'Training%'
                ids = conn.execute(text(
                    "select dataset_name, dataset_id, target_classes from records where partition_attribution like'" + training_class + "'")).fetchall()
            else:
                training_class = 'Test'
                ids = conn.execute(text(
                    "select dataset_name, dataset_id, target_classes from records where partition_attribution like'" + training_class + "'")).fetchall()
                training_class = 'CV'
                ids += conn.execute(text(
                    "select dataset_name, dataset_id, target_classes from records where partition_attribution like'" + training_class + "'")).fetchall()
        self.training_class = training_class
        self.segment_length = segment_length
        self.all_limb = all_limb
        self.ids = ids
        self.label_names = label_names

        all_labs = []
        for i, lab in enumerate(self.label_names):
            all_labs.append(np.array([lab in row[-1] for row in self.ids]))
        all_labs = np.stack(all_labs, axis=-1)
        self.negative_class = negative_class
        if self.negative_class:
            last_lab = (all_labs.sum(axis=-1) == 0).astype(int)[:, np.newaxis]
            all_labs = np.concatenate([all_labs, last_lab], axis=-1)
        self.labels = all_labs.astype(int)
        if self.return_weights:
            pos_w = self.labels.sum(axis=0)
            neg_w = (1-self.labels).sum(axis=0)
            pos_w = 1/pos_w
            neg_w = 1/neg_w
            pos_w /= (pos_w + neg_w)
            neg_w /= (pos_w + neg_w)
            self.weights = self.labels.astype(float)*pos_w + (1-self.labels.astype(float))*neg_w
        # self.weights = self.labels * weights[np.newaxis]
        # In our experiments we focus on the
        # following six common types of diagnosis:
        # AF - Atrial fibrillation;
        # TAb - T wave abnormal;
        # QAb - Q wave abnormal;
        # VPB - Ventricular premature beats;
        # LAD - Left axis deviation;
        # SA - Sinus arrhythmia.

        #  1st degree AV block(1dAVb), IAVB
        #  right bundle branch block (RBBB),
        #  left bundle branch block (LBBB),
        #  sinus bradycardia (SB),
        #  atrial fibrillation (AF),
        #  sinus tachycardia (ST). STach
        #  The abnormalities are not mutually exclusive, so the probabilities do not necessarily sum to one.

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        row = self.ids[idx]
        path_to_raw_data = os.path.join(self.data_path, row.dataset_name, f'{row.dataset_id}.npz')
        #lab_lst = row[-1].split('-')
        #labels = torch.tensor([lab in lab_lst for lab in self.label_names]).to(int)
        labels = torch.tensor(self.labels[idx])
        npz = np.load(path_to_raw_data)
        ecg = torch.tensor(npz['data'])
        T = ecg.shape[1]

        if T > self.segment_length:
            start = torch.randint(0, T-self.segment_length, size=(1,))[0]
            end = start + self.segment_length
            ecg = ecg[:, start:end]
        if T < self.segment_length:  # padd with zeros
            pad = self.segment_length - T
            ecg = torch.cat([ecg, torch.zeros((ecg.shape[0], pad))], dim=1)
        if self.all_limb:
            if self.return_weights:
                return ecg.to(torch.float32), labels, torch.tensor(self.weights[idx])
            return ecg.to(torch.float32), labels
        if self.return_weights:
            return torch.cat([ecg[:3], ecg[6:]]).to(torch.float32), labels, torch.tensor(self.weights[idx])
        return torch.cat([ecg[:3], ecg[6:]]).to(torch.float32), labels



class PhysionetECG(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, database_path: str,  # categories_to_filter: List[str],
                 segment_length: int = 2048,
                 normalized: str = 'no',
                 training_class: bool = 'Training',
                 dataset_names: List = ['ptb-xl'],
                 all: bool = False,
                 return_beat_id: bool = False,
                 all_limb=False,
                 **kwargs):
        engine = create_engine(f'sqlite:///{database_path}/database.db')
        with engine.connect() as conn:
            query_string = '(' + ' or '.join([f"dataset_name = '{i}'" for i in dataset_names]) + ')'
            if all:
                ids = conn.execute(text(
                    "select dataset_name, dataset_id, sex, age, target_classes from records where " + query_string + " and age IS NOT NULL and sex IS NOT NULL and target_classes is NOT NULL group by dataset_name, dataset_id, n_beats")).fetchall()
            elif 'Train' in training_class and 'CV' in training_class:
                training_class = 'Training%' # if training_class == 'Training' else training_class
                ids = conn.execute(text(
                    "select dataset_name, dataset_id, sex, age, target_classes from records where partition_attribution like '" + training_class + "' and  " + query_string + " and age IS NOT NULL and sex IS NOT NULL and target_classes is NOT NULL group by dataset_name, dataset_id, n_beats, sex, age, target_classes")).fetchall()
                training_class = 'CV'
                ids += conn.execute(text(
                    "select dataset_name, dataset_id, sex, age, target_classes from records where partition_attribution like '" + training_class + "' and  " + query_string + " and age IS NOT NULL and sex IS NOT NULL and target_classes is NOT NULL group by dataset_name, dataset_id, n_beats, sex, age, target_classes")).fetchall()
            else:
                training_class = 'Training%' if training_class == 'Training' else training_class
                ids = conn.execute(text(
                    "select dataset_name, dataset_id, sex, age, target_classes from records where partition_attribution like '" + training_class + "' and  " + query_string + " and age IS NOT NULL and sex IS NOT NULL and target_classes is NOT NULL group by dataset_name, dataset_id, n_beats, sex, age, target_classes")).fetchall()

        ignored_ids = ['E00841', 'E00906', 'E00915', 'E00947', 'E08668', 'E08686',
                       'E08744', 'E08767', 'E08822', 'E08876', 'E09904', 'E09923',
                       'E09965', 'E09980', 'E09983', 'E09988', 'E10002',  # training
                       'E08366', 'E08717']  # cv
        self._ids = [id for id in ids if not (id.dataset_id in ignored_ids and id.dataset_name == 'georgia')]

        self.database_path = database_path
        self.signal_length = segment_length
        # self.estimate_noise_std = estimate_noise_std
        self.normalized = normalized
        self.return_beat_id = return_beat_id
        self.all_limb = all_limb

    def _get_ecg(self, n:int) -> Tuple:
        row = self._ids[n]
        path_to_raw_data = os.path.join(self.database_path, row.dataset_name, f'{row.dataset_id}.npz')

        npz = np.load(path_to_raw_data)
        ecg = npz['data']
        T = ecg.shape[1]
        start = 0
        if T > self.signal_length:
            start = torch.randint(0, T-self.signal_length, size=(1,))[0]
            end = start + self.signal_length
            ecg = ecg[:, start:end]

        if not self.all_limb:
            ecg = np.concatenate([ecg[:3], ecg[6:]])
        ecg = ecg.astype(np.float16)
        if len(self.noise_path) > 0:
            # windowing
            #start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
            #                             size=(2,),replace=True)
            #leads = np.random.choice(a=ecg.shape[0], size=(2,), replace=False)
            start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
                                         size=(ecg.shape[0],),replace=True)
            leads = np.arange(ecg.shape[0])
            np.random.shuffle(leads)
            # corruption -> 2 leads une seule lead ?
            noise = np.stack([self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,0]
                              if k < ecg.shape[0] /2
                              else self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,1]
                              for k in range(ecg.shape[0])])
            beat_max_value = np.max(ecg, axis=1) - np.min(ecg, axis=1)
            noise_max_value = np.max(noise, axis=1) - np.min(noise, axis=1)
            Ase = noise_max_value / beat_max_value[leads]
            denoised_ecg = ecg.copy().T
            noise = noise/Ase[:, np.newaxis]*self.alphas[n]
            ecg[leads] += noise
        if self.normalized == 'per_lead':
            max_val = np.abs(ecg).max(axis=-1).clip(1e-4, 10)[..., None]
        elif self.normalized == 'global':
            max_val = np.abs(ecg).max().clip(1e-4, 10)
        else:
            max_val = 1.
        ecg = ecg / max_val
        ecg = ecg.T

        features = np.array([row.sex == 'Male',
                             row.sex=='Female',
                             (row.age - 50) / 50], dtype=np.float16)
        if self.return_beat_id:
            return ecg, features, start, n
        if len(self.noise_path) > 0:
            return denoised_ecg, features, ecg, noise, leads
        return ecg, features

    def __getitem__(self, n: int) -> Tuple:
        return self._get_ecg(n)

    def __len__(self) -> int:
        return len(self._ids)

class LQT_ECG(Dataset):
    """s
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, database_path: str, segment_length: int = 176,
                 normalized: str = 'global',
                 return_beat_id: bool = False, all_limb=False):
        self.database_path = database_path
        self.annotations = pd.read_csv(os.path.join(database_path, 'annotations_clean.csv'))
        self.annotations = self.annotations[self.annotations['status'].isin(['POS', 'NEG'])]
        self.signal_length = segment_length
        self.normalized = normalized
        self.return_beat_id = return_beat_id
        self.all_limb = all_limb

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int]:
        row = self.annotations.iloc[n]
        file_name = row.id
        path_to_raw_data = os.path.join(self.database_path, file_name + '_beats.npy')
        n_beats = row.n_beats
        beat_index = np.random.randint(low=0,
                                       high=n_beats,
                                       size=(1,))[0]

        ecg = read_memmap_from_position(filename=path_to_raw_data,
                                        shape=(12, n_beats, self.signal_length),
                                        beat_index=beat_index)
        if not self.all_limb:
            ecg = np.concatenate([ecg[:3], ecg[6:]])
        ecg = ecg.astype(np.float16)
        if self.normalized == 'per_lead':
            ecg = ecg / np.abs(ecg).max(axis=-1).clip(1e-4, 10)[..., None]
        if self.normalized == 'global':
            ecg = ecg / np.abs(ecg).max().clip(1e-4, 10)
        ecg = ecg.T
        fp = np.memmap(filename=os.path.join(self.database_path, f'{file_name}_rr.npy'),
                       mode='r',
                       dtype='int',
                       shape=(n_beats,),
                       offset=0)
        rr = np.asarray(fp[beat_index])

        features = np.array([row.sex == 'M',
                             row.sex =='F',
                             (rr - 125) / 125,
                             (row.age - 50) / 50], dtype=np.float16)
        label = int(row.status == 'POS')
        if self.return_beat_id:
            return ecg, features, beat_index, n, label
        return ecg, features, label

    def __len__(self) -> int:
        return self.annotations.shape[0]




class PhysionetRerun(PhysionetECG):

    def __init__(self, npz_path: str):
        npz = np.load(npz_path)
        self._ids = np.stack([npz['dataset'], npz['patient_id'], npz['n_beat']], axis=1)
        self._beat_id = npz['beat_id']
        self._target_ecg = npz['target_ecg']

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int]:
        ecg = self._target_ecg[n]
        beat_index = self._beat_id[n]
        return ecg, beat_index, n


def numpy_collate(batch, n_devices=0):
    if isinstance(batch[0], np.ndarray):
        batch = np.stack(batch)
        if n_devices == 0:
            return batch
        else:
            return batch.reshape(n_devices, -1, *batch.shape[1:])
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples, n_devices=n_devices) for samples in transposed]

    else:
        if n_devices > 0:
            return np.array(batch).reshape(n_devices, -1)
        else:
            return np.array(batch)


if __name__ == '__main__':
    dataset = PhysionetECG(database_path='/mnt/data/gabriel/physionet.org/beats_db_more_meta',
                           categories_to_filter=["NSR", "SB", "STach", "SA"], normalized=True, training_class=True,
                           estimate_std=False)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=512,
                            shuffle=True,
                            num_workers=10,
                            drop_last=True,
                            collate_fn=numpy_collate)
    fun = jit(lambda x, y: random.split(x, y), static_argnums=1)
    res = fun(random.PRNGKey(0), len(dataloader))
    t1 = time.time()
    for i, _ in enumerate(zip(dataloader, res)):
        pass
    t2 = time.time()
    print(i, t2 - t1)
