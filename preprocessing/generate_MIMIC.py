import os
os.environ["OMP_NUM_THREADS"] = "50"
import sys
sys.path.append('.')
sys.path.append('..')
from tqdm import tqdm
import wfdb
from functools import partial
import numpy as np
from preprocessing import ptbxl_tools
from multiprocessing import Pool
import glob
from sklearn.model_selection import KFold


def create_npz_file_old(i, y_arr, headers, database_path, output_path, freq=250):
    y = y_arr[i]
    header = headers.iloc[i]
    signal, _ = wfdb.rdsamp(os.path.join(database_path, str(header['filename_hr'])))
    signal = ptbxl_tools.filter_ecg(freq, 500, signal.T)
    file_n = header['filename_hr'].split('/')[-1]
    np.savez(os.path.join(output_path, f'{file_n}.npz'),
             ecg=signal,
             patient_id=int(header['patient_id']),
             age=float(header['age']),
             sex=int(header['sex']),
             all_labels=header['all_scp']
             )


def process_file(file_p, target_fs=100):
    signal, header = wfdb.rdsamp(file_p)
    fs = header['fs']
    signal = ptbxl_tools.filter_ecg(target_fs, fs, signal.T)
    if signal.shape[0] == 12:
        return signal, '/'.join(file_p.split('/')[-3:-1])
    signalIII = signal[1] - signal[0]
    # aVL=(I-III)/2, aVF=(II+III)/2
    # and aVR=-(I+II)/2.
    aVR = -(signal[0]+signal[1])/2
    aVF = (signal[1]+signalIII)/2
    aVL = (signal[0]-signalIII)/2
    limb_leads = np.stack([signalIII, aVR, aVF, aVL])
    return np.concatenate([signal[:2], limb_leads, signal[2:]]), '/'.join(file_p.split('/')[-3:-1])


def create_npz_file(folder_p, results_path, target_fs):
    folder_name = folder_p.split('/')[-1]
    # print(f'doing {folder_name}: {i+1}/{len(all_folders)}')
    all_files = glob.glob(os.path.join(folder_p, '*/*/*.hea'))
    all_files = ['.'.join(file_p.split('.')[:-1]) for file_p in all_files]
    all_signals, all_prefix = [], []
    for file_p in tqdm(all_files, total=len(all_files)):
        # read file_p
        signal, prefix = process_file(file_p, target_fs=target_fs)
        keepit = bool(np.prod(1 - np.isnan(signal[0]).astype(int)))
        if keepit:
            all_signals.append(signal)
            all_prefix.append(prefix)
    np.savez(os.path.join(results_path, folder_name+'.npz'),
             ecg=np.stack(all_signals), info=np.array(all_prefix))


if __name__ == '__main__':
    data_source = '/mnt/Reseau/Signal/lisa/physionet.org/files/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files'  # sys.argv[1]
    #data_source = '/gpfsdsdir/dataset/MIMIC-IV-ECG/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files'
    freq = 100 # sys.argv[2]  # 125  # 250
    #data_destination = '/lustre/fsn1/projects/rech/vpd/udw33dp/mimic'
    data_destination = '/mnt/Reseau/Signal/lisa/mimic' #physionet.org/files/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'  # sys.argv[3]
    data_destination = os.path.join(data_destination, f'processed_{freq}Hz')  # '/mnt/data/lisa/physionet.org/ecg_10s_db'  # sys.argv[2]
    os.makedirs(os.path.join(data_destination, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_destination, 'val'), exist_ok=True)
    os.makedirs(os.path.join(data_destination, 'test'), exist_ok=True)

    all_folders = np.array(glob.glob(os.path.join(data_source, 'p*')))
    skf = KFold(n_splits=100, shuffle=True, random_state=0)
    for fold, inds in enumerate(skf.split(np.arange(len(all_folders)))):
        break
    train_folders = all_folders[inds[0]]
    val_folders = all_folders[inds[1]]
    skf = KFold(n_splits=2, shuffle=True, random_state=0)
    for fold, inds in enumerate(skf.split(np.arange(len(val_folders)))):
        break
    test_folders = val_folders[inds[1]]
    val_folders = val_folders[inds[0]]

    # for file_p in test_folders:
    #     create_npz_file(file_p, os.path.join(data_destination, 'test'), target_fs=freq)  # 20s for 850 samples

    process_fn = partial(create_npz_file, results_path=os.path.join(data_destination, 'test'), target_fs=freq)
    with Pool(5) as p:
        print(p.map(process_fn, tqdm(test_folders, total=len(test_folders))))

    process_fn = partial(create_npz_file, results_path=os.path.join(data_destination, 'val'), target_fs=freq)
    with Pool(5) as p:
        print(p.map(process_fn, tqdm(val_folders, total=len(val_folders))))

    process_fn = partial(create_npz_file, results_path=os.path.join(data_destination, 'train'), target_fs=freq)
    with Pool(50) as p:
        print(p.map(process_fn, tqdm(train_folders, total=len(train_folders))))
