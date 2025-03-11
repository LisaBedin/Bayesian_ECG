import sys
sys.path.append('.')
sys.path.append('..')
import os
from tqdm import tqdm
import wfdb
from functools import partial
import numpy as np
from preprocessing import ptbxl_tools
from multiprocessing import Pool

def create_npz_file(i, y_arr, headers, database_path, output_path, freq=250):
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


if __name__ == '__main__':
    # data_source = '/mnt/data/lisa/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords'
    # data_source = '/mnt/data/gabriel/physionet.org/files/challenge-2021/1.0.3/training'  # sys.argv[1]
    data_source = '/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3/'  # sys.argv[1]  # '/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3/'
    #data_source = '/lustre/fsn1/projects/rech/vpd/udw33dp/physionet.org/files/ptb-xl/1.0.3/'
    freq = 100 #  int(sys.argv[2])  # 125  # 250
    data_destination = os.path.join(data_source, f'processed_{freq}Hz')  # '/mnt/data/lisa/physionet.org/ecg_10s_db'  # sys.argv[2]
    os.makedirs(os.path.join(data_destination, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_destination, 'test'), exist_ok=True)
    task = 'all'

    raw_labels = ptbxl_tools.load_codes(data_source)
    labels = ptbxl_tools.compute_label_aggregations(raw_labels, data_source, task)
    labels, Y, mlb = ptbxl_tools.select_data(labels, task, min_samples=0, outputfolder=data_destination)
    y_train = Y[labels.strat_fold < 9]
    df_train = labels[labels.strat_fold < 9]
    y_test = Y[labels.strat_fold == 10]
    df_test = labels[labels.strat_fold == 10]
    # np.where( mlb.transform([['SARRH']])==1)

    y_val = Y[labels.strat_fold == 9]
    df_val = labels[labels.strat_fold == 9]
    df_val.to_csv(os.path.join(os.path.join(data_destination, 'val/headers.csv')))
    np.save(os.path.join(data_destination, 'val/labels.npy'), y_val)

    df_train.to_csv(os.path.join(os.path.join(data_destination, 'train/headers.csv')))
    np.save(os.path.join(data_destination, 'train/labels.npy'), y_train)
    df_test.to_csv(os.path.join(os.path.join(data_destination, 'test/headers.csv')))
    np.save(os.path.join(data_destination, 'test/labels.npy'), y_test)

    process_fn = partial(create_npz_file, y_arr=y_val, headers=df_val, database_path=data_source,
                         output_path=os.path.join(data_destination, 'val'), freq=freq)
    with Pool(65) as p:  # 13 minutes
        print(p.map(process_fn, tqdm(np.arange(df_val.shape[0]))))


    process_fn = partial(create_npz_file, y_arr=y_test, headers=df_test, database_path=data_source,
                         output_path=os.path.join(data_destination, 'test'), freq=freq)
    with Pool(65) as p:  # 13 minutes
        print(p.map(process_fn, tqdm(np.arange(df_test.shape[0]))))

    process_fn = partial(create_npz_file, y_arr=y_train, headers=df_train, database_path=data_source, output_path=os.path.join(data_destination, 'train'), freq=freq)
    with Pool(65) as p:
        print(p.map(process_fn, tqdm(np.arange(df_train.shape[0]))))
