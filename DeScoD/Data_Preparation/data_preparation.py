import numpy as np
import _pickle as pickle
from DeScoD.Data_Preparation import Prepare_QTDatabase, Prepare_NSTDB

def Data_Preparation(noise_version=2, noise_type='bw'):

    print('Getting the Data ready ... ')

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    Prepare_QTDatabase.prepare()
    Prepare_NSTDB.prepare()

    # Load QT Database
    with open('/mnt/Reseau/Signal/lisa/ECG_data/DeScoD_data/QTDatabase.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('/mnt/Reseau/Signal/lisa/ECG_data/DeScoD_data/NoiseBWL.pkl', 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    if noise_type == 'bw':
        [bw_signals,_,_] = nstdb
    elif noise_type == 'em':
        [_,bw_signals,_] = nstdb
    #[_, em_signals, _ ] = nstdb
    #[_, _, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    

    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]



    #####################################
    # Data split
    #####################################
    if noise_version == 1:
        noise_test = bw_noise_channel2_b
        noise_train = bw_noise_channel1_a
    elif noise_version == 2:
        noise_test = np.stack([bw_noise_channel1_b, bw_noise_channel2_b], axis=-1)
        noise_train = np.stack([bw_noise_channel2_a, bw_noise_channel2_a], axis=-1)
    else:
        raise Exception("Sorry, noise_version should be 1 or 2")

    #####################################
    # QTDatabase
    #####################################

    beats_train = []
    beats_test = {}
    padding_test = {}
    
    '''
    test_set = ['qt-database-1.0.0/sel123',  # Record from MIT-BIH Arrhythmia Database
                'qt-database-1.0.0/sel233',  # Record from MIT-BIH Arrhythmia Database

                'qt-database-1.0.0/sel302',  # Record from MIT-BIH ST Change Database
                'qt-database-1.0.0/sel307',  # Record from MIT-BIH ST Change Database

                'qt-database-1.0.0/sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'qt-database-1.0.0/sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database

                'qt-database-1.0.0/sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'qt-database-1.0.0/sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database

                'qt-database-1.0.0/sele0106',  # Record from European ST-T Database
                'qt-database-1.0.0/sele0121',  # Record from European ST-T Database

                'qt-database-1.0.0/sel32',  # Record from ``sudden death'' patients from BIH
                'qt-database-1.0.0/sel49',  # Record from ``sudden death'' patients from BIH

                'qt-database-1.0.0/sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'qt-database-1.0.0/sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]
    '''
    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database   "['MLII', 'V5']
                'sel233',  # Record from MIT-BIH Arrhythmia Database   ['MLII', 'V1']
               #
               # 'sel302',  # Record from MIT-BIH ST Change Database ['ECG1', 'ECG2']
               # 'sel307',  # Record from MIT-BIH ST Change Database ['ECG1', 'ECG2']
               #
               # 'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database ['ECG1', 'ECG2']
               # 'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database ['ECG1', 'ECG2']
               #
               # 'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database ['ECG1', 'ECG2']
               # 'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database ['ECG1', 'ECG2']

                'sele0106',  # Record from European ST-T Database ['D3', 'V3']
                'sele0121',  # Record from European ST-T Database ['V4', 'D3']

               #  'sel32',  # Record from ``sudden death'' patients from BIH ['ECG1', 'ECG2']
               #  'sel49',  # Record from ``sudden death'' patients from BIH ['ECG1', 'ECG2']
               # 
               #  'sel14046',  # Record from MIT-BIH Long-Term ECG Database ['ECG1', 'ECG2']
               #  'sel15814',  # Record from MIT-BIH Long-Term ECG Database ['ECG1', 'ECG2']
                ]
    
    skip_beats = 0
    test_beats_skiped = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        full_size = 0
        if signal_name in test_set:
            beat_size = []
            all_beats = []
        for b in qtdb[signal_name]:
            b_np = np.zeros((samples, 2))
            b_sq = np.array(b)
            
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                if signal_name in test_set:
                    test_beats_skiped += 1
                continue
            full_size += b_sq.shape[0]
            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0:1] + b_sq[-1:]) / 2

            if signal_name in test_set:
                all_beats.append(b_np)
                beat_size.append(b_sq.shape[0])
                if full_size > int(1024 * 360 / 100):
                    print(full_size)
                    break
            else:
                beats_train.append(b_np)
        if signal_name in test_set:
            beats_test[signal_name] = np.array(all_beats)
            padding_test[signal_name] = np.array(beat_size)

    for signal_name in test_set:
        print(signal_name, beats_test[signal_name].shape)
    
    sn_train = []
    noise_index = 0
    
    # Adding noise to train
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(10):  # len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    # rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    #rnd_test = np.random.randint(low=150, high=200, size=len(beats_test)) / 100

    # Saving the random array so we can use it on the amplitude segmentation tables
    # np.save(f'/mnt/Reseau/Signal/lisa/ECG_data/DeScoD_data/rnd_test_{noise_version}{noise_type}.npy', rnd_test)
    # print('rnd_test shape: ' + str(rnd_test.shape))
    sn_test = {signal_name: [] for signal_name in test_set}

    for signal_name, beat_lst in beats_test.items():
        pad_lst = padding_test[signal_name]
        noise_start = noise_index
        for i in range(len(beat_lst)):
            noise = noise_test[noise_index:noise_index + samples]
            beat_max_value = np.max(beat_lst) - np.min(beat_lst) # np.max(beat_lst[i]) - np.min(beat_lst[i])
            noise_max_value = np.max(noise_test[noise_start:noise_start+int(1024*360/100)-16]) - np.min(noise_test[noise_start:noise_start+int(1024*360/100)-16])
            Ase = noise_max_value / beat_max_value
            alpha = 1 / Ase  #  rnd_test[i] / Ase
            signal_noise = beat_lst[i] + alpha * noise

            sn_test[signal_name].append(signal_noise)
            noise_index += pad_lst[i]

        if noise_index > (len(noise_test) - int(1024*360/100)-16):
            noise_index = 0

    for signal_name in test_set:
        sn_test[signal_name] = np.array(sn_test[signal_name])

    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    
    # X_test = np.array(sn_test)
    # y_test = np.array(beats_test)
    
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    # X_test = np.expand_dims(X_test, axis=2)
    # y_test = np.expand_dims(y_test, axis=2)


    Dataset = [X_train, y_train, sn_test, beats_test, padding_test]

    print('Dataset ready to use.')

    return Dataset

if __name__ == '__main__':
    Dataset = Data_Preparation()