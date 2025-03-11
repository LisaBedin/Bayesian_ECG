from learn.Training import Train_model
from compute_metrics.compute_metrics import Compute_metrics
from compute_metrics.compute_peaks import Compute_peaks
import argparse



def main(data_path, val_path, save_path, seed, device, Train, save_results = "Results/", epoch = 10, batch_size = 256):
    if Train == True:
        print("Train Model")
        Train_model(data_path, val_path, seed, device, epoch, batch_size, save_path)
    else:
        print("Compute Metrics")
        Compute_metrics(data_path,save_path+"/Model.pth", save_results, seed, device)
        print("\nCompute Peaks")
        Compute_peaks(data_path,save_path+"/Model.pth", save_results, seed, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='program for training ECGrecover on PTB-XL base')
    parser.add_argument('--data_train',
                        default='/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3/processed_100Hz/train',
                        type=str, help='Path to data (precise file for calculating metrics / training folder)')
    parser.add_argument('--data_val',
                        default='/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3/processed_100Hz/val',
                        type=str, help='Path to data (precise file for calculating metrics / training folder)')
    parser.add_argument('--save_path',
                        default='/mnt/data/lisa/ecg_results/ECGrecover',
                        type=str, help='Path where model is to be saved (Training) or where model is located (calculates metrics)')
    parser.add_argument('--seed', default=0, type=int, help='Seed for data mixing')
    parser.add_argument('--device', default=0, type=int, help='GPU to be used')
    
    #parser.add_argument('--Train', action='store_true', help='parameter to specify if you want to train the model')

    parser.add_argument('--save_results',
                        default='/mnt/data/lisa/ecg_results/ECGrecover',
                        type=str, help='Where metrics tables should be saved')
    
    parser.add_argument('--epoch', type=int, default = 100, help='Number of epochs for model training')
    parser.add_argument('--batch_size',type=int, default = 256, help='Batch size for model training')
    #parser.add_argument('--Verbose', action='store_true', help='Verbose for the training')

    args = parser.parse_args()
    args.Train = True
    args.Verbose = True
    main(args.data_train, args.data_val, args.save_path, args.seed, args.device, args.Train, args.save_results , args.epoch, args.batch_size)
