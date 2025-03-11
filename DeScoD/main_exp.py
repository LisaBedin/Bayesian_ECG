import argparse
import torch
import datetime
import json
import yaml
import os
import wfdb
from scipy.signal import resample

from main_model import DDPM
from denoising_model_small import ConditionalModel
from utils import train, evaluate
import sys
sys.path.append('..')
from diffusion_prior.dataloaders import PtbXLDataset

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="DDPM for ECG")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--noise_type", type=str, default="bw")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--n_type', type=int, default=0, help='noise version')
    args = parser.parse_args()
    print(args)
    
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    foldername = "/mnt/data/lisa/ecg_results/DeScoD/checkpoints/noise_type_" + str(args.noise_type) + str(args.n_type) + "/"
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    noise_type = args.noise_type  # 'bw'
    noise_path = '/mnt/data/lisa/physionet.org/mit-bih-noise-stress-test-database-1.0.0/' + noise_type
    record = wfdb.rdrecord(noise_path).__dict__
    noise = record['p_signal']
    f_s_prev = record['fs']
    new_s = int(round(noise.shape[0] / f_s_prev * 100))
    noise_stress = resample(noise, num=new_s)[:, args.n_type]

    train_set = PtbXLDataset('/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3',
                             training_class='train',
                             all_limb=False, segment_length=1024,
                             sampling_rate=100,
                            label_names=None,
                            noise=noise_stress)

    val_set = PtbXLDataset('/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3',
                             training_class='val',
                             all_limb=False, segment_length=1024,
                             sampling_rate=100,
                            label_names=None,
                            noise=noise_stress)

    test_set = PtbXLDataset('/mnt/data/lisa/physionet.org/files/ptb-xl_clean/1.0.3',
                             training_class='test',
                             all_limb=False, segment_length=1024,
                             sampling_rate=100,
                            label_names=None,
                            noise=noise_stress)

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
    
    #base_model = ConditionalModel(64,8,4).to(args.device)
    base_model = ConditionalModel(config['train']['feats']).to(args.device)
    model = DDPM(base_model, config, args.device).cuda()
    
    train(model, config['train'], train_loader, args.device, 
          valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)
    checkpoint = torch.load(os.path.join(foldername, 'model.pth'))
    model.load_state_dict(checkpoint)
    #eval final
    print('eval final')
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #eval best
    print('eval best')
    foldername = "./check_points/noise_type_" + str(1) + "/"
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path))
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #don't use before final model is determined
    print('eval test')
    evaluate(model, test_loader, 1, args.device, foldername=foldername)
    
    
    
    
    
    
    
    
    
    
