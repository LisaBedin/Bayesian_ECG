import json
import torch
import sys
sys.path.append('.')
import os
from tqdm import tqdm
from resnet import ResNet1d
from dataloader import BatchDataloader
import torch.optim as optim
import numpy as np
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from diffusion_prior.dataloaders import dataloader
from diffusion_prior.utils import calc_diffusion_hyperparams, local_directory, print_size  #
from diffusion_prior.models import construct_model
from posterior_samplers.mgps import load_vdps_sampler
import torch.nn.functional as F
from sklearn.metrics import recall_score, balanced_accuracy_score
import torch.nn.init as init


def evaluation_report(y_true, y_pred):
    lab_lst = ['AF', 'TAb', 'QAb', 'VPB', 'LAD', 'SA', 'LBBB', 'RBBB']
    recall_dic, specific_dic = {}, {}
    for i, lab in enumerate(lab_lst):
        recall = recall_score(y_true[:, i], y_pred[:, i], pos_label=1)
        specificity = recall_score(y_true[:, i], y_pred[:, i], pos_label=0)
        recall_dic[lab] = recall
        specific_dic[lab] = specificity
    return recall_dic, specific_dic


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def BCE(pred_ages, ages, weights):
    # diff = ages.flatten() - pred_ages.flatten()
    # loss = torch.sum(weights.flatten() * diff * diff)
    pos_w = 1 / ages.sum()
    neg_w = 1 / (1-ages).sum()
    pos_w /= (pos_w + neg_w)
    neg_w /= (pos_w + neg_w)
    batch_w = ages * pos_w + (1-ages) * neg_w
    loss = F.binary_cross_entropy_with_logits(pred_ages, ages, #   weight=weights,
                           reduction='sum')
    return loss

def BCE_weight(pred_ages, ages, weights, pos_w):
    # diff = ages.flatten() - pred_ages.flatten()
    # loss = torch.sum(weights.flatten() * diff * diff)
    # pos_w = 1 / ages.sum()
    # neg_w = 1 / (1-ages).sum()
    # pos_w /= (pos_w + neg_w)
    # neg_w /= (pos_w + neg_w)
    # batch_w = ages * pos_w + (1-ages) * neg_w
    # loss = F.binary_cross_entropy(pred_ages, ages,   weight=batch_w,
                                   #                        reduction='sum')
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w).cuda())(pred_ages, ages)
    return loss

def train(ep, dataload, model, device, optimizer, loss_fn=BCE):
    model.train()
    total_loss = 0
    total_recall = 0
    total_specificity = 0
    total_acc = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for traces, ages, weights in dataload:
        # traces = traces.transpose(1, 2)
        traces, ages, weights = traces.to(device), ages.to(torch.float32).to(device), weights.to(device)
        ages = ages.flatten()  # F.one_hot(ages.to(int).flatten()).to(torch.float32)
        weights = weights.flatten()
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces)
        loss = loss_fn(pred_ages, ages, weights)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        total_loss += loss.detach().cpu().numpy()
        y_pred = (nn.Sigmoid()(pred_ages.detach()).cpu().numpy() > 0.5).astype(int)
        recall = recall_score(ages.cpu(), y_pred, pos_label=1)
        specificity = recall_score(ages.cpu(), y_pred, pos_label=0)
        total_recall += recall
        total_specificity += specificity
        total_acc += balanced_accuracy_score(ages.cpu().numpy(), y_pred)  # y_pred == ages.cpu().numpy()).sum() / len(y_pred)
        n_entries += 1  # bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries, total_recall / n_entries, total_specificity / n_entries, total_acc / n_entries


def eval(ep, dataload, model, device, loss_fn=BCE):
    model.eval()
    total_loss = 0
    n_entries = 0
    total_recall = 0
    total_specificity = 0
    total_acc = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for traces, ages, weights in dataload:
        # traces = traces.transpose(1, 2)
        traces, ages, weights = traces.to(device), ages.to(torch.float32).to(device), weights.to(device)
        ages = ages.flatten()  # F.one_hot(ages.to(int).flatten()).to(torch.float32)
        weights = weights.flatten()
        with torch.no_grad():
            # Forward pass
            pred_ages = model(traces)
            loss = loss_fn(pred_ages, ages, weights)
            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            y_pred = (nn.Sigmoid()(pred_ages.detach()).cpu().numpy() > 0.5).astype(int)
            recall = recall_score(ages.cpu(), y_pred, pos_label=1)
            specificity = recall_score(ages.cpu(), y_pred, pos_label=0)
            total_recall += recall
            total_specificity += specificity
            total_acc += balanced_accuracy_score(ages.cpu().numpy(),
                                                 y_pred)  # y_pred == ages.cpu().numpy()).sum() / len(y_pred)
            n_entries += 1  # bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries, total_recall / n_entries, total_specificity / n_entries, total_acc / n_entries


@hydra.main(version_base=None, config_path="../sashimi/configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    torch.manual_seed(args.seed)
    print(args)
    # Set device
    device = torch.device('cuda:0')  #  if args.cuda else 'cpu')
    num_gpus = torch.cuda.device_count()
    tqdm.write("Building data loaders...")
    # cfg.dataset.negative_class = True
    cfg.dataset.return_weights = True
    cfg.dataset.training_class = 'test'
    cfg.dataset.all_limb = True
    cfg.dataset.label_names = [cfg.algo.downstream_classification]
    valid_loader = dataloader(cfg.dataset, batch_size=2048, num_gpus=num_gpus, unconditional=cfg.model.unconditional, shuffle=False)
    cfg.dataset.training_class = 'train'
    train_loader = dataloader(cfg.dataset, batch_size=args.batch_size, num_gpus=num_gpus, unconditional=cfg.model.unconditional, shuffle=True)
    local_path, output_directory = local_directory(None, cfg.train.results_path,
                                                   cfg.model, cfg.diffusion,
                                                   cfg.dataset,
                                                   'waveforms')
    output_directory = os.path.join(output_directory, f"georgia_ribeiro_{args.loss_type}_{cfg.algo.downstream_classification}_lr{args.lr}_bs{args.batch_size}")
    args.folder = folder = output_directory
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
    model.to(device=device)
    model.apply(weights_init)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    # best_loss = np.Inf
    best_acc = 0
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'train_recall', 'train_specificity', 'train_acc',
                                    'valid_recall', 'valid_specificity', 'valid_acc'])
    #'''
    pos_w = 1./train_loader.dataset.labels.sum()
    neg_w = 1./(1 - train_loader.dataset.labels).sum()
    pos_w /= (pos_w + neg_w)
    neg_w /= (pos_w + neg_w)
    weight = torch.tensor([neg_w, pos_w]).cuda()
    #'''
    if args.loss_type == 'BCE':
        loss_fn = BCE  # nn.BCELoss() # weight=torch.tensor([neg_w, pos_w]).to(device))  # WithLogitsLoss()  # pos_weight=torch.tensor([pos_w]).to(device))  # pos_weight=train_loader.dataset.inv_counts.to(device))  # CrossEntropyLoss(weight=train_loader.dataset.inv_counts.to(device), reduction='sum')
        #elif 'BCEweight':
        #     loss_fn = BCE_weight
    else:
        loss_fn = lambda x0, x1, x2: BCE_weight(x0, x1, x2, min(20, pos_w/neg_w))
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w).cuda())
    for ep in range(start_epoch, args.epochs):
        train_loss, train_recall, train_spec, train_acc = train(ep, train_loader, model, device, optimizer, loss_fn)
        valid_loss, val_recall, val_spec, val_acc = eval(ep, valid_loader, model, device, loss_fn)
        # Save best model
        if val_acc > best_acc:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_acc = val_acc
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\n'
                   'Train Recall {:.6f}\t' \
                   'Train Spec {:.6f}\n' \
                   'Train Acc {:.6f}\n'
                   'Valid Recall {:.6f}\t' \
                   'Valid Spec {:.6f}\n' \
                    'Valid Acc {:.6f}'
                   .format(ep, train_loss, valid_loss, learning_rate,
                           train_recall, train_spec, train_acc, val_recall, val_spec, val_acc))
        print(' ')
        print(' ')

        # Save history
        history = pd.concat([history, pd.DataFrame({
            "epoch": [ep], "train_loss": [train_loss],
                                  "valid_loss": [valid_loss],
                            'train_recall': [train_recall],
            'train_specificity': [train_spec],
            'train_acc': [train_acc],
            'valid_recall': [val_recall],
            'valid_specificity': [val_spec],
            'valid_acc': [val_acc],
                                        "lr": [learning_rate]})], axis=0, ignore_index=True)
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")
    print(folder)
    print(' ')
    X_test, y_test, w_test = [], [], []
    for X_test_, y_test_, w_test_ in valid_loader:
        X_test.append(X_test_)
        y_test.append(y_test_) # .flatten())
        w_test.append(w_test_)
    y_test = torch.cat(y_test).numpy()
    w_test = torch.cat(w_test)
    X_test = torch.cat(X_test)
    model.eval()
    with ((torch.no_grad())):
        y_logit = model(X_test.to(device))
        # y_score = nn.Sigmoid()(y_logit)
        y_score = y_logit.detach().cpu().numpy()
    y_pred = (nn.Sigmoid()(y_score) > 0.5).astype(int)
    recall = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    print(f'{cfg.algo.downstream_classification}: recall={recall:.4f}, specificity={specificity:.4f}')
    # print(evaluation_report(y_test, y_pred))
    print('ok')
    # y_pred = np.zeros_like(y_score)
    # y_pred[:, np.argmax(y_score, axis=1)] = 1
    # print(evaluation_report(y_test, y_pred))


if __name__ == "__main__":
    import h5py
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=200, # 100,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=1024,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=128, # 128,  # 64,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001, # 01,  # 3e-4, # 01,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--loss_type", type=str, default='BCEweight',
                        help='loss type')
    parser.add_argument("--lr_factor", type=float, default=0.99,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256], #  320],
                        help='filter size in resnet layers (default: [128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[ 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    # parser.add_argument('--folder', default='/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_ptbxl100_L1024/waveforms/georgia_ribeiro',
    #                     help='output folder (default: ./out)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset', default='',
                        help='by default consider the ids are just the order')
    parser.add_argument('--age_col', default='age',
                        help='column with the age in csv file.')
    parser.add_argument('--ids_col', default=None,
                        help='column with the ids in csv file.')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                             'The rest is for training')
    # parser.add_argument('path_to_traces',
    #                     help='path to file containing ECG traces')
    # parser.add_argument('path_to_csv',
    #                     help='path to csv file containing attributes.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    main()
