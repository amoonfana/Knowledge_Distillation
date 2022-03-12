# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

from DL.DataReader import TimeSeries, TimeSeries_KD
from Net.ROCKET import ROCKET
from Net.ConvNet1d import ConvNet1d
from Net.ResNet1d import resnet1d18, resnet1d34, resnet1d50, resnet1d101, resnet1d152
from Net.InceptionTime import InceptionTime
from DL.Loss import LabelSmooth, KD, KDCT, KDCR
from DL.Trainer_KD import fit_KD
from DL.Predictor import predict
    
# def predict_TCH(dataset_info, batch_size, device):
#     outputs_tch = []
#     kwargs = {'num_workers': 0, 'pin_memory': True}
    
#     for _, row in dataset_info.iterrows():
#         dataset = row["dataset"]
        
#         TS_pred = TimeSeries("../../UCRArchive_2018/{}/{}_TRAIN.tsv".format(dataset, dataset))
#         pred_loader = torch.utils.data.DataLoader(TS_pred, batch_size=batch_size, shuffle=False, **kwargs)
#         model_TCH = torch.load('../models/{}.pkl'.format(dataset)).to(device)
        
#         output_tch = predict(pred_loader, model_TCH, device).cpu()
#         outputs_tch.append(output_tch.numpy())
        
#         _, pred = output_tch.max(1)
#         num_correct = (pred == TS_pred.Y).sum().item()
#         acc = float(num_correct) / output_tch.shape[0]
        
#     return outputs_tch

dataset_info = pd.read_csv("ds_info.csv")
results = pd.DataFrame(index = dataset_info["dataset"], columns = ["acc_mean", "acc_std", "acc_best", "time_train", "time_val"], data = 0)

# Set up hyper-parameters
n_runs = 3              # 5 default
learning_rate = 1e-3    # 1e-3 default
n_epochs = 512          # 256 default
batch_size = 64         # 128 default
patience = 80
log_interval = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True}

# outputs_tch = predict_TCH(dataset_info, batch_size, device)

# Generate soft labels by the teacher model
outputs_tch = []
# aaa = []
for _, row in dataset_info.iterrows():
    dataset = row["dataset"]
    
    TS_pred = TimeSeries("../../UCRArchive_2018/{}/{}_TRAIN.tsv".format(dataset, dataset))
    pred_loader = torch.utils.data.DataLoader(TS_pred, batch_size=batch_size, shuffle=False, **kwargs)
    model_TCH = torch.load('../models/{}.pkl'.format(dataset)).to(device)
    
    output_tch = predict(pred_loader, model_TCH, device).cpu()
    outputs_tch.append(output_tch.numpy())
    
    # _, pred = output_tch.max(1)
    # num_correct = (pred == TS_pred.Y).sum().item()
    # acc = float(num_correct) / output_tch.shape[0]
    # aaa.append([num_correct, output_tch.shape[0], acc])
    
torch.cuda.empty_cache()

# Train the student model
for idx, row in dataset_info.iterrows():
    dataset = row["dataset"]
    n_classes = row["numClasses"]
    
    TS_Train = TimeSeries_KD("../../UCRArchive_2018/{}/{}_TRAIN.tsv".format(dataset, dataset), outputs_tch[idx])
    TS_Test = TimeSeries("../../UCRArchive_2018/{}/{}_Test.tsv".format(dataset, dataset))
    
    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(TS_Train, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(TS_Test, batch_size=batch_size, shuffle=False, **kwargs)
    
    accs = np.zeros(n_runs)
    times_train = np.zeros(n_runs)
    times_val = np.zeros(n_runs)
    for i in range(n_runs):
        # Set up networks, loss functions and optimizers
        # model = ROCKET(1, TS_Train.ts_len, n_output//2) # n_output must be multiple of 2
        # model = ConvNet1d(n_output, TS_Train.ts_len)
        # model = resnet1d18(n_output)
        model = InceptionTime(1, n_classes, depth=3)
        
        # loss_fn_train = nn.CrossEntropyLoss()
        # loss_fn_train = LabelSmooth(0.1)
        # loss_fn_train = KD(0.9, 1)
        loss_fn_train = KDCT(0.9)
        # loss_fn_train = KDCR(1, 1)
        loss_fn_val = nn.CrossEntropyLoss()
        
        model = model.to(device)
        loss_fn_train = loss_fn_train.to(device)
        loss_fn_val = loss_fn_val.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        # optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}, {'params': loss_fn.parameters(), 'lr': learning_rate}], weight_decay=0.01)
        # optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': learning_rate}, {'params': loss_fn.parameters(), 'lr': learning_rate}], momentum=0.8)
        
        scheduler = lr_scheduler.StepLR(optimizer, step_size=(n_epochs//7 - 1), gamma=0.5, last_epoch=-1)
        
        # Train the network
        accs[i], times_train[i], times_val[i] = fit_KD(train_loader,
                                                    val_loader,
                                                    model,
                                                    loss_fn_train,
                                                    loss_fn_val,
                                                    optimizer,
                                                    scheduler,
                                                    n_epochs,
                                                    patience,
                                                    device,
                                                    "../models/{}.pkl".format(dataset),
                                                    log_interval)
        
        torch.cuda.empty_cache()
    
    results.loc[dataset, "acc_mean"] = accs.mean()
    results.loc[dataset, "acc_std"] = accs.std()
    results.loc[dataset, "acc_best"] = accs.max()
    results.loc[dataset, "time_train"] = times_train.mean()
    results.loc[dataset, "time_val"] = times_val.mean()
    
results.to_csv("../results/STU.csv")