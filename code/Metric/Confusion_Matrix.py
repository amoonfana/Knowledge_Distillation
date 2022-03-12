import numpy as np
import pandas as pd
from collections import Counter
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from DL.DataReader import TimeSeries
from DL.Predictor import predict

dataset_info = pd.read_csv("../ds_info.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True}

for _, row in dataset_info.iterrows():
    dataset = row["dataset"]
    
    if dataset != "InsectWingbeatSound":
        continue
    
    TS_pred = TimeSeries("../../../UCRArchive_2018/{}/{}_Test.tsv".format(dataset, dataset))
    pred_loader = torch.utils.data.DataLoader(TS_pred, batch_size=64, shuffle=False, **kwargs)
    model = torch.load('../../models/{}.pkl'.format(dataset)).to(device)
    
    output = predict(pred_loader, model, device).cpu()
    
    _, pred = output.max(1)
    
    # a = np.array([4, 3, 1, 2])
    # j = np.argsort(-a)
    # a = a[j]
    
    pred = pred.numpy()
    true = TS_pred.Y.numpy()
    hist_class = Counter(true)
    hist_class = np.array(list(hist_class.values()))
    idx = np.argsort(-hist_class)
    
    cm = confusion_matrix(true, pred)
    
    cls_acc = np.zeros(cm.shape[0])
    for i in range(cls_acc.shape[0]):
        cls_acc[i] = cm[i][i]/cm[i].sum()
        
    hist_class = hist_class[idx]
    cls_acc = cls_acc[idx]
    
    # plt.figure(figsize=(8,8))
    # plt.tick_params(labelsize=20)
    # plt.xlabel(dataset+"Sorted class indices",fontsize=28)
    # plt.ylabel("Number",fontsize=28)
    
    # x1 = np.array(range(len(hist_class)))
    # plt.bar(x1, hist_class, color="#41b6e6")
    
    # plt.figure(figsize=(8,8))
    # plt.tick_params(labelsize=20)
    # plt.xlabel("Sorted class indices",fontsize=28)
    # plt.ylabel("Number",fontsize=28)
    
    # x1 = np.array(range(len(cls_acc)))
    # plt.bar(x1, cls_acc, color="#41b6e6")
    
    # plt.show()

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(8,8))
plt.matshow(cm, cmap = "Blues")
cb = plt.colorbar(shrink=0.8, aspect=16)
cb.ax.tick_params(labelsize=12)

plt.tick_params(labelsize=12)
plt.title("预测标签",fontsize=18)
# plt.xlabel("预测标签",fontsize=18)
plt.ylabel("真实标签",fontsize=18)

for x in range(cm.shape[0]):
    for y in range(cm.shape[1]):
        plt.annotate(cm[x,y], xy=(x,y), horizontalalignment="center", verticalalignment="center",fontsize=8)