import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Softmax cross entropy + L2 regularization loss
# class LinearCE_L2Reg(nn.Module):
#     def __init__(self, model, gamma):
#         super(LinearCE_L2Reg, self).__init__()
#         self.model = model
#         self.gamma = gamma
            
#     def forward(self, output, label):
#         loss = F.cross_entropy(output, label)
        
#         loss_l2 = torch.tensor([0], dtype = torch.float32)
#         for param in self.model.parameters():
#             loss_l2 += torch.norm(param, 2)
            
#         return loss + self.gamma*loss_l2
        
# Softmax cross entropy + Label smoothing loss
class LabelSmooth(nn.Module):
    def __init__(self, epsilon):
        super(LabelSmooth, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, label):
        log_p = F.log_softmax(output, dim=1)
        pos = F.nll_loss(log_p, label)
        neg = -log_p.sum(dim=1).mean()/output.shape[1]
        loss = (1-self.epsilon)*pos + self.epsilon*neg
        
        return loss
    
# Knowledge distillation
class KD(nn.Module):
    def __init__(self, alpha, T):
        super(KD, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, output_stu, output_tch, label):
        loss_stu = F.cross_entropy(output_stu, label)
        
        label_stu = F.log_softmax(output_stu/self.T, dim=1)
        label_tch = F.softmax(output_tch/self.T, dim=1)
        loss_tch = F.kl_div(label_stu, label_tch) * self.T * self.T
        
        loss = loss_stu*(1-self.alpha) + loss_tch*self.alpha
        
        return loss

# Knowledge distillation with calibration by traslating
class KDCT(nn.Module):
    def __init__(self, alpha):
        super(KDCT, self).__init__()
        self.alpha = alpha
        
    def forward(self, output_stu, output_tch, label):
        loss_stu = F.cross_entropy(output_stu, label)
        
        label_stu = F.log_softmax(output_stu, dim=1)
        label_tch = F.softmax(output_tch, dim=1)
        
        # Traslating
        with torch.no_grad():
            _, pred = label_tch.max(1)
            idx = (pred!=label)
            
            if label_tch[idx].shape[0]>0:
                label_oh = torch.zeros_like(label_tch).scatter_(1, label.unsqueeze(1), 1)
                D = label_oh[idx] - label_tch[idx]
                d = torch.norm(D, dim=1)
                D = (D.T/d).T
                label_tch[idx] = label_tch[idx] + 0.7071*D
        
        loss_tch = F.kl_div(label_stu, label_tch)
        
        loss = loss_stu*(1-self.alpha) + loss_tch*self.alpha
        
        return loss
        # return loss_tch

# Knowledge distillation with calibration by reordering
class KDCR(nn.Module):
    def __init__(self, alpha, T):
        super(KDCR, self).__init__()
        self.alpha = alpha
        self.T = T
        
    def forward(self, output_stu, output_tch, label):
        loss_stu = F.cross_entropy(output_stu, label)
        
        # Reordering
        with torch.no_grad():
            _, pred = output_tch.max(1)
            idx = (pred!=label)
            l = label[idx]
            if l.shape[0]>0:
                o = output_tch[idx]
                _, i = torch.sort(o, descending=True)
                for j in range(l.shape[0]):
                    tmp = o[j][i[j][0]].item()
                    k = 0
                    while i[j][k] != l[j]:
                        o[j][i[j][k]] = o[j][i[j][k+1]]
                        k += 1
                    o[j][i[j][k]] = tmp
                output_tch[idx] = o
        
        # label_stu = F.log_softmax(output_stu, dim=1)
        # label_tch = F.softmax(output_tch, dim=1)
        # loss_tch = F.kl_div(label_stu, label_tch)
        label_stu = F.log_softmax(output_stu/self.T, dim=1)
        label_tch = F.softmax(output_tch/self.T, dim=1)
        loss_tch = F.kl_div(label_stu, label_tch) * self.T * self.T
        
        loss = loss_stu*(1-self.alpha) + loss_tch*self.alpha
        
        return loss

# # Knowledge distillation with calibration
# class LinearCE_KDC(nn.Module):
#     def __init__(self, alpha):
#         super(LinearCE_KDC, self).__init__()
#         self.alpha = alpha
        
#     def forward(self, output_stu, output_tch, label):
#         loss_stu = F.cross_entropy(output_stu, label)
        
#         output_stu = F.log_softmax(output_stu, dim=1)
#         # loss_tch = torch.mul(output_tch, output_stu)
#         # loss_tch = -loss_tch.mean()
#         loss_tch = F.kl_div(output_stu, output_tch)
        
#         loss = loss_stu*(1-self.alpha) + loss_tch*self.alpha
        
#         return loss
        