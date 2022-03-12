# -*- coding: utf-8 -*-
import torch
import time
from DL.Trainer import val_epoch

def fit_KD(train_loader, val_loader, model, loss_fn_train, loss_fn_val, optimizer, scheduler, n_epochs, patience, device, save_path, log_interval):
    time_train_total = 0
    time_val_avg = 0
    best_val_acc = 0
    es_cnt = 0
    
    for epoch in range(0, n_epochs):
        # Train stage
        time_train = time.perf_counter()
        train_loss, train_acc = train_KD_epoch(train_loader, model, loss_fn_train, optimizer, device, log_interval)
        time_train_total += (time.perf_counter() - time_train)
        if epoch % log_interval == 0:
            print('Epoch: {}/{}. Train loss: {:.6f}\t Accuracy: {:.6f}'.format(epoch + 1, n_epochs, train_loss, train_acc))
        
        scheduler.step()
        
        # Validation stage
        time_val = time.perf_counter()
        val_loss, val_acc = val_epoch(val_loader, model, loss_fn_val, device)
        time_val_avg += (time.perf_counter() - time_val)
        if epoch % log_interval == 0:
            print('Epoch: {}/{}. Valid loss: {:.6f}\t Accuracy: {:.6f}'.format(epoch + 1, n_epochs, val_loss, val_acc))
        
    #     # Early stopping
    #     if val_acc > best_val_acc:
    #         es_cnt = 0
    #         best_val_acc = val_acc
    #     else:
    #         es_cnt += 1
            
    #         if es_cnt > patience:
    #             break
            
    # return best_val_acc, time_train_total, time_val_avg/n_epochs
    return val_acc, time_train_total, time_val_avg/n_epochs

def train_KD_epoch(train_loader, model, loss_fn_train, optimizer, device, log_interval):
    accum_loss = 0
    accum_acc = 0
    
    model.train()
    for batch_idx, (data, output_tch, label) in enumerate(train_loader):
        data = data.to(device)
        output_tch = output_tch.to(device)
        label = label.to(device)

        #Input-->model-->output-->loss_fn_train-->loss
        output_stu = model(data)
        loss = loss_fn_train(output_stu, output_tch, label)
        
        #Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Accumulated loss
        accum_loss += loss.item()
        #Accumulated accuracy
        _, pred = output_stu.max(1)
        num_correct = (pred == label).sum().item()
        acc = float(num_correct) / data.shape[0]
        accum_acc += acc

        #Print loss and accuracy while training
        # if batch_idx % log_interval == 0:
        #     print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), accum_loss/(batch_idx+1), accum_acc/(batch_idx+1)))
        
    return accum_loss/len(train_loader), accum_acc/len(train_loader)