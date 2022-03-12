# -*- coding: utf-8 -*-
import torch

def predict(pred_loader, model, device):
    output = None
    
    with torch.no_grad():
        model.eval()
        for data, _ in pred_loader:
            data = data.to(device)
            if output is None:
                output = model(data)
            else:
                output = torch.cat([output, model(data)], dim=0)

    return output