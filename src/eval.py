from transformers import SamModel, SamProcessor
import torch
from dataset import S1S2Dataset
import numpy as np
import rasterio
import os
from torch.utils.data import DataLoader

def compute_metrics(preds, labels):
    metrics = {}
    # IOU
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    metrics["iou"] = iou.mean().item()
    
    # Dice
    dice = (2.0 * intersection + 1e-6) / (preds.float().sum((1, 2)) + labels.float().sum((1, 2)) + 1e-6)
    metrics["dice"] = dice.mean().item()
    
    # Recall
    recall = intersection / (labels.float().sum((1, 2)) + 1e-6)
    metrics["recall"] = recall.mean().item()
    
    # Precision
    precision = intersection / (preds.float().sum((1, 2)) + 1e-6)
    metrics["precision"] = precision.mean().item()
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load dataset 
    processor = SamProcessor.from_pretrained(checkpoint_path)
    dataset = S1S2Dataset("split/val/img", "split/val/msk", processor)
    val_loader = DataLoader(dataset, batch_size=32 , shuffle=False)
    
    
    # load model
    model = SamModel.from_pretrained(checkpoint_path)
    
    for i, data in enumerate(val_loader):
        inputs, labels = data
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output = False)
            preds = torch.sigmoid(outputs.logits).round()
            metrics = compute_metrics(preds, labels)
            # save metrics 
            print(metrics)

    
    