"""
========================================
训练器模块
========================================

包含训练和验证相关的函数
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前epoch编号
        
    Returns:
        平均损失和准确率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到设备
        images = batch['images'].to(device)
        num_images = batch['num_images']
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(images, num_images, input_ids, attention_mask)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {correct/total*100:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        平均损失、准确率、预测结果和真实标签
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    all_video_ids = []
    all_titles = []
    all_dataset_folders = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 将数据移动到设备
            images = batch['images'].to(device)
            num_images = batch['num_images']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            # 获取元数据
            video_ids = batch['video_id']
            titles = batch['title']
            dataset_folders = batch.get('dataset_folder', [''] * len(video_ids))
            
            # 前向传播
            logits = model(images, num_images, input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # 获取预测结果
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_video_ids.extend(video_ids)
            all_titles.extend(titles)
            all_dataset_folders.extend(dataset_folders)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels), np.array(all_probs), all_video_ids, all_titles, all_dataset_folders

