"""
========================================
图像编码器模块
========================================

使用预训练ResNet50提取图像特征，通过注意力机制融合多张缩略图
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    图像编码器
    
    使用预训练的ResNet50提取图像特征，并通过注意力机制融合多张缩略图的特征
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 加载预训练的ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 先在原始 ResNet 上冻结/解冻，再转为 Sequential。此前在
        # Sequential 上按 "layer4" 名称匹配会失败，导致整个主干被冻结。
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 注意力机制：学习每张图片的重要性权重
        # 用于融合多张缩略图的特征
        self.attention = nn.Sequential(
            nn.Linear(config.IMAGE_FEATURE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # 特征投影层，将图像特征映射到融合空间
        self.projection = nn.Sequential(
            nn.Linear(config.IMAGE_FEATURE_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )

    def set_layer4_trainable(self, enabled):
        """在 warmup 和微调阶段之间切换 ResNet layer4。"""
        for param in self.backbone[7].parameters():  # ResNet child index 7 is layer4
            param.requires_grad = enabled
    
    def forward(self, images, num_images):
        """
        前向传播
        
        Args:
            images: 图像张量 [batch_size, MAX_IMAGES, 3, H, W]
            num_images: 每个样本的实际图片数量 [batch_size]
            
        Returns:
            融合后的图像特征 [batch_size, HIDDEN_DIM]
        """
        batch_size, max_images, c, h, w = images.shape
        
        # 将所有图片展平处理
        # [batch_size * MAX_IMAGES, 3, H, W]
        images_flat = images.view(-1, c, h, w)
        
        # 通过ResNet提取特征
        # [batch_size * MAX_IMAGES, 2048, 1, 1]
        features = self.backbone(images_flat)
        
        # 展平特征
        # [batch_size * MAX_IMAGES, 2048]
        features = features.view(-1, self.config.IMAGE_FEATURE_DIM)
        
        # 恢复batch维度
        # [batch_size, MAX_IMAGES, 2048]
        features = features.view(batch_size, max_images, -1)
        
        # 计算注意力权重
        # [batch_size, MAX_IMAGES, 1]
        attention_weights = self.attention(features)
        
        # 创建mask，屏蔽padding的图片
        mask = torch.zeros(batch_size, max_images, 1, device=features.device)
        for i, n in enumerate(num_images):
            mask[i, :n, 0] = 1.0
        
        # 应用mask（将padding位置的注意力设为很大的负数）
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 处理全为-inf的情况（当num_images为0时）
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # 加权求和，融合多张图片的特征
        # [batch_size, 2048]
        pooled_features = (features * attention_weights).sum(dim=1)
        
        # 投影到隐藏空间
        # [batch_size, HIDDEN_DIM]
        output = self.projection(pooled_features)
        
        return output
