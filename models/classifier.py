"""
========================================
多模态分类器模块
========================================

融合图像和文本特征，进行二分类预测
"""

import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class MultiModalClassifier(nn.Module):
    """
    多模态分类器
    
    融合图像和文本特征，进行二分类预测
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 图像编码器
        self.image_encoder = ImageEncoder(config)
        
        # 文本编码器
        self.text_encoder = TextEncoder(config)
        
        # 融合层
        # 输入维度 = 图像特征维度 + 文本特征维度 = 2 * HIDDEN_DIM
        self.fusion = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )
        
        # 分类头
        self.classifier = nn.Linear(config.HIDDEN_DIM // 2, 1)
    
    def forward(self, images, num_images, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            images: 图像张量 [batch_size, MAX_IMAGES, 3, H, W]
            num_images: 实际图片数量 [batch_size]
            input_ids: BERT输入ID [batch_size, MAX_TEXT_LENGTH]
            attention_mask: 注意力掩码 [batch_size, MAX_TEXT_LENGTH]
            
        Returns:
            分类logits [batch_size, 1]
        """
        # 提取图像特征
        image_features = self.image_encoder(images, num_images)
        
        # 提取文本特征
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # 特征融合（拼接）
        # [batch_size, HIDDEN_DIM * 2]
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # 融合层处理
        fused = self.fusion(combined_features)
        
        # 分类预测
        logits = self.classifier(fused)
        
        return logits
