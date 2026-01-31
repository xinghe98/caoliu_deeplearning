"""
========================================
自定义损失函数模块
========================================

包含处理类别不平衡的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - 专门用于处理类别不平衡问题的损失函数
    
    论文: "Focal Loss for Dense Object Detection" (RetinaNet)
    https://arxiv.org/abs/1708.02002
    
    核心思想: 降低易分类样本的权重，让模型更关注难分类的样本
    
    公式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: 正样本的权重因子，用于平衡正负样本。
               如果正样本少，设置 alpha > 0.5
        gamma: 聚焦参数，控制易分类样本的下调程度。
               gamma=0 时退化为普通交叉熵
               gamma=2 是论文推荐值
        reduction: 'none' | 'mean' | 'sum'
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型的原始输出 (logits)，形状 [batch_size]
            targets: 真实标签，形状 [batch_size]，值为 0 或 1
        """
        # 确保输入形状一致
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算概率
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算 p_t
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, pt = 1-p if y=0
        
        # 计算 alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算 Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    """
    带标签平滑的二元交叉熵损失
    
    标签平滑可以防止模型过度自信，提高泛化能力
    
    Args:
        smoothing: 平滑系数，范围 [0, 1]
                   0 表示不平滑（原始标签）
                   0.1 表示将 0->0.05, 1->0.95
        pos_weight: 正样本权重，用于处理类别不平衡
    """
    
    def __init__(self, smoothing=0.1, pos_weight=None):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # 应用标签平滑
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                inputs, targets_smooth, pos_weight=self.pos_weight
            )
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets_smooth)


class CombinedLoss(nn.Module):
    """
    组合损失函数：Focal Loss + Label Smoothing BCE Loss
    
    综合两种方法的优点：
    1. Focal Loss 处理类别不平衡
    2. Label Smoothing 防止过拟合
    
    Args:
        alpha: Focal Loss 的正样本权重
        gamma: Focal Loss 的聚焦参数
        smoothing: 标签平滑系数
        focal_weight: Focal Loss 的权重
        bce_weight: BCE Loss 的权重
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.1, 
                 focal_weight=0.7, bce_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_loss = LabelSmoothingBCELoss(smoothing=smoothing)
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.focal_weight * focal + self.bce_weight * bce


def get_class_weights(labels, device):
    """
    计算类别权重，用于WeightedRandomSampler
    
    Args:
        labels: 所有样本的标签
        device: 计算设备
    
    Returns:
        每个样本的权重
    """
    import numpy as np
    
    class_counts = np.bincount(labels.astype(int))
    n_samples = len(labels)
    n_classes = len(class_counts)
    
    # 计算每个类别的权重（样本越少，权重越大）
    class_weights = n_samples / (n_classes * class_counts)
    
    # 为每个样本分配其类别的权重
    sample_weights = class_weights[labels.astype(int)]
    
    return torch.DoubleTensor(sample_weights)
