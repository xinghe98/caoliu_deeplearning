"""
========================================
配置参数模块
========================================

集中管理所有超参数和配置项
"""

import torch


class Config:
    """
    训练配置类，集中管理所有超参数
    """
    # 数据相关
    DATA_DIR = "/Users/xinghe/Downloads/深度学习"  # 数据集根目录
    DATASET_FOLDERS = ["数据集1", "数据集2"]        # 数据集文件夹名
    MAX_IMAGES_PER_VIDEO = 5                       # 每个视频最多使用的缩略图数量
    
    # 模型相关
    IMAGE_SIZE = 224                               # 图像输入尺寸（ResNet要求224x224）
    IMAGE_FEATURE_DIM = 2048                       # ResNet50输出的特征维度
    TEXT_FEATURE_DIM = 768                         # BERT输出的特征维度
    HIDDEN_DIM = 512                               # 融合层隐藏层维度
    DROPOUT_RATE = 0.3                             # Dropout比率，防止过拟合
    
    # 训练相关
    BATCH_SIZE = 16                                # 批次大小
    LEARNING_RATE = 1e-4                           # 学习率
    NUM_EPOCHS = 20                                # 训练轮数
    WEIGHT_DECAY = 1e-5                            # L2正则化系数
    EARLY_STOPPING_PATIENCE = 5                    # 早停的耐心轮数
    
    # BERT相关
    BERT_MODEL_NAME = 'bert-base-chinese'          # 使用中文BERT
    MAX_TEXT_LENGTH = 128                          # 文本最大长度
    
    # 保存相关
    MODEL_SAVE_PATH = "best_model.pth"             # 最佳模型保存路径
    
    # 随机种子
    RANDOM_SEED = 42


def get_device():
    """
    获取计算设备
    优先使用GPU（CUDA），其次MPS（Apple Silicon），最后CPU
    
    Returns:
        torch.device: 计算设备
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✓ 使用 CUDA GPU 进行训练")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ 使用 Apple Silicon MPS 进行训练")
    else:
        device = torch.device('cpu')
        print("✓ 使用 CPU 进行训练")
    return device
