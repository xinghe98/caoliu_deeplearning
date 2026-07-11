"""
========================================
配置参数模块
========================================

集中管理所有超参数和配置项
"""

import os
import torch


class Config:
    """
    训练配置类，集中管理所有超参数
    """
    # 数据相关
    # 动态获取脚本所在目录作为数据根目录，兼容本地和服务器环境
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDERS = ["数据集1", "数据集2", "数据集3", "数据集4", "数据集5"]  # 数据集文件夹名
    MAX_IMAGES_PER_VIDEO = 5                       # 每个视频最多使用的缩略图数量
    
    # 模型相关
    IMAGE_SIZE = 224                               # 图像输入尺寸（ResNet要求224x224）
    IMAGE_FEATURE_DIM = 2048                       # ResNet50输出的特征维度
    TEXT_FEATURE_DIM = 768                         # BERT输出的特征维度
    HIDDEN_DIM = 512                               # 融合层隐藏层维度
    DROPOUT_RATE = 0.3                             # 降低分类头欠拟合风险
    NORMALIZE_MODALITIES = True                    # 图文特征分别归一化后再融合
    
    # 训练相关
    BATCH_SIZE = 32                                # 批次大小（增大以稳定梯度）
    LEARNING_RATE = 5e-5                           # 分类头学习率
    BACKBONE_LEARNING_RATE = 1e-5                  # ResNet layer4 / BERT后两层学习率
    WARMUP_EPOCHS = 3                              # 仅训练分类头的轮数
    NUM_EPOCHS = 23                                # warmup 后最多再训练20轮
    WEIGHT_DECAY = 1e-4                            # L2正则化系数（提高防止过拟合）
    EARLY_STOPPING_PATIENCE = 5                    # 以 PR-AUC 早停
    
    # 使用标准 BCE 保持概率可校准；类别取舍由验证集阈值决定
    USE_FOCAL_LOSS = False
    USE_WEIGHTED_SAMPLER = False
    TARGET_PRECISION = 0.90                        # 默认“好看”筛选的最低精确率
    VALIDATION_FRACTION = 0.2
    EXTERNAL_TEST_FOLDERS = ["数据集3"]            # 训练期间完全隔离的外部测试集
    
    # 对抗性数据增强（防止模型依赖图片数量判断）
    USE_ADVERSARIAL_AUGMENT = False                 # 是否启用对抗性数据增强
    SINGLE_IMAGE_DROP_PROB = 0.1                   # 随机裁剪为单图的概率
    
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
