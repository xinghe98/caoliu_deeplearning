"""
========================================
视频吸引力预测模型 - 主训练脚本
========================================

使用方法：
    python train.py

作者: AI Assistant
日期: 2026-01-31
"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer

# 导入自定义模块
from config import Config, get_device
from dataset import load_dataset, VideoDataset
from models import MultiModalClassifier
from utils import train_one_epoch, validate, set_seed, plot_training_history


def main():
    """
    主函数：完整的训练流程
    """
    print("="*60)
    print("       视频吸引力预测模型 - 多模态深度学习训练")
    print("="*60)
    
    # 获取设备
    device = get_device()
    
    # 初始化配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    
    # ==================== 1. 加载数据 ====================
    print("\n[1/6] 加载数据集...")
    df = load_dataset(config)
    
    # ==================== 2. 划分数据集 ====================
    print("\n[2/6] 划分数据集...")
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2,           # 20%作为验证集
        stratify=df['label'],    # 分层采样，保持正负样本比例
        random_state=config.RANDOM_SEED
    )
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # ==================== 3. 初始化分词器 ====================
    print("\n[3/6] 加载BERT分词器...")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    print(f"✓ 已加载 {config.BERT_MODEL_NAME} 分词器")
    
    # ==================== 4. 创建数据加载器 ====================
    print("\n[4/6] 创建数据加载器...")
    train_dataset = VideoDataset(train_df, config, tokenizer, is_training=True)
    val_dataset = VideoDataset(val_df, config, tokenizer, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Mac上建议设为0避免多进程问题
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"✓ 训练集批次数: {len(train_loader)}")
    print(f"✓ 验证集批次数: {len(val_loader)}")
    
    # ==================== 5. 初始化模型 ====================
    print("\n[5/6] 初始化模型...")
    model = MultiModalClassifier(config).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 定义损失函数（带权重的二元交叉熵，处理类别不平衡）
    pos_weight = torch.tensor(
        [(df['label'] == 0).sum() / (df['label'] == 1).sum()],
        dtype=torch.float32,  # MPS不支持float64，必须使用float32
        device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"正样本权重: {pos_weight.item():.2f}")
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    
    # ==================== 6. 开始训练 ====================
    print("\n[6/6] 开始训练...")
    print("="*60)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, predictions, labels = validate(
            model, val_loader, criterion, device
        )
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        print(f"\n  训练损失: {train_loss:.4f} | 训练准确率: {train_acc*100:.2f}%")
        print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc*100:.2f}%")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(config.DATA_DIR, config.MODEL_SAVE_PATH))
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ! 验证损失未改善 ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
        # 早停
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n早停: 验证损失连续{config.EARLY_STOPPING_PATIENCE}个epoch未改善")
            break
    
    # 绘制训练曲线
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(config.DATA_DIR, "training_history.png")
    )
    
    # ==================== 最终评估 ====================
    print("\n" + "="*60)
    print("最终评估（使用最佳模型）")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(config.DATA_DIR, config.MODEL_SAVE_PATH))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc, final_predictions, final_labels = validate(
        model, val_loader, criterion, device
    )
    
    print("\n分类报告:")
    print(classification_report(
        final_labels, final_predictions,
        target_names=['不好看 (0)', '好看 (1)']
    ))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(final_labels, final_predictions))
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳验证准确率: {checkpoint['val_acc']*100:.2f}%")
    print(f"模型已保存到: {os.path.join(config.DATA_DIR, config.MODEL_SAVE_PATH)}")
    print("="*60)


if __name__ == "__main__":
    main()
