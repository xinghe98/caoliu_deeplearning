# 视频吸引力预测模型

## 📖 项目简介

本项目是一个**多模态深度学习模型**，通过结合视频缩略图（图像）和标题（文本）来预测视频是否"好看"。

## 📁 项目结构

```
深度学习/
├── 数据集1/                    # 数据集1
├── 数据集2/                    # 数据集2
│
├── models/                     # 模型模块
│   ├── __init__.py            # 模块导出
│   ├── image_encoder.py       # 图像编码器 (ResNet50 + 注意力机制)
│   ├── text_encoder.py        # 文本编码器 (BERT)
│   └── classifier.py          # 多模态分类器
│
├── utils/                      # 工具模块
│   ├── __init__.py            # 模块导出
│   ├── trainer.py             # 训练和验证函数
│   └── helpers.py             # 辅助函数（随机种子、绘图）
│
├── config.py                   # 配置参数
├── dataset.py                  # 数据集加载和预处理
├── train.py                    # 🔥 主训练脚本
├── predict.py                  # 推理/预测脚本
├── requirements.txt            # 依赖包列表
└── README.md                   # 本文档
```

## 🏗️ 模型架构

```
                    ┌─────────────────┐
                    │   视频缩略图    │
                    │ (1-5张图片)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   ResNet50      │ ← 预训练的图像特征提取器
                    │   (image_encoder.py)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  注意力机制     │ ← 学习每张图片的重要性权重
                    └────────┬────────┘
                             │
         图像特征 ───────────┤
                             │
                    ┌────────▼────────┐
                    │   特征融合      │ ← classifier.py
                    └────────┬────────┘
                             │
         文本特征 ───────────┘
              ↑
         ┌────┴────┐
         │  BERT   │ ← text_encoder.py
         └─────────┘
              ↑
         ┌────┴────┐
         │  标题   │
         └─────────┘
                             │
                    ┌────────▼────────┐
                    │   预测结果      │
                    │   0=不好看      │
                    │   1=好看        │
                    └─────────────────┘
```

## 🛠️ 环境配置

### 1. 创建Python环境

```bash
conda create -n video_predict python=3.10
conda activate video_predict
```

### 2. 安装PyTorch

**Mac (Apple Silicon M1/M2/M3):**
```bash
pip install torch torchvision
```

**Linux/Windows (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 训练模型

```bash
cd /Users/xinghe/Downloads/深度学习
python train.py
```

### 预测新视频

```bash
python predict.py \
    --video_folder "/path/to/video_thumbnails" \
    --title "视频标题"
```

## ⚙️ 配置参数

编辑 `config.py` 可以调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_IMAGES_PER_VIDEO` | 5 | 每个视频最多使用的缩略图数量 |
| `IMAGE_SIZE` | 224 | 图像输入尺寸 |
| `HIDDEN_DIM` | 512 | 融合层隐藏层维度 |
| `DROPOUT_RATE` | 0.3 | Dropout比率 |
| `BATCH_SIZE` | 16 | 批次大小（内存不足时减小） |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `NUM_EPOCHS` | 20 | 训练轮数 |
| `EARLY_STOPPING_PATIENCE` | 5 | 早停的耐心轮数 |
| `MAX_TEXT_LENGTH` | 128 | 标题最大长度 |

## 📊 模块说明

### models/image_encoder.py
- 使用预训练 ResNet50 提取每张图片特征
- 通过注意力机制自动学习每张图片的重要性
- 支持变长图片数量（1-5张不等）

### models/text_encoder.py
- 使用预训练 bert-base-chinese 模型
- 提取标题的语义特征
- 只微调最后2层Transformer

### models/classifier.py
- 融合图像和文本特征
- 多层全连接网络进行分类

### utils/trainer.py
- `train_one_epoch()`: 训练一个epoch
- `validate()`: 验证模型性能

### utils/helpers.py
- `set_seed()`: 设置随机种子确保可复现
- `plot_training_history()`: 绘制训练曲线

## 🔧 常见问题

### Q: 内存不足怎么办？
A: 在 `config.py` 中减小 `BATCH_SIZE`（如改为8或4）

### Q: 如何在代码中使用模型？

```python
from predict import Predictor

# 初始化预测器
predictor = Predictor("best_model.pth")

# 预测单个视频
result = predictor.predict(
    folder_path="/path/to/video_folder",
    title="视频标题"
)

print(f"预测结果: {result['label']}")
print(f"预测概率: {result['probability']:.2%}")
```

## 📜 许可证

MIT License
