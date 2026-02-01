# 视频吸引力预测

多模态深度学习模型，通过视频缩略图和标题预测视频是否"好看"。

## 项目结构

```
├── 数据集1/                 # 训练数据
├── 数据集2/
├── 数据集3/
├── models/                  # 模型定义
│   ├── image_encoder.py     # 图像编码器 (ResNet50 + Attention)
│   ├── text_encoder.py      # 文本编码器 (BERT)
│   └── classifier.py        # 融合分类器
├── utils/                   # 工具函数
│   ├── trainer.py           # 训练逻辑
│   └── helpers.py           # 辅助函数
├── config.py                # 配置参数
├── dataset.py               # 数据加载
├── train.py                 # 训练脚本
├── predict.py               # 预测脚本
└── best_model.pth           # 训练好的模型
```

## 模型架构

```
视频缩略图 (1-5张)
      │
      ▼
  ResNet50 ──► Attention ──► 图像特征
                                  │
                                  ├──► 特征融合 ──► 分类结果 (好看/不好看)
                                  │
  标题 ──► BERT ────────────► 文本特征
```

## 环境配置

```bash
# 创建环境
conda create -n video_predict python=3.10
conda activate video_predict

# 安装 PyTorch (根据显卡选择)
pip install torch torchvision                                    # CPU / Mac
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
python train.py
```

### 预测

```bash
# 单个视频
python predict.py --video_folder "./数据集1/video_01" --title "视频标题"

# 批量预测数据集
python predict.py --dataset_dir "./数据集1"

# 预测全部数据集
python predict.py --predict_all

# 从 CSV 预测
python predict.py --csv_file "./data/index.csv" --data_root "./data"
```

### 在代码中调用

```python
from predict import Predictor

predictor = Predictor()  # 自动加载 best_model.pth

# 单个预测
result = predictor.predict("./数据集1/video_01", "视频标题")
print(result['label'])        # "好看" 或 "不好看"
print(result['probability'])  # 好看的概率

# 批量预测
df = predictor.predict_from_dataset("./数据集1")
```

## 配置参数

编辑 `config.py` 调整：

| 参数                      | 默认值 | 说明                      |
| ------------------------- | ------ | ------------------------- |
| `MAX_IMAGES_PER_VIDEO`    | 5      | 每视频最多图片数          |
| `IMAGE_SIZE`              | 224    | 图像尺寸                  |
| `BATCH_SIZE`              | 16     | 批次大小 (显存不够就改小) |
| `LEARNING_RATE`           | 5e-5   | 学习率                    |
| `NUM_EPOCHS`              | 30     | 训练轮数                  |
| `DROPOUT_RATE`            | 0.5    | Dropout                   |
| `EARLY_STOPPING_PATIENCE` | 7      | 早停耐心值                |

## 预测输出字段

批量预测生成的 CSV 包含：

| 字段                  | 说明                      |
| --------------------- | ------------------------- |
| `predict_probability` | 好看的概率 (0-1)          |
| `predict_label`       | 预测标签                  |
| `predict_result`      | 预测值 (1=好看, 0=不好看) |
| `predict_confidence`  | 置信度                    |
| `predict_status`      | 状态 (success/error)      |

## 常见问题

**显存不足？** → 减小 `BATCH_SIZE`

**准确率低？** → 增加训练数据，调整 `LEARNING_RATE`

## License

MIT
