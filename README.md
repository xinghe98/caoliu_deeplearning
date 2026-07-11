# 视频吸引力预测

多模态二分类项目：使用视频缩略图和标题，预测内容是否符合标注的“好看”偏好。模型由 ResNet-50 图像编码器、中文 BERT 文本编码器和融合分类头组成。

新版训练流程会按内容分组切分数据、以 PR-AUC 选择模型、校准预测概率并保存业务阈值；推理不再固定使用 `0.5` 阈值。

## 环境配置

建议使用 Python 3.10 或更高版本。

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning
python -m pip install -r requirements.txt
```

首次训练或推理需要本地存在、或能够下载以下预训练权重：

- `bert-base-chinese`
- ResNet-50 ImageNet 权重

如果机器无法访问 Hugging Face，请先在可联网机器下载并配置本地缓存后再运行。

## 数据集格式

训练数据放在项目根目录的 `数据集1` 到 `数据集5` 中。每个数据集目录必须包含 `index.csv` 和对应图片目录：

```text
数据集4/
├── index.csv
├── video_0001/
│   ├── image_01.jpg
│   └── image_02.jpg
└── video_0002/
    └── image_01.jpg
```

`index.csv` 至少应包含下列字段：

```csv
video_id,title,download_link,label
video_0001,视频标题 A,magnet:?xt=urn:btih:xxxx,1
video_0002,视频标题 B,magnet:?xt=urn:btih:yyyy,0
```

- `video_id`：必须与图片文件夹同名。
- `title`：标题文本；可以为空，但不建议为空。
- `download_link`：用于内容去重；没有可留空。
- `label`：人工真实标签，`1` 表示符合偏好，`0` 表示不符合。

训练时会过滤无图片、重复内容及标签冲突内容。相同下载链接或规范化标题会被视为同一内容组，不会跨训练集和验证集。

默认将 `数据集3` 作为锁定外部测试集；不要把它用于日常调参。新增已标注数据建议放入 `数据集4` 或 `数据集5`。

## 训练

首次使用新版代码训练：

```powershell
python train.py
```

显存不足时降低批次大小：

```powershell
python train.py --batch-size 8
```

指定总训练轮数：

```powershell
python train.py --epochs 30
```

训练分为两个阶段：前 3 轮仅训练融合与分类层，之后微调 ResNet-50 的 `layer4` 和 BERT 最后两层。模型按验证集 PR-AUC 保存最佳版本，并在验证集上选择满足默认 90% precision 目标的业务阈值。

训练完成后会生成：

- `best_model.pth`：最佳模型、阈值、温度校准参数和训练配置。
- `split_manifest.csv`：本次训练/验证/外部测试划分。
- `validation_predictions.csv`：验证集逐样本预测。
- `validation_error_cases.csv`：验证集误判集合，含假阳性/假阴性类型。
- `external_test_predictions.csv`：数据集3上的外部测试预测。
- `external_test_error_cases.csv`：外部测试误判集合。
- `evaluation_report.json`：PR-AUC、precision、recall、F0.5、Brier score 等指标。
- `training_history.json`：每轮训练损失、训练准确率、验证损失和验证 PR-AUC。
- `training_history.png`：训练曲线。

### 继续训练

仅当当前 `best_model.pth` 由新版训练脚本生成时，才可以继续训练：

```powershell
python train.py --resume --epochs 40
```

项目早期生成的旧 `best_model.pth` 没有新版融合结构和训练元数据，不能直接使用 `--resume`。保留旧模型可用于预测；首次升级训练请使用 `python train.py` 从头训练，并在开始前备份模型：

```powershell
Copy-Item best_model.pth best_model.before_retrain.pth
```

## 预测

```powershell
# 单个视频
python predict.py --video_folder ".\数据集1\video_01" --title "视频标题"

# 批量预测一个数据集
python predict.py --dataset_dir ".\数据集1"

# 从 CSV 批量预测
python predict.py --csv_file ".\data\index.csv" --data_root ".\data"
```

新版模型会自动读取 checkpoint 中保存的温度和业务阈值。旧模型则兼容使用原始阈值 `0.5`。

Python 调用：

```python
from predict import Predictor

predictor = Predictor()
result = predictor.predict('./数据集1/video_01', '视频标题')

print(result['label'])
print(result['probability'])
print(result['decision_threshold'])
```

## 预测输出

批量预测结果包含：

| 字段 | 说明 |
| --- | --- |
| `predict_probability` | 温度校准后的“好看”概率 |
| `predict_result` | 依据业务阈值得到的 0/1 结果 |
| `predict_label` | `好看` 或 `不好看` |
| `predict_confidence` | 当前预测类别的概率 |
| `predict_status` | `success` 或 `error` |

单条/API 输出额外包含 `decision_threshold` 与 `model_version`。

## 标注建议

新爬取数据应先人工标注，再加入训练集。模型预测可用于排序待标注样本，但不要直接把预测结果当成训练标签。优先复核概率接近决策阈值的样本，并定期加入模型高置信度但人工判断错误的样本；这比单纯增加大量重复样本更能提升模型对个人偏好的拟合。

## 常见问题

**显存不足？** 使用 `python train.py --batch-size 8`，必要时继续减小。

**模型下载失败？** 确认首次运行时能访问 Hugging Face，或预先准备 `bert-base-chinese` 与 ResNet 权重缓存。

**验证准确率高但实际筛选差？** 以 `evaluation_report.json` 中的 PR-AUC、precision 和 recall 为准；补充真实业务数据的人工标签后重新训练。

## License

MIT
