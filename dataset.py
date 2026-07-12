"""
========================================
数据集模块
========================================

包含数据加载和预处理相关的类和函数
"""

import os
import glob
import random
import re
import shutil
import stat
import tempfile
import zipfile
import weakref
from contextlib import AbstractContextManager
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold


def _normalise_title(value):
    """Create a conservative duplicate key without changing the model text."""
    if pd.isna(value):
        return ""
    return re.sub(r'\s+', '', str(value).strip().lower())


def _add_group_ids(dataframe):
    """Group exact duplicate links/titles so leakage cannot cross a split."""
    df = dataframe.copy()
    links = df.get('download_link', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    titles = df.get('title', pd.Series('', index=df.index)).map(_normalise_title)
    df['_group_id'] = [f'link:{link}' if link else f'title:{title}' for link, title in zip(links, titles)]
    # Rows without either identifier remain isolated rather than being grouped together.
    empty = df['_group_id'].isin(['link:', 'title:'])
    df.loc[empty, '_group_id'] = [f'row:{index}' for index in df.index[empty]]
    return df


def split_train_validation(dataframe, validation_fraction=0.2, random_state=42):
    """Deterministic stratified group split that keeps duplicate content together."""
    if '_group_id' not in dataframe:
        raise ValueError('Dataframe must contain _group_id')
    group_labels = dataframe.drop_duplicates('_group_id').groupby('label').size()
    group_count = dataframe['_group_id'].nunique()
    n_splits = min(5, group_count, int(group_labels.min()))
    if n_splits < 2:
        raise ValueError('至少需要两个不同内容组才能切分训练集和验证集')
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    target = validation_fraction
    best = None
    for train_indices, val_indices in splitter.split(dataframe, dataframe['label'], dataframe['_group_id']):
        ratio_distance = abs(len(val_indices) / len(dataframe) - target)
        class_distance = abs(dataframe.iloc[val_indices]['label'].mean() - dataframe['label'].mean())
        score = ratio_distance + class_distance
        if best is None or score < best[0]:
            best = (score, train_indices, val_indices)
    _, train_indices, val_indices = best
    return (dataframe.iloc[train_indices].reset_index(drop=True),
            dataframe.iloc[val_indices].reset_index(drop=True))


def load_dataset(config):
    """
    加载所有数据集的CSV文件并合并
    
    Args:
        config: 配置对象
        
    Returns:
        合并后的DataFrame，包含所有视频的信息
    """
    all_data = []
    
    for folder in config.DATASET_FOLDERS:
        csv_path = os.path.join(config.DATA_DIR, folder, "index.csv")
        if os.path.exists(csv_path):
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            # 添加数据集来源列，方便后续定位图片
            df['dataset_folder'] = folder
            all_data.append(df)
            print(f"✓ 从 {folder} 加载了 {len(df)} 条数据")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 数据清洗：移除标签为空的行
    combined_df = combined_df.dropna(subset=['label'])
    # 确保标签是整数类型
    combined_df['label'] = combined_df['label'].astype(int)
    
    # 过滤掉对应图片文件夹不存在的记录
    valid_indices = []
    for idx, row in combined_df.iterrows():
        folder_path = os.path.join(
            config.DATA_DIR, 
            row['dataset_folder'], 
            row['video_id']
        )
        if os.path.exists(folder_path):
            # 检查文件夹中是否有图片
            images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                     glob.glob(os.path.join(folder_path, "*.gif")) + \
                     glob.glob(os.path.join(folder_path, "*.png")) + \
                     glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                     glob.glob(os.path.join(folder_path, "*.webp"))
            if len(images) > 0:
                valid_indices.append(idx)
    
    combined_df = combined_df.loc[valid_indices].reset_index(drop=True)
    combined_df = _add_group_ids(combined_df)

    # 同一内容组出现相反标签时无法得到可靠监督信号；明确排除而非偏向正类。
    group_label_counts = combined_df.groupby('_group_id')['label'].nunique()
    conflicting_groups = group_label_counts[group_label_counts > 1].index
    if len(conflicting_groups):
        conflicting_rows = combined_df[combined_df['_group_id'].isin(conflicting_groups)]
        print(f"⚠ 排除 {len(conflicting_rows)} 条标签冲突记录（{len(conflicting_groups)} 个内容组），请人工复核")
        combined_df = combined_df[~combined_df['_group_id'].isin(conflicting_groups)]

    # 保留每个内容组的一条记录，避免重复样本在训练中被放大。
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['_group_id'], keep='first').reset_index(drop=True)
    if before_dedup != len(combined_df):
        print(f"✓ 移除 {before_dedup - len(combined_df)} 条重复内容记录")
    
    print(f"\n{'='*50}")
    print(f"数据集统计信息:")
    print(f"{'='*50}")
    print(f"总样本数: {len(combined_df)}")
    print(f"正样本数 (好看=1): {(combined_df['label'] == 1).sum()}")
    print(f"负样本数 (不好看=0): {(combined_df['label'] == 0).sum()}")
    print(f"正负样本比例: {(combined_df['label'] == 1).sum() / len(combined_df) * 100:.2f}%")
    print(f"{'='*50}\n")
    
    return combined_df


TRAINING_PACKAGE_MAX_FILES = 200
TRAINING_PACKAGE_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_ALLOWED_PACKAGE_SPLITS = frozenset({'train', 'validation', 'production_shadow_test', 'external_test'})


class TrainingPackage(AbstractContextManager):
    """A loaded training package whose temporary extraction is explicitly owned."""

    def __init__(self, dataframe, root: str, temp_dir: str | None):
        self.dataframe = dataframe
        self.root = root
        self._temp_dir = temp_dir
        self._finalizer = weakref.finalize(self, shutil.rmtree, temp_dir, ignore_errors=True) if temp_dir else None

    def close(self) -> None:
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        if self._finalizer:
            self._finalizer.detach()

    def __getitem__(self, key):
        return self.dataframe[key]

    def __getattr__(self, name):
        return getattr(self.dataframe, name)

    def __exit__(self, *_exc) -> None:
        self.close()


def _safe_extract_training_package(archive: zipfile.ZipFile, target: Path) -> None:
    members = archive.infolist()
    if len(members) > TRAINING_PACKAGE_MAX_FILES:
        raise ValueError('训练包文件数量超过限制')
    if len({member.filename for member in members}) != len(members):
        raise ValueError('训练包包含重复文件名')
    total = 0
    for member in members:
        name = member.filename
        destination = (target / name).resolve()
        if not name or Path(name).is_absolute() or not destination.is_relative_to(target.resolve()):
            raise ValueError('训练包包含非法路径')
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError('训练包不能包含符号链接')
        total += member.file_size
        if total > TRAINING_PACKAGE_MAX_UNCOMPRESSED_BYTES:
            raise ValueError('训练包解压后体积超过限制')
    archive.extractall(target)


def _validate_split_manifest(df, split_df):
    required = {'content_id', 'label', 'image_paths'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"训练包 manifest.csv 缺少列: {', '.join(sorted(missing))}")
    if df['content_id'].isna().any() or df['content_id'].astype(str).str.strip().eq('').any() or df['content_id'].duplicated().any():
        raise ValueError('训练包 content_id 必须非空且唯一')
    if split_df is not None:
        if not {'content_id', 'split'}.issubset(split_df.columns):
            raise ValueError('split_manifest.csv 必须包含 content_id 与 split 列')
        if split_df['content_id'].isna().any() or split_df['content_id'].duplicated().any():
            raise ValueError('split_manifest.csv 的 content_id 必须非空且唯一')
        split_map = split_df.set_index('content_id')['split']
        missing_ids = set(df['content_id']) - set(split_map.index)
        extra_ids = set(split_map.index) - set(df['content_id'])
        if missing_ids or extra_ids:
            raise ValueError('split_manifest.csv 必须与 manifest.csv 的 content_id 完全一致')
        df = df.drop(columns=['split'], errors='ignore').copy()
        df['split'] = df['content_id'].map(split_map)
    if 'split' not in df.columns:
        raise ValueError('训练包缺少 split 列，且没有 split_manifest.csv')
    if df['split'].isna().any() or not set(df['split'].astype(str)).issubset(_ALLOWED_PACKAGE_SPLITS):
        raise ValueError('训练包包含缺失或不支持的 split')
    labels = pd.to_numeric(df['label'], errors='coerce')
    if labels.isna().any() or not set(labels.astype(int)).issubset({0, 1}) or not (labels == labels.astype(int)).all():
        raise ValueError('训练包 label 必须为 0 或 1')
    df = df.copy()
    df['label'] = labels.astype(int)
    return df


def load_training_package(package_path, split_manifest=None) -> TrainingPackage:
    """Load a training ZIP/directory; use it as a context manager to clean ZIP extraction."""
    package_path = os.path.abspath(package_path)
    temp_dir = None
    if os.path.isfile(package_path) and package_path.lower().endswith('.zip'):
        temp_dir = tempfile.mkdtemp(prefix='training_package_')
        try:
            with zipfile.ZipFile(package_path) as archive:
                _safe_extract_training_package(archive, Path(temp_dir))
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        root = temp_dir
    else:
        root = package_path

    manifest_path = os.path.join(root, 'manifest.csv')
    if not os.path.isfile(manifest_path):
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileNotFoundError(f'训练包缺少 manifest.csv: {manifest_path}')
    try:
        df = pd.read_csv(manifest_path, encoding='utf-8-sig')
        split_path = split_manifest or os.path.join(root, 'split_manifest.csv')
        split_df = pd.read_csv(split_path, encoding='utf-8-sig') if os.path.isfile(split_path) else None
        df = _validate_split_manifest(df, split_df)
    except Exception:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    records = []
    for _, row in df.iterrows():
        split = str(row['split'])
        if split in {'external_test'}:
            # Allowed for evaluation loaders only.
            pass
        image_field = str(row.get('image_paths') or '')
        rel_paths = [part.strip() for part in image_field.split(';') if part.strip()]
        abs_paths = []
        for rel_path in rel_paths:
            path = (Path(root) / rel_path).resolve()
            if Path(rel_path).is_absolute() or not path.is_relative_to(Path(root).resolve()):
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(f'训练包图片路径非法: {rel_path}')
            if not path.is_file():
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(f'训练包引用的图片不存在: {rel_path}')
            abs_paths.append(str(path))
        if not abs_paths:
            # Fall back to content image directory.
            folder = os.path.join(root, 'images', str(row.get('content_id', '')))
            abs_paths = sorted(
                glob.glob(os.path.join(folder, '*.jpg'))
                + glob.glob(os.path.join(folder, '*.jpeg'))
                + glob.glob(os.path.join(folder, '*.png'))
                + glob.glob(os.path.join(folder, '*.gif'))
                + glob.glob(os.path.join(folder, '*.webp'))
            )
        if not abs_paths:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"训练包样本缺少图片: {row['content_id']}")
        records.append({
            'dataset_folder': 'package',
            'video_id': str(row.get('content_id') or row.get('video_id')),
            'title': row.get('title', ''),
            'label': int(row['label']),
            'download_link': row.get('download_link', ''),
            'split': split,
            'image_paths': abs_paths,
            '_group_id': str(row.get('content_group_id') or row.get('content_id') or row.get('video_id')),
            '_package_root': root,
            '_package_temp': temp_dir,
        })
    if not records:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError('训练包没有可用样本')
    return TrainingPackage(pd.DataFrame(records), root, temp_dir)


class VideoDataset(Dataset):
    """
    视频数据集类，用于加载图像缩略图和标题文本
    
    继承自PyTorch的Dataset类，实现__len__和__getitem__方法
    """
    
    def __init__(self, dataframe, config, tokenizer, transform=None, is_training=True, package_mode=False):
        """
        初始化数据集
        
        Args:
            dataframe: 包含视频信息的DataFrame
            config: 配置对象
            tokenizer: BERT分词器
            transform: 图像变换操作
            is_training: 是否为训练模式
            package_mode: 是否从训练包显式路径加载图片
        """
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.package_mode = package_mode or ('image_paths' in self.df.columns)
        
        # 定义图像变换
        if transform is None:
            if is_training:
                # 训练时使用数据增强
                self.transform = transforms.Compose([
                    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                    transforms.RandomRotation(10),           # 随机旋转±10度
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet均值
                        std=[0.229, 0.224, 0.225]    # ImageNet标准差
                    )
                ])
            else:
                # 验证/测试时不使用数据增强
                self.transform = transforms.Compose([
                    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.df)
    
    def _load_images(self, folder_path):
        """
        加载视频缩略图
        
        Args:
            folder_path: 图片文件夹路径
            
        Returns:
            图像张量列表和有效图片数量
        """
        # 获取所有图片路径（支持jpg、gif、png混合）
        image_paths = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.gif")) +
            glob.glob(os.path.join(folder_path, "*.png")) +
            glob.glob(os.path.join(folder_path, "*.jpeg")) +
            glob.glob(os.path.join(folder_path, "*.webp"))
        )
        
        # 对抗性数据增强：训练时随机将多图样本裁剪为单图
        # 这样可以防止模型依赖"图片数量"这个虚假特征
        use_single_image = False
        if (self.is_training and 
            hasattr(self.config, 'USE_ADVERSARIAL_AUGMENT') and 
            self.config.USE_ADVERSARIAL_AUGMENT and
            len(image_paths) > 1):
            # 以配置的概率随机裁剪为单图
            drop_prob = getattr(self.config, 'SINGLE_IMAGE_DROP_PROB', 0.3)
            if random.random() < drop_prob:
                use_single_image = True
                # 随机选择一张图片
                image_paths = [random.choice(image_paths)]
        
        images = []
        for img_path in image_paths[:self.config.MAX_IMAGES_PER_VIDEO]:
            try:
                # 打开图片并转换为RGB格式（处理GIF取第一帧）
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # 应用变换
                img_tensor = self.transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"警告: 无法加载图片 {img_path}: {e}")
                continue
        
        # 如果没有成功加载任何图片，创建一个空白图片
        if len(images) == 0:
            blank_img = torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
            images.append(blank_img)
        
        # 记录实际图片数量
        num_images = len(images)
        
        # 填充到固定数量的图片
        while len(images) < self.config.MAX_IMAGES_PER_VIDEO:
            # 用零张量填充
            images.append(torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        
        # 堆叠成一个张量 [MAX_IMAGES, 3, H, W]
        images_tensor = torch.stack(images)
        
        return images_tensor, num_images
    
    
    def _encode_text(self, title):
        """
        使用BERT分词器编码标题文本
        
        Args:
            title: 视频标题字符串
            
        Returns:
            input_ids, attention_mask 张量
        """
        # 处理空标题
        if pd.isna(title) or title.strip() == "":
            title = "[PAD]"
        
        # BERT分词
        encoding = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_TEXT_LENGTH,
            return_tensors='pt'
        )
        
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
    
    def _load_images_from_path_list(self, image_paths):
        images = []
        paths = list(image_paths) if isinstance(image_paths, (list, tuple)) else []
        for img_path in paths[: self.config.MAX_IMAGES_PER_VIDEO]:
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(self.transform(img))
            except Exception as exc:
                print(f'警告: 无法加载图片 {img_path}: {exc}')
                continue
        if len(images) == 0:
            images.append(torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        num_images = len(images)
        while len(images) < self.config.MAX_IMAGES_PER_VIDEO:
            images.append(torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        return torch.stack(images), num_images

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像、文本和标签的字典
        """
        row = self.df.iloc[idx]

        if self.package_mode and 'image_paths' in row and isinstance(row['image_paths'], (list, tuple)):
            images, num_images = self._load_images_from_path_list(row['image_paths'])
        else:
            folder_path = os.path.join(
                self.config.DATA_DIR,
                row['dataset_folder'],
                row['video_id']
            )
            images, num_images = self._load_images(folder_path)
        
        # 编码文本
        input_ids, attention_mask = self._encode_text(row['title'])
        
        # 获取标签
        label = row['label']
        
        return {
            'images': images,                        # [MAX_IMAGES, 3, H, W]
            'num_images': num_images,                # 实际图片数量
            'input_ids': input_ids,                  # [MAX_TEXT_LENGTH]
            'attention_mask': attention_mask,        # [MAX_TEXT_LENGTH]
            'label': torch.tensor(label, dtype=torch.float32),
            'video_id': row['video_id'],
            'title': str(row['title']),
            'dataset_folder': row['dataset_folder']
        }
