"""
========================================
数据集模块
========================================

包含数据加载和预处理相关的类和函数
"""

import os
import glob
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
    
    # 去重处理：根据download_link去重，保留label为1的记录
    if 'download_link' in combined_df.columns:
        # 先移除download_link为空的重复检查
        has_link = combined_df['download_link'].notna() & (combined_df['download_link'] != '')
        df_with_link = combined_df[has_link].copy()
        df_without_link = combined_df[~has_link].copy()
        
        if len(df_with_link) > 0:
            # 统计去重前的数量
            before_count = len(df_with_link)
            
            # 按download_link分组，优先保留label=1的记录
            # 方法：先按label降序排序（1在前），然后对每个download_link保留第一条
            df_with_link = df_with_link.sort_values('label', ascending=False)
            df_with_link = df_with_link.drop_duplicates(subset=['download_link'], keep='first')
            
            # 统计去重后的数量
            after_count = len(df_with_link)
            duplicates_removed = before_count - after_count
            
            if duplicates_removed > 0:
                print(f"⚠ 发现 {duplicates_removed} 条重复的下载链接，已去重（保留label=1的记录）")
        
        # 合并有链接和无链接的数据
        combined_df = pd.concat([df_with_link, df_without_link], ignore_index=True)
    
    # 去重处理：根据标题去重，防止模型记住"标题-标签"关联
    if 'title' in combined_df.columns:
        before_count = len(combined_df)
        # 按 label 降序排序（优先保留 label=1），然后去重
        combined_df = combined_df.sort_values('label', ascending=False)
        combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
        duplicates_removed = before_count - len(combined_df)
        if duplicates_removed > 0:
            print(f"⚠ 发现 {duplicates_removed} 条重复标题，已去重（保留label=1的记录）")
    
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
    
    print(f"\n{'='*50}")
    print(f"数据集统计信息:")
    print(f"{'='*50}")
    print(f"总样本数: {len(combined_df)}")
    print(f"正样本数 (好看=1): {(combined_df['label'] == 1).sum()}")
    print(f"负样本数 (不好看=0): {(combined_df['label'] == 0).sum()}")
    print(f"正负样本比例: {(combined_df['label'] == 1).sum() / len(combined_df) * 100:.2f}%")
    print(f"{'='*50}\n")
    
    return combined_df


class VideoDataset(Dataset):
    """
    视频数据集类，用于加载图像缩略图和标题文本
    
    继承自PyTorch的Dataset类，实现__len__和__getitem__方法
    """
    
    def __init__(self, dataframe, config, tokenizer, transform=None, is_training=True):
        """
        初始化数据集
        
        Args:
            dataframe: 包含视频信息的DataFrame
            config: 配置对象
            tokenizer: BERT分词器
            transform: 图像变换操作
            is_training: 是否为训练模式
        """
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        
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
            glob.glob(os.path.join(folder_path, "*.png"))
        )
        
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
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像、文本和标签的字典
        """
        row = self.df.iloc[idx]
        
        # 构建图片文件夹路径
        folder_path = os.path.join(
            self.config.DATA_DIR,
            row['dataset_folder'],
            row['video_id']
        )
        
        # 加载图像
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
            'title': str(row['title'])
        }
