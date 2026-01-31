"""
========================================
视频吸引力预测模型 - 推理脚本
========================================

使用方法：
    python predict.py --video_folder <视频缩略图文件夹路径> --title <视频标题>

作者: AI Assistant
日期: 2026-01-31
"""

import os
import glob
import argparse
from PIL import Image
import torch
from torchvision import transforms
from transformers import BertTokenizer

# 导入自定义模块
from config import Config, get_device
from models import MultiModalClassifier


class Predictor:
    """
    预测器类，封装模型加载和预测逻辑
    """
    
    def __init__(self, model_path, config=None):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config if config else Config()
        self.device = get_device()
        
        # 初始化模型
        self.model = MultiModalClassifier(self.config).to(self.device)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ 模型加载成功，验证准确率: {checkpoint['val_acc']*100:.2f}%")
        
        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_images(self, folder_path):
        """
        加载视频缩略图
        
        Args:
            folder_path: 图片文件夹路径
            
        Returns:
            图像张量和有效图片数量
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
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_tensor = self.transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"警告: 无法加载图片 {img_path}: {e}")
                continue
        
        if len(images) == 0:
            raise ValueError(f"无法从 {folder_path} 加载任何图片")
        
        num_images = len(images)
        
        # 填充到固定数量
        while len(images) < self.config.MAX_IMAGES_PER_VIDEO:
            images.append(torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        
        images_tensor = torch.stack(images).unsqueeze(0)  # 添加batch维度
        
        return images_tensor, num_images
    
    def _encode_text(self, title):
        """
        编码标题文本
        
        Args:
            title: 视频标题
            
        Returns:
            input_ids和attention_mask张量
        """
        if not title or title.strip() == "":
            title = "[PAD]"
        
        encoding = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_TEXT_LENGTH,
            return_tensors='pt'
        )
        
        return encoding['input_ids'], encoding['attention_mask']
    
    def predict(self, folder_path, title):
        """
        对单个视频进行预测
        
        Args:
            folder_path: 视频缩略图文件夹路径
            title: 视频标题
            
        Returns:
            预测结果字典，包含预测类别、概率和判断
        """
        # 加载图像
        images, num_images = self._load_images(folder_path)
        images = images.to(self.device)
        
        # 编码文本
        input_ids, attention_mask = self._encode_text(title)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model(images, [num_images], input_ids, attention_mask)
            probability = torch.sigmoid(logits).item()
        
        # 判断结果
        is_attractive = probability > 0.5
        
        return {
            'probability': probability,
            'prediction': 1 if is_attractive else 0,
            'label': '好看' if is_attractive else '不好看',
            'confidence': probability if is_attractive else (1 - probability)
        }
    
    def predict_batch(self, video_list):
        """
        批量预测多个视频
        
        Args:
            video_list: 视频信息列表，每个元素是 (folder_path, title) 元组
            
        Returns:
            预测结果列表
        """
        results = []
        for folder_path, title in video_list:
            try:
                result = self.predict(folder_path, title)
                result['folder_path'] = folder_path
                result['title'] = title
                results.append(result)
            except Exception as e:
                results.append({
                    'folder_path': folder_path,
                    'title': title,
                    'error': str(e)
                })
        
        return results


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(description='视频吸引力预测')
    parser.add_argument('--video_folder', type=str, required=True,
                        help='视频缩略图文件夹路径')
    parser.add_argument('--title', type=str, required=True,
                        help='视频标题')
    parser.add_argument('--model_path', type=str, 
                        default='/Users/xinghe/Downloads/深度学习/best_model.pth',
                        help='模型权重文件路径')
    
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = Predictor(args.model_path)
    
    # 进行预测
    result = predictor.predict(args.video_folder, args.title)
    
    # 输出结果
    print("\n" + "="*50)
    print("预测结果")
    print("="*50)
    print(f"视频标题: {args.title}")
    print(f"预测结果: {result['label']}")
    print(f"预测概率: {result['probability']:.2%}")
    print(f"置信度: {result['confidence']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
