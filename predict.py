"""
视频吸引力预测 - 推理脚本

用法：
    单个预测:   python predict.py --video_folder ./数据集1/video_01 --title "标题"
    批量预测:   python predict.py --dataset_dir ./数据集1
    全部数据集: python predict.py --predict_all
"""

import os
import glob
import argparse
from datetime import datetime
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from transformers import BertTokenizer
from tqdm import tqdm

from config import Config, get_device
from models import MultiModalClassifier


class Predictor:
    """模型预测器"""
    
    def __init__(self, model_path=None, config=None):
        self.config = config if config else Config()
        self.device = get_device()
        
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'best_model.pth'
            )
        
        self.model = MultiModalClassifier(self.config).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        val_acc = checkpoint.get('val_acc', 0)
        print(f"✓ 模型加载成功，验证准确率: {val_acc*100:.2f}%")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_images(self, folder_path):
        """加载缩略图"""
        image_paths = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.jpeg")) +
            glob.glob(os.path.join(folder_path, "*.gif")) +
            glob.glob(os.path.join(folder_path, "*.png")) +
            glob.glob(os.path.join(folder_path, "*.webp"))
        )
        
        images = []
        for img_path in image_paths[:self.config.MAX_IMAGES_PER_VIDEO]:
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(self.transform(img))
            except:
                continue
        
        if len(images) == 0:
            raise ValueError(f"无法从 {folder_path} 加载图片")
        
        num_images = len(images)
        
        # 补齐到固定数量
        while len(images) < self.config.MAX_IMAGES_PER_VIDEO:
            images.append(torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        
        return torch.stack(images).unsqueeze(0), num_images
    
    def _encode_text(self, title):
        """文本编码"""
        if not title or (isinstance(title, str) and title.strip() == "") or pd.isna(title):
            title = "[PAD]"
        
        encoding = self.tokenizer(
            str(title),
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_TEXT_LENGTH,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']
    
    def predict(self, folder_path, title):
        """单个预测"""
        images, num_images = self._load_images(folder_path)
        images = images.to(self.device)
        
        input_ids, attention_mask = self._encode_text(title)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            logits = self.model(images, [num_images], input_ids, attention_mask)
            prob = torch.sigmoid(logits).item()
        
        is_good = prob > 0.5
        return {
            'probability': prob,
            'probability_good': prob,
            'probability_bad': 1 - prob,
            'prediction': 1 if is_good else 0,
            'label': '好看' if is_good else '不好看',
            'confidence': prob if is_good else (1 - prob)
        }
    
    def predict_batch(self, video_list, show_progress=True):
        """批量预测"""
        results = []
        iterator = tqdm(video_list, desc="预测中") if show_progress else video_list
        
        for folder_path, title in iterator:
            try:
                result = self.predict(folder_path, title)
                result['folder_path'] = folder_path
                result['title'] = title
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'folder_path': folder_path,
                    'title': title,
                    'status': 'error',
                    'error': str(e)
                })
        return results
    
    def predict_from_csv(self, csv_path, data_root=None, output_path=None, 
                         video_id_col='video_id', title_col='title'):
        """从CSV文件批量预测"""
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✓ 加载 {len(df)} 条数据")
        
        if data_root is None:
            data_root = os.path.dirname(csv_path)
        
        video_list = []
        for _, row in df.iterrows():
            folder_path = os.path.join(data_root, row[video_id_col])
            video_list.append((folder_path, row.get(title_col, "")))
        
        results = self.predict_batch(video_list)
        
        result_df = df.copy()
        result_df['predict_probability'] = [r.get('probability', None) for r in results]
        result_df['predict_label'] = [r.get('label', 'error') for r in results]
        result_df['predict_result'] = [r.get('prediction', None) for r in results]
        result_df['predict_confidence'] = [r.get('confidence', None) for r in results]
        result_df['predict_status'] = [r.get('status', 'unknown') for r in results]
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = csv_path.replace('.csv', f'_predictions_{timestamp}.csv')
        
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 结果保存至: {output_path}")
        
        self._print_stats(result_df)
        return result_df
    
    def predict_from_dataset(self, dataset_dir, output_path=None):
        """从数据集目录预测"""
        csv_path = os.path.join(dataset_dir, 'index.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 index.csv: {csv_path}")
        
        return self.predict_from_csv(csv_path, dataset_dir, output_path)
    
    def predict_all_datasets(self, data_dir=None, output_path=None):
        """预测全部数据集"""
        if data_dir is None:
            data_dir = self.config.DATA_DIR
        
        all_results = []
        
        for folder in self.config.DATASET_FOLDERS:
            dataset_dir = os.path.join(data_dir, folder)
            csv_path = os.path.join(dataset_dir, 'index.csv')
            
            if os.path.exists(csv_path):
                print(f"\n{'='*50}")
                print(f"处理: {folder}")
                print(f"{'='*50}")
                
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
                df['dataset_folder'] = folder
                
                video_list = []
                for _, row in df.iterrows():
                    folder_path = os.path.join(dataset_dir, row['video_id'])
                    video_list.append((folder_path, row.get('title', "")))
                
                results = self.predict_batch(video_list)
                
                df['predict_probability'] = [r.get('probability', None) for r in results]
                df['predict_label'] = [r.get('label', 'error') for r in results]
                df['predict_result'] = [r.get('prediction', None) for r in results]
                df['predict_confidence'] = [r.get('confidence', None) for r in results]
                df['predict_status'] = [r.get('status', 'unknown') for r in results]
                
                all_results.append(df)
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(data_dir, f'all_predictions_{timestamp}.csv')
            
            combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 全部结果保存至: {output_path}")
            
            self._print_stats(combined_df)
            return combined_df
        
        return None
    
    def _print_stats(self, df):
        """打印统计"""
        print(f"\n{'='*50}")
        print("统计信息")
        print(f"{'='*50}")
        
        success_df = df[df['predict_status'] == 'success']
        error_count = len(df) - len(success_df)
        
        print(f"总数: {len(df)}")
        print(f"成功: {len(success_df)}")
        print(f"失败: {error_count}")
        
        if len(success_df) > 0:
            good = (success_df['predict_result'] == 1).sum()
            bad = (success_df['predict_result'] == 0).sum()
            print(f"\n好看: {good} ({good/len(success_df)*100:.1f}%)")
            print(f"不好看: {bad} ({bad/len(success_df)*100:.1f}%)")
            print(f"平均概率: {success_df['predict_probability'].mean():.2%}")
            
            if 'label' in df.columns:
                correct = (success_df['predict_result'] == success_df['label']).sum()
                print(f"\n准确率: {correct/len(success_df)*100:.2f}%")
        
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='视频吸引力预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py --video_folder "./数据集1/video_01" --title "标题"
  python predict.py --dataset_dir "./数据集1"
  python predict.py --csv_file "./data/index.csv" --data_root "./data"
  python predict.py --predict_all
        """
    )
    
    parser.add_argument('--video_folder', type=str, help='视频文件夹路径')
    parser.add_argument('--title', type=str, default="", help='视频标题')
    parser.add_argument('--dataset_dir', type=str, help='数据集目录')
    parser.add_argument('--csv_file', type=str, help='CSV文件路径')
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--predict_all', action='store_true', help='预测全部数据集')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出路径')
    
    args = parser.parse_args()
    
    predictor = Predictor(args.model_path)
    
    if args.predict_all:
        predictor.predict_all_datasets(output_path=args.output)
        
    elif args.dataset_dir:
        predictor.predict_from_dataset(args.dataset_dir, output_path=args.output)
        
    elif args.csv_file:
        predictor.predict_from_csv(args.csv_file, data_root=args.data_root, output_path=args.output)
        
    elif args.video_folder:
        result = predictor.predict(args.video_folder, args.title)
        
        print("\n" + "="*50)
        print("预测结果")
        print("="*50)
        print(f"标题: {args.title}")
        print(f"结果: {result['label']}")
        print(f"好看概率: {result['probability_good']:.2%}")
        print(f"置信度: {result['confidence']:.2%}")
        print("="*50)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
