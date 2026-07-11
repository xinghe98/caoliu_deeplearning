"""Train a calibrated multimodal attractiveness classifier.

The workflow deliberately keeps configured external folders out of model selection.
"""

import argparse
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config import Config, get_device
from dataset import VideoDataset, load_dataset, split_train_validation
from models import MultiModalClassifier
from utils import (
    apply_temperature, binary_metrics, fit_temperature, plot_training_history,
    select_threshold, set_seed, train_one_epoch, validate,
)

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='视频吸引力预测模型训练')
    parser.add_argument('--resume', action='store_true', help='从新版 checkpoint 继续训练')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖总训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖批次大小')
    return parser.parse_args()


def config_snapshot(config):
    return {name: getattr(config, name) for name in dir(config) if name.isupper()}


def data_manifest_hash(dataframe):
    columns = [column for column in ('dataset_folder', 'video_id', 'title', 'download_link', 'label', '_group_id') if column in dataframe]
    payload = dataframe[columns].fillna('').astype(str).sort_values(columns).to_csv(index=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def make_optimizer(model, config):
    head_params, backbone_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('image_encoder.backbone') or name.startswith('text_encoder.bert'):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    groups = [{'params': head_params, 'lr': config.LEARNING_RATE}]
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': config.BACKBONE_LEARNING_RATE})
    return optim.AdamW(groups, weight_decay=config.WEIGHT_DECAY)


def set_finetuning_stage(model, enabled):
    model.image_encoder.set_layer4_trainable(enabled)
    model.text_encoder.set_last_layers_trainable(enabled)


def evaluate(model, loader, criterion, device, target_precision):
    val_loss, _, _, labels, _, video_ids, titles, folders, logits = validate(model, loader, criterion, device)
    temperature = fit_temperature(logits, labels)
    probabilities = apply_temperature(logits, temperature)
    threshold, threshold_info = select_threshold(labels, probabilities, target_precision)
    metrics = binary_metrics(labels, probabilities, threshold)
    metrics['loss'] = float(val_loss)
    metrics['temperature'] = temperature
    metrics['threshold_selection'] = threshold_info
    samples = pd.DataFrame({
        'dataset_folder': folders,
        'video_id': video_ids,
        'title': titles,
        'true_label': labels.astype(int),
        'logit': logits,
        'probability': probabilities,
        'prediction': (probabilities >= threshold).astype(int),
    })
    return metrics, samples


def write_prediction_artifacts(samples, output_dir, prefix, threshold):
    """Write all predictions plus a focused false-positive/false-negative set."""
    samples = samples.copy()
    samples['decision_threshold'] = float(threshold)
    samples.to_csv(
        os.path.join(output_dir, f'{prefix}_predictions.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    errors = samples[samples['prediction'] != samples['true_label']].copy()
    errors['error_type'] = np.where(
        errors['prediction'].eq(1), 'false_positive', 'false_negative'
    )
    errors = errors.sort_values('probability', ascending=False)
    errors.to_csv(
        os.path.join(output_dir, f'{prefix}_error_cases.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    return int(len(errors))


def main():
    args = parse_args()
    config = Config()
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size

    set_seed(config.RANDOM_SEED)
    device = get_device()
    df = load_dataset(config)
    external_folders = set(config.EXTERNAL_TEST_FOLDERS)
    external_df = df[df['dataset_folder'].isin(external_folders)].copy().reset_index(drop=True)
    development_df = df[~df['dataset_folder'].isin(external_folders)].copy().reset_index(drop=True)
    if len(external_df) and external_df['label'].nunique() < 2:
        raise ValueError('外部测试集必须同时包含正、负样本')
    train_df, val_df = split_train_validation(
        development_df, config.VALIDATION_FRACTION, config.RANDOM_SEED
    )
    print(f'训练集: {len(train_df)} | 验证集: {len(val_df)} | 锁定外部测试集: {len(external_df)}')

    manifest = pd.concat([
        train_df.assign(split='train'), val_df.assign(split='validation'),
        external_df.assign(split='external_test'),
    ], ignore_index=True)
    manifest.to_csv(os.path.join(config.DATA_DIR, 'split_manifest.csv'), index=False, encoding='utf-8-sig')
    manifest_hash = data_manifest_hash(manifest)

    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    train_loader = DataLoader(VideoDataset(train_df, config, tokenizer, is_training=True), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(VideoDataset(val_df, config, tokenizer, is_training=False), batch_size=config.BATCH_SIZE, shuffle=False)
    external_loader = None
    if len(external_df):
        external_loader = DataLoader(VideoDataset(external_df, config, tokenizer, is_training=False), batch_size=config.BATCH_SIZE, shuffle=False)

    model = MultiModalClassifier(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_path = os.path.join(config.DATA_DIR, config.MODEL_SAVE_PATH)
    start_epoch = 0
    best_pr_auc = -float('inf')
    patience = 0
    history = {'train_loss': [], 'train_accuracy': [], 'validation_loss': [], 'validation_pr_auc': []}

    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_config' not in checkpoint:
            raise ValueError('旧 checkpoint 缺少训练配置，无法安全续训；请从头训练新版模型。')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_pr_auc = checkpoint.get('metrics', {}).get('pr_auc', -float('inf'))
        history = checkpoint.get('history', history)
        print(f'从 epoch {start_epoch + 1} 继续训练，最佳 PR-AUC: {best_pr_auc:.4f}')

    optimizer = None
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        finetune = epoch >= config.WARMUP_EPOCHS
        set_finetuning_stage(model, finetune)
        if optimizer is None or (epoch == config.WARMUP_EPOCHS and start_epoch <= config.WARMUP_EPOCHS):
            optimizer = make_optimizer(model, config)
            stage = '主干微调' if finetune else '分类头预热'
            print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS} — {stage}')

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics, samples = evaluate(model, val_loader, criterion, device, config.TARGET_PRECISION)
        history['train_loss'].append(float(train_loss))
        history['train_accuracy'].append(float(train_accuracy))
        history['validation_loss'].append(metrics['loss'])
        history['validation_pr_auc'].append(metrics['pr_auc'])
        print(
            f"训练 loss={train_loss:.4f}, acc={train_accuracy:.2%} | "
            f"验证 PR-AUC={metrics['pr_auc']:.4f}, precision={metrics['precision']:.2%}, "
            f"recall={metrics['recall']:.2%}, threshold={metrics['threshold']:.3f}"
        )

        if metrics['pr_auc'] > best_pr_auc:
            best_pr_auc, patience = metrics['pr_auc'], 0
            checkpoint = {
                'format_version': 2,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': config_snapshot(config),
                'data_manifest_hash': manifest_hash,
                'metrics': metrics,
                'decision_threshold': metrics['threshold'],
                'temperature': metrics['temperature'],
                'history': history,
            }
            torch.save(checkpoint, checkpoint_path)
            write_prediction_artifacts(
                samples, config.DATA_DIR, 'validation', metrics['threshold']
            )
            print('✓ 已按验证 PR-AUC 保存最佳模型')
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print('早停：验证 PR-AUC 连续未提升')
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    report = {'validation': checkpoint['metrics'], 'data_manifest_hash': manifest_hash}
    if external_loader is not None:
        _, _, _, labels, _, video_ids, titles, folders, logits = validate(model, external_loader, criterion, device)
        probabilities = apply_temperature(logits, checkpoint['temperature'])
        report['external_test'] = binary_metrics(labels, probabilities, checkpoint['decision_threshold'])
        external_samples = pd.DataFrame({
            'dataset_folder': folders, 'video_id': video_ids, 'title': titles,
            'true_label': labels.astype(int), 'logit': logits, 'probability': probabilities,
            'prediction': (probabilities >= checkpoint['decision_threshold']).astype(int),
        })
        report['external_test_error_count'] = write_prediction_artifacts(
            external_samples, config.DATA_DIR, 'external_test', checkpoint['decision_threshold']
        )
        print(f"外部测试 PR-AUC={report['external_test']['pr_auc']:.4f}, precision={report['external_test']['precision']:.2%}")
    validation_samples = pd.read_csv(os.path.join(config.DATA_DIR, 'validation_predictions.csv'), encoding='utf-8-sig')
    report['validation_error_count'] = int(
        (validation_samples['prediction'] != validation_samples['true_label']).sum()
    )
    with open(os.path.join(config.DATA_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with open(os.path.join(config.DATA_DIR, 'training_history.json'), 'w', encoding='utf-8') as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    plot_training_history(history['train_loss'], history['validation_loss'], history['train_accuracy'], history['validation_pr_auc'], os.path.join(config.DATA_DIR, 'training_history.png'))


if __name__ == '__main__':
    main()
