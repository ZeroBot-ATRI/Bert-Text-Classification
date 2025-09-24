#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT文本分类训练脚本
支持使用train.txt作为训练数据
"""

import os
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import DataLoader_Manager
from model import ModelManager, ModelConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, optimizer, scheduler, 
                device, num_epochs=10, save_dir='checkpoints'):
    """训练模型"""
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            # 数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 更新进度条
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions
        
        # 验证阶段
        val_acc = 0.0
        val_loss = 0.0
        if val_loader:
            model.eval()
            val_total_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                    logits = outputs['logits']
                    
                    val_total_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss = val_total_loss / len(val_loader)
            val_acc = val_correct / val_total
        
        # 打印结果
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if val_loader:
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_loader and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_model')
            os.makedirs(best_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_model_path, 'pytorch_model.bin'))
            logger.info(f"保存最佳模型 (Val Acc: {val_acc:.4f})")
        
        # 定期保存
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}')
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'pytorch_model.bin'))
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model')
    os.makedirs(final_model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_model_path, 'pytorch_model.bin'))
    
    logger.info("训练完成!")

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='BERT文本分类训练')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--pretrain_dir', type=str, default='pretrain/bert-base-chinese', help='预训练模型目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备配置
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据管理器
    data_manager = DataLoader_Manager(args.data_dir, args.pretrain_dir, args.max_length)
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset, test_dataset = data_manager.create_datasets()
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=args.batch_size
    )
    
    # 创建配置
    config = ModelConfig()
    config['pretrain_dir'] = args.pretrain_dir
    config['num_classes'] = data_manager.get_num_classes()
    config['learning_rate'] = args.learning_rate
    
    # 创建模型
    model_manager = ModelManager(config)
    model = model_manager.create_model()
    model.to(device)
    
    # 计算训练步数并创建优化器
    num_training_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = model_manager.create_optimizer(num_training_steps)
    
    logger.info(f"训练参数:")
    logger.info(f"  训练样本数: {len(train_dataset)}")
    logger.info(f"  验证样本数: {len(val_dataset) if val_dataset else 0}")
    logger.info(f"  测试样本数: {len(test_dataset)}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  训练轮数: {args.num_epochs}")
    logger.info(f"  学习率: {args.learning_rate}")
    logger.info(f"  类别数: {config['num_classes']}")
    
    # 开始训练
    train_model(model, train_loader, val_loader, optimizer, scheduler,
                device, args.num_epochs, args.save_dir)

if __name__ == "__main__":
    main()