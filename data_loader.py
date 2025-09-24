#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
用于处理文本分类数据，包括数据读取、清洗、tokenization等
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import re
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """文本分类数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 文本预处理
        text = self.clean_text(text)
        
        # tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def clean_text(self, text):
        """清洗文本数据"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空格
        text = text.strip()
        return text

class DataLoader_Manager:
    """数据加载管理器"""
    
    def __init__(self, data_dir, pretrain_dir, max_length=512):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
            pretrain_dir: 预训练模型目录路径
            max_length: 最大序列长度
        """
        self.data_dir = data_dir
        self.pretrain_dir = pretrain_dir
        self.max_length = max_length
        self.tokenizer = None
        self.label_to_id = {}
        self.id_to_label = {}
        
        # 初始化tokenizer
        self._load_tokenizer()
        # 加载类别标签
        self._load_labels()
    
    def _load_tokenizer(self):
        """加载BERT tokenizer"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_dir)
            logger.info(f"成功加载tokenizer，词汇表大小: {self.tokenizer.vocab_size}")
        except Exception as e:
            logger.error(f"加载tokenizer失败: {e}")
            raise
    
    def _load_labels(self):
        """加载类别标签"""
        class_file = os.path.join(self.data_dir, 'class.txt')
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            
            # 创建标签映射
            self.label_to_id = {label: idx for idx, label in enumerate(labels)}
            self.id_to_label = {idx: label for idx, label in enumerate(labels)}
            
            logger.info(f"成功加载 {len(labels)} 个类别标签")
            logger.info(f"类别标签: {labels}")
            
        except Exception as e:
            logger.error(f"加载类别标签失败: {e}")
            raise
    
    def load_data_from_file(self, file_path):
        """
        从文件加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            texts: 文本列表
            labels: 标签列表
        """
        texts = []
        labels = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 假设格式为: text\tlabel
                        parts = line.rsplit('\t', 1)
                        if len(parts) == 2:
                            text, label = parts
                            texts.append(text)
                            labels.append(int(label))
                        else:
                            logger.warning(f"跳过格式错误的行: {line}")
            
            logger.info(f"从 {file_path} 加载了 {len(texts)} 条数据")
            return texts, labels
            
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            raise
    
    def create_datasets(self, train_ratio=0.8, val_ratio=0.1, random_state=42):
        """
        创建训练、验证和测试数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_state: 随机种子
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # 加载训练数据 (使用train.txt)
        train_texts, train_labels = self.load_data_from_file(
            os.path.join(self.data_dir, 'train.txt')
        )
        
        # 加载测试数据
        test_texts, test_labels = self.load_data_from_file(
            os.path.join(self.data_dir, 'test.txt')
        )
        
        # 将训练数据分割为训练集和验证集
        if val_ratio > 0:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels,
                test_size=val_ratio,
                random_state=random_state,
                stratify=train_labels
            )
        else:
            val_texts, val_labels = [], []
        
        # 创建数据集
        train_dataset = TextDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        
        val_dataset = TextDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        ) if val_texts else None
        
        test_dataset = TextDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )
        
        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset) if val_dataset else 0}")
        logger.info(f"测试集样本数: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset, 
                          batch_size=16, num_workers=0):
        """
        创建数据加载器
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            test_dataset: 测试数据集
            batch_size: 批次大小
            num_workers: 工作进程数
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ) if val_dataset else None
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_num_classes(self):
        """获取类别数量"""
        return len(self.label_to_id)
    
    def get_label_names(self):
        """获取类别名称列表"""
        return list(self.label_to_id.keys())

def analyze_data_distribution(data_loader_manager):
    """分析数据分布"""
    logger.info("=== 数据分布分析 ===")
    
    # 分析训练数据分布 (使用train.txt)
    train_texts, train_labels = data_loader_manager.load_data_from_file(
        os.path.join(data_loader_manager.data_dir, 'train.txt')
    )
    
    # 统计各类别数量
    label_counts = {}
    for label in train_labels:
        label_name = data_loader_manager.id_to_label[label]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    logger.info("训练数据各类别分布:")
    for label_name, count in sorted(label_counts.items()):
        logger.info(f"  {label_name}: {count} 样本")
    
    # 分析文本长度分布
    text_lengths = [len(text) for text in train_texts]
    logger.info(f"文本长度统计:")
    logger.info(f"  平均长度: {np.mean(text_lengths):.2f}")
    logger.info(f"  最小长度: {min(text_lengths)}")
    logger.info(f"  最大长度: {max(text_lengths)}")
    logger.info(f"  中位数长度: {np.median(text_lengths):.2f}")

if __name__ == "__main__":
    # 测试数据加载功能
    data_dir = "data"
    pretrain_dir = "pretrain/bert-base-chinese"
    
    # 创建数据加载管理器
    data_manager = DataLoader_Manager(data_dir, pretrain_dir)
    
    # 分析数据分布
    analyze_data_distribution(data_manager)
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = data_manager.create_datasets()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=16
    )
    
    # 测试数据加载
    for batch in train_loader:
        print("样本批次形状:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        break