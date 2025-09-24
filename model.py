#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT模型配置和微调架构
实现基于BERT的文本分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel, 
    BertConfig, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertClassifier(nn.Module):
    """基于BERT的文本分类器"""
    
    def __init__(self, pretrain_dir, num_classes, dropout_rate=0.1, 
                 freeze_bert=False, hidden_size=768):
        """
        初始化BERT分类器
        
        Args:
            pretrain_dir: 预训练BERT模型目录
            num_classes: 分类类别数
            dropout_rate: dropout比率
            freeze_bert: 是否冻结BERT参数
            hidden_size: 隐藏层大小
        """
        super(BertClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 加载预训练BERT模型
        try:
            self.bert = BertModel.from_pretrained(pretrain_dir)
            logger.info(f"成功加载预训练BERT模型: {pretrain_dir}")
        except Exception as e:
            logger.error(f"加载BERT模型失败: {e}")
            raise
        
        # 是否冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT参数已冻结")
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # 初始化分类层权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            labels: 标签 (可选，用于计算损失)
            
        Returns:
            outputs: 包含logits和loss的字典
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的表示
        pooled_output = bert_outputs.pooler_output
        
        # 添加dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        # 如果提供了标签，计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs

class AdvancedBertClassifier(nn.Module):
    """高级BERT分类器，包含更多特性"""
    
    def __init__(self, pretrain_dir, num_classes, dropout_rate=0.1,
                 use_pooler=True, hidden_dims=None, activation='relu'):
        """
        初始化高级BERT分类器
        
        Args:
            pretrain_dir: 预训练BERT模型目录
            num_classes: 分类类别数
            dropout_rate: dropout比率
            use_pooler: 是否使用pooler输出
            hidden_dims: 隐藏层维度列表，如[512, 256]
            activation: 激活函数类型
        """
        super(AdvancedBertClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.use_pooler = use_pooler
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(pretrain_dir)
        
        # 获取BERT隐藏层大小
        bert_hidden_size = self.bert.config.hidden_size
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建分类头
        if hidden_dims is None:
            # 简单分类头
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(bert_hidden_size, num_classes)
            )
        else:
            # 多层分类头
            layers = []
            input_dim = bert_hidden_size
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    self.activation,
                    nn.Dropout(dropout_rate)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类头权重"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """前向传播"""
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if self.use_pooler:
            # 使用pooler输出
            pooled_output = bert_outputs.pooler_output
        else:
            # 使用[CLS]标记的最后一层隐藏状态
            sequence_output = bert_outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]  # [CLS] token
        
        # 分类
        logits = self.classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs

class ModelConfig:
    """模型配置类"""
    
    def __init__(self, config_path=None):
        """
        初始化模型配置
        
        Args:
            config_path: 配置文件路径
        """
        self.default_config = {
            'model_type': 'bert_classifier',  # 或 'advanced_bert_classifier'
            'pretrain_dir': 'pretrain/bert-base-chinese',
            'num_classes': 10,
            'dropout_rate': 0.1,
            'freeze_bert': False,
            'hidden_size': 768,
            'use_pooler': True,
            'hidden_dims': None,  # [512, 256] for multi-layer classifier
            'activation': 'relu',
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'max_grad_norm': 1.0,
            'label_smoothing': 0.0
        }
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self.default_config.copy()
    
    def load_config(self, config_path):
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            self.config = self.default_config.copy()
            self.config.update(loaded_config)
            logger.info(f"成功加载配置文件: {config_path}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = self.default_config.copy()
    
    def save_config(self, config_path):
        """保存配置到文件"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class ModelManager:
    """模型管理器"""
    
    def __init__(self, config):
        """
        初始化模型管理器
        
        Args:
            config: 模型配置对象
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def create_model(self):
        """创建模型"""
        if self.config['model_type'] == 'bert_classifier':
            self.model = BertClassifier(
                pretrain_dir=self.config['pretrain_dir'],
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['dropout_rate'],
                freeze_bert=self.config['freeze_bert'],
                hidden_size=self.config['hidden_size']
            )
        elif self.config['model_type'] == 'advanced_bert_classifier':
            self.model = AdvancedBertClassifier(
                pretrain_dir=self.config['pretrain_dir'],
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['dropout_rate'],
                use_pooler=self.config['use_pooler'],
                hidden_dims=self.config['hidden_dims'],
                activation=self.config['activation']
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.config['model_type']}")
        
        logger.info(f"成功创建 {self.config['model_type']} 模型")
        return self.model
    
    def create_optimizer(self, num_training_steps):
        """创建优化器和学习率调度器"""
        if self.model is None:
            raise ValueError("请先创建模型")
        
        # 分别设置BERT和分类头的学习率
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and "bert" in n],
                "weight_decay": self.config['weight_decay'],
                "lr": self.config['learning_rate']
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and "bert" in n],
                "weight_decay": 0.0,
                "lr": self.config['learning_rate']
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "bert" not in n],
                "weight_decay": self.config['weight_decay'],
                "lr": self.config['learning_rate'] * 10  # 分类头使用更高学习率
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        logger.info("成功创建优化器和学习率调度器")
        return self.optimizer, self.scheduler
    
    def save_model(self, save_dir, epoch=None):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型状态字典
        model_path = os.path.join(save_dir, 'pytorch_model.bin')
        torch.save(self.model.state_dict(), model_path)
        
        # 保存配置
        config_path = os.path.join(save_dir, 'config.json')
        self.config.save_config(config_path)
        
        # 保存优化器状态（如果有）
        if self.optimizer is not None:
            optimizer_path = os.path.join(save_dir, 'optimizer.bin')
            torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # 保存调度器状态（如果有）
        if self.scheduler is not None:
            scheduler_path = os.path.join(save_dir, 'scheduler.bin')
            torch.save(self.scheduler.state_dict(), scheduler_path)
        
        logger.info(f"模型已保存到: {save_dir}")
    
    def load_model(self, model_dir, load_optimizer=False):
        """加载模型"""
        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            self.config.load_config(config_path)
        
        # 创建并加载模型
        self.create_model()
        model_path = os.path.join(model_dir, 'pytorch_model.bin')
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            logger.info(f"成功加载模型: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载优化器（如果需要）
        if load_optimizer:
            optimizer_path = os.path.join(model_dir, 'optimizer.bin')
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path))
                logger.info("成功加载优化器状态")
            
            scheduler_path = os.path.join(model_dir, 'scheduler.bin')
            if os.path.exists(scheduler_path):
                self.scheduler.load_state_dict(torch.load(scheduler_path))
                logger.info("成功加载调度器状态")
        
        return self.model

def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型参数统计:")
    logger.info(f"  总参数数: {total_params:,}")
    logger.info(f"  可训练参数数: {trainable_params:,}")
    logger.info(f"  冻结参数数: {total_params - trainable_params:,}")
    
    return total_params, trainable_params

if __name__ == "__main__":
    # 测试模型创建
    config = ModelConfig()
    config['num_classes'] = 10
    
    model_manager = ModelManager(config)
    model = model_manager.create_model()
    
    # 统计参数
    count_parameters(model)
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 创建虚拟输入
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones((batch_size, seq_length)).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # 前向传播测试
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels)
        print(f"输出logits形状: {outputs['logits'].shape}")
        print(f"损失值: {outputs['loss'].item()}")