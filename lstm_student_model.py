#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双向LSTM学生模型
用于知识蒸馏，从BERT教师模型学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiLSTMStudentModel(nn.Module):
    """双向LSTM学生模型"""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, 
                 num_layers=2, num_classes=10, dropout_rate=0.1,
                 pretrained_embeddings=None, freeze_embeddings=False):
        """
        初始化双向LSTM学生模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数
            dropout_rate: dropout比率
            pretrained_embeddings: 预训练词嵌入矩阵
            freeze_embeddings: 是否冻结词嵌入
        """
        super(BiLSTMStudentModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 词嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, 
                freeze=freeze_embeddings
            )
            logger.info("使用预训练词嵌入")
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            logger.info("使用随机初始化词嵌入")
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM输出维度
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 用于知识蒸馏的特征投影层
        self.feature_projector = nn.Linear(hidden_dim * 2, 768)  # 投影到BERT维度
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_features=False):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 (可选)
            return_features: 是否返回中间特征用于蒸馏
            
        Returns:
            outputs: 包含logits和loss的字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 双向LSTM
        lstm_out, (hidden, cell) = self.lstm(embeddings)  # [batch_size, seq_len, hidden_dim*2]
        
        # 注意力机制
        if attention_mask is not None:
            # 创建注意力掩码 (True表示需要注意的位置)
            attn_mask = attention_mask == 0  # 反转掩码，因为MultiheadAttention使用True表示忽略
        else:
            attn_mask = None
        
        # 自注意力
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out, 
            key_padding_mask=attn_mask
        )  # [batch_size, seq_len, hidden_dim*2]
        
        # 池化：使用attention mask进行加权平均
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(attn_out).float()
            pooled_output = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_output = attn_out.mean(dim=1)  # [batch_size, hidden_dim*2]
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        outputs = {'logits': logits}
        
        # 特征投影（用于知识蒸馏）
        if return_features:
            projected_features = self.feature_projector(pooled_output)
            outputs['features'] = projected_features
            outputs['lstm_features'] = pooled_output
        
        # 计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs

class EnhancedBiLSTMStudentModel(nn.Module):
    """增强版双向LSTM学生模型，包含更多特性"""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512,
                 num_layers=2, num_classes=10, dropout_rate=0.1,
                 use_attention=True, use_residual=True, 
                 pretrained_embeddings=None):
        """
        初始化增强版双向LSTM学生模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数
            dropout_rate: dropout比率
            use_attention: 是否使用注意力机制
            use_residual: 是否使用残差连接
            pretrained_embeddings: 预训练词嵌入矩阵
        """
        super(EnhancedBiLSTMStudentModel, self).__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim
        
        # 词嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多层双向LSTM
        self.lstm_layers = nn.ModuleList()
        input_dim = embedding_dim
        
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim,
                    hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout_rate if i < num_layers - 1 else 0
                )
            )
            input_dim = hidden_dim * 2
        
        # 注意力机制
        if use_attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 特征投影层（用于知识蒸馏）
        self.feature_projector = nn.Linear(hidden_dim * 2, 768)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_features=False):
        """前向传播"""
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 多层双向LSTM
        for i, lstm_layer in enumerate(self.lstm_layers):
            lstm_out, _ = lstm_layer(x)
            
            # 残差连接（维度匹配时）
            if self.use_residual and x.shape[-1] == lstm_out.shape[-1]:
                lstm_out = lstm_out + x
            
            x = lstm_out
        
        # 注意力机制
        if self.use_attention:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask == 0
            
            attn_out, _ = self.attention_layer(x, x, x, key_padding_mask=attn_mask)
            x = attn_out + x  # 残差连接
        
        # 层归一化
        x = self.layer_norm(x)
        
        # 池化
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).float()
            pooled_output = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_output = x.mean(dim=1)
        
        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        # 特征投影
        if return_features:
            projected_features = self.feature_projector(pooled_output)
            outputs['features'] = projected_features
            outputs['lstm_features'] = pooled_output
        
        # 损失计算
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs

class LSTMModelConfig:
    """LSTM学生模型配置类"""
    
    def __init__(self):
        self.config = {
            'model_type': 'bilstm_student',  # 或 'enhanced_bilstm_student'
            'vocab_size': 21128,  # BERT词汇表大小
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'num_classes': 10,
            'dropout_rate': 0.1,
            'use_attention': True,
            'use_residual': True,
            'freeze_embeddings': False,
            'use_pretrained_embeddings': True,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0
        }
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def get(self, key, default=None):
        return self.config.get(key, default)

def create_lstm_student_model(config, tokenizer=None):
    """
    创建LSTM学生模型
    
    Args:
        config: 模型配置
        tokenizer: BERT tokenizer（用于获取词汇表大小和预训练嵌入）
        
    Returns:
        model: LSTM学生模型
    """
    # 获取词汇表大小
    if tokenizer is not None:
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = config['vocab_size']
    
    # 预训练词嵌入（可选）
    pretrained_embeddings = None
    if config.get('use_pretrained_embeddings', False) and tokenizer is not None:
        # 这里可以加载BERT的词嵌入作为初始化
        # 为了简化，我们使用随机初始化
        logger.info("注意：当前使用随机初始化，可以考虑使用BERT词嵌入初始化")
    
    # 创建模型
    if config['model_type'] == 'bilstm_student':
        model = BiLSTMStudentModel(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=config['freeze_embeddings']
        )
    elif config['model_type'] == 'enhanced_bilstm_student':
        model = EnhancedBiLSTMStudentModel(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            use_attention=config['use_attention'],
            use_residual=config['use_residual'],
            pretrained_embeddings=pretrained_embeddings
        )
    else:
        raise ValueError(f"不支持的模型类型: {config['model_type']}")
    
    logger.info(f"成功创建 {config['model_type']} 模型")
    return model

def count_lstm_parameters(model):
    """统计LSTM模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"LSTM学生模型参数统计:")
    logger.info(f"  总参数数: {total_params:,}")
    logger.info(f"  可训练参数数: {trainable_params:,}")
    
    return total_params, trainable_params

if __name__ == "__main__":
    # 测试模型创建
    from transformers import BertTokenizer
    
    # 创建配置
    config = LSTMModelConfig()
    config['num_classes'] = 10
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('pretrain/bert-base-chinese')
    
    # 创建模型
    model = create_lstm_student_model(config, tokenizer)
    
    # 统计参数
    count_lstm_parameters(model)
    
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
        outputs = model(input_ids, attention_mask, labels, return_features=True)
        print(f"输出logits形状: {outputs['logits'].shape}")
        print(f"特征形状: {outputs['features'].shape}")
        print(f"损失值: {outputs['loss'].item()}")