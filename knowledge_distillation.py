#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识蒸馏训练器
实现BERT教师模型向LSTM学生模型的知识蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        初始化知识蒸馏损失
        
        Args:
            temperature: 蒸馏温度，控制软目标的平滑程度
            alpha: 蒸馏损失权重，(1-alpha)为硬目标损失权重
        """
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型输出 [batch_size, num_classes]
            teacher_logits: 教师模型输出 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            
        Returns:
            total_loss: 总损失
            distill_loss: 蒸馏损失
            hard_loss: 硬目标损失
        """
        # 软目标蒸馏损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 硬目标损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, distill_loss, hard_loss

class FeatureDistillationLoss(nn.Module):
    """特征层知识蒸馏损失"""
    
    def __init__(self, student_dim=512, teacher_dim=768, loss_type='mse'):
        """
        初始化特征蒸馏损失
        
        Args:
            student_dim: 学生模型特征维度
            teacher_dim: 教师模型特征维度
            loss_type: 损失类型 ('mse', 'cosine')
        """
        super(FeatureDistillationLoss, self).__init__()
        self.loss_type = loss_type
        
        # 特征对齐层
        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, student_features, teacher_features):
        """
        计算特征蒸馏损失
        
        Args:
            student_features: 学生模型特征 [batch_size, student_dim]
            teacher_features: 教师模型特征 [batch_size, teacher_dim]
            
        Returns:
            feature_loss: 特征蒸馏损失
        """
        # 特征对齐
        projected_student = self.projector(student_features)
        
        if self.loss_type == 'mse':
            feature_loss = self.loss_fn(projected_student, teacher_features.detach())
        elif self.loss_type == 'cosine':
            target = torch.ones(projected_student.size(0)).to(projected_student.device)
            feature_loss = self.loss_fn(
                projected_student, 
                teacher_features.detach(), 
                target
            )
        else:
            feature_loss = torch.tensor(0.0).to(projected_student.device)
        
        return feature_loss

class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, teacher_model, student_model, config):
        """
        初始化蒸馏训练器
        
        Args:
            teacher_model: 教师模型 (BERT)
            student_model: 学生模型 (LSTM)
            config: 训练配置
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # 获取设备
        self.device = next(student_model.parameters()).device
        
        # 损失函数
        self.distill_loss = KnowledgeDistillationLoss(
            temperature=config.get('temperature', 4.0),
            alpha=config.get('alpha', 0.7)
        )
        
        self.feature_loss = FeatureDistillationLoss(
            student_dim=config.get('student_feature_dim', 1024),
            teacher_dim=config.get('teacher_feature_dim', 768),
            loss_type=config.get('feature_loss_type', 'mse')
        )
        
        # 将损失函数移动到正确的设备
        self.distill_loss.to(self.device)
        self.feature_loss.to(self.device)
        
        # 优化器
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.best_val_acc = 0.0
        self.train_history = []
    
    def setup_optimizer(self, num_training_steps):
        """设置优化器和学习率调度器"""
        # 分层学习率
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student_model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get('weight_decay', 1e-4),
            },
            {
                "params": [p for n, p in self.student_model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        
        # 特征对齐层的参数
        if hasattr(self.feature_loss, 'projector') and not isinstance(self.feature_loss.projector, nn.Identity):
            optimizer_grouped_parameters.extend([
                {
                    "params": self.feature_loss.projector.parameters(),
                    "weight_decay": self.config.get('weight_decay', 1e-4),
                    "lr": self.config.get('learning_rate', 1e-3) * 0.1  # 较小的学习率
                }
            ])
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get('learning_rate', 1e-3),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.get('warmup_steps', 1000),
            num_training_steps=num_training_steps
        )
        
        logger.info("优化器和调度器设置完成")
    
    def train_epoch(self, train_loader, device):
        """训练一个epoch"""
        self.teacher_model.eval()  # 教师模型始终在评估模式
        self.student_model.train()
        
        total_loss = 0
        total_distill_loss = 0
        total_hard_loss = 0
        total_feature_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            # 数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 教师模型推理（不计算梯度）
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs['logits']
                
                # 获取教师模型特征
                # 对于BERT分类器，我们需要获取BERT的输出特征
                if hasattr(self.teacher_model, 'bert'):
                    # 获取BERT的pooler输出
                    bert_outputs = self.teacher_model.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_features = bert_outputs.pooler_output
                else:
                    # 备用方案：如果输出中有特征信息
                    if 'pooler_output' in teacher_outputs:
                        teacher_features = teacher_outputs['pooler_output']
                    elif hasattr(teacher_outputs, 'pooler_output'):
                        teacher_features = teacher_outputs.pooler_output
                    else:
                        # 最后的备用方案：使用logits的特征维度进行随机填充（仅用于测试）
                        teacher_features = torch.randn(teacher_logits.size(0), 768).to(teacher_logits.device)
            
            # 学生模型推理
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=True
            )
            student_logits = student_outputs['logits']
            # 使用投影后的特征进行蒸饦对齐
            student_features = student_outputs['features']
            
            # 计算损失
            # 1. 知识蒸馏损失
            distill_total_loss, distill_loss, hard_loss = self.distill_loss(
                student_logits, teacher_logits, labels
            )
            
            # 2. 特征蒸馏损失
            feature_loss = self.feature_loss(student_features, teacher_features)
            
            # 3. 总损失
            total_batch_loss = (
                distill_total_loss + 
                self.config.get('feature_loss_weight', 0.1) * feature_loss
            )
            
            # 反向传播
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), 
                    self.config.get('max_grad_norm', 1.0)
                )
                
                self.optimizer.step()
                
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            total_distill_loss += distill_loss.item()
            total_hard_loss += hard_loss.item()
            total_feature_loss += feature_loss.item()
            
            predictions = torch.argmax(student_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 更新进度条
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.config.get('learning_rate', 1e-3)
            progress_bar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'distill': f'{distill_loss.item():.4f}',
                'feature': f'{feature_loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        avg_distill_loss = total_distill_loss / len(train_loader)
        avg_hard_loss = total_hard_loss / len(train_loader)
        avg_feature_loss = total_feature_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return {
            'loss': avg_loss,
            'distill_loss': avg_distill_loss,
            'hard_loss': avg_hard_loss,
            'feature_loss': avg_feature_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self, val_loader, device):
        """评估模型"""
        self.student_model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_loader, val_loader, device, num_epochs=10, save_dir='distilled_models'):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置优化器
        num_training_steps = len(train_loader) * num_epochs
        self.setup_optimizer(num_training_steps)
        
        logger.info(f"开始知识蒸馏训练，共 {num_epochs} 个epoch")
        logger.info(f"学生模型参数数量: {sum(p.numel() for p in self.student_model.parameters()):,}")
        
        for epoch in range(num_epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, device)
            
            # 验证
            val_metrics = self.evaluate(val_loader, device) if val_loader else {}
            
            # 记录历史
            epoch_history = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            }
            self.train_history.append(epoch_history)
            
            # 打印结果
            logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                       f"Distill: {train_metrics['distill_loss']:.4f}, "
                       f"Feature: {train_metrics['feature_loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}")
            
            if val_metrics:
                logger.info(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                           f"Acc: {val_metrics['accuracy']:.4f}, "
                           f"F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics and val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                best_model_path = os.path.join(save_dir, 'best_student_model')
                self.save_student_model(best_model_path)
                logger.info(f"保存最佳学生模型 (Val Acc: {val_metrics['accuracy']:.4f})")
            
            # 定期保存
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                checkpoint_path = os.path.join(save_dir, f'student_model_epoch_{epoch+1}')
                self.save_student_model(checkpoint_path)
        
        # 保存最终模型和训练历史
        final_model_path = os.path.join(save_dir, 'final_student_model')
        self.save_student_model(final_model_path)
        
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        
        logger.info("知识蒸馏训练完成!")
        return self.train_history
    
    def save_student_model(self, save_path):
        """保存学生模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型状态字典
        model_path = os.path.join(save_path, 'pytorch_model.bin')
        torch.save(self.student_model.state_dict(), model_path)
        
        # 保存配置
        config_path = os.path.join(save_path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            # 处理不可序列化的配置项
            serializable_config = {}
            for k, v in self.config.items():
                try:
                    json.dumps(v)
                    serializable_config[k] = v
                except:
                    serializable_config[k] = str(v)
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        # 保存特征对齐层（如果存在）
        if hasattr(self.feature_loss, 'projector') and not isinstance(self.feature_loss.projector, nn.Identity):
            projector_path = os.path.join(save_path, 'feature_projector.bin')
            torch.save(self.feature_loss.projector.state_dict(), projector_path)

def create_distillation_config():
    """创建默认蒸馏配置"""
    return {
        # 蒸馏超参数
        'temperature': 4.0,
        'alpha': 0.7,  # 蒸馏损失权重
        'feature_loss_weight': 0.1,  # 特征损失权重
        'feature_loss_type': 'mse',  # 'mse' or 'cosine'
        
        # 模型维度
        'student_feature_dim': 1024,  # 双向LSTM: hidden_dim * 2
        'teacher_feature_dim': 768,   # BERT隐藏维度
        
        # 训练超参数
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
        'save_every': 5,
        
        # 其他配置
        'use_amp': False,  # 混合精度训练
        'log_interval': 100
    }

if __name__ == "__main__":
    # 测试知识蒸馏组件
    print("知识蒸馏训练器创建成功!")
    
    # 测试损失函数
    batch_size, num_classes = 4, 10
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 知识蒸馏损失
    kd_loss = KnowledgeDistillationLoss()
    total_loss, distill_loss, hard_loss = kd_loss(student_logits, teacher_logits, labels)
    print(f"蒸馏损失测试 - 总损失: {total_loss:.4f}, 蒸馏: {distill_loss:.4f}, 硬目标: {hard_loss:.4f}")
    
    # 特征蒸馏损失
    student_features = torch.randn(batch_size, 512)
    teacher_features = torch.randn(batch_size, 768)
    
    feature_loss_fn = FeatureDistillationLoss(512, 768)
    feature_loss = feature_loss_fn(student_features, teacher_features)
    print(f"特征损失测试: {feature_loss:.4f}")