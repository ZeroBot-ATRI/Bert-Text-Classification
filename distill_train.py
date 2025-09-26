#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识蒸馏训练脚本
使用BERT教师模型训练双向LSTM学生模型
"""

import os
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
import json

from data_loader import DataLoader_Manager
from model import BertClassifier, ModelConfig, ModelManager
from lstm_student_model import create_lstm_student_model, LSTMModelConfig, count_lstm_parameters
from knowledge_distillation import DistillationTrainer, create_distillation_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_teacher_model(teacher_model_path, config, device):
    """加载训练好的BERT教师模型"""
    logger.info(f"加载教师模型: {teacher_model_path}")
    
    # 创建模型管理器
    model_manager = ModelManager(config)
    teacher_model = model_manager.create_model()
    
    # 加载模型权重
    model_file = os.path.join(teacher_model_path, 'pytorch_model.bin')
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location=device)
        teacher_model.load_state_dict(state_dict)
        logger.info("成功加载教师模型权重")
    else:
        raise FileNotFoundError(f"教师模型文件不存在: {model_file}")
    
    teacher_model.to(device)
    teacher_model.eval()  # 设为评估模式
    
    return teacher_model

def create_student_model(config, tokenizer, device):
    """创建学生模型"""
    logger.info("创建LSTM学生模型")
    
    student_model = create_lstm_student_model(config, tokenizer)
    student_model.to(device)
    
    # 统计参数数量
    count_lstm_parameters(student_model)
    
    return student_model

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='BERT到LSTM的知识蒸馏训练')
    
    # 数据和模型路径
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='数据目录')
    parser.add_argument('--teacher_model_path', type=str, default='checkpoints/best_model',
                       help='教师模型路径')
    parser.add_argument('--pretrain_dir', type=str, default='pretrain/bert-base-chinese',
                       help='预训练BERT模型目录')
    parser.add_argument('--save_dir', type=str, default='distilled_models',
                       help='学生模型保存目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    
    # 学生模型参数
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='词嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--student_model_type', type=str, default='enhanced_bilstm_student',
                       choices=['bilstm_student', 'enhanced_bilstm_student'],
                       help='学生模型类型')
    
    # 蒸馏参数
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='蒸馏损失权重')
    parser.add_argument('--feature_loss_weight', type=float, default=0.1,
                       help='特征损失权重')
    parser.add_argument('--feature_loss_type', type=str, default='mse',
                       choices=['mse', 'cosine'],
                       help='特征损失类型')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--no_cuda', action='store_true',
                       help='不使用GPU')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备配置
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存训练参数
    with open(os.path.join(args.save_dir, 'training_args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # 创建数据管理器
    logger.info("准备数据...")
    data_manager = DataLoader_Manager(args.data_dir, args.pretrain_dir, args.max_length)
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset, test_dataset = data_manager.create_datasets()
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=args.batch_size
    )
    
    # 获取类别数量
    num_classes = data_manager.get_num_classes()
    logger.info(f"类别数量: {num_classes}")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_dir)
    
    # 创建教师模型配置并加载模型
    teacher_config = ModelConfig()
    teacher_config['pretrain_dir'] = args.pretrain_dir
    teacher_config['num_classes'] = num_classes
    
    teacher_model = load_teacher_model(args.teacher_model_path, teacher_config, device)
    
    # 创建学生模型配置
    student_config = LSTMModelConfig()
    student_config['model_type'] = args.student_model_type
    student_config['vocab_size'] = tokenizer.vocab_size
    student_config['embedding_dim'] = args.embedding_dim
    student_config['hidden_dim'] = args.hidden_dim
    student_config['num_layers'] = args.num_layers
    student_config['num_classes'] = num_classes
    student_config['learning_rate'] = args.learning_rate
    
    # 创建学生模型
    student_model = create_student_model(student_config, tokenizer, device)
    
    # 创建蒸馏配置
    distill_config = create_distillation_config()
    
    # 获取学生模型的实际特征维度
    # 创建一个临时的前向传播来确定特征维度
    with torch.no_grad():
        dummy_input = torch.randint(0, 1000, (2, 10)).to(device)
        dummy_mask = torch.ones((2, 10)).to(device)
        dummy_outputs = student_model(dummy_input, dummy_mask, return_features=True)
        actual_student_feature_dim = dummy_outputs['features'].shape[-1]  # 投影后的维度
        lstm_feature_dim = dummy_outputs['lstm_features'].shape[-1]  # 原始LSTM维度
    
    logger.info(f"学生模型LSTM特征维度: {lstm_feature_dim}")
    logger.info(f"学生模型投影后特征维度: {actual_student_feature_dim}")
    
    distill_config.update({
        'temperature': args.temperature,
        'alpha': args.alpha,
        'feature_loss_weight': args.feature_loss_weight,
        'feature_loss_type': args.feature_loss_type,
        'student_feature_dim': actual_student_feature_dim,  # 使用投影后的维度
        'teacher_feature_dim': 768,  # BERT
        'learning_rate': args.learning_rate,
        'use_amp': args.use_amp
    })
    
    # 创建蒸馏训练器
    trainer = DistillationTrainer(teacher_model, student_model, distill_config)
    
    # 打印训练信息
    logger.info("=" * 60)
    logger.info("知识蒸馏训练配置:")
    logger.info(f"  教师模型: BERT ({sum(p.numel() for p in teacher_model.parameters()):,} 参数)")
    logger.info(f"  学生模型: {args.student_model_type} ({sum(p.numel() for p in student_model.parameters()):,} 参数)")
    logger.info(f"  参数压缩比: {sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.1f}x")
    logger.info(f"  训练样本数: {len(train_dataset)}")
    logger.info(f"  验证样本数: {len(val_dataset) if val_dataset else 0}")
    logger.info(f"  测试样本数: {len(test_dataset)}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  训练轮数: {args.num_epochs}")
    logger.info(f"  学习率: {args.learning_rate}")
    logger.info(f"  蒸馏温度: {args.temperature}")
    logger.info(f"  蒸馏权重: {args.alpha}")
    logger.info("=" * 60)
    
    # 开始训练
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    training_time = time.time() - start_time
    logger.info(f"训练完成! 总用时: {training_time/3600:.2f} 小时")
    
    # 在测试集上评估最佳模型
    logger.info("在测试集上评估最佳学生模型...")
    best_model_path = os.path.join(args.save_dir, 'best_student_model', 'pytorch_model.bin')
    
    test_metrics = None  # 初始化变量
    if os.path.exists(best_model_path):
        # 加载最佳模型
        student_model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # 评估
        test_metrics = trainer.evaluate(test_loader, device)
        
        logger.info("测试集评估结果:")
        logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {test_metrics['precision']:.4f}")
        logger.info(f"  召回率: {test_metrics['recall']:.4f}")
        logger.info(f"  F1分数: {test_metrics['f1']:.4f}")
        
        # 保存测试结果
        test_results = {
            'test_metrics': test_metrics,
            'training_time_hours': training_time / 3600,
            'model_compression_ratio': sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()),
            'args': vars(args)
        }
        
        with open(os.path.join(args.save_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"测试结果已保存到: {os.path.join(args.save_dir, 'test_results.json')}")
    
    # 与教师模型对比
    logger.info("\n" + "=" * 60)
    logger.info("教师模型vs学生模型对比:")
    
    # 评估教师模型在测试集上的性能
    teacher_model.eval()
    teacher_correct = 0
    teacher_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = teacher_model(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=1)
            teacher_correct += (predictions == labels).sum().item()
            teacher_total += labels.size(0)
    
    teacher_accuracy = teacher_correct / teacher_total
    
    logger.info(f"教师模型(BERT)测试准确率: {teacher_accuracy:.4f}")
    if test_metrics is not None:
        logger.info(f"学生模型(LSTM)测试准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"性能保持率: {test_metrics['accuracy']/teacher_accuracy*100:.1f}%")
    else:
        logger.info("学生模型测试结果不可用（未找到最佳模型文件）")
    logger.info(f"模型大小压缩: {sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.1f}x")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()