#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识蒸馏测试脚本
验证完整的知识蒸馏流程
"""

import torch
import os
from transformers import BertTokenizer

from data_loader import DataLoader_Manager
from model import BertClassifier, ModelConfig, ModelManager
from lstm_student_model import create_lstm_student_model, LSTMModelConfig, count_lstm_parameters
from knowledge_distillation import DistillationTrainer, create_distillation_config

def test_distillation_pipeline():
    """测试知识蒸馏完整流程"""
    print("=" * 60)
    print("开始测试知识蒸馏流程")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据准备
    print("\n1. 准备数据...")
    data_manager = DataLoader_Manager('data', 'pretrain/bert-base-chinese', max_length=128)
    train_dataset, val_dataset, test_dataset = data_manager.create_datasets()
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=4
    )
    
    num_classes = data_manager.get_num_classes()
    tokenizer = BertTokenizer.from_pretrained('pretrain/bert-base-chinese')
    
    print(f"数据加载完成 - 类别数: {num_classes}")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset) if val_dataset else 0}")
    
    # 加载教师模型
    print("\n2. 加载教师模型...")
    teacher_config = ModelConfig()
    teacher_config['pretrain_dir'] = 'pretrain/bert-base-chinese'
    teacher_config['num_classes'] = num_classes
    
    model_manager = ModelManager(teacher_config)
    teacher_model = model_manager.create_model()
    
    # 加载预训练权重
    teacher_model_path = 'checkpoints/best_model/pytorch_model.bin'
    if os.path.exists(teacher_model_path):
        state_dict = torch.load(teacher_model_path, map_location=device)
        teacher_model.load_state_dict(state_dict)
        print("成功加载教师模型权重")
    else:
        print("警告: 教师模型权重不存在，使用随机初始化")
    
    teacher_model.to(device)
    teacher_model.eval()
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"教师模型参数数量: {teacher_params:,}")
    
    # 创建学生模型
    print("\n3. 创建学生模型...")
    student_config = LSTMModelConfig()
    student_config['model_type'] = 'enhanced_bilstm_student'
    student_config['vocab_size'] = tokenizer.vocab_size
    student_config['embedding_dim'] = 128  # 较小的嵌入维度用于测试
    student_config['hidden_dim'] = 256    # 较小的隐藏维度用于测试
    student_config['num_layers'] = 2
    student_config['num_classes'] = num_classes
    
    student_model = create_lstm_student_model(student_config, tokenizer)
    student_model.to(device)
    
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"学生模型参数数量: {student_params:,}")
    print(f"压缩比: {teacher_params / student_params:.1f}x")
    
    # 测试前向传播
    print("\n4. 测试模型前向传播...")
    
    # 获取一个小批次数据
    batch = next(iter(train_loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    print(f"输入形状: {input_ids.shape}")
    
    # 教师模型前向传播
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids, attention_mask)
        teacher_logits = teacher_outputs['logits']
        teacher_features = teacher_outputs.get('pooler_output', None)
        if teacher_features is None:
            # 如果没有pooler_output，使用last_hidden_state的[CLS]位置
            teacher_features = teacher_model.bert(input_ids, attention_mask).last_hidden_state[:, 0, :]
    
    print(f"教师模型输出形状: {teacher_logits.shape}")
    print(f"教师模型特征形状: {teacher_features.shape}")
    
    # 学生模型前向传播
    student_outputs = student_model(input_ids, attention_mask, return_features=True)
    student_logits = student_outputs['logits']
    student_features = student_outputs['features']
    
    print(f"学生模型输出形状: {student_logits.shape}")
    print(f"学生模型特征形状: {student_features.shape}")
    print(f"学生模型原始特征形状: {student_outputs['lstm_features'].shape}")
    
    # 创建蒸馏训练器
    print("\n5. 创建蒸馏训练器...")
    distill_config = create_distillation_config()
    
    # 根据实际的特征维度设置配置
    actual_student_dim = student_outputs['lstm_features'].shape[-1]  # 原始特征维度
    projected_student_dim = student_outputs['features'].shape[-1]  # 投影后特征维度
    print(f"实际学生模型特征维度: {actual_student_dim}")
    print(f"投影后学生模型特征维度: {projected_student_dim}")
    
    distill_config.update({
        'temperature': 3.0,
        'alpha': 0.8,
        'feature_loss_weight': 0.1,
        'student_feature_dim': projected_student_dim,  # 使用投影后的维度
        'teacher_feature_dim': 768,
        'learning_rate': 5e-4,
        'warmup_steps': 100
    })
    
    trainer = DistillationTrainer(teacher_model, student_model, distill_config)
    
    # 短时间训练测试
    print("\n6. 开始短时间训练测试 (1个epoch)...")
    
    # 创建小的数据加载器用于测试
    small_train_dataset = torch.utils.data.Subset(train_dataset, range(min(20, len(train_dataset))))
    small_train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=4, shuffle=True)
    
    small_val_dataset = torch.utils.data.Subset(val_dataset, range(min(10, len(val_dataset)))) if val_dataset else None
    small_val_loader = torch.utils.data.DataLoader(small_val_dataset, batch_size=4, shuffle=False) if small_val_dataset else None
    
    try:
        history = trainer.train(
            train_loader=small_train_loader,
            val_loader=small_val_loader,
            device=device,
            num_epochs=1,
            save_dir='test_distilled_models'
        )
        
        print("\n训练测试完成!")
        print(f"最终训练损失: {history[-1]['train']['loss']:.4f}")
        print(f"最终训练准确率: {history[-1]['train']['accuracy']:.4f}")
        
        if history[-1]['val']:
            print(f"验证准确率: {history[-1]['val']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"训练过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n7. 测试学生模型推理...")
    
    # 测试学生模型独立推理
    student_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in small_train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = student_model(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"学生模型在小测试集上的准确率: {accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("知识蒸馏流程测试完成!")
    print("✅ 所有组件工作正常")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_distillation_pipeline()
    if success:
        print("\n🎉 知识蒸馏系统测试成功！可以开始正式训练。")
    else:
        print("\n❌ 测试失败，请检查错误信息。")