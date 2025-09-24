#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的BERT文本分类演示
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

def main():
    print("=" * 50)
    print("BERT中文文本分类演示")
    print("=" * 50)
    
    # 配置
    data_dir = "data"
    pretrain_dir = "pretrain/bert-base-chinese"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 加载类别名称
    class_file = os.path.join(data_dir, 'class.txt')
    with open(class_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"加载了 {len(class_names)} 个类别:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    print()
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrain_dir)
    print("tokenizer加载成功")
    
    # 加载模型
    model_path = "checkpoints/best_model/pytorch_model.bin"
    if os.path.exists(model_path):
        print("找到训练好的模型")
        model = BertForSequenceClassification.from_pretrained(
            pretrain_dir,
            num_labels=len(class_names)
        )
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("使用预训练模型（未微调）")
        model = BertForSequenceClassification.from_pretrained(
            pretrain_dir,
            num_labels=len(class_names)
        )
    
    model.to(device)
    model.eval()
    print("模型加载成功")
    
    # 演示文本
    demo_texts = [
        "A股市场今日大幅上涨，沪指收盘涨幅超过2%",
        "教育部发布新的高考改革方案", 
        "中国男篮在亚运会上获得金牌",
        "房地产市场调控政策进一步收紧",
        "科学家发现新的治疗癌症的方法"
    ]
    
    print("\n开始演示预测:")
    print("-" * 50)
    
    for text in demo_texts:
        # 编码
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # 结果
        pred_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][pred_id].item()
        pred_class = class_names[pred_id]
        
        print(f"文本: {text}")
        print(f"预测类别: {pred_class}")
        print(f"置信度: {confidence:.4f}")
        print("-" * 50)
    
    print("演示完成!")

if __name__ == "__main__":
    main()