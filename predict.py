#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT文本分类预测脚本
用于对新文本进行分类预测，支持单个文本和批量文本预测
"""

import os
import argparse
import logging
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
import time

from model import ModelManager, ModelConfig
from data_loader import DataLoader_Manager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationPredictor:
    """文本分类预测器"""
    
    def __init__(self, model_dir, device=None):
        """
        初始化预测器
        
        Args:
            model_dir: 模型保存目录
            device: 指定设备，None表示自动选择
        """
        self.model_dir = model_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        self.config = ModelConfig(os.path.join(model_dir, 'config.json'))
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        
        # 加载模型和tokenizer
        self._load_model()
        self._load_tokenizer()
        self._load_labels()
        
        logger.info(f"预测器初始化完成，使用设备: {self.device}")
    
    def _load_model(self):
        """加载模型"""
        try:
            model_manager = ModelManager(self.config)
            self.model = model_manager.load_model(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"成功加载模型: {self.model_dir}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _load_tokenizer(self):
        """加载tokenizer"""
        try:
            pretrain_dir = self.config['pretrain_dir']
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_dir)
            logger.info(f"成功加载tokenizer: {pretrain_dir}")
        except Exception as e:
            logger.error(f"加载tokenizer失败: {e}")
            raise
    
    def _load_labels(self):
        """加载标签映射"""
        try:
            # 尝试从data目录加载标签
            data_dir = "data"  # 默认数据目录
            class_file = os.path.join(data_dir, 'class.txt')
            
            if os.path.exists(class_file):
                with open(class_file, 'r', encoding='utf-8') as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]
                
                self.label_to_id = {label: idx for idx, label in enumerate(labels)}
                self.id_to_label = {idx: label for idx, label in enumerate(labels)}
                
                logger.info(f"成功加载 {len(labels)} 个类别标签")
            else:
                # 如果没有找到class.txt，创建默认标签
                num_classes = self.config['num_classes']
                self.id_to_label = {i: f"class_{i}" for i in range(num_classes)}
                self.label_to_id = {f"class_{i}": i for i in range(num_classes)}
                logger.warning(f"未找到class.txt，使用默认标签")
                
        except Exception as e:
            logger.error(f"加载标签映射失败: {e}")
            raise
    
    def predict_single(self, text, return_probabilities=False, top_k=None):
        """
        预测单个文本
        
        Args:
            text: 输入文本
            return_probabilities: 是否返回概率
            top_k: 返回前k个预测结果
            
        Returns:
            预测结果字典
        """
        # 文本预处理
        text = str(text).strip()
        if not text:
            return {"error": "输入文本为空"}
        
        # tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.get('max_length', 512),
            return_tensors='pt'
        )
        
        # 移到设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
        
        # 处理结果
        probs = probabilities.cpu().numpy()[0]
        predicted_class_id = np.argmax(probs)
        predicted_class_name = self.id_to_label[predicted_class_id]
        confidence = probs[predicted_class_id]
        
        result = {
            'text': text,
            'predicted_class': predicted_class_name,
            'predicted_class_id': int(predicted_class_id),
            'confidence': float(confidence)
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.id_to_label[i]: float(prob) 
                for i, prob in enumerate(probs)
            }
        
        if top_k:
            top_indices = np.argsort(probs)[::-1][:top_k]
            result['top_predictions'] = [
                {
                    'class': self.id_to_label[i],
                    'class_id': int(i),
                    'probability': float(probs[i])
                }
                for i in top_indices
            ]
        
        return result
    
    def predict_batch(self, texts, batch_size=32, return_probabilities=False, 
                     show_progress=True):
        """
        批量预测文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            return_probabilities: 是否返回概率
            show_progress: 是否显示进度条
            
        Returns:
            预测结果列表
        """
        results = []
        
        # 处理空文本
        texts = [str(text).strip() if text else "" for text in texts]
        
        # 批量处理
        for i in tqdm(range(0, len(texts), batch_size), 
                      desc="预测进度", disable=not show_progress):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._predict_batch_internal(
                batch_texts, return_probabilities
            )
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(self, batch_texts, return_probabilities=False):
        """内部批量预测函数"""
        # tokenization
        encodings = self.tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.get('max_length', 512),
            return_tensors='pt'
        )
        
        # 移到设备
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
        
        # 处理结果
        probs = probabilities.cpu().numpy()
        predicted_class_ids = np.argmax(probs, axis=1)
        
        results = []
        for i, text in enumerate(batch_texts):
            predicted_class_id = predicted_class_ids[i]
            predicted_class_name = self.id_to_label[predicted_class_id]
            confidence = probs[i][predicted_class_id]
            
            result = {
                'text': text,
                'predicted_class': predicted_class_name,
                'predicted_class_id': int(predicted_class_id),
                'confidence': float(confidence)
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    self.id_to_label[j]: float(prob) 
                    for j, prob in enumerate(probs[i])
                }
            
            results.append(result)
        
        return results
    
    def predict_from_file(self, input_file, output_file=None, 
                         text_column='text', batch_size=32):
        """
        从文件预测
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            text_column: 文本列名
            batch_size: 批次大小
        """
        logger.info(f"从文件预测: {input_file}")
        
        # 读取输入文件
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        elif input_file.endswith('.txt'):
            # 纯文本文件，每行一个文本
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines()]
            df = pd.DataFrame({text_column: texts})
        else:
            raise ValueError("不支持的文件格式，请使用csv、json或txt格式")
        
        if text_column not in df.columns:
            raise ValueError(f"文件中未找到列 '{text_column}'")
        
        # 批量预测
        texts = df[text_column].tolist()
        results = self.predict_batch(texts, batch_size, return_probabilities=True)
        
        # 添加预测结果到DataFrame
        df['predicted_class'] = [r['predicted_class'] for r in results]
        df['predicted_class_id'] = [r['predicted_class_id'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        # 添加所有类别的概率
        for class_id, class_name in self.id_to_label.items():
            df[f'prob_{class_name}'] = [
                r['probabilities'][class_name] for r in results
            ]
        
        # 保存结果
        if output_file:
            if output_file.endswith('.csv'):
                df.to_csv(output_file, index=False, encoding='utf-8')
            elif output_file.endswith('.json'):
                df.to_json(output_file, orient='records', force_ascii=False, indent=2)
            else:
                # 默认保存为CSV
                output_file = output_file + '.csv'
                df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"预测结果已保存到: {output_file}")
        
        return df
    
    def interactive_predict(self):
        """交互式预测"""
        logger.info("=== 交互式文本分类预测 ===")
        logger.info("输入文本进行分类预测，输入 'quit' 退出")
        logger.info(f"可用类别: {list(self.id_to_label.values())}")
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    print("请输入有效文本")
                    continue
                
                # 预测
                start_time = time.time()
                result = self.predict_single(text, return_probabilities=True, top_k=3)
                end_time = time.time()
                
                # 显示结果
                print(f"\n预测结果 (耗时: {end_time - start_time:.3f}s):")
                print(f"文本: {result['text']}")
                print(f"预测类别: {result['predicted_class']}")
                print(f"置信度: {result['confidence']:.4f}")
                
                print("\n前3个预测:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    print(f"  {i}. {pred['class']}: {pred['probability']:.4f}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"预测出错: {e}")
        
        print("退出交互式预测")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT文本分类预测')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'file', 'interactive'], 
                       default='interactive', help='预测模式')
    parser.add_argument('--text', type=str, help='单个文本预测时的输入文本')
    parser.add_argument('--texts', type=str, nargs='+', help='批量文本预测时的输入文本列表')
    parser.add_argument('--input_file', type=str, help='输入文件路径')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--text_column', type=str, default='text', help='文本列名')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--top_k', type=int, default=3, help='返回前k个预测结果')
    parser.add_argument('--device', type=str, help='指定设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设备配置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建预测器
    predictor = TextClassificationPredictor(args.model_dir, device)
    
    # 根据模式执行预测
    if args.mode == 'single':
        if not args.text:
            print("单个文本预测模式需要提供 --text 参数")
            return
        
        result = predictor.predict_single(
            args.text, return_probabilities=True, top_k=args.top_k
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.mode == 'batch':
        if not args.texts:
            print("批量文本预测模式需要提供 --texts 参数")
            return
        
        results = predictor.predict_batch(
            args.texts, batch_size=args.batch_size, return_probabilities=True
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
    
    elif args.mode == 'file':
        if not args.input_file:
            print("文件预测模式需要提供 --input_file 参数")
            return
        
        df = predictor.predict_from_file(
            args.input_file, args.output_file, 
            args.text_column, args.batch_size
        )
        print(f"成功预测 {len(df)} 条文本")
    
    elif args.mode == 'interactive':
        predictor.interactive_predict()

if __name__ == "__main__":
    main()