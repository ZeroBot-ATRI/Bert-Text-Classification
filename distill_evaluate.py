#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识蒸馏模型评估脚本
评估蒸馏后的LSTM学生模型性能
"""

import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
import pandas as pd

from data_loader import DataLoader_Manager
from model import BertClassifier, ModelConfig, ModelManager
from lstm_student_model import create_lstm_student_model, LSTMModelConfig
from knowledge_distillation import DistillationTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, data_manager, device):
        self.data_manager = data_manager
        self.device = device
        self.class_names = data_manager.get_label_names()
    
    def evaluate_model(self, model, test_loader, model_name="Model"):
        """评估单个模型"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        total_time = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                start_time = time.time()
                outputs = model(input_ids, attention_mask)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        # 推理时间
        avg_inference_time = total_time / len(test_loader.dataset) * 1000  # ms per sample
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist(),
            'avg_inference_time_ms': avg_inference_time,
            'total_inference_time_s': total_time,
            'predictions': all_predictions,
            'labels': all_labels,
            'logits': all_logits
        }
        
        return results
    
    def compare_models(self, teacher_results, student_results):
        """比较教师模型和学生模型"""
        logger.info("\n" + "=" * 80)
        logger.info("模型对比分析")
        logger.info("=" * 80)
        
        # 基础指标对比
        logger.info(f"{'指标':<15} {'教师模型(BERT)':<20} {'学生模型(LSTM)':<20} {'差异':<15}")
        logger.info("-" * 70)
        
        accuracy_diff = student_results['accuracy'] - teacher_results['accuracy']
        logger.info(f"{'准确率':<15} {teacher_results['accuracy']:<20.4f} {student_results['accuracy']:<20.4f} {accuracy_diff:+.4f}")
        
        precision_diff = student_results['precision'] - teacher_results['precision']
        logger.info(f"{'精确率':<15} {teacher_results['precision']:<20.4f} {student_results['precision']:<20.4f} {precision_diff:+.4f}")
        
        recall_diff = student_results['recall'] - teacher_results['recall']
        logger.info(f"{'召回率':<15} {teacher_results['recall']:<20.4f} {student_results['recall']:<20.4f} {recall_diff:+.4f}")
        
        f1_diff = student_results['f1'] - teacher_results['f1']
        logger.info(f"{'F1分数':<15} {teacher_results['f1']:<20.4f} {student_results['f1']:<20.4f} {f1_diff:+.4f}")
        
        # 推理速度对比
        speed_up = teacher_results['avg_inference_time_ms'] / student_results['avg_inference_time_ms']
        logger.info(f"{'推理时间(ms)':<15} {teacher_results['avg_inference_time_ms']:<20.4f} {student_results['avg_inference_time_ms']:<20.4f} {speed_up:.1f}x faster")
        
        # 性能保持率
        performance_retention = student_results['accuracy'] / teacher_results['accuracy'] * 100
        logger.info(f"\n性能保持率: {performance_retention:.1f}%")
        
        return {
            'accuracy_diff': accuracy_diff,
            'precision_diff': precision_diff,
            'recall_diff': recall_diff,
            'f1_diff': f1_diff,
            'speed_up': speed_up,
            'performance_retention': performance_retention
        }
    
    def generate_classification_report(self, results, save_path=None):
        """生成分类报告"""
        report = classification_report(
            results['labels'], 
            results['predictions'],
            target_names=self.class_names,
            output_dict=True
        )
        
        report_str = classification_report(
            results['labels'], 
            results['predictions'],
            target_names=self.class_names
        )
        
        logger.info(f"\n{results['model_name']} 分类报告:")
        logger.info("\n" + report_str)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"{results['model_name']} 分类报告:\n")
                f.write(report_str)
        
        return report
    
    def plot_confusion_matrix(self, results, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'{results["model_name"]} 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_performance_comparison(self, teacher_results, student_results, save_path=None):
        """绘制性能对比图"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        teacher_scores = [teacher_results[m] for m in metrics]
        student_scores = [student_results[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, teacher_scores, width, label='教师模型(BERT)', alpha=0.8)
        bars2 = ax.bar(x + width/2, student_scores, width, label='学生模型(LSTM)', alpha=0.8)
        
        ax.set_xlabel('评估指标')
        ax.set_ylabel('分数')
        ax.set_title('教师模型vs学生模型性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(['准确率', '精确率', '召回率', 'F1分数'])
        ax.legend()
        
        # 在柱子上添加数值
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能对比图已保存到: {save_path}")
        
        plt.show()
        plt.close()
    
    def analyze_prediction_confidence(self, results, save_path=None):
        """分析预测置信度"""
        logits = np.array(results['logits'])
        predictions = np.array(results['predictions'])
        labels = np.array(results['labels'])
        
        # 计算softmax概率
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        max_probs = np.max(probs, axis=1)
        
        # 正确和错误预测的置信度分布
        correct_mask = predictions == labels
        correct_confidence = max_probs[correct_mask]
        incorrect_confidence = max_probs[~correct_mask]
        
        plt.figure(figsize=(12, 5))
        
        # 置信度分布
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidence, bins=20, alpha=0.7, label='正确预测', density=True)
        plt.hist(incorrect_confidence, bins=20, alpha=0.7, label='错误预测', density=True)
        plt.xlabel('预测置信度')
        plt.ylabel('密度')
        plt.title('预测置信度分布')
        plt.legend()
        
        # 置信度vs准确率
        plt.subplot(1, 2, 2)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        
        for i in range(len(bins) - 1):
            mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(predictions[mask] == labels[mask])
                bin_accuracies.append(acc)
            else:
                bin_accuracies.append(0)
        
        plt.plot(bin_centers, bin_accuracies, 'o-', label='实际准确率')
        plt.plot([0, 1], [0, 1], '--', label='理想校准线')
        plt.xlabel('预测置信度')
        plt.ylabel('准确率')
        plt.title('置信度校准曲线')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"置信度分析图已保存到: {save_path}")
        
        plt.show()
        plt.close()
        
        return {
            'avg_correct_confidence': np.mean(correct_confidence),
            'avg_incorrect_confidence': np.mean(incorrect_confidence),
            'confidence_gap': np.mean(correct_confidence) - np.mean(incorrect_confidence)
        }

def load_student_model(model_path, config, tokenizer, device):
    """加载学生模型"""
    student_model = create_lstm_student_model(config, tokenizer)
    
    model_file = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location=device)
        student_model.load_state_dict(state_dict)
        logger.info(f"成功加载学生模型: {model_file}")
    else:
        raise FileNotFoundError(f"学生模型文件不存在: {model_file}")
    
    student_model.to(device)
    return student_model

def load_teacher_model(model_path, config, device):
    """加载教师模型"""
    model_manager = ModelManager(config)
    teacher_model = model_manager.create_model()
    
    model_file = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location=device)
        teacher_model.load_state_dict(state_dict)
        logger.info(f"成功加载教师模型: {model_file}")
    else:
        raise FileNotFoundError(f"教师模型文件不存在: {model_file}")
    
    teacher_model.to(device)
    return teacher_model

def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='知识蒸馏模型评估')
    
    # 数据和模型路径
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--teacher_model_path', type=str, default='checkpoints/best_model',
                       help='教师模型路径')
    parser.add_argument('--student_model_path', type=str, default='distilled_models/best_student_model',
                       help='学生模型路径')
    parser.add_argument('--pretrain_dir', type=str, default='pretrain/bert-base-chinese',
                       help='预训练BERT模型目录')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='评估结果输出目录')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    
    # 学生模型参数 (需要与训练时保持一致)
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='词嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--student_model_type', type=str, default='enhanced_bilstm_student',
                       choices=['bilstm_student', 'enhanced_bilstm_student'],
                       help='学生模型类型')
    
    # 其他参数
    parser.add_argument('--no_cuda', action='store_true',
                       help='不使用GPU')
    parser.add_argument('--save_plots', action='store_true',
                       help='保存图表')
    
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据管理器
    data_manager = DataLoader_Manager(args.data_dir, args.pretrain_dir, args.max_length)
    
    # 创建测试数据加载器
    _, _, test_dataset = data_manager.create_datasets()
    _, _, test_loader = data_manager.create_dataloaders(
        None, None, test_dataset, batch_size=args.batch_size
    )
    
    num_classes = data_manager.get_num_classes()
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_dir)
    
    # 创建评估器
    evaluator = ModelEvaluator(data_manager, device)
    
    # 加载并评估教师模型
    logger.info("评估教师模型...")
    teacher_config = ModelConfig()
    teacher_config['pretrain_dir'] = args.pretrain_dir
    teacher_config['num_classes'] = num_classes
    
    teacher_model = load_teacher_model(args.teacher_model_path, teacher_config, device)
    teacher_results = evaluator.evaluate_model(teacher_model, test_loader, "教师模型(BERT)")
    
    # 加载并评估学生模型
    logger.info("评估学生模型...")
    student_config = LSTMModelConfig()
    student_config['model_type'] = args.student_model_type
    student_config['vocab_size'] = tokenizer.vocab_size
    student_config['embedding_dim'] = args.embedding_dim
    student_config['hidden_dim'] = args.hidden_dim
    student_config['num_layers'] = args.num_layers
    student_config['num_classes'] = num_classes
    
    student_model = load_student_model(args.student_model_path, student_config, tokenizer, device)
    student_results = evaluator.evaluate_model(student_model, test_loader, "学生模型(LSTM)")
    
    # 比较模型
    comparison = evaluator.compare_models(teacher_results, student_results)
    
    # 生成分类报告
    teacher_report = evaluator.generate_classification_report(
        teacher_results, 
        os.path.join(args.output_dir, 'teacher_classification_report.txt')
    )
    
    student_report = evaluator.generate_classification_report(
        student_results,
        os.path.join(args.output_dir, 'student_classification_report.txt')
    )
    
    # 绘制图表
    if args.save_plots:
        # 混淆矩阵
        evaluator.plot_confusion_matrix(
            teacher_results,
            os.path.join(args.output_dir, 'teacher_confusion_matrix.png')
        )
        
        evaluator.plot_confusion_matrix(
            student_results,
            os.path.join(args.output_dir, 'student_confusion_matrix.png')
        )
        
        # 性能对比
        evaluator.plot_performance_comparison(
            teacher_results, student_results,
            os.path.join(args.output_dir, 'performance_comparison.png')
        )
        
        # 置信度分析
        teacher_confidence = evaluator.analyze_prediction_confidence(
            teacher_results,
            os.path.join(args.output_dir, 'teacher_confidence_analysis.png')
        )
        
        student_confidence = evaluator.analyze_prediction_confidence(
            student_results,
            os.path.join(args.output_dir, 'student_confidence_analysis.png')
        )
    
    # 保存评估结果
    evaluation_results = {
        'teacher_results': {k: v for k, v in teacher_results.items() if k not in ['predictions', 'labels', 'logits']},
        'student_results': {k: v for k, v in student_results.items() if k not in ['predictions', 'labels', 'logits']},
        'comparison': comparison,
        'model_info': {
            'teacher_params': sum(p.numel() for p in teacher_model.parameters()),
            'student_params': sum(p.numel() for p in student_model.parameters()),
            'compression_ratio': sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()),
            'test_samples': len(test_dataset)
        },
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"评估结果已保存到: {args.output_dir}")
    
    # 输出总结
    logger.info("\n" + "=" * 80)
    logger.info("评估总结")
    logger.info("=" * 80)
    logger.info(f"模型压缩比: {evaluation_results['model_info']['compression_ratio']:.1f}x")
    logger.info(f"推理加速比: {comparison['speed_up']:.1f}x")
    logger.info(f"性能保持率: {comparison['performance_retention']:.1f}%")
    logger.info(f"准确率损失: {-comparison['accuracy_diff']:.4f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()