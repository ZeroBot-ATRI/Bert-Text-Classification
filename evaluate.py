#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估和性能分析工具
包含详细的模型评估指标、混淆矩阵、分类报告等
"""

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from predict import TextClassificationPredictor
from data_loader import DataLoader_Manager
from model import ModelManager, ModelConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_dir, data_dir, device=None):
        """
        初始化评估器
        
        Args:
            model_dir: 模型目录
            data_dir: 数据目录
            device: 设备
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化预测器
        self.predictor = TextClassificationPredictor(model_dir, device)
        
        # 加载数据管理器
        config = ModelConfig(os.path.join(model_dir, 'config.json'))
        pretrain_dir = config['pretrain_dir']
        self.data_manager = DataLoader_Manager(data_dir, pretrain_dir)
        
        logger.info("评估器初始化完成")
    
    def evaluate_on_dataset(self, dataset_type='test', batch_size=32):
        """
        在数据集上评估模型
        
        Args:
            dataset_type: 数据集类型 ('test', 'dev')
            batch_size: 批次大小
            
        Returns:
            评估结果字典
        """
        logger.info(f"在 {dataset_type} 数据集上评估模型...")
        
        # 加载数据
        if dataset_type == 'test':
            file_path = os.path.join(self.data_dir, 'test.txt')
        elif dataset_type == 'dev':
            file_path = os.path.join(self.data_dir, 'dev.txt')
        else:
            raise ValueError("dataset_type 必须是 'test' 或 'dev'")
        
        texts, true_labels = self.data_manager.load_data_from_file(file_path)
        
        # 批量预测
        predictions = self.predictor.predict_batch(
            texts, batch_size=batch_size, 
            return_probabilities=True, show_progress=True
        )
        
        # 提取预测标签和概率
        pred_labels = [p['predicted_class_id'] for p in predictions]
        pred_probs = np.array([
            [p['probabilities'][self.predictor.id_to_label[i]] 
             for i in range(len(self.predictor.id_to_label))]
            for p in predictions
        ])
        
        # 计算评估指标
        results = self._calculate_metrics(true_labels, pred_labels, pred_probs)
        results['dataset_type'] = dataset_type
        results['num_samples'] = len(texts)
        
        return results, predictions, texts, true_labels
    
    def _calculate_metrics(self, true_labels, pred_labels, pred_probs):
        """计算评估指标"""
        # 基本指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # 宏平均指标
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='macro', zero_division=0
        )
        
        # 微平均指标
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='micro', zero_division=0
        )
        
        # 每个类别的指标
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0
        )
        
        # AUC指标（多分类）
        try:
            # 二值化标签用于计算多分类AUC
            n_classes = len(self.predictor.id_to_label)
            y_true_bin = label_binarize(true_labels, classes=list(range(n_classes)))
            
            if n_classes == 2:
                # 二分类情况
                auc_score = roc_auc_score(true_labels, pred_probs[:, 1])
            else:
                # 多分类情况
                auc_score = roc_auc_score(y_true_bin, pred_probs, 
                                        average='weighted', multi_class='ovr')
        except Exception as e:
            logger.warning(f"计算AUC失败: {e}")
            auc_score = None
        
        # 构建结果字典
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'auc': auc_score,
            'class_metrics': {
                self.predictor.id_to_label[i]: {
                    'precision': class_precision[i],
                    'recall': class_recall[i],
                    'f1': class_f1[i],
                    'support': int(class_support[i])
                }
                for i in range(len(class_precision))
            }
        }
        
        return results
    
    def plot_confusion_matrix(self, true_labels, pred_labels, 
                            save_path=None, figsize=(10, 8)):
        """绘制混淆矩阵"""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=figsize)
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.predictor.id_to_label.values()),
                   yticklabels=list(self.predictor.id_to_label.values()))
        
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_classification_report(self, true_labels, pred_labels, 
                                 save_path=None, figsize=(12, 8)):
        """绘制分类报告热力图"""
        # 获取分类报告
        report = classification_report(
            true_labels, pred_labels,
            target_names=list(self.predictor.id_to_label.values()),
            output_dict=True,
            zero_division=0
        )
        
        # 转换为DataFrame
        df = pd.DataFrame(report).iloc[:-1, :].T  # 排除最后的汇总行
        df = df.iloc[:, :-1]  # 排除support列用于热力图
        
        plt.figure(figsize=figsize)
        sns.heatmap(df, annot=True, fmt='.3f', cmap='Blues', 
                   cbar_kws={'label': '分数'})
        plt.title('分类报告')
        plt.xlabel('指标')
        plt.ylabel('类别')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分类报告图已保存到: {save_path}")
        
        plt.show()
        
        return report
    
    def plot_roc_curves(self, true_labels, pred_probs, 
                       save_path=None, figsize=(12, 10)):
        """绘制ROC曲线"""
        n_classes = len(self.predictor.id_to_label)
        
        # 二值化标签
        y_true_bin = label_binarize(true_labels, classes=list(range(n_classes)))
        
        if n_classes == 2:
            # 二分类情况
            fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率')
            plt.ylabel('真正率')
            plt.title('ROC曲线')
            plt.legend(loc="lower right")
        else:
            # 多分类情况
            plt.figure(figsize=figsize)
            
            # 计算每个类别的ROC曲线
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 绘制每个类别的ROC曲线
            colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
            for i, color in zip(range(n_classes), colors):
                class_name = self.predictor.id_to_label[i]
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率')
            plt.ylabel('真正率')
            plt.title('多分类ROC曲线')
            plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC曲线已保存到: {save_path}")
        
        plt.show()
    
    def analyze_predictions(self, predictions, texts, true_labels, 
                          save_path=None, top_k=5):
        """分析预测结果"""
        results = []
        
        for i, (pred, text, true_label) in enumerate(zip(predictions, texts, true_labels)):
            pred_label = pred['predicted_class_id']
            confidence = pred['confidence']
            correct = (pred_label == true_label)
            
            result = {
                'index': i,
                'text': text,
                'true_label': true_label,
                'true_class': self.predictor.id_to_label[true_label],
                'pred_label': pred_label,
                'pred_class': pred['predicted_class'],
                'confidence': confidence,
                'correct': correct
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # 分析统计
        logger.info("=== 预测结果分析 ===")
        logger.info(f"总样本数: {len(df)}")
        logger.info(f"预测正确数: {df['correct'].sum()}")
        logger.info(f"准确率: {df['correct'].mean():.4f}")
        
        # 置信度分析
        logger.info(f"平均置信度: {df['confidence'].mean():.4f}")
        logger.info(f"正确预测平均置信度: {df[df['correct']]['confidence'].mean():.4f}")
        logger.info(f"错误预测平均置信度: {df[~df['correct']]['confidence'].mean():.4f}")
        
        # 最高置信度的错误预测
        wrong_predictions = df[~df['correct']].nlargest(top_k, 'confidence')
        if not wrong_predictions.empty:
            logger.info(f"\n最高置信度的 {len(wrong_predictions)} 个错误预测:")
            for _, row in wrong_predictions.iterrows():
                logger.info(f"  文本: {row['text'][:50]}...")
                logger.info(f"  真实: {row['true_class']}, 预测: {row['pred_class']}, "
                          f"置信度: {row['confidence']:.4f}")
        
        # 最低置信度的正确预测
        correct_predictions = df[df['correct']].nsmallest(top_k, 'confidence')
        if not correct_predictions.empty:
            logger.info(f"\n最低置信度的 {len(correct_predictions)} 个正确预测:")
            for _, row in correct_predictions.iterrows():
                logger.info(f"  文本: {row['text'][:50]}...")
                logger.info(f"  类别: {row['true_class']}, 置信度: {row['confidence']:.4f}")
        
        # 保存详细结果
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8')
            logger.info(f"详细预测结果已保存到: {save_path}")
        
        return df
    
    def generate_report(self, output_dir='evaluation_results'):
        """生成完整的评估报告"""
        logger.info("生成完整评估报告...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 在测试集上评估
        results, predictions, texts, true_labels = self.evaluate_on_dataset('test')
        
        # 保存评估指标
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成图表
        pred_labels = [p['predicted_class_id'] for p in predictions]
        pred_probs = np.array([
            [p['probabilities'][self.predictor.id_to_label[i]] 
             for i in range(len(self.predictor.id_to_label))]
            for p in predictions
        ])
        
        # 混淆矩阵
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(true_labels, pred_labels, save_path=cm_path)
        
        # 分类报告
        report_path = os.path.join(output_dir, 'classification_report.png')
        self.plot_classification_report(true_labels, pred_labels, save_path=report_path)
        
        # ROC曲线
        roc_path = os.path.join(output_dir, 'roc_curves.png')
        self.plot_roc_curves(true_labels, pred_probs, save_path=roc_path)
        
        # 预测分析
        analysis_path = os.path.join(output_dir, 'prediction_analysis.csv')
        self.analyze_predictions(predictions, texts, true_labels, save_path=analysis_path)
        
        # 生成文本报告
        report_text_path = os.path.join(output_dir, 'evaluation_report.txt')
        self._generate_text_report(results, report_text_path)
        
        logger.info(f"评估报告已生成到: {output_dir}")
        
        return results
    
    def _generate_text_report(self, results, save_path):
        """生成文本格式的评估报告"""
        report_lines = [
            "=" * 50,
            "BERT文本分类模型评估报告",
            "=" * 50,
            "",
            f"模型目录: {self.model_dir}",
            f"数据集: {results['dataset_type']}",
            f"样本数量: {results['num_samples']}",
            "",
            "整体性能指标:",
            f"  准确率 (Accuracy): {results['accuracy']:.4f}",
            f"  加权精确率: {results['precision_weighted']:.4f}",
            f"  加权召回率: {results['recall_weighted']:.4f}",
            f"  加权F1分数: {results['f1_weighted']:.4f}",
            "",
            "宏平均指标:",
            f"  精确率: {results['precision_macro']:.4f}",
            f"  召回率: {results['recall_macro']:.4f}",
            f"  F1分数: {results['f1_macro']:.4f}",
            "",
            "微平均指标:",
            f"  精确率: {results['precision_micro']:.4f}",
            f"  召回率: {results['recall_micro']:.4f}",
            f"  F1分数: {results['f1_micro']:.4f}",
            ""
        ]
        
        if results['auc'] is not None:
            report_lines.append(f"AUC分数: {results['auc']:.4f}")
            report_lines.append("")
        
        report_lines.append("各类别详细指标:")
        for class_name, metrics in results['class_metrics'].items():
            report_lines.extend([
                f"  {class_name}:",
                f"    精确率: {metrics['precision']:.4f}",
                f"    召回率: {metrics['recall']:.4f}",
                f"    F1分数: {metrics['f1']:.4f}",
                f"    样本数: {metrics['support']}",
                ""
            ])
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT模型评估工具')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--dataset', type=str, choices=['test', 'dev'], default='test', help='评估数据集')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, help='指定设备')
    
    args = parser.parse_args()
    
    # 设备配置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_dir, args.data_dir, device)
    
    # 生成评估报告
    results = evaluator.generate_report(args.output_dir)
    
    # 打印主要指标
    logger.info("=== 评估完成 ===")
    logger.info(f"准确率: {results['accuracy']:.4f}")
    logger.info(f"F1分数(加权): {results['f1_weighted']:.4f}")
    logger.info(f"F1分数(宏平均): {results['f1_macro']:.4f}")

if __name__ == "__main__":
    main()