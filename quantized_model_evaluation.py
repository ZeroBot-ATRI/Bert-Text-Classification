#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化模型评估脚本
比较原始模型和量化模型的性能、推理速度、内存占用等指标
"""

import os
import time
import json
import logging
import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from evaluate import ModelEvaluator
from model import ModelManager, ModelConfig
from model_optimization import ModelOptimizer
from true_quantization import TrueModelOptimizer, TrueQuantizedLinear
from data_loader import DataLoader_Manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizedModelEvaluator:
    """量化模型评估器"""
    
    def __init__(self, original_model_path: str, data_dir: str = "data"):
        """
        初始化评估器
        
        Args:
            original_model_path: 原始模型路径
            data_dir: 数据目录
        """
        self.original_model_path = original_model_path
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载原始模型
        config = ModelConfig()
        self.model_manager = ModelManager(config)
        self.original_model = self.model_manager.load_model(original_model_path)
        
        # 加载数据管理器
        self.data_manager = DataLoader_Manager(data_dir, config['pretrain_dir'])
        
        # 评估结果存储
        self.evaluation_results = {}
        
        logger.info("量化模型评估器初始化完成")
    
    def get_model_size_info(self, model: nn.Module, model_name: str) -> Dict:
        """获取模型大小信息"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        # 统计量化层信息
        quantized_layers = 0
        total_linear_layers = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_linear_layers += 1
            elif isinstance(module, TrueQuantizedLinear):
                quantized_layers += 1
                total_linear_layers += 1
        
        return {
            'model_name': model_name,
            'total_size_mb': total_size / 1024 / 1024,
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_params': sum(p.numel() for p in model.parameters()),
            'quantized_layers': quantized_layers,
            'total_linear_layers': total_linear_layers,
            'quantization_ratio': quantized_layers / total_linear_layers if total_linear_layers > 0 else 0
        }
    
    def measure_inference_speed(self, model: nn.Module, test_texts: List[str], 
                              num_runs: int = 10) -> Dict:
        """测量推理速度"""
        model.eval()
        model.to(self.device)
        
        # 准备测试数据
        test_sample = test_texts[:min(100, len(test_texts))]  # 使用前100个样本
        
        # 预热
        with torch.no_grad():
            for text in test_sample[:5]:
                try:
                    # 这里需要根据你的模型输入格式调整
                    inputs = self.data_manager.tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    _ = model(input_ids, attention_mask)
                except Exception as e:
                    logger.warning(f"预热时出错: {e}")
                    break
        
        # 正式测试
        times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                for text in test_sample:
                    try:
                        inputs = self.data_manager.tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=512,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        input_ids = inputs['input_ids'].to(self.device)
                        attention_mask = inputs['attention_mask'].to(self.device)
                        
                        _ = model(input_ids, attention_mask)
                    except Exception as e:
                        logger.warning(f"推理时出错: {e}")
                        continue
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            'avg_time_per_batch': avg_time,
            'std_time_per_batch': std_time,
            'avg_time_per_sample': avg_time / len(test_sample),
            'throughput_samples_per_sec': len(test_sample) / avg_time,
            'num_samples': len(test_sample),
            'num_runs': num_runs
        }
    
    def measure_memory_usage(self, model: nn.Module) -> Dict:
        """测量内存占用"""
        # GPU内存（如果可用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated()
            
            model.to(self.device)
            
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = (gpu_memory_after - gpu_memory_before) / 1024 / 1024
        else:
            gpu_memory_used = 0
        
        # CPU内存
        process = psutil.Process()
        cpu_memory_info = process.memory_info()
        
        return {
            'gpu_memory_mb': gpu_memory_used,
            'cpu_memory_mb': cpu_memory_info.rss / 1024 / 1024,
            'cpu_memory_percent': process.memory_percent()
        }
    
    def evaluate_model_accuracy(self, model: nn.Module, model_name: str, 
                              dataset_type: str = 'test') -> Dict:
        """评估模型精度"""
        logger.info(f"评估 {model_name} 在 {dataset_type} 数据集上的精度...")
        
        # 加载测试数据
        if dataset_type == 'test':
            file_path = os.path.join(self.data_dir, 'test.txt')
        else:
            file_path = os.path.join(self.data_dir, 'dev.txt')
        
        texts, true_labels = self.data_manager.load_data_from_file(file_path)
        
        # 进行预测
        model.eval()
        model.to(self.device)
        
        predictions = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (text, true_label) in enumerate(tqdm(zip(texts, true_labels), 
                                                       desc=f"评估{model_name}")):
                try:
                    inputs = self.data_manager.tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask)
                    
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    predicted_label = torch.argmax(logits, dim=1).item()
                    predictions.append(predicted_label)
                    
                    if predicted_label == true_label:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"预测第 {i} 个样本时出错: {e}")
                    predictions.append(0)  # 默认预测
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def compare_models(self, quantized_model_paths: List[str]) -> Dict:
        """比较原始模型和量化模型"""
        logger.info("开始比较原始模型和量化模型...")
        
        # 加载测试数据（用于速度测试）
        test_file = os.path.join(self.data_dir, 'test.txt')
        test_texts, test_labels = self.data_manager.load_data_from_file(test_file)
        test_texts = test_texts[:50]  # 限制样本数量以加快测试
        
        comparison_results = {
            'models': {},
            'summary': {}
        }
        
        # 评估原始模型
        logger.info("评估原始模型...")
        
        original_size_info = self.get_model_size_info(self.original_model, "原始模型")
        original_speed_info = self.measure_inference_speed(self.original_model, test_texts)
        original_memory_info = self.measure_memory_usage(self.original_model)
        original_accuracy_info = self.evaluate_model_accuracy(self.original_model, "原始模型")
        
        comparison_results['models']['original'] = {
            'size': original_size_info,
            'speed': original_speed_info,
            'memory': original_memory_info,
            'accuracy': original_accuracy_info
        }
        
        # 评估量化模型
        for i, quantized_path in enumerate(quantized_model_paths):
            model_name = f"量化模型_{i+1}"
            
            if not os.path.exists(quantized_path):
                logger.warning(f"量化模型路径不存在: {quantized_path}")
                continue
            
            logger.info(f"评估 {model_name}...")
            
            try:
                # 加载量化模型
                optimizer = ModelOptimizer(self.original_model_path)
                optimizer.load_optimized_model(quantized_path)
                quantized_model = optimizer.optimized_model
                
                if quantized_model is None:
                    logger.warning(f"加载量化模型失败: {quantized_path}")
                    continue
                
                # 评估各项指标
                size_info = self.get_model_size_info(quantized_model, model_name)
                speed_info = self.measure_inference_speed(quantized_model, test_texts)
                memory_info = self.measure_memory_usage(quantized_model)
                accuracy_info = self.evaluate_model_accuracy(quantized_model, model_name)
                
                comparison_results['models'][f'quantized_{i+1}'] = {
                    'path': quantized_path,
                    'size': size_info,
                    'speed': speed_info,
                    'memory': memory_info,
                    'accuracy': accuracy_info
                }
                
            except Exception as e:
                logger.error(f"评估量化模型 {quantized_path} 时出错: {e}")
                continue
        
        # 计算改善比例
        self._calculate_improvements(comparison_results)
        
        return comparison_results
    
    def _calculate_improvements(self, comparison_results: Dict):
        """计算改善比例"""
        if 'original' not in comparison_results['models']:
            return
        
        original = comparison_results['models']['original']
        
        comparison_results['summary'] = {
            'original_model': {
                'size_mb': original['size']['total_size_mb'],
                'accuracy': original['accuracy']['accuracy'],
                'inference_speed': original['speed']['throughput_samples_per_sec']
            },
            'quantized_models': {}
        }
        
        for key, model_data in comparison_results['models'].items():
            if key == 'original':
                continue
            
            quantized = model_data
            
            # 计算压缩比
            size_reduction = (1 - quantized['size']['total_size_mb'] / 
                            original['size']['total_size_mb']) * 100
            
            # 计算精度损失
            accuracy_loss = (original['accuracy']['accuracy'] - 
                           quantized['accuracy']['accuracy']) * 100
            
            # 计算速度提升
            speed_improvement = (quantized['speed']['throughput_samples_per_sec'] / 
                               original['speed']['throughput_samples_per_sec'] - 1) * 100
            
            comparison_results['summary']['quantized_models'][key] = {
                'size_mb': quantized['size']['total_size_mb'],
                'size_reduction_percent': size_reduction,
                'accuracy': quantized['accuracy']['accuracy'],
                'accuracy_loss_percent': accuracy_loss,
                'inference_speed': quantized['speed']['throughput_samples_per_sec'],
                'speed_improvement_percent': speed_improvement
            }
    
    def save_evaluation_results(self, results: Dict, save_dir: str = "evaluation_results"):
        """保存评估结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(save_dir, 'quantized_model_evaluation.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成报告
        report_file = os.path.join(save_dir, 'quantized_model_report.txt')
        self._generate_report(results, report_file)
        
        # 生成对比图表
        self._plot_comparison_charts(results, save_dir)
        
        logger.info(f"评估结果已保存到: {save_dir}")
    
    def _generate_report(self, results: Dict, report_file: str):
        """生成评估报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("量化模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 摘要
            if 'summary' in results:
                f.write("📊 评估摘要\n")
                f.write("-" * 30 + "\n")
                
                summary = results['summary']
                
                # 原始模型信息
                if 'original_model' in summary:
                    orig = summary['original_model']
                    f.write(f"原始模型:\n")
                    f.write(f"  模型大小: {orig['size_mb']:.2f} MB\n")
                    f.write(f"  准确率: {orig['accuracy']:.4f}\n")
                    f.write(f"  推理速度: {orig['inference_speed']:.2f} 样本/秒\n\n")
                
                # 量化模型对比
                if 'quantized_models' in summary:
                    for model_name, model_data in summary['quantized_models'].items():
                        f.write(f"{model_name}:\n")
                        f.write(f"  模型大小: {model_data['size_mb']:.2f} MB "
                               f"(减少 {model_data['size_reduction_percent']:.1f}%)\n")
                        f.write(f"  准确率: {model_data['accuracy']:.4f} "
                               f"(损失 {model_data['accuracy_loss_percent']:.2f}%)\n")
                        f.write(f"  推理速度: {model_data['inference_speed']:.2f} 样本/秒 "
                               f"(提升 {model_data['speed_improvement_percent']:.1f}%)\n\n")
            
            # 详细结果
            f.write("\n📋 详细评估结果\n")
            f.write("-" * 30 + "\n")
            
            for model_key, model_data in results['models'].items():
                f.write(f"\n{model_data['size']['model_name']}:\n")
                f.write(f"  模型大小: {model_data['size']['total_size_mb']:.2f} MB\n")
                f.write(f"  参数数量: {model_data['size']['total_params']:,}\n")
                f.write(f"  量化层数: {model_data['size']['quantized_layers']}/{model_data['size']['total_linear_layers']}\n")
                f.write(f"  准确率: {model_data['accuracy']['accuracy']:.4f}\n")
                f.write(f"  推理速度: {model_data['speed']['throughput_samples_per_sec']:.2f} 样本/秒\n")
                f.write(f"  GPU内存: {model_data['memory']['gpu_memory_mb']:.2f} MB\n")
    
    def _plot_comparison_charts(self, results: Dict, save_dir: str):
        """绘制对比图表"""
        if 'summary' not in results or 'quantized_models' not in results['summary']:
            return
        
        summary = results['summary']
        
        # 准备数据
        model_names = ['原始模型']
        sizes = [summary['original_model']['size_mb']]
        accuracies = [summary['original_model']['accuracy']]
        speeds = [summary['original_model']['inference_speed']]
        
        for model_name, model_data in summary['quantized_models'].items():
            model_names.append(model_name.replace('quantized_', '量化模型'))
            sizes.append(model_data['size_mb'])
            accuracies.append(model_data['accuracy'])
            speeds.append(model_data['inference_speed'])
        
        # 创建对比图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 模型大小对比
        bars1 = ax1.bar(model_names, sizes, color=['blue', 'red', 'green', 'orange'][:len(sizes)])
        ax1.set_title('模型大小对比')
        ax1.set_ylabel('大小 (MB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, size in zip(bars1, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{size:.1f}MB', ha='center', va='bottom')
        
        # 准确率对比
        bars2 = ax2.bar(model_names, accuracies, color=['blue', 'red', 'green', 'orange'][:len(accuracies)])
        ax2.set_title('准确率对比')
        ax2.set_ylabel('准确率')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars2, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 推理速度对比
        bars3 = ax3.bar(model_names, speeds, color=['blue', 'red', 'green', 'orange'][:len(speeds)])
        ax3.set_title('推理速度对比')
        ax3.set_ylabel('样本/秒')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, speed in zip(bars3, speeds):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{speed:.1f}', ha='center', va='bottom')
        
        # 压缩效果图
        if len(sizes) > 1:
            reduction_rates = [(1 - size/sizes[0]) * 100 for size in sizes[1:]]
            bars4 = ax4.bar(model_names[1:], reduction_rates, color=['red', 'green', 'orange'][:len(reduction_rates)])
            ax4.set_title('模型压缩效果')
            ax4.set_ylabel('压缩率 (%)')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, rate in zip(bars4, reduction_rates):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'quantized_model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("对比图表已保存")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='量化模型评估')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model',
                       help='原始模型路径')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--quantized_models', type=str, nargs='+',
                       default=['optimized_models/nf4_quantized', 
                               'optimized_models/combined_pruned_0.2_nf4'],
                       help='量化模型路径列表')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = QuantizedModelEvaluator(args.model_path, args.data_dir)
    
    # 进行比较评估
    results = evaluator.compare_models(args.quantized_models)
    
    # 保存结果
    evaluator.save_evaluation_results(results, args.output_dir)
    
    # 打印摘要
    print("\n" + "="*60)
    print("📊 量化模型评估完成")
    print("="*60)
    
    if 'summary' in results:
        summary = results['summary']
        
        if 'original_model' in summary:
            orig = summary['original_model']
            print(f"\n原始模型:")
            print(f"  大小: {orig['size_mb']:.2f} MB")
            print(f"  准确率: {orig['accuracy']:.4f}")
            print(f"  速度: {orig['inference_speed']:.2f} 样本/秒")
        
        if 'quantized_models' in summary:
            for model_name, model_data in summary['quantized_models'].items():
                print(f"\n{model_name}:")
                print(f"  大小: {model_data['size_mb']:.2f} MB (减少 {model_data['size_reduction_percent']:.1f}%)")
                print(f"  准确率: {model_data['accuracy']:.4f} (损失 {model_data['accuracy_loss_percent']:.2f}%)")
                print(f"  速度: {model_data['inference_speed']:.2f} 样本/秒 (提升 {model_data['speed_improvement_percent']:.1f}%)")
    
    print(f"\n详细结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()