#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型优化模块
实现NF4量化和剪枝优化
"""

import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
import json
import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import BitsAndBytesConfig
from model import ModelManager, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """模型优化器，支持量化和剪枝"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        初始化模型优化器
        
        Args:
            model_path: 训练好的模型路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.config_path = config_path
        self.original_model = None
        self.optimized_model = None
        self.optimization_config = {
            'quantization': {
                'enabled': False,
                'method': 'nf4',  # nf4, int8
                'compute_dtype': torch.float16,
                'quant_type': "nf4",
                'use_double_quant': True,
                'load_in_4bit': True
            },
            'pruning': {
                'enabled': False,
                'method': 'magnitude',  # magnitude, structured
                'sparsity_ratio': 0.2,
                'structured_n': 2,
                'structured_m': 4,
                'layers_to_prune': ['bert.encoder.layer', 'classifier']
            }
        }
        
        self.load_model()
    
    def load_model(self):
        """加载原始模型"""
        try:
            # 加载配置
            if self.config_path and os.path.exists(self.config_path):
                config = ModelConfig(self.config_path)
            else:
                config = ModelConfig()
            
            # 创建模型管理器并加载模型
            model_manager = ModelManager(config)
            self.original_model = model_manager.load_model(self.model_path)
            
            logger.info(f"成功加载原始模型: {self.model_path}")
            
            # 统计原始模型大小
            self._log_model_stats(self.original_model, "原始模型")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _log_model_stats(self, model, model_name):
        """记录模型统计信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小（MB）
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = (param_size + buffer_size) / 1024 / 1024
        
        logger.info(f"{model_name}统计信息:")
        logger.info(f"  总参数数: {total_params:,}")
        logger.info(f"  可训练参数数: {trainable_params:,}")
        logger.info(f"  模型大小: {model_size:.2f} MB")
    
    def apply_nf4_quantization(self, save_path: Optional[str] = None):
        """
        应用NF4量化
        
        Args:
            save_path: 量化模型保存路径
        """
        logger.info("开始应用NF4量化...")
        
        try:
            # 设置BitsAndBytesConfig进行NF4量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # 创建量化模型
            self.optimized_model = copy.deepcopy(self.original_model)
            
            # 应用量化到BERT层
            self._quantize_bert_layers()
            
            # 记录量化后的统计信息
            self._log_model_stats(self.optimized_model, "NF4量化后模型")
            
            # 保存量化模型
            if save_path:
                self.save_optimized_model(save_path, "nf4_quantized")
            
            logger.info("NF4量化完成!")
            
        except Exception as e:
            logger.error(f"NF4量化失败: {e}")
            raise
    
    def _quantize_bert_layers(self):
        """量化BERT层"""
        # 这里实现简化的量化逻辑
        # 实际的NF4量化需要使用bitsandbytes库
        for name, module in self.optimized_model.named_modules():
            if isinstance(module, nn.Linear) and 'bert' in name:
                # 简化的权重量化示例
                weight = module.weight.data
                # 将权重量化到4位
                weight_quantized = self._quantize_weights_nf4(weight)
                module.weight.data = weight_quantized
    
    def _quantize_weights_nf4(self, weights):
        """
        简化的NF4权重量化
        实际应用中应使用bitsandbytes库
        """
        # 计算量化参数
        min_val = weights.min()
        max_val = weights.max()
        
        # 量化到4位 (16个级别)
        scale = (max_val - min_val) / 15.0
        zero_point = min_val
        
        # 量化
        quantized = torch.round((weights - zero_point) / scale)
        quantized = torch.clamp(quantized, 0, 15)
        
        # 反量化
        dequantized = quantized * scale + zero_point
        
        return dequantized.to(weights.dtype)
    
    def apply_magnitude_pruning(self, sparsity_ratio: float = 0.2, save_path: Optional[str] = None):
        """
        应用幅度剪枝
        
        Args:
            sparsity_ratio: 剪枝比例
            save_path: 剪枝模型保存路径
        """
        logger.info(f"开始应用幅度剪枝 (稀疏度: {sparsity_ratio})...")
        
        try:
            if self.optimized_model is None:
                self.optimized_model = copy.deepcopy(self.original_model)
            
            # 收集要剪枝的模块
            modules_to_prune = []
            
            for name, module in self.optimized_model.named_modules():
                if isinstance(module, nn.Linear):
                    # 检查是否在要剪枝的层中
                    should_prune = any(layer_pattern in name 
                                     for layer_pattern in self.optimization_config['pruning']['layers_to_prune'])
                    if should_prune:
                        modules_to_prune.append((module, 'weight'))
            
            logger.info(f"将对 {len(modules_to_prune)} 个线性层进行剪枝")
            
            # 应用全局非结构化剪枝
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity_ratio,
            )
            
            # 移除剪枝掩码，使剪枝永久化
            for module, param_name in modules_to_prune:
                prune.remove(module, param_name)
            
            # 记录剪枝后的统计信息
            self._log_pruning_stats()
            self._log_model_stats(self.optimized_model, f"剪枝后模型 (稀疏度: {sparsity_ratio})")
            
            # 保存剪枝模型
            if save_path:
                self.save_optimized_model(save_path, f"pruned_{sparsity_ratio}")
            
            logger.info("幅度剪枝完成!")
            
        except Exception as e:
            logger.error(f"剪枝失败: {e}")
            raise
    
    def apply_structured_pruning(self, n: int = 2, m: int = 4, save_path: Optional[str] = None):
        """
        应用结构化剪枝 (N:M模式)
        
        Args:
            n: 在每m个元素中保留n个
            m: 块大小
            save_path: 剪枝模型保存路径
        """
        logger.info(f"开始应用结构化剪枝 ({n}:{m} 模式)...")
        
        try:
            if self.optimized_model is None:
                self.optimized_model = copy.deepcopy(self.original_model)
            
            # 收集要剪枝的模块
            modules_to_prune = []
            
            for name, module in self.optimized_model.named_modules():
                if isinstance(module, nn.Linear):
                    should_prune = any(layer_pattern in name 
                                     for layer_pattern in self.optimization_config['pruning']['layers_to_prune'])
                    if should_prune:
                        modules_to_prune.append((module, 'weight'))
            
            logger.info(f"将对 {len(modules_to_prune)} 个线性层进行结构化剪枝")
            
            # 应用结构化剪枝
            for module, param_name in modules_to_prune:
                # 应用RandomStructured剪枝作为N:M剪枝的近似
                prune.random_structured(
                    module, 
                    name=param_name,
                    amount=1 - (n / m),  # 剪枝比例
                    dim=0
                )
                # 移除剪枝掩码
                prune.remove(module, param_name)
            
            # 记录剪枝后的统计信息
            self._log_model_stats(self.optimized_model, f"结构化剪枝后模型 ({n}:{m})")
            
            # 保存剪枝模型
            if save_path:
                self.save_optimized_model(save_path, f"structured_pruned_{n}_{m}")
            
            logger.info("结构化剪枝完成!")
            
        except Exception as e:
            logger.error(f"结构化剪枝失败: {e}")
            raise
    
    def _log_pruning_stats(self):
        """记录剪枝统计信息"""
        total_params = 0
        zero_params = 0
        
        for module in self.optimized_model.modules():
            if isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"剪枝统计:")
        logger.info(f"  总参数数: {total_params:,}")
        logger.info(f"  零参数数: {zero_params:,}")
        logger.info(f"  实际稀疏度: {sparsity:.4f}")
    
    def apply_combined_optimization(self, sparsity_ratio: float = 0.2, save_path: Optional[str] = None):
        """
        应用组合优化：先剪枝再量化
        
        Args:
            sparsity_ratio: 剪枝比例
            save_path: 优化模型保存路径
        """
        logger.info("开始应用组合优化 (剪枝 + NF4量化)...")
        
        # 先应用剪枝
        self.apply_magnitude_pruning(sparsity_ratio)
        
        # 再应用量化
        self._quantize_bert_layers()
        
        # 记录最终统计信息
        self._log_model_stats(self.optimized_model, f"组合优化后模型 (剪枝{sparsity_ratio} + NF4)")
        
        # 保存组合优化模型
        if save_path:
            self.save_optimized_model(save_path, f"combined_pruned_{sparsity_ratio}_nf4")
        
        logger.info("组合优化完成!")
    
    def evaluate_model_performance(self, model, test_loader, device):
        """
        评估模型性能
        
        Args:
            model: 待评估模型
            test_loader: 测试数据加载器
            device: 设备
            
        Returns:
            评估结果字典
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                logits = outputs['logits']
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def compare_models(self, test_loader, device):
        """
        比较原始模型和优化模型的性能
        
        Args:
            test_loader: 测试数据加载器
            device: 设备
        """
        if self.optimized_model is None:
            logger.warning("优化模型不存在，无法进行比较")
            return
        
        logger.info("开始比较模型性能...")
        
        # 评估原始模型
        original_results = self.evaluate_model_performance(
            self.original_model.to(device), test_loader, device
        )
        
        # 评估优化模型
        optimized_results = self.evaluate_model_performance(
            self.optimized_model.to(device), test_loader, device
        )
        
        # 打印比较结果
        logger.info("模型性能比较:")
        logger.info(f"原始模型    - 准确率: {original_results['accuracy']:.4f}, 损失: {original_results['loss']:.4f}")
        logger.info(f"优化模型    - 准确率: {optimized_results['accuracy']:.4f}, 损失: {optimized_results['loss']:.4f}")
        
        # 计算性能差异
        acc_diff = optimized_results['accuracy'] - original_results['accuracy']
        loss_diff = optimized_results['loss'] - original_results['loss']
        
        logger.info(f"性能差异    - 准确率差异: {acc_diff:+.4f}, 损失差异: {loss_diff:+.4f}")
        
        return {
            'original': original_results,
            'optimized': optimized_results,
            'difference': {
                'accuracy': acc_diff,
                'loss': loss_diff
            }
        }
    
    def _make_config_serializable(self, config):
        """
        将配置中的torch类型转换为可序列化的格式
        """
        import torch
        
        def convert_value(value):
            if isinstance(value, torch.dtype):
                return str(value)  # 如 'torch.float16'
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            else:
                return value
        
        return convert_value(config)
    
    def save_optimized_model(self, save_dir: str, optimization_type: str):
        """
        保存优化后的模型
        
        Args:
            save_dir: 保存目录
            optimization_type: 优化类型
        """
        if self.optimized_model is None:
            logger.warning("没有优化模型可保存")
            return
        
        full_save_path = os.path.join(save_dir, optimization_type)
        os.makedirs(full_save_path, exist_ok=True)
        
        # 保存模型状态字典
        model_path = os.path.join(full_save_path, 'pytorch_model.bin')
        torch.save(self.optimized_model.state_dict(), model_path)
        
        # 保存优化配置 (处理不可序列化的torch类型)
        config_path = os.path.join(full_save_path, 'optimization_config.json')
        serializable_config = self._make_config_serializable(self.optimization_config)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化模型已保存到: {full_save_path}")
    
    def load_optimized_model(self, model_dir: str):
        """
        加载优化后的模型
        
        Args:
            model_dir: 模型目录
        """
        model_path = os.path.join(model_dir, 'pytorch_model.bin')
        config_path = os.path.join(model_dir, 'optimization_config.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载优化配置
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.optimization_config = json.load(f)
        
        # 创建模型实例并加载权重
        self.optimized_model = copy.deepcopy(self.original_model)
        state_dict = torch.load(model_path, map_location='cpu')
        self.optimized_model.load_state_dict(state_dict)
        
        logger.info(f"成功加载优化模型: {model_dir}")

def main():
    """主函数，演示模型优化"""
    # 设置路径
    model_path = "checkpoints/best_model"  # 训练好的模型路径
    config_path = "config.json"
    save_dir = "optimized_models"
    
    try:
        # 创建模型优化器
        optimizer = ModelOptimizer(model_path, config_path)
        
        # 应用不同的优化策略
        logger.info("=" * 50)
        logger.info("1. 应用NF4量化")
        optimizer.apply_nf4_quantization(save_dir)
        
        logger.info("=" * 50)
        logger.info("2. 应用幅度剪枝 (稀疏度: 0.2)")
        optimizer.apply_magnitude_pruning(0.2, save_dir)
        
        logger.info("=" * 50)
        logger.info("3. 应用结构化剪枝 (2:4)")
        optimizer.apply_structured_pruning(2, 4, save_dir)
        
        logger.info("=" * 50)
        logger.info("4. 应用组合优化")
        optimizer.apply_combined_optimization(0.15, save_dir)
        
        logger.info("模型优化完成!")
        
    except Exception as e:
        logger.error(f"模型优化失败: {e}")

if __name__ == "__main__":
    main()