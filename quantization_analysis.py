#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化效果分析脚本
分析为什么量化后模型大小没有减少的问题
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from model_optimization import ModelOptimizer
from model import ModelManager, ModelConfig

def get_model_size(model):
    """
    计算模型大小（字节）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size, param_size, buffer_size

def analyze_weight_distribution(model, name_prefix="模型"):
    """
    分析权重分布
    """
    print(f"\n{name_prefix}权重分析:")
    print("-" * 40)
    
    total_params = 0
    linear_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            total_params += weight.numel()
            linear_params += weight.numel()
            
            print(f"层: {name}")
            print(f"  形状: {weight.shape}")
            print(f"  数据类型: {weight.dtype}")
            print(f"  元素大小: {weight.element_size()} 字节")
            print(f"  参数数量: {weight.numel():,}")
            print(f"  层大小: {weight.numel() * weight.element_size() / 1024 / 1024:.2f} MB")
            print(f"  权重范围: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
            print(f"  权重标准差: {weight.std().item():.6f}")
            
            # 检查是否有零权重（剪枝效果）
            zero_count = (weight == 0).sum().item()
            sparsity = zero_count / weight.numel()
            print(f"  零权重数量: {zero_count:,}")
            print(f"  稀疏度: {sparsity:.4f}")
            print()
    
    print(f"总线性层参数: {linear_params:,}")
    print(f"总模型参数: {total_params:,}")

def simulate_true_nf4_quantization(weight):
    """
    模拟真正的NF4量化（4位存储）
    """
    # NF4量化
    min_val = weight.min()
    max_val = weight.max()
    
    # 量化到4位 (16个级别: 0-15)
    scale = (max_val - min_val) / 15.0
    zero_point = min_val
    
    # 量化
    quantized = torch.round((weight - zero_point) / scale)
    quantized = torch.clamp(quantized, 0, 15)
    
    # 转换为4位存储（使用uint8模拟，每个uint8存储2个4位数）
    quantized_uint8 = quantized.to(torch.uint8)
    
    # 计算真实4位存储大小
    original_size = weight.numel() * weight.element_size()
    quantized_size = (weight.numel() + 1) // 2  # 每个字节存储2个4位数
    scale_size = scale.element_size()
    zero_point_size = zero_point.element_size()
    
    total_quantized_size = quantized_size + scale_size + zero_point_size
    
    return {
        'original_size': original_size,
        'quantized_size': total_quantized_size,
        'compression_ratio': original_size / total_quantized_size,
        'scale': scale,
        'zero_point': zero_point,
        'quantized_data': quantized_uint8
    }

def analyze_quantization_effect():
    """
    分析量化效果
    """
    print("=" * 60)
    print("量化效果分析")
    print("=" * 60)
    
    # 加载原始模型
    try:
        config = ModelConfig()
        model_manager = ModelManager(config)
        original_model = model_manager.load_model("checkpoints/best_model")
        print(f"模型类型: {type(original_model)}")
        
        if original_model is None:
            print("错误：模型加载失败")
            return
        
        print("成功加载原始模型")
        
        # 分析原始模型
        total_size, param_size, buffer_size = get_model_size(original_model)
        print(f"\n原始模型大小分析:")
        print(f"  参数大小: {param_size / 1024 / 1024:.2f} MB")
        print(f"  缓冲区大小: {buffer_size / 1024 / 1024:.2f} MB")
        print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")
        
        analyze_weight_distribution(original_model, "原始")
        
        # 加载量化模型
        print("\n" + "=" * 60)
        print("加载当前量化模型")
        print("=" * 60)
        
        quantized_model_path = "optimized_models/nf4_quantized"
        if os.path.exists(quantized_model_path):
            optimizer = ModelOptimizer("checkpoints/best_model")
            optimizer.load_optimized_model(quantized_model_path)
            
            quantized_total_size, quantized_param_size, quantized_buffer_size = get_model_size(optimizer.optimized_model)
            print(f"\n当前量化模型大小分析:")
            print(f"  参数大小: {quantized_param_size / 1024 / 1024:.2f} MB")
            print(f"  缓冲区大小: {quantized_buffer_size / 1024 / 1024:.2f} MB")
            print(f"  总大小: {quantized_total_size / 1024 / 1024:.2f} MB")
            
            analyze_weight_distribution(optimizer.optimized_model, "当前量化")
            
            # 比较大小
            size_reduction = (total_size - quantized_total_size) / total_size * 100
            print(f"\n大小对比:")
            print(f"  原始模型: {total_size / 1024 / 1024:.2f} MB")
            print(f"  量化模型: {quantized_total_size / 1024 / 1024:.2f} MB")
            print(f"  大小减少: {size_reduction:.2f}%")
            
            if abs(size_reduction) < 1:
                print("⚠️  警告：模型大小几乎没有减少，说明量化没有生效！")
            
        # 模拟真正的NF4量化效果
        print("\n" + "=" * 60)
        print("模拟真正的NF4量化效果")
        print("=" * 60)
        
        total_original_size = 0
        total_quantized_size = 0
        
        for name, module in original_model.named_modules():
            if isinstance(module, nn.Linear) and 'bert' in name:
                weight = module.weight
                result = simulate_true_nf4_quantization(weight)
                
                total_original_size += result['original_size']
                total_quantized_size += result['quantized_size']
                
                print(f"层: {name}")
                print(f"  原始大小: {result['original_size'] / 1024 / 1024:.2f} MB")
                print(f"  量化大小: {result['quantized_size'] / 1024 / 1024:.2f} MB")
                print(f"  压缩比: {result['compression_ratio']:.2f}x")
                print()
        
        overall_compression = total_original_size / total_quantized_size
        size_reduction_true = (total_original_size - total_quantized_size) / total_original_size * 100
        
        print(f"真正NF4量化效果:")
        print(f"  被量化层原始大小: {total_original_size / 1024 / 1024:.2f} MB")
        print(f"  被量化层量化后大小: {total_quantized_size / 1024 / 1024:.2f} MB")
        print(f"  压缩比: {overall_compression:.2f}x")
        print(f"  大小减少: {size_reduction_true:.2f}%")
        
        # 估算整个模型的量化效果
        estimated_total_quantized = total_size - total_original_size + total_quantized_size
        estimated_reduction = (total_size - estimated_total_quantized) / total_size * 100
        
        print(f"\n整个模型的预期量化效果:")
        print(f"  原始模型: {total_size / 1024 / 1024:.2f} MB")
        print(f"  预期量化后: {estimated_total_quantized / 1024 / 1024:.2f} MB")
        print(f"  预期减少: {estimated_reduction:.2f}%")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()

def provide_solution():
    """
    提供解决方案
    """
    print("\n" + "=" * 60)
    print("解决方案")
    print("=" * 60)
    
    print("当前量化实现的问题:")
    print("1. 执行了伪量化：量化后立即反量化回原数据类型")
    print("2. 没有真正减少存储空间")
    print("3. 没有使用bitsandbytes库的真正NF4量化")
    
    print("\n正确的量化实现应该:")
    print("1. 使用bitsandbytes库进行真正的4位量化")
    print("2. 或者实现自定义的4位存储格式")
    print("3. 只在前向传播时进行反量化")
    print("4. 存储量化参数（scale, zero_point）")
    
    print("\n推荐解决方案:")
    print("1. 使用bitsandbytes库的load_in_4bit功能")
    print("2. 使用transformers库的4位量化支持")
    print("3. 实现真正的4位存储和加载逻辑")

if __name__ == "__main__":
    analyze_quantization_effect()
    provide_solution()