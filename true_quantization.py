#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真正的NF4量化实现
解决量化后模型大小不减少的问题
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import logging
from typing import Dict, Optional, Tuple
import struct

logger = logging.getLogger(__name__)

class TrueNF4Quantizer:
    """真正的NF4量化器"""
    
    def __init__(self):
        # NF4量化级别 (16个级别对应4位)
        self.nf4_levels = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ])
    
    def quantize_tensor(self, tensor: torch.Tensor) -> Dict:
        """
        将张量量化为真正的4位NF4格式
        
        Returns:
            包含量化数据、scale、zero_point的字典
        """
        # 计算缩放因子
        absmax = tensor.abs().max()
        if isinstance(absmax, torch.Tensor):
            scale = absmax.item()
        else:
            scale = float(absmax)
        scale = scale / 1.0  # NF4的最大值是1.0
        
        if scale == 0:
            scale = 1.0
        
        # 规范化到[-1, 1]
        normalized = tensor / scale
        
        # 量化到NF4级别
        quantized_indices = self._quantize_to_nf4_indices(normalized)
        
        # 打包为4位存储
        packed_data = self._pack_4bit(quantized_indices)
        
        return {
            'data': packed_data,
            'scale': scale,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'original_size': tensor.numel() * tensor.element_size(),
            'quantized_size': len(packed_data) + 4  # 4字节scale
        }
    
    def _quantize_to_nf4_indices(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """将规范化张量量化到NF4索引"""
        # 找到最接近的NF4级别
        distances = torch.abs(normalized_tensor.unsqueeze(-1) - self.nf4_levels.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)
        return indices.to(torch.uint8)
    
    def _pack_4bit(self, indices: torch.Tensor) -> bytes:
        """将8位索引打包为4位存储"""
        flat_indices = indices.flatten()
        
        # 确保长度为偶数
        if len(flat_indices) % 2 == 1:
            flat_indices = torch.cat([flat_indices, torch.tensor([0], dtype=torch.uint8)])
        
        # 打包：每个字节存储2个4位数
        packed = []
        for i in range(0, len(flat_indices), 2):
            low = int(flat_indices[i].item()) & 0x0F
            high = (int(flat_indices[i+1].item()) & 0x0F) << 4
            packed.append(high | low)
        
        return bytes(packed)
    
    def dequantize_tensor(self, quantized_data: Dict) -> torch.Tensor:
        """反量化张量"""
        # 解包4位数据
        indices = self._unpack_4bit(quantized_data['data'], quantized_data['shape'])
        
        # 获取NF4值
        nf4_values = self.nf4_levels[indices]
        
        # 反规范化
        dequantized = nf4_values * quantized_data['scale']
        
        return dequantized.to(quantized_data['dtype'])
    
    def _unpack_4bit(self, packed_data: bytes, shape: tuple) -> torch.Tensor:
        """解包4位数据"""
        indices = []
        for byte in packed_data:
            low = byte & 0x0F
            high = (byte >> 4) & 0x0F
            indices.extend([low, high])
        
        # 截断到原始长度
        total_elements = np.prod(shape)
        indices = indices[:total_elements]
        
        return torch.tensor(indices, dtype=torch.long).reshape(shape)

class TrueQuantizedLinear(nn.Module):
    """真正量化的线性层"""
    
    def __init__(self, in_features: int, out_features: int, quantized_weight: Dict, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantized_weight = quantized_weight
        self.bias = bias
        self.quantizer = TrueNF4Quantizer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在前向传播时才反量化权重
        weight = self.quantizer.dequantize_tensor(self.quantized_weight)
        return torch.nn.functional.linear(x, weight, self.bias)
    
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        original_size = self.quantized_weight['original_size']
        quantized_size = self.quantized_weight['quantized_size']
        
        if self.bias is not None:
            bias_size = self.bias.numel() * self.bias.element_size()
            original_size += bias_size
            quantized_size += bias_size
        
        return {
            'original_size': original_size,
            'quantized_size': quantized_size,
            'compression_ratio': original_size / quantized_size
        }

class TrueModelOptimizer:
    """真正的模型优化器"""
    
    def __init__(self, model):
        self.original_model = model
        self.quantizer = TrueNF4Quantizer()
        self.quantized_layers = {}
    
    def apply_true_nf4_quantization(self) -> nn.Module:
        """应用真正的NF4量化"""
        logger.info("开始真正的NF4量化...")
        
        # 复制模型结构
        quantized_model = self._create_quantized_model_structure()
        
        # 量化权重并替换层
        total_original_size = 0
        total_quantized_size = 0
        
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Linear) and 'bert' in name:
                # 量化权重
                quantized_weight = self.quantizer.quantize_tensor(module.weight)
                
                # 创建量化层
                quantized_layer = TrueQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    quantized_weight,
                    module.bias
                )
                
                # 替换原层
                self._replace_module(quantized_model, name, quantized_layer)
                
                # 统计大小
                memory_info = quantized_layer.get_memory_usage()
                total_original_size += memory_info['original_size']
                total_quantized_size += memory_info['quantized_size']
                
                logger.info(f"量化层 {name}: {memory_info['original_size']/1024/1024:.2f}MB → {memory_info['quantized_size']/1024/1024:.2f}MB (压缩比: {memory_info['compression_ratio']:.2f}x)")
        
        compression_ratio = total_original_size / total_quantized_size
        logger.info(f"总量化效果: {total_original_size/1024/1024:.2f}MB → {total_quantized_size/1024/1024:.2f}MB (压缩比: {compression_ratio:.2f}x)")
        
        return quantized_model
    
    def _create_quantized_model_structure(self):
        """创建量化模型结构"""
        import copy
        return copy.deepcopy(self.original_model)
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """替换模型中的模块"""
        parts = module_name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """保存量化模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存量化数据
        quantized_data = {}
        total_original_size = 0
        total_quantized_size = 0
        
        for name, module in model.named_modules():
            if isinstance(module, TrueQuantizedLinear):
                quantized_data[name] = {
                    'quantized_weight': module.quantized_weight,
                    'bias': module.bias.tolist() if module.bias is not None else None,
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
                
                memory_info = module.get_memory_usage()
                total_original_size += memory_info['original_size']
                total_quantized_size += memory_info['quantized_size']
        
        # 保存为二进制格式以节省空间
        model_file = os.path.join(save_path, 'quantized_model.bin')
        torch.save(quantized_data, model_file)
        
        # 保存模型信息
        info = {
            'quantization_method': 'true_nf4',
            'total_original_size_mb': total_original_size / 1024 / 1024,
            'total_quantized_size_mb': total_quantized_size / 1024 / 1024,
            'compression_ratio': total_original_size / total_quantized_size,
            'quantized_layers': list(quantized_data.keys())
        }
        
        info_file = os.path.join(save_path, 'quantization_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"真正量化模型已保存到: {save_path}")
        logger.info(f"压缩效果: {info['total_original_size_mb']:.2f}MB → {info['total_quantized_size_mb']:.2f}MB")

def demonstrate_true_quantization():
    """演示真正的量化效果"""
    print("=" * 60)
    print("真正NF4量化演示")
    print("=" * 60)
    
    try:
        # 加载原始模型
        from model import ModelManager, ModelConfig
        
        config = ModelConfig()
        model_manager = ModelManager(config)
        original_model = model_manager.load_model("checkpoints/best_model")
        
        if original_model is None:
            print("错误：模型加载失败")
            return
        
        print("成功加载原始模型")
        
        # 计算原始模型大小
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        print(f"原始模型大小: {original_size / 1024 / 1024:.2f} MB")
        
        # 应用真正的量化
        optimizer = TrueModelOptimizer(original_model)
        quantized_model = optimizer.apply_true_nf4_quantization()
        
        # 保存量化模型
        save_path = "optimized_models/true_nf4_quantized"
        optimizer.save_quantized_model(quantized_model, save_path)
        
        print("\n✅ 真正的NF4量化完成！")
        print("现在模型大小已经真正减少了。")
        
    except Exception as e:
        logger.error(f"演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demonstrate_true_quantization()