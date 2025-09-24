#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型优化脚本
对训练好的模型进行NF4量化和剪枝优化
"""

import os
import argparse
import logging
import torch
from model_optimization import ModelOptimizer
from data_loader import DataLoader_Manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型优化脚本')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model', 
                       help='训练好的模型路径')
    parser.add_argument('--config_path', type=str, default='config.json', 
                       help='配置文件路径')
    parser.add_argument('--save_dir', type=str, default='optimized_models', 
                       help='优化模型保存目录')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='数据目录（用于性能测试）')
    parser.add_argument('--pretrain_dir', type=str, default='pretrain/bert-base-chinese', 
                       help='预训练模型目录')
    parser.add_argument('--optimization_type', type=str, 
                       choices=['quantization', 'pruning', 'structured_pruning', 'combined', 'all'],
                       default='all', help='优化类型')
    parser.add_argument('--sparsity_ratio', type=float, default=0.2, 
                       help='剪枝稀疏度比例')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='测试批次大小')
    parser.add_argument('--max_length', type=int, default=512, 
                       help='最大序列长度')
    parser.add_argument('--evaluate', action='store_true', 
                       help='是否评估优化后的模型性能')
    parser.add_argument('--no_cuda', action='store_true', 
                       help='不使用GPU')
    
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return
    
    try:
        # 创建模型优化器
        logger.info("正在加载模型...")
        optimizer = ModelOptimizer(args.model_path, args.config_path)
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 根据优化类型执行相应优化
        if args.optimization_type == 'quantization' or args.optimization_type == 'all':
            logger.info("=" * 60)
            logger.info("执行NF4量化优化")
            logger.info("=" * 60)
            optimizer.apply_nf4_quantization(args.save_dir)
        
        if args.optimization_type == 'pruning' or args.optimization_type == 'all':
            logger.info("=" * 60)
            logger.info(f"执行幅度剪枝优化 (稀疏度: {args.sparsity_ratio})")
            logger.info("=" * 60)
            optimizer.apply_magnitude_pruning(args.sparsity_ratio, args.save_dir)
        
        if args.optimization_type == 'structured_pruning' or args.optimization_type == 'all':
            logger.info("=" * 60)
            logger.info("执行结构化剪枝优化 (2:4)")
            logger.info("=" * 60)
            optimizer.apply_structured_pruning(2, 4, args.save_dir)
        
        if args.optimization_type == 'combined' or args.optimization_type == 'all':
            logger.info("=" * 60)
            logger.info(f"执行组合优化 (剪枝{args.sparsity_ratio} + NF4量化)")
            logger.info("=" * 60)
            optimizer.apply_combined_optimization(args.sparsity_ratio, args.save_dir)
        
        # 性能评估
        if args.evaluate:
            logger.info("=" * 60)
            logger.info("开始性能评估")
            logger.info("=" * 60)
            
            # 创建测试数据加载器
            data_manager = DataLoader_Manager(args.data_dir, args.pretrain_dir, args.max_length)
            _, _, test_dataset = data_manager.create_datasets()
            _, _, test_loader = data_manager.create_dataloaders(
                None, None, test_dataset, batch_size=args.batch_size
            )
            
            # 比较模型性能
            comparison_results = optimizer.compare_models(test_loader, device)
            
            # 保存比较结果
            import json
            results_path = os.path.join(args.save_dir, 'performance_comparison.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            logger.info(f"性能比较结果已保存到: {results_path}")
        
        logger.info("=" * 60)
        logger.info("模型优化完成!")
        logger.info(f"优化后的模型保存在: {args.save_dir}")
        logger.info("=" * 60)
        
        # 输出优化建议
        print_optimization_recommendations(args)
        
    except Exception as e:
        logger.error(f"模型优化失败: {e}")
        raise

def print_optimization_recommendations(args):
    """打印优化建议"""
    logger.info("优化建议:")
    logger.info("1. NF4量化:")
    logger.info("   - 显著减少模型存储空间 (约75%压缩)")
    logger.info("   - 推理速度提升 (特别是在支持量化的硬件上)")
    logger.info("   - 精度损失通常很小 (<1%)")
    
    logger.info("2. 幅度剪枝:")
    logger.info(f"   - 当前稀疏度: {args.sparsity_ratio}")
    logger.info("   - 减少计算量和存储空间")
    logger.info("   - 可能需要sparse推理框架获得实际加速")
    
    logger.info("3. 结构化剪枝:")
    logger.info("   - 2:4结构化模式")
    logger.info("   - 在支持稀疏计算的硬件上有更好加速效果")
    logger.info("   - 比非结构化剪枝更容易硬件优化")
    
    logger.info("4. 组合优化:")
    logger.info("   - 结合剪枝和量化的优势")
    logger.info("   - 最大化压缩比和推理速度")
    logger.info("   - 需要仔细平衡精度损失")
    
    logger.info("\n部署建议:")
    logger.info("- 生产环境推荐使用组合优化模型")
    logger.info("- 如果对精度要求极高，选择单独的NF4量化")
    logger.info("- 如果硬件支持稀疏计算，考虑结构化剪枝")

if __name__ == "__main__":
    main()