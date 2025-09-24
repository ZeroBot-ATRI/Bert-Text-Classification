#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT文本分类项目统一管理脚本
提供训练、预测、评估、API服务等功能的统一入口
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectManager:
    """项目管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_files = {
            'data': ['class.txt', 'train.txt', 'test.txt'],
            'pretrain/bert-base-chinese': ['config.json', 'vocab.txt'],
            'scripts': ['train.py', 'predict.py', 'evaluate.py', 'app.py']
        }
    
    def check_environment(self):
        """检查环境和依赖"""
        logger.info("检查项目环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 7:
            logger.error(f"Python版本过低: {python_version.major}.{python_version.minor}, 需要Python 3.7+")
            return False
        
        # 检查必要文件
        missing_files = []
        for dir_name, files in self.required_files.items():
            dir_path = self.project_root / dir_name
            if dir_name == 'scripts':
                # 脚本文件在根目录
                dir_path = self.project_root
            
            for file_name in files:
                file_path = dir_path / file_name
                if not file_path.exists():
                    missing_files.append(str(file_path))
        
        if missing_files:
            logger.error(f"缺少必要文件: {missing_files}")
            return False
        
        # 检查Python包
        required_packages = [
            'torch', 'transformers', 'sklearn', 'pandas', 
            'numpy', 'fastapi', 'uvicorn', 'requests'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少Python包: {missing_packages}")
            logger.info(f"请运行: pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("✓ 环境检查通过")
        return True
    
    def analyze_data(self):
        """分析数据分布"""
        logger.info("分析数据分布...")
        try:
            cmd = [sys.executable, "data_loader.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✓ 数据分析完成")
                print(result.stdout)
            else:
                logger.error(f"数据分析失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"数据分析出错: {e}")
            return False
        
        return True
    
    def train_model(self, **kwargs):
        """训练模型"""
        logger.info("开始训练模型...")
        
        cmd = [sys.executable, "train.py"]
        
        # 添加命令行参数
        for key, value in kwargs.items():
            if key.startswith('no_'):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        try:
            logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✓ 模型训练完成")
                return True
            else:
                logger.error("模型训练失败")
                return False
        except Exception as e:
            logger.error(f"训练出错: {e}")
            return False
    
    def evaluate_model(self, model_dir="checkpoints/best_model", **kwargs):
        """评估模型"""
        logger.info("开始评估模型...")
        
        cmd = [sys.executable, "evaluate.py", "--model_dir", model_dir]
        
        # 添加命令行参数
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✓ 模型评估完成")
                return True
            else:
                logger.error("模型评估失败")
                return False
        except Exception as e:
            logger.error(f"评估出错: {e}")
            return False
    
    def start_api_server(self, **kwargs):
        """启动API服务"""
        logger.info("启动API服务...")
        
        cmd = [sys.executable, "start_api.py"]
        
        # 添加命令行参数
        for key, value in kwargs.items():
            if key == 'reload' and value:
                cmd.append(f"--{key}")
            elif key != 'reload':
                cmd.extend([f"--{key}", str(value)])
        
        try:
            subprocess.run(cmd, cwd=self.project_root)
        except KeyboardInterrupt:
            logger.info("API服务已停止")
        except Exception as e:
            logger.error(f"API服务出错: {e}")
    
    def test_api(self, url="http://localhost:8000"):
        """测试API服务"""
        logger.info("测试API服务...")
        
        cmd = [sys.executable, "test_api.py", "--url", url]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✓ API测试完成")
                return True
            else:
                logger.error("API测试失败")
                return False
        except Exception as e:
            logger.error(f"API测试出错: {e}")
            return False
    
    def interactive_predict(self, model_dir="checkpoints/best_model"):
        """交互式预测"""
        logger.info("启动交互式预测...")
        
        cmd = [sys.executable, "predict.py", 
               "--model_dir", model_dir, 
               "--mode", "interactive"]
        
        try:
            subprocess.run(cmd, cwd=self.project_root)
        except KeyboardInterrupt:
            logger.info("交互式预测已结束")
        except Exception as e:
            logger.error(f"交互式预测出错: {e}")
    
    def demo(self):
        """运行演示"""
        logger.info("运行演示...")
        
        cmd = [sys.executable, "demo.py"]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✓ 演示完成")
                return True
            else:
                logger.error("演示失败")
                return False
        except Exception as e:
            logger.error(f"演示出错: {e}")
            return False
    
    def show_status(self):
        """显示项目状态"""
        logger.info("=" * 60)
        logger.info("BERT文本分类项目状态")
        logger.info("=" * 60)
        
        # 检查模型文件
        model_paths = [
            "checkpoints/best_model/pytorch_model.bin",
            "checkpoints/final_model/pytorch_model.bin"
        ]
        
        trained_models = []
        for model_path in model_paths:
            if (self.project_root / model_path).exists():
                trained_models.append(model_path)
        
        if trained_models:
            logger.info(f"✓ 已训练模型: {trained_models}")
        else:
            logger.info("✗ 未找到已训练的模型")
        
        # 检查数据文件大小
        data_files = ['train.txt', 'test.txt']
        for file_name in data_files:
            file_path = self.project_root / 'data' / file_name
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"✓ {file_name}: {file_size / 1024 / 1024:.2f} MB")
            else:
                logger.info(f"✗ {file_name}: 不存在")
        
        logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='BERT文本分类项目管理器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 检查环境
    subparsers.add_parser('check', help='检查环境和依赖')
    
    # 显示状态
    subparsers.add_parser('status', help='显示项目状态')
    
    # 分析数据
    subparsers.add_parser('analyze', help='分析数据分布')
    
    # 训练模型
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    train_parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    train_parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    train_parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    train_parser.add_argument('--no_cuda', action='store_true', help='不使用GPU')
    
    # 评估模型
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model_dir', type=str, default='checkpoints/best_model', help='模型目录')
    eval_parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    
    # 启动API服务
    api_parser = subparsers.add_parser('api', help='启动API服务')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    api_parser.add_argument('--port', type=int, default=8000, help='服务端口')
    api_parser.add_argument('--reload', action='store_true', help='开启热重载')
    
    # 测试API
    test_parser = subparsers.add_parser('test', help='测试API服务')
    test_parser.add_argument('--url', type=str, default='http://localhost:8000', help='API地址')
    
    # 交互式预测
    predict_parser = subparsers.add_parser('predict', help='交互式预测')
    predict_parser.add_argument('--model_dir', type=str, default='checkpoints/best_model', help='模型目录')
    
    # 演示
    subparsers.add_parser('demo', help='运行演示')
    
    # 完整流程
    full_parser = subparsers.add_parser('full', help='完整流程：训练->评估->演示')
    full_parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    full_parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    full_parser.add_argument('--no_cuda', action='store_true', help='不使用GPU')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ProjectManager()
    
    if args.command == 'check':
        manager.check_environment()
    
    elif args.command == 'status':
        manager.show_status()
    
    elif args.command == 'analyze':
        if manager.check_environment():
            manager.analyze_data()
    
    elif args.command == 'train':
        if manager.check_environment():
            train_kwargs = {
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'max_length': args.max_length,
                'no_cuda': args.no_cuda
            }
            manager.train_model(**train_kwargs)
    
    elif args.command == 'evaluate':
        if manager.check_environment():
            eval_kwargs = {
                'output_dir': args.output_dir
            }
            manager.evaluate_model(args.model_dir, **eval_kwargs)
    
    elif args.command == 'api':
        if manager.check_environment():
            api_kwargs = {
                'host': args.host,
                'port': args.port,
                'reload': args.reload
            }
            manager.start_api_server(**api_kwargs)
    
    elif args.command == 'test':
        manager.test_api(args.url)
    
    elif args.command == 'predict':
        if manager.check_environment():
            manager.interactive_predict(args.model_dir)
    
    elif args.command == 'demo':
        if manager.check_environment():
            manager.demo()
    
    elif args.command == 'full':
        if not manager.check_environment():
            return
        
        logger.info("开始完整流程...")
        
        # 1. 分析数据
        if not manager.analyze_data():
            return
        
        # 2. 训练模型
        train_kwargs = {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'no_cuda': args.no_cuda
        }
        if not manager.train_model(**train_kwargs):
            return
        
        # 3. 评估模型
        if not manager.evaluate_model():
            return
        
        # 4. 演示
        manager.demo()
        
        logger.info("✓ 完整流程执行完成！")

if __name__ == "__main__":
    main()