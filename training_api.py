#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练和模型管理API扩展
"""

import os
import json
import subprocess
import sys
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

# 全局训练状态
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'best_accuracy': 0.0,
    'progress': 0.0,
    'start_time': None,
    'log_messages': []
}

def get_system_status() -> Dict[str, Any]:
    """获取系统状态"""
    try:
        # CPU和内存使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # 磁盘使用率
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        
        # GPU信息
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = None
        if gpu_available:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            except:
                gpu_memory_used = 0
        
        return {
            'cpu_usage': round(cpu_percent, 1),
            'memory_usage': round(memory.percent, 1),
            'gpu_available': gpu_available,
            'gpu_memory_used': gpu_memory_used,
            'disk_usage': round(disk_percent, 1)
        }
    except Exception as e:
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_available': False,
            'gpu_memory_used': None,
            'disk_usage': 0.0,
            'error': str(e)
        }

def get_model_files() -> List[Dict[str, Any]]:
    """获取模型文件列表"""
    models = []
    checkpoints_dir = Path("checkpoints")
    
    if not checkpoints_dir.exists():
        return models
    
    for model_dir in checkpoints_dir.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "pytorch_model.bin"
            if model_file.exists():
                stat = model_file.stat()
                size_mb = stat.st_size / 1024 / 1024
                created_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
                
                # 尝试读取准确率信息
                accuracy = None
                metrics_file = model_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            accuracy = metrics.get('accuracy', metrics.get('eval_accuracy'))
                    except:
                        pass
                
                models.append({
                    'name': model_dir.name,
                    'path': str(model_file),
                    'size_mb': round(size_mb, 2),
                    'created_time': created_time,
                    'accuracy': accuracy
                })
    
    return sorted(models, key=lambda x: x['created_time'], reverse=True)

def start_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """启动训练"""
    global training_status
    
    if training_status['is_training']:
        return {'success': False, 'message': '已有训练任务在进行中'}
    
    try:
        # 构建训练命令
        cmd = [
            sys.executable, "train.py",
            "--batch_size", str(config.get('batch_size', 16)),
            "--num_epochs", str(config.get('num_epochs', 3)),
            "--learning_rate", str(config.get('learning_rate', 2e-5)),
            "--max_length", str(config.get('max_length', 512))
        ]
        
        # 更新训练状态
        training_status.update({
            'is_training': True,
            'current_epoch': 0,
            'total_epochs': config.get('num_epochs', 3),
            'current_loss': 0.0,
            'best_accuracy': 0.0,
            'progress': 0.0,
            'start_time': datetime.now().isoformat(),
            'log_messages': []
        })
        
        # 启动训练进程（后台）
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return {'success': True, 'message': '训练已启动'}
        
    except Exception as e:
        training_status['is_training'] = False
        return {'success': False, 'message': f'启动训练失败: {str(e)}'}

def stop_training() -> Dict[str, Any]:
    """停止训练"""
    global training_status
    
    try:
        # 查找并终止训练进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'train.py' in ' '.join(proc.info['cmdline'] or []):
                    proc.terminate()
                    proc.wait(timeout=5)
                    break
            except:
                continue
        
        training_status.update({
            'is_training': False,
            'current_epoch': 0,
            'total_epochs': 0,
            'current_loss': 0.0,
            'progress': 0.0
        })
        
        return {'success': True, 'message': '训练已停止'}
        
    except Exception as e:
        return {'success': False, 'message': f'停止训练失败: {str(e)}'}

def get_training_status() -> Dict[str, Any]:
    """获取训练状态"""
    return training_status.copy()

def delete_model(model_name: str) -> Dict[str, Any]:
    """删除模型"""
    try:
        model_dir = Path("checkpoints") / model_name
        if not model_dir.exists():
            return {'success': False, 'message': '模型不存在'}
        
        # 删除模型目录
        import shutil
        shutil.rmtree(model_dir)
        
        return {'success': True, 'message': f'模型 {model_name} 已删除'}
        
    except Exception as e:
        return {'success': False, 'message': f'删除模型失败: {str(e)}'}

def evaluate_model(model_name: str) -> Dict[str, Any]:
    """评估模型"""
    try:
        model_dir = Path("checkpoints") / model_name
        if not model_dir.exists():
            return {'success': False, 'message': '模型不存在'}
        
        # 运行评估命令
        cmd = [sys.executable, "evaluate.py", "--model_dir", str(model_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return {'success': True, 'message': '评估完成', 'output': result.stdout}
        else:
            return {'success': False, 'message': '评估失败', 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'message': '评估超时'}
    except Exception as e:
        return {'success': False, 'message': f'评估失败: {str(e)}'}

def get_data_info() -> Dict[str, Any]:
    """获取数据信息"""
    info = {
        'train_size': 0,
        'test_size': 0,
        'classes_count': 0,
        'data_ready': False
    }
    
    try:
        # 检查训练数据
        train_file = Path("data/train.txt")
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                info['train_size'] = sum(1 for _ in f)
        
        # 检查测试数据
        test_file = Path("data/test.txt")
        if test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                info['test_size'] = sum(1 for _ in f)
        
        # 检查类别文件
        class_file = Path("data/class.txt")
        if class_file.exists():
            with open(class_file, 'r', encoding='utf-8') as f:
                info['classes_count'] = sum(1 for line in f if line.strip())
        
        info['data_ready'] = all([
            info['train_size'] > 0,
            info['test_size'] > 0,
            info['classes_count'] > 0
        ])
        
    except Exception as e:
        info['error'] = str(e)
    
    return info