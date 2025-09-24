#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT文本分类FastAPI服务
提供RESTful API接口进行文本分类预测
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import logging
import uvicorn
from datetime import datetime
import json
import subprocess
import sys
from pathlib import Path

# 导入训练管理功能
from training_api import (
    get_system_status, get_model_files, start_training, stop_training,
    get_training_status, delete_model, evaluate_model, get_data_info
)

# 导入模型优化功能
from optimization_api import (
    start_model_optimization, get_optimization_status, get_optimized_models,
    test_optimized_model, predict_with_optimized_model, delete_optimized_model,
    predict_with_original_model
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="BERT文本分类API",
    description="基于BERT的中文文本分类服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# 全局变量
model = None
tokenizer = None
class_names = []
device = None

# Pydantic模型定义
class TextInput(BaseModel):
    text: str = Field(..., description="要分类的文本", example="A股市场今日大幅上涨")
    
class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="要分类的文本列表", 
                            example=["A股市场今日大幅上涨", "教育部发布新政策"])
    
class PredictionResponse(BaseModel):
    text: str = Field(..., description="输入文本")
    predicted_class: str = Field(..., description="预测类别")
    predicted_class_id: int = Field(..., description="预测类别ID")
    confidence: float = Field(..., description="置信度")
    probabilities: Optional[Dict[str, float]] = Field(None, description="所有类别的概率分布")
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="批量预测结果")
    total_count: int = Field(..., description="预测文本总数")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="模型名称")
    num_classes: int = Field(..., description="分类类别数")
    class_names: List[str] = Field(..., description="类别名称列表")
    device: str = Field(..., description="运行设备")
    model_loaded: bool = Field(..., description="模型是否已加载")

class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="时间戳")
    model_loaded: bool = Field(..., description="模型是否已加载")

# 新增：训练和模型管理相关的数据模型
class TrainingConfig(BaseModel):
    batch_size: int = Field(16, description="批次大小")
    num_epochs: int = Field(3, description="训练轮数")
    learning_rate: float = Field(2e-5, description="学习率")
    max_length: int = Field(512, description="最大序列长度")
    
class TrainingStatus(BaseModel):
    is_training: bool = Field(..., description="是否正在训练")
    current_epoch: int = Field(0, description="当前轮数")
    total_epochs: int = Field(0, description="总轮数")
    current_loss: float = Field(0.0, description="当前损失")
    best_accuracy: float = Field(0.0, description="最佳准确率")
    progress: float = Field(0.0, description="训练进度百分比")

class ModelFileInfo(BaseModel):
    name: str = Field(..., description="模型名称")
    path: str = Field(..., description="模型路径")
    size_mb: float = Field(..., description="模型大小(MB)")
    created_time: str = Field(..., description="创建时间")
    accuracy: Optional[float] = Field(None, description="模型准确率")
    
class SystemStatus(BaseModel):
    cpu_usage: float = Field(..., description="CPU使用率")
    memory_usage: float = Field(..., description="内存使用率")
    gpu_available: bool = Field(..., description="GPU是否可用")
    gpu_memory_used: Optional[float] = Field(None, description="GPU内存使用(MB)")
    disk_usage: float = Field(..., description="磁盘使用率")

def load_class_names(data_dir: str = "data") -> List[str]:
    """加载类别名称"""
    class_file = os.path.join(data_dir, 'class.txt')
    try:
        with open(class_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"成功加载 {len(classes)} 个类别")
        return classes
    except Exception as e:
        logger.error(f"加载类别文件失败: {e}")
        raise

def load_model_and_tokenizer():
    """加载模型和tokenizer"""
    global model, tokenizer, class_names, device
    
    try:
        # 设备配置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 配置路径
        data_dir = "data"
        pretrain_dir = "pretrain/bert-base-chinese"
        
        # 加载类别名称
        class_names = load_class_names(data_dir)
        
        # 加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(pretrain_dir)
        logger.info("tokenizer加载成功")
        
        # 查找模型文件
        model_paths = [
            "checkpoints/best_model/pytorch_model.bin",
            "checkpoints/final_model/pytorch_model.bin",
            "checkpoints/checkpoint_epoch_1/pytorch_model.bin"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            logger.info(f"找到训练好的模型: {model_path}")
            # 加载模型架构
            model = BertForSequenceClassification.from_pretrained(
                pretrain_dir,
                num_labels=len(class_names)
            )
            # 加载权重
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("训练好的模型加载成功")
        else:
            logger.warning("未找到训练好的模型，使用预训练BERT模型（未微调）")
            model = BertForSequenceClassification.from_pretrained(
                pretrain_dir,
                num_labels=len(class_names)
            )
        
        model.to(device)
        model.eval()
        logger.info("模型初始化完成")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def predict_text(text: str, return_probabilities: bool = False) -> Dict[str, Any]:
    """预测单个文本"""
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 文本预处理
        text = str(text).strip()
        if not text:
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        
        # tokenization
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # 移到设备
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # 处理结果
        probs = probabilities.cpu().numpy()[0]
        predicted_class_id = int(probs.argmax())
        predicted_class_name = class_names[predicted_class_id]
        confidence = float(probs[predicted_class_id])
        
        result = {
            'text': text,
            'predicted_class': predicted_class_name,
            'predicted_class_id': predicted_class_id,
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_names[i]: float(prob) for i, prob in enumerate(probs)
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    logger.info("正在启动BERT文本分类服务...")
    try:
        load_model_and_tokenizer()
        logger.info("服务启动成功！")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.get("/", summary="前端页面")
async def frontend():
    """返回前端页面"""
    return FileResponse("frontend/index.html")

@app.get("/admin.html", summary="管理控制台")
async def admin_console():
    """返回管理控制台页面"""
    return FileResponse("frontend/admin.html")

@app.get("/api", summary="API根路径")
async def api_root():
    """API根路径，返回API信息"""
    return {
        "message": "BERT文本分类API服务",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "frontend_url": "/"
    }

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )

@app.get("/info", response_model=ModelInfo, summary="模型信息")
async def get_model_info():
    """获取模型信息"""
    return ModelInfo(
        model_name="BERT-base-chinese",
        num_classes=len(class_names),
        class_names=class_names,
        device=str(device),
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse, summary="单文本预测")
async def predict_single_text(input_data: TextInput):
    """
    对单个文本进行分类预测
    
    - **text**: 要分类的文本内容
    
    返回预测的类别、置信度和所有类别的概率分布
    """
    result = predict_text(input_data.text, return_probabilities=True)
    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="批量文本预测")
async def predict_batch_texts(input_data: BatchTextInput):
    """
    对多个文本进行批量分类预测
    
    - **texts**: 要分类的文本列表
    
    返回每个文本的预测结果
    """
    if len(input_data.texts) > 100:
        raise HTTPException(status_code=400, detail="批量预测最多支持100个文本")
    
    predictions = []
    for text in input_data.texts:
        try:
            result = predict_text(text, return_probabilities=True)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            # 对于单个文本的错误，返回错误信息
            predictions.append(PredictionResponse(
                text=text,
                predicted_class="error",
                predicted_class_id=-1,
                confidence=0.0,
                probabilities=None
            ))
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions)
    )

@app.post("/predict/simple", summary="简单预测接口")
async def predict_simple(input_data: TextInput):
    """
    简单的预测接口，只返回预测类别和置信度
    
    - **text**: 要分类的文本内容
    
    返回简化的预测结果
    """
    result = predict_text(input_data.text, return_probabilities=False)
    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"]
    }

@app.get("/classes", summary="获取所有分类类别")
async def get_classes():
    """获取所有支持的分类类别"""
    return {
        "classes": [{"id": i, "name": name} for i, name in enumerate(class_names)],
        "total_count": len(class_names)
    }

# ====================
# 训练和模型管理API
# ====================

@app.get("/api/system/status", summary="获取系统状态")
async def get_system_status_api():
    """获取系统资源使用情况"""
    return get_system_status()

@app.get("/api/models", summary="获取模型列表")
async def get_models_api():
    """获取所有已训练的模型列表"""
    return {"models": get_model_files()}

@app.delete("/api/models/{model_name}", summary="删除模型")
async def delete_model_api(model_name: str):
    """删除指定的模型"""
    return delete_model(model_name)

@app.post("/api/models/{model_name}/evaluate", summary="评估模型")
async def evaluate_model_api(model_name: str):
    """评估指定的模型"""
    return evaluate_model(model_name)

@app.get("/api/training/status", summary="获取训练状态")
async def get_training_status_api():
    """获取当前训练状态"""
    return get_training_status()

@app.post("/api/training/start", summary="开始训练")
async def start_training_api(config: dict):
    """开始模型训练"""
    return start_training(config)

@app.post("/api/training/stop", summary="停止训练")
async def stop_training_api():
    """停止当前训练"""
    return stop_training()

@app.get("/api/data/info", summary="获取数据信息")
async def get_data_info_api():
    """获取训练数据信息"""
    return get_data_info()

# ====================
# 模型优化API
# ====================

@app.get("/api/optimization/status", summary="获取优化状态")
async def get_optimization_status_api():
    """获取当前优化状态"""
    return get_optimization_status()

@app.post("/api/optimization/start", summary="开始模型优化")
async def start_optimization_api(config: dict):
    """开始模型优化"""
    return start_model_optimization(config)

@app.get("/api/optimized-models", summary="获取优化模型列表")
async def get_optimized_models_api():
    """获取所有优化后的模型列表"""
    return {"models": get_optimized_models()}

@app.post("/api/optimized-models/{model_name}/test", summary="测试优化模型")
async def test_optimized_model_api(model_name: str, input_data: dict):
    """测试优化后的模型"""
    text = input_data.get("text", "")
    return test_optimized_model(model_name, text)

@app.post("/api/optimized-models/{model_name}/predict", summary="使用优化模型预测")
async def predict_with_optimized_model_api(model_name: str, input_data: dict):
    """使用优化后的模型进行预测"""
    text = input_data.get("text", "")
    return predict_with_optimized_model(model_name, text)

@app.post("/api/models/{model_name}/predict", summary="使用原始模型预测")
async def predict_with_original_model_api(model_name: str, input_data: dict):
    """使用原始模型进行预测"""
    text = input_data.get("text", "")
    return predict_with_original_model(model_name, text)

@app.delete("/api/optimized-models/{model_name}", summary="删除优化模型")
async def delete_optimized_model_api(model_name: str):
    """删除优化后的模型"""
    return delete_optimized_model(model_name)

if __name__ == "__main__":
    # 运行服务
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )