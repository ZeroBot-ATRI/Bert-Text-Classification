# BERT文本分类项目

基于BERT的中文文本分类系统，支持微调训练、预测和FastAPI部署。

## 🎆 特性亮点

- ✅ **完整的数据管理**: 支持大规模train.txt数据集加载和预处理
- ✅ **高效的BERT微调**: 基于中文BERT预训练模型的文本分类
- ✅ **FastAPI部署**: 提供RESTful API接口，支持单文本和批量预测
- ✅ **统一管理**: 一个命令行工具管理所有功能
- ✅ **详细评估**: 混淆矩阵、ROC曲线、分类报告等
- ✅ **灵活配置**: 支持多种模型架构和超参数调整

## 项目结构

```
├── data/                    # 数据目录
│   ├── class.txt           # 类别标签文件
│   ├── dev.txt             # 训练数据文件
│   └── test.txt            # 测试数据文件
├── pretrain/               # 预训练模型目录
│   └── bert-base-chinese/  # BERT中文预训练模型
├── data_loader.py          # 数据加载和预处理
├── model.py                # BERT模型定义
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── evaluate.py            # 评估脚本
├── run.py                 # 主运行脚本
├── config.json            # 配置文件
└── README.md              # 说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- transformers 4.0+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

## 安装依赖

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm tensorboard
```

## 数据格式

### 类别标签文件 (data/class.txt)
每行一个类别名称：
```
finance
realty
stocks
education
science
society
politics
sports
game
entertainment
```

### 训练/测试数据文件 (data/dev.txt, data/test.txt)
每行格式：`文本内容\t类别ID`
```
## 📁 项目结构

```
d:\py\0923/                    # 项目根目录
├── 🚀 启动脚本
│   ├── start.py              # 一键启动脚本 (推荐)
│   ├── start_api.py          # API服务启动脚本  
│   ├── start_web.py          # Web界面启动脚本
│   └── manage.py             # 完整项目管理工具
│
├── 🌐 Web界面
│   └── frontend/             # 前端文件目录
│       ├── index.html        # Web界面主文件
│       └── README.md         # 前端使用说明
│
├── 🤖 核心模块
│   ├── app.py                # FastAPI应用主文件
│   ├── model.py              # BERT模型定义
│   ├── data_loader.py        # 数据加载器
│   ├── train.py              # 模型训练脚本
│   ├── predict.py            # 预测功能模块
│   └── evaluate.py           # 模型评估工具
│
├── 🧪 测试和演示
│   ├── test_api.py           # API完整测试脚本
│   └── demo.py               # 命令行演示工具
│
├── 📊 数据和模型
│   ├── data/                 # 数据目录
│   │   ├── train.txt         # 训练数据 (188万+样本)
│   │   ├── test.txt          # 测试数据
│   │   ├── class.txt         # 类别标签
│   │   └── ...
│   ├── checkpoints/          # 模型检查点
│   │   ├── best_model/       # 最佳模型
│   │   └── final_model/      # 最终模型
│   └── pretrain/             # 预训练模型
│       └── bert-base-chinese/
│
└── 📄 配置和文档
    ├── README.md             # 项目文档
    ├── requirements.txt      # Python依赖
    └── config.json           # 模型配置
```

## 🚀 快速开始

### 🌍 **Web界面使用** (推荐)
```bash
# 一键启动Web界面
python start.py web

# 或者直接使用默认命令
python start.py
```

✨ **Web界面特性**:
- 🎨 **现代化设计**: 响应式布局，支持桌面和移动设备
- 🔍 **智能预测**: 输入中文文本，一键获得分类结果
- 📊 **可视化结果**: 动态图表展示各类别预测概率
- 💡 **示例文本**: 点击示例快速体验分类功能
- ⚙️ **实时状态**: 显示API服务和模型加载状态

🌐 **访问地址**:
- Web界面: http://127.0.0.1:8000/
- API文档: http://127.0.0.1:8000/docs

### 🔧 **命令行使用**

### 1. 检查环境和数据
```bash
python run.py --action check
```

### 2. 测试数据加载
```bash
python run.py --action test
```

### 3. 训练模型
```bash
python run.py --action train
```

### 4. 评估模型
```bash
python run.py --action evaluate
```

### 5. 预测
```bash
python run.py --action predict
```

### 6. 一键运行全流程
```bash
python run.py --action all
```

## 详细使用说明

### 训练

```bash
python train.py \
    --data_dir data \
    --pretrain_dir pretrain/bert-base-chinese \
    --save_dir checkpoints \
    --config_file config.json \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --seed 42
```

训练参数说明：
- `--data_dir`: 数据目录路径
- `--pretrain_dir`: 预训练BERT模型路径
- `--save_dir`: 模型保存目录
- `--config_file`: 配置文件路径
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--max_length`: 最大序列长度
- `--seed`: 随机种子
- `--no_cuda`: 不使用GPU训练

### 预测

#### 交互式预测
```bash
python predict.py --model_dir checkpoints/best_model --mode interactive
```

#### 单个文本预测
```bash
python predict.py \
    --model_dir checkpoints/best_model \
    --mode single \
    --text "这是一个测试文本"
```

#### 批量文本预测
```bash
python predict.py \
    --model_dir checkpoints/best_model \
    --mode batch \
    --texts "文本1" "文本2" "文本3"
```

#### 文件预测
```bash
python predict.py \
    --model_dir checkpoints/best_model \
    --mode file \
    --input_file input.csv \
    --output_file output.csv \
    --text_column text
```

### 评估

```bash
python evaluate.py \
    --model_dir checkpoints/best_model \
    --data_dir data \
    --output_dir evaluation_results \
    --dataset test \
    --batch_size 32
```

评估将生成：
- 混淆矩阵图
- 分类报告图
- ROC曲线图
- 详细预测分析
- 评估指标JSON文件
- 文本评估报告

## 配置文件

`config.json` 包含所有模型和训练配置：

```json
{
  "model_type": "bert_classifier",
  "pretrain_dir": "pretrain/bert-base-chinese",
  "num_classes": 10,
  "dropout_rate": 0.1,
  "freeze_bert": false,
  "hidden_size": 768,
  "learning_rate": 2e-5,
  "weight_decay": 0.01,
  "warmup_steps": 500,
  "max_grad_norm": 1.0,
  "num_epochs": 10,
  "patience": 5,
  "max_length": 512
}
```

### 主要配置参数说明

- `model_type`: 模型类型，支持 "bert_classifier" 和 "advanced_bert_classifier"
- `pretrain_dir`: 预训练BERT模型目录
- `num_classes`: 分类类别数量
- `dropout_rate`: Dropout率
- `freeze_bert`: 是否冻结BERT参数
- `learning_rate`: 学习率
- `weight_decay`: 权重衰减
- `warmup_steps`: 学习率预热步数
- `num_epochs`: 训练轮数
- `patience`: 早停的耐心值
- `max_length`: 最大序列长度

## 高级功能

### 1. 自定义模型架构

可以使用高级BERT分类器，支持多层分类头：

```json
{
  "model_type": "advanced_bert_classifier",
  "hidden_dims": [512, 256],
  "activation": "relu",
  "use_pooler": true
}
```

### 2. 混合精度训练

在训练脚本中可以启用混合精度训练以提高效率：

```python
# 在train.py中添加混合精度支持
from torch.cuda.amp import autocast, GradScaler
```

### 3. 模型集成

可以训练多个模型并进行集成预测以提高性能。

### 4. 数据增强

可以在数据预处理阶段添加数据增强技术：
- 回译
- 同义词替换
- 随机删除/插入

## 训练监控

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir checkpoints/tensorboard
```

## 性能优化建议

1. **批次大小**: 根据GPU内存调整，通常16-32
2. **学习率**: BERT通常使用2e-5到5e-5
3. **序列长度**: 根据数据分布设置，通常128-512
4. **训练轮数**: 通常3-10个epoch，使用早停防止过拟合
5. **梯度裁剪**: 使用1.0的最大梯度范数

## 常见问题

### Q: 显存不足怎么办？
A: 
- 减小batch_size
- 减小max_length  
- 使用梯度累积
- 启用混合精度训练

### Q: 训练速度太慢怎么办？
A:
- 使用GPU训练
- 增大batch_size（在显存允许的情况下）
- 使用多GPU并行训练
- 启用混合精度训练

### Q: 模型效果不好怎么办？
A:
- 检查数据质量和分布
- 调整学习率
- 增加训练轮数
- 尝试不同的模型架构
- 使用数据增强技术

### Q: 如何处理类别不平衡？
A:
- 使用类别权重
- 数据采样平衡
- 焦点损失函数
- 评估时使用F1-score而非准确率

## ⚡ 模型优化功能

本项目提供完整的模型优化解决方案，支持NF4量化和多种剪枝策略，在保持高精度的前提下显著减少模型大小和推理时间。

### 🔧 优化功能

#### NF4量化优化
- **4位标准化浮点量化**: 将模型权重从32位压缩到4位
- **硬件加速支持**: 集成bitsandbytes库，支持GPU加速量化
- **选择性量化**: 智能选择重要层进行量化，保护关键参数
- **预期效果**: 模型大小减少75%，精度损失通常<0.5%

#### 多种剪枝策略
- **幅度剪枝**: 基于权重大小的非结构化剪枝
- **智能剪枝**: 层级自适应剪枝，不同层使用不同稀疏度
- **结构化剪枝**: N:M模式剪枝，硬件友好
- **块稀疏剪枝**: 块级别稀疏化，提升推理效率
- **渐进式剪枝**: 逐步增加稀疏度，保持模型精度

#### 组合优化策略
- **剪枝+量化**: 先剪枝后量化，最大化压缩比
- **自适应优化**: 根据层的重要性调整优化强度
- **精度保护**: 保护重要层（如嵌入层、池化层）

### 🚀 优化使用方法

#### Web界面优化（推荐）
1. 启动Web界面：`python start.py web`
2. 访问管理控制台：http://127.0.0.1:8000/admin.html
3. 在"模型优化"标签页中选择优化策略
4. 设置参数并开始优化
5. 在"模型性能比较"区域测试优化效果

#### 命令行优化
```bash
# 对已训练的模型进行所有类型优化
python optimize_model.py --model_path checkpoints/best_model --optimization_type all

# 仅进行NF4量化
python optimize_model.py --model_path checkpoints/best_model --optimization_type quantization

# 组合优化（剪枝20% + NF4量化）
python optimize_model.py --model_path checkpoints/best_model --optimization_type combined --sparsity_ratio 0.2
```

#### 编程接口
```python
from simple_advanced_optimization import SimpleAdvancedOptimizer

# 创建优化器
optimizer = SimpleAdvancedOptimizer("checkpoints/best_model")

# 应用NF4量化
optimizer.apply_nf4_quantization("optimized_models")

# 应用智能剪枝（20%稀疏度）
optimizer.apply_intelligent_pruning(0.2, "optimized_models")

# 应用组合优化
optimizer.apply_combined_optimization(0.15, "optimized_models")
```

### 📊 优化效果预期

| 优化方案 | 模型大小压缩 | 精度保持率 | 推理加速 | 适用场景 |
|----------|-------------|------------|----------|----------|
| 仅NF4量化 | 75% | 99.5%+ | 1.5x | 高精度要求 |
| 15%剪枝+量化 | 80% | 99%+ | 2.0x | 生产环境 |
| 25%剪枝+量化 | 85% | 98%+ | 2.3x | 移动设备 |
| 块稀疏+量化 | 80% | 98.5%+ | 2.5x | 专用硬件 |

### 🛠️ 优化建议

#### 不同场景推荐
- **生产环境**: 使用15%剪枝+NF4量化，平衡精度和性能
- **移动设备**: 使用25%剪枝+NF4量化，最大化压缩比
- **高精度要求**: 仅使用NF4量化，避免剪枝
- **实时推理**: 使用块稀疏剪枝+量化，配合专用硬件

#### 最佳实践
1. **备份原始模型**: 始终保留未优化的原始模型作为备份
2. **分步验证**: 每个优化步骤后都要验证模型性能
3. **渐进优化**: 从保守参数开始，逐步增加优化强度
4. **A/B测试**: 在生产环境中逐步切换到优化模型

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系开发者。