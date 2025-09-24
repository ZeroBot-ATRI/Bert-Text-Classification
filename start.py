#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键启动脚本 - BERT文本分类项目
简化版启动脚本，快速启动API服务或执行常用任务
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """打印项目横幅"""
    print("=" * 60)
    print("    🚀 BERT中文文本分类项目 - 一键启动")
    print("=" * 60)

def check_basic_files():
    """检查基本文件是否存在"""
    required_files = [
        "app.py",
        "data/class.txt", 
        "pretrain/bert-base-chinese/config.json"
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print(f"❌ 缺少必要文件: {missing}")
        return False
    
    return True

def start_web_interface():
    """启动Web界面"""
    print(f"🌍 启动Web界面...")
    print("-" * 40)
    
    # 直接使用app.py启动服务
    import uvicorn
    try:
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Web界面已停止")
    except Exception as e:
        print(f"❌ Web界面启动失败: {e}")

def start_api(host="127.0.0.1", port=8000, reload=False):
    """启动API服务"""
    print(f"🌟 启动FastAPI服务...")
    print(f"📍 服务地址: http://{host}:{port}")
    print(f"📚 API文档: http://{host}:{port}/docs")
    print(f"🔄 热重载: {'开启' if reload else '关闭'}")
    print("-" * 40)
    
    # 直接使用uvicorn启动
    import uvicorn
    try:
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 API服务已停止")

def test_api(url="http://127.0.0.1:8000"):
    """测试API服务"""
    print(f"🧪 测试API服务: {url}")
    print("-" * 40)
    
    # 使用内置的快速测试
    try:
        import requests
        import json
        
        # 测试用例
        test_cases = [
            "股票市场今天表现不错",
            "教育部发布新的招生政策", 
            "中国男篮获得亚运会冠军",
            "房地产市场调控政策收紧"
        ]
        
        # 1. 健康检查
        print("1. 健康检查...")
        response = requests.get(f"{url}/health")
        health = response.json()
        print(f"   状态: {health['status']}")
        print(f"   模型已加载: {health['model_loaded']}")
        
        # 2. 单文本预测测试
        print("\n2. 预测测试...")
        for i, text in enumerate(test_cases, 1):
            try:
                response = requests.post(f"{url}/predict/simple", 
                                       json={"text": text})
                result = response.json()
                print(f"   {i}. {text}")
                print(f"      → {result['predicted_class']} ({result['confidence']:.3f})")
            except Exception as e:
                print(f"   {i}. {text} → 错误: {e}")
        
        print(f"\n✅ 快速测试完成！")
        
    except ImportError:
        print("❌ 缺少requests模块，使用完整测试脚本")
        cmd = [sys.executable, "test_api.py"]
        subprocess.run(cmd)
    except Exception as e:
        if 'requests' in str(type(e)) and 'ConnectionError' in str(type(e)):
            print("❌ 无法连接到API服务，请确保服务已启动")
        else:
            print(f"❌ 测试失败: {e}")

def show_status():
    """显示项目状态"""
    print("📊 项目状态:")
    print("-" * 40)
    
    # 检查模型文件
    model_files = [
        "checkpoints/best_model/pytorch_model.bin",
        "checkpoints/final_model/pytorch_model.bin"
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            found_models.append(model_file)
    
    if found_models:
        print(f"✅ 已训练模型: {len(found_models)} 个")
        for model in found_models:
            print(f"   - {model}")
    else:
        print("❌ 未找到已训练的模型")
    
    # 检查数据文件
    data_files = ["data/train.txt", "data/test.txt", "data/class.txt"]
    for data_file in data_files:
        if os.path.exists(data_file):
            size_mb = os.path.getsize(data_file) / 1024 / 1024
            print(f"✅ {data_file}: {size_mb:.2f} MB")
        else:
            print(f"❌ {data_file}: 不存在")

def quick_demo():
    """快速演示"""
    print("🎯 运行快速演示...")
    print("-" * 40)
    
    if os.path.exists("demo.py"):
        subprocess.run([sys.executable, "demo.py"])
    elif os.path.exists("predict.py"):
        subprocess.run([sys.executable, "predict.py", "--mode", "interactive"])
    else:
        print("❌ 未找到演示文件")

def manage_project():
    """调用项目管理器"""
    print("🛠️  调用项目管理器...")
    print("-" * 40)
    subprocess.run([sys.executable, "manage.py"] + sys.argv[2:])

def show_help():
    """显示帮助信息"""
    print_banner()
    print()
    print("📋 使用方法:")
    print()
    print("  python start.py web                   # 启动Web界面 (推荐)")
    print("  python start.py api                   # 启动API服务")
    print("  python start.py api --dev             # 启动API服务 (开发模式)")
    print("  python start.py api --port 8080       # 指定端口启动")
    print("  python start.py test                  # 测试API服务")
    print("  python start.py status                # 查看项目状态")
    print("  python start.py demo                  # 运行演示")
    print("  python start.py manage [args...]      # 调用完整管理器")
    print()
    print("🔧 API服务参数:")
    print("  --host HOST                           # 服务主机 (默认: 127.0.0.1)")
    print("  --port PORT                           # 服务端口 (默认: 8000)")
    print("  --dev                                 # 开发模式 (热重载)")
    print()
    print("💡 示例:")
    print("  python start.py                       # 直接启动Web界面")
    print("  python start.py web                   # 启动Web界面")
    print("  python start.py api --dev --port 8080 # 开发模式,端口8080")
    print("  python start.py manage train          # 训练模型")
    print("  python start.py manage full           # 完整流程")

def main():
    print_banner()
    
    # 检查基本环境
    if not check_basic_files():
        print("\n❌ 环境检查失败，请确保项目文件完整")
        return
    
    # 解析命令行参数
    if len(sys.argv) == 1:
        # 默认启动Web界面
        start_web_interface()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'web':
        start_web_interface()
    
    elif command in ['api', 'start', 'serve']:
        parser = argparse.ArgumentParser(description='启动API服务')
        parser.add_argument('command', help='命令')
        parser.add_argument('--host', default='127.0.0.1', help='服务主机')
        parser.add_argument('--port', type=int, default=8000, help='服务端口')
        parser.add_argument('--dev', action='store_true', help='开发模式(热重载)')
        
        args = parser.parse_args()
        start_api(args.host, args.port, args.dev)
    
    elif command == 'test':
        test_api()
    
    elif command == 'status':
        show_status()
    
    elif command == 'demo':
        quick_demo()
    
    elif command == 'manage':
        manage_project()
    
    elif command in ['help', '-h', '--help']:
        show_help()
    
    else:
        print(f"❌ 未知命令: {command}")
        print("💡 使用 'python start.py help' 查看帮助")

if __name__ == "__main__":
    main()