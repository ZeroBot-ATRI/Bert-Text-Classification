#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI服务测试客户端
"""

import requests
import json
import time
import argparse

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """测试健康检查接口"""
        print("=" * 50)
        print("测试健康检查接口")
        print("=" * 50)
        
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return response.status_code == 200
        except Exception as e:
            print(f"请求失败: {e}")
            return False
    
    def test_model_info(self):
        """测试模型信息接口"""
        print("=" * 50)
        print("测试模型信息接口")
        print("=" * 50)
        
        try:
            response = requests.get(f"{self.base_url}/info")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return response.status_code == 200
        except Exception as e:
            print(f"请求失败: {e}")
            return False
    
    def test_single_prediction(self):
        """测试单文本预测"""
        print("=" * 50)
        print("测试单文本预测接口")
        print("=" * 50)
        
        test_texts = [
            "A股市场今日大幅上涨，沪指收盘涨幅超过2%",
            "教育部发布新的高考改革方案",
            "中国男篮在亚运会上获得金牌",
            "房地产市场调控政策进一步收紧",
            "科学家发现新的治疗癌症的方法"
        ]
        
        for text in test_texts:
            try:
                data = {"text": text}
                response = requests.post(f"{self.base_url}/predict", json=data)
                
                print(f"输入文本: {text}")
                print(f"状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"预测类别: {result['predicted_class']}")
                    print(f"置信度: {result['confidence']:.4f}")
                    print("前3个概率:")
                    if result.get('probabilities'):
                        sorted_probs = sorted(result['probabilities'].items(), 
                                            key=lambda x: x[1], reverse=True)
                        for i, (class_name, prob) in enumerate(sorted_probs[:3]):
                            print(f"  {i+1}. {class_name}: {prob:.4f}")
                else:
                    print(f"错误响应: {response.text}")
                
                print("-" * 30)
                time.sleep(0.5)  # 避免请求过快
                
            except Exception as e:
                print(f"请求失败: {e}")
                continue
    
    def test_batch_prediction(self):
        """测试批量预测"""
        print("=" * 50)
        print("测试批量预测接口")
        print("=" * 50)
        
        test_texts = [
            "股市今天表现不错",
            "学校开始放暑假了",
            "篮球比赛很精彩"
        ]
        
        try:
            data = {"texts": test_texts}
            response = requests.post(f"{self.base_url}/predict/batch", json=data)
            
            print(f"输入文本数量: {len(test_texts)}")
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"预测结果数量: {result['total_count']}")
                
                for i, pred in enumerate(result['predictions']):
                    print(f"\n文本 {i+1}: {pred['text']}")
                    print(f"预测类别: {pred['predicted_class']}")
                    print(f"置信度: {pred['confidence']:.4f}")
            else:
                print(f"错误响应: {response.text}")
                
        except Exception as e:
            print(f"请求失败: {e}")
    
    def test_simple_prediction(self):
        """测试简单预测接口"""
        print("=" * 50)
        print("测试简单预测接口")
        print("=" * 50)
        
        try:
            data = {"text": "今天股票涨了很多"}
            response = requests.post(f"{self.base_url}/predict/simple", json=data)
            
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            
        except Exception as e:
            print(f"请求失败: {e}")
    
    def test_get_classes(self):
        """测试获取类别接口"""
        print("=" * 50)
        print("测试获取类别接口")
        print("=" * 50)
        
        try:
            response = requests.get(f"{self.base_url}/classes")
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"类别总数: {result['total_count']}")
                print("所有类别:")
                for class_info in result['classes']:
                    print(f"  {class_info['id']}: {class_info['name']}")
            else:
                print(f"错误响应: {response.text}")
                
        except Exception as e:
            print(f"请求失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始API测试...")
        
        # 等待服务启动
        print("等待服务启动...")
        time.sleep(2)
        
        tests = [
            ("健康检查", self.test_health),
            ("模型信息", self.test_model_info),
            ("获取类别", self.test_get_classes),
            ("单文本预测", self.test_single_prediction),
            ("批量预测", self.test_batch_prediction),
            ("简单预测", self.test_simple_prediction),
        ]
        
        success_count = 0
        for test_name, test_func in tests:
            try:
                print(f"\n正在执行: {test_name}")
                result = test_func()
                if result is not False:
                    success_count += 1
                    print(f"✓ {test_name} 通过")
                else:
                    print(f"✗ {test_name} 失败")
            except Exception as e:
                print(f"✗ {test_name} 出错: {e}")
            
            time.sleep(1)
        
        print("=" * 50)
        print(f"测试完成！通过 {success_count}/{len(tests)} 个测试")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='测试BERT文本分类API')
    parser.add_argument('--url', type=str, default='http://localhost:8000', 
                       help='API服务地址')
    parser.add_argument('--test', type=str, 
                       choices=['health', 'info', 'single', 'batch', 'simple', 'classes', 'all'],
                       default='all', help='要执行的测试')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test == 'all':
        tester.run_all_tests()
    elif args.test == 'health':
        tester.test_health()
    elif args.test == 'info':
        tester.test_model_info()
    elif args.test == 'single':
        tester.test_single_prediction()
    elif args.test == 'batch':
        tester.test_batch_prediction()
    elif args.test == 'simple':
        tester.test_simple_prediction()
    elif args.test == 'classes':
        tester.test_get_classes()

if __name__ == "__main__":
    main()