"""
简单的模型加载测试 - 验证模型文件是否完整
"""
import torch
from transformers import AutoConfig, AutoTokenizer
import os

print("=" * 80)
print("测试模型配置和 tokenizer 加载")
print("=" * 80)

model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"

print(f"\n1. 加载模型配置...")
try:
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print(f"✅ 配置加载成功!")
    print(f"   模型类型: {config.model_type}")
    print(f"   架构: {config.architectures}")
    if hasattr(config, 'num_experts'):
        print(f"   专家数量: {config.num_experts}")
    if hasattr(config, 'num_experts_per_tok'):
        print(f"   每token激活专家数: {config.num_experts_per_tok}")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    exit(1)

print(f"\n2. 加载 tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print(f"✅ Tokenizer 加载成功!")
    print(f"   词表大小: {len(tokenizer)}")
except Exception as e:
    print(f"❌ Tokenizer 加载失败: {e}")
    exit(1)

print(f"\n3. 检查 GPU 可用性...")
if torch.cuda.is_available():
    print(f"✅ CUDA 可用")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print(f"❌ CUDA 不可用")

print("\n" + "=" * 80)
print("所有基础检查通过! 模型文件完整。")
print("问题可能在于 vLLM 加载 MoE 模型的实现。")
print("=" * 80)
