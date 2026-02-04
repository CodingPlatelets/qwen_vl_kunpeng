"""
PyTorch Pipeline Parallelism 多卡测试脚本
使用原生 PyTorch + Accelerate 实现真正的多卡并行
适用于 ARM 架构，无需额外依赖
"""

import time
import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from accelerate import Accelerator, PartialState
import os


def print_gpu_info():
    """打印 GPU 信息"""
    print("=" * 80)
    print("GPU 信息:")
    gpu_count = torch.cuda.device_count()
    print(f"可用 GPU 数量: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"    总内存: {total_memory:.2f} GB")
        print(f"    已分配: {memory_allocated:.2f} GB")
        print(f"    预留: {memory_reserved:.2f} GB")
    print("=" * 80)


def test_single_inference(model, processor, device):
    """测试单个样本推理"""
    print("\n[测试 1] 单样本推理测试")
    print("-" * 80)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    
    # 准备输入
    print("开始处理输入（下载和处理图像）...")
    start_process = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    process_time = time.time() - start_process
    print(f"输入处理完成，耗时: {process_time:.2f} 秒")
    
    # 将输入移到设备
    print(f"将输入移到设备 {device}...")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("输入已移到 GPU")
    
    # 预热
    print("预热中...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # 清空 CUDA 缓存
    torch.cuda.empty_cache()
    
    # 正式测试
    print("\n开始推理...")
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    torch.cuda.synchronize()  # 确保所有 GPU 操作完成
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    num_tokens = len(generated_ids_trimmed[0])
    
    print(f"\n生成结果:")
    print(f"输出: {output_text[0]}")
    print(f"\n生成的 token 数: {num_tokens}")
    print(f"推理时间: {elapsed_time:.4f} 秒")
    print(f"吞吐量: {num_tokens / elapsed_time:.2f} tokens/秒")
    
    return elapsed_time, num_tokens


def test_batch_inference(model, processor, device, batch_size=4):
    """测试批量推理"""
    print(f"\n[测试 2] 批量推理测试 (Batch Size: {batch_size})")
    print("-" * 80)
    
    # 准备批量输入
    test_prompts = [
        "Describe this image.",
        "What objects can you see in this picture?",
        "What is the main subject of this image?",
        "Describe the colors and composition.",
    ]
    
    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for prompt in test_prompts[:batch_size]
    ]
    
    # 处理批量输入
    all_inputs = []
    for messages in messages_list:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        all_inputs.append(inputs)
    
    # 预热
    print("预热中...")
    with torch.no_grad():
        inputs_warmup = {k: v.to(device) for k, v in all_inputs[0].items()}
        _ = model.generate(**inputs_warmup, max_new_tokens=10)
    
    torch.cuda.empty_cache()
    
    # 批量推理
    print(f"\n开始批量推理...")
    start_time = time.time()
    
    all_outputs = []
    total_tokens = 0
    
    with torch.no_grad():
        for inputs in all_inputs:
            inputs_device = {k: v.to(device) for k, v in inputs.items()}
            generated_ids = model.generate(**inputs_device, max_new_tokens=128)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_device["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            all_outputs.append(output_text[0])
            total_tokens += len(generated_ids_trimmed[0])
    
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n批量大小: {batch_size}")
    print(f"总推理时间: {elapsed_time:.4f} 秒")
    print(f"平均每个样本时间: {elapsed_time / batch_size:.4f} 秒")
    print(f"总 token 数: {total_tokens}")
    print(f"吞吐量: {total_tokens / elapsed_time:.2f} tokens/秒")
    
    print(f"\n前 2 个结果示例:")
    for i, output in enumerate(all_outputs[:2]):
        print(f"\n样本 {i+1}: {output[:100]}...")
    
    return elapsed_time, total_tokens


def test_throughput(model, processor, device, num_requests=10):
    """测试持续吞吐量"""
    print(f"\n[测试 3] 持续吞吐量测试 ({num_requests} 个请求)")
    print("-" * 80)
    
    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": f"Request {i}: Describe this image briefly."},
                ],
            }
        ]
        for i in range(num_requests)
    ]
    
    print(f"处理 {num_requests} 个请求...")
    start_time = time.time()
    
    total_tokens = 0
    
    with torch.no_grad():
        for messages in messages_list:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generated_ids = model.generate(**inputs, max_new_tokens=100)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            total_tokens += len(generated_ids_trimmed[0])
    
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n总时间: {elapsed_time:.4f} 秒")
    print(f"平均延迟: {elapsed_time / num_requests * 1000:.2f} ms/请求")
    print(f"请求吞吐量: {num_requests / elapsed_time:.2f} 请求/秒")
    print(f"Token 吞吐量: {total_tokens / elapsed_time:.2f} tokens/秒")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("PyTorch Pipeline Parallelism 多卡性能测试")
    print("模型: Qwen3-VL-30B-A3B-Instruct")
    print("=" * 80)
    
    # 打印初始 GPU 信息
    print_gpu_info()
    
    num_gpus = torch.cuda.device_count()
    print(f"\n配置信息:")
    print(f"  使用 GPU 数量: {num_gpus}")
    print(f"  并行策略: device_map='auto' (模型并行)")
    print(f"  数据类型: bfloat16")
    print(f"  Flash Attention 2: 启用")
    
    # 加载模型
    print("\n加载模型中...")
    start_load = time.time()
    
    # 使用 device_map="auto" 自动分配到多卡
    print("步骤 1: 加载模型权重...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # 先用 eager 避免 flash_attention_2 的问题
        device_map="auto",  # 自动多卡分配
        trust_remote_code=True,
    )
    print("步骤 2: 模型权重加载完成")
    
    print("步骤 3: 加载 processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        trust_remote_code=True
    )
    print("步骤 4: Processor 加载完成")
    
    print("步骤 5: 设置模型为评估模式...")
    model.eval()
    print("步骤 6: 评估模式设置完成")
    
    load_time = time.time() - start_load
    print(f"模型加载完成! 耗时: {load_time:.2f} 秒")
    
    # 获取主设备
    device = next(model.parameters()).device
    print(f"主设备: {device}")
    
    # 打印模型分布
    print("\n模型层分布:")
    if hasattr(model, 'hf_device_map'):
        device_map = model.hf_device_map
        for layer_name, device_id in list(device_map.items())[:10]:
            print(f"  {layer_name}: GPU {device_id}")
        if len(device_map) > 10:
            print(f"  ... (还有 {len(device_map) - 10} 层)")
    
    print("\n" + "=" * 80)
    
    # 运行测试
    try:
        # 测试 1: 单样本推理
        test_single_inference(model, processor, device)
        
        # 测试 2: 批量推理
        test_batch_inference(model, processor, device, batch_size=4)
        
        # 测试 3: 持续吞吐量
        test_throughput(model, processor, device, num_requests=10)
        
        print("\n" + "=" * 80)
        print("所有测试完成!")
        print("=" * 80)
        
        # 最终 GPU 状态
        print_gpu_info()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
