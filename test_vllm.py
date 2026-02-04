"""
vLLM 多卡并行测试脚本 - Qwen3-VL-30B-A3B-Instruct
支持 Tensor Parallelism 实现真正的多卡并行推理
"""

import time
import torch
from vllm import LLM, SamplingParams
from typing import List
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
        print(f"    已分配内存: {memory_allocated:.2f} GB")
        print(f"    预留内存: {memory_reserved:.2f} GB")
    print("=" * 80)


def test_single_inference(llm: LLM, num_gpus: int):
    """测试单个样本推理"""
    print("\n[测试 1] 单样本推理测试")
    print("-" * 80)

    # 准备输入 - vLLM OpenAI 兼容格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                },
            ],
        }
    ]

    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=128,
    )

    # 预热
    print("预热中...")
    _ = llm.chat(messages=[messages], sampling_params=sampling_params)

    # 正式测试
    print(f"\n使用 {num_gpus} 张 GPU 进行推理...")
    start_time = time.time()

    outputs = llm.chat(messages=[messages], sampling_params=sampling_params)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 输出结果
    print(f"\n生成结果:")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"输出: {generated_text}")
        print(f"\n生成的 token 数: {len(output.outputs[0].token_ids)}")

    print(f"\n推理时间: {elapsed_time:.4f} 秒")
    print(
        f"吞吐量: {len(output.outputs[0].token_ids) / elapsed_time:.2f} tokens/秒")


def test_batch_inference(llm: LLM, batch_size: int, num_gpus: int):
    """测试批量推理"""
    print(f"\n[测试 2] 批量推理测试 (Batch Size: {batch_size})")
    print("-" * 80)

    # 准备批量输入
    test_prompts = [
        "Describe this image.",
        "What objects can you see in this picture?",
        "What is the main subject of this image?",
        "Describe the colors and composition.",
        "What is happening in this scene?",
    ]

    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                        },
                    },
                ],
            }
        ]
        for prompt in test_prompts[:batch_size]
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    # 预热
    print("预热中...")
    _ = llm.chat(messages=messages_list[:1], sampling_params=sampling_params)

    # 批量推理
    print(f"\n使用 {num_gpus} 张 GPU 进行批量推理...")
    start_time = time.time()

    outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 统计结果
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    print(f"\n批量大小: {batch_size}")
    print(f"总推理时间: {elapsed_time:.4f} 秒")
    print(f"平均每个样本时间: {elapsed_time / batch_size:.4f} 秒")
    print(f"总 token 数: {total_tokens}")
    print(f"吞吐量: {total_tokens / elapsed_time:.2f} tokens/秒")
    print(
        f"并行效率提升: {batch_size / (elapsed_time / (elapsed_time / batch_size)):.2f}x")

    # 显示部分结果
    print(f"\n前 2 个结果示例:")
    for i, output in enumerate(outputs[:2]):
        print(f"\n样本 {i+1}:")
        print(f"  输出: {output.outputs[0].text[:100]}...")


def test_throughput(llm: LLM, num_requests: int, num_gpus: int):
    """测试持续吞吐量"""
    print(f"\n[测试 3] 持续吞吐量测试 ({num_requests} 个请求)")
    print("-" * 80)

    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Request {i}: Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                        },
                    },
                ],
            }
        ]
        for i in range(num_requests)
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )

    print(f"使用 {num_gpus} 张 GPU 处理 {num_requests} 个请求...")
    start_time = time.time()

    outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)

    end_time = time.time()
    elapsed_time = end_time - start_time

    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    print(f"\n总时间: {elapsed_time:.4f} 秒")
    print(f"平均延迟: {elapsed_time / num_requests * 1000:.2f} ms/请求")
    print(f"请求吞吐量: {num_requests / elapsed_time:.2f} 请求/秒")
    print(f"Token 吞吐量: {total_tokens / elapsed_time:.2f} tokens/秒")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("vLLM 多卡并行性能测试 - Qwen3-VL-30B-A3B-Instruct")
    print("=" * 80)

    # 打印 GPU 信息
    print_gpu_info()

    # 配置参数
    num_gpus = torch.cuda.device_count()
    tensor_parallel_size = 4

    print(f"\n配置信息:")
    print(f"  Tensor Parallel Size: {tensor_parallel_size}")
    print(f"  GPU Memory Utilization: 0.90")
    print(f"  Max Model Length: 4096")

    # 初始化 vLLM
    print("\n加载模型中...")
    start_load = time.time()

    # 设置环境变量启用调试日志
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    # os.environ["OMP_NUM_THREADS"] = "1"

    max_model_len = 8192

    # llm = LLM(
    #     model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    #     tensor_parallel_size=tensor_parallel_size,  # 多卡并行
    #     gpu_memory_utilization=0.85,
    #     trust_remote_code=True,
    #     enable_expert_parallel=True,
    #     max_model_len=max_model_len,
    #     dtype="bfloat16",
    #     mm_encoder_tp_mode="data",
    #     # 禁用一些可能导致问题的优化
    #     enforce_eager=True,  # 禁用 CUDA graph，避免卡住
    #     max_num_seqs=256,     # 最大批处理序列数
    # )

    llm = LLM(
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.90,  # 给系统留一点 buffer
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,    
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 1},  # 明确限制每条消息的图片数，减轻 Profiling 负担
    )

    load_time = time.time() - start_load
    print(f"模型加载完成! 耗时: {load_time:.2f} 秒")

    # 运行测试
    try:
        # 测试 1: 单样本推理
        test_single_inference(llm, tensor_parallel_size)

        # 测试 2: 批量推理
        test_batch_inference(llm, batch_size=5, num_gpus=tensor_parallel_size)

        # # 测试 3: 持续吞吐量
        # test_throughput(llm, num_requests=10, num_gpus=tensor_parallel_size)

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
