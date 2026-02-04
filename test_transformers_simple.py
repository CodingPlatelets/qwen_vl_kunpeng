"""
使用 Transformers + device_map 实现简单多卡推理
这是目前最稳定的 Qwen3-VL-30B-A3B 多卡方案
"""
import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
import time

def process_vision_info(messages):
    """处理视觉输入"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message["content"], list):
            for content in message["content"]:
                if content.get("type") == "image":
                    image_inputs.append(content["image"])
                elif content.get("type") == "video":
                    video_inputs.append(content["video"])
    
    return image_inputs or None, video_inputs or None


print("=" * 80)
print("Transformers 多卡推理测试 - Qwen3-VL-30B-A3B-Instruct")
print("=" * 80)

# GPU 信息
print(f"\nGPU 数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 加载模型 - 使用 device_map="auto" 自动分配到多卡
print("\n加载模型中...")
start_time = time.time()

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 自动多卡分配
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    trust_remote_code=True
)

load_time = time.time() - start_time
print(f"✅ 模型加载成功! 耗时: {load_time:.2f} 秒")

# 打印模型分布情况
print("\n模型分布情况:")
if hasattr(model, 'hf_device_map'):
    device_map = model.hf_device_map
    # 统计各 GPU 上的模块数量
    device_counts = {}
    for module_name, device in device_map.items():
        device_str = str(device)
        if device_str not in device_counts:
            device_counts[device_str] = 0
        device_counts[device_str] += 1
    
    for device, count in sorted(device_counts.items()):
        print(f"  {device}: {count} 个模块")

# 打印 GPU 内存使用
print("\nGPU 内存使用:")
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"  GPU {i}: 已分配 {allocated:.2f} GB, 预留 {reserved:.2f} GB")

# 测试推理
print("\n开始推理测试...")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]

# 准备输入
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# 生成
print("生成中...")
start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
    )

generation_time = time.time() - start_time

# 解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(f"\n生成结果:")
print(f"{output_text}")
print(f"\n生成耗时: {generation_time:.2f} 秒")
print(f"生成 token 数: {len(generated_ids_trimmed[0])}")
print(f"吞吐量: {len(generated_ids_trimmed[0]) / generation_time:.2f} tokens/秒")

print("\n" + "=" * 80)
print("测试完成!")
print("=" * 80)
