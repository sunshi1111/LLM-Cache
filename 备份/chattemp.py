import argparse
import contextlib
import random
import os
import numpy as np
import torch
import time
import hashlib

from gemma import config
from gemma import model1 as gemma_model

@contextlib.contextmanager
def set_default_tensor_type(dtype: torch.dtype):
    """设置默认的torch dtype"""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

def get_prompt_hash(prompt):
    """生成提示文本的哈希值，用于标识"""
    return hashlib.md5(prompt.encode('utf-8', errors='ignore')).hexdigest()[:12]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--variant", type=str, default="2b",
                        choices=["2b", "2b-v2", "7b", "9b", "27b"])
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--output_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--debug", action='store_true', help="显示调试信息")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载模型
    print("正在加载模型...")
    model_config = config.get_model_config(args.variant)
    model_config.dtype = "float32" if args.device == "cpu" else "float16"
    model_config.quant = args.quant

    device = torch.device(args.device)
    dtype = torch.float32 if args.device == "cpu" else torch.float16

    with set_default_tensor_type(dtype):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(args.ckpt)
        model = model.to(device).eval()
    print("模型加载完成！")

    # 对话历史
    conversation_history = []
    system_prompt = "你是一个有用的AI助手。请提供有帮助、安全、准确的回答。每次只生成一个回复，不要继续对话，不要自问自答。"

    # 会话标识
    session_id = int(time.time())

    print("开始与Gemma模型对话，输入'exit'退出\n")
    print(f"系统: {system_prompt}\n")

    try:
        turn = 0

        while True:
            turn += 1
            user_input = input("用户: ")
            if user_input.lower() == "exit":
                break

            # 添加用户输入到对话历史
            conversation_history.append({"role": "user", "content": user_input})

            # 构建当前完整对话提示
            current_full_prompt = system_prompt.encode('utf-8', errors='ignore').decode('utf-8')
            for msg in conversation_history:
                if msg["role"] == "user":
                    current_full_prompt += f"\n\n用户: {str(msg['content']).encode('utf-8', errors='ignore').decode('utf-8')}"
                else:
                    current_full_prompt += f"\n\nGemma: {str(msg['content']).encode('utf-8', errors='ignore').decode('utf-8')}"

            # 添加最后的提示词
            current_full_prompt += "\n\nGemma:"

            # 计算当前完整提示的哈希值
            current_full_prompt_hash = get_prompt_hash(current_full_prompt)

            if args.debug:
                print(f"\n[调试] 轮次 {turn} - 完整对话提示")
                print(f"[调试] 提示长度: {len(model.tokenizer.encode(current_full_prompt))} tokens")
                print(f"[调试] 提示哈希: {current_full_prompt_hash}")

            start_time = time.time()

            # 生成回复，不使用KV缓存特性
            with torch.no_grad():
                response = model.generate(
                    current_full_prompt,
                    device,
                    output_len=args.output_len,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50
                )

            end_time = time.time()

            if args.debug:
                print(f"\n[调试] 轮次 {turn} 响应:\n{response}")
                print(f"[调试] 生成时间: {end_time - start_time:.2f} 秒")

            # 显示回复
            print(f"Gemma: {response}")

            # 保存助手回复
            conversation_history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\n对话已结束")

    print("感谢使用Gemma!")

if __name__ == "__main__":
    main()