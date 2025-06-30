import argparse
import contextlib
import random
import os
import numpy as np
import torch
import time
import hashlib
import subprocess

from gemma import config
from gemma import model as gemma_model

@contextlib.contextmanager
def set_default_tensor_type(dtype: torch.dtype):
    """设置默认的torch dtype"""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

def get_prompt_hash(prompt):
    """生成提示文本的哈希值，用于缓存标识"""
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

    # KV缓存相关参数
    parser.add_argument("--cache-dir", type=str, default="./chat_cache",
                      help="本地保存KV缓存的目录")
    parser.add_argument("--no-cache", action="store_true",
                      help="禁用KV缓存功能")

    # Crail相关参数
    parser.add_argument("--use-crail", action="store_true",
                      help="使用Crail存储KV缓存")
    parser.add_argument("--crail-cache-dir", type=str, default="/kvcache",
                      help="Crail中存储KV缓存的目录")
    parser.add_argument("--crail-jar", type=str,
                      default="/home/ms-admin/sunshi/crail-example/target/crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar",
                      help="Crail客户端JAR路径")
    parser.add_argument("--crail-conf", type=str,
                      default="/home/ms-admin/sunshi/crail/conf",
                      help="Crail配置目录路径")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建本地缓存目录
    if not args.no_cache and not args.use_crail and not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # 设置Crail环境变量
    if args.use_crail:
        os.environ["CRAIL_KVCACHE_JAR"] = args.crail_jar
        os.environ["CRAIL_CONF_DIR"] = args.crail_conf

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
    system_prompt = "你是一个有用的AI助手。请提供有帮助、安全、准确的回答。不要自问自答。"

    # 会话标识
    session_id = int(time.time())

    # 最新保存的完整对话缓存路径
    last_full_prompt_cache = None
    # 上一轮的提示文本(带哈希)
    last_full_prompt = None
    last_full_prompt_hash = None

    print("开始与Gemma模型对话，输入'exit'退出\n")
    print(f"系统: {system_prompt}\n")

    if args.use_crail:
        print(f"Crail KV缓存已启用，将保存在: {args.crail_cache_dir}\n")
    elif not args.no_cache:
        print(f"本地KV缓存已启用，将保存在: {args.cache_dir}\n")

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
            current_full_prompt = system_prompt
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

            # 确定本轮生成策略:
            # 1. 第一轮对话: 从头生成
            # 2. 后续轮次: 尝试重用上一轮的KV Cache

            # 缓存文件命名
            if args.use_crail:
                current_cache_path = f"{args.crail_cache_dir}/session_{session_id}_turn_{turn}_{current_full_prompt_hash}.pt"
            else:
                current_cache_path = os.path.join(args.cache_dir, f"session_{session_id}_turn_{turn}_{current_full_prompt_hash}.pt")

            # 生成参数
            gen_kwargs = {}

            # 对于非首轮对话，尝试重用上一轮的KV Cache
            if turn > 1 and last_full_prompt and last_full_prompt_cache:
                # 找到上一轮提示与当前提示的公共部分长度
                common_prefix_len = 0
                for i in range(min(len(last_full_prompt), len(current_full_prompt))):
                    if last_full_prompt[i] == current_full_prompt[i]:
                        common_prefix_len += 1
                    else:
                        break

                # 只有当存在足够的公共前缀时才重用KV Cache
                if common_prefix_len > len(system_prompt):
                    if args.debug:
                        common_prefix = current_full_prompt[:common_prefix_len]
                        print(f"[调试] 发现公共前缀: {common_prefix_len} 字符")
                        print(f"[调试] 公共前缀结束于: \"{common_prefix[-20:]}\"")

                    common_prefix_tokens = len(model.tokenizer.encode(current_full_prompt[:common_prefix_len]))
                    total_tokens = len(model.tokenizer.encode(current_full_prompt))

                    if args.debug:
                        print(f"[调试] 公共前缀tokens: {common_prefix_tokens}/{total_tokens}")

                    # 设置KV Cache加载参数
                    if args.use_crail:
                        gen_kwargs["load_kv_cache_crail_path"] = last_full_prompt_cache
                        if args.debug:
                            print(f"[调试] 从Crail加载上一轮KV Cache: {last_full_prompt_cache}")
                    else:
                        gen_kwargs["load_kv_cache_path"] = last_full_prompt_cache
                        if args.debug:
                            print(f"[调试] 从本地加载上一轮KV Cache: {last_full_prompt_cache}")

            # 设置KV Cache保存参数
            if args.use_crail:
                gen_kwargs["save_kv_cache_crail_path"] = current_cache_path
                if args.debug:
                    print(f"[调试] 将保存KV Cache到Crail: {current_cache_path}")
            else:
                gen_kwargs["save_kv_cache_path"] = current_cache_path
                if args.debug:
                    print(f"[调试] 将保存KV Cache到本地: {current_cache_path}")

            start_time = time.time()

            # 生成回复，使用KV缓存特性
            with torch.no_grad():
                response = model.generate(
                    current_full_prompt,
                    device,
                    output_len=args.output_len,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50,
                    **gen_kwargs
                )

            end_time = time.time()

            if args.debug:
                print(f"\n[调试] 轮次 {turn} 响应:\n{response}")
                print(f"[调试] 生成时间: {end_time - start_time:.2f} 秒")

            # 显示回复
            print(f"Gemma: {response}")

            # 保存助手回复
            conversation_history.append({"role": "assistant", "content": response})

            # 更新上一轮完整提示和缓存路径
            last_full_prompt = current_full_prompt
            last_full_prompt_hash = current_full_prompt_hash
            last_full_prompt_cache = current_cache_path

    except KeyboardInterrupt:
        print("\n对话已结束")

    print("感谢使用Gemma!")

if __name__ == "__main__":
    main()
