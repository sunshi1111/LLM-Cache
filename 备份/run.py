# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import random
import time
import hashlib

import numpy as np
import torch
from absl import app, flags

from gemma import config
from gemma import model as gemma_model

# 定义命令行参数
FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt', None, '模型检查点文件路径。', required=True)
flags.DEFINE_string('variant', '2b', '模型变体。')
flags.DEFINE_string('device', 'cpu', '运行模型的设备。')
flags.DEFINE_integer('output_len', 1024, '生成回复的最大长度。')
flags.DEFINE_integer('seed', 12345, '随机种子。')
flags.DEFINE_boolean('quant', False, '是否使用量化。')
flags.DEFINE_float('temperature', 1.0, '生成文本的温度参数')
flags.DEFINE_float('top_p', 0.95, 'Top-p 采样参数')
flags.DEFINE_integer('top_k', 64, 'Top-k 采样参数')
flags.DEFINE_string('system_prompt', "你是一个有用的AI助手。请提供有帮助、安全、准确的回答。", '系统提示词')
flags.DEFINE_boolean('debug', False, '是否显示调试信息')

# 定义有效的文本模型变体
_VALID_MODEL_VARIANTS = ['2b', '2b-v2', '7b', '9b', '27b', '1b']

# 定义有效设备
_VALID_DEVICES = ['cpu', 'cuda']

# 验证 'variant' 参数的函数
def validate_variant(variant):
    if variant not in _VALID_MODEL_VARIANTS:
        raise ValueError(f'无效的变体: {variant}。有效的变体有: {_VALID_MODEL_VARIANTS}')
    return True

# 验证 'device' 参数的函数
def validate_device(device):
    if device not in _VALID_DEVICES:
        raise ValueError(f'无效的设备: {device}。有效的设备有: {_VALID_DEVICES}')
    return True

# 为 'variant' 参数注册验证器
flags.register_validator('variant', validate_variant, message='无效的模型变体。')

# 为 'device' 参数注册验证器
flags.register_validator('device', validate_device, message='无效的设备。')

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """设置默认的torch dtype"""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def get_prompt_hash(prompt):
    """生成提示文本的哈希值，用于调试标识"""
    return hashlib.md5(prompt.encode('utf-8', errors='ignore')).hexdigest()[:12]

def main(_):
    # 构造模型配置
    model_config = config.get_model_config(FLAGS.variant)
    model_config.dtype = "float32" if FLAGS.device == "cpu" else "float16"
    model_config.quant = FLAGS.quant

    # 设置随机种子
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # 创建模型并加载权重
    print("正在加载模型...")
    device = torch.device(FLAGS.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(FLAGS.ckpt)
        model = model.to(device).eval()
    print("模型加载完成！")

    # 对话历史
    conversation_history = []
    system_prompt = FLAGS.system_prompt

    # 输出会话开始信息
    print("======================================")
    print("开始与 Gemma 模型对话，输入'exit'退出")
    print(f"系统: {system_prompt}")
    print("======================================\n")

    turn = 0
    try:
        while True:
            turn += 1
            user_input = input("用户: ")
            
            if user_input.lower() == "exit":
                break
                
            if user_input.strip() == "":
                continue

            # 添加用户输入到对话历史
            conversation_history.append({"role": "user", "content": user_input})

            # 构建完整的对话提示
            current_full_prompt = system_prompt
            for msg in conversation_history:
                if msg["role"] == "user":
                    current_full_prompt += f"\n\n用户: {msg['content']}"
                else:
                    current_full_prompt += f"\n\nGemma: {msg['content']}"
            # 添加最后的提示符
            current_full_prompt += "\n\nGemma:"

            # 显示调试信息
            if FLAGS.debug:
                prompt_hash = get_prompt_hash(current_full_prompt)
                print(f"\n[调试] 轮次 {turn}:")
                print(f"[调试] 提示哈希: {prompt_hash}")
                print(f"[调试] 完整提示: \n{current_full_prompt}")
                print(f"[调试] 生成参数: temperature={FLAGS.temperature}, top_p={FLAGS.top_p}, top_k={FLAGS.top_k}")

            # 计时并生成响应
            start_time = time.time()
            with torch.no_grad():
                response = model.generate(
                    current_full_prompt,
                    device,
                    output_len=FLAGS.output_len,
                    temperature=FLAGS.temperature,
                    top_p=FLAGS.top_p,
                    top_k=FLAGS.top_k
                )
            end_time = time.time()

            # 显示调试信息
            if FLAGS.debug:
                print(f"[调试] 生成时间: {end_time - start_time:.2f} 秒")
            
            # 显示模型回复
            print(f"Gemma: {response}\n")
            
            # 保存回复到对话历史
            conversation_history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\n对话已结束")

    print("感谢使用 Gemma!")

if __name__ == "__main__":
    app.run(main)