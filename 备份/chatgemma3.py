import contextlib
import random
import os
import time
import hashlib
import sys
import io

from absl import app
from absl import flags
import numpy as np
import torch

from gemma import config
from gemma import gemma3_model
from gemma.model import reset_timer, TOTAL_SECOND_LOOP_DURATION

# Define flags
FLAGS = flags.FLAGS

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to the checkpoint file.', required=True
)
_VARIANT = flags.DEFINE_string('variant', '4b', 'Model variant.')
_DEVICE = flags.DEFINE_string('device', 'cpu', 'Device to run the model on.')
_OUTPUT_LEN = flags.DEFINE_integer(
    'output_len', 50, 'Length of the output sequence.'
)
_SEED = flags.DEFINE_integer('seed', 12345, 'Random seed.')
_QUANT = flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')
_INTERACTIVE = flags.DEFINE_boolean('interactive', True, 'Enable interactive conversation mode.')
_DEBUG = flags.DEFINE_boolean('debug', False, '显示调试信息')

# KV缓存相关参数
_CACHE_DIR = flags.DEFINE_string('cache_dir', './gemma3_cache', '保存KV缓存的本地目录')
_NO_CACHE = flags.DEFINE_boolean('no_cache', False, '禁用KV缓存功能')
_CACHE_ID = flags.DEFINE_string('cache_id', None, '指定KV缓存标识符，若不指定则使用随机生成的ID')

# Crail相关参数
_USE_CRAIL = flags.DEFINE_boolean('use_crail', False, '使用Crail存储KV缓存')
_CRAIL_CACHE_DIR = flags.DEFINE_string('crail_cache_dir', '/kvcache', 'Crail中存储KV缓存的目录')
_CRAIL_JAR = flags.DEFINE_string('crail_jar', 
                              '/home/ms-admin/sunshi/crail-example/target/crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar',
                              'Crail客户端JAR路径')
_CRAIL_CONF = flags.DEFINE_string('crail_conf', 
                               '/home/ms-admin/sunshi/crail/conf',
                               'Crail配置目录路径')

# Define valid model variants
_VALID_MODEL_VARIANTS = ['4b', '12b', '27b_v3']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']


# Validator function for the 'variant' flag
def validate_variant(variant):
  if variant not in _VALID_MODEL_VARIANTS:
    raise ValueError(
        f'Invalid variant: {variant}. Valid variants are:'
        f' {_VALID_MODEL_VARIANTS}'
    )
  return True


# Validator function for the 'device' flag
def validate_device(device):
  if device not in _VALID_DEVICES:
    raise ValueError(
        f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}'
    )
  return True


# Register the validator for the 'variant' flag
flags.register_validator(
    'variant', validate_variant, message='Invalid model variant.'
)

# Register the validator for the 'device' flag
flags.register_validator('device', validate_device, message='Invalid device.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)


def format_message(role, content):
  """Format a message for the model."""
  return f"<start_of_turn>{role} {content}<end_of_turn>\n"


def get_prompt_hash(prompt):
  """生成提示文本的哈希值，用于缓存标识"""
  return hashlib.md5(prompt.encode('utf-8', errors='ignore')).hexdigest()[:12]


def interactive_chat(model, device, output_len=50, use_cache=True, cache_dir=None, 
                    use_crail=False, crail_cache_dir=None, cache_id=None, debug=False):
  """运行交互式对话，支持KV缓存复用，并自动删除过期缓存"""
  print("\n===== 开始交互式对话 =====")
  print("输入 'exit' 或 'quit' 结束对话")
  reset_timer()
  # 创建缓存目录
  if use_cache and not use_crail and cache_dir:
    os.makedirs(cache_dir, exist_ok=True)
  
  # 会话标识符
  session_id = cache_id or int(time.time())
  
  # 对话历史
  conversation_history = ""
  
  # 缓存路径历史记录（保留最近两轮）
  kv_cache_paths = []
  
  # 最近的KV缓存路径和提示
  last_kv_cache_path = None
  last_full_prompt = None
  last_full_prompt_hash = None
  
  turn = 0
  
  while True:
    turn += 1
    user_input = input("\n用户: ")
    if user_input.lower() in ["exit", "quit"]:
      print("对话结束")
      break
    if turn > 1:  # 第一轮已经在函数开始时重置过了
      reset_timer()
    # 构建当前完整对话
    current_full_prompt = conversation_history
    current_full_prompt += format_message("user", user_input)
    current_full_prompt += "<start_of_turn>model"
    
    # 计算当前提示的哈希值
    current_full_prompt_hash = get_prompt_hash(current_full_prompt)
    
    # 构建缓存路径
    if use_crail:
      current_cache_path = f"{crail_cache_dir}/session_{session_id}_turn_{turn}_{current_full_prompt_hash}.pt"
    else:
      current_cache_path = os.path.join(cache_dir, f"session_{session_id}_turn_{turn}_{current_full_prompt_hash}.pt") if cache_dir else None
    
    if debug:
      print(f"\n[调试] 轮次 {turn} - 完整对话提示")
      print(f"[调试] 提示哈希: {current_full_prompt_hash}")
      if use_cache:
        print(f"[调试] 当前缓存路径: {current_cache_path}")
    
    # 生成参数
    gen_kwargs = {}
    
    # 设置KV缓存加载和保存路径
    if use_cache:
      # 尝试复用上一轮缓存
      if turn > 1 and last_kv_cache_path and last_full_prompt:
        # 找到上一轮提示与当前提示的公共部分长度
        common_prefix_len = 0
        for i in range(min(len(last_full_prompt), len(current_full_prompt))):
          if last_full_prompt[i] == current_full_prompt[i]:
            common_prefix_len += 1
          else:
            break
            
        if debug and common_prefix_len > 0:
          common_prefix = current_full_prompt[:common_prefix_len]
          print(f"[调试] 发现公共前缀: {common_prefix_len} 字符")
          print(f"[调试] 公共前缀结束于: \"{common_prefix[-20:] if len(common_prefix) >= 20 else common_prefix}\"")
        
        if common_prefix_len > 0:
          if use_crail:
            gen_kwargs["load_kv_cache_crail_path"] = last_kv_cache_path
            if debug:
              print(f"[调试] 从Crail加载上一轮KV Cache: {last_kv_cache_path}")
          else:
            gen_kwargs["load_kv_cache_path"] = last_kv_cache_path
            if debug:
              print(f"[调试] 从本地加载上一轮KV Cache: {last_kv_cache_path}")
      
      # 保存本轮缓存
      if use_crail:
        gen_kwargs["save_kv_cache_crail_path"] = current_cache_path
        if debug:
          print(f"[调试] 将保存KV Cache到Crail: {current_cache_path}")
      else:
        gen_kwargs["save_kv_cache_path"] = current_cache_path
        if debug:
          print(f"[调试] 将保存KV Cache到本地: {current_cache_path}")
    
    # 生成回复
    start_time = time.time()
    result = model.generate(
      [[current_full_prompt]],
      device,
      output_len=output_len,
      temperature=0.7,
      top_p=0.95,
      top_k=50,
      **gen_kwargs
    )
    end_time = time.time()
    
    # 提取模型回复
    model_response = result[0].split("<start_of_turn>model")[-1].strip()
    print(f"\n助手: {model_response}")
    
    if debug:
      print(f"[调试] 生成时间: {end_time - start_time:.2f} 秒")
    
    # 更新对话历史
    conversation_history += format_message("user", user_input)
    conversation_history += format_message("model", model_response)
    
    # 更新缓存路径历史
    if use_cache and not use_crail and current_cache_path:
      kv_cache_paths.append(current_cache_path)
      
      # 删除上上轮缓存（保留最近两轮）
      if len(kv_cache_paths) > 2:
        old_cache_path = kv_cache_paths.pop(0)  # 移除并获取最旧的缓存路径
        try:
          if os.path.exists(old_cache_path):
            os.remove(old_cache_path)
            if debug:
              print(f"[调试] 已删除旧缓存: {old_cache_path}")
        except Exception as e:
          if debug:
            print(f"[调试] 删除旧缓存失败: {str(e)}")
    
    # 更新最近的缓存路径和提示
    last_kv_cache_path = current_cache_path
    last_full_prompt = current_full_prompt
    last_full_prompt_hash = current_full_prompt_hash

def main(_):
  # Construct the model config.
  model_config = config.get_model_config(_VARIANT.value)
  model_config.dtype = 'float32'
  model_config.quant = _QUANT.value

  # Seed random.
  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)

  # 设置KV缓存选项
  use_cache = not _NO_CACHE.value
  use_crail = _USE_CRAIL.value
  cache_dir = _CACHE_DIR.value if use_cache and not use_crail else None
  crail_cache_dir = _CRAIL_CACHE_DIR.value if use_cache and use_crail else None
  cache_id = _CACHE_ID.value
  debug = _DEBUG.value
  
  # 设置Crail环境变量
  if use_crail:
    os.environ["CRAIL_KVCACHE_JAR"] = _CRAIL_JAR.value
    os.environ["CRAIL_CONF_DIR"] = _CRAIL_CONF.value
  
  # 缓存信息提示
  if use_cache:
    if use_crail:
      print(f"Crail KV缓存已启用，存储目录: {crail_cache_dir}")
    else:
      print(f"本地KV缓存已启用，存储目录: {cache_dir}")
      os.makedirs(cache_dir, exist_ok=True)

  # Create the model and load the weights.
  device = torch.device(_DEVICE.value)
  with _set_default_tensor_type(model_config.get_dtype()):
    model = gemma3_model.Gemma3ForMultimodalLM(model_config)
    #model.load_state_dict(torch.load(_CKPT.value)['model_state_dict'])
    model.load_weights(_CKPT.value)
    
    model = model.to(device).eval()
  print('Model loading done')

  if _INTERACTIVE.value:
    # 启动交互式对话模式，传递KV缓存参数
    interactive_chat(
      model, 
      device, 
      _OUTPUT_LEN.value, 
      use_cache=use_cache,
      cache_dir=cache_dir,
      use_crail=use_crail,
      crail_cache_dir=crail_cache_dir,
      cache_id=cache_id,
      debug=debug
    )
  else:
    # 常规测试样例（非交互式）
    # Generate text only.
    result = model.generate(
        [
            [
                '<start_of_turn>user The capital of Italy'
                ' is?<end_of_turn>\n<start_of_turn>model'
            ],
            [
                '<start_of_turn>user What is your'
                ' purpose?<end_of_turn>\n<start_of_turn>model'
            ],
        ],
        device,
        output_len=_OUTPUT_LEN.value,
    )

    # Print the results.
    print('======================================')
    print(f'Text only RESULT: {result}')
    print('======================================')


if __name__ == '__main__':
  app.run(main)