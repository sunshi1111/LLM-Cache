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

from absl import app
from absl import flags
import numpy as np
import torch

from gemma import config
from gemma import gemma3_model1
from gemma.model1 import reset_timer, TOTAL_SECOND_LOOP_DURATION

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


def interactive_chat(model, device, output_len=50):
  """Run an interactive chat session with the model."""
  print("\n===== 开始交互式对话 =====")
  print("输入 'exit' 或 'quit' 结束对话")
  
  conversation_history = ""
  turn = 0
  while True:
    turn += 1
    user_input = input("\n用户: ")
    if user_input.lower() in ["exit", "quit"]:
      print("对话结束")
      break
    if turn > 1:  # 第一轮已经在函数开始时重置过了
      reset_timer()
    # 将用户输入添加到对话历史
    conversation_history += format_message("user", user_input)
    conversation_history += "<start_of_turn>model"
    
    # 生成回复
    result = model.generate(
        [[conversation_history]],
        device,
        output_len=output_len
    )
    
    # 提取模型回复
    model_response = result[0].split("<start_of_turn>model")[-1].strip()
    print(f"\n助手: {model_response}")
    
    # 更新对话历史
    conversation_history = conversation_history.replace("<start_of_turn>model", "")
    conversation_history += format_message("model", model_response)


def main(_):
  # Construct the model config.
  model_config = config.get_model_config(_VARIANT.value)
  model_config.dtype = 'float32'
  model_config.quant = _QUANT.value

  # Seed random.
  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)

  # Create the model and load the weights.
  device = torch.device(_DEVICE.value)
  with _set_default_tensor_type(model_config.get_dtype()):
    model = gemma3_model1.Gemma3ForMultimodalLM(model_config)
    model.load_state_dict(torch.load(_CKPT.value)['model_state_dict'])
    # model.load_weights(_CKPT.value)
    model = model.to(device).eval()
  print('Model loading done')

  if _INTERACTIVE.value:
    # 启动交互式对话模式
    interactive_chat(model, device, _OUTPUT_LEN.value)
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