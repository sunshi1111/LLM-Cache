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
"""Inference-only Gemma 3 multimodal model implementation."""
import subprocess
import torch
import os
import time
import io
import sys
import json
import gc
from torch import nn
from PIL import Image
from typing import Any, List, Sequence, Tuple, Union, Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor



from . import model as gemma_model
from . import config as gemma_config
from . import gemma3_preprocessor
from . import tokenizer
from .siglip_vision import siglip_vision_model

class Gemma3ForMultimodalLM(nn.Module):
  """Gemma3 model for multimodal causal LM."""
  def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
    super().__init__()
    self.dtype = config.get_dtype()
    assert config.architecture == gemma_config.Architecture.GEMMA_3
    self.config = config
    max_seq_len = config.max_position_embeddings
    head_dim = config.head_dim
    vocab_size = config.vocab_size
    self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
    self.text_token_embedder = gemma_model.Embedding(vocab_size, config.hidden_size, config.quant)
    self.model = gemma_model.GemmaModel(config)
    self.sampler = gemma_model.Sampler(vocab_size, config)

    if config.vision_config is None:
      raise ValueError('vision_config must be provided for Gemma3.')
    self.siglip_vision_model = siglip_vision_model.SiglipVisionModel(config.vision_config)
    # transformer/embedder/mm_soft_embedding_norm
    self.mm_soft_embedding_norm = gemma_model.RMSNorm(config.vision_config.embedding_dim,
                                                           eps = config.rms_norm_eps)
    # transformer/embedder/mm_input_projection
    self.mm_input_projection = gemma_model.Linear(config.vision_config.embedding_dim, config.hidden_size, config.quant)

    if config.rope_wave_length is None:
      raise ValueError('rope_wave_length must be provided for Gemma3.')
    rope_lengths = config.rope_wave_length
    defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }
    self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
              gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]
          ))
    self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
              gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]
          ), rope_scaling_factor=config.rope_scaling_factor)
    self._kv_cache_save_executor = ThreadPoolExecutor(max_workers=2)
    self._active_save_tasks = []
    
  
  def save_kv_cache_async(self, kv_caches, file_path=None, crail_path=None, max_seq_len=None, 
                       prompt_len=None, cached_len=0, current_len=None, incremental=False):
    """异步保存KV缓存的函数，支持增量保存"""
    
    # 创建KV缓存的深度拷贝，以防在保存过程中被修改
    # 我们必须先将张量移动到CPU并分离
    detached_kv_caches = [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_caches]
    
    def _save_task():
        try:
            save_start_time = time.time()
            
            if incremental:
                # 增量保存模式
                if current_len is None:
                    raise ValueError("进行增量保存时必须指定current_len")
                
                if crail_path:
                    self.save_kv_cache_crail_incremental(
                        detached_kv_caches, crail_path, 
                        cached_len, current_len, 
                        max_seq_len, prompt_len
                    )
                elif file_path:
                    self.save_kv_cache_incremental(
                        detached_kv_caches, file_path,
                        cached_len, current_len,
                        max_seq_len, prompt_len
                    )
            else:
                # 全量保存模式
                if crail_path:
                    self.save_kv_cache_crail(detached_kv_caches, crail_path, max_seq_len, prompt_len)
                elif file_path:
                    self.save_kv_cache(detached_kv_caches, file_path, max_seq_len, prompt_len)
                
        except Exception as e:
            import traceback
            print(f"[后台] 异步保存KV缓存失败: {str(e)}")
            traceback.print_exc()
    
    # 提交保存任务到线程池执行器
    task = self._kv_cache_save_executor.submit(_save_task)
    self._active_save_tasks.append(task)
    
    # 清理已完成的任务
    self._active_save_tasks = [t for t in self._active_save_tasks if not t.done()]
    
    return task


  def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
    ):
    self.register_buffer(
            name, gemma_model.precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=rope_scaling_factor)
        )
  
  def load_kv_cache(self, file_path: str, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"找不到KV cache文件: {file_path}")

    try:
      # 加载保存的数据
      saved_data = torch.load(file_path, map_location='cpu')

      # 验证模型配置
      saved_config = saved_data.get("model_config", {})
      for key, value in saved_config.items():
        current_value = getattr(self.config, key, None)
        if current_value != value:
          print(f"警告: 加载的KV cache配置不匹配。{key}: 已保存={value}, 当前={current_value}")

      # 获取一些元数据，如果存在
      original_max_seq_len = saved_data.get("max_seq_len", None)
      original_prompt_len = saved_data.get("prompt_len", None)

      # 将KV cache移动到指定设备
      kv_caches = []
      for k, v in saved_data["kv_caches"]:
        # 添加到返回列表
        kv_caches.append((k.to(device), v.to(device)))

      print(f"成功从 {file_path} 加载KV cache，时间戳: {saved_data.get('timestamp')}")

      # 同时返回元数据信息，以便generate函数使用
      metadata = {
          "original_max_seq_len": original_max_seq_len,
          "original_prompt_len": original_prompt_len
      }

      return kv_caches, metadata

    except Exception as e:
      print(f"加载KV cache失败: {str(e)}")
      # 如果出错，返回None
      return None, {}

  def save_kv_cache(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
                 file_path: str, max_seq_len=None, prompt_len=None) -> None:
    try:
      # 创建目录(如果不存在)
      os.makedirs(os.path.dirname(file_path), exist_ok=True)

      # 构建保存数据
      save_data = {
          "kv_caches": [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_caches],
          "timestamp": time.time(),
          "max_seq_len": max_seq_len,  # 存储最大序列长度
          "prompt_len": prompt_len,    # 存储提示长度
          "model_config": {
              "hidden_size": self.config.hidden_size,
              "num_hidden_layers": self.config.num_hidden_layers,
              "num_attention_heads": self.config.num_attention_heads,
              "num_key_value_heads": self.config.num_key_value_heads,
              "head_dim": self.config.head_dim
          }
      }

      # 保存到文件
      with open(file_path, 'wb') as f:
        torch.save(save_data, f)

      
    except Exception as e:
      print(f"保存KV cache失败: {str(e)}")

  def save_kv_cache_crail(self, kv_caches, crail_path, max_seq_len=None, prompt_len=None):
    try:
      # 构建保存数据
      save_data = {
          "kv_caches": [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_caches],
          "timestamp": time.time(),
          "max_seq_len": max_seq_len, 
          "prompt_len": prompt_len,
          "model_config": {
              "hidden_size": self.config.hidden_size,
              "num_hidden_layers": self.config.num_hidden_layers,
              "num_attention_heads": self.config.num_attention_heads,
              "num_key_value_heads": self.config.num_key_value_heads,
              "head_dim": self.config.head_dim
          }
      }

      # 准备Java程序命令
      jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar")
      crail_conf_dir = os.environ.get("CRAIL_CONF_DIR", "/home/ms-admin/sunshi/crail/conf")
      
      cmd = [
          "java",
          "-Djava.library.path=/home/ms-admin/sunshi/crail/lib",
          f"-Dcrail.conf.dir={crail_conf_dir}",
          "-cp", f"{jar_path}:{crail_conf_dir}:/home/ms-admin/sunshi/crail/jars/*",
          "com.example.CrailKVCacheManager",
          "upload-stream",  # 使用流上传命令
          crail_path
      ]

      print(f"开始通过流将KV cache上传到Crail: {crail_path}")
      
      # 使用BytesIO在内存中序列化数据
      buffer = io.BytesIO()
      torch.save(save_data, buffer)
      buffer.seek(0)
      
      # 获取数据大小进行日志记录
      data_size = buffer.getbuffer().nbytes
      print(f"序列化后的KV cache大小: {data_size/1024/1024:.2f} MB")
      
      # 方法1：使用communicate一次性提供输入并等待进程完成
      process = subprocess.Popen(
          cmd,
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          bufsize=-1
      )
      
      stdout, stderr = process.communicate(input=buffer.getvalue())
      
      if process.returncode != 0:
        error_msg = stderr.decode("utf-8")
        print(f"Java进程stderr输出: {error_msg}")
        raise RuntimeError(f"上传到Crail失败，返回码: {process.returncode}")

      stderr_output = stderr.decode("utf-8")
      print(f"上传详情: {stderr_output.strip()}")
      print(f"KV cache已成功上传到Crail: {crail_path}")
      
    except Exception as e:
      print(f"保存KV cache到Crail失败: {str(e)}")
      import traceback
      traceback.print_exc()

  def load_kv_cache_crail(self, crail_path: str, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """使用内存流直接从Crail加载KV cache，不使用磁盘临时文件"""
    
    try:
      # 准备Java程序命令
      jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar")
      crail_conf_dir = os.environ.get("CRAIL_CONF_DIR", "/home/ms-admin/sunshi/crail/conf")
      
      cmd = [
          "java",
          "-Djava.library.path=/home/ms-admin/sunshi/crail/lib",
          f"-Dcrail.conf.dir={crail_conf_dir}",
          "-cp", f"{jar_path}:{crail_conf_dir}:/home/ms-admin/sunshi/crail/jars/*",
          "com.example.CrailKVCacheManager",
          "download-stream",  # 使用流下载命令
          crail_path
      ]

      print(f"开始从Crail通过流下载KV cache: {crail_path}")
      
      # 启动Java进程，设置stdout为PIPE，使用更大的缓冲区
      process = subprocess.Popen(
          cmd,
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          bufsize=32 * 1024 * 1024  # 增大缓冲区
      )

      # 使用bytearray替代字节串拼接
      buffer = bytearray()
      chunk_size = 16 * 1024 * 1024  # 增大到16MB
      bytes_read = 0
      start_time = time.time()
      last_report_time = start_time
      
      while True:
        chunk = process.stdout.read(chunk_size)
        if not chunk:
          break
        
        bytes_read += len(chunk)
        buffer.extend(chunk)  # 使用extend而非+=
      
        # 每秒更新一次进度，减少I/O开销
        current_time = time.time()
        if current_time - last_report_time >= 1.0:
          elapsed = current_time - start_time
          speed = bytes_read / (1024 * 1024 * elapsed) if elapsed > 0 else 0
          print(f"已接收: {bytes_read/1024/1024:.2f} MB, 速度: {speed:.2f} MB/s", end='\r')
          sys.stdout.flush()
          last_report_time = current_time
      
      # 等待Java进程完成并检查返回码
      stderr_output = process.stderr.read().decode("utf-8")
      return_code = process.wait()
      if return_code != 0:
        print(f"Java进程stderr输出: {stderr_output}")
        raise RuntimeError(f"从Crail下载失败，返回码: {return_code}")

      total_time = time.time() - start_time
      avg_speed = len(buffer) / (1024 * 1024 * total_time) if total_time > 0 else 0
      print(f"\n下载完成: {len(buffer)/1024/1024:.2f} MB, 平均速度: {avg_speed:.2f} MB/s")
      print(f"下载详情: {stderr_output.strip()}")
      
      # 在内存中加载数据
      buffer_io = io.BytesIO(buffer)
      saved_data = torch.load(buffer_io, map_location='cpu')
      
      # 验证模型配置
      saved_config = saved_data.get("model_config", {})
      for key, value in saved_config.items():
        current_value = getattr(self.config, key, None)
        if current_value != value:
          print(f"警告: 加载的KV cache配置不匹配。{key}: 已保存={value}, 当前={current_value}")

      # 获取一些元数据，如果存在
      original_max_seq_len = saved_data.get("max_seq_len", None)
      original_prompt_len = saved_data.get("prompt_len", None)

      # 将KV cache移动到指定设备
      kv_caches = []
      for k, v in saved_data["kv_caches"]:
        # 添加到返回列表
        kv_caches.append((k.to(device), v.to(device)))

      print(f"成功从Crail加载KV cache，时间戳: {saved_data.get('timestamp')}")

      # 同时返回元数据信息，以便generate函数使用
      metadata = {
          "original_max_seq_len": original_max_seq_len,
          "original_prompt_len": original_prompt_len
      }

      return kv_caches, metadata

    except Exception as e:
      print(f"从Crail加载KV cache失败: {str(e)}")
      import traceback
      traceback.print_exc()
      return None, {}
  def merge_incremental_kv_cache(self, base_kv_caches, incremental_file, device):
    """合并基础KV缓存和增量KV缓存"""
    try:
        # 加载增量缓存数据
        saved_data = torch.load(incremental_file, map_location='cpu')
        
        if not saved_data.get("is_incremental", False):
            print("提供的文件不是增量缓存文件，无需合并")
            return base_kv_caches, {}
        
        cached_len = saved_data.get("cached_len", 0)
        current_len = saved_data.get("current_len", 0)
        incremental_kv_caches = saved_data.get("incremental_kv_caches", [])
        
        # 验证基础缓存长度是否匹配
        for i, (base_k, base_v) in enumerate(base_kv_caches):
            if base_k.shape[1] < cached_len:
                raise ValueError(f"基础缓存长度({base_k.shape[1]})小于增量缓存的基础长度({cached_len})，无法合并")
        
        # 合并缓存
        merged_kv_caches = []
        for i, ((base_k, base_v), (inc_k, inc_v)) in enumerate(zip(base_kv_caches, incremental_kv_caches)):
            # 确保形状匹配
            batch, seq_len, num_heads, head_dim = base_k.shape
            inc_batch, inc_seq_len, inc_num_heads, inc_head_dim = inc_k.shape
            
            if (batch != inc_batch or num_heads != inc_num_heads or head_dim != inc_head_dim):
                raise ValueError(f"层{i}的基础缓存和增量缓存维度不匹配")
            
            # 如果需要，扩展基础缓存以适应增量缓存
            if seq_len < current_len:
                new_k = torch.zeros((batch, current_len, num_heads, head_dim), 
                                  dtype=base_k.dtype, device=device)
                new_v = torch.zeros((batch, current_len, num_heads, head_dim),
                                  dtype=base_v.dtype, device=device)
                
                # 复制基础缓存
                new_k[:, :seq_len, :, :] = base_k
                new_v[:, :seq_len, :, :] = base_v
                
                # 复制增量缓存部分
                new_k[:, cached_len:cached_len+inc_seq_len, :, :] = inc_k.to(device)
                new_v[:, cached_len:cached_len+inc_seq_len, :, :] = inc_v.to(device)
                
                merged_kv_caches.append((new_k, new_v))
            else:
                # 基础缓存足够大，直接更新相应部分
                base_k[:, cached_len:cached_len+inc_seq_len, :, :] = inc_k.to(device)
                base_v[:, cached_len:cached_len+inc_seq_len, :, :] = inc_v.to(device)
                merged_kv_caches.append((base_k, base_v))
        
        # 同时返回元数据信息
        metadata = {
            "original_max_seq_len": saved_data.get("max_seq_len"),
            "original_prompt_len": current_len,  # 更新为合并后的长度
            "is_merged": True
        }
        
        print(f"成功合并增量KV缓存，基础长度: {cached_len}，当前长度: {current_len}")
        
        return merged_kv_caches, metadata
        
    except Exception as e:
        print(f"合并增量KV缓存失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return base_kv_caches, {}
  def save_kv_cache_incremental(self, kv_caches, file_path, cached_len, current_len, max_seq_len=None, prompt_len=None):
    """保存KV缓存的增量部分（从cached_len到current_len的部分）"""
    try:
        # 创建目录(如果不存在)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 只保存增量部分
        incremental_kv_caches = []
        for k, v in kv_caches:
            # 提取增量部分 [batch, cached_len:current_len, heads, dim]
            inc_k = k[:, cached_len:current_len, :, :].detach().cpu()
            inc_v = v[:, cached_len:current_len, :, :].detach().cpu()
            incremental_kv_caches.append((inc_k, inc_v))

        # 构建保存数据
        save_data = {
            "incremental_kv_caches": incremental_kv_caches,
            "timestamp": time.time(),
            "max_seq_len": max_seq_len,  
            "prompt_len": prompt_len,
            "cached_len": cached_len,     # 原来已缓存的长度
            "current_len": current_len,   # 当前缓存的总长度
            "is_incremental": True,       # 标记这是增量保存
            "model_config": {
                "hidden_size": self.config.hidden_size,
                "num_hidden_layers": self.config.num_hidden_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "head_dim": self.config.head_dim
            }
        }

        # 保存到文件
        with open(file_path, 'wb') as f:
            torch.save(save_data, f)
        
        print(f"增量KV cache已保存到 {file_path}，增量大小: {current_len - cached_len} tokens")
      
    except Exception as e:
        print(f"保存增量KV cache失败: {str(e)}")
  def save_kv_cache_crail_incremental(self, kv_caches, crail_path, cached_len, current_len, max_seq_len=None, prompt_len=None):
    """通过流将KV缓存的增量部分保存到Crail"""
    try:
        # 只保存增量部分
        incremental_kv_caches = []
        for k, v in kv_caches:
            # 提取增量部分 [batch, cached_len:current_len, heads, dim]
            inc_k = k[:, cached_len:current_len, :, :].detach().cpu()
            inc_v = v[:, cached_len:current_len, :, :].detach().cpu()
            incremental_kv_caches.append((inc_k, inc_v))

        # 构建保存数据
        save_data = {
            "incremental_kv_caches": incremental_kv_caches,
            "timestamp": time.time(),
            "max_seq_len": max_seq_len,
            "prompt_len": prompt_len,
            "cached_len": cached_len,     # 原来已缓存的长度
            "current_len": current_len,   # 当前缓存的总长度
            "is_incremental": True,       # 标记这是增量保存
            "model_config": {
                "hidden_size": self.config.hidden_size,
                "num_hidden_layers": self.config.num_hidden_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "head_dim": self.config.head_dim
            }
        }

        # 准备Java程序命令
        jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar")
        crail_conf_dir = os.environ.get("CRAIL_CONF_DIR", "/home/ms-admin/sunshi/crail/conf")
        
        cmd = [
            "java",
            "-Djava.library.path=/home/ms-admin/sunshi/crail/lib",
            f"-Dcrail.conf.dir={crail_conf_dir}",
            "-cp", f"{jar_path}:{crail_conf_dir}:/home/ms-admin/sunshi/crail/jars/*",
            "com.example.CrailKVCacheManager",
            "upload-stream",  # 使用流上传命令
            crail_path
        ]

        print(f"开始通过流将增量KV cache上传到Crail: {crail_path}，增量大小: {current_len - cached_len} tokens")
        
        # 使用BytesIO在内存中序列化数据
        buffer = io.BytesIO()
        torch.save(save_data, buffer)
        buffer.seek(0)
        
        # 获取数据大小进行日志记录
        data_size = buffer.getbuffer().nbytes
        print(f"序列化后的增量KV cache大小: {data_size/1024/1024:.2f} MB")
        
        # 使用communicate一次性提供输入并等待进程完成
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=-1
        )
        
        stdout, stderr = process.communicate(input=buffer.getvalue())
        
        if process.returncode != 0:
            error_msg = stderr.decode("utf-8")
            print(f"Java进程stderr输出: {error_msg}")
            raise RuntimeError(f"上传到Crail失败，返回码: {process.returncode}")

        stderr_output = stderr.decode("utf-8")
        print(f"上传详情: {stderr_output.strip()}")
        print(f"增量KV cache已成功上传到Crail: {crail_path}")
        
    except Exception as e:
        print(f"保存增量KV cache到Crail失败: {str(e)}")
        import traceback
        traceback.print_exc()
  def load_kv_cache_with_incremental(self, file_path: str, device: torch.device):
    """加载KV缓存，支持增量缓存文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到KV cache文件: {file_path}")

    try:
        # 加载保存的数据
        saved_data = torch.load(file_path, map_location='cpu')
        
        # 检查是否是增量缓存
        is_incremental = saved_data.get("is_incremental", False)
        
        # 验证模型配置
        saved_config = saved_data.get("model_config", {})
        for key, value in saved_config.items():
            current_value = getattr(self.config, key, None)
            if current_value != value:
                print(f"警告: 加载的KV cache配置不匹配。{key}: 已保存={value}, 当前={current_value}")

        # 获取元数据
        original_max_seq_len = saved_data.get("max_seq_len", None)
        original_prompt_len = saved_data.get("prompt_len", None)
        
        if is_incremental:
            # 这是增量缓存
            cached_len = saved_data.get("cached_len", 0)
            current_len = saved_data.get("current_len", 0)
            incremental_kv_caches = saved_data.get("incremental_kv_caches", [])
            
            print(f"加载的是增量KV cache，原缓存长度: {cached_len}，当前长度: {current_len}")
            
            # 需要创建完整的KV缓存并将增量部分填充进去
            # 注意：这里需要基础缓存，如果没有基础缓存，将无法恢复完整的KV缓存
            raise NotImplementedError("加载增量缓存需要先加载基础缓存，请先加载基础缓存")
            
        else:
            # 这是完整缓存
            kv_caches = []
            for k, v in saved_data["kv_caches"]:
                # 添加到返回列表
                kv_caches.append((k.to(device), v.to(device)))

            print(f"成功从 {file_path} 加载完整KV cache，时间戳: {saved_data.get('timestamp')}")

        # 同时返回元数据信息，以便generate函数使用
        metadata = {
            "original_max_seq_len": original_max_seq_len,
            "original_prompt_len": original_prompt_len,
            "is_incremental": is_incremental
        }
        
        if is_incremental:
            metadata.update({
                "cached_len": cached_len,
                "current_len": current_len
            })

        return kv_caches, metadata

    except Exception as e:
        print(f"加载KV cache失败: {str(e)}")
        # 如果出错，返回None
        return None, {}
    
  @torch.no_grad()
  def forward(self,
              input_token_ids: torch.Tensor, # B x L
              image_patches: torch.Tensor, # B x N x C x H x W (3x896x896)
              image_presence_mask: torch.Tensor, # B x N
              input_positions: torch.Tensor,
              kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
              mask: torch.Tensor,
              output_positions: torch.Tensor,
              temperatures: Union[torch.Tensor, None],
              top_ps: torch.Tensor,
              top_ks: torch.Tensor,
              local_mask: torch.Tensor | None = None,
              cached_len: int = 0,
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = {}
    freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
        self.local_freqs_cis.index_select(0, input_positions)
    )
    freqs_cis[gemma_config.AttentionType.GLOBAL] = (
        self.global_freqs_cis.index_select(0, input_positions)
    )
    hidden_states = self.text_token_embedder(input_token_ids)
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
    hidden_states = hidden_states * normalizer
    if image_patches is not None and self.config.vision_config is not None:
      # the input has images
      B, N, C, H, W = image_patches.shape
      # Flatten and Pass to SiglipVisionModel, and apply SiglipVisionModel Exit
      flattened_input = image_patches.reshape(B * N, C, H, W)  # (B*N)xCxHxW
      image_embeddings = self.siglip_vision_model(flattened_input)  # (B*N)xUxD
      image_embeddings = self.mm_soft_embedding_norm(image_embeddings)  # (B*N) x U x D
      image_embeddings = self.mm_input_projection(image_embeddings)  # (B*N) x U x model_dim
      hidden_states = self.populate_image_embeddings(
          hidden_states.clone(),
          image_embeddings.clone(),
          input_token_ids.clone(),
          image_presence_mask.clone(),
      )

    kv_write_indices = input_positions

    hidden_states = self.model(
        hidden_states=hidden_states,
        freqs_cis=freqs_cis,
        kv_write_indices=kv_write_indices,
        kv_caches=kv_caches,
        mask=mask,
        local_mask=local_mask,
        cached_len=cached_len,
    )
    embedder_weight = self.text_token_embedder.weight
    if self.config.quant:
      embedder_weight = (
          embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1))

    next_tokens, logits = self.sampler(
        embedding=embedder_weight,
        hidden_states=hidden_states,
        output_positions=output_positions,
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
    )
    return next_tokens, logits

  def populate_image_embeddings(self,
                                hidden_states: torch.Tensor, # B x L x model_dim
                                image_embeddings: torch.Tensor, # (B*N) x U x model_dim
                                input_token_ids: torch.Tensor, # B x L
                                image_presence_mask: torch.Tensor, # B x N
                                ):
    batch_size, seq_len, model_dim = hidden_states.shape
    # Step 1 of 2: Fetch valid image embeddings
    # flatten indices of valid image embeddings
    valid_image_embeddings_indices = torch.nonzero(image_presence_mask.flatten(), as_tuple=False).squeeze()
    # num_valid_images x model_dim
    valid_image_embeddings = image_embeddings.index_select(0, valid_image_embeddings_indices)

    # Step 2 of 2: Replace image embeddings at right places.
    image_placeholder_mask = input_token_ids == self.tokenizer.image_token_placeholder_id
    image_placeholder_indices = image_placeholder_mask.flatten().nonzero(as_tuple=False).squeeze()

    hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
    hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(-1, self.config.hidden_size)
    return hidden_states.reshape(batch_size, seq_len, model_dim).contiguous()

  def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int):
    batch_size = input_ids.shape[0]
    causal_mask = torch.tril(torch.ones((batch_size, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device))
    image_token_mask = input_ids == self.tokenizer.image_token_placeholder_id
    # Pad the mask to the left with 0. This is to make sure the boundary
    # detection works correctly. Boundary (starting index of image patch) is
    # detected when the value changes from 0 to 1.
    padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
    # Find the boundary (starting index) of the image tokens patch.
    boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
    # Number the boundary.
    # boundary:
    # [[False, False,  True, False, False],
    #  [False,  True, False,  True, False]]
    # numbered_boundary:
    # [[0, 0, 1, 1, 1],
    #  [0, 1, 1, 2, 2]]
    numbered_boundary = torch.cumsum(boundary, dim=-1)

    # image_token_mask:
    # [[False, False,  True,  True, False],
    #  [True,  True, False,  True, True]]
    # numbered_boundary:
    # [[0, 0, 1, 1, 1],
    #  [1, 1, 1, 2, 2]]
    # q_block_indices:
    # [[0, 0, 1, 1, 0],
    #  [1, 1, 0, 2, 2]]
    q_block_indices = image_token_mask * numbered_boundary
    kv_block_indices = q_block_indices
    # Test the equality of vertical and horizontal numbered patches
    # to create the bidirectional mask.
    bidirectional_mask = torch.logical_and(
        kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
        q_block_indices.unsqueeze(-1) > 0,
    )
    attention_mask = torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
    # The upper triangular matrix's diagonal is shifted by sliding window size
    # before doing logical 'and' with attention mask. This is to make sure the
    # local attention is within the sliding window.
    local_mask = torch.logical_and(
        attention_mask,
        torch.triu(torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device), diagonal=-(self.config.sliding_window_size-1))
    )
    return attention_mask, local_mask

  def generate(
        self,
        prompts: Sequence[Sequence[Union[str, Image.Image]]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        save_kv_cache_path: Optional[str] = None,
        save_kv_cache_crail_path: Optional[str] = None,
        load_kv_cache_path: Optional[str] = None,
        load_kv_cache_crail_path: Optional[str] = None,
    ) -> Sequence[str]:
    """Generates responses for given prompts using Gemma model."""
    # Inference only.
    timing = {}
    timing['start'] = time.time()

    print("│ 1. 对提示进行tokenize...                                   │")
    tokenize_start = time.time()
    processing_result = gemma3_preprocessor.tokenize_raw_input(
        self.tokenizer, prompts, self.config, output_len, device
    )
    batch_size = processing_result["batch_size"]
    user_input_token_ids = processing_result["user_input_token_ids"]
    image_batch = processing_result["image_batch"]
    min_prompt_len = processing_result["min_prompt_len"]
    max_prompt_len = processing_result["max_prompt_len"]
    total_seq_len = processing_result["max_seq_len"]
    image_presence_mask = processing_result["image_presence_mask"]

    # Create attention mask.
    min_dtype = torch.finfo(self.dtype).min
    if self.config.sliding_window_size is None:
      raise ValueError('gemma 3 model requires sliding_window size')
    boolean_mask, local_boolean_mask = self.create_attention_mask(user_input_token_ids, total_seq_len)
    mask_tensor = torch.where(boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()
    local_mask_tensor = torch.where(local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()
    tokenize_end = time.time()
    timing['1_tokenize'] = tokenize_end - tokenize_start
    print(f"│ - Tokenize耗时: {timing['1_tokenize']:.4f}秒               │")
    print(f"│ - 批次大小: {batch_size}, 最大提示长度: {max_prompt_len}      │")
    # 2. 加载KV缓存和准备前向传播
    print("│ 2. 准备模型前向传播...                                     │")
    forward_prep_start = time.time()
    
    # 尝试加载KV cache
    kv_caches = None
    kv_cache_metadata = {}
    force_regenerate = False  # 标记是否需要强制重新生成

    # 加载KV缓存的尝试
    if load_kv_cache_crail_path:
      try:
        print(f"│ - 从Crail加载KV缓存: {load_kv_cache_crail_path}          │")
        kv_cache_load_start = time.time()
        kv_caches, kv_cache_metadata = self.load_kv_cache_crail(load_kv_cache_crail_path, device)
        kv_cache_load_end = time.time()
        timing['kv_cache_load'] = kv_cache_load_end - kv_cache_load_start
        
        if kv_caches is None:
          print(f"│ ! 无法从Crail加载KV cache，将重新生成                   │")
          force_regenerate = True
        else:
          print(f"│ - KV缓存加载耗时: {timing['kv_cache_load']:.4f}秒          │")
      except Exception as e:
        print(f"│ ! 从Crail加载KV cache失败: {str(e)[:30]}...             │")
        force_regenerate = True
    elif load_kv_cache_path:
      try:
        print(f"│ - 从本地加载KV缓存: {load_kv_cache_path}               │")
        kv_cache_load_start = time.time()
        kv_caches, kv_cache_metadata = self.load_kv_cache(load_kv_cache_path, device)
        kv_cache_load_end = time.time()
        timing['kv_cache_load'] = kv_cache_load_end - kv_cache_load_start
        
        if kv_caches is None:
          print(f"│ ! 无法从本地文件加载KV cache，将重新生成                │")
          force_regenerate = True
        else:
          print(f"│ - KV缓存加载耗时: {timing['kv_cache_load']:.4f}秒          │")
      except Exception as e:
        print(f"│ ! 从本地文件加载KV cache失败，将重新生成                  │")
        force_regenerate = True

    # 3. 构建与调整KV cache
    print("│ 3. 构建与调整KV cache...                                  │")
    kv_build_start = time.time()
    
    # 如果没有成功加载KV cache，或者加载的缓存维度不匹配，则创建新的
    if kv_caches is None or force_regenerate:
      print("│ - 创建新的KV cache...                                   │")
      kv_caches = []
      for _ in range(self.config.num_hidden_layers):
        size = (batch_size, total_seq_len, self.config.num_key_value_heads,
                self.config.head_dim)
        dtype = self.config.get_dtype()
        k_cache = torch.zeros(size=size, dtype=dtype, device=device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=device)
        kv_caches.append((k_cache, v_cache))
    else:
      # 验证并调整KV cache的尺寸以匹配当前序列长度
      print("│ - 调整KV cache维度...                                   │")
      adjusted_kv_caches = []

      for layer_idx, (k_cache, v_cache) in enumerate(kv_caches):
        batch, seq_len, num_heads, head_dim = k_cache.shape

        # 如果维度不匹配，调整KV缓存
        if seq_len != total_seq_len:
          print(f"│ - 调整第{layer_idx}层KV cache: {seq_len} -> {total_seq_len}     │")

          # 创建新的缓存张量
          dtype = k_cache.dtype
          new_k = torch.zeros((batch, total_seq_len, num_heads, head_dim),
                            dtype=dtype, device=device)
          new_v = torch.zeros((batch, total_seq_len, num_heads, head_dim),
                            dtype=dtype, device=device)

          # 复制数据，处理尺寸不匹配
          copy_len = min(seq_len, total_seq_len)
          new_k[:, :copy_len, :, :] = k_cache[:, :copy_len, :, :]
          new_v[:, :copy_len, :, :] = v_cache[:, :copy_len, :, :]

          adjusted_kv_caches.append((new_k, new_v))
        else:
          adjusted_kv_caches.append((k_cache, v_cache))

      kv_caches = adjusted_kv_caches

    cached_len = 0  # 默认为0，表示没有缓存

    if kv_caches is not None and 'original_prompt_len' in kv_cache_metadata:
      cached_len = kv_cache_metadata.get('original_prompt_len', 0)
      print(f"│ - 已加载KV缓存，包含{cached_len}个token                      │")

    kv_build_end = time.time()
    timing['3_kv_build'] = kv_build_end - kv_build_start
    print(f"│ - KV缓存构建/调整耗时: {timing['3_kv_build']:.4f}秒              │")
    
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                        self.tokenizer.pad_id,
                                        dtype=torch.int64, device=device)
    token_ids_tensor = user_input_token_ids.to(device)
    for i in range(batch_size):
      p = user_input_token_ids[i]
      input_token_ids_tensor[i, :min_prompt_len] = p[:min_prompt_len]

    input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)
    prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperature else torch.FloatTensor(
        [temperature] * batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

    forward_prep_end = time.time()
    timing['2_forward_prep'] = forward_prep_end - forward_prep_start
    print(f"│ - 总前向传播准备耗时: {timing['2_forward_prep']:.4f}秒            │")
    # 记录开始时间
    first_token_start = time.time()
    generation_start = time.time()
    # Prefill up to min_prompt_len tokens, then treat other prefill as
    # decode and ignore output.
    for i in range(total_seq_len - min_prompt_len):
      next_token_ids, _ = self(
        input_token_ids=input_token_ids_tensor,
        image_patches=image_batch,
        image_presence_mask=image_presence_mask,
        input_positions=input_positions_tensor,
        kv_caches=kv_caches,
        mask=curr_mask_tensor,
        output_positions=output_positions_tensor,
        temperatures=temperatures_tensor,
        top_ps=top_ps_tensor,
        top_ks=top_ks_tensor,
        local_mask=curr_local_mask_tensor,
        cached_len=cached_len,
      )
      # 记录第一个token的时间
      if i == 0:
        first_token_end = time.time()
        timing['first_token'] = first_token_end - first_token_start
        print(f"│ - 第一个token生成耗时: {timing['first_token']:.4f}秒          │")
        
      curr_prompt_mask = prompt_mask_tensor.index_select(
        1, output_index).squeeze(dim=1)
      curr_token_ids = token_ids_tensor.index_select(
        1, output_index).squeeze(dim=1)
      output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                     next_token_ids).unsqueeze(dim=1)
      token_ids_tensor.index_copy_(1, output_index, output_token_ids)

      input_token_ids_tensor = output_token_ids
      input_positions_tensor = output_index.unsqueeze(dim=-1)
      curr_mask_tensor = mask_tensor.index_select(2,
                                                  input_positions_tensor)
      curr_local_mask_tensor = local_mask_tensor.index_select(
        2, input_positions_tensor
      ) if local_mask_tensor is not None else None
      output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
      output_index = output_index + 1
      image_batch = None
      image_presence_mask = None
    # 每生成10个token打印一次进度
      if i % 10 == 0 and i > 0:
        print(f"│ - 已生成 {i}/{total_seq_len - min_prompt_len} tokens                 │")
                    
                # 如果所有序列都结束了，提前退出循环
      if all(next_token_ids == self.tokenizer.eos_id):
        print("│ - 所有序列已生成完成，提前结束                          │")
        break
    generation_end = time.time()
    timing['4_generation'] = generation_end - generation_start
    
    print(f"│ - 总生成循环耗时: {timing['4_generation']:.4f}秒                │")
    
    
    
    
    
    
    decode_start = time.time()        
    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
      output = tokens
      if self.tokenizer.eos_id in output:
        eos_index = output.index(self.tokenizer.eos_id)
        output = output[:eos_index]
      results.append(self.tokenizer.decode(output))
    

    
    

    decode_end = time.time()
    timing['decode'] = decode_end - decode_start
    print(f"│ - 解码耗时: {timing['decode']:.4f}秒                          │")
    timing['total'] = time.time() - timing['start']


    print("│                                                           │")
    print("│ 性能分析总结:                                             │")
    #print(f"│ - 1. Tokenize:   {timing['1_tokenize']:.4f}秒 ({timing['1_tokenize']/timing['total']*100:.1f}%)          │")
    #print(f"│ - 2. 前向传播准备:{timing['2_forward_prep']:.4f}秒 ({timing['2_forward_prep']/timing['total']*100:.1f}%)          │")
    #print(f"│ - 3. KV缓存构建: {timing['3_kv_build']:.4f}秒 ({timing['3_kv_build']/timing['total']*100:.1f}%)          │")
    #print(f"│ - 4. 生成循环:   {timing['4_generation']:.4f}秒 ({timing['4_generation']/timing['total']*100:.1f}%)          │")
    #print(f"│ - 总计:         {timing['total']:.4f}秒 (100%)                       │")
    #print("└─────────────────────────────────────────────────────────────┘")
    from gemma.model import TOTAL_SECOND_LOOP_DURATION
    print(f"│ - 第一个token生成耗时: {TOTAL_SECOND_LOOP_DURATION:.6f}秒")

    if save_kv_cache_crail_path or save_kv_cache_path:
        
        
    # 是否进行增量保存
        if 'original_prompt_len' in kv_cache_metadata:
            # 我们有已加载的KV缓存，只保存增量部分
            original_len = kv_cache_metadata.get('original_prompt_len', 0)
            current_len = output_index.item()  # 当前生成到的位置
            
            print(f"│ - 进行增量保存，原始长度: {original_len}, 当前长度: {current_len}  │")
            
            self.save_kv_cache_async(
                kv_caches,
                file_path=save_kv_cache_path,
                crail_path=save_kv_cache_crail_path,
                max_seq_len=total_seq_len,
                prompt_len=max_prompt_len,
                cached_len=original_len,
                current_len=current_len,
                incremental=True
            )
        else:
            # 全量保存
            print("│ - 进行全量KV缓存保存                                  │")
            self.save_kv_cache_async(
                kv_caches,
                file_path=save_kv_cache_path,
                crail_path=save_kv_cache_crail_path,
                max_seq_len=total_seq_len,
                prompt_len=max_prompt_len
            )




    


    return results

  def load_weights(self, model_path: str):
    if os.path.isfile(model_path):
      self.load_state_dict(
        torch.load(
          model_path, mmap=True, weights_only=True,
        )['model_state_dict'],
        strict=False,
      )
    else:
      index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
      with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
      shard_files = list(set(index["weight_map"].values()))
      for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        del state_dict  # Save memory.
        gc.collect()