import subprocess
import io
import json
import gc
import os
import time
import sys
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union

from gemma import config as gemma_config
from gemma import tokenizer


class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_logit_softcapping: Optional[float],
        query_pre_attn_scalar: Optional[int],
        head_dim: int,
        quant: bool,
        attn_type: gemma_config.AttentionType,
        sliding_window_size: Optional[int] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=quant)
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            quant=quant)

        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
        ):
            all_ones = torch.ones_like(mask)
            sliding_mask = torch.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=gemma_config.AttentionType.GLOBAL,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=attn_type,
            sliding_window_size=config.sliding_window_size,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(config))
            elif config.architecture == gemma_config.Architecture.GEMMA_2:
                attn_type = (
                    config.attn_types[i]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config)
        self.sampler = Sampler(vocab_size, config)

        # Pre-compute rotary embedding table.
        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = precompute_freqs_cis(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)


    def load_kv_cache(self,file_path: str, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:

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

    def save_kv_cache(self,kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
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

            print(f"KV cache已保存到: {file_path}")
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


    def load_kv_cache_crail(self,crail_path: str, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """使用内存流直接从Crail加载KV cache，不使用磁盘临时文件"""
        
        try:
            # 准备Java程序命令
            jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar")
            crail_conf_dir = os.environ.get("CRAIL_CONF_DIR", "/home/ms-admin/sunshi/crail/conf")
            
            cmd = [
                "java",
                f"-Dcrail.conf.dir={crail_conf_dir}",
                "-cp", f"{jar_path}:{crail_conf_dir}:/home/ms-admin/sunshi/crail/jars/*",
                "com.example.CrailKVCacheManager",
                "download-stream",  # 使用流下载命令
                crail_path
            ]

            print(f"开始从Crail通过流下载KV cache: {crail_path}")
            
            # 启动Java进程，设置stdout为PIPE
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=-1  # 使用系统默认缓冲
            )

            # 收集来自Java进程的标准输出流
            data = b''
            chunk_size = 8 * 1024 * 1024  # 8MB chunks
            bytes_read = 0
            
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                    
                bytes_read += len(chunk)
                data += chunk
                print(f"已接收: {bytes_read/1024/1024:.2f} MB", end='\r')
                sys.stdout.flush()
            
            # 等待Java进程完成并检查返回码
            stderr_output = process.stderr.read().decode("utf-8")
            return_code = process.wait()
            if return_code != 0:
                print(f"Java进程stderr输出: {stderr_output}")
                raise RuntimeError(f"从Crail下载失败，返回码: {return_code}")

            print(f"\n下载详情: {stderr_output.strip()}")
            print(f"成功从Crail接收数据，大小: {len(data)/1024/1024:.2f} MB")
            
            # 在内存中加载数据
            buffer = io.BytesIO(data)
            saved_data = torch.load(buffer, map_location='cpu')
            
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
    
   
        
    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = self.freqs_cis.index_select(0, input_positions)
        kv_write_indices = input_positions

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
        )
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    
    
    
    
    
    def generate(
            self,
            prompts: Union[str, Sequence[str]],
            device: Any,
            output_len: int = 100,
            temperature: Union[float, None] = 0.95,
            top_p: float = 1.0,
            top_k: int = 100,
        ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model with detailed timing."""
        import time
        
        # 创建计时字典
        timing = {}
        timing['start'] = time.time()
        
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│ 开始生成过程...                                            │")
        
        # 1. 对提示进行tokenize
        print("│ 1. 对提示进行tokenize...                                   │")
        tokenize_start = time.time()
        
        # 如果提供了单个提示，将其视为批次大小为1
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings
        
        tokenize_end = time.time()
        timing['1_tokenize'] = tokenize_end - tokenize_start
        print(f"│ - Tokenize耗时: {timing['1_tokenize']:.4f}秒               │")
        print(f"│ - 批次大小: {batch_size}, 最大提示长度: {max_prompt_len}      │")
        
        # 2. 准备前向传播
        print("│ 2. 准备模型前向传播...                                     │")
        forward_prep_start = time.time()
        
        # 3. 构建KV cache
        print("│ 3. 构建KV cache...                                  │")
        kv_build_start = time.time()
        
        # 创建新的KV cache
        print("│ - 创建新的KV cache...                                   │")
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))
            
        kv_build_end = time.time()
        timing['3_kv_build'] = kv_build_end - kv_build_start
        print(f"│ - KV缓存构建耗时: {timing['3_kv_build']:.4f}秒              │")
        
        # 准备tensor
        print("│ - 准备输入张量...                                        │")
        tensor_prep_start = time.time()
        
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                    self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                            dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)
            
        tensor_prep_end = time.time()
        timing['tensor_prep'] = tensor_prep_end - tensor_prep_start
        print(f"│ - 张量准备耗时: {timing['tensor_prep']:.4f}秒                   │")
        
        forward_prep_end = time.time()
        timing['2_forward_prep'] = forward_prep_end - forward_prep_start
        print(f"│ - 总前向传播准备耗时: {timing['2_forward_prep']:.4f}秒            │")
        
        # 4. 执行生成循环
        print("│ 4. 执行生成循环...                                        │")
        generation_start = time.time()
        first_token_start = time.time()
        
        # 预填充阶段计时
        prefill_start = time.time()
        
        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            step_start = time.time()
            
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
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
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1
            
            step_end = time.time()
            
            # 每生成10个token打印一次进度
            if i % 10 == 0 and i > 0:
                print(f"│ - 已生成 {i}/{max_seq_len - min_prompt_len} tokens                 │")
                
            # 如果所有序列都结束了，提前退出循环
            if all(next_token_ids == self.tokenizer.eos_id):
                print("│ - 所有序列已生成完成，提前结束                          │")
                break
        
        prefill_end = time.time()
        timing['prefill'] = prefill_end - prefill_start
        
        generation_end = time.time()
        timing['4_generation'] = generation_end - generation_start
        print(f"│ - 预填充阶段耗时: {timing['prefill']:.4f}秒                    │")
        print(f"│ - 总生成循环耗时: {timing['4_generation']:.4f}秒                │")
        print(f"│ - 平均每token生成耗时: {timing['4_generation']/max(1, output_len):.4f}秒 │")
        
        # Detokenization.
        decode_start = time.time()
        
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))
        
        decode_end = time.time()
        timing['decode'] = decode_end - decode_start
        print(f"│ - 解码耗时: {timing['decode']:.4f}秒                          │")
        
        # 总结时间统计
        timing['total'] = time.time() - timing['start']
        print("│                                                           │")
        print("│ 性能分析总结:                                             │")
        print(f"│ - 1. Tokenize:   {timing['1_tokenize']:.4f}秒 ({timing['1_tokenize']/timing['total']*100:.1f}%)          │")
        print(f"│ - 2. 前向传播准备:{timing['2_forward_prep']:.4f}秒 ({timing['2_forward_prep']/timing['total']*100:.1f}%)          │")
        print(f"│ - 3. KV缓存构建: {timing['3_kv_build']:.4f}秒 ({timing['3_kv_build']/timing['total']*100:.1f}%)          │")
        print(f"│ - 4. 生成循环:   {timing['4_generation']:.4f}秒 ({timing['4_generation']/timing['total']*100:.1f}%)          │")
        print(f"│ - 总计:         {timing['total']:.4f}秒 (100%)                       │")
        print("└─────────────────────────────────────────────────────────────┘")
        
        # 如果提供的是单个字符串作为输入，则返回单个字符串
        return results[0] if is_str_prompt else results
        
    
   



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
