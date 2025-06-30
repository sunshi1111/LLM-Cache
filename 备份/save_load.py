def save_kv_cache(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
                 file_path: str, prompt_len: int, max_seq_len: int) -> None:

    # 创建目录(如果不存在)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # 构建保存数据
        save_data = {
            "kv_caches": [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_caches],
            "prompt_len": prompt_len,
            "max_seq_len": max_seq_len,
            "timestamp": time.time(),
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
        print(f"包含prompt长度: {prompt_len}")

    def load_kv_cache(self, file_path: str, device: Any) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, int]:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到KV cache文件: {file_path}")

        print(f"正在从 {file_path} 加载KV cache...")
    # 加载数据
        save_data = torch.load(file_path, map_location="cpu")

    # 验证模型配置匹配
        model_config = save_data["model_config"]
        if (model_config["hidden_size"] != self.config.hidden_size or
            model_config["num_hidden_layers"] != self.config.num_hidden_layers or
            model_config["num_attention_heads"] != self.config.num_attention_heads or
            model_config["num_key_value_heads"] != self.config.num_key_value_heads or
            model_config["head_dim"] != self.config.head_dim):
            raise ValueError("加载的KV cache与当前模型配置不匹配")

    # 恢复KV cache并移到指定设备
        kv_caches = [(k.to(device), v.to(device)) for k, v in save_data["kv_caches"]]
        prompt_len = save_data["prompt_len"]
        max_seq_len = save_data.get("max_seq_len", kv_caches[0][0].shape[1])
        print(f"已加载KV cache，prompt长度为 {prompt_len}，最大序列长度为 {max_seq_len}")
        return kv_caches, prompt_len, max_seq_len