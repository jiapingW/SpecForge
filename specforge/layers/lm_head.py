import math
from typing import Optional
import json
import os
import glob

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from huggingface_hub import snapshot_download

from specforge.distributed import get_draft_tp_group, shard_tensor
from specforge.utils import padding

class ParallelLMHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = get_draft_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        # tp-related
        self.out_features_per_shard = math.ceil(out_features / self.tp_size)
        self.padded_out_features = (
            self.out_features_per_shard * self.tp_size - out_features
        )
        assert (
            self.out_features_per_shard * self.tp_size
            == out_features + self.padded_out_features
        )

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_shard, self.in_features, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features_per_shard, **factory_kwargs))
            if bias
            else None
        )

        # init params
        self.reset_parameters()

        # handle weight loading
        self._register_load_state_dict_pre_hook(self.shard_state_dict)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "ParallelLMHead":
        # 1. 加载配置获取维度
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        
        # 2. 实例化对象 (此时会在各卡创建空的 Parameter)
        instance = cls(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False, # 大多数开源模型如 Llama/Qwen 为 False
            dtype=torch.bfloat16
        )

        # 3. 权重定位与加载 (仅在主进程读取磁盘，或者各卡并行读取以减少通信)
        # 这里沿用 TargetHead 的逻辑定位权重文件
        if not os.path.exists(model_path):
            actual_model_path = snapshot_download(repo_id=model_path, cache_dir=cache_dir)
        else:
            actual_model_path = model_path

        # 找到 index.json
        index_files = glob.glob(os.path.join(actual_model_path, "*.index.json"))
        if not index_files:
            raise FileNotFoundError(f"No index.json found in {actual_model_path}")
        
        with open(index_files[0], "r") as f:
            index_json = json.load(f)
        
        ckpt_file = index_json["weight_map"][lm_head_key]
        full_ckpt_path = os.path.join(actual_model_path, ckpt_file)

        # 4. 读取完整权重 (此处由于后续 Hook 会 shard，所以需要加载完整的权重字典)
        if full_ckpt_path.endswith(".safetensors"):
            with safe_open(full_ckpt_path, framework="pt") as f:
                full_weight = f.get_tensor(lm_head_key)
        else:
            state_dict = torch.load(full_ckpt_path, map_location="cpu")
            full_weight = state_dict[lm_head_key]

        # 5. 调用 load_state_dict，触发 shard_state_dict hook
        # 注意：这里构造的 key 要对应 self.weight 的变量名 "weight"
        instance.load_state_dict({"weight": full_weight}, strict=False)
        
        # 6. 移动到设备并设为 eval 模式
        instance = instance.cuda().eval()
        return instance

    def shard_state_dict(self, state_dict, *args):
        if "weight" in state_dict:
            value = state_dict["weight"]

            # pad this value if it is not divisible by the TP size
            if value.shape[0] % self.tp_size != 0:
                padding_size = self.tp_size - value.shape[0] % self.tp_size
                value = F.pad(value, (0, 0, 0, padding_size))
            state_dict["weight"] = shard_tensor(value, self.tp_group, 0)

        if "bias" in state_dict:
            value = state_dict["bias"]

            # pad this value if it is not divisible by the TP size
            if value.shape[0] % self.tp_size != 0:
                padding_size = self.tp_size - value.shape[0] % self.tp_size
                value = F.pad(value, (0, padding_size))
            state_dict["bias"] = shard_tensor(value, self.tp_group, 0)

    def forward(self, hidden: torch.Tensor, gather_output: bool = False):
        """
        hidden: [B, T, H] or [N, H]
        returns:
          - if gather_output=False: local logits [*, local_vocab] and (start,end) for stitching
          - if gather_output=True:  full logits [*, vocab] via all-gather (use for inference)
        """
        orig_shape = hidden.shape
        hidden = hidden.reshape(-1, self.in_features)  # [N, H]

        local_logits = hidden @ self.weight.T  # [N, local_vocab]

        if self.bias is not None:
            local_logits = local_logits + self.bias

        if not gather_output or self.tp_size == 1:
            return local_logits.view(
                *orig_shape[:-1], self.out_features_per_shard
            ).contiguous()
        else:
            # all-gather shards along vocab dim
            chunks = [torch.empty_like(local_logits) for _ in range(self.tp_size)]
            dist.all_gather(chunks, local_logits, group=self.tp_group)
            full = torch.cat(chunks, dim=-1)[
                :, : self.out_features
            ]  # trim padding from ceil-div
            return full.view(*orig_shape[:-1], self.out_features).contiguous()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return f"ParallelLMHead(in_features={self.in_features}, out_features={self.out_features_per_shard}, tp_size={self.tp_size}, tp_rank={self.tp_rank})"


    def preprocess(self, input_ids, target, loss_mask):
        """
        对输入数据进行预处理。
        
        参数:
            input_ids: 原始输入 ID (通常是 List[List[int]] 或未对齐的 Tensor)
            target: 目标 Label (通常是 List[List[int]])
            loss_mask: 损失掩码 [Batch, SeqLen]
            
        返回:
            处理后的 input_ids, target, loss_mask (均为 Tensor)
        """
        # 1. 应用 Padding (调用你提供的 padding 工具函数)
        # 在 TP 环境下，padding 必须确保在所有 Rank 上产生的 Max Length 是一致的
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        # 2. 调整 loss_mask 的维度
        # 从 [Batch, SeqLen] 变为 [Batch, SeqLen, 1]
        # 这样做是为了后续方便与 Logits [Batch, SeqLen, local_vocab] 进行广播乘法
        if isinstance(loss_mask, torch.Tensor):
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(target.device)
        else:
            # 如果 loss_mask 还是 list，先转 tensor
            loss_mask = torch.tensor(loss_mask, device=target.device)[..., None]

        return input_ids, target, loss_mask