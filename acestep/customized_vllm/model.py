"""Qwen3 causal LM built with plain nn.Linear/Embedding (no tensor parallelism)."""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3Config

from acestep.customized_vllm.layers import RMSNorm, SiluAndMul, get_rope
from acestep.customized_vllm.attention import Attention
from acestep.customized_vllm.context import get_context


def _packed_qkv_linear(hidden_size, num_heads, num_kv_heads, head_dim, bias):
    """Create a packed Q/K/V projection with a shard-aware weight loader."""
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    linear = nn.Linear(hidden_size, q_size + 2 * kv_size, bias=bias)

    def loader(param, weight, shard_id):
        offsets = {"q": 0, "k": q_size, "v": q_size + kv_size}
        sizes = {"q": q_size, "k": kv_size, "v": kv_size}
        param.data.narrow(0, offsets[shard_id], sizes[shard_id]).copy_(weight)

    linear.weight.weight_loader = loader
    if bias:
        linear.bias.weight_loader = loader
    return linear


def _packed_gate_up_linear(hidden_size, intermediate_size):
    """Create a packed gate+up projection with a shard-aware weight loader."""
    linear = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)

    def loader(param, weight, shard_id):
        param.data.narrow(0, shard_id * intermediate_size, intermediate_size).copy_(weight)

    linear.weight.weight_loader = loader
    return linear


class Qwen3Attention(nn.Module):
    """Qwen3 attention block with packed QKV, RoPE, and optional QK-norm."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        has_bias = getattr(config, "attention_bias", True)

        self.qkv_proj = _packed_qkv_linear(
            config.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim, has_bias,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = get_rope(
            self.head_dim, config.max_position_embeddings,
            getattr(config, "rope_theta", 1000000),
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)
        if not has_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._has_bias = has_bias

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self._has_bias:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))


class Qwen3MLP(nn.Module):
    """Qwen3 MLP with packed gate+up projection and SiLU gating."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_up_proj = _packed_gate_up_linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 causal language model for inference."""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, positions):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(positions, h, residual)
        h, _ = self.norm(h, residual)
        return h

    def compute_logits(self, hidden_states):
        """Extract last-token hidden states (prefill) and project to vocab logits."""
        ctx = get_context()
        if ctx.is_prefill:
            hidden_states = hidden_states[ctx.cu_seqlens_q[1:] - 1].contiguous()
        return self.lm_head(hidden_states)
