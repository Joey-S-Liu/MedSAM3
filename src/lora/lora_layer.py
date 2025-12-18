"""
LoRA (Low-Rank Adaptation) Layer Implementation for SAM3

This module implements LoRA layers that can be injected into transformer models
for efficient fine-tuning.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionLoRA(nn.Module):
    """
    Custom MultiheadAttention that doesn't use F.multi_head_attention_forward,
    allowing LoRA to be properly applied to Q, K, V, and output projections.

    This replaces nn.MultiheadAttention to enable LoRA on all projection layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        # Copy weights from existing MHA
        in_proj_weight: Optional[torch.Tensor] = None,
        in_proj_bias: Optional[torch.Tensor] = None,
        out_proj_weight: Optional[torch.Tensor] = None,
        out_proj_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Separate Q, K, V projections (instead of fused in_proj)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize from existing MHA weights if provided
        if in_proj_weight is not None:
            # Split in_proj_weight into q, k, v
            self.q_proj.weight.data = in_proj_weight[:embed_dim, :].clone()
            self.k_proj.weight.data = in_proj_weight[embed_dim:2*embed_dim, :].clone()
            self.v_proj.weight.data = in_proj_weight[2*embed_dim:, :].clone()

        if in_proj_bias is not None:
            self.q_proj.bias.data = in_proj_bias[:embed_dim].clone()
            self.k_proj.bias.data = in_proj_bias[embed_dim:2*embed_dim].clone()
            self.v_proj.bias.data = in_proj_bias[2*embed_dim:].clone()

        if out_proj_weight is not None:
            self.out_proj.weight.data = out_proj_weight.clone()

        if out_proj_bias is not None:
            self.out_proj.bias.data = out_proj_bias.clone()

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using separate Q, K, V projections so LoRA works.
        """
        # Handle batch_first
        if self.batch_first:
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            tgt_len, batch_size, _ = query.shape
            src_len = key.shape[0]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V - LoRA is applied here through the Linear layers
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask - handle various input formats
        if attn_mask is not None:
            # attn_weights shape: (batch, num_heads, tgt_len, src_len)
            if attn_mask.dim() == 2:
                # (tgt_len, src_len) -> (1, 1, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # Could be (batch, tgt_len, src_len) or (batch*num_heads, tgt_len, src_len)
                if attn_mask.shape[0] == batch_size:
                    # (batch, tgt_len, src_len) -> (batch, 1, tgt_len, src_len)
                    attn_mask = attn_mask.unsqueeze(1)
                elif attn_mask.shape[0] == batch_size * self.num_heads:
                    # (batch*num_heads, tgt_len, src_len) -> (batch, num_heads, tgt_len, src_len)
                    attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
                else:
                    # Unknown format, try to broadcast
                    attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:
                # Already (batch, num_heads, tgt_len, src_len) or similar
                pass

            # Expand to match attn_weights if needed
            if attn_mask.shape != attn_weights.shape:
                attn_mask = attn_mask.expand_as(attn_weights)

            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)

        # Output projection - LoRA is applied here
        attn_output = self.out_proj(attn_output)

        # Convert back if not batch_first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear transformation.

    This implements the LoRA technique from "LoRA: Low-Rank Adaptation of Large Language Models"
    where a linear layer's weight W is augmented with a low-rank update: W' = W + BA
    where B is (out_features x r) and A is (r x in_features), with r << min(in_features, out_features)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of the low-rank matrices (r)
        alpha: Scaling factor for LoRA (controls the magnitude of updates)
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform initialization for A and zero for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Apply dropout to input
        x_dropout = self.dropout(x)

        # Low-rank adaptation: (x @ A^T) @ B^T
        result = F.linear(x_dropout, self.lora_B @ self.lora_A)

        # Scale by alpha/r
        return result * self.scaling

    def merge_weights(self) -> torch.Tensor:
        """
        Merge LoRA weights into a single weight matrix.

        Returns:
            Merged weight matrix of shape (out_features, in_features)
        """
        return (self.lora_B @ self.lora_A) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.

    This wraps an existing nn.Linear layer and adds a LoRA layer in parallel.
    The original linear layer's weights are frozen.

    Args:
        linear: Original linear layer to adapt
        rank: Rank of LoRA matrices
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store original linear layer and freeze it
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        # Store dimensions for compatibility
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    @property
    def weight(self):
        """Expose weight for compatibility with nn.Linear."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose bias for compatibility with nn.Linear."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original linear transformation + LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Original linear transformation (frozen)
        result = self.linear(x)

        # Add LoRA adaptation
        result = result + self.lora(x)

        return result

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into the original linear layer.

        Returns:
            New linear layer with merged weights
        """
        merged_weight = self.linear.weight.data + self.lora.merge_weights()

        new_linear = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )
        new_linear.weight.data = merged_weight
        if self.linear.bias is not None:
            new_linear.bias.data = self.linear.bias.data.clone()

        return new_linear
