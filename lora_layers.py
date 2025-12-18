"""
LoRA (Low-Rank Adaptation) implementation for SAM3 model fine-tuning.
Supports selective application to different transformer components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple
import math


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
            # Input: (batch, seq, embed_dim)
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            # Input: (seq, batch, embed_dim)
            tgt_len, batch_size, _ = query.shape
            src_len = key.shape[0]
            # Convert to batch_first for easier processing
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V - LoRA is applied here through the Linear layers
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        # (batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)
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
            # key_padding_mask: (batch, src_len), True = ignore
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, embed_dim)
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
    LoRA layer that replaces a linear layer with low-rank adaptation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank matrices (r in the paper)
        alpha: Scaling factor (typically set to rank)
        dropout: Dropout probability for LoRA weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation: x @ (A @ B) * scaling
        """
        # x shape: (..., in_features)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Combines the original frozen linear layer with a LoRA layer.

    Exposes weight/bias properties to maintain compatibility with modules
    that access these attributes directly (e.g., nn.MultiheadAttention).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Freeze the original layer
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Store original layer attributes for compatibility
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    @property
    def weight(self) -> torch.Tensor:
        """Proxy to original layer's weight for compatibility with nn.MultiheadAttention."""
        return self.original_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Proxy to original layer's bias for compatibility with nn.MultiheadAttention."""
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original output + LoRA output
        """
        return self.original_layer(x) + self.lora(x)


class LoRAConfig:
    """
    Configuration for LoRA application to SAM3 model.

    Args:
        rank: Rank of LoRA matrices
        alpha: Scaling factor
        dropout: Dropout probability
        target_modules: Which modules to apply LoRA to
        apply_to_vision_encoder: Whether to apply LoRA to vision encoder
        apply_to_text_encoder: Whether to apply LoRA to text encoder
        apply_to_geometry_encoder: Whether to apply LoRA to geometry encoder
        apply_to_detr_encoder: Whether to apply LoRA to DETR encoder
        apply_to_detr_decoder: Whether to apply LoRA to DETR decoder
        apply_to_mask_decoder: Whether to apply LoRA to mask decoder
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        # Component-level control
        apply_to_vision_encoder: bool = True,
        apply_to_text_encoder: bool = True,
        apply_to_geometry_encoder: bool = False,
        apply_to_detr_encoder: bool = True,
        apply_to_detr_decoder: bool = True,
        apply_to_mask_decoder: bool = False,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default target modules: Q, K, V projections for different architectures
        # - q_proj, k_proj, v_proj: Standard separate projections (LLaMA-style)
        # - qkv: Fused Q/K/V projection (ViT-style, used in SAM3 vision backbone)
        # - proj: Output projection in vision backbone (different from out_proj)
        # - out_proj: Output projection in MultiheadAttention
        # - c_fc, c_proj: MLP layers in CLIP-style language backbone
        # - linear1, linear2: FFN layers in transformer encoder/decoder
        if target_modules is None:
            target_modules = [
                # Standard attention projections
                "q_proj", "k_proj", "v_proj", "out_proj",
                # Vision backbone (ViT-style)
                "qkv",  # Fused Q/K/V projection
                "proj",  # Output projection (note: will also match out_proj, c_proj)
                "fc1", "fc2",  # MLP layers in vision backbone
                # Language backbone (CLIP-style) MLP
                "c_fc", "c_proj",
                # Transformer FFN layers
                "linear1", "linear2",
            ]
        self.target_modules = set(target_modules)

        # Component flags
        self.apply_to_vision_encoder = apply_to_vision_encoder
        self.apply_to_text_encoder = apply_to_text_encoder
        self.apply_to_geometry_encoder = apply_to_geometry_encoder
        self.apply_to_detr_encoder = apply_to_detr_encoder
        self.apply_to_detr_decoder = apply_to_detr_decoder
        self.apply_to_mask_decoder = apply_to_mask_decoder

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "apply_to_vision_encoder": self.apply_to_vision_encoder,
            "apply_to_text_encoder": self.apply_to_text_encoder,
            "apply_to_geometry_encoder": self.apply_to_geometry_encoder,
            "apply_to_detr_encoder": self.apply_to_detr_encoder,
            "apply_to_detr_decoder": self.apply_to_detr_decoder,
            "apply_to_mask_decoder": self.apply_to_mask_decoder,
        }


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to specified modules in the SAM3 model.

    This function:
    1. Replaces nn.MultiheadAttention with MultiheadAttentionLoRA (enables LoRA on Q/K/V/out_proj)
    2. Applies LoRA to all matching Linear layers

    Args:
        model: SAM3 model to apply LoRA to
        config: LoRA configuration

    Returns:
        Model with LoRA applied
    """

    # CRITICAL: Freeze all base model parameters first
    for param in model.parameters():
        param.requires_grad = False

    def should_apply_lora_to_component(module_name: str) -> bool:
        """Check component-level flags to determine if we should apply LoRA."""
        if ("vision_encoder" in module_name or "vision_backbone" in module_name) and not config.apply_to_vision_encoder:
            return False
        if ("text_encoder" in module_name or "language_backbone" in module_name) and not config.apply_to_text_encoder:
            return False
        if "geometry_encoder" in module_name and not config.apply_to_geometry_encoder:
            return False
        if ("detr_encoder" in module_name or "transformer.encoder" in module_name) and not config.apply_to_detr_encoder:
            return False
        if ("detr_decoder" in module_name or "transformer.decoder" in module_name) and not config.apply_to_detr_decoder:
            return False
        if "mask_decoder" in module_name and not config.apply_to_mask_decoder:
            return False
        return True

    def should_apply_lora(module_name: str) -> bool:
        """Determine if LoRA should be applied to this module."""
        if not should_apply_lora_to_component(module_name):
            return False

        # Check if module name matches target modules
        module_basename = module_name.split('.')[-1]

        # Direct basename match (e.g., "qkv", "proj", "linear1", etc.)
        if module_basename in config.target_modules:
            return True

        # Also check for substring match for flexibility
        for target in config.target_modules:
            if target in module_basename:
                return True

        return False

    # Track replacements
    mha_replaced = []
    lora_modules_applied = []

    # STEP 1: Replace nn.MultiheadAttention with MultiheadAttentionLoRA
    # This enables LoRA to be applied to Q, K, V, and out_proj inside MHA
    mha_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if should_apply_lora_to_component(name):
                mha_to_replace.append((name, module))

    for name, mha in mha_to_replace:
        # Get parent module and attribute name
        *parent_path, attr_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Create replacement with separate Q, K, V projections
        new_mha = MultiheadAttentionLoRA(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            batch_first=mha.batch_first,
            in_proj_weight=mha.in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            out_proj_weight=mha.out_proj.weight,
            out_proj_bias=mha.out_proj.bias if mha.out_proj.bias is not None else None,
        )

        # Freeze the new MHA parameters
        for param in new_mha.parameters():
            param.requires_grad = False

        setattr(parent, attr_name, new_mha)
        mha_replaced.append(name)

    print(f"Replaced {len(mha_replaced)} nn.MultiheadAttention modules with MultiheadAttentionLoRA")

    # STEP 2: Apply LoRA to all matching Linear layers
    # Now includes q_proj, k_proj, v_proj, out_proj from the replaced MHA modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Get parent module and attribute name
            *parent_path, attr_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)

            # Replace with LoRA linear
            lora_linear = LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
            setattr(parent, attr_name, lora_linear)
            lora_modules_applied.append(name)

    print(f"Applied LoRA to {len(lora_modules_applied)} modules:")
    for module_name in lora_modules_applied[:15]:  # Show first 15
        print(f"  - {module_name}")
    if len(lora_modules_applied) > 15:
        print(f"  ... and {len(lora_modules_applied) - 15} more")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from the model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA weights (not the full model).

    Args:
        model: Model with LoRA layers
        save_path: Path to save LoRA weights
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B

    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")


def load_lora_weights(model: nn.Module, load_path: str):
    """
    Load LoRA weights into a model.

    Args:
        model: Model with LoRA layers
        load_path: Path to LoRA weights
    """
    lora_state_dict = torch.load(load_path)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"Loaded LoRA weights from {load_path}")
