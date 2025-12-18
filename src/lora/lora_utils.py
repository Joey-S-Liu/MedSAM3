"""
Utilities for injecting LoRA into SAM3 models and managing LoRA parameters.
"""

import re
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from .lora_layer import LinearWithLoRA, MultiheadAttentionLoRA


class LoRAConfig:
    """
    Configuration for LoRA injection into SAM3 models.

    Args:
        rank: Rank of LoRA matrices
        alpha: LoRA scaling factor
        dropout: Dropout probability
        target_modules: List of module patterns to apply LoRA to.
                       Can include: 'q_proj', 'k_proj', 'v_proj', 'out_proj',
                       'linear1', 'linear2', 'all'
    """

    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default: apply to all attention projections and FFN layers
        # Supports multiple naming conventions:
        # - q_proj, k_proj, v_proj, out_proj: Standard separate projections
        # - qkv: Fused Q/K/V projection (ViT-style, used in SAM3 vision backbone)
        # - proj: Output projection in vision backbone
        # - c_fc, c_proj: MLP layers in CLIP-style language backbone
        # - linear1, linear2: FFN layers in transformer encoder/decoder
        if target_modules is None:
            target_modules = [
                # Standard attention projections
                "q_proj", "k_proj", "v_proj", "out_proj",
                # Vision backbone (ViT-style)
                "qkv",  # Fused Q/K/V projection
                "proj",  # Output projection in vision backbone (attn.proj)
                "fc1", "fc2",  # MLP layers in vision backbone
                # Language backbone (CLIP-style) MLP
                "c_fc", "c_proj",
                # Transformer FFN layers
                "linear1", "linear2",
            ]

        self.target_modules = set(target_modules)

        # If 'all' is specified, enable all possible modules
        if "all" in self.target_modules:
            self.target_modules = {
                # Standard attention projections
                "q_proj", "k_proj", "v_proj", "out_proj",
                # Vision backbone (ViT-style)
                "qkv",  # Fused Q/K/V
                "proj",  # Output projection
                "fc1", "fc2",  # MLP layers
                # Language backbone (CLIP-style)
                "c_fc", "c_proj",
                # Transformer FFN
                "linear1", "linear2",
                # MultiheadAttention internal (if accessible)
                "in_proj",
            }


def _should_inject_lora(name: str, target_modules: Set[str]) -> bool:
    """
    Check if a module should have LoRA injected based on its name.

    Args:
        name: Module name
        target_modules: Set of target module patterns

    Returns:
        True if LoRA should be injected
    """
    # Get the module basename (last part of the name)
    module_basename = name.split(".")[-1]

    # Direct basename match - most reliable method
    if module_basename in target_modules:
        return True

    # Check for substring matches in basename
    # This handles cases like "out_proj" matching "proj" target
    for target in target_modules:
        if target in module_basename:
            return True

    # Check for substring matches in full name
    # This handles patterns like "self_attn" appearing in the path
    for target in target_modules:
        if target in name:
            # Make sure we're matching a meaningful component
            # Avoid false positives by checking it's a component boundary
            parts = name.split(".")
            for part in parts:
                if target == part or target in part:
                    return True

    return False


def _is_inside_multihead_attention(model: nn.Module, module_name: str) -> bool:
    """Check if the module is a direct child of nn.MultiheadAttention."""
    parts = module_name.split('.')
    if len(parts) < 2:
        return False
    # Get the parent module path
    parent_path = '.'.join(parts[:-1])
    # Find the parent module
    parent = model
    for p in parent_path.split('.'):
        if hasattr(parent, p):
            parent = getattr(parent, p)
        else:
            return False
    return isinstance(parent, nn.MultiheadAttention)


def inject_lora_into_model(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True,
) -> nn.Module:
    """
    Inject LoRA layers into a SAM3 model.

    This function:
    1. Replaces nn.MultiheadAttention with MultiheadAttentionLoRA (enables LoRA on Q/K/V/out_proj)
    2. Applies LoRA to all matching Linear layers

    Args:
        model: SAM3 model to inject LoRA into
        config: LoRA configuration
        verbose: If True, print injection details

    Returns:
        Modified model with LoRA injected
    """
    # STEP 1: Replace nn.MultiheadAttention with MultiheadAttentionLoRA
    # This enables LoRA to be applied to Q, K, V, and out_proj inside MHA
    mha_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            mha_to_replace.append((name, module))

    mha_replaced_count = 0
    for name, mha in mha_to_replace:
        # Get parent module and attribute name
        *parent_path, attr_name = name.split(".")
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

        setattr(parent, attr_name, new_mha)
        mha_replaced_count += 1

        if verbose:
            print(f"Replaced MHA: {name}")

    if verbose:
        print(f"\nReplaced {mha_replaced_count} nn.MultiheadAttention with MultiheadAttentionLoRA")

    # STEP 2: Freeze all parameters before applying LoRA
    for param in model.parameters():
        param.requires_grad = False

    # STEP 3: Apply LoRA to all matching Linear layers
    injected_count = 0
    total_lora_params = 0

    for name, module in model.named_modules():
        # Skip if not a Linear layer
        if not isinstance(module, nn.Linear):
            continue

        # Check if we should inject LoRA
        if not _should_inject_lora(name, config.target_modules):
            continue

        # Get parent module and attribute name
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Replace with LoRA layer
        lora_layer = LinearWithLoRA(
            linear=module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )

        setattr(parent, attr_name, lora_layer)

        # Count parameters
        lora_params = sum(p.numel() for p in lora_layer.lora.parameters())
        total_lora_params += lora_params
        injected_count += 1

        if verbose:
            print(
                f"Injected LoRA into {name}: "
                f"{module.in_features}x{module.out_features} "
                f"-> {lora_params:,} trainable params"
            )

    if verbose:
        print(f"\nTotal LoRA injections: {injected_count}")
        print(f"Total LoRA parameters: {total_lora_params:,}")

        # Calculate total model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(
            f"Trainable ratio: {100 * trainable_params / total_params:.2f}%"
        )

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []

    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            lora_params.extend(module.lora.parameters())

    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters as a state dict.

    Args:
        model: Model with LoRA layers

    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            lora_state_dict[f"{name}.lora.lora_A"] = module.lora.lora_A.data
            lora_state_dict[f"{name}.lora.lora_B"] = module.lora.lora_B.data

    return lora_state_dict


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    Load LoRA parameters from a state dict.

    Args:
        model: Model with LoRA layers
        state_dict: State dict containing LoRA parameters
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            lora_a_key = f"{name}.lora.lora_A"
            lora_b_key = f"{name}.lora.lora_B"

            if lora_a_key in state_dict:
                module.lora.lora_A.data = state_dict[lora_a_key]
            if lora_b_key in state_dict:
                module.lora.lora_B.data = state_dict[lora_b_key]


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into the base model.

    This converts all LinearWithLoRA layers back to regular Linear layers
    with the LoRA weights merged in.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with LoRA weights merged
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LinearWithLoRA):
            # Get parent module and attribute name
            *parent_path, attr_name = name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)

            # Replace with merged linear layer
            merged_linear = module.merge_weights()
            setattr(parent, attr_name, merged_linear)

    return model


def print_trainable_parameters(model: nn.Module):
    """
    Print summary of trainable parameters in the model.

    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )
