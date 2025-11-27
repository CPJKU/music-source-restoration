"""
LoRA (Low-Rank Adaptation) implementation for RoFormer models.

This module provides LoRA layers that can be optionally applied to linear layers
in the RoFormer architecture to enable memory-efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    LoRA-wrapped linear layer.
    
    Applies low-rank adaptation to a linear layer:
    output = (W + B @ A) @ x + bias
    
    where:
    - W is the original weight (frozen)
    - A and B are low-rank matrices (trainable)
    - x is the input
    
    Args:
        linear_layer: The original linear layer to wrap
        r: LoRA rank (dimension of the low-rank matrices)
        lora_alpha: Scaling factor for LoRA (typically r or 2*r)
        lora_dropout: Dropout rate for LoRA
        enable: Whether to enable LoRA (if False, behaves as original layer)
    """
    
    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        enable: bool = True
    ):
        super().__init__()
        
        self.enable = enable
        
        if not enable:
            # If LoRA is disabled, just use the original layer
            self.linear = linear_layer
            return
        
        # Store original layer attributes
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # Freeze original weights
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Initialize LoRA parameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA A matrix (initialized to zeros)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        
        # LoRA B matrix (initialized to zeros, so initial delta is zero)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Reset LoRA parameters
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA parameters."""
        if not self.enable:
            return
        
        # Initialize A with Kaiming uniform (like linear layer init)
        nn.init.kaiming_uniform_(self.lora_A, a=2.236)
        # Initialize B to zero (so initial delta is zero, preserving original behavior)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-wrapped linear layer."""
        if not self.enable:
            return self.linear(x)
        
        # Original output
        result = self.linear(x)
        
        # LoRA adaptation
        x_dropout = self.lora_dropout(x)
        lora_output = F.linear(x_dropout, self.lora_B @ self.lora_A, None)
        
        # Scale and add to original output
        result = result + lora_output * self.scaling
        
        return result
    
    def merge_weights(self):
        """
        Merge LoRA weights into the original linear layer.
        After merging, the model behaves as if LoRA was never applied.
        """
        if not self.enable:
            return
        
        # Compute delta weight
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        
        # Add to original weight
        with torch.no_grad():
            self.linear.weight.data += delta_weight
    
    def unmerge_weights(self):
        """
        Unmerge LoRA weights (reverse of merge_weights).
        This requires keeping a copy of the original weights, so it's not implemented here.
        """
        raise NotImplementedError("Unmerge is not implemented. Save original weights before merging if needed.")


class LoRALinearQKV(nn.Module):
    """
    LoRA-wrapped linear layer that handles QKV (query, key, value) projections.
    
    This is used for attention layers where the linear layer projects to 3*head_dim
    dimensions, and we may want to apply LoRA to Q, K, and V separately.
    
    Args:
        linear_layer: The original linear layer (projects to 3 * dim_inner)
        r: LoRA rank
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout rate for LoRA
        enable_lora: List of 3 booleans indicating whether to apply LoRA to [Q, K, V]
    """
    
    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        enable_lora: list = [True, False, True]
    ):
        super().__init__()
        
        assert len(enable_lora) == 3, "enable_lora must be a list of 3 booleans [Q, K, V]"
        assert linear_layer.out_features % 3 == 0, "Output features must be divisible by 3 for QKV"
        
        self.enable_lora = enable_lora
        self.linear = linear_layer
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
        
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.dim_per_head = self.out_features // 3
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize LoRA parameters for Q, K, V
        # Register parameters directly as attributes (since ParameterList doesn't support None)
        self.lora_As = []
        self.lora_Bs = []
        
        for i, enable in enumerate(enable_lora):
            if enable:
                # LoRA A matrix
                lora_A = nn.Parameter(torch.zeros(r, self.in_features))
                nn.init.kaiming_uniform_(lora_A, a=2.236)
                self.register_parameter(f'lora_A_{i}', lora_A)
                self.lora_As.append(lora_A)
                
                # LoRA B matrix (initialized to zero)
                lora_B = nn.Parameter(torch.zeros(self.dim_per_head, r))
                nn.init.zeros_(lora_B)
                self.register_parameter(f'lora_B_{i}', lora_B)
                self.lora_Bs.append(lora_B)
            else:
                # Placeholder to maintain indexing
                self.lora_As.append(None)
                self.lora_Bs.append(None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-wrapped QKV layer."""
        # Original output
        result = self.linear(x)  # [..., 3 * dim_per_head]
        
        # Split into Q, K, V
        q, k, v = result.split(self.dim_per_head, dim=-1)
        
        x_dropout = self.lora_dropout(x)
        
        # Apply LoRA to each if enabled
        if self.enable_lora[0] and self.lora_As[0] is not None:
            q_lora = F.linear(x_dropout, self.lora_Bs[0] @ self.lora_As[0], None)
            q = q + q_lora * self.scaling
        
        if self.enable_lora[1] and self.lora_As[1] is not None:
            k_lora = F.linear(x_dropout, self.lora_Bs[1] @ self.lora_As[1], None)
            k = k + k_lora * self.scaling
        
        if self.enable_lora[2] and self.lora_As[2] is not None:
            v_lora = F.linear(x_dropout, self.lora_Bs[2] @ self.lora_As[2], None)
            v = v + v_lora * self.scaling
        
        # Concatenate back
        result = torch.cat([q, k, v], dim=-1)
        
        return result


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    enable_lora_qkv: list = [True, False, True],
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Apply LoRA to linear layers in a model.
    
    Args:
        model: The model to apply LoRA to
        r: LoRA rank
        lora_alpha: Scaling factor
        lora_dropout: Dropout rate
        enable_lora_qkv: Which of Q, K, V to apply LoRA to in attention layers
        target_modules: List of module names to apply LoRA to. If None, applies to all linear layers.
    
    Returns:
        Model with LoRA applied
    """
    # For now, we'll apply LoRA specifically in the Attention class
    # This function is here for future extensibility
    return model

