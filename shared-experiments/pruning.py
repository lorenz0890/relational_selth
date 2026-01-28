"""
Model-Agnostic Random Pruning Module

This module works with:
- Temporal GNNs (T1, T2)
- XIMP (hierarchical molecular GNNs)

"""

import torch
from torch import Tensor
from torch.nn import Module
from typing import Dict, Tuple, Optional, List


def prune_at_initialization(
    model: Module,
    prune_ratio: float,
    exclude_bias: bool = True,
    exclude_patterns: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Tensor]:
    if not 0.0 <= prune_ratio <= 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}")

    if exclude_patterns is None:
        # Default exclusions: normalization layers
        exclude_patterns = ['bn', 'batch_norm', 'layer_norm', 'ln', 'norm']

    masks = {}
    total_params = 0
    pruned_params = 0

    # Apply pruning to each weight matrix
    for name, param in model.named_parameters():
        # Skip bias terms
        if exclude_bias and 'bias' in name:
            continue

        # Skip layers matching exclusion patterns
        if any(pattern in name.lower() for pattern in exclude_patterns):
            if 'weight' in name or 'bias' in name:
                continue

        # Only prune weight matrices
        if 'weight' in name:
            # SELTH-consistent random pruning: M ~ Bernoulli(1 - prune_ratio)
            # Each weight independently kept with probability (1 - ρ)
            # Store as bool for memory efficiency
            mask = torch.rand_like(param.data) >= prune_ratio

            masks[name] = mask

            # Apply mask immediately (convert to float for multiplication)
            param.data *= mask.to(param.dtype)

            # Track statistics
            layer_total = mask.numel()
            layer_pruned = (~mask).sum().item()  # Count False values
            total_params += layer_total
            pruned_params += layer_pruned

            if verbose:
                print(f"{name:50s}: {layer_total:8d} params, "
                      f"{layer_pruned:8d} pruned ({100*layer_pruned/layer_total:5.2f}%)")

    # Print summary
    if verbose and total_params > 0:
        actual_prune_ratio = pruned_params / total_params
        print(f"\n{'='*70}")
        print(f"SELTH-Compliant Random Pruning Summary:")
        print(f"  Method: Random (M ~ Bernoulli(1-ρ))")
        print(f"  Target prune ratio: {prune_ratio:.2%}")
        print(f"  Actual prune ratio: {actual_prune_ratio:.2%}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Pruned parameters: {pruned_params:,}")
        print(f"  Active parameters: {total_params - pruned_params:,}")
        print(f"\n✓ SELTH-compliant: Masks are i.i.d. Bernoulli(1-{prune_ratio})")
        print(f"{'='*70}\n")

    return masks


def apply_pruning_masks(model: Module, masks: Dict[str, Tensor]) -> None:
    """
    Apply pruning masks to keep pruned weights at zero during training.
    This MUST be called after EVERY optimizer.step() to ensure
    pruned weights don't accumulate gradients and drift from zero.

    """
    for name, param in model.named_parameters():
        if name in masks:
            # Convert bool mask to same dtype as param for multiplication
            mask = masks[name].to(param.dtype) if masks[name].dtype == torch.bool else masks[name]

            # Element-wise multiplication: zero out pruned weights
            param.data *= mask

            # CRITICAL: Also zero gradients for pruned weights
            # Without this, gradients accumulate for pruned weights, wasting computation
            # and potentially causing pruned weights to drift from zero
            if param.grad is not None:
                param.grad *= mask


def count_parameters(model: Module, masks: Dict[str, Tensor]) -> Tuple[int, int]:
    """
    Count total and active (non-pruned) parameters in a PyTorch model.
    """
    total = 0
    active = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count

        if name in masks:
            # Count non-zero (active) parameters
            mask = masks[name]
            if mask.dtype == torch.bool:
                active += mask.sum().item()
            else:
                active += (mask != 0).sum().item()
        else:
            # No mask = all parameters active
            active += param_count

    return total, active


def get_sparsity_stats(model: Module, masks: Dict[str, Tensor]) -> Dict[str, any]:

    #Compute sparsity statistics for a pruned model.

    total_params = 0
    zero_params = 0
    layer_sparsities = {}

    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name]
            layer_total = mask.numel()
            # Handle both bool and float masks
            if mask.dtype == torch.bool:
                layer_zeros = (~mask).sum().item()  # Count False values
            else:
                layer_zeros = (mask == 0).sum().item()

            total_params += layer_total
            zero_params += layer_zeros

            layer_sparsities[name] = layer_zeros / layer_total

    global_sparsity = zero_params / total_params if total_params > 0 else 0.0

    return {
        'global_sparsity': global_sparsity,
        'total_params': total_params,
        'zero_params': zero_params,
        'active_params': total_params - zero_params,
        'layer_sparsities': layer_sparsities
    }


def verify_masks(
    model: Module,
    masks: Dict[str, Tensor],
    tolerance: float = 1e-7
) -> bool:

    #This function ensures that the mask M is correctly applied and that pruned weights haven't drifted from zero during training.


    all_valid = True

    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name]

            # Get indices where mask is False (pruned weights)
            if mask.dtype == torch.bool:
                pruned_indices = ~mask
            else:
                pruned_indices = mask == 0

            # Check if pruned weights are zero
            pruned_weights = param.data[pruned_indices]

            if not torch.all(torch.abs(pruned_weights) <= tolerance):
                max_violation = torch.max(torch.abs(pruned_weights)).item()
                print(f"WARNING: Mask violation in {name}")
                print(f"  Max absolute value of pruned weight: {max_violation:.2e}")
                all_valid = False

    return all_valid


def save_masks(masks: Dict[str, Tensor], filepath: str) -> None:
    #Save pruning masks to file for reproducibility 
    torch.save(masks, filepath)
    print(f"Saved pruning masks to {filepath}")


def load_masks(filepath: str) -> Dict[str, Tensor]:
    #Load pruning masks from file
    masks = torch.load(filepath, weights_only=True)
    print(f"Loaded pruning masks from {filepath}")
    return masks


def get_compression_ratio(model: Module, masks: Dict[str, Tensor]) -> float:
    #Compute compression ratio achieved by pruning.
    stats = get_sparsity_stats(model, masks)
    if stats['active_params'] == 0:
        return float('inf')
    return stats['total_params'] / stats['active_params']


