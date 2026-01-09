import math
import numpy as np
import copy
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

def _binom2(n: int) -> float:
    """Compute n choose 2 as in the bounds."""
    return 0.5 * n * (n - 1)

def gamma_rgnn_lower_bound(
    N_max: float,
    s_min: float,
    m_min: float,
    N_max_comb: float,
    s_min_comb: float,
    m_min_comb: float,
    rho: float,
    L: int,
    M: int,
    B: int,
    use_simplified: bool = True,
) -> float:
    def _binom2(n: float) -> float:
        return 0.5 * n * (n - 1)

    if use_simplified:
        N_tilde = max(N_max, N_max_comb)
        s_tilde = min(s_min, s_min_comb)
        m_tilde = min(m_min, m_min_comb)

        base = 1.0 - _binom2(N_tilde) * (rho ** (s_tilde * m_tilde))
        base = max(0.0, min(1.0, base))
        prob = base ** (L * (B * M + 1))
    else:
        base_branch = 1.0 - _binom2(N_max) * (rho ** (s_min * m_min))
        base_comb   = 1.0 - _binom2(N_max_comb) * (rho ** (s_min_comb * m_min_comb))

        base_branch = max(0.0, min(1.0, base_branch))
        base_comb   = max(0.0, min(1.0, base_comb))

        prob = ((base_branch ** (B * M)) * base_comb) ** L

    return max(0.0, min(1.0, prob))


def mlp_madds(num_inputs: int, width: int, depth: int, sparsity: float = 0.0) -> float:
    if num_inputs <= 0 or width <= 0 or depth <= 0:
        return 0.0
    sparsity = min(max(sparsity, 0.0), 1.0)
    density = 1.0 - sparsity
    weights_per_layer = width * width
    effective_weights_per_layer = density * weights_per_layer
    total_weights = depth * effective_weights_per_layer
    return num_inputs * total_weights

def rgnn_madds_estimate(
    N_max: int,
    N_max_comb: int,
    m_min: float,
    m_min_comb: float,
    B: int,
    M: int,
    L: int,
    sparsity: float = 0.0,
) -> float:
    if L <= 0:
        return 0.0
    branch_madds_per_layer = B * 1 * mlp_madds(N_max, m_min, M, sparsity)
    combine_madds_per_layer = mlp_madds(N_max_comb, m_min_comb, M, sparsity)
    return L * (branch_madds_per_layer + combine_madds_per_layer)


def masked_state_dict(
    base_sd: Dict[str, torch.Tensor],
    rho: float,
    rng: torch.Generator,
    prune_bias: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Apply Bernoulli(rho) mask to all floating tensors except biases (unless prune_bias=True).
    """
    out = {}
    for k, v in base_sd.items():
        if (not prune_bias) and (k.endswith(".bias") or k.endswith("bias")):
            out[k] = v.clone()
            continue
        if not torch.is_floating_point(v):
            out[k] = v.clone()
            continue
        probs = torch.full_like(v, float(rho), device=v.device)
        mask = torch.bernoulli(probs, generator=rng)
        out[k] = v * mask
    return out

# -----------------------------
# Extremely simple RGNN
# -----------------------------
def mean_aggregate(num_nodes: int, edge_index: torch.LongTensor, h: torch.Tensor) -> torch.Tensor:
    """
    Mean aggregation over directed edges:
      out[v] = mean_{u -> v} h[u]
    """
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(h)
    out.index_add_(0, dst, h[src])

    deg = torch.zeros((num_nodes,), device=h.device, dtype=h.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    deg = deg.clamp_min(1.0).unsqueeze(1)
    return out / deg

# -----------------------------
# Extremely simple RGNN
# -----------------------------
def sum_aggregate(num_nodes: int, edge_index: torch.LongTensor, h: torch.Tensor) -> torch.Tensor:
    """
    Sum aggregation over directed edges:
      out[v] = sum_{u -> v} h[u]

    Shapes:
      edge_index: [2, E]
      h:         [N, d]
      out:       [N, d]
    """
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(h)
    out.index_add_(0, dst, h[src])
    return out