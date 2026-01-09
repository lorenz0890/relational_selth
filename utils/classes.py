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

# -----------------------------
# Toy multi-relational graphs
# -----------------------------
@dataclass
class RelGraph:
    num_nodes: int
    edge_index_by_rel: List[torch.LongTensor]  # list of [2, E_r] tensors

    def to(self, dev: torch.device) -> "RelGraph":
        self.edge_index_by_rel = [ei.to(dev) for ei in self.edge_index_by_rel]
        return self

class TinyRGNN(nn.Module):
    """
    Minimal branches+combine RGNN:
      h0 = tanh(W_in x)
      for each layer:
         for each relation r: m_r = tanh(W_r meanAgg_r(h))
         h = tanh(W_c concat(h, m_1..m_R))
      graph_emb = sum_nodes(h)
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)

        self.branch_linears = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)])
            for _ in range(num_layers)
        ])
        self.combine_linears = nn.ModuleList([
            #nn.Linear(hidden_dim * (num_relations + 1), hidden_dim, bias=False)
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])

    @torch.no_grad()
    def embed(self, g: RelGraph) -> torch.Tensor:
        """Return graph embedding vector [hidden_dim]."""
        g = copy.deepcopy(g).to(next(self.parameters()).device)
        x = node_features_degrees(g)  # already on correct device
        h = torch.tanh(self.input_proj(x))

        for l in range(self.num_layers):
            outs = []
            for r in range(self.num_relations):
                agg = sum_aggregate(g.num_nodes, g.edge_index_by_rel[r], h) #mean
                outs.append(torch.tanh(self.branch_linears[l][r](agg)))
            comb_in = torch.stack([h] + outs, dim=0).sum(dim=0)#torch.cat([h] + outs, dim=-1)
            h = torch.tanh(self.combine_linears[l](comb_in))

        return h.sum(dim=0)