import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch import nn


# ----------------------------
# 1) Tracker: N (distinct inputs) + S_min (min L0 separation)
# ----------------------------

@dataclass
class L0Stats:
    S_min: float
    S_mean: float
    S_std: float
    S_max: float


class ExpressivityTracker:
    """
    Distinctness + minimal L0 separation under a tolerance.
    Treat each row as one "input vector".
    """
    def __init__(self, tolerance: float = 1e-6, device: str = "cpu"):
        self.tol = float(tolerance)
        self.device = device

    def _as_2d(self, x: torch.Tensor) -> torch.Tensor:
        # Interpret last dim as feature dim; flatten everything else into batch.
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x.reshape(-1, x.size(-1))

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize so that "equal within tol" -> equal integer representation.
        This is a pragmatic proxy. If you want stricter behavior, change rounding rule.
        """
        x2 = self._as_2d(x).to(torch.float32)
        q = torch.round(x2 / self.tol).to(torch.int32)
        return q

    @torch.no_grad()
    def compute_N(self, x: torch.Tensor) -> int:
        q = self._quantize(x).to(self.device)
        # unique rows
        uq = torch.unique(q, dim=0)
        return int(uq.size(0))

    @torch.no_grad()
    def compute_S_min_l0(
        self,
        x: torch.Tensor,
        max_unique_for_exact: int = 6000,
        approx_pairs: int = 20000,
        seed: int = 0,
    ) -> Tuple[float, L0Stats]:
        """
        Returns:
          S_min: minimal L0 distance between ANY TWO DISTINCT observed inputs (after quantization)
          plus simple distribution stats over the sampled/exact pairwise distances.

        Exact all-pairs is O(U^2 * d). If U is large, we approximate by random pairing.
        """
        q = self._quantize(x).to(self.device)
        uq = torch.unique(q, dim=0)  # shape [U, d]
        U, d = uq.size(0), uq.size(1)

        if U <= 1:
            stats = L0Stats(S_min=float("inf"), S_mean=0.0, S_std=0.0, S_max=0.0)
            return float("inf"), stats

        def l0_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a: [B,d], b: [B,d] -> [B]
            return (a != b).sum(dim=1).to(torch.float32)

        # Exact mode if small enough
        if U <= max_unique_for_exact:
            S_min = float(d)
            dists_for_stats: List[torch.Tensor] = []

            # loop over anchors; compare to all subsequent
            for i in range(U - 1):
                anchor = uq[i].unsqueeze(0)                # [1,d]
                others = uq[i + 1 :]                       # [U-i-1,d]
                dist_i = (others != anchor).sum(dim=1)     # [U-i-1]
                # minimal positive distance (all are distinct, so >= 1)
                min_i = int(dist_i.min().item())
                if min_i < S_min:
                    S_min = float(min_i)
                    if S_min == 1.0:
                        # can't get smaller than 1 among distinct vectors
                        pass
                dists_for_stats.append(dist_i.to(torch.float32))

            all_d = torch.cat(dists_for_stats, dim=0)
            stats = L0Stats(
                S_min=float(S_min),
                S_mean=float(all_d.mean().item()),
                S_std=float(all_d.std(unbiased=False).item()),
                S_max=float(all_d.max().item()),
            )
            return float(S_min), stats

        # Approx mode: random pairs
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        # sample indices for pairs
        i1 = torch.randint(0, U, (approx_pairs,), generator=g, device=self.device)
        i2 = torch.randint(0, U, (approx_pairs,), generator=g, device=self.device)

        # avoid i1 == i2 (resample those positions once)
        same = (i1 == i2)
        if same.any():
            i2[same] = (i2[same] + 1) % U

        dists = l0_dist(uq[i1], uq[i2])
        S_min = float(dists.min().item())

        stats = L0Stats(
            S_min=float(S_min),
            S_mean=float(dists.mean().item()),
            S_std=float(dists.std(unbiased=False).item()),
            S_max=float(dists.max().item()),
        )
        return float(S_min), stats


# ----------------------------
# 2) Hook selection logic: "MLP or Linear, but not Linear inside an MLP"
# ----------------------------

def _is_mlp_sequential(m: nn.Module) -> bool:
    # Treat any Sequential containing at least one Linear as an "MLP"
    if not isinstance(m, nn.Sequential):
        return False
    return any(isinstance(sm, nn.Linear) for sm in m.modules())


def _collect_linear_descendants(root: nn.Module) -> set:
    ids = set()
    for sm in root.modules():
        if isinstance(sm, nn.Linear):
            ids.add(id(sm))
    return ids


def get_ximp_modules_to_hook(model: nn.Module) -> List[Tuple[str, nn.Module, str]]:
    """
    Returns list of (name, module, kind) where kind in {"MLP", "Linear"}.

    - Hooks ALL "MLP" = Sequential(...) that contains Linear(s)
    - Hooks ALL Linear layers NOT nested under any such Sequential MLP
    """
    mlp_linears_ids = set()

    # First pass: find all MLP Sequentials and mark their Linear descendants for exclusion
    for name, m in model.named_modules():
        if _is_mlp_sequential(m):
            mlp_linears_ids |= _collect_linear_descendants(m)

    targets: List[Tuple[str, nn.Module, str]] = []
    for name, m in model.named_modules():
        if _is_mlp_sequential(m):
            targets.append((name, m, "MLP"))
        elif isinstance(m, nn.Linear) and (id(m) not in mlp_linears_ids):
            targets.append((name, m, "Linear"))

    return targets


# ----------------------------
# 3) Measurement for XIMP
# ----------------------------

@torch.no_grad()
def measure_expressivity_ximp(
    model: nn.Module,
    loader,  # PyG DataLoader (yields Batch objects) or any iterable of "data"
    device: torch.device,
    logger=None,
    max_batches: int = 1,
    tolerance: float = 1e-6,
    max_unique_for_exact: int = 6000,
    approx_pairs: int = 20000,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Measures, for each:
      - MLP (Sequential with Linear(s), e.g., atom_convs.*.nn and rg_convs.*.*.nn)
      - standalone Linear (not inside such a Sequential)
    the following on OBSERVED INPUTS (i.e., tensors *fed into* that transform):
      - N: number of distinct inputs (under tolerance)
      - S_min: minimal L0 separation between two distinct inputs (under tolerance)
    """

    model.eval()
    tracker = ExpressivityTracker(tolerance=tolerance, device="cpu")  # compute on CPU

    # Storage: module_name -> list[Tensor] of inputs
    measurements: Dict[str, List[torch.Tensor]] = {}

    hooks = []

    def make_pre_hook(name: str):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            # detach and move to CPU for later aggregation
            x_cpu = x.detach().to("cpu")
            measurements.setdefault(name, []).append(x_cpu)
        return _hook

    # Register hooks
    targets = get_ximp_modules_to_hook(model)
    if logger:
        logger.debug(f"Registering {len(targets)} hooks (MLP + standalone Linear).")
    for name, module, kind in targets:
        # pre-hook captures inputs before transformation
        h = module.register_forward_pre_hook(make_pre_hook(f"{kind}:{name}"))
        hooks.append(h)

    # Forward over dataset
    batch_count = 0
    for data in loader:
        if max_batches is not None and batch_count >= max_batches:
            break
        data = data.to(device)
        _ = model(data)
        batch_count += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute metrics per module
    results: Dict[str, Dict[str, Any]] = {}
    for key, xs in measurements.items():
        if not xs:
            continue
        X = torch.cat([x.reshape(-1, x.size(-1)) if x.dim() > 1 else x.unsqueeze(0) for x in xs], dim=0)

        N = tracker.compute_N(X)
        S_min, S_stats = tracker.compute_S_min_l0(
            X,
            max_unique_for_exact=max_unique_for_exact,
            approx_pairs=approx_pairs,
            seed=seed,
        )

        results[key] = {
            "N": N,
            "total_inputs": int(X.size(0)),
            "input_dim": int(X.size(1)),
            "S_min_l0": S_min,
            "S_mean_l0": S_stats.S_mean,
            "S_std_l0": S_stats.S_std,
            "S_max_l0": S_stats.S_max,
            "uniqueness_ratio": (N / X.size(0)) if X.size(0) > 0 else 0.0,
        }

        if logger:
            logger.debug(
                f"{key} | N={N}/{X.size(0)} ({100.0*N/max(1,X.size(0)):.1f}%), "
                f"S_min(L0)={S_min}"
            )

    return results


# ----------------------------
# 4) Example usage (PyG)
# ----------------------------
"""
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Ximp(...).to(device)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

res = measure_expressivity_ximp(
    model=model,
    loader=loader,
    device=device,
    logger=logger,
    max_batches=5,
    tolerance=1e-6,
    max_unique_for_exact=4000,  # set lower if things blow up
    approx_pairs=20000,
    seed=0,
)

# res keys look like:
# "MLP:atom_convs.0.nn", "MLP:rg_convs.1.3.nn", "Linear:rg2raw_lins.0.2", ...
"""
