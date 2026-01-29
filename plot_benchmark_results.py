#imports + path setup

from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

REPO_ROOT = Path.cwd()  # adjust if needed
RESULTS_DIR = REPO_ROOT / "temporal-graph-nn" / "results_selth"#_26_01_2026"#"results_selth_16_01_2026"
json_files = sorted(p for p in RESULTS_DIR.rglob("*.json") if p.is_file())

def deep_merge(a, b):
    """Recursively merge dict b into dict a (returns a new dict)."""
    a = deepcopy(a)
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = v
    return a

runs = defaultdict(list)
for fp in json_files:
    runs[fp.parent].append(fp)

rows = []
errors = []

for run_dir, files in runs.items():
    merged = {}
    file_list = []

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            # If a JSON is a list store it under a key derived from filename
            # rather than exploding into many rows.
            if isinstance(obj, list):
                obj = {fp.stem: obj}
            elif not isinstance(obj, dict):
                obj = {fp.stem: obj}

            merged = deep_merge(merged, obj)
            file_list.append(str(fp))

        except Exception as e:
            errors.append({"file": str(fp), "error": repr(e)})

    # Normalize merged dict into a single flat row
    flat = pd.json_normalize(merged, sep=".")
    row = flat.iloc[0].to_dict()

    # provenance
    rel_run = run_dir.relative_to(RESULTS_DIR)
    row["__run_dir"] = str(rel_run)
    row["__n_files"] = len(files)
    row["__files"] = file_list

    rows.append(row)

df_a = pd.DataFrame(rows)

print("Runs:", len(df_a), "Columns:", df_a.shape[1])
print("JSON read errors:", len(errors))

# Now filtering makes sense (dataset exists in config.json and will be merged in)
df_a = df_a.dropna(subset=["dataset"]).reset_index(drop=True)

# Drop all-NaN columns (after filtering)
df_a = df_a.dropna(axis=1, how="all")

#  path setup
REPO_ROOT = Path.cwd()  # adjust if needed
RESULTS_DIR = REPO_ROOT / "ximp" / "results_selth"#_26_01_2026"#/ "ximp" / "results_selth_16_01_2026"
json_files = sorted(p for p in RESULTS_DIR.rglob("*.json") if p.is_file())

def deep_merge(a, b):
    """Recursively merge dict b into dict a (returns a new dict)."""
    a = deepcopy(a)
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = v
    return a

runs = defaultdict(list)
for fp in json_files:
    runs[fp.parent].append(fp)

rows = []
errors = []

for run_dir, files in runs.items():
    merged = {}
    file_list = []

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            # If a JSON is a list, store it under a key derived from filename
            # rather than exploding into many rows.
            if isinstance(obj, list):
                obj = {fp.stem: obj}
            elif not isinstance(obj, dict):
                obj = {fp.stem: obj}

            merged = deep_merge(merged, obj)
            file_list.append(str(fp))

        except Exception as e:
            errors.append({"file": str(fp), "error": repr(e)})

    # Normalize merged dict into a single flat row
    flat = pd.json_normalize(merged, sep=".")
    row = flat.iloc[0].to_dict()

    # provenance
    rel_run = run_dir.relative_to(RESULTS_DIR)
    row["__run_dir"] = str(rel_run)
    row["__n_files"] = len(files)
    row["__files"] = file_list

    rows.append(row)

df_b = pd.DataFrame(rows)

print("Runs:", len(df_b), "Columns:", df_b.shape[1])
print("JSON read errors:", len(errors))

# Now filtering makes sense (dataset exists in config.json and will be merged in)
df_b = df_b.dropna(subset=["config.task"]).reset_index(drop=True)

# Drop all-NaN columns (after filtering)
df_b = df_b.dropna(axis=1, how="all")

print(df_b.__run_dir)
df_b['dataset'] = df_b['__run_dir'].apply(lambda x : x.split('_')[0])

df_a = df_a[["dataset", "config.model_flavour", "config.prune_ratio", "results.test_ndcg" , "expressivity.pre_ratio", "model.compression_ratio"]]
df_b = df_b[["dataset", "config.repr_model", "config.prune_ratio", "expressivity.post_pruning.seperable_final_embeddings_ratio", 
             "performance.test_mae", "model.compression_ratio"]]
df_b.columns = ['dataset', 'model', 'prune', 'sep', 'test_performance', "compression"]
df_a.columns = ['dataset', 'model', 'prune', 'sep', 'test_performance', "compression"]




def minmax_01(s: pd.Series, higher_is_better: bool = True, eps: float = 0.0) -> pd.Series:
    """
    Min-max normalize to [eps, 1]. If higher_is_better=False, smaller values map to larger scores.
    eps=0.0 -> [0,1], eps=0.1 -> [0.1,1].
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)
    lo = x.min(skipna=True)
    hi = x.max(skipna=True)

    # handle constant/degenerate case
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # if all equal (or all NaN), make them all 1.0 (or NaN stays NaN)
        out = pd.Series(np.where(x.notna(), 1.0, np.nan), index=x.index)
        return eps + (1 - eps) * out

    norm = (x - lo) / (hi - lo)          # in [0,1]
    if not higher_is_better:
        norm = 1.0 - norm                # flip so higher = better

    return eps + (1 - eps) * norm


EPS = 0.0 

# df_a: NDCG (higher is better) — per dataset
df_a["test_performance"] = (
    df_a.groupby("dataset")["test_performance"]
        .transform(lambda s: minmax_01(s, higher_is_better=True, eps=EPS))
)

# df_b: MAE (lower is better) — per dataset
df_b["test_performance"] = (
    df_b.groupby("dataset")["test_performance"]
        .transform(lambda s: minmax_01(s, higher_is_better=False, eps=EPS))
)

df_new = pd.concat([df_a, df_b])
df_new["model"] = df_new["model"].replace({
    "T1": r"$TGNN_{loc}$",
    "T2": r"$TGNN_{glob}$",
})
df_new['dataset'] = df_new['dataset'].apply(lambda x : x.split('-')[1] if '-' in x else x)
df_new['model'] = df_new['model']+'/'+df_new['dataset']


for col in ["sep", "compression"]:
    df_new[col] = (
        df_new.groupby("model")[col]
            .transform(lambda s: minmax_01(s, higher_is_better=True, eps=EPS))
    )
df_new_full = dc(df_new)

df_new = df_new[df_new['prune'] >=0] # only need 0.0 for probability


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

alpha_sig = 0.05

# 3 pairwise correlations:
pairs = [
    ("sep", "test_performance"),
    ("sep", "compression"),
    ("test_performance", "compression"),
]

pair_label = {
    ("sep", "test_performance"): "Sep. vs Perf.",
    ("sep", "compression"): "Sep. vs Comp.",
    ("test_performance", "compression"): "Perf. vs Comp.",
}

def spearman_corr(sub: pd.DataFrame, a: str, b: str):
    x = sub[a]
    y = sub[b]
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 2:
        return np.nan, np.nan, n
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), n

# -------------------------------------------------------------------
# Compute correlations per model (marker encodes model, color encodes pair)
# -------------------------------------------------------------------
rows = []
for model_name, sub in df_new.groupby("model"):
    for a, b in pairs:
        rho, p, n = spearman_corr(sub, a, b)
        rows.append({"model": model_name, "a": a, "b": b, "pair": (a, b), "rho": rho, "p": p, "n": n})

corr_df = pd.DataFrame(rows).dropna(subset=["rho", "p"]).copy()
corr_df["neglog10p"] = -np.log10(corr_df["p"].clip(lower=1e-300))

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
S = 55
EDGE_LW = 0.4
GRID_LW = 0.6
THR_LW = 1.0

# color encodes which correlation pair
color_map = {
    ("sep", "test_performance"): "tab:brown",
    ("sep", "compression"): "tab:blue",
    ("test_performance", "compression"): "tab:orange",
}

# marker encodes model (auto-assign from a cycle)
marker_cycle = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "h", "*", "p", "8"]
models = list(corr_df["model"].unique())
marker_map = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(models)}

# -------------------------------------------------------------------
# Plot (volcano-style): x=rho, y=-log10(p)
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.1, 1.5), constrained_layout=True)

for (a, b) in pairs:
    for m in models:
        d = corr_df[(corr_df["pair"] == (a, b)) & (corr_df["model"] == m)]
        if d.empty:
            continue
        ax.scatter(
            d["rho"], d["neglog10p"],
            s=S,
            marker=marker_map[m],
            alpha=1.0,
            color=color_map.get((a, b), "tab:gray"),
            edgecolors="black",
            linewidths=EDGE_LW
        )

# significance threshold (horizontal)
ax.axhline(-np.log10(alpha_sig), linewidth=THR_LW, linestyle="--", alpha=0.6)
# rho=0 reference (vertical)
ax.axvline(0.0, linewidth=THR_LW, linestyle="--", alpha=0.6)

ax.set_title(" ")
ax.set_xlabel("Spearman's $\\rho_{sm}$.")
ax.set_ylabel(r"$-\log_{10}(p)$")
ax.grid(True, linestyle="-", linewidth=GRID_LW, alpha=0.25)
ax.set_axisbelow(True)

# sensible limits
ax.set_xlim(-1.05, 1.05)
#ymax = max(2.0, float(corr_df["neglog10p"].max()) + 1.0) if len(corr_df) else 2.0
ax.set_ylim(7*1e-1, 10**2+1000)#min(22, ymax))
ax.set_yscale('log')


# -------------------------------------------------------------------
# Legends (pair colors + model markers)
# -------------------------------------------------------------------
pair_handles = [
    Line2D([0], [0], marker="o", linestyle="None", markersize=7,
           markerfacecolor=color_map[(a, b)], markeredgecolor="black",
           label=pair_label.get((a, b), f"{a} vs {b}"))
    for (a, b) in pairs
]

model_handles = [
    Line2D([0], [0], marker=marker_map[m], linestyle="None", markersize=7,
           markerfacecolor="white", markeredgecolor="black",
           label=str(m))
    for m in models
]

# Put legends to the right; adjust for many models
fig.subplots_adjust(right=0.72)

leg1 = ax.legend(
    handles=pair_handles,
    loc="upper left",
    bbox_to_anchor=(0.00, 1.33),
    borderaxespad=0.0,
    frameon=True,
    title=None,
    ncol=3
)
ax.add_artist(leg1)

ax.legend(
    handles=model_handles,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0.0,
    frameon=True,
    title=None,
    ncol=1
)

out_path = "corr_volcano_models_pairs.pdf"
plt.savefig(out_path, bbox_inches="tight", format="pdf", dpi=300)
#plt.show()
print("Saved:", out_path)


def conditional_prob_per_model(
    df_new_full: pd.DataFrame,
    *,
    model_col="model",
    prune_col="prune",
    sep_col="sep",
    perf_col="test_performance",
    dense_value=0.0,
    pruned_condition=lambda p: p > 0,
    margin=0.02,
    bins=None,
    quantile_bins=None,
):
    df = df_new_full.copy()

    dense  = df[df[prune_col] == dense_value].copy()
    pruned = df[df[prune_col].apply(pruned_condition)].copy()

    # baseline dense performance per model (model already includes dataset in df)
    dense_base = (
        dense.groupby([model_col], dropna=False)[perf_col]
             .mean()
             .rename("dense_perf")
             .reset_index()
    )

    pruned = pruned.merge(dense_base, on=[model_col], how="inner")
    pruned["as_good"] = pruned[perf_col] >= (pruned["dense_perf"] - margin)
    #print(pruned["dense_perf"], pruned[perf_col], flush=True)
    # bin sep globally
    x = pd.to_numeric(pruned[sep_col], errors="coerce")
    if quantile_bins is not None:
        pruned["sep_bin"] = pd.qcut(x, q=quantile_bins, duplicates="drop")
    else:
        pruned["sep_bin"] = pd.cut(x, bins=(bins if bins is not None else 10), include_lowest=True)

    out = (
        pruned.groupby([model_col, "sep_bin"], observed=True, dropna=False)  # observed=True removes empty bins
              .agg(
                  p_as_good=("as_good", "mean"),
                  n=("as_good", "size"),
                  sep_min=(sep_col, "min"),
                  sep_max=(sep_col, "max"),
                  dense_perf=("dense_perf", "first"),
                  pruned_perf_mean=(perf_col, "mean"),
              )
              .reset_index()
              .sort_values([model_col, "sep_min", "sep_max"])
    )

    return out, pruned



# Example
out_model, pruned_with_flags = conditional_prob_per_model(
    df_new_full,
    margin=1e-4,#1e-4,#0.001,#1e-4,#0.00,
    #quantile_bins= 15,
    bins=np.linspace(0.0,1,25, endpoint=True)
)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

alpha_sig = 0.05

# =========================================================
# (A) LEFT PANEL: volcano correlations per model
# =========================================================
pairs = [
    ("sep", "test_performance"),
    ("sep", "compression"),
    ("test_performance", "compression"),
]
pair_label = {
    ("sep", "test_performance"): "Sep., Perf.", #"Sep. vs Perf."
    ("sep", "compression"): "Sep., Comp.",
    ("test_performance", "compression"): "Perf., Comp.",
}

def spearman_corr(sub: pd.DataFrame, a: str, b: str):
    x = sub[a]
    y = sub[b]
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 2:
        return np.nan, np.nan, n
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), n

# keep only prune >= 0 (dense included)
df_new = df_new[df_new["prune"] >= 0].copy()

rows = []
for model_name, sub in df_new.groupby("model"):
    for a, b in pairs:
        rho, p, n = spearman_corr(sub, a, b)
        rows.append({"model": model_name, "pair": (a, b), "rho": rho, "p": p, "n": n})

corr_df = pd.DataFrame(rows).dropna(subset=["rho", "p"]).copy()
corr_df["neglog10p"] = -np.log10(corr_df["p"].clip(lower=1e-300))

# styling (shared)
S = 55
EDGE_LW = 0.4
GRID_LW = 0.6
THR_LW = 1.0

color_map_left = {
    ("sep", "test_performance"): "tab:brown",
    ("sep", "compression"): "tab:blue",
    ("test_performance", "compression"): "tab:orange",
}
color_right = "tab:purple"


# marker encodes model (shared across BOTH panels)
marker_cycle = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "h", "*", "p", "8"]
models = list(pd.unique(df_new["model"]))  # all models, even if some correlations drop
marker_map = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(models)}

# =========================================================
# (B) RIGHT PANEL: p_as_good vs sep-bin midpoint per model
# Requires out_model from conditional_prob_per_model(...):
# columns: model, sep_bin, p_as_good, n, sep_min, sep_max, ...
# =========================================================
assert "out_model" in globals(), "Expected `out_model` in globals() (from conditional_prob_per_model)."

tmp = out_model.copy()
tmp["p_as_good"] = pd.to_numeric(tmp["p_as_good"], errors="coerce")
tmp["n"] = pd.to_numeric(tmp["n"], errors="coerce").fillna(0).astype(int)
tmp["sep_min"] = pd.to_numeric(tmp["sep_min"], errors="coerce")
tmp["sep_max"] = pd.to_numeric(tmp["sep_max"], errors="coerce")
tmp["sep_mid"] = 0.5 * (tmp["sep_min"] + tmp["sep_max"])

# optional: drop empty bins
tmp = tmp[tmp["n"] > 0].copy()

# (optional) size encode sample count (keeps style but adds information)
# for fixed size, just set size = S.
nmax = max(1, int(tmp["n"].max() if len(tmp) else 1))
def size_from_n(n):
    # map n in [1,nmax] -> [0.7S, 1.3S]
    return (0.7 + 0.6 * (n / nmax)) * S

# =========================================================
# FIGURE: 1 row, 2 columns, shared marker legend
# =========================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.1, 1.7),   # adjust for more height
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1.0, 1.0],
                 "wspace": 0.56 #54
                }
)

# ---------- Left: volcano ----------
for (a, b) in pairs:
    dpair = corr_df[corr_df["pair"] == (a, b)]
    for m in models:
        d = dpair[dpair["model"] == m]
        if d.empty:
            continue
        ax1.scatter(
            d["rho"], d["neglog10p"],
            s=S,
            marker=marker_map[m],
            alpha=1.0,
            color=color_map_left.get((a, b), "tab:gray"),
            edgecolors="black",
            linewidths=EDGE_LW
        )

ax1.axhline(-np.log10(alpha_sig), linewidth=THR_LW, linestyle="--", alpha=0.6)
ax1.axvline(0.0, linewidth=THR_LW, linestyle="--", alpha=0.6)
ax1.set_xlabel("Spearman's $\\rho_{sm}$.")
ax1.set_ylabel(r"$-\log_{10}(p)$")
ax1.grid(True, linestyle="-", linewidth=GRID_LW, alpha=0.25)
ax1.set_axisbelow(True)
ax1.set_xlim(-1.05, 1.05)

# log-y look
ax1.set_yscale("log")
ax1.set_ylim(7e-1, 1e2 + 1000)

ax1.set_title("(a)")

# ---------- Right: p_as_good vs sep ----------
for m in models:
    dm = tmp[tmp["model"] == m]
    
    if dm.empty:
        continue

    sizes = np.array([size_from_n(int(n)) for n in dm["n"].to_numpy()])
    
    ax2.scatter(
        dm["sep_mid"], dm["p_as_good"],
        s=S,#s=sizes,
        marker=marker_map[m],
        alpha=1.0,
        color=color_right,
        edgecolors="black",
        linewidths=EDGE_LW
    )
    
    #ax2.plot(dm["sep_mid"].values, dm["p_as_good"].values)


ax2.set_xlabel("Separability") #(bin midpoint)
ax2.set_ylabel(r"$P (\mathrm{WT}|\mathrm{Sep.})$")#(r"$p_{\mathrm{as\ good}}$")
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, linestyle="-", linewidth=GRID_LW, alpha=0.25)
ax2.set_axisbelow(True)
ax2.set_title("(b)")

# =========================================================
# Legends:
# - Pair legend (colors) above panel (a)
# - Single shared MODEL legend (markers) on the right of the whole figure
# =========================================================
pair_handles = [
    Line2D([0], [0], marker="o", linestyle="None", markersize=7,
           markerfacecolor=color_map_left[(a, b)], markeredgecolor="black",
           label=pair_label[(a, b)])
    for (a, b) in pairs
]
right_handle = Line2D(
    [0], [0],
    marker="o", linestyle="None", markersize=7,
    markerfacecolor=color_right, markeredgecolor="black",
    label=r"$p_{\mathrm{as\ good}}$"
)
right_handle = Line2D(
    [0], [0],
    marker="o", linestyle="None", markersize=7,
    markerfacecolor=color_right, markeredgecolor="black",
    label=r"$P (\mathrm{WT}|\mathrm{Sep.})$"
)


pair_handles = pair_handles + [right_handle]

model_handles = [
    Line2D([0], [0], marker=marker_map[m], linestyle="None", markersize=7,
           markerfacecolor="white", markeredgecolor="black",
           label=str(m))
    for m in models
]

# leave space for the shared model legend
fig.subplots_adjust(right=0.575) #0.58

# pair legend above left panel
leg_pairs = ax1.legend(
    handles=pair_handles,
    loc="upper left",
    bbox_to_anchor=(3.925, 1.0),#4.075
    borderaxespad=0.0,
    frameon=True,
    ncol=1,
    handlelength=1.0
)
ax1.add_artist(leg_pairs)
ax2.yaxis.labelpad = 0
ax1.yaxis.labelpad = 0

# shared model legend (markers) for BOTH panels
fig.legend(
    handles=model_handles,
    loc="upper left",
    bbox_to_anchor=(0.5825, 0.88),#0.60
    borderaxespad=0.0,
    frameon=True,
    ncol=1,
    handlelength=1.0
)

out_path = "corr_volcano_and_pasgood.pdf"
plt.savefig(out_path, bbox_inches="tight", format="pdf", dpi=300)
#plt.show()
print("Saved:", out_path)

