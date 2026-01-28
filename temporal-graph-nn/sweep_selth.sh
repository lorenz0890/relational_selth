#!/usr/bin/env bash
set -euo pipefail

# ---- Grid definitions ----
mapfile -t X_VALUES < <(python - <<'PY'
import numpy as np
xs = np.linspace(0.0, 1.0, 200, endpoint=False)
for x in xs:
    print(f"{x:.8f}")
#x = 0.0
print(f"{x:.8f}")
PY
)

Y_VALUES=(T2 T1)
Z_VALUES=(561 1105 1729 2465 2821) # 6601 8911 10585 15841 29341) #Carmichael numbers #(3 5 17 257 65537) # Fermat numbers
D_VALUES=(16) #This wont be used bc of a bug in the codebase (it's set in hyper), but kept for consistency. Marco should fix it.

# ---- Progress bookkeeping ----
total=$(( ${#X_VALUES[@]} * ${#Y_VALUES[@]} * ${#Z_VALUES[@]} * ${#D_VALUES[@]} ))
done_count=0

echo "Total runs: ${total}"

# ---- Run all combinations ----
for x in "${X_VALUES[@]}"; do
  for y in "${Y_VALUES[@]}"; do
    for z in "${Z_VALUES[@]}"; do
      for d in "${D_VALUES[@]}"; do
        done_count=$((done_count + 1))
        pct=$(( 100 * done_count / total ))

        printf "[%3d%%] (%d/%d) prune_ratio=%s flavour=%s seed=%s embed_dim=%s\n" \
          "${pct}" "${done_count}" "${total}" "${x}" "${y}" "${z}" "${d}"

        python run_selth_experiment.py \
          --dataset tgbn-trade \
          --flavour "${y}" \
          --embed_dim "${d}" \
          --prune_ratio "${x}" \
          --seed "${z}" \
          --max_epochs 10 \
          --patience 100
      done
    done
  done
done

echo "Done. Ran ${total} jobs."
