#!/usr/bin/env bash
set -euo pipefail

# X: linspace from 0.10 to 0.99 with 30 values (inclusive)
mapfile -t X_VALUES < <(python - <<'PY'
import numpy as np
for x in np.linspace(0.0, 1.0, 200, endpoint=False):
    print(f"{x:.8f}")
#x=0.0
print(f"{x:.8f}")
PY
)

# Z: seeds
Z_VALUES=(561 1105 1729 2465 2821) # 6601 8911 10585 15841 29341) #Carmichael numbers #(3 5 17 257 65537) # Fermat numbers

# Progress bookkeeping
total=$(( ${#X_VALUES[@]} * ${#Z_VALUES[@]} ))
done_count=0

echo "Total runs: ${total}"

for x in "${X_VALUES[@]}"; do
  for z in "${Z_VALUES[@]}"; do
    done_count=$((done_count + 1))
    pct=$(( 100 * done_count / total ))

    printf "[%3d%%] (%d/%d) prune_ratio=%s seed=%s\n" \
      "${pct}" "${done_count}" "${total}" "${x}" "${z}"

    python ./ximp/run_selth_experiment.py \
      --prune_ratio "${x}" \
      --seed "${z}" \
      --task admet \
      --target_task "HLM" \
      --repr_model XIMP \
      --hidden_channels 16 \
      --out_channels 16 \
      --encoding_dim 16 \
      --proj_hidden_dim 16 \
      --epochs 50 \
      --batch_size 128 \
      --num_cv_folds 2 \
      --num_cv_bins 1 \
      --scaffold_split_val_sz 0.1 \
      --use_erg True \
      --use_jt True \
      --jt_coarsity 1 \
      --rg_embedding_dim 16

    python ./ximp/run_selth_experiment.py \
      --prune_ratio "${x}" \
      --seed "${z}" \
      --task potency \
      --target_task "pIC50 (MERS-CoV Mpro)" \
      --repr_model XIMP \
      --hidden_channels 16 \
      --out_channels 16 \
      --encoding_dim 16 \
      --proj_hidden_dim 16 \
      --epochs 50 \
      --batch_size 128 \
      --num_cv_folds 2 \
      --num_cv_bins 1 \
      --scaffold_split_val_sz 0.1 \
      --use_erg True \
      --use_jt True \
      --jt_coarsity 1 \
      --rg_embedding_dim 16
  done
done

echo "Done. Ran ${total} jobs."
