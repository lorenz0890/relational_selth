#!/usr/bin/env python3
import re
import os
from pathlib import Path

results = []

for logfile in sorted(Path("logs").glob("*.log")):
    content = logfile.read_text()
    
    # Extract dataset and sparsity
    name = logfile.stem
    parts = name.split("_")
    
    # Determine dataset type and name
    if "esol" in name.lower() or "freesolv" in name.lower() or "lipo" in name.lower():
        dataset_type = "MoleculeNet"
        dataset = parts[0]
    elif "admet" in name:
        dataset_type = "Polaris-ADMET"
        dataset = "_".join(parts[1:-1])
    elif "potency" in name:
        dataset_type = "Polaris-Potency"
        dataset = "_".join(parts[1:-1])
    else:
        continue
    
    # Extract sparsity
    sparsity_match = re.search(r'(\d+)pct', name)
    if sparsity_match:
        sparsity = int(sparsity_match.group(1))
    else:
        continue
    
    # Extract N (Post-training)
    n_match = re.search(r'Post-training.*?N \(unique graphs\): (\d+) / (\d+) \(([0-9.]+)%\)', content, re.DOTALL)
    if n_match:
        n_unique = int(n_match.group(1))
        n_total = int(n_match.group(2))
        uniqueness = float(n_match.group(3))
    else:
        n_unique = n_total = uniqueness = None
    
    # Extract Test MAE
    mae_match = re.search(r'Test MAE: ([0-9.]+)', content)
    if mae_match:
        test_mae = float(mae_match.group(1))
    else:
        test_mae = None
    
    # Extract Compression
    comp_match = re.search(r'Compression: ([0-9.]+)×', content)
    if comp_match:
        compression = float(comp_match.group(1))
    else:
        compression = None
    
    if n_unique is not None:
        results.append({
            'type': dataset_type,
            'dataset': dataset,
            'sparsity': sparsity,
            'n_unique': n_unique,
            'n_total': n_total,
            'uniqueness': uniqueness,
            'test_mae': test_mae,
            'compression': compression
        })

# Group by dataset
datasets = {}
for r in results:
    key = (r['type'], r['dataset'])
    if key not in datasets:
        datasets[key] = []
    datasets[key].append(r)

# Print summary by dataset type
for dataset_type in ["MoleculeNet", "Polaris-ADMET", "Polaris-Potency"]:
    print(f"\n{'='*80}")
    print(f"{dataset_type} Results")
    print(f"{'='*80}\n")
    
    for key, data in sorted(datasets.items()):
        if key[0] != dataset_type:
            continue
        
        dataset_name = key[1]
        data = sorted(data, key=lambda x: x['sparsity'])
        
        print(f"## {dataset_name}")
        print(f"| Sparsity | N (Post) | Total | Uniqueness | Test MAE | Compression |")
        print(f"|----------|----------|-------|------------|----------|-------------|")
        
        for r in data:
            print(f"| {r['sparsity']}% | {r['n_unique']} | {r['n_total']} | {r['uniqueness']:.1f}% | {r['test_mae']:.3f} | {r['compression']:.2f}× |")
        
        # Check SELTH validation
        baseline_n = data[0]['n_unique'] if data else None
        if baseline_n:
            preserved_70 = any(r['sparsity'] == 70 and r['n_unique'] == baseline_n for r in data)
            n_90 = next((r['n_unique'] for r in data if r['sparsity'] == 90), None)
            
            if preserved_70:
                if n_90 and abs(n_90 - baseline_n) <= baseline_n * 0.05:
                    print(f"\n**SELTH Status**: ✅ **PERFECT** - N preserved through 90%\n")
                else:
                    print(f"\n**SELTH Status**: ✅ **VALIDATED** - N preserved through 70%\n")
            else:
                # Check variance
                max_n = max(r['n_unique'] for r in data)
                min_n = min(r['n_unique'] for r in data)
                if (max_n - min_n) / max_n < 0.1:
                    print(f"\n**SELTH Status**: ✅ **VALIDATED** - N variation < 10%\n")
                else:
                    print(f"\n**SELTH Status**: ⚠️  NEEDS REVIEW - N varies by {((max_n - min_n) / max_n * 100):.1f}%\n")

print(f"\n{'='*80}")
print("Summary Statistics")
print(f"{'='*80}\n")
print(f"Total experiments completed: {len(results)}")
print(f"Dataset types: {len(set(r['type'] for r in results))}")
print(f"Unique datasets: {len(datasets)}")
print(f"\nSparsity levels tested: {sorted(set(r['sparsity'] for r in results))}")
