#!/usr/bin/env python3
"""
SELTH Validation Experiment Runner for XIMP (Molecular GNNs)

Validates the Strong Expressive Lottery Ticket Hypothesis (SELTH) on XIMP architecture with expressivity tracking.

Usage:
    python run_selth_experiment.py --repr_model GIN --prune_ratio 0.9 --seed 42
    python run_selth_experiment.py --repr_model XIMP --prune_ratio 0.5 --seed 42 --use_erg True

"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'shared-experiments'))

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch_geometric.loader import DataLoader
import pandas as pd

# XIMP imports
from src.models import TrainerModel, create_proj_model, create_repr_model
from src.data import MoleculeNetDataset, PolarisDataset
from src.utils import scaffold_split

# Pruning utilities
from pruning import (
    prune_at_initialization,
    apply_pruning_masks,
    get_sparsity_stats,
    count_parameters
)

from src.metric_utils import *

class SELTHXIMPExperiment:
    """SELTH validation experiment for XIMP molecular GNNs."""

    def __init__(self, params: dict):
        self.params = params
        self.device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        self.output_dir = self._create_output_dir()

        # Setup logging
        self.log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.log(f"SELTH XIMP Experiment")
        self.log(f"Output: {self.output_dir}")
        self.log(f"Device: {self.device}")

    def _create_output_dir(self) -> Path:
        """Create experiment output directory."""
        model_name = self.params['repr_model']
        prune_pct = prune_pct = str(self.params['prune_ratio'] * 100).replace('.', '_')#int(self.params['prune_ratio'] * 100)
        seed = self.params['seed']
        task = self.params['task']

        dir_name = f"{task}_{model_name}_prune{prune_pct}pct_seed{seed}"
        output_dir = Path('results_selth') / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)

        return output_dir

    def log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def _init_dataset(self):
        """Initialize datasets."""
        self.log("\nInitializing dataset...")

        if self.params["task"] in ["potency", "selectivity"]:
            dataset = PolarisDataset(
                root=f"./data/polaris/{self.params['task']}/",
                task=self.params["task"],
                target_task=self.params["target_task"],
                use_erg=self.params["use_erg"], #Lorenz: we need to pass these variables
                use_jt=self.params["use_jt"], #Lorenz: we need to pass these variables
                jt_coarsity = self.params["jt_coarsity"], #Lorenz: we need to pass these variables

            )
        elif self.params["task"] == "admet":
            dataset = PolarisDataset(
                root="./data/polaris/admet/",
                task=self.params["task"],
                target_task=self.params["target_task"],
                use_erg=self.params["use_erg"], #Lorenz: we need to pass these variables
                use_jt=self.params["use_jt"], #Lorenz: we need to pass these variables
                jt_coarsity = self.params["jt_coarsity"], #Lorenz: we need to pass these variables
            )
        else:
            dataset_wrapper = MoleculeNetDataset(
                root="./data/moleculenet/",
                target_task=self.params["target_task"],
                use_erg=self.params["use_erg"],
                use_jt=self.params["use_jt"],
                jt_coarsity=self.params["jt_coarsity"],
            )
            dataset = dataset_wrapper.create_dataset()

        # Scaffold split
        train_scaffold, test_scaffold = scaffold_split(
            dataset,
            test_size=self.params["scaffold_split_val_sz"],
        )

        self.dataset = dataset
        self.train_scaffold = train_scaffold
        self.test_scaffold = test_scaffold

        self.log(f"Dataset: {self.params['task']}")
        self.log(f"Train size: {len(train_scaffold)}")
        self.log(f"Test size: {len(test_scaffold)}")

        # Store dataset info for expressivity
        if hasattr(dataset, '__len__'):
            self.num_graphs = len(dataset)
        else:
            self.num_graphs = len(train_scaffold) + len(test_scaffold)

    def _init_model(self):
        """Initialize model."""
        self.log("\nInitializing model...")

        # Create representation model
        repr_model = create_repr_model(self.params)

        # Create projection model
        proj_model = create_proj_model(self.params)

        # Combine into trainer model
        self.model = TrainerModel(
            repr_model=repr_model,
            proj_model=proj_model
        ).to(self.device)

        # Count parameters before pruning
        self.total_params_initial = sum(p.numel() for p in self.model.parameters())
        self.log(f"Model: {self.params['repr_model']}")
        self.log(f"Total parameters: {self.total_params_initial:,}")

    def measure_expressivity(self, dataloader, stage: str = ""):
        """
        Measure expressivity on molecular graphs.

        For molecular GNNs, we measure:
        - seperable_final_embeddings: Number of unique graph representations
        - S_min: Minimum separation between graph embeddings
        """
        self.log(f"\n{'='*80}")
        self.log(f"EXPRESSIVITY MEASUREMENT: {stage}")
        self.log(f"{'='*80}")

        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                # Get graph-level embeddings (after pooling)
                emb = self.model.repr_model(batch)
                if isinstance(emb, tuple):
                    emb = emb[0]  # Take first output if tuple
                embeddings.append(emb.cpu())

        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings, dim=0)

        # Compute seperable_final_embeddings (unique graphs)
        tolerance = 1e-4
        unique_count = 0
        used = torch.zeros(len(all_embeddings), dtype=torch.bool)

        for i in range(len(all_embeddings)):
            if used[i]:
                continue
            dists = torch.norm(all_embeddings - all_embeddings[i], dim=1)
            similar = dists < tolerance
            used |= similar
            unique_count += 1

        seperable_final_embeddings = unique_count
        total = len(all_embeddings)

        metrics = {
            #'seperable_final_embeddings': seperable_final_embeddings, #Lorenz: i renamed N for something more correct, N was not the same as in the temporal setting
            #'total_graphs': total,
            'seperable_final_embeddings_ratio': seperable_final_embeddings / total if total > 0 else 0.0, #Lorenz: i renamed uniqueness_ratio for something more correct, N was not the same as in the temporal setting
            #'embedding_dim': all_embeddings.shape[1],
            #'S_min': S_min, #These are all computed incorrectly
            #'S_mean': S_mean,
            #'S_std': S_std
        }

        self.log(f"seperable_final_embeddings (unique graphs): {seperable_final_embeddings} / {total} ({100*seperable_final_embeddings/total:.1f}%)")
        self.log(f"Embedding dimension: {all_embeddings.shape[1]}")

        return metrics

    def run(self) -> Dict[str, Any]:
        """Run SELTH validation experiment."""
        torch.manual_seed(self.params['seed'])
        np.random.seed(self.params['seed'])

        # Initialize dataset and model
        self._init_dataset()
        self._init_model()

        # Create dataloader for expressivity measurement
        expressivity_eval_dataloader = DataLoader( #Lorenz: Renamed this for clarity
            self.dataset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )

        # Apply pruning
        masks = None
        #if self.params['prune_ratio'] > 0:
        self.log(f"\n{'='*80}")
        self.log(f"PRUNING: {self.params['prune_ratio']*100:.0f}% random pruning")
        self.log(f"{'='*80}")

        start_time = time.time()
        masks = prune_at_initialization(self.model, self.params['prune_ratio'])
        pruning_time = time.time() - start_time

        total_params, active_params = count_parameters(self.model, masks)

        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Active parameters: {active_params:,}")
        self.log(f"Compression ratio: {total_params/active_params:.2f}×")
        self.log(f"Pruning time: {pruning_time:.2f}s")

        # Measure post-pruning expressivity
        expr_post_prune = self.measure_expressivity(expressivity_eval_dataloader, "Post-pruning (before training)") # What we get here is graph level seperability

        # Compare
        self.log(f"\n{'='*80}")
        self.log(f"COMPARISON: Dense vs Pruned")
        self.log(f"{'='*80}")

        #else:
            #expr_post_prune = expr_pre.copy()
            #active_params = self.total_params_initial

        # Training
        self.log(f"\n{'='*80}")
        self.log(f"TRAINING")
        self.log(f"{'='*80}")

        training_start = time.time()
        self.train_with_cv(masks)
        training_time = time.time() - training_start

        self.log(f"Training completed in {training_time:.1f}s")

        # Final evaluation
        self.log(f"\n{'='*80}")
        self.log(f"FINAL EVALUATION")
        self.log(f"{'='*80}")

        preds = self.predict(self.test_scaffold)
        preds = [pred[1] for pred in preds]
        mae = mean_absolute_error(preds, self.test_scaffold.y)

        self.log(f"Test MAE: {mae:.6f}")

        # Compile results
        results = {
            'config': {
                'task': self.params['task'],
                'repr_model': self.params['repr_model'],
                'prune_ratio': self.params['prune_ratio'],
                'seed': self.params['seed'],
            },
            'model': {
                'total_params': self.total_params_initial,
                'active_params': active_params,
                'compression_ratio': self.total_params_initial / active_params if active_params > 0 else 1.0
            },
            'performance': {
                'test_mae': mae,
                'mean_val_loss': self.params.get('mean_val_loss', 0),
                'std_val_loss': self.params.get('std_val_loss', 0),
            },
            'expressivity': {
                #'pre_pruning': expr_pre,
                'post_pruning': expr_post_prune,
                #'post_training': expr_post_train
            },
            'training_time_seconds': training_time
        }

        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.log(f"\nResults saved to: {self.output_dir}")

        return results

    def train_with_cv(self, masks=None):
        """Train with cross-validation (adapted from Trainer)."""
        smiles = self.train_scaffold.smiles
        labels = self.train_scaffold.y.view(-1).tolist()

        y_binned = pd.qcut(labels, q=self.params["num_cv_bins"], labels=False)
        '''
        skf = StratifiedKFold(n_splits=self.params["num_cv_folds"], shuffle=True, random_state=42)

        val_loss_list = []

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(smiles, y_binned)):
            self.log(f"\nFold {fold_idx + 1}/{self.params['num_cv_folds']}")

            # Reinitialize model for each fold
            self._init_model()
            if masks is not None:
                # Reapply pruning
                masks = prune_at_initialization(self.model, self.params['prune_ratio'], verbose=False)

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.params['lr'],
                weight_decay=self.params['weight_decay']
            )

            train_fold = self.train_scaffold[train_idx]
            valid_fold = self.train_scaffold[valid_idx]

            train_loader = DataLoader(train_fold, batch_size=self.params['batch_size'], shuffle=True)
            valid_loader = DataLoader(valid_fold, batch_size=self.params['batch_size'], shuffle=False)

            # Train for this fold
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            for epoch in range(self.params['epochs']):
                # Train
                self.model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()

                    pred = self.model(batch)
                    loss = nn.functional.l1_loss(pred, batch.y.view(-1, 1).float())

                    loss.backward()

                    # Reapply masks
                    if masks is not None:
                        apply_pruning_masks(self.model, masks)

                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validate
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in valid_loader:
                        batch = batch.to(self.device)
                        pred = self.model(batch)
                        loss = nn.functional.l1_loss(pred, batch.y.view(-1, 1).float())
                        val_loss += loss.item()

                val_loss /= len(valid_loader)

                if epoch % 10 == 0:
                    self.log(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.log(f"  Early stopping at epoch {epoch}")
                        break

            val_loss_list.append(best_val_loss)

        self.params['mean_val_loss'] = np.mean(val_loss_list)
        self.params['std_val_loss'] = np.std(val_loss_list)
        
        self.log(f"\nCV Results: Val Loss = {self.params['mean_val_loss']:.4f} ± {self.params['std_val_loss']:.4f}")
        '''
        # Final training on full train set
        self.log("\nFinal training on full train set...")
        self._init_model()
        if masks is not None:
            masks = prune_at_initialization(self.model, self.params['prune_ratio'], verbose=False)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay']
        )

        train_loader = DataLoader(self.train_scaffold, batch_size=self.params['batch_size'], shuffle=True)

        for epoch in range(self.params['epochs']):
            self.model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch)
                loss = nn.functional.l1_loss(pred, batch.y.view(-1, 1).float())
                loss.backward()

                if masks is not None:
                    apply_pruning_masks(self.model, masks)

                optimizer.step()

    def predict(self, dataset):
        """Predict on dataset."""
        self.model.eval()
        predictions = []

        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                for p, y in zip(pred.cpu().numpy(), batch.y.cpu().numpy()):
                    predictions.append((y[0], p[0]))

        return predictions


def main():
    parser = argparse.ArgumentParser(description='SELTH Validation for XIMP')

    # Core SELTH parameters
    parser.add_argument('--prune_ratio', type=float, required=True, help='Pruning ratio (0.0-0.9)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')

    # Task parameters
    parser.add_argument('--task', default='potency', help='Task name')
    parser.add_argument('--target_task', default='pIC50 (MERS-CoV Mpro)', help='Target task')

    # Model parameters
    parser.add_argument('--repr_model', default='GIN', help='Model: GIN, GCN, GAT, XIMP, HIMP')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels')
    parser.add_argument('--out_channels', type=int, default=64, help='Output channels')
    parser.add_argument('--encoding_dim', type=int, default=8, help='Encoding dimension')
    parser.add_argument('--proj_hidden_dim', type=int, default=64, help='Projection hidden dim')
    parser.add_argument('--out_dim', type=int, default=1, help='Output dimension')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')

    # Dataset parameters
    parser.add_argument('--num_cv_folds', type=int, default=5, help='CV folds')
    parser.add_argument('--num_cv_bins', type=int, default=10, help='CV bins')
    parser.add_argument('--scaffold_split_val_sz', type=float, default=0.1, help='Validation split')

    # XIMP parameters
    parser.add_argument('--use_erg', type=bool, default=False, help='Use ERG')
    parser.add_argument('--use_jt', type=bool, default=False, help='Use Junction Tree')
    parser.add_argument('--jt_coarsity', type=int, default=1, help='JT coarsity')
    parser.add_argument('--rg_embedding_dim', type=int, default=8, help='RG embedding dim')
    parser.add_argument('--radius', type=int, default=2, help='ECFP radius')

    args = parser.parse_args()
    params = vars(args)

    # Run experiment
    experiment = SELTHXIMPExperiment(params)
    results = experiment.run()

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Model: {params['repr_model']}")
    print(f"Pruning: {params['prune_ratio']*100:.0f}%")
    print(f"Test MAE: {results['performance']['test_mae']:.6f}")
    print(f"Compression: {results['model']['compression_ratio']:.2f}×")
    #print(f"Expressivity (N): {results['expressivity']['post_training']['seperable_final_embeddings']}/{results['expressivity']['post_training']['total_graphs']}")
    print("="*80)


if __name__ == '__main__':
    main()
