#!/usr/bin/env python3
"""
SELTH Validation Experiment Runner for Temporal GNNs

Validates the Strong Expressive Lottery Ticket Hypothesis (SELTH) on temporal
graph neural networks with comprehensive metrics tracking.

"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any

# Add paths
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'shared-experiments'))
sys.path.append(str(Path(__file__).parent / 'selth'))

import torch
from torch import optim
import torch.nn.functional as F
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
import numpy as np

# Original model code (unchanged)
from models import Model
import hyper

# Pruning utilities
from pruning import (
    prune_at_initialization,
    apply_pruning_masks,
    get_sparsity_stats,
    count_parameters
)

# Gradient diversity is optional (not implemented yet)
try:
    from selth.metrics.gradient_diversity import GradientDiversityTracker
    HAS_GRADIENT_TRACKING = True
except ImportError:
    GradientDiversityTracker = None
    HAS_GRADIENT_TRACKING = False

from selth.utils.experiment_tracker import ExperimentTracker
from selth.utils.logging_config import setup_logging
from selth.config.experiment_config import ExperimentConfig, create_config_from_args


def batches(dataset):
    """Iterator of [begin, end) batches."""
    i = 0
    for time, y_map in dataset.dataset.label_dict.items():
        j = i + 1
        offset = 1
        while dataset.ts[j] < time:
            if j + offset < len(dataset.ts) and dataset.ts[j + offset] < time:
                j += offset
                offset *= 2
            else:
                offset //= 2
                j += 1

        projection = list(y_map.keys())
        y = torch.tensor(np.array([y_map[k] for k in projection]), dtype=torch.float)
        projection = torch.tensor(projection, dtype=torch.long)
        yield i, j, projection, y
        i = j



def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a single SELTH validation experiment.


    """
    # Setup logging
    logger = setup_logging(
        config.experiment_name,
        log_dir=config.get_output_dir() / 'logs'
    )

    logger.info("="*80)
    logger.info(f"SELTH VALIDATION EXPERIMENT: {config.experiment_name}")
    logger.info("="*80)

    # Setup experiment tracker
    tracker = ExperimentTracker(config.experiment_name)
    tracker.record_config({
        'dataset': config.dataset,
        'embed_dim': config.embed_dim,
        'prune_ratio': config.prune_ratio,
        'prune_method': config.prune_method,
        'seed': config.seed,
        'model_flavour': config.model_flavour
    })

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Set EMBED dimension
    hyper.EMBED = config.embed_dim # Lorenz: This doesnt work. Always uses 32 from the file...

    # Set batch size if provided (override default)
    if config.batch_size is not None:
        hyper.BATCH = config.batch_size

    logger.info(f"\nConfiguration:")
    logger.info(f"  Dataset: {config.dataset}")
    logger.info(f"  Embedding dimension: {config.embed_dim}")
    logger.info(f"  Pruning ratio: {config.prune_ratio:.0%}")
    logger.info(f"  Random seed: {config.seed}")
    if config.batch_size is not None:
        logger.info(f"  Batch size: {config.batch_size} (custom)")

    # Load dataset
    logger.info(f"\nLoading dataset: {config.dataset}")
    dataset = PyGNodePropPredDataset(name=config.dataset, root=config.dataset_root)
    dataset.process_data()

    src = dataset.src.to(hyper.DEVICE)
    dst = dataset.dst.to(hyper.DEVICE)
    ts = dataset.ts.to(hyper.DEVICE)

    total_nodes = max(int(src.max()), int(dst.max().item())) + 1
    total_events = len(ts)
    num_train = int(dataset.train_mask.sum())
    num_val = int(dataset.val_mask.sum())
    num_test = int(dataset.test_mask.sum())

    logger.info(f"  Nodes: {total_nodes:,}")
    logger.info(f"  Events: {total_events:,}")
    logger.info(f"  Train/Val/Test: {num_train:,} / {num_val:,} / {num_test:,}")

    # Create model
    model = Model(config.model_flavour, total_nodes, total_events, dataset.num_classes)
    model = model.to(hyper.DEVICE)

    # Count initial parameters (no masks = all active)
    total_params_initial = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel: {config.model_flavour}")
    logger.info(f"  Total parameters: {total_params_initial:,}")

    #exit(-1)
    # Apply pruning
    masks = None
    if config.prune_ratio > 0:
        logger.info("\n" + "="*80)
        logger.info(f"PRUNING: Applying {config.prune_ratio:.0%} {config.prune_method} pruning")
        logger.info("="*80)

        start_time = time.time()
        # Note: pruning.py only supports random pruning (SELTH-compliant)
        masks = prune_at_initialization(model, config.prune_ratio)
        pruning_time = time.time() - start_time

        sparsity_stats = get_sparsity_stats(model, masks)
        total_params, active_params = count_parameters(model, masks)

        logger.info(f"  Total parameters:  {total_params:,}")
        logger.info(f"  Active parameters: {active_params:,}")
        logger.info(f"  Compression ratio: {total_params/active_params:.2f}×")
        logger.info(f"  Pruning time: {pruning_time:.2f}s")

        tracker.record_timing('pruning', pruning_time)
    else:
        active_params = total_params_initial

    # Training setup
    optimiser = hyper.make_optimiser(model)
    evaluator = Evaluator(name=config.dataset)

    epoch = 0
    best_validation = float('+inf')
    patience = config.patience
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_ndcg': [],
        'epochs': [],
        'best_epoch': 0,
        'best_val_loss': float('+inf')
    }

    # Forward function
    def forward(current, after, project, y):
        hs = model.embed(src, dst, ts, after)
        root = src[current:after]
        pos_dst = dst[current:after]
        embedding = hs[-1][project]
        prediction = model.predict_node(embedding)
        loss = F.cross_entropy(prediction, y.to(hyper.DEVICE))
        model.remember(hs, root, pos_dst, current)
        return loss, embedding

    logger.info("PRE-TRAINING SEPARABILITY")
    embeddings = None
    batch_iterator = batches(dataset)
    with torch.no_grad():
        for current, after, projection, y in batch_iterator: # This is now for the entire dataset. We should also try to predict test set prediction based on training set separability
            _, embedding = forward(current, after, projection, y)
            embeddings = embedding.detach().cpu() if embeddings is None else torch.cat([embeddings, embedding.detach().cpu()], dim=0)
            #print(embeddings.size(), embeddings.device, flush=True)
        eps = 1e-4
        h_q = (embeddings / eps).round()  # quantize
        ratio = torch.unique(h_q, dim=0).size(0) / embeddings.size(0)
        #print(ratio, flush=True)
        #exit(-1)
    logger.info(f"pre_training_separability = {ratio:.4f}")

    # Training loop
    logger.info("\n" + "="*80)
    logger.info("TRAINING")
    logger.info("="*80)

    training_start = time.time()
    while True:
        epoch += 1

        # === TRAIN ===
        batch_iterator = batches(dataset)
        model.train()

        train_loss = 0
        train_batches = 0

        embeddings = None
        for current, after, projection, y in batch_iterator:
            loss, embedding = forward(current, after, projection, y)
            embeddings = embedding if embeddings is None else torch.cat([embeddings, embedding], dim=0)
            loss.backward()

            optimiser.step()

            # CRITICAL: Re-apply pruning masks
            if masks:
                apply_pruning_masks(model, masks)

            optimiser.zero_grad()

            train_loss += loss.item()
            train_batches += 1

            if after > num_train:
                break

        train_loss /= train_batches
        #exit(-1)

        # === VALIDATION ===
        model.eval()
        validation_loss = 0
        val_batches = 0

        with torch.no_grad():
            for current, after, projection, y in batch_iterator:
                loss,_ = forward(current, after, projection, y)
                validation_loss += loss.item()
                val_batches += 1

                if after > num_train + num_val:
                    break

        validation_loss /= val_batches

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(validation_loss)
        history['epochs'].append(epoch)

        logger.info(
            f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
            f"Val Loss = {validation_loss:.4f}",
        )

        # === EARLY STOPPING ===
        if validation_loss < best_validation:
            best_validation = validation_loss
            patience = config.patience
            history['best_epoch'] = epoch
            history['best_val_loss'] = validation_loss

            # Save best model
            output_dir = config.get_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / 'best_model.pt')

            logger.debug("  ← New best model saved")
        else:
            patience -= 1

        if patience < 0 or epoch >= config.max_epochs:
            logger.info(f"\nTraining stopped at epoch {epoch}")
            break

    training_time = time.time() - training_start
    logger.info(f"Training completed in {training_time:.1f}s ({training_time/60:.1f}min)")
    tracker.record_timing('training', training_time)

    # Load best model for evaluation
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))

    # === TEST EVALUATION ===
    logger.info("\n" + "="*80)
    logger.info("EVALUATION: Test Set")
    logger.info("="*80)

    model.eval()
    test_loss = 0
    test_batches = 0
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        batch_iterator = batches(dataset)

        for current, after, projection, y in batch_iterator:
            if after <= num_train + num_val:
                continue

            loss,_ = forward(current, after, projection, y)
            test_loss += loss.item()
            test_batches += 1

            # Collect predictions for NDCG
            hs = model.embed(src, dst, ts, after)
            embedding = hs[-1][projection]
            prediction = model.predict_node(embedding)

            y_pred_list.append(prediction.cpu())
            y_true_list.append(y.cpu())

    test_loss /= test_batches

    # Compute NDCG
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)

    input_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
        "eval_metric": [dataset.dataset.eval_metric]
    }
    result_dict = evaluator.eval(input_dict)
    test_ndcg = result_dict[dataset.dataset.eval_metric]

    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test {dataset.dataset.eval_metric.upper()}: {test_ndcg:.6f}")


    # Compile results
    results = {
        'config': {
            'dataset': config.dataset,
            'embed_dim': config.embed_dim,
            'prune_ratio': config.prune_ratio,
            'prune_method': config.prune_method,
            'seed': config.seed
        },
        'model': {
            'total_params': total_params_initial,
            'active_params': active_params,
            'compression_ratio': total_params_initial / active_params if active_params > 0 else 1.0
        },
        'training': {
            'epochs': epoch,
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss'],
            'training_time_seconds': training_time,
            'history': history
        },
        'performance': {
            'test_loss': test_loss,
            'test_ndcg': test_ndcg
        },
        'expressivity': {
            'pre_ratio': ratio
        },
    }

    # Record results in tracker
    tracker.record_results({
        'test_ndcg': test_ndcg,
        'epochs': epoch,
        'active_params': active_params,
    })


    # Save everything
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir)

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save tracker metadata
    tracker.save(output_dir)

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(tracker.get_summary())

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SELTH Validation for Temporal GNNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--embed_dim', type=int, required=True,
                       help='Embedding dimension (0 = no embeddings)')
    parser.add_argument('--prune_ratio', type=float, required=True,
                       help='Pruning ratio (0.0 = no pruning, 0.9 = 90%% pruned)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed for reproducibility')

    # Dataset
    parser.add_argument('--dataset', type=str, default='tgbn-trade',
                       help='TGB dataset name')
    parser.add_argument('--dataset_root', type=str, default='datasets',
                       help='Root directory for datasets')

    # Model
    parser.add_argument('--flavour', type=str, default='T1',
                       choices=['T1', 'T2'],
                       help='Model architecture')

    # Pruning
    parser.add_argument('--prune_method', type=str, default='random',
                       choices=['random', 'magnitude'],
                       help='Pruning method')

    # Training
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: 1024, use 256 for large datasets like genre)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--max_epochs', type=int, default=200,
                       help='Maximum training epochs')

    # Metrics
    parser.add_argument('--no_expressivity', action='store_true',
                       help='Disable expressivity measurement (faster)')
    parser.add_argument('--track_gradients', action='store_true',
                       help='Enable gradient diversity tracking (slower)')

    # Output
    parser.add_argument('--results_dir', type=str, default='results_selth',
                       help='Base directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name (auto-generated if not provided)')

    args = parser.parse_args()

    # Create config
    args.measure_expressivity = not args.no_expressivity
    args.measure_gradients = args.track_gradients
    config = create_config_from_args(args)

    # Run experiment
    results = run_experiment(config)

    # Print summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(f"Experiment: {config.experiment_name}")
    print(f"Test NDCG:  {results['performance']['test_ndcg']:.6f}")
    print(f"Epochs:     {results['training']['epochs']}")
    print(f"Time:       {results['training']['training_time_seconds']:.1f}s")
    print("="*80)

    return results


if __name__ == '__main__':
    main()