"""
Centralized Configuration for SELTH Experiments

Provides type-safe configuration management.

"""

from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    """
    Configuration for a single SELTH experiment.

    Attributes:
        # Dataset
        dataset: str = Dataset name (e.g., 'tgbn-trade')
        dataset_root: str = Root directory for datasets

        # Model
        model_flavour: str = Model architecture ('T1', 'T2', etc.)
        embed_dim: int = Embedding dimension (0 = no embeddings)

        # Pruning
        prune_ratio: float = Pruning ratio (0.0 = no pruning, 0.9 = 90% pruned)
        prune_method: str = Pruning method ('random', 'magnitude')

        # Training
        seed: int = Random seed for reproducibility
        patience: int = Early stopping patience
        max_epochs: int = Maximum training epochs

        # Evaluation
        eval_metric: str = Evaluation metric ('ndcg', 'mrr')
        measure_expressivity: bool = Whether to measure expressivity
        measure_gradients: bool = Whether to track gradients

        # Output
        results_dir: Path = Base directory for results
        experiment_name: str = Unique experiment identifier
    """

    # Dataset
    dataset: str = 'tgbn-trade'
    dataset_root: str = 'datasets'

    # Model
    model_flavour: str = 'T1'
    embed_dim: int = 32

    # Pruning
    prune_ratio: float = 0.0
    prune_method: str = 'random'

    # Training
    seed: int = 42
    batch_size: Optional[int] = None  # None = use default from hyper.py (1024)
    patience: int = 5
    max_epochs: int = 200

    # Evaluation
    eval_metric: str = 'ndcg'
    measure_expressivity: bool = True
    measure_gradients: bool = False

    # Output
    results_dir: Path = Path('results_selth')
    experiment_name: Optional[str] = None

    def __post_init__(self):
        """Generate experiment name if not provided."""
        if self.experiment_name is None:
            prune_pct = str(self.prune_ratio * 100).replace('.', '_')#int(self.prune_ratio * 100) #TODO this results in overriding results for granularities < 1.0%
            self.experiment_name = (
                f"{self.dataset}_"
                f"{self.model_flavour}_"
                f"embed{self.embed_dim}_"
                f"prune{prune_pct}pct_"
                f"seed{self.seed}"
            )

        # Convert paths
        self.results_dir = Path(self.results_dir)
        self.dataset_root = str(self.dataset_root)

    def get_output_dir(self) -> Path:
        """Get output directory for this experiment."""
        return self.results_dir / self.experiment_name

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save configuration to JSON.

        Args:
            output_dir: Directory to save to (default: self.get_output_dir())

        Returns:
            config_path: Path to saved config file
        """
        if output_dir is None:
            output_dir = self.get_output_dir()

        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / 'config.json'

        # Convert to dict and handle Path objects
        config_dict = asdict(self)
        config_dict['results_dir'] = str(config_dict['results_dir'])

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        return config_path

    @classmethod
    def load(cls, config_path: Path) -> 'ExperimentConfig':
        """
        Load configuration from JSON.

        Args:
            config_path: Path to config file

        Returns:
            config: Loaded configuration
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def __str__(self) -> str:
        """Get string representation."""
        lines = ["ExperimentConfig:"]
        for key, value in asdict(self).items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def create_config_from_args(args) -> ExperimentConfig:
    """
    Create ExperimentConfig from argparse.Namespace.

    Args:
        args: Parsed command-line arguments

    Returns:
        config: ExperimentConfig instance
    """
    return ExperimentConfig(
        dataset=getattr(args, 'dataset', 'tgbn-trade'),
        dataset_root=getattr(args, 'dataset_root', 'datasets'),
        model_flavour=getattr(args, 'flavour', 'T1'),
        embed_dim=getattr(args, 'embed_dim', 32),
        prune_ratio=getattr(args, 'prune_ratio', 0.0),
        prune_method=getattr(args, 'prune_method', 'random'),
        seed=getattr(args, 'seed', 42),
        batch_size=getattr(args, 'batch_size', None),
        patience=getattr(args, 'patience', 5),
        max_epochs=getattr(args, 'max_epochs', 200),
        eval_metric=getattr(args, 'eval_metric', 'ndcg'),
        measure_expressivity=getattr(args, 'measure_expressivity', True),
        measure_gradients=getattr(args, 'measure_gradients', False),
        results_dir=Path(getattr(args, 'results_dir', 'results_selth')),
        experiment_name=getattr(args, 'experiment_name', None)
    )
