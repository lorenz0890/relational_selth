"""
Experiment Tracker for Reproducibility

Tracks all experiment metadata
    - Git commit hash
    - Command-line arguments
    - Random seeds
    - Software versions
    - Hardware information
    - Execution time
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import socket


class ExperimentTracker:
    """
    Tracks experiment metadata for reproducibility.

    Usage:
        ```python
        tracker = ExperimentTracker(experiment_name="selth_experiment")

        ```
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()

        # Initialize metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'environment': self._collect_environment_info(),
            'config': {},
            'results': {},
            'timing': {}
        }

    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information."""
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': socket.gethostname(),
            'cpu': platform.processor(),
            'python_executable': sys.executable
        }

        # Try to get Git info
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            env_info['git_commit'] = git_hash

            # Check if working directory is clean
            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            env_info['git_clean'] = len(git_status) == 0

        except (subprocess.CalledProcessError, FileNotFoundError):
            env_info['git_commit'] = 'unknown'
            env_info['git_clean'] = False

        # Try to get package versions
        try:
            import torch
            env_info['torch_version'] = torch.__version__

            try:
                import torch_geometric
                env_info['torch_geometric_version'] = torch_geometric.__version__
            except ImportError:
                pass

            try:
                from tgb import __version__ as tgb_version
                env_info['tgb_version'] = tgb_version
            except ImportError:
                pass

        except ImportError:
            pass

        return env_info

    def record_config(self, config: Dict[str, Any]) -> None:
        """
        Record experiment configuration.

        Args:
            config: Configuration dict (hyperparameters, settings, etc.)
        """
        self.metadata['config'].update(config)

    def record_results(self, results: Dict[str, Any]) -> None:
        """
        Record experiment results.

        Args:
            results: Results dict (metrics, scores, etc.)
        """
        self.metadata['results'].update(results)

    def record_timing(self, stage: str, duration_seconds: float) -> None:
        """
        Record timing for a stage.

        Args:
            stage: Name of stage (e.g., 'training', 'evaluation')
            duration_seconds: Duration in seconds
        """
        self.metadata['timing'][stage] = duration_seconds

    def finalize(self) -> None:
        """Finalize metadata (add end time and total duration)."""
        self.metadata['end_time'] = datetime.now().isoformat()
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metadata['total_duration_seconds'] = duration

    def save(self, output_dir: Path) -> Path:
        """
        Save metadata to JSON file.

        Args:
            output_dir: Directory to save metadata

        Returns:
            metadata_path: Path to saved metadata file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Finalize before saving
        self.finalize()

        # Save to JSON
        metadata_path = output_dir / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        return metadata_path

    def get_summary(self) -> str:
        """
        Get human-readable summary of experiment.

        Returns:
            summary: Formatted summary string
        """
        lines = [
            "="*80,
            f"Experiment: {self.experiment_name}",
            "="*80,
            "",
            "Environment:",
            f"  Platform: {self.metadata['environment']['platform']}",
            f"  Python: {self.metadata['environment']['python_version'].split()[0]}",
        ]

        if 'torch_version' in self.metadata['environment']:
            lines.append(f"  PyTorch: {self.metadata['environment']['torch_version']}")

        if 'git_commit' in self.metadata['environment']:
            git_status = "clean" if self.metadata['environment'].get('git_clean') else "dirty"
            lines.append(f"  Git: {self.metadata['environment']['git_commit'][:8]} ({git_status})")

        lines.extend([
            "",
            "Configuration:",
        ])

        for key, value in self.metadata['config'].items():
            lines.append(f"  {key}: {value}")

        if self.metadata['results']:
            lines.extend([
                "",
                "Results:",
            ])
            for key, value in self.metadata['results'].items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.6f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("="*80)

        return "\n".join(lines)
