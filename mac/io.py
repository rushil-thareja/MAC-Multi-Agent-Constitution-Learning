"""
Input/Output utilities for data processing and file management.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class ConfigLoader:
    """Loads and validates configuration."""

    _REQUIRED_FIELDS = [
        'model', 'data', 'algorithm', 'random_seed',
        'output', 'validation', 'tokenization', 'prompts'
    ]

    @staticmethod
    def load_config(config_path: Path) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ConfigLoader._validate(config)
        return config

    @staticmethod
    def from_dict(config: Dict) -> Dict:
        """Accept an already-built config dict (for programmatic use)."""
        ConfigLoader._validate(config)
        return config

    @staticmethod
    def _validate(config: Dict) -> None:
        for field in ConfigLoader._REQUIRED_FIELDS:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")


class RunManager:
    """Manages run directories and metadata."""
    
    def __init__(self, config: Dict):
        """
        Initialize run manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.runs_dir = Path(config['output']['runs_dir'])
        self.runs_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = self.runs_dir / timestamp
        self.current_run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.current_run_dir / "constitutions").mkdir(exist_ok=True)
        (self.current_run_dir / "docs").mkdir(exist_ok=True)
        (self.current_run_dir / "summaries").mkdir(exist_ok=True)
        (self.current_run_dir / "logs").mkdir(exist_ok=True)
    
    def save_run_config(self) -> None:
        """Save run configuration."""
        config_file = self.current_run_dir / "run_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get_run_dir(self) -> Path:
        """Get current run directory."""
        return self.current_run_dir
    
    def create_doc_dir(self, doc_id: str) -> Path:
        """
        Create directory for document processing.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document directory path
        """
        doc_dir = self.current_run_dir / "docs" / doc_id
        doc_dir.mkdir(exist_ok=True)
        return doc_dir
    
    def create_iter_dir(self, doc_id: str, iter_id: int) -> Path:
        """
        Create directory for iteration.
        
        Args:
            doc_id: Document ID
            iter_id: Iteration ID
            
        Returns:
            Iteration directory path
        """
        doc_dir = self.create_doc_dir(doc_id)
        iter_dir = doc_dir / f"iter_{iter_id:03d}"
        iter_dir.mkdir(exist_ok=True)
        return iter_dir
    
    def save_predictions(self, doc_id: str, iter_id: int, 
                        predictions: List[int], constitution_version: int) -> None:
        """
        Save predictions for an iteration.
        
        Args:
            doc_id: Document ID
            iter_id: Iteration ID
            predictions: List of predicted private token indices
            constitution_version: Constitution version used
        """
        iter_dir = self.create_iter_dir(doc_id, iter_id)
        
        predictions_data = {
            'doc_id': doc_id,
            'iter_id': iter_id,
            'constitution_version': constitution_version,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        predictions_file = iter_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
    
    def save_constitution_reference(self, doc_id: str, iter_id: int,
                                  constitution_version: int, constitution_text: str) -> None:
        """
        Save reference to constitution version used.
        
        Args:
            doc_id: Document ID
            iter_id: Iteration ID
            constitution_version: Constitution version
            constitution_text: Constitution text
        """
        iter_dir = self.create_iter_dir(doc_id, iter_id)
        
        ref_file = iter_dir / "constitution_ref.txt"
        with open(ref_file, 'w') as f:
            f.write(f"# Constitution v{constitution_version:04d} used for {doc_id} iter {iter_id}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(constitution_text)
    
    def save_summary(self, filename: str, content: str) -> None:
        """
        Save summary to summaries directory.
        
        Args:
            filename: Summary filename
            content: Summary content
        """
        summary_file = self.current_run_dir / "summaries" / filename
        with open(summary_file, 'w') as f:
            f.write(content)


def create_run_manager(config: Dict) -> RunManager:
    """
    Factory function to create run manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RunManager instance
    """
    return RunManager(config)
