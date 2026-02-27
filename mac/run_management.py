"""
Run management system for Constitutional Classifier.
Handles run-specific logging, progress tracking, and constitution evolution.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class RunManager:
    """Manages individual runs with timestamped directories and organized logging."""

    def __init__(self, config: Dict):
        """Initialize run manager."""
        self.config = config

        # Create timestamped run directory using configured runs_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        use_lora = config.get('agent_models', {}).get('use_lora_agents', False)
        lora_suffix = "_lora" if use_lora else ""
        self.run_id = f"run_{timestamp}{lora_suffix}"
        self.timestamp = timestamp
        runs_dir = Path(config.get('output', {}).get('runs_dir', 'runs'))

        # Generate hierarchical path
        self.run_dir = self._generate_run_path(config, runs_dir, timestamp)
        
        # Create subdirectories
        self.llm_dir = self.run_dir / "llm_interactions"
        self.constitution_dir = self.run_dir / "constitution"
        self.progress_dir = self.run_dir / "progress"
        self.metrics_dir = self.run_dir / "metrics"
        
        # Create all directories
        for dir_path in [self.llm_dir, self.constitution_dir, self.progress_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Disabled: current_run symlink causes conflicts with parallel runs
        # current_run_link = runs_dir / "current_run"
        # if current_run_link.exists() or current_run_link.is_symlink():
        #     current_run_link.unlink()
        # rel_path = self.run_dir.relative_to(runs_dir)
        # current_run_link.symlink_to(rel_path)
        
        # Initialize progress tracking
        self.progress_file = self.progress_dir / "run_status.txt"
        self.constitution_log = self.constitution_dir / "constitution_log.txt"
        self.constitution_versions = self.constitution_dir / "constitution_versions.json"
        
        # Initialize files
        self._init_progress_file()
        self._init_constitution_files()
        self._create_run_metadata()

        # Update run index
        self._update_run_index(runs_dir)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _generate_run_path(self, config: Dict, runs_dir: Path, timestamp: str) -> Path:
        """Generate self-describing hierarchical run path from config parameters."""

        # Extract parameters
        dataset = config.get('data', {}).get('active_dataset', 'unknown')
        epochs = config.get('algorithm', {}).get('num_epochs', 0)
        maintain_pool = config.get('algorithm', {}).get('maintain_pool', False)
        use_lora = config.get('agent_models', {}).get('use_lora_agents', False)

        # Level 1: dataset_Xepochs_{pool|nopool}[_lora]
        poolflag = "pool" if maintain_pool else "nopool"
        loraflag = "_lora" if use_lora else ""
        level1 = f"{dataset}_{epochs}epochs_{poolflag}{loraflag}"

        # Level 2: strategy and key parameters
        if maintain_pool:
            strategy = config.get('algorithm', {}).get('pool_selection', {}).get('strategy', 'default')
            epsilon = config.get('algorithm', {}).get('pool_selection', {}).get('epsilon', 0.0)
            delta = config.get('algorithm', {}).get('pool_update_min_delta', 0.0)

            max_fn = config.get('error_limits', {}).get('max_fn_phrases', 32)
            max_fp = config.get('error_limits', {}).get('max_fp_phrases', 32)

            # Build compact param slug
            level2 = f"{strategy}-eps{epsilon}-d{delta}-fn{max_fn}-fp{max_fp}"
        else:
            # No pool mode: simpler path
            level2 = "direct"

        # Combine: runs/{level1}/{level2}/{timestamp}/
        return runs_dir / level1 / level2 / timestamp

    def _create_run_metadata(self):
        """Create tags.json for programmatic indexing."""
        tags = {
            'timestamp': self.timestamp,
            'run_path': str(self.run_dir),
            'dataset': self.config.get('data', {}).get('active_dataset'),
            'num_epochs': self.config.get('algorithm', {}).get('num_epochs'),
            'maintain_pool': self.config.get('algorithm', {}).get('maintain_pool'),
            'pool_strategy': self.config.get('algorithm', {}).get('pool_selection', {}).get('strategy'),
            'epsilon': self.config.get('algorithm', {}).get('pool_selection', {}).get('epsilon'),
            'pool_update_delta': self.config.get('algorithm', {}).get('pool_update_min_delta'),
            'max_fn_phrases': self.config.get('error_limits', {}).get('max_fn_phrases'),
            'max_fp_phrases': self.config.get('error_limits', {}).get('max_fp_phrases'),
            'batch_api': self.config.get('algorithm', {}).get('pool_use_batch_api'),
            'provider': self.config.get('model', {}).get('provider'),
            'model_name': self.config.get('model', {}).get('model_name'),
            'evaluation_strategy': self.config.get('evaluation', {}).get('strategy'),
            'random_seed': self.config.get('random_seed')
        }

        tags_file = self.run_dir / "tags.json"
        with open(tags_file, 'w') as f:
            json.dump(tags, f, indent=2)

    def _update_run_index(self, runs_dir: Path):
        """Append current run to runs/index.csv for easy searching."""
        import csv

        index_file = runs_dir / "index.csv"

        # Create if doesn't exist
        file_exists = index_file.exists()

        with open(index_file, 'a', newline='') as f:
            fieldnames = ['timestamp', 'path', 'dataset', 'epochs', 'pool', 'strategy', 'epsilon', 'delta', 'provider', 'model']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': self.timestamp,
                'path': str(self.run_dir.relative_to(runs_dir)),
                'dataset': self.config.get('data', {}).get('active_dataset'),
                'epochs': self.config.get('algorithm', {}).get('num_epochs'),
                'pool': self.config.get('algorithm', {}).get('maintain_pool'),
                'strategy': self.config.get('algorithm', {}).get('pool_selection', {}).get('strategy'),
                'epsilon': self.config.get('algorithm', {}).get('pool_selection', {}).get('epsilon'),
                'delta': self.config.get('algorithm', {}).get('pool_update_min_delta'),
                'provider': self.config.get('model', {}).get('provider'),
                'model': self.config.get('model', {}).get('model_name')
            })

    def _init_progress_file(self):
        """Initialize the progress monitoring file."""
        with open(self.progress_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONSTITUTIONAL CLASSIFIER RUN STATUS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.config.get('model', {}).get('model_name', 'Unknown')}\n")
            f.write(f"Documents to Process: {self.config.get('algorithm', {}).get('DK', 'Unknown')}\n")
            f.write(f"Max Rules: {self.config.get('algorithm', {}).get('DR', 'Unknown')}\n")
            f.write("-" * 80 + "\n")
            f.write("STATUS: Initializing...\n")
            f.write("\n")
    
    def _init_constitution_files(self):
        """Initialize constitution tracking files."""
        # Constitution log
        with open(self.constitution_log, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONSTITUTION EVOLUTION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n")
            f.write("Version 0: Empty Constitution (Initial State)\n")
            f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("  Rules: None\n")
            f.write("\n")
        
        # Constitution versions JSON
        initial_version = {
            "version": 0,
            "timestamp": datetime.now().isoformat(),
            "rules": [],
            "rule_count": 0,
            "change_type": "initial",
            "description": "Empty constitution (initial state)"
        }
        
        with open(self.constitution_versions, 'w') as f:
            json.dump([initial_version], f, indent=2)
    
    def update_progress(self, status: str, doc_current: int = None, doc_total: int = None, 
                       constitution_version: int = None, rule_count: int = None,
                       current_phase: str = None, performance_summary: str = None):
        """Update the progress monitoring file."""
        
        # Read current content
        with open(self.progress_file, 'r') as f:
            lines = f.readlines()
        
        # Find and update status line
        updated_lines = []
        for line in lines:
            if line.startswith("STATUS:"):
                updated_lines.append(f"STATUS: {status}\n")
            else:
                updated_lines.append(line)
        
        # Add current state information
        if doc_current is not None and doc_total is not None:
            updated_lines.append(f"Document Progress: {doc_current}/{doc_total}\n")
        
        if constitution_version is not None:
            updated_lines.append(f"Constitution Version: v{constitution_version}\n")
        
        if rule_count is not None:
            updated_lines.append(f"Rules Count: {rule_count}\n")
        
        if current_phase:
            updated_lines.append(f"Current Phase: {current_phase}\n")
        
        if performance_summary:
            updated_lines.append(f"Performance: {performance_summary}\n")
        
        updated_lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        updated_lines.append("\n")
        
        # Write updated content
        with open(self.progress_file, 'w') as f:
            f.writelines(updated_lines)
    
    def log_constitution_change(self, new_version: int, change_type: str, rule_text: str = None,
                              rule_index: int = None, performance_impact: Dict = None,
                              full_constitution_text: str = None):
        """Log a constitution change."""
        timestamp = datetime.now()
        
        # Update constitution log (human-readable)
        with open(self.constitution_log, 'a') as f:
            f.write(f"Version {new_version}: {change_type.title()}\n")
            f.write(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if change_type == "add" and rule_text:
                f.write(f"  Added Rule: {rule_text}\n")
            elif change_type == "edit" and rule_text and rule_index is not None:
                f.write(f"  Modified Rule {rule_index}: {rule_text}\n")
            elif change_type == "reject":
                f.write(f"  Rejected Change: Insufficient improvement\n")
            
            if performance_impact:
                f1_before = performance_impact.get('f1_before', 0)
                f1_after = performance_impact.get('f1_after', 0)
                f.write(f"  Performance Impact: F1 {f1_before:.3f} → {f1_after:.3f}\n")
            
            f.write("\n")
        
        # Save full constitution text file
        if full_constitution_text and change_type != "reject":
            constitution_file = self.constitution_dir / f"constitution_v{new_version:04d}.txt"
            with open(constitution_file, 'w') as f:
                f.write(f"# Constitutional Classifier - Constitution v{new_version:04d}\n")
                f.write(f"# Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Change: {change_type.title()}\n")
                if rule_text:
                    f.write(f"# Latest Rule: {rule_text}\n")
                f.write("#" + "="*80 + "\n\n")
                f.write(full_constitution_text)
                f.write("\n")
        
        # Update constitution versions JSON
        try:
            with open(self.constitution_versions, 'r') as f:
                versions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            versions = []
        
        version_entry = {
            "version": new_version,
            "timestamp": timestamp.isoformat(),
            "change_type": change_type,
            "rule_text": rule_text,
            "rule_index": rule_index,
            "performance_impact": performance_impact
        }
        
        versions.append(version_entry)
        
        with open(self.constitution_versions, 'w') as f:
            json.dump(versions, f, indent=2)
    
    def get_llm_log_dir(self) -> Path:
        """Get the directory for LLM interaction logs."""
        return self.llm_dir
    
    def get_progress_file(self) -> Path:
        """Get the path to the progress monitoring file."""
        return self.progress_file
    
    def get_constitution_log(self) -> Path:
        """Get the path to the constitution log."""
        return self.constitution_log
    
    def get_run_dir(self) -> Path:
        """Get the run directory path."""
        return self.run_dir
    
    def save_run_config(self) -> None:
        """Save run configuration to the run directory."""
        import yaml
        config_file = self.run_dir / "run_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def save_iteration_metrics(self, doc_id: str, iter_id: int, constitution_version: int,
                             metrics: Dict, evaluation_results: Dict = None):
        """
        Save iteration metrics and generate plots.
        
        Args:
            doc_id: Document ID
            iter_id: Iteration ID  
            constitution_version: Constitution version
            metrics: Metrics dictionary with precision, recall, f1, etc.
            evaluation_results: Evaluation results with tp, fp, fn, etc.
        """
        timestamp = datetime.now()
        
        # Save metrics JSON
        metrics_file = self.metrics_dir / f"{doc_id}_iter_{iter_id:03d}_v{constitution_version:04d}.json"
        
        metrics_data = {
            "timestamp": timestamp.isoformat(),
            "doc_id": doc_id,
            "iter_id": iter_id,
            "constitution_version": constitution_version,
            "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics,
            "evaluation": evaluation_results
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Create plots if matplotlib is available
        if PLOTTING_AVAILABLE:
            self._create_metrics_plots(doc_id, iter_id, constitution_version, metrics, evaluation_results)
    
    def _create_metrics_plots(self, doc_id: str, iter_id: int, constitution_version: int,
                            metrics, evaluation_results: Dict = None):
        """Create and save metrics plots."""
        
        # Extract metrics values
        if hasattr(metrics, '__dict__'):
            m = metrics.__dict__
        else:
            m = metrics
        
        precision = m.get('precision', 0)
        recall = m.get('recall', 0) 
        f1 = m.get('f1', 0)
        
        # Extract counts
        if evaluation_results:
            tp = evaluation_results.get('tp', 0)
            fp = evaluation_results.get('fp', 0)
            fn = evaluation_results.get('fn', 0)
        else:
            tp = fp = fn = 0
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Performance Metrics Bar Chart
        metrics_names = ['Precision', 'Recall', 'F1']
        metrics_values = [precision, recall, f1]
        colors = ['#2E8B57', '#4682B4', '#DC143C']
        
        bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_title(f'Performance Metrics - {doc_id} Iter {iter_id}')
        ax1.set_ylabel('Score')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Confusion Matrix Components
        counts = [tp, fp, fn]
        count_labels = ['True Positive', 'False Positive', 'False Negative']
        count_colors = ['#228B22', '#FF6347', '#FF4500']
        
        bars2 = ax2.bar(count_labels, counts, color=count_colors, alpha=0.7)
        ax2.set_title(f'Error Analysis - {doc_id} Iter {iter_id}')
        ax2.set_ylabel('Count')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom')
        
        # 3. F1 Score Gauge
        ax3.pie([f1, 1-f1], labels=['F1 Score', ''], colors=['#32CD32', '#F0F0F0'],
                startangle=90, counterclock=False, wedgeprops={'width': 0.3})
        ax3.set_title(f'F1 Score: {f1:.3f}')
        
        # 4. Constitution Info
        ax4.text(0.1, 0.8, f'Document: {doc_id}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Iteration: {iter_id}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Constitution v{constitution_version}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f'Precision: {precision:.3f}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.3, f'Recall: {recall:.3f}', fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, f'F1: {f1:.3f}', fontsize=10, transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Iteration Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.metrics_dir / f"{doc_id}_iter_{iter_id:03d}_v{constitution_version:04d}_metrics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a simple line plot for F1 progression (if multiple iterations exist)
        self._create_progression_plot(doc_id)
    
    def _create_progression_plot(self, doc_id: str):
        """Create F1 progression plot for a document."""
        if not PLOTTING_AVAILABLE:
            return
        
        # Collect all metrics files for this document
        metrics_files = list(self.metrics_dir.glob(f"{doc_id}_iter_*.json"))
        if len(metrics_files) < 2:
            return  # Need at least 2 iterations for progression
        
        # Sort by iteration number
        metrics_files.sort()
        
        iterations = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for file in metrics_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                iter_id = data['iter_id']
                metrics = data['metrics']
                
                iterations.append(iter_id)
                f1_scores.append(metrics.get('f1', 0))
                precisions.append(metrics.get('precision', 0))
                recalls.append(metrics.get('recall', 0))
                
            except (json.JSONDecodeError, KeyError):
                continue
        
        if not iterations:
            return
        
        # Create progression plot
        plt.figure(figsize=(10, 6))
        
        plt.plot(iterations, f1_scores, 'o-', label='F1 Score', color='#DC143C', linewidth=2, markersize=6)
        plt.plot(iterations, precisions, 's-', label='Precision', color='#2E8B57', linewidth=2, markersize=6)
        plt.plot(iterations, recalls, '^-', label='Recall', color='#4682B4', linewidth=2, markersize=6)
        
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title(f'Performance Progression - Document {doc_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value annotations for F1 scores
        for i, (x, y) in enumerate(zip(iterations, f1_scores)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save progression plot
        progression_file = self.metrics_dir / f"{doc_id}_progression.png"
        plt.savefig(progression_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_run_summary(self, final_metrics: Dict, total_time: float):
        """Save final run summary."""
        summary_file = self.run_dir / "run_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RUN SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Time: {total_time:.2f} seconds\n")
            f.write(f"Model Used: {self.config.get('model', {}).get('model_name', 'Unknown')}\n")
            f.write("\n")
            f.write("FINAL METRICS:\n")
            f.write("-" * 40 + "\n")
            
            if final_metrics:
                # Handle RunMetrics object
                if hasattr(final_metrics, '__dict__'):
                    for key, value in final_metrics.__dict__.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.3f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                else:
                    # Handle dictionary
                    for key, value in final_metrics.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.3f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
            
            f.write("\n")
            f.write("Files Generated:\n")
            f.write(f"  - Progress Log: {self.progress_file}\n")
            f.write(f"  - Constitution Log: {self.constitution_log}\n")
            f.write(f"  - LLM Interactions: {self.llm_dir}/\n")
            f.write(f"  - Metrics: {self.metrics_dir}/\n")


# Global run manager instance
_current_run_manager: Optional[RunManager] = None


def initialize_run_manager(config: Dict) -> RunManager:
    """Initialize the global run manager."""
    global _current_run_manager
    _current_run_manager = RunManager(config)
    return _current_run_manager


def get_current_run_manager() -> Optional[RunManager]:
    """Get the current run manager instance."""
    return _current_run_manager