"""
Plotting utilities for constitutional classifier metrics visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json


class MetricsPlotter:
    """Handles plotting of metrics during constitutional classifier runs."""
    
    def __init__(self, run_dir: Path):
        """
        Initialize metrics plotter.
        
        Args:
            run_dir: Run directory for saving plots
        """
        self.run_dir = Path(run_dir)
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_iteration_metrics(self, doc_id: str, metrics_history: List[Dict[str, Any]]) -> None:
        """
        Plot metrics evolution across iterations for a single document.
        
        Args:
            doc_id: Document ID
            metrics_history: List of metrics dictionaries from each iteration
        """
        if not metrics_history:
            return
        
        # Extract metrics data
        iterations = []
        f1_scores = []
        precisions = []
        recalls = []
        fn_counts = []
        fp_counts = []
        
        for i, metrics_dict in enumerate(metrics_history):
            metrics = metrics_dict.get('metrics', metrics_dict)  # Handle nested structure
            
            iterations.append(i)
            f1_scores.append(metrics.get('f1', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            fn_counts.append(metrics.get('fn', 0))
            fp_counts.append(metrics.get('fp', 0))
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Constitutional Classifier Metrics Evolution - Document {doc_id}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: F1, Precision, Recall
        ax1.plot(iterations, f1_scores, 'o-', color='#2E8B57', label='F1 Score', linewidth=3)
        ax1.plot(iterations, precisions, 's-', color='#4169E1', label='Precision', linewidth=2)
        ax1.plot(iterations, recalls, '^-', color='#DC143C', label='Recall', linewidth=2)
        ax1.set_title('Performance Metrics', fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: False Negatives and False Positives
        ax2.plot(iterations, fn_counts, 'o-', color='#FF6347', label='False Negatives', linewidth=3)
        ax2.plot(iterations, fp_counts, 's-', color='#FF8C00', label='False Positives', linewidth=3)
        ax2.set_title('Error Counts', fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: F1 Score Focus
        ax3.plot(iterations, f1_scores, 'o-', color='#2E8B57', linewidth=4, markersize=8)
        ax3.fill_between(iterations, f1_scores, alpha=0.3, color='#2E8B57')
        ax3.set_title('F1 Score Evolution', fontweight='bold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('F1 Score')
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)
        
        # Add F1 value annotations
        for i, f1 in enumerate(f1_scores):
            ax3.annotate(f'{f1:.3f}', (i, f1), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        # Plot 4: Combined Error Reduction
        total_errors = [fn + fp for fn, fp in zip(fn_counts, fp_counts)]
        ax4.plot(iterations, total_errors, 'o-', color='#8B008B', linewidth=3, label='Total Errors')
        ax4.bar(iterations, fn_counts, alpha=0.6, color='#FF6347', label='False Negatives')
        ax4.bar(iterations, fp_counts, bottom=fn_counts, alpha=0.6, color='#FF8C00', label='False Positives')
        ax4.set_title('Error Reduction Progress', fontweight='bold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Error Count')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / f"{doc_id}_metrics_evolution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved metrics plot: {plot_file}")
        
        # Show plot (optional - comment out for headless)
        # plt.show()
        
        # Close to free memory
        plt.close()
    
    def plot_run_summary(self, document_metrics: List[Dict[str, Any]]) -> None:
        """
        Plot summary metrics across all documents in the run.
        
        Args:
            document_metrics: List of final document metrics
        """
        if not document_metrics:
            return
        
        doc_ids = [str(m['doc_id']) for m in document_metrics]
        final_f1s = [m['final_f1'] for m in document_metrics]
        improvements = [m['improvement'] for m in document_metrics]
        iterations = [m['iterations'] for m in document_metrics]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Constitutional Classifier Run Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Final F1 Scores by Document
        bars1 = ax1.bar(doc_ids, final_f1s, color='#2E8B57', alpha=0.7)
        ax1.set_title('Final F1 Scores by Document', fontweight='bold')
        ax1.set_xlabel('Document ID')
        ax1.set_ylabel('Final F1 Score')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, f1 in zip(bars1, final_f1s):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: F1 Improvement by Document
        colors = ['#4169E1' if imp > 0 else '#DC143C' for imp in improvements]
        bars2 = ax2.bar(doc_ids, improvements, color=colors, alpha=0.7)
        ax2.set_title('F1 Improvement by Document', fontweight='bold')
        ax2.set_xlabel('Document ID')
        ax2.set_ylabel('F1 Improvement')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, imp in zip(bars2, improvements):
            y_pos = bar.get_height() + (0.01 if imp >= 0 else -0.02)
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{imp:+.3f}', ha='center', va='bottom' if imp >= 0 else 'top', 
                    fontweight='bold')
        
        # Plot 3: Iterations per Document
        bars3 = ax3.bar(doc_ids, iterations, color='#FF8C00', alpha=0.7)
        ax3.set_title('Iterations per Document', fontweight='bold')
        ax3.set_xlabel('Document ID')
        ax3.set_ylabel('Number of Iterations')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, iters in zip(bars3, iterations):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{iters}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Summary Statistics
        mean_f1 = np.mean(final_f1s)
        mean_improvement = np.mean(improvements)
        total_iterations = sum(iterations)
        
        ax4.text(0.5, 0.8, f'Mean Final F1: {mean_f1:.3f}', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2E8B57", alpha=0.7, edgecolor='black'))
        
        ax4.text(0.5, 0.6, f'Mean Improvement: {mean_improvement:+.3f}', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#4169E1", alpha=0.7, edgecolor='black'))
        
        ax4.text(0.5, 0.4, f'Total Iterations: {total_iterations}', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FF8C00", alpha=0.7, edgecolor='black'))
        
        ax4.text(0.5, 0.2, f'Documents Processed: {len(document_metrics)}', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#8B008B", alpha=0.7, edgecolor='black'))
        
        ax4.set_title('Run Statistics', fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / "run_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved run summary plot: {plot_file}")
        
        # Close to free memory
        plt.close()
    
    def plot_unified_progress(self, progress_data: List[Dict[str, Any]], chunk_boundaries: List[int] = None) -> None:
        """
        Create unified progress visualization showing all metrics over time with chunk boundaries.
        
        Args:
            progress_data: List of metrics dictionaries from each iteration/batch
            chunk_boundaries: List of iteration numbers where chunks change
        """
        if not progress_data:
            return
        
        # Extract metrics data
        iterations = []
        f1_scores = []
        precisions = []
        recalls = []
        fn_counts = []
        fp_counts = []
        
        for i, data in enumerate(progress_data):
            metrics = data.get('metrics', data)  # Handle nested structure
            
            iterations.append(i)
            f1_scores.append(metrics.get('f1', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            fn_counts.append(metrics.get('fn', 0))
            fp_counts.append(metrics.get('fp', 0))
        
        # Create unified plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(16, 10))
        fig.suptitle('Constitutional Classifier Training Progress', fontsize=16, fontweight='bold')
        
        # Primary y-axis: Percentages (P, R, F1)
        ax1.set_xlabel('Iteration/Batch Number', fontweight='bold')
        ax1.set_ylabel('Performance Metrics (0-1)', fontweight='bold', color='black')
        
        # Plot performance metrics
        line1 = ax1.plot(iterations, f1_scores, 'o-', color='#2E8B57', label='F1 Score', linewidth=3, markersize=6)
        line2 = ax1.plot(iterations, precisions, 's-', color='#4169E1', label='Precision', linewidth=2, markersize=5)
        line3 = ax1.plot(iterations, recalls, '^-', color='#DC143C', label='Recall', linewidth=2, markersize=5)
        
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        
        # Secondary y-axis: Counts (FN, FP)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Error Counts', fontweight='bold', color='#8B008B')
        
        # Plot error counts
        line4 = ax2.plot(iterations, fn_counts, 'o-', color='#FF6347', label='False Negatives', linewidth=3, markersize=6)
        line5 = ax2.plot(iterations, fp_counts, 's-', color='#FF8C00', label='False Positives', linewidth=3, markersize=6)
        
        ax2.tick_params(axis='y', labelcolor='#8B008B')
        
        # Add vertical red lines for chunk boundaries
        if chunk_boundaries:
            max_count = max(max(fn_counts, default=0), max(fp_counts, default=0))
            for boundary in chunk_boundaries:
                if boundary > 0 and boundary < len(iterations):
                    ax1.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    ax2.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Add chunk boundary label
                    ax1.text(boundary, 0.95, f'Chunk {boundary}', rotation=90, 
                            verticalalignment='top', horizontalalignment='right',
                            color='red', fontweight='bold', alpha=0.8)
        
        # Create unified legend
        lines = line1 + line2 + line3 + line4 + line5
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        
        # Add chunk boundary legend if applicable
        if chunk_boundaries:
            ax1.plot([], [], color='red', linestyle='--', linewidth=2, alpha=0.8, label='Chunk Boundaries')
            # Update legend to include chunk boundaries
            lines.append(ax1.lines[-1])
            labels.append('Chunk Boundaries')
            ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        
        # Add final metrics annotation
        if iterations:
            final_f1 = f1_scores[-1]
            final_fn = fn_counts[-1]
            final_fp = fp_counts[-1]
            
            ax1.text(0.98, 0.02, 
                    f'Final: F1={final_f1:.3f}, FN={final_fn}, FP={final_fp}',
                    transform=ax1.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='black'),
                    fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / "training_progress.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved unified training progress plot: {plot_file}")
        
        # Close to free memory
        plt.close()


def create_plotter(run_dir: Path) -> MetricsPlotter:
    """
    Factory function to create metrics plotter.
    
    Args:
        run_dir: Run directory
        
    Returns:
        MetricsPlotter instance
    """
    return MetricsPlotter(run_dir)