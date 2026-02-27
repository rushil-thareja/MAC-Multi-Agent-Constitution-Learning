"""
Metrics and evaluation system for constitutional classifier.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""
    doc_id: str
    iter_id: int
    constitution_version: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fn_indices: List[int]
    fp_indices: List[int]
    timestamp: str


@dataclass
class DocumentMetrics:
    """Final metrics for a document."""
    doc_id: str
    iterations: int
    final_constitution_version: int
    initial_f1: float
    final_f1: float
    improvement: float
    final_precision: float
    final_recall: float
    timestamp: str


@dataclass
class RunMetrics:
    """Aggregate metrics for entire run."""
    total_documents: int
    final_constitution_version: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    mean_improvement: float
    timestamp: str


class MetricsTracker:
    """Tracks and manages metrics throughout the run."""
    
    def __init__(self, run_dir: Path):
        """
        Initialize metrics tracker.
        
        Args:
            run_dir: Run directory for saving metrics
        """
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.iteration_metrics = []
        self.document_metrics = []
        self.run_metrics = None
    
    def record_iteration(self, doc_id: str, iter_id: int, constitution_version: int,
                        evaluation_results: Dict, text: str = None) -> IterationMetrics:
        """
        Record metrics for an iteration.
        
        Args:
            doc_id: Document ID
            iter_id: Iteration ID
            constitution_version: Constitution version used
            evaluation_results: Results from evaluator
            text: Original text for context (optional)
            
        Returns:
            IterationMetrics object
        """
        # Convert phrase lists to indices for compatibility
        fn_phrases = evaluation_results.get('fn_phrases', [])
        fp_phrases = evaluation_results.get('fp_phrases', [])
        
        metrics = IterationMetrics(
            doc_id=doc_id,
            iter_id=iter_id,
            constitution_version=constitution_version,
            tp=evaluation_results['tp'],
            fp=evaluation_results['fp'],
            fn=evaluation_results['fn'],
            precision=evaluation_results['precision'],
            recall=evaluation_results['recall'],
            f1=evaluation_results['f1'],
            fn_indices=fn_phrases,  # Store phrases instead of indices
            fp_indices=fp_phrases,  # Store phrases instead of indices
            timestamp=datetime.now().isoformat()
        )
        
        self.iteration_metrics.append(metrics)
        self._save_iteration_metrics(metrics, text)
        
        return metrics
    
    def record_document_completion(self, doc_id: str) -> DocumentMetrics:
        """
        Record completion of document processing.
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentMetrics object
        """
        # Get all iterations for this document
        doc_iterations = [m for m in self.iteration_metrics if m.doc_id == doc_id]
        
        if not doc_iterations:
            raise ValueError(f"No iterations found for document {doc_id}")
        
        # Sort by iteration ID
        doc_iterations.sort(key=lambda x: x.iter_id)
        
        initial_metrics = doc_iterations[0]
        final_metrics = doc_iterations[-1]
        
        metrics = DocumentMetrics(
            doc_id=doc_id,
            iterations=len(doc_iterations),
            final_constitution_version=final_metrics.constitution_version,
            initial_f1=initial_metrics.f1,
            final_f1=final_metrics.f1,
            improvement=final_metrics.f1 - initial_metrics.f1,
            final_precision=final_metrics.precision,
            final_recall=final_metrics.recall,
            timestamp=datetime.now().isoformat()
        )
        
        self.document_metrics.append(metrics)
        self._save_document_metrics(metrics)
        
        return metrics
    
    def compute_run_metrics(self, final_constitution_version: int) -> RunMetrics:
        """
        Compute aggregate metrics for the entire run.
        
        Args:
            final_constitution_version: Final constitution version
            
        Returns:
            RunMetrics object
        """
        if not self.document_metrics:
            raise ValueError("No document metrics to aggregate")
        
        # Compute macro averages (average of per-document metrics)
        macro_precision = sum(dm.final_precision for dm in self.document_metrics) / len(self.document_metrics)
        macro_recall = sum(dm.final_recall for dm in self.document_metrics) / len(self.document_metrics)
        macro_f1 = sum(dm.final_f1 for dm in self.document_metrics) / len(self.document_metrics)
        
        # Compute micro averages (aggregate all TPs, FPs, FNs)
        final_iterations = []
        for doc_metrics in self.document_metrics:
            doc_id = doc_metrics.doc_id
            doc_iter_metrics = [m for m in self.iteration_metrics 
                              if m.doc_id == doc_id and 
                              m.constitution_version == doc_metrics.final_constitution_version]
            if doc_iter_metrics:
                final_iterations.append(doc_iter_metrics[-1])
        
        total_tp = sum(m.tp for m in final_iterations)
        total_fp = sum(m.fp for m in final_iterations)
        total_fn = sum(m.fn for m in final_iterations)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Mean improvement
        mean_improvement = sum(dm.improvement for dm in self.document_metrics) / len(self.document_metrics)
        
        self.run_metrics = RunMetrics(
            total_documents=len(self.document_metrics),
            final_constitution_version=final_constitution_version,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            mean_improvement=mean_improvement,
            timestamp=datetime.now().isoformat()
        )
        
        self._save_run_metrics()
        return self.run_metrics
    
    def get_phrase_errors(self, doc_id: str, iter_id: int, text: str = None) -> Tuple[List[str], List[str], str]:
        """
        Get phrase-level error examples for rule proposal.
        
        Args:
            doc_id: Document ID
            iter_id: Iteration ID  
            text: Original text for context generation
            
        Returns:
            Tuple of (fn_phrases, fp_phrases, error_context)
        """
        # Find iteration metrics
        iteration_metrics = None
        for m in self.iteration_metrics:
            if m.doc_id == doc_id and m.iter_id == iter_id:
                iteration_metrics = m
                break
        
        if not iteration_metrics:
            return [], [], ""
        
        fn_phrases = iteration_metrics.fn_indices  # Now contains phrases
        fp_phrases = iteration_metrics.fp_indices  # Now contains phrases
        
        # Generate context if text is available
        error_context = ""
        if text:
            from .phrase_utils import create_phrase_converter
            converter = create_phrase_converter()
            error_context = converter.generate_error_context(fn_phrases, fp_phrases, text)
        
        return fn_phrases, fp_phrases, error_context
    
    def _get_token_context(self, tokens: List[str], target_idx: int, 
                          window_size: int) -> str:
        """Get context window around a token."""
        start_idx = max(0, target_idx - window_size)
        end_idx = min(len(tokens), target_idx + window_size + 1)
        
        context_tokens = tokens[start_idx:end_idx]
        relative_idx = target_idx - start_idx
        
        # Highlight target token
        if 0 <= relative_idx < len(context_tokens):
            context_tokens[relative_idx] = f"**{context_tokens[relative_idx]}**"
        
        return " ".join(context_tokens)
    
    def _get_phrase_context(self, phrase: str, text: str, context_chars: int = 100) -> str:
        """Get context around a phrase in text."""
        if not text or not phrase:
            return ""
        
        phrase_lower = phrase.lower()
        text_lower = text.lower()
        
        # Find phrase in text
        start_pos = text_lower.find(phrase_lower)
        if start_pos == -1:
            return f"Context not found for phrase: {phrase}"
        
        end_pos = start_pos + len(phrase)
        
        # Get context
        context_start = max(0, start_pos - context_chars)
        context_end = min(len(text), end_pos + context_chars)
        
        context = text[context_start:context_end]
        
        # Highlight the phrase
        phrase_in_context = text[start_pos:end_pos]
        highlighted_context = context.replace(phrase_in_context, f"**{phrase_in_context}**")
        
        return highlighted_context
    
    def _save_iteration_metrics(self, metrics: IterationMetrics, text: str = None) -> None:
        """Save iteration metrics to disk."""
        # Create document directory
        doc_dir = self.metrics_dir / metrics.doc_id
        doc_dir.mkdir(exist_ok=True)
        
        # Create iteration directory
        iter_dir = doc_dir / f"iter_{metrics.iter_id:03d}"
        iter_dir.mkdir(exist_ok=True)
        
        # Save metrics JSON
        metrics_file = iter_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Save deltas (FN/FP details) - now phrase-based
        deltas_file = iter_dir / "deltas.json"
        
        fn_phrases = metrics.fn_indices  # Contains phrases now
        fp_phrases = metrics.fp_indices  # Contains phrases now
        
        fn_details = []
        for phrase in fn_phrases:
            fn_details.append({
                'phrase': phrase,
                'context': self._get_phrase_context(phrase, text) if text else ""
            })
        
        fp_details = []
        for phrase in fp_phrases:
            fp_details.append({
                'phrase': phrase,
                'context': self._get_phrase_context(phrase, text) if text else ""
            })
        
        deltas = {
            'false_negative_phrases': fn_details,
            'false_positive_phrases': fp_details,
            'constitution_version': metrics.constitution_version
        }
        
        with open(deltas_file, 'w') as f:
            json.dump(deltas, f, indent=2)
    
    def _save_document_metrics(self, metrics: DocumentMetrics) -> None:
        """Save document metrics to disk."""
        doc_file = self.metrics_dir / f"{metrics.doc_id}_summary.json"
        with open(doc_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def _save_run_metrics(self) -> None:
        """Save run metrics to disk."""
        if not self.run_metrics:
            return
        
        # Save JSON
        run_file = self.metrics_dir / "run_summary.json"
        with open(run_file, 'w') as f:
            json.dump(asdict(self.run_metrics), f, indent=2)
        
        # Save CSV for per-document metrics
        csv_file = self.metrics_dir / "per_document_metrics.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.document_metrics:
                writer = csv.DictWriter(f, fieldnames=asdict(self.document_metrics[0]).keys())
                writer.writeheader()
                for metrics in self.document_metrics:
                    writer.writerow(asdict(metrics))
    
    def export_all_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics for reporting.
        
        Returns:
            Dict with all metrics data
        """
        return {
            'iteration_metrics': [asdict(m) for m in self.iteration_metrics],
            'document_metrics': [asdict(m) for m in self.document_metrics],
            'run_metrics': asdict(self.run_metrics) if self.run_metrics else None
        }


class ObjectiveFunction:
    """Defines objective function for constitutional optimization."""
    
    @staticmethod
    def evaluate(metrics_before: Dict, metrics_after: Dict, 
                 objective_type: str = "minimize_fn_then_fp") -> float:
        """
        Evaluate objective improvement.
        
        Args:
            metrics_before: Metrics before change
            metrics_after: Metrics after change  
            objective_type: Type of objective function
            
        Returns:
            Objective score (higher is better)
        """
        if objective_type == "minimize_fn_then_fp":
            # Primary: minimize FN, secondary: minimize FP, tertiary: maximize F1
            fn_before = metrics_before.get('fn', 0)
            fn_after = metrics_after.get('fn', 0)
            
            fp_before = metrics_before.get('fp', 0)
            fp_after = metrics_after.get('fp', 0)
            
            f1_before = metrics_before.get('f1', 0)
            f1_after = metrics_after.get('f1', 0)
            
            # Calculate improvements
            fn_improvement = fn_before - fn_after  # Positive if FN reduced
            fp_improvement = fp_before - fp_after  # Positive if FP reduced
            f1_improvement = f1_after - f1_before  # Positive if F1 increased
            
            # Weight: FN >> FP >> F1
            score = (fn_improvement * 1000 + 
                    fp_improvement * 10 + 
                    f1_improvement)
            
            return score
        
        else:
            # Default to F1 improvement
            return metrics_after.get('f1', 0) - metrics_before.get('f1', 0)
    
    @staticmethod
    def should_accept_change(metrics_before: Dict, metrics_after: Dict,
                           min_improvement: float = 0.01,
                           objective_type: str = "minimize_fn_then_fp") -> bool:
        """
        Determine if a change should be accepted.
        
        Args:
            metrics_before: Metrics before change
            metrics_after: Metrics after change
            min_improvement: Minimum improvement threshold
            objective_type: Type of objective function
            
        Returns:
            True if change should be accepted
        """
        score = ObjectiveFunction.evaluate(metrics_before, metrics_after, objective_type)
        return score >= min_improvement


def create_metrics_tracker(run_dir: Path) -> MetricsTracker:
    """
    Factory function to create metrics tracker.
    
    Args:
        run_dir: Run directory
        
    Returns:
        MetricsTracker instance
    """
    return MetricsTracker(run_dir)