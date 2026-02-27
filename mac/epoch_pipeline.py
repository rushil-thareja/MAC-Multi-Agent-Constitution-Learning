"""
Epoch-based constitutional training pipeline.
Implements batch training with constitutional rule evolution across multiple epochs.
"""

import json
import sys
import random
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

from .io import ConfigLoader
from .agents import create_agents, get_token_tracker, reset_token_tracker
from .constitution import create_constitution_manager
from .scoring import create_metrics_tracker, ObjectiveFunction
from .run_management import initialize_run_manager
from .plotting import create_plotter
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
from .batch_processor import (
    create_batch_processor, 
    create_batch_decision_maker, 
    create_batch_rule_modifier,
    BatchResult
)

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """Results from processing a complete epoch."""
    epoch: int
    batch_results: List[BatchResult]
    constitutional_changes: List[Dict[str, Any]]
    epoch_metrics: Dict[str, float]
    constitution_version_start: int
    constitution_version_end: int
    epoch_duration: float


@dataclass
class TrainingResult:
    """Final training results."""
    total_epochs: int
    epoch_results: List[EpochResult]
    holdout_results: List[Dict[str, float]]
    best_constitution_info: Dict[str, Any]
    final_constitution_version: int
    training_metrics: Dict[str, float]
    holdout_metrics: Dict[str, float]
    total_training_time: float
    token_usage: Optional[Dict[str, Any]] = None  # Token tracking summary
    baseline_holdout_metrics: Optional[Dict[str, float]] = None    # Baseline (no rules) holdout metrics


class ConstitutionalEvolution:
    """Tracks constitutional rule changes and evolution across training."""
    
    def __init__(self, run_dir: Path):
        """Initialize constitutional evolution tracker."""
        self.run_dir = run_dir
        self.evolution_log = []
        self.rule_history = {}  # Track when rules were created/modified
        
    def log_change(self, epoch: int, batch_index: int, decision: Dict[str, Any], 
                   batch_result: BatchResult, old_version: int, new_version: int) -> None:
        """
        Log a constitutional change.
        
        Args:
            epoch: Current epoch number
            batch_index: Current batch index
            decision: Decision made by BatchDecisionMaker
            batch_result: BatchResult that triggered the change
            old_version: Constitution version before change
            new_version: Constitution version after change
        """
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch_index': batch_index,
            'batch_id': batch_result.batch_id,
            'action': decision['action'],
            'rule_index': decision.get('rule_index'),
            'reasoning': decision.get('reasoning', ''),
            'old_version': old_version,
            'new_version': new_version,
            'documents_processed': batch_result.doc_ids,
            'fn_count': len(batch_result.aggregated_fn_phrases),
            'fp_count': len(batch_result.aggregated_fp_phrases),
            'batch_f1': batch_result.batch_metrics['f1'],
            'performance_impact': {
                'fn_phrases_addressed': batch_result.aggregated_fn_phrases[:5],  # Sample
                'fp_phrases_addressed': batch_result.aggregated_fp_phrases[:5]   # Sample
            }
        }
        
        self.evolution_log.append(change_record)
        
        # Update rule history tracking
        if decision['action'] == 'add':
            # For ADD, use the new version number as the rule was added at the end
            rule_key = f"rule_{new_version - 1}"  # New rule index
            self.rule_history[rule_key] = {
                'created_epoch': epoch,
                'created_batch': batch_index,
                'last_modified_epoch': epoch,
                'last_modified_batch': batch_index,
                'modification_count': 1
            }
        elif decision['action'] == 'edit':
            # For EDIT, use the provided rule index
            rule_key = f"rule_{decision['rule_index']}"
            if rule_key in self.rule_history:
                self.rule_history[rule_key]['last_modified_epoch'] = epoch
                self.rule_history[rule_key]['last_modified_batch'] = batch_index
                self.rule_history[rule_key]['modification_count'] += 1
            else:
                # If not tracked yet, initialize it
                self.rule_history[rule_key] = {
                    'created_epoch': epoch,
                    'created_batch': batch_index,
                    'last_modified_epoch': epoch,
                    'last_modified_batch': batch_index,
                    'modification_count': 1
                }
    
    def save_evolution_log(self) -> None:
        """Save evolution log to disk."""
        evolution_file = self.run_dir / "constitutions" / "evolution_log.json"
        evolution_file.parent.mkdir(exist_ok=True)
        
        with open(evolution_file, 'w') as f:
            json.dump({
                'evolution_log': self.evolution_log,
                'rule_history': self.rule_history,
                'summary': {
                    'total_changes': len(self.evolution_log),
                    'add_actions': sum(1 for change in self.evolution_log if change['action'] == 'add'),
                    'edit_actions': sum(1 for change in self.evolution_log if change['action'] == 'edit'),
                    'remove_actions': sum(1 for change in self.evolution_log if change['action'] == 'remove'),
                }
            }, f, indent=2)
        
        logger.info(f"Saved constitutional evolution log with {len(self.evolution_log)} changes")


class EpochBatchConstitutionalPipeline:
    """Main pipeline for epoch-based constitutional training."""

    def __init__(self, config_path: Path = None, config_dict: Dict = None, display=None):
        """
        Initialize epoch-batch constitutional pipeline.

        Args:
            config_path: Path to configuration file.
            config_dict: Pre-built config dict (programmatic).
                         Exactly one of config_path or config_dict must be provided.
            display: Optional MACDisplay for Rich CLI output.
        """
        # Load configuration from file or dict
        if config_dict is not None:
            self.config = ConfigLoader.from_dict(config_dict)
        elif config_path is not None:
            self.config = ConfigLoader.load_config(config_path)
        else:
            raise ValueError("Provide either config_path or config_dict")

        # Set random seed for reproducibility
        random.seed(self.config['random_seed'])

        self.metric = None
        self.error_analyzer = None
        self._trainset = None
        self._holdout = None
        self.display = display

        # Algorithm parameters
        self.training_mode = self.config['algorithm'].get('training_mode', 'epoch_batch_constitutional')
        self.num_epochs = self.config['algorithm'].get('num_epochs', 5)
        self.batch_size = self.config['algorithm'].get('batch_size', 4)
        self.num_batches = self.config['algorithm'].get('num_batches', 8)
        self.training_docs = self.config['algorithm'].get('training_docs', 32)
        self.holdout_docs = self.config['algorithm'].get('holdout_docs',
                            self.config['algorithm'].get('validation_docs', 8))
        self.holdout_batches = self.config['algorithm'].get('holdout_batches',
                               self.config['algorithm'].get('validation_batches', 1))

        # Initialize components
        self.run_manager = initialize_run_manager(self.config)

        # Setup logging EARLY so all subsequent output is controlled
        self._setup_logging()

        # Suppress matplotlib tight_layout warnings
        import warnings
        warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*Tight layout.*", category=UserWarning)

        # Prompt adaptation: optionally rewrite prompts for new task via external model
        prompts_dir = Path(self.config['prompts']['templates_dir'])
        meta_cfg = self.config.get('meta_model', {})
        if meta_cfg.get('enabled', False):
            from .prompt_adaptation import PromptAdapter, TARGET_PROMPTS
            adapter = PromptAdapter(meta_cfg, meta_cfg.get('task', {}))
            adapted_dir = self.run_manager.get_run_dir() / "adapted_prompts"
            if self.display:
                adapter.quiet = True
                task_desc = meta_cfg.get('task', {}).get('description', 'this task')
                self.display.adaptation_start(task_desc, len(TARGET_PROMPTS))
                prompts_dir = adapter.adapt_all_prompts(
                    prompts_dir, adapted_dir,
                    on_agent_start=lambda **kw: self.display.adaptation_agent_start(**kw),
                    on_progress=lambda **kw: self.display.adaptation_agent_done(**kw),
                )
                _log = json.loads((adapted_dir / "adaptation_log.json").read_text())
                self.display.adaptation_complete(
                    _log["summary"]["succeeded"], _log["summary"]["total"]
                )
            else:
                prompts_dir = adapter.adapt_all_prompts(prompts_dir, adapted_dir)
        self.agents = create_agents(self.config, prompts_dir, quiet=bool(self.display))

        # Update agents to use new logging directory
        for agent in self.agents.values():
            agent.set_log_dir(self.run_manager.get_llm_log_dir())

        self.constitution = create_constitution_manager(self.run_manager.get_run_dir(), self.config.get('algorithm', {}))
        self.metrics_tracker = create_metrics_tracker(self.run_manager.get_run_dir())
        self.plotter = create_plotter(self.run_manager.get_run_dir())

        # Create batch processing components
        self.batch_processor = create_batch_processor(self.agents, self.config)
        self.batch_decision_maker = create_batch_decision_maker(self.agents, self.config)
        self.batch_rule_modifier = create_batch_rule_modifier(self.agents, self.config)

        # Create evolution tracker
        self.evolution_tracker = ConstitutionalEvolution(self.run_manager.get_run_dir())

        # Save run configuration
        self.run_manager.save_run_config()

        logger.info(f"Initialized EpochBatchConstitutionalPipeline:")
        logger.info(f"  Training mode: {self.training_mode}")
        logger.info(f"  Epochs: {self.num_epochs}, Batches: {self.num_batches}, Batch size: {self.batch_size}")
        logger.info(f"  Training docs: {self.training_docs}, Holdout docs: {self.holdout_docs}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config['output']['log_level']
        log_file = self.run_manager.get_run_dir() / "logs" / "epoch_pipeline.log"
        log_file.parent.mkdir(exist_ok=True)

        handlers = [logging.FileHandler(log_file)]
        if not self.display:
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Override any existing logging config
        )

    @classmethod
    def init_from_examples(cls, trainset, holdout, metric,
                           config_path=None, config_dict=None,
                           error_analyzer=None,
                           display=None, error_context_formatter=None):
        """Create a pipeline in examples mode (DSPy-style API).

        Args:
            trainset: List[Example] for training.
            holdout: List[Example] for holdout evaluation.
            metric: Callable(prediction, gold) -> float.
            config_path: Optional path to YAML config.
            config_dict: Optional pre-built config dict.
            error_analyzer: Optional error analyzer (defaults to DefaultErrorAnalyzer).
            display: Optional MACDisplay for Rich CLI output.

        Returns:
            Configured EpochBatchConstitutionalPipeline in examples mode.
        """
        pipeline = cls(
            config_path=Path(config_path) if config_path else None,
            config_dict=config_dict,
            display=display,
        )
        pipeline.metric = metric
        pipeline.error_analyzer = error_analyzer
        pipeline._trainset = trainset
        pipeline._holdout = holdout

        # Override doc counts from the actual data
        pipeline.training_docs = len(trainset)
        pipeline.holdout_docs = len(holdout)

        # Recompute num_batches if needed
        if pipeline.training_docs < pipeline.batch_size * pipeline.num_batches:
            pipeline.num_batches = max(1, pipeline.training_docs // pipeline.batch_size)

        # Thread error context formatter to batch processor
        pipeline.error_context_formatter = error_context_formatter
        pipeline.batch_processor.error_context_formatter = error_context_formatter

        logger.info(f"[examples-mode] Initialized with {len(trainset)} train, "
                    f"{len(holdout)} holdout examples")
        return pipeline

    def _limit_error_phrases(self, fn_phrases: List[str], fp_phrases: List[str]) -> Tuple[List[str], List[str]]:
        """
        Limit FN and FP phrases based on config error_limits.

        Args:
            fn_phrases: List of false negative phrases
            fp_phrases: List of false positive phrases

        Returns:
            Tuple of (limited_fn_phrases, limited_fp_phrases)
        """
        error_limits = self.config.get('error_limits', {})
        max_fn = error_limits.get('max_fn_phrases', 12)
        max_fp = error_limits.get('max_fp_phrases', 12)
        selection_method = error_limits.get('phrase_selection', 'random')

        # Apply limits to FN phrases
        limited_fn = fn_phrases.copy() if fn_phrases else []
        if len(limited_fn) > max_fn:
            if selection_method == 'random':
                limited_fn = random.sample(limited_fn, max_fn)
            else:
                limited_fn = limited_fn[:max_fn]

        # Apply limits to FP phrases
        limited_fp = fp_phrases.copy() if fp_phrases else []
        if len(limited_fp) > max_fp:
            if selection_method == 'random':
                limited_fp = random.sample(limited_fp, max_fp)
            else:
                limited_fp = limited_fp[:max_fp]

        if len(fn_phrases or []) > max_fn or len(fp_phrases or []) > max_fp:
            logger.info(f"🔢 LIMITED PHRASES: FN {len(fn_phrases or [])}→{len(limited_fn)}, "
                       f"FP {len(fp_phrases or [])}→{len(limited_fp)} (method: {selection_method})")

        return limited_fn, limited_fp

    def run(self) -> TrainingResult:
        """
        Run the complete epoch-based constitutional training.

        Returns:
            TrainingResult with training outcomes
        """
        logger.info("Starting Epoch-Batch Constitutional Training")
        start_time = time.time()

        # Reset token tracker at start of training
        reset_token_tracker()
        logger.info("[tokens] Token tracker reset for new training run")

        # Load and split documents
        train_docs, holdout_docs = self._load_and_split_documents()
        logger.info(f"Loaded {len(train_docs)} training, {len(holdout_docs)} holdout docs")
        
        # Update progress
        self.run_manager.update_progress(
            status="Starting epoch-based training",
            doc_current=0,
            doc_total=len(train_docs),
            constitution_version=0,
            rule_count=0,
            current_phase="Training Initialization"
        )
        
        # Run epoch-based training
        epoch_results = []
        holdout_results = []
        self.best_constitution_info = {
            'version': 0,
            'f1_score': 0.0,
            'fn_count': float('inf'),
            'epoch': 0,
            'reasoning': 'Initial empty constitution'
        }
        
        # Validation checkpoint configuration
        checkpoint_cfg = self.config.get('algorithm', {}).get('validation_checkpoints', {})
        checkpoints_enabled = checkpoint_cfg.get('enabled', False)
        run_at_start = checkpoint_cfg.get('run_at_start', True)
        checkpoint_frequency = checkpoint_cfg.get('frequency', 5)

        # Store intermediate validation results (batch_index -> metrics)
        self.intermediate_validation_results = []

        # ── Baseline evaluation (empty constitution, no rules) ──────────
        # Always run baselines regardless of checkpoint config
        logger.info("[baseline] Running baseline evaluation (empty constitution)...")

        baseline_holdout_metrics = self._evaluate_holdout(holdout_docs, epoch=0)
        self.baseline_holdout_metrics = baseline_holdout_metrics
        logger.info(f"[baseline] Holdout baseline F1={baseline_holdout_metrics['f1']:.3f}")

        # Run initial holdout eval before training starts
        if checkpoints_enabled and run_at_start:
            if not self.display:
                print(f"[holdout-checkpoint] Running initial holdout eval before training (baseline)...")
            logger.info("[holdout-checkpoint] Running initial holdout eval before training")
            try:
                initial_holdout_metrics = self._evaluate_holdout(holdout_docs, epoch=0)
                initial_holdout_metrics['checkpoint_type'] = 'initial'
                initial_holdout_metrics['global_batch_index'] = 0
                initial_holdout_metrics['constitution_version'] = self.constitution.version
                self.intermediate_validation_results.append(initial_holdout_metrics)
                if not self.display:
                    print(f"[holdout-checkpoint] Initial holdout F1: {initial_holdout_metrics['f1']:.3f} (before training)")
                logger.info(f"[holdout-checkpoint] Initial F1={initial_holdout_metrics['f1']:.3f}")
            except Exception as e:
                logger.warning(f"[holdout-checkpoint] Initial holdout eval failed: {e}")

        # Create epoch progress bar (skip tqdm when rich display active)
        if self.display:
            epoch_iter = range(1, self.num_epochs + 1)
        else:
            epoch_iter = tqdm(range(1, self.num_epochs + 1), desc="Epochs", unit="epoch", leave=True)

        for epoch in epoch_iter:
            if not self.display and hasattr(epoch_iter, 'set_description'):
                epoch_iter.set_description(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"Starting Epoch {epoch}/{self.num_epochs}")

            # Pass holdout docs and checkpoint settings to epoch runner
            epoch_result = self._run_epoch(
                epoch, train_docs,
                holdout_docs=holdout_docs,
                checkpoint_cfg={
                    'enabled': checkpoints_enabled,
                    'frequency': checkpoint_frequency,
                }
            )
            epoch_results.append(epoch_result)

            # Run holdout evaluation after each epoch
            holdout_metrics = self._evaluate_holdout(holdout_docs, epoch)
            holdout_results.append(holdout_metrics)

            # Get metrics for logging
            current_f1 = holdout_metrics['f1']
            current_fn = holdout_metrics['fn_count']

            # Track best constitution from holdout evaluation
            # (works even when validation_checkpoints are disabled)
            is_new_best = (current_f1 > self.best_constitution_info['f1_score'] or
                (current_f1 == self.best_constitution_info['f1_score'] and
                 current_fn < self.best_constitution_info['fn_count']))
            if is_new_best:
                self.best_constitution_info.update({
                    'version': self.constitution.version,
                    'f1_score': current_f1,
                    'fn_count': current_fn,
                    'epoch': epoch,
                    'reasoning': f'Best holdout F1={current_f1:.3f} at epoch {epoch}'
                })
                logger.info(f"New best constitution: v{self.constitution.version} "
                            f"(F1={current_f1:.3f}, Epoch {epoch})")

            # Update progress
            self.run_manager.update_progress(
                status=f"Completed epoch {epoch}/{self.num_epochs}",
                doc_current=epoch * len(train_docs),
                doc_total=self.num_epochs * len(train_docs),
                constitution_version=self.constitution.version,
                rule_count=len(self.constitution.rules),
                current_phase=f"Epoch {epoch} Complete",
                performance_summary=f"Holdout F1={current_f1:.3f}, FN={current_fn}"
            )

            logger.info(f"Epoch {epoch} completed - Constitution v{self.constitution.version}, "
                       f"{len(self.constitution.rules)} rules, Holdout F1={current_f1:.3f}")
        
        # Load best constitution for final evaluation
        logger.info(f"Loading best constitution for final evaluation: {self.best_constitution_info['reasoning']}")
        if self.best_constitution_info['version'] > 0:
            self.constitution.load_constitution(self.best_constitution_info['version'])
            logger.info(f"Loaded constitution v{self.best_constitution_info['version']} for final holdout evaluation")

        # Final evaluation on holdout documents with best constitution
        logger.info("Running final evaluation on holdout documents")
        holdout_metrics = self._evaluate_holdout(holdout_docs, epoch=0)

        # Compute baseline deltas
        baseline_f1 = self.baseline_holdout_metrics.get('f1', 0.0)
        final_f1 = holdout_metrics.get('f1', 0.0)
        delta = final_f1 - baseline_f1
        logger.info(f"[deltas] Holdout: {baseline_f1:.3f} → {final_f1:.3f} (Δ={delta:+.3f})")

        # Calculate training metrics
        training_metrics = self._calculate_training_metrics(epoch_results)
        
        total_time = time.time() - start_time
        
        # Save evolution log
        self.evolution_tracker.save_evolution_log()
        
        # Generate final training report
        # Get final token usage summary
        token_tracker = get_token_tracker()
        token_usage = token_tracker.get_summary()

        # Log final token summary (suppressed when display active)
        if not self.display:
            print(f"\n{'='*60}")
            print(f"[tokens] FINAL TOKEN USAGE SUMMARY")
            print(f"{'='*60}")
            print(f"[tokens] Total prompt tokens:     {token_usage['total_prompt_tokens']:,}")
            print(f"[tokens] Total completion tokens: {token_usage['total_completion_tokens']:,}")
            print(f"[tokens] Total tokens:            {token_usage['total_tokens']:,}")
            print(f"[tokens] Total LLM calls:         {token_usage['total_calls']:,}")
            print(f"[tokens] Per-agent breakdown:")
            for agent, stats in token_usage['per_agent'].items():
                print(f"[tokens]   {agent}: prompt={stats['prompt']:,} completion={stats['completion']:,} total={stats['total']:,} calls={stats['calls']}")
            print(f"{'='*60}\n")

        logger.info(f"[tokens] Final: prompt={token_usage['total_prompt_tokens']} completion={token_usage['total_completion_tokens']} total={token_usage['total_tokens']} calls={token_usage['total_calls']}")

        training_result = TrainingResult(
            total_epochs=self.num_epochs,
            epoch_results=epoch_results,
            holdout_results=holdout_results,
            best_constitution_info=self.best_constitution_info,
            final_constitution_version=self.best_constitution_info['version'],  # Use best constitution
            training_metrics=training_metrics,
            holdout_metrics=holdout_metrics,
            total_training_time=total_time,
            token_usage=token_usage,
            baseline_holdout_metrics=self.baseline_holdout_metrics,
        )

        # Save training results
        self._save_training_results(training_result)
        
        # Generate comprehensive plots
        self._create_training_plots(training_result)
        
        # Final progress update
        self.run_manager.update_progress(
            status="TRAINING COMPLETED",
            doc_current=len(train_docs) * self.num_epochs,
            doc_total=len(train_docs) * self.num_epochs,
            constitution_version=self.constitution.version,
            rule_count=len(self.constitution.rules),
            current_phase="Training Complete",
            performance_summary=f"Holdout F1={holdout_metrics['f1']:.3f}, Rules={len(self.constitution.rules)}"
        )

        # Compute and save final run metrics (creates run_summary.json)
        # Note: In epoch-batch mode, document_metrics may not be populated (uses batch results instead)
        logger.info("Computing final run metrics...")
        try:
            final_metrics = self.metrics_tracker.compute_run_metrics(
                final_constitution_version=self.constitution.version
            )
            logger.info(f"✓ Saved run metrics: F1={final_metrics.micro_f1:.3f}, Documents={final_metrics.total_documents}")
        except ValueError as e:
            # In epoch-batch mode, document-level metrics aren't tracked the same way
            logger.info(f"Skipping run metrics aggregation (epoch-batch mode): {e}")
            logger.info(f"Final results saved in training_results.json with Holdout F1={holdout_metrics['f1']:.3f}")

        logger.info("Epoch-Batch Constitutional Training completed successfully!")
        logger.info(f"Final constitution: v{self.constitution.version} with {len(self.constitution.rules)} rules")
        logger.info(f"Holdout performance: F1={holdout_metrics['f1']:.3f}")
        
        return training_result
    
    def _load_and_split_documents(self):
        """
        Load documents and split into training and holdout sets.

        Returns:
            Tuple of (training_documents, holdout_documents)
            Each is a List[Example].
        """
        logger.info(f"[examples-mode] Using {len(self._trainset)} train, "
                    f"{len(self._holdout)} holdout examples")
        return self._trainset, self._holdout
    
    def _load_documents_from_split(self, split_dir: Path, max_docs: int) -> List[Tuple[Path, Path]]:
        """
        Load documents from a split directory using batch files.
        
        Args:
            split_dir: Path to split directory (e.g., train/, validation/, test/)
            max_docs: Maximum number of documents to load
            
        Returns:
            List of (txt_path, ann_path) tuples
        """
        documents = []
        batches_dir = split_dir / "batches"
        
        if batches_dir.exists():
            # Load from batch files
            batch_files = sorted(batches_dir.glob("batch_*.txt"))
            
            for batch_file in batch_files:
                if len(documents) >= max_docs:
                    break
                    
                # Read document IDs from batch file
                with open(batch_file, 'r') as f:
                    doc_ids = [line.strip() for line in f if line.strip()]
                
                # Convert to file paths
                for doc_id in doc_ids:
                    if len(documents) >= max_docs:
                        break
                        
                    txt_path = split_dir / f"{doc_id}.txt"
                    ann_path = split_dir / f"{doc_id}.ann"
                    
                    if txt_path.exists() and ann_path.exists():
                        documents.append((txt_path, ann_path))
                        
            logger.info(f"Loaded {len(documents)} documents from {len(batch_files)} batch files in {split_dir.name}")
        else:
            # Fallback: load all documents in split directory
            txt_files = list(split_dir.glob("*.txt"))
            txt_files.sort()  # Deterministic order
            
            for txt_file in txt_files[:max_docs]:
                ann_file = txt_file.with_suffix('.ann')
                if ann_file.exists():
                    documents.append((txt_file, ann_file))
                    
            logger.info(f"Loaded {len(documents)} documents directly from {split_dir.name} (no batch files)")
        
        return documents
    
    def _run_epoch(self, epoch: int, train_docs,
                   holdout_docs=None,
                   checkpoint_cfg: Dict = None) -> EpochResult:
        """
        Run a complete training epoch.

        Args:
            epoch: Current epoch number
            train_docs: List of training document pairs
            holdout_docs: List of holdout document pairs (for checkpoints)
            checkpoint_cfg: Checkpoint configuration

        Returns:
            EpochResult with epoch outcomes
        """
        epoch_start_time = time.time()
        constitution_version_start = self.constitution.version

        batch_results = []
        constitutional_changes = []

        # Checkpoint configuration
        checkpoint_cfg = checkpoint_cfg or {}
        checkpoints_enabled = checkpoint_cfg.get('enabled', False)
        checkpoint_frequency = checkpoint_cfg.get('frequency', 5)

        # Display epoch start
        if self.display:
            self.display.epoch_start(epoch, self.num_epochs, len(self.constitution.rules))

        # Process batches (skip tqdm when rich display active)
        if self.display:
            batch_iter = range(self.num_batches)
        else:
            batch_iter = tqdm(range(self.num_batches), desc=f"Epoch {epoch} Batches", unit="batch", leave=False)

        for batch_index in batch_iter:
            if not self.display and hasattr(batch_iter, 'set_description'):
                batch_iter.set_description(f"E{epoch} Batch {batch_index + 1}/{self.num_batches}")
            logger.info(f"Processing Epoch {epoch}, Batch {batch_index + 1}/{self.num_batches}")
            
            # Get batch documents
            batch_start = batch_index * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(train_docs))
            batch_doc_pairs = train_docs[batch_start:batch_end]
            
            if not batch_doc_pairs:
                logger.warning(f"No documents for batch {batch_index + 1}, skipping")
                continue

            # Process batch
            _snap0 = get_token_tracker().snapshot()
            batch_result = self._process_batch(epoch, batch_index, batch_doc_pairs)
            batch_results.append(batch_result)
            _snap1 = get_token_tracker().snapshot()

            # Display batch results
            if self.display:
                self.display.batch_scored(epoch, batch_index, self.num_batches,
                                          batch_result, self.constitution,
                                          total_tokens=_snap1['tokens'],
                                          total_calls=_snap1['calls'])
                self.display.step_tokens("Annotate",
                    _snap1['tokens'] - _snap0['tokens'],
                    _snap1['calls'] - _snap0['calls'])
            else:
                print(f"[tokens]   Annotate: {_snap1['tokens'] - _snap0['tokens']:,} tok ({_snap1['calls'] - _snap0['calls']} calls)")

            # Make constitutional decision
            old_version = self.constitution.version
            old_rules = list(self.constitution.rules)  # Save rules before change for potential revert
            decision = self.batch_decision_maker.make_constitutional_decision(
                self.constitution, batch_result
            )
            _snap2 = get_token_tracker().snapshot()

            # Display decision
            if self.display:
                self.display.decision(decision, batch_result)
                self.display.step_tokens("Decide",
                    _snap2['tokens'] - _snap1['tokens'],
                    _snap2['calls'] - _snap1['calls'])
            else:
                print(f"[tokens]   Decide: {_snap2['tokens'] - _snap1['tokens']:,} tok ({_snap2['calls'] - _snap1['calls']} calls)")

            # Execute constitutional change
            new_version = None
            rule_change_accepted = True  # Track if change was validated and accepted
            _snap3 = get_token_tracker().snapshot()  # before propose/edit
            if decision['action'] != 'no_change':
                new_version = self.batch_rule_modifier.execute_rule_change(
                    decision, self.constitution, batch_result
                )
                _snap_propose = get_token_tracker().snapshot()
                _propose_tok = _snap_propose['tokens'] - _snap3['tokens']
                _propose_calls = _snap_propose['calls'] - _snap3['calls']
                _propose_label = "Propose" if decision['action'] == 'add' else "Edit"
                if self.display:
                    self.display.step_tokens(_propose_label, _propose_tok, _propose_calls)
                else:
                    print(f"[tokens]   {_propose_label}: {_propose_tok:,} tok ({_propose_calls} calls)")

                # Display rule change
                if new_version and self.display:
                    rule_text = ""
                    rule_idx = decision.get('rule_index')
                    if decision['action'] == 'add' and self.constitution.rules:
                        rule_text = self.constitution.rules[-1]
                    elif decision['action'] == 'edit' and rule_idx is not None and rule_idx < len(self.constitution.rules):
                        rule_text = self.constitution.rules[rule_idx]
                    self.display.rule_change(
                        action=decision['action'],
                        old_version=old_version,
                        new_version=new_version,
                        rule_text=rule_text,
                        rule_index=rule_idx,
                    )

                if new_version:
                    # Validate rule change and retry with temperatures if needed
                    held_out_cfg = self.config.get('algorithm', {}).get('held_out_validation', {})
                    use_held_out = held_out_cfg.get('enabled', False)

                    if use_held_out:
                        # Use a DIFFERENT batch for validation (prevents overfitting)
                        held_out_batch_index = (batch_index + 1) % self.num_batches
                        if held_out_batch_index == batch_index:
                            held_out_batch_index = batch_index

                        held_out_start = held_out_batch_index * self.batch_size
                        held_out_end = min(held_out_start + self.batch_size, len(train_docs))
                        held_out_doc_pairs = train_docs[held_out_start:held_out_end]

                        if not self.display:
                            print(f"[held-out-validation] Using batch {held_out_batch_index + 1} for validation (training on batch {batch_index + 1})")
                        logger.info(f"[held-out-validation] Validating on batch {held_out_batch_index + 1} instead of batch {batch_index + 1}")

                        held_out_batch_docs = self.batch_processor.prepare_batch_from_examples(held_out_doc_pairs)
                        original_constitution_text = '\n'.join(old_rules) if old_rules else None
                        try:
                            if original_constitution_text:
                                original_f1 = self._evaluate_single_constitution_direct(
                                    constitution_text=original_constitution_text,
                                    batch_docs=held_out_batch_docs
                                )
                                if not self.display:
                                    print(f"[held-out-validation] Original F1 on held-out batch: {original_f1:.3f}")
                            else:
                                original_f1 = batch_result.batch_metrics.get('f1', 0.0)
                                if not self.display:
                                    print(f"[held-out-validation] No prior rules, using batch F1 as baseline: {original_f1:.3f}")
                        except Exception as e:
                            logger.warning(f"[held-out-validation] Failed to evaluate original on held-out: {e}")
                            original_f1 = batch_result.batch_metrics.get('f1', 0.0)

                        validation_batch_docs = held_out_batch_docs
                    else:
                        # Validate on same batch
                        original_f1 = batch_result.batch_metrics.get('f1', 0.0)
                        validation_batch_docs = self.batch_processor.prepare_batch_from_examples(batch_doc_pairs)

                    _snap_val0 = get_token_tracker().snapshot()
                    validation_result = self._validate_and_retry_rule_change(
                        batch_docs=validation_batch_docs,
                        original_f1=original_f1,
                        decision=decision,
                        batch_result=batch_result,
                        old_rules=old_rules,
                        old_version=old_version
                    )
                    _snap_val1 = get_token_tracker().snapshot()
                    _val_tok = _snap_val1['tokens'] - _snap_val0['tokens']
                    _val_calls = _snap_val1['calls'] - _snap_val0['calls']

                    rule_change_accepted = validation_result.get('accepted', False)

                    # Display validation result
                    if self.display:
                        new_f1 = validation_result.get('new_f1', 0.0) or 0.0
                        delta = validation_result.get('improvement', 0.0) or 0.0
                        self.display.rule_validated(
                            accepted=rule_change_accepted,
                            old_f1=original_f1,
                            new_f1=new_f1,
                            delta=delta,
                            retry_attempt=validation_result.get('retry_attempt'),
                            constitution=self.constitution if rule_change_accepted else None,
                        )
                        self.display.step_tokens("Validate", _val_tok, _val_calls)
                    else:
                        print(f"[tokens]   Validate: {_val_tok:,} tok ({_val_calls} calls)")

                    if not rule_change_accepted:
                        new_version = None
                        logger.info(f"[rule-validate] Rule change rejected, keeping v{old_version}")
                        # Record rejection in proposer/editor memory
                        action = decision.get('action', '')
                        agent_key = 'new_rule_proposer' if action == 'add' else 'new_rule_editor'
                        if agent_key in self.agents:
                            self.agents[agent_key].remember(f"REJECTED (val didn't improve)")
                    else:
                        new_version = self.constitution.version
                        if validation_result.get('retry_attempt'):
                            logger.info(f"[rule-validate] Rule accepted via retry attempt {validation_result['retry_attempt']}")

                    # Only log if change was accepted
                    if new_version and rule_change_accepted:
                        # Log constitutional change
                        self.evolution_tracker.log_change(
                            epoch, batch_index, decision, batch_result, old_version, new_version
                        )

                        constitutional_changes.append({
                            'epoch': epoch,
                            'batch_index': batch_index,
                            'decision': decision,
                            'old_version': old_version,
                            'new_version': new_version
                        })

                        logger.info(f"Constitutional change: {decision['action']} -> v{new_version}")

            # Save batch results
            self._save_batch_results(epoch, batch_index, batch_result, decision)

            # Run validation checkpoint if enabled and at checkpoint frequency
            global_batch_index = (epoch - 1) * self.num_batches + (batch_index + 1)
            if (checkpoints_enabled and
                checkpoint_frequency > 0 and
                holdout_docs is not None and
                global_batch_index % checkpoint_frequency == 0):
                try:
                    if not self.display:
                        print(f"[holdout-checkpoint] Running holdout eval at batch {global_batch_index} (every {checkpoint_frequency} batches)...")
                    logger.info(f"[holdout-checkpoint] Checkpoint at global batch {global_batch_index}")

                    checkpoint_metrics = self._evaluate_holdout(holdout_docs, epoch)
                    checkpoint_metrics['checkpoint_type'] = 'periodic'
                    checkpoint_metrics['global_batch_index'] = global_batch_index
                    checkpoint_metrics['epoch'] = epoch
                    checkpoint_metrics['batch_index'] = batch_index
                    checkpoint_metrics['constitution_version'] = self.constitution.version

                    # Store in intermediate results
                    if hasattr(self, 'intermediate_validation_results'):
                        self.intermediate_validation_results.append(checkpoint_metrics)

                    # Update best constitution if this checkpoint is better
                    checkpoint_f1 = checkpoint_metrics['f1']
                    checkpoint_fn = checkpoint_metrics.get('fn_count', float('inf'))
                    is_new_best = (checkpoint_f1 > self.best_constitution_info['f1_score'] or
                        (checkpoint_f1 == self.best_constitution_info['f1_score'] and
                         checkpoint_fn < self.best_constitution_info['fn_count']))
                    if is_new_best:
                        self.best_constitution_info.update({
                            'version': self.constitution.version,
                            'f1_score': checkpoint_f1,
                            'fn_count': checkpoint_fn,
                            'epoch': epoch,
                            'batch_index': batch_index,
                            'global_batch_index': global_batch_index,
                            'reasoning': f'Best holdout F1={checkpoint_f1:.3f} at batch {global_batch_index}'
                        })
                        logger.info(f"New best constitution: v{self.constitution.version} "
                                    f"(F1={checkpoint_f1:.3f}, Batch {global_batch_index})")
                        if not self.display:
                            print(f"[holdout-checkpoint] New best constitution: v{self.constitution.version} "
                                  f"(F1={checkpoint_f1:.3f})")

                    # Display checkpoint
                    if self.display:
                        self.display.checkpoint(
                            global_batch=global_batch_index,
                            val_f1=checkpoint_f1,
                            const_version=self.constitution.version,
                            n_rules=len(self.constitution.rules),
                            is_new_best=is_new_best,
                        )
                    else:
                        print(f"[holdout-checkpoint] Batch {global_batch_index}: Holdout F1={checkpoint_metrics['f1']:.3f}, "
                              f"Constitution v{self.constitution.version}")
                    logger.info(f"[holdout-checkpoint] Batch {global_batch_index}: F1={checkpoint_metrics['f1']:.3f}")
                except Exception as e:
                    logger.warning(f"[holdout-checkpoint] Checkpoint at batch {global_batch_index} failed: {e}")

            # Log token usage for this batch
            token_tracker = get_token_tracker()
            batch_tokens = token_tracker.current_batch_tokens
            if not self.display:
                print(f"[tokens] Batch {batch_index + 1}: prompt={batch_tokens['prompt']:,} completion={batch_tokens['completion']:,} total={batch_tokens['total']:,} calls={batch_tokens['calls']}")
            logger.info(f"[tokens] Batch {batch_index + 1}: prompt={batch_tokens['prompt']} completion={batch_tokens['completion']} total={batch_tokens['total']} calls={batch_tokens['calls']}")

            # Mark end of batch for token tracking
            token_tracker.end_batch()

        # Calculate epoch metrics
        epoch_metrics = self._calculate_epoch_metrics(batch_results)

        # Log epoch token totals
        token_tracker = get_token_tracker()
        prompt_total, completion_total, total_tokens = token_tracker.get_totals()
        if not self.display:
            print(f"[tokens] Epoch {epoch} cumulative: prompt={prompt_total:,} completion={completion_total:,} total={total_tokens:,}")
        logger.info(f"[tokens] Epoch {epoch} cumulative: prompt={prompt_total} completion={completion_total} total={total_tokens}")

        epoch_duration = time.time() - epoch_start_time
        constitution_version_end = self.constitution.version

        logger.info(f"Epoch {epoch} completed in {epoch_duration:.1f}s - "
                   f"Constitution: v{constitution_version_start} -> v{constitution_version_end}")

        epoch_result = EpochResult(
            epoch=epoch,
            batch_results=batch_results,
            constitutional_changes=constitutional_changes,
            epoch_metrics=epoch_metrics,
            constitution_version_start=constitution_version_start,
            constitution_version_end=constitution_version_end,
            epoch_duration=epoch_duration
        )

        # Display epoch summary
        if self.display:
            self.display.epoch_end(epoch_result, self.constitution)

        return epoch_result

    def _evaluate_single_constitution_direct(self, constitution_text: str,
                                              batch_docs) -> float:
        """Evaluate a constitution text on batch docs and return F1.

        Creates a lightweight wrapper so the batch processor can call
        .get_text(), .rules, and .version — then delegates to existing scorer.
        """
        class _ConstWrap:
            def __init__(self, text):
                self.text = text
                self.rules = [r for r in text.strip().splitlines() if r.strip()]
                self.version = 0
            def get_text(self):
                return self.text

        wrap = _ConstWrap(constitution_text)
        result = self.batch_processor.process_batch_with_metric(
            batch_docs, wrap, epoch=0, batch_index=0,
            metric=self.metric, error_analyzer=self.error_analyzer
        )
        return result.batch_metrics.get('f1', 0.0)

    def _validate_and_retry_rule_change(
        self,
        batch_docs: List[Any],
        original_f1: float,
        decision: Dict,
        batch_result: Any,
        old_rules: List[str],
        old_version: int
    ) -> Dict[str, Any]:
        """
        Validate rule change by checking if F1 improved. If not, retry with higher temperatures.
        Validates rule changes and retries with different temperatures if needed.

        Args:
            batch_docs: Batch documents for evaluation
            original_f1: F1 score before the rule change
            decision: Decision dict from constitutional decision maker
            batch_result: Batch result with FN/FP phrases
            old_rules: Rules before the change (for reverting)
            old_version: Constitution version before the change

        Returns:
            Dict with validation status and metrics
        """
        if not self.display:
            print(f"[rule-validate] Validating rule change (v{old_version} → v{self.constitution.version})")
            print(f"[rule-validate] Original F1: {original_f1:.3f}")

        # Minimum F1 improvement required for rule acceptance
        delta = self.config.get('algorithm', {}).get('min_f1_delta',
                    self.config.get('algorithm', {}).get('pool_update_min_delta', 0.01))

        # Get current constitution text (the modified version)
        modified_constitution_text = self.constitution.get_text()

        # Evaluate modified constitution on batch documents
        try:
            new_f1 = self._evaluate_single_constitution_direct(
                constitution_text=modified_constitution_text,
                batch_docs=batch_docs
            )
        except Exception as e:
            logger.error(f"[rule-validate] Evaluation failed: {e}")
            if not self.display:
                print(f"[rule-validate] Evaluation failed, reverting change")
            # Revert to old state
            self.constitution.rules = old_rules[:]
            self.constitution.version = old_version
            return {
                'accepted': False,
                'new_f1': None,
                'improvement': None,
                'threshold_met': False,
                'reverted': True,
                'error': str(e)
            }

        # Calculate improvement
        improvement = new_f1 - original_f1
        # Always accept the first rule when constitution was empty — we need a starting point
        is_first_rule = len(old_rules) == 0
        threshold_met = new_f1 > original_f1 or (is_first_rule and new_f1 >= original_f1)

        if not self.display:
            print(f"[rule-validate] New F1: {new_f1:.3f} (Δ={improvement:+.3f})")

        if threshold_met:
            if not self.display:
                print(f"[rule-validate] ✓ Rule change accepted: F1 {original_f1:.3f} → {new_f1:.3f} "
                      f"(Δ={improvement:+.3f}, strict improvement)")
            return {
                'accepted': True,
                'new_f1': new_f1,
                'improvement': improvement,
                'threshold_met': True,
                'reverted': False
            }
        else:
            if not self.display:
                print(f"[rule-validate] ✗ Rule regressed: F1 {original_f1:.3f} → {new_f1:.3f} "
                      f"(Δ={improvement:+.3f})")

            # Check if retry is enabled
            retry_cfg = self.config.get('algorithm', {}).get('update_retries', {})
            if (retry_cfg.get('enabled', False) and
                decision['action'] in ['add', 'edit']):

                max_attempts = retry_cfg.get('max_attempts', 1)
                temperatures = retry_cfg.get('temperature', [1.0])

                # Try additional attempts with different temperatures
                for attempt in range(1, max_attempts):
                    if attempt >= len(temperatures):
                        break

                    temp = temperatures[attempt]
                    if not self.display:
                        print(f"[rule-validate-retry] Attempt {attempt + 1}/{max_attempts} with temperature={temp}")
                    logger.info(f"[rule-validate-retry] Retrying with temperature={temp}")

                    # Revert to pre-modification state
                    self.constitution.rules = old_rules[:]
                    self.constitution.version = old_version

                    # Re-run the action with new temperature using new agents
                    try:
                        # Apply phrase limits from config
                        limited_fn, limited_fp = self._limit_error_phrases(
                            batch_result.aggregated_fn_phrases,
                            batch_result.aggregated_fp_phrases
                        )

                        if decision['action'] == 'add':
                            # Use NewRuleProposerAgent for ADD action
                            new_rule_proposal = self.agents['new_rule_proposer'].propose_rule(
                                constitution_text=self.constitution.get_text(),
                                error_context=batch_result.batch_context,
                                previous_reasoning=decision.get('reasoning', ''),
                                previous_rejections=[],
                                temperature=temp
                            )
                            new_rule_text = new_rule_proposal.get('rule_text', '').strip()
                            if new_rule_text:
                                retry_version = self.constitution.add_rule(new_rule_text, batch_result.batch_id, 0)
                            else:
                                retry_version = None

                        elif decision['action'] == 'edit':
                            rule_index = decision.get('rule_index')
                            if rule_index is not None and 0 <= rule_index < len(self.constitution.rules):
                                old_rule = self.constitution.rules[rule_index]
                                # Use NewRuleEditorAgent for EDIT action
                                new_rule_proposal = self.agents['new_rule_editor'].edit_rule(
                                    rule_index=rule_index,
                                    constitution_text=self.constitution.get_text(),
                                    error_context=batch_result.batch_context,
                                    previous_reasoning=decision.get('reasoning', ''),
                                    previous_rejections=[],
                                    temperature=temp
                                )
                                new_rule_text = new_rule_proposal.get('rule_text', '').strip()
                                if new_rule_text and new_rule_text != old_rule:
                                    retry_version = self.constitution.edit_rule(rule_index, new_rule_text, batch_result.batch_id, 0)
                                else:
                                    retry_version = None
                            else:
                                retry_version = None
                        else:
                            retry_version = None

                        if retry_version is None:
                            if not self.display:
                                print(f"[rule-validate-retry] No valid rule change generated, skipping")
                            continue

                        # Evaluate the retry attempt
                        retry_constitution_text = self.constitution.get_text()
                        retry_f1 = self._evaluate_single_constitution_direct(
                            constitution_text=retry_constitution_text,
                            batch_docs=batch_docs
                        )

                        retry_improvement = retry_f1 - original_f1
                        retry_threshold_met = retry_f1 > (original_f1 + delta)

                        if not self.display:
                            print(f"[rule-validate-retry] Retry F1: {retry_f1:.3f} (Δ={retry_improvement:+.3f})")

                        if retry_threshold_met:
                            if not self.display:
                                print(f"[rule-validate-retry] ✓ Retry successful! F1 {original_f1:.3f} → {retry_f1:.3f}")
                            logger.info(f"[rule-validate-retry] Retry succeeded with temperature={temp}")

                            return {
                                'accepted': True,
                                'new_f1': retry_f1,
                                'improvement': retry_improvement,
                                'threshold_met': True,
                                'reverted': False,
                                'retry_attempt': attempt + 1,
                                'retry_temperature': temp
                            }
                        else:
                            if not self.display:
                                print(f"[rule-validate-retry] ✗ Retry insufficient (Δ={retry_improvement:+.3f} ≤ {delta:.3f})")

                    except Exception as e:
                        logger.error(f"[rule-validate-retry] Retry attempt {attempt + 1} failed: {e}")
                        if not self.display:
                            print(f"[rule-validate-retry] Error during retry: {e}")
                        continue

                # All retries exhausted - revert to original
                if not self.display:
                    print(f"[rule-validate-retry] All {max_attempts} attempts exhausted, reverting change")
                logger.info(f"[rule-validate-retry] All retry attempts failed, reverting")

            # Revert to original state since no improvement
            self.constitution.rules = old_rules[:]
            self.constitution.version = old_version
            if not self.display:
                print(f"[rule-validate] Reverted to v{old_version} (change rejected)")

            return {
                'accepted': False,
                'new_f1': new_f1,
                'improvement': improvement,
                'threshold_met': False,
                'reverted': True
            }

    def _process_batch(self, epoch: int, batch_index: int,
                      batch_doc_pairs) -> BatchResult:
        """
        Process a single batch of documents.

        Args:
            epoch: Current epoch number
            batch_index: Current batch index
            batch_doc_pairs: List[Example]

        Returns:
            BatchResult with batch processing outcomes
        """
        batch_docs = self.batch_processor.prepare_batch_from_examples(batch_doc_pairs)
        return self.batch_processor.process_batch_with_metric(
            batch_docs, self.constitution, epoch, batch_index,
            self.metric, self.error_analyzer
        )
    
    def _save_batch_results(self, epoch: int, batch_index: int, batch_result: BatchResult,
                           decision: Dict[str, Any]) -> None:
        """Save batch results to disk."""
        epoch_dir = self.run_manager.get_run_dir() / "training" / f"epoch_{epoch:02d}"
        # Keep batch indexing consistent (1-based for display)
        batch_dir = epoch_dir / f"batch_{batch_index + 1:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch results - handle dataclass serialization
        batch_file = batch_dir / "batch_results.json"
        batch_data = {
            'batch_id': batch_result.batch_id,
            'epoch': batch_result.epoch,
            'batch_index': batch_result.batch_index,
            'doc_ids': batch_result.doc_ids,
            'aggregated_fn_phrases': batch_result.aggregated_fn_phrases[:20],  # Limit size
            'aggregated_fp_phrases': batch_result.aggregated_fp_phrases[:20],  # Limit size
            'batch_context': batch_result.batch_context[:1000],  # Limit size
            'batch_metrics': batch_result.batch_metrics,
            'num_individual_results': len(batch_result.individual_results)
        }
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        # Save decision
        decision_file = batch_dir / "decision.json"  
        with open(decision_file, 'w') as f:
            json.dump(decision, f, indent=2)
    
    def _calculate_epoch_metrics(self, batch_results: List[BatchResult]) -> Dict[str, float]:
        """Calculate aggregated metrics for an epoch."""
        if not batch_results:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'fn_count': 0, 'fp_count': 0}
        
        # Aggregate metrics across batches
        total_tp = sum(result.batch_metrics['tp'] for result in batch_results)
        total_fp = sum(result.batch_metrics['fp'] for result in batch_results)
        total_fn = sum(result.batch_metrics['fn'] for result in batch_results)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'fn_count': sum(len(result.aggregated_fn_phrases) for result in batch_results),
            'fp_count': sum(len(result.aggregated_fp_phrases) for result in batch_results),
            'num_batches': len(batch_results)
        }
    
    def _evaluate_holdout(self, holdout_docs, epoch: int) -> Dict[str, float]:
        """Evaluate current constitution on holdout documents."""
        if not holdout_docs:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'fn_count': 0, 'fp_count': 0}

        logger.info(f"Evaluating constitution v{self.constitution.version} on {len(holdout_docs)} holdout documents")

        batch_docs = self.batch_processor.prepare_batch_from_examples(holdout_docs)
        holdout_result = self.batch_processor.process_batch_with_metric(
            batch_docs, self.constitution, epoch=epoch, batch_index=99,
            metric=self.metric, error_analyzer=self.error_analyzer
        )

        # Save holdout results for this epoch
        holdout_dir = self.run_manager.get_run_dir() / "holdout"
        holdout_dir.mkdir(exist_ok=True)

        holdout_file = holdout_dir / f"epoch_{epoch:02d}_holdout_results.json"
        with open(holdout_file, 'w') as f:
            holdout_data = asdict(holdout_result)
            holdout_data.update({
                'constitution_version': self.constitution.version,
                'epoch': epoch
            })
            json.dump(holdout_data, f, indent=2)

        # Display holdout evaluation results
        if self.display:
            self.display.holdout_eval(
                holdout_metrics=holdout_result.batch_metrics,
                individual_results=holdout_result.individual_results,
                const_version=self.constitution.version,
                error_reports=holdout_result.error_reports,
            )

        # Extract metrics with counts
        metrics = holdout_result.batch_metrics.copy()
        metrics['fn_count'] = len(holdout_result.aggregated_fn_phrases)
        metrics['fp_count'] = len(holdout_result.aggregated_fp_phrases)

        logger.info(f"Holdout evaluation (Epoch {epoch}): F1={metrics['f1']:.3f}, "
                   f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                   f"FN={metrics['fn_count']}, FP={metrics['fp_count']}")

        return metrics
    
    def _calculate_training_metrics(self, epoch_results: List[EpochResult]) -> Dict[str, float]:
        """Calculate overall training metrics."""
        if not epoch_results:
            return {'final_f1': 0.0, 'avg_f1': 0.0, 'f1_improvement': 0.0}
        
        f1_scores = [result.epoch_metrics['f1'] for result in epoch_results]
        
        return {
            'final_f1': f1_scores[-1],
            'initial_f1': f1_scores[0],
            'avg_f1': sum(f1_scores) / len(f1_scores),
            'f1_improvement': f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0.0,
            'total_constitutional_changes': sum(len(result.constitutional_changes) for result in epoch_results)
        }
    
    def _save_training_results(self, training_result: TrainingResult) -> None:
        """Save final training results."""
        results_file = self.run_manager.get_run_dir() / "training_results.json"

        # Convert to serializable format
        serializable_result = {
            'total_epochs': training_result.total_epochs,
            'final_constitution_version': training_result.final_constitution_version,
            'training_metrics': training_result.training_metrics,
            'holdout_metrics': training_result.holdout_metrics,
            'total_training_time': training_result.total_training_time,
            'token_usage': training_result.token_usage,  # Include token tracking
            'baseline_holdout_metrics': training_result.baseline_holdout_metrics,
            'best_constitution_info': training_result.best_constitution_info,
            'epoch_summary': [
                {
                    'epoch': result.epoch,
                    'constitution_changes': len(result.constitutional_changes),
                    'f1_score': result.epoch_metrics['f1'],
                    'constitution_version_end': result.constitution_version_end
                }
                for result in training_result.epoch_results
            ]
        }

        with open(results_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)

        logger.info(f"Saved training results to {results_file}")

        # Save intermediate validation checkpoint results
        if hasattr(self, 'intermediate_validation_results') and self.intermediate_validation_results:
            checkpoints_file = self.run_manager.get_run_dir() / "validation_checkpoints.json"
            with open(checkpoints_file, 'w') as f:
                json.dump(self.intermediate_validation_results, f, indent=2)
            logger.info(f"Saved {len(self.intermediate_validation_results)} validation checkpoints to {checkpoints_file}")
    
    def _create_training_plots(self, training_result: TrainingResult) -> None:
        """Create comprehensive training visualization plots."""
        plots_dir = self.run_manager.get_run_dir() / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Training Progress Plot (F1, Precision, Recall across epochs)
        self._plot_training_progress(training_result, plots_dir)
        
        # 2. Constitutional Evolution Plot (Rules added/edited/removed)
        self._plot_constitutional_evolution(training_result, plots_dir)
        
        # 3. Batch-level Performance Heatmap
        self._plot_batch_performance_heatmap(training_result, plots_dir)
        
        # 4. Error Distribution Plot (FN/FP counts)
        self._plot_error_distribution(training_result, plots_dir)
        
        # 5. Final Holdout Performance Summary
        self._plot_holdout_performance_summary(training_result, plots_dir)
        
        # 6. Rule Impact Analysis
        self._plot_rule_impact_analysis(plots_dir)
        
        # 7. FN/FP Phrase Evolution (MOST IMPORTANT)
        self._plot_fn_fp_phrase_evolution(training_result, plots_dir)

        # 8. Aggregated FN/FP Reduction Analysis
        self._plot_aggregated_error_reduction(training_result, plots_dir)

        # 9. Validation Checkpoint Plot (batch-level validation scores)
        self._plot_validation_checkpoints(plots_dir)

        logger.info(f"Created training plots in {plots_dir}")
    
    def _plot_training_progress(self, training_result: TrainingResult, plots_dir: Path) -> None:
        """Plot F1, Precision, Recall progression across epochs with intermediate checkpoints."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Load intermediate validation checkpoints to get epoch 0 baseline
        checkpoints_file = self.run_manager.get_run_dir() / "validation_checkpoints.json"
        epoch0_metrics = None
        if checkpoints_file.exists():
            try:
                with open(checkpoints_file, 'r') as f:
                    checkpoints = json.load(f)
                # Find the initial checkpoint (global_batch_index = 0 or checkpoint_type = 'initial')
                for cp in checkpoints:
                    if cp.get('checkpoint_type') == 'initial' or cp.get('global_batch_index', -1) == 0:
                        epoch0_metrics = cp
                        break
            except Exception as e:
                logger.warning(f"Failed to load epoch 0 metrics: {e}")

        # Collect training metrics per epoch - START WITH EPOCH 0 if available
        epochs = []
        f1_scores = []
        precisions = []
        recalls = []
        fn_counts = []
        fp_counts = []

        # Add epoch 0 (baseline before training) if we have it
        if epoch0_metrics:
            epochs.append(0)
            f1_scores.append(epoch0_metrics.get('f1', 0.0))
            precisions.append(epoch0_metrics.get('precision', 0.0))
            recalls.append(epoch0_metrics.get('recall', 0.0))
            fn_counts.append(epoch0_metrics.get('fn_count', 0))
            fp_counts.append(epoch0_metrics.get('fp_count', 0))

        for epoch_result in training_result.epoch_results:
            epochs.append(epoch_result.epoch)
            f1_scores.append(epoch_result.epoch_metrics['f1'])
            precisions.append(epoch_result.epoch_metrics['precision'])
            recalls.append(epoch_result.epoch_metrics['recall'])
            fn_counts.append(epoch_result.epoch_metrics.get('fn_count', 0))
            fp_counts.append(epoch_result.epoch_metrics.get('fp_count', 0))

        # Collect validation metrics per epoch - START WITH EPOCH 0 if available
        val_f1_scores = []
        val_precisions = []
        val_recalls = []
        val_fn_counts = []
        val_fp_counts = []

        # Add epoch 0 validation baseline
        if epoch0_metrics:
            val_f1_scores.append(epoch0_metrics.get('f1', 0.0))
            val_precisions.append(epoch0_metrics.get('precision', 0.0))
            val_recalls.append(epoch0_metrics.get('recall', 0.0))
            val_fn_counts.append(epoch0_metrics.get('fn_count', 0))
            val_fp_counts.append(epoch0_metrics.get('fp_count', 0))

        for val_result in training_result.holdout_results:
            val_f1_scores.append(val_result['f1'])
            val_precisions.append(val_result['precision'])
            val_recalls.append(val_result['recall'])
            val_fn_counts.append(val_result['fn_count'])
            val_fp_counts.append(val_result['fp_count'])

        # Load intermediate validation checkpoints if they exist
        checkpoints_file = self.run_manager.get_run_dir() / "validation_checkpoints.json"
        checkpoint_batches = []
        checkpoint_f1s = []
        checkpoint_versions = []
        if checkpoints_file.exists():
            try:
                with open(checkpoints_file, 'r') as f:
                    checkpoints = json.load(f)
                for cp in checkpoints:
                    checkpoint_batches.append(cp.get('global_batch_index', 0))
                    checkpoint_f1s.append(cp.get('f1', 0.0))
                    checkpoint_versions.append(cp.get('constitution_version', 0))
            except Exception as e:
                logger.warning(f"Failed to load validation checkpoints: {e}")

        # Plot 1: F1 Score Progression (now includes intermediate checkpoints)
        ax1 = axes[0, 0]

        # If we have intermediate checkpoints, plot them as main line
        if checkpoint_batches and len(checkpoint_batches) > 1:
            # Plot checkpoint F1 as stepped line showing actual progression
            ax1.plot(checkpoint_batches, checkpoint_f1s, 'o-', color='#FF6347', linewidth=2,
                     markersize=6, alpha=0.8, label='Validation F1 (checkpoints)')

            # Add version annotations for key changes
            prev_version = -1
            for i, (batch, f1, version) in enumerate(zip(checkpoint_batches, checkpoint_f1s, checkpoint_versions)):
                if version != prev_version:
                    ax1.annotate(f'v{version}', (batch, f1), textcoords="offset points",
                                xytext=(0, 8), ha='center', fontsize=7, alpha=0.7)
                    prev_version = version

            ax1.set_xlabel('Global Batch Index')
            # Also show epoch markers
            num_batches = self.num_batches
            for epoch in epochs:
                epoch_batch = epoch * num_batches
                ax1.axvline(x=epoch_batch, color='gray', linestyle=':', alpha=0.3)
                ax1.text(epoch_batch, ax1.get_ylim()[1] * 0.95, f'E{epoch}', ha='center', fontsize=8, alpha=0.5)
        else:
            # Fallback: plot epoch-level validation F1
            ax1.plot(epochs, val_f1_scores, 's-', color='#FF6347', linewidth=2, markersize=8, label='Validation F1')
            ax1.set_xlabel('Epoch')

        # Training F1 (epoch level)
        ax1.plot(epochs if not checkpoint_batches else [e * self.num_batches for e in epochs],
                 f1_scores, 'o--', color='#DC143C', linewidth=2, markersize=8, alpha=0.6, label='Training F1 (epoch)')
        ax1.axhline(y=training_result.holdout_metrics['f1'], color='green', linestyle='--', label='Holdout F1')
        
        # Mark best constitution
        best_epoch = training_result.best_constitution_info['epoch']
        best_f1 = training_result.best_constitution_info['f1_score']

        # Convert best epoch to batch index if using batch-level x-axis
        if checkpoint_batches and len(checkpoint_batches) > 1:
            best_x = best_epoch * self.num_batches
        else:
            best_x = best_epoch

        ax1.axvline(x=best_x, color='purple', linestyle=':', alpha=0.7, label='Best Constitution')
        ax1.plot(best_x, best_f1, 'D', color='purple', markersize=10, markerfacecolor='yellow',
                markeredgecolor='purple', markeredgewidth=2)

        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score Progression (Train/Holdout)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Add best constitution annotation
        ax1.annotate(f'Best: v{training_result.best_constitution_info["version"]}\nF1={best_f1:.3f}',
                    (best_x, best_f1), textcoords="offset points",
                    xytext=(10, 10), ha='left', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Highlight epoch 0 baseline if available
        if epoch0_metrics:
            baseline_f1 = epoch0_metrics.get('f1', 0.0)
            baseline_x = 0 if not checkpoint_batches else 0
            # Mark baseline with a special marker
            ax1.plot(baseline_x, baseline_f1, 'o', color='blue', markersize=12,
                    markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2,
                    label='Baseline (Epoch 0)', zorder=10)
            ax1.annotate(f'Baseline\nF1={baseline_f1:.3f}',
                        (baseline_x, baseline_f1), textcoords="offset points",
                        xytext=(-50, -30), ha='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))

            # Add improvement from baseline to best
            improvement = best_f1 - baseline_f1
            improvement_pct = (improvement / max(baseline_f1, 0.001)) * 100
            ax1.text(0.02, 0.02, f'Improvement from Baseline: {baseline_f1:.3f} → {best_f1:.3f} (+{improvement:.3f}, +{improvement_pct:.1f}%)',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot 2: Precision/Recall Progression
        ax2 = axes[0, 1]
        ax2.plot(epochs, precisions, 's-', color='#2E8B57', linewidth=2, markersize=8, label='Train Precision')
        ax2.plot(epochs, recalls, '^-', color='#4682B4', linewidth=2, markersize=8, label='Train Recall')
        ax2.plot(epochs, val_precisions, 's--', color='#90EE90', linewidth=2, markersize=6, label='Val Precision')
        ax2.plot(epochs, val_recalls, '^--', color='#87CEEB', linewidth=2, markersize=6, label='Val Recall')

        # Mark epoch 0 baseline
        if epoch0_metrics and 0 in epochs:
            ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Baseline')

        # Mark best constitution
        ax2.axvline(x=best_epoch, color='purple', linestyle=':', alpha=0.7)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall Progression (Train/Val)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Plot 3: Error Counts (Train vs Validation)
        ax3 = axes[1, 0]
        x_pos = range(len(epochs))
        width = 0.2
        
        # Training error bars
        ax3.bar([p - 1.5*width for p in x_pos], fn_counts, width, label='Train FN', color='#FF6347', alpha=0.8)
        ax3.bar([p - 0.5*width for p in x_pos], fp_counts, width, label='Train FP', color='#FFA500', alpha=0.8)
        
        # Validation error bars
        ax3.bar([p + 0.5*width for p in x_pos], val_fn_counts, width, label='Val FN', color='#DC143C', alpha=0.6)
        ax3.bar([p + 1.5*width for p in x_pos], val_fp_counts, width, label='Val FP', color='#FF8C00', alpha=0.6)
        
        # Mark best constitution
        ax3.axvline(x=best_epoch-1, color='purple', linestyle=':', alpha=0.7)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Error Count')
        ax3.set_title('False Negatives/Positives (Train vs Validation)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(epochs)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Training Summary Stats
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Summary text
        best_info = training_result.best_constitution_info
        final_val = training_result.holdout_results[-1] if training_result.holdout_results else {}
        
        summary_text = f"""Training Summary
        
Total Epochs: {training_result.total_epochs}
Training Time: {training_result.total_training_time:.1f} seconds

Best Constitution (Holdout-Selected):
  Version: v{best_info['version']} (Epoch {best_info['epoch']})
  Holdout F1: {best_info['f1_score']:.3f}
  Holdout FN: {best_info['fn_count']}

Final Holdout Performance:
  F1 Score: {training_result.holdout_metrics['f1']:.3f}
  Precision: {training_result.holdout_metrics['precision']:.3f}
  Recall: {training_result.holdout_metrics['recall']:.3f}

Performance Gap:
  Best→Final Holdout F1: {best_info['f1_score']:.3f} → {training_result.holdout_metrics['f1']:.3f}

Selected Constitution: v{training_result.final_constitution_version}
"""
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Epoch-Based Constitutional Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_constitutional_evolution(self, training_result: TrainingResult, plots_dir: Path) -> None:
        """Plot constitutional changes over time."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Load evolution log
        evolution_file = self.run_manager.get_run_dir() / "constitutions" / "evolution_log.json"
        if evolution_file.exists():
            with open(evolution_file, 'r') as f:
                evolution_data = json.load(f)
                evolution_log = evolution_data['evolution_log']
        else:
            evolution_log = []
        
        # Plot 1: Rule Changes Timeline
        ax1 = axes[0]
        
        epochs = []
        batches = []
        actions = []
        colors = {'add': 'green', 'edit': 'blue', 'remove': 'red'}
        
        for change in evolution_log:
            epochs.append(change['epoch'])
            batches.append(change['batch_index'])
            actions.append(change['action'])
        
        for i, (epoch, batch, action) in enumerate(zip(epochs, batches, actions)):
            x = epoch + batch * 0.4  # Spread batches within epoch
            ax1.scatter(x, i + 1, c=colors.get(action, 'gray'), s=100, alpha=0.7)
            ax1.text(x + 0.05, i + 1, action.upper(), fontsize=8)
        
        ax1.set_xlabel('Epoch.Batch')
        ax1.set_ylabel('Change Number')
        ax1.set_title('Constitutional Changes Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        for action, color in colors.items():
            ax1.scatter([], [], c=color, s=100, label=action.title())
        ax1.legend(loc='upper left')
        
        # Plot 2: Constitution Versions and F1 Scores
        ax2 = axes[1]
        
        versions = []
        f1_scores = []
        
        for change in evolution_log:
            versions.append(change['new_version'])
            f1_scores.append(change['batch_f1'])
        
        ax2.plot(versions, f1_scores, 'o-', color='purple', linewidth=2, markersize=8)
        ax2.set_xlabel('Constitution Version')
        ax2.set_ylabel('Batch F1 Score')
        ax2.set_title('Performance by Constitution Version')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add annotations
        for i, (v, f1) in enumerate(zip(versions, f1_scores)):
            ax2.annotate(f'{f1:.2f}', (v, f1), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)
        
        plt.suptitle('Constitutional Evolution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'constitutional_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_performance_heatmap(self, training_result: TrainingResult, plots_dir: Path) -> None:
        """Create heatmap of batch-level performance."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create performance matrix
        performance_matrix = []
        batch_labels = []
        
        for epoch_result in training_result.epoch_results:
            epoch_row = []
            for batch_result in epoch_result.batch_results:
                epoch_row.append(batch_result.batch_metrics['f1'])
                if len(performance_matrix) == 0:  # Only add batch labels once
                    batch_labels.append(f"Batch {batch_result.batch_index + 1}")
            performance_matrix.append(epoch_row)
        
        # Create heatmap
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(batch_labels)))
        ax.set_xticklabels(batch_labels)
        ax.set_yticks(range(len(training_result.epoch_results)))
        ax.set_yticklabels([f"Epoch {i+1}" for i in range(len(training_result.epoch_results))])
        
        # Add text annotations
        for i in range(len(performance_matrix)):
            for j in range(len(performance_matrix[i])):
                text = ax.text(j, i, f'{performance_matrix[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1 Score', rotation=270, labelpad=15)
        
        ax.set_title('Batch-Level F1 Score Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'batch_performance_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, training_result: TrainingResult, plots_dir: Path) -> None:
        """Plot distribution of errors across training."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect error data per batch
        all_fn_counts = []
        all_fp_counts = []
        batch_ids = []
        
        for epoch_result in training_result.epoch_results:
            for batch_result in epoch_result.batch_results:
                batch_ids.append(f"E{epoch_result.epoch}B{batch_result.batch_index + 1}")
                all_fn_counts.append(len(batch_result.aggregated_fn_phrases))
                all_fp_counts.append(len(batch_result.aggregated_fp_phrases))
        
        # Plot 1: FN/FP Distribution
        ax1 = axes[0, 0]
        x_pos = range(len(batch_ids))
        width = 0.35
        ax1.bar([p - width/2 for p in x_pos], all_fn_counts, width, label='FN', color='#FF6347', alpha=0.7)
        ax1.bar([p + width/2 for p in x_pos], all_fp_counts, width, label='FP', color='#FFA500', alpha=0.7)
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Error Count')
        ax1.set_title('Error Distribution Across Batches')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(batch_ids, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Error Ratio
        ax2 = axes[0, 1]
        error_ratios = [fp/max(fn, 1) for fn, fp in zip(all_fn_counts, all_fp_counts)]
        ax2.plot(range(len(batch_ids)), error_ratios, 'o-', color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('FP/FN Ratio')
        ax2.set_title('False Positive to False Negative Ratio')
        ax2.set_xticks(range(len(batch_ids)))
        ax2.set_xticklabels(batch_ids, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal FP/FN')
        ax2.legend()
        
        # Plot 3: Cumulative Errors
        ax3 = axes[1, 0]
        cumulative_fn = [sum(all_fn_counts[:i+1]) for i in range(len(all_fn_counts))]
        cumulative_fp = [sum(all_fp_counts[:i+1]) for i in range(len(all_fp_counts))]
        ax3.plot(range(len(batch_ids)), cumulative_fn, 's-', color='#FF6347', linewidth=2, label='Cumulative FN')
        ax3.plot(range(len(batch_ids)), cumulative_fp, '^-', color='#FFA500', linewidth=2, label='Cumulative FP')
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('Cumulative Error Count')
        ax3.set_title('Cumulative Error Accumulation')
        ax3.set_xticks(range(len(batch_ids)))
        ax3.set_xticklabels(batch_ids, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""Error Statistics
        
Total False Negatives: {sum(all_fn_counts)}
Total False Positives: {sum(all_fp_counts)}

Average FN per Batch: {sum(all_fn_counts)/len(all_fn_counts):.1f}
Average FP per Batch: {sum(all_fp_counts)/len(all_fp_counts):.1f}

Min FN: {min(all_fn_counts)}
Max FN: {max(all_fn_counts)}

Min FP: {min(all_fp_counts)}
Max FP: {max(all_fp_counts)}

Final Batch FN: {all_fn_counts[-1] if all_fn_counts else 0}
Final Batch FP: {all_fp_counts[-1] if all_fp_counts else 0}
"""
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Error Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_holdout_performance_summary(self, training_result: TrainingResult, plots_dir: Path) -> None:
        """Create comprehensive holdout performance visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        holdout_metrics = training_result.holdout_metrics

        # Plot 1: Holdout Metrics Bar Chart
        ax1 = axes[0, 0]
        metrics_names = ['Precision', 'Recall', 'F1']
        metrics_values = [holdout_metrics['precision'], holdout_metrics['recall'], holdout_metrics['f1']]
        colors = ['#2E8B57', '#4682B4', '#DC143C']

        bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_title('Holdout Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # Plot 2: Confusion Matrix Components
        ax2 = axes[0, 1]
        counts = [holdout_metrics['tp'], holdout_metrics['fp'], holdout_metrics['fn']]
        count_labels = ['True Positive', 'False Positive', 'False Negative']
        count_colors = ['#228B22', '#FF6347', '#FF4500']

        bars2 = ax2.bar(count_labels, counts, color=count_colors, alpha=0.7)
        ax2.set_title('Holdout Predictions Breakdown')
        ax2.set_ylabel('Count')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars2, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom')

        # Plot 3: Training vs Holdout Comparison
        ax3 = axes[1, 0]
        categories = ['F1 Score', 'Precision', 'Recall']
        train_values = [
            training_result.training_metrics['final_f1'],
            training_result.epoch_results[-1].epoch_metrics['precision'] if training_result.epoch_results else 0,
            training_result.epoch_results[-1].epoch_metrics['recall'] if training_result.epoch_results else 0
        ]
        holdout_values = [holdout_metrics['f1'], holdout_metrics['precision'], holdout_metrics['recall']]

        x = range(len(categories))
        width = 0.35
        ax3.bar([i - width/2 for i in x], train_values, width, label='Training', color='blue', alpha=0.7)
        ax3.bar([i + width/2 for i in x], holdout_values, width, label='Holdout', color='green', alpha=0.7)
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Score')
        ax3.set_title('Training vs Holdout Performance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 1)

        # Plot 4: Holdout Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""Holdout Summary

Documents Evaluated: {holdout_metrics['num_docs']}
Final Constitution: v{training_result.final_constitution_version}

Performance:
  F1 Score: {holdout_metrics['f1']:.3f}
  Precision: {holdout_metrics['precision']:.3f}
  Recall: {holdout_metrics['recall']:.3f}

Predictions:
  True Positives: {holdout_metrics['tp']}
  False Positives: {holdout_metrics['fp']}
  False Negatives: {holdout_metrics['fn']}

Generalization Gap:
  Training F1: {training_result.training_metrics['final_f1']:.3f}
  Holdout F1: {holdout_metrics['f1']:.3f}
  Gap: {abs(training_result.training_metrics['final_f1'] - holdout_metrics['f1']):.3f}
"""
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Holdout Performance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'holdout_performance_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rule_impact_analysis(self, plots_dir: Path) -> None:
        """Analyze and plot the impact of individual rules."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Load evolution log for rule impact data
        evolution_file = self.run_manager.get_run_dir() / "constitutions" / "evolution_log.json"
        if not evolution_file.exists():
            plt.close()
            return
        
        with open(evolution_file, 'r') as f:
            evolution_data = json.load(f)
            evolution_log = evolution_data['evolution_log']
            rule_history = evolution_data.get('rule_history', {})
        
        # Plot 1: F1 Impact per Rule Change
        ax1 = axes[0]
        
        changes = []
        f1_before = []
        f1_after = []
        
        for i, change in enumerate(evolution_log):
            changes.append(f"{change['action'].upper()} {change.get('rule_index', 'NEW')}")
            
            # Get F1 before (from previous change or 0)
            if i > 0:
                f1_before.append(evolution_log[i-1]['batch_f1'])
            else:
                f1_before.append(0)
            
            f1_after.append(change['batch_f1'])
        
        x_pos = range(len(changes))
        width = 0.35
        ax1.bar([p - width/2 for p in x_pos], f1_before, width, label='Before', color='red', alpha=0.7)
        ax1.bar([p + width/2 for p in x_pos], f1_after, width, label='After', color='green', alpha=0.7)
        ax1.set_xlabel('Rule Change')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score Impact of Rule Changes')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(changes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Rule Modification Frequency
        ax2 = axes[1]
        
        if rule_history:
            rule_names = list(rule_history.keys())
            mod_counts = [rule_history[r]['modification_count'] for r in rule_names]
            
            ax2.bar(range(len(rule_names)), mod_counts, color='purple', alpha=0.7)
            ax2.set_xlabel('Rule')
            ax2.set_ylabel('Modification Count')
            ax2.set_title('Rule Modification Frequency')
            ax2.set_xticks(range(len(rule_names)))
            ax2.set_xticklabels(rule_names, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Rule Impact Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'rule_impact_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_fn_fp_phrase_evolution(self, training_result: TrainingResult, plots_dir: Path) -> None:
        """Plot the evolution of FN/FP phrases - MOST IMPORTANT VISUALIZATION."""
        fig = plt.figure(figsize=(20, 14))
        
        # Load all batch results to get actual phrases
        all_batch_data = []
        for epoch_result in training_result.epoch_results:
            for batch_result in epoch_result.batch_results:
                batch_info = {
                    'epoch': epoch_result.epoch,
                    'batch': batch_result.batch_index,
                    'batch_id': f"E{epoch_result.epoch}B{batch_result.batch_index+1}",
                    'fn_phrases': batch_result.aggregated_fn_phrases,
                    'fp_phrases': batch_result.aggregated_fp_phrases,
                    'fn_count': len(batch_result.aggregated_fn_phrases),
                    'fp_count': len(batch_result.aggregated_fp_phrases),
                    'f1': batch_result.batch_metrics['f1']
                }
                all_batch_data.append(batch_info)
        
        # Create 3x2 subplot grid
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
        
        # Plot 1: FN Count Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        batch_labels = [d['batch_id'] for d in all_batch_data]
        fn_counts = [d['fn_count'] for d in all_batch_data]
        colors = ['red' if i == 0 else 'orange' for i in range(len(fn_counts))]
        bars1 = ax1.bar(range(len(batch_labels)), fn_counts, color=colors, alpha=0.7)
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('FN Count')
        ax1.set_title('False Negative Count Evolution (Lower is Better)', fontweight='bold')
        ax1.set_xticks(range(len(batch_labels)))
        ax1.set_xticklabels(batch_labels, rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels and change indicators
        for i, (bar, value) in enumerate(zip(bars1, fn_counts)):
            ax1.text(bar.get_x() + bar.get_width()/2., value + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold')
            if i > 0:
                change = fn_counts[i] - fn_counts[i-1]
                color = 'green' if change < 0 else 'red'
                ax1.text(bar.get_x() + bar.get_width()/2., value/2,
                        f"{change:+d}", ha='center', va='center', 
                        color=color, fontweight='bold', fontsize=10)
        
        # Plot 2: FP Count Evolution  
        ax2 = fig.add_subplot(gs[0, 1])
        fp_counts = [d['fp_count'] for d in all_batch_data]
        colors = ['red' if i == 0 else 'orange' for i in range(len(fp_counts))]
        bars2 = ax2.bar(range(len(batch_labels)), fp_counts, color=colors, alpha=0.7)
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('FP Count')
        ax2.set_title('False Positive Count Evolution (Lower is Better)', fontweight='bold')
        ax2.set_xticks(range(len(batch_labels)))
        ax2.set_xticklabels(batch_labels, rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels and change indicators
        for i, (bar, value) in enumerate(zip(bars2, fp_counts)):
            ax2.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                    str(value), ha='center', va='bottom', fontweight='bold')
            if i > 0:
                change = fp_counts[i] - fp_counts[i-1]
                color = 'green' if change < 0 else 'red'
                ax2.text(bar.get_x() + bar.get_width()/2., value/2,
                        f"{change:+d}", ha='center', va='center',
                        color=color, fontweight='bold', fontsize=10)
        
        # Plot 3: FN/FP Ratio Evolution
        ax3 = fig.add_subplot(gs[1, 0])
        fn_fp_ratios = [d['fn_count']/(d['fp_count']+1) for d in all_batch_data]
        ax3.plot(range(len(batch_labels)), fn_fp_ratios, 'o-', color='purple', 
                linewidth=2, markersize=8)
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('FN/FP Ratio')
        ax3.set_title('FN/FP Ratio Evolution (Target: Minimize FN)', fontweight='bold')
        ax3.set_xticks(range(len(batch_labels)))
        ax3.set_xticklabels(batch_labels, rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        for i, (x, y) in enumerate(zip(range(len(batch_labels)), fn_fp_ratios)):
            ax3.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)
        
        # Plot 4: Total Error Reduction
        ax4 = fig.add_subplot(gs[1, 1])
        total_errors = [d['fn_count'] + d['fp_count'] for d in all_batch_data]
        ax4.plot(range(len(batch_labels)), total_errors, 's-', color='darkred',
                linewidth=2, markersize=8, label='Total Errors')
        ax4.plot(range(len(batch_labels)), fn_counts, '^-', color='red',
                linewidth=2, markersize=6, label='FN', alpha=0.7)
        ax4.plot(range(len(batch_labels)), fp_counts, 'v-', color='orange',
                linewidth=2, markersize=6, label='FP', alpha=0.7)
        ax4.set_xlabel('Batch')
        ax4.set_ylabel('Error Count')
        ax4.set_title('Total Error Trajectory', fontweight='bold')
        ax4.set_xticks(range(len(batch_labels)))
        ax4.set_xticklabels(batch_labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Example FN Phrases (Most Important)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Get first and last batch FN/FP phrases for comparison
        first_batch = all_batch_data[0]
        last_batch = all_batch_data[-1]
        
        # Create comparison text
        comparison_text = "FN/FP PHRASE EVOLUTION ANALYSIS\n"
        comparison_text += "="*80 + "\n\n"
        
        comparison_text += f"INITIAL STATE (Epoch 1, Batch 1):\n"
        comparison_text += f"  FN Count: {first_batch['fn_count']}, FP Count: {first_batch['fp_count']}\n"
        comparison_text += f"  Sample FN Phrases:\n"
        for phrase in first_batch['fn_phrases'][:8]:
            comparison_text += f"    • {phrase}\n"
        comparison_text += f"  Sample FP Phrases:\n"
        for phrase in first_batch['fp_phrases'][:5]:
            comparison_text += f"    • {phrase}\n"
        
        comparison_text += f"\nFINAL STATE (Epoch {last_batch['epoch']}, Batch {last_batch['batch']+1}):\n"
        comparison_text += f"  FN Count: {last_batch['fn_count']}, FP Count: {last_batch['fp_count']}\n"
        comparison_text += f"  Sample FN Phrases (Remaining):\n"
        for phrase in last_batch['fn_phrases'][:8]:
            comparison_text += f"    • {phrase}\n"
        comparison_text += f"  Sample FP Phrases (Remaining):\n"
        for phrase in last_batch['fp_phrases'][:5]:
            comparison_text += f"    • {phrase}\n"
        
        comparison_text += f"\nOVERALL REDUCTION:\n"
        comparison_text += f"  FN Reduction: {first_batch['fn_count']} → {last_batch['fn_count']} "
        comparison_text += f"({first_batch['fn_count'] - last_batch['fn_count']} phrases eliminated, "
        comparison_text += f"{((first_batch['fn_count'] - last_batch['fn_count'])/max(first_batch['fn_count'], 1)*100):.1f}%)\n"
        comparison_text += f"  FP Reduction: {first_batch['fp_count']} → {last_batch['fp_count']} "
        comparison_text += f"({first_batch['fp_count'] - last_batch['fp_count']} phrases eliminated, "
        comparison_text += f"{((first_batch['fp_count'] - last_batch['fp_count'])/max(first_batch['fp_count'], 1)*100):.1f}%)\n"
        
        ax5.text(0.05, 0.95, comparison_text, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('FALSE NEGATIVE / FALSE POSITIVE PHRASE EVOLUTION (PRIMARY METRICS)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(plots_dir / 'fn_fp_phrase_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_aggregated_error_reduction(self, training_result: TrainingResult, plots_dir: Path) -> None:
        """Plot aggregated error reduction across epochs - showing actual improvement."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Collect data per epoch
        epoch_data = []
        for epoch_result in training_result.epoch_results:
            # Aggregate all FN/FP from the epoch
            all_fn = set()
            all_fp = set()
            for batch_result in epoch_result.batch_results:
                all_fn.update(batch_result.aggregated_fn_phrases)
                all_fp.update(batch_result.aggregated_fp_phrases)
            
            epoch_data.append({
                'epoch': epoch_result.epoch,
                'unique_fn': len(all_fn),
                'unique_fp': len(all_fp),
                'f1': epoch_result.epoch_metrics['f1'],
                'precision': epoch_result.epoch_metrics['precision'],
                'recall': epoch_result.epoch_metrics['recall'],
                'sample_fn': list(all_fn)[:10],
                'sample_fp': list(all_fp)[:10]
            })
        
        epochs = [d['epoch'] for d in epoch_data]
        
        # Plot 1: Unique FN Reduction
        ax1 = axes[0, 0]
        unique_fn = [d['unique_fn'] for d in epoch_data]
        bars1 = ax1.bar(epochs, unique_fn, color='red', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Unique FN Phrases')
        ax1.set_title('Unique False Negatives per Epoch', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars1, unique_fn):
            ax1.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # Add reduction percentage
        if len(unique_fn) > 1:
            reduction = ((unique_fn[0] - unique_fn[-1]) / max(unique_fn[0], 1)) * 100
            ax1.text(0.5, 0.95, f"Total Reduction: {reduction:.1f}%",
                    transform=ax1.transAxes, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Plot 2: Unique FP Reduction
        ax2 = axes[0, 1]
        unique_fp = [d['unique_fp'] for d in epoch_data]
        bars2 = ax2.bar(epochs, unique_fp, color='orange', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Unique FP Phrases')
        ax2.set_title('Unique False Positives per Epoch', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars2, unique_fp):
            ax2.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # Add reduction percentage
        if len(unique_fp) > 1:
            reduction = ((unique_fp[0] - unique_fp[-1]) / max(unique_fp[0], 1)) * 100
            ax2.text(0.5, 0.95, f"Total Reduction: {reduction:.1f}%",
                    transform=ax2.transAxes, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Plot 3: Combined Error Reduction
        ax3 = axes[0, 2]
        total_errors = [d['unique_fn'] + d['unique_fp'] for d in epoch_data]
        ax3.plot(epochs, total_errors, 'o-', color='darkred', linewidth=3, 
                markersize=10, label='Total Unique Errors')
        ax3.fill_between(epochs, 0, total_errors, alpha=0.3, color='red')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Total Unique Errors')
        ax3.set_title('Total Unique Error Reduction', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        for x, y in zip(epochs, total_errors):
            ax3.annotate(str(y), (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontweight='bold')
        
        # Plot 4: F1 vs FN Relationship
        ax4 = axes[1, 0]
        f1_scores = [d['f1'] for d in epoch_data]
        ax4.plot(unique_fn, f1_scores, 'o-', color='blue', linewidth=2, markersize=8)
        ax4.set_xlabel('Unique FN Count')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score vs False Negatives', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.invert_xaxis()  # Lower FN on the right
        
        for fn, f1 in zip(unique_fn, f1_scores):
            ax4.annotate(f'E{epochs[unique_fn.index(fn)]}', (fn, f1),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        # Plot 5: Precision-Recall Trade-off
        ax5 = axes[1, 1]
        precisions = [d['precision'] for d in epoch_data]
        recalls = [d['recall'] for d in epoch_data]
        
        ax5.plot(recalls, precisions, 'o-', color='green', linewidth=2, markersize=8)
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Precision-Recall Trade-off', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        
        for i, (r, p) in enumerate(zip(recalls, precisions)):
            ax5.annotate(f'E{epochs[i]}', (r, p), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)
        
        # Plot 6: Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        initial = epoch_data[0]
        final = epoch_data[-1]
        
        summary_text = f"""AGGREGATED ERROR REDUCTION SUMMARY
{'='*40}

INITIAL (Epoch 1):
  Unique FN: {initial['unique_fn']}
  Unique FP: {initial['unique_fp']}
  Total Errors: {initial['unique_fn'] + initial['unique_fp']}
  F1 Score: {initial['f1']:.3f}

FINAL (Epoch {final['epoch']}):
  Unique FN: {final['unique_fn']}
  Unique FP: {final['unique_fp']}
  Total Errors: {final['unique_fn'] + final['unique_fp']}
  F1 Score: {final['f1']:.3f}

IMPROVEMENTS:
  FN Reduced by: {initial['unique_fn'] - final['unique_fn']} phrases
  FP Reduced by: {initial['unique_fp'] - final['unique_fp']} phrases
  Total Reduced: {(initial['unique_fn'] + initial['unique_fp']) - (final['unique_fn'] + final['unique_fp'])} phrases
  F1 Improved by: {(final['f1'] - initial['f1']):.3f}

PERCENTAGE IMPROVEMENTS:
  FN: {((initial['unique_fn'] - final['unique_fn'])/max(initial['unique_fn'], 1)*100):.1f}% reduction
  FP: {((initial['unique_fp'] - final['unique_fp'])/max(initial['unique_fp'], 1)*100):.1f}% reduction
  F1: {((final['f1'] - initial['f1'])/max(initial['f1'], 0.01)*100):.1f}% improvement
"""
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('AGGREGATED ERROR REDUCTION ANALYSIS', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'aggregated_error_reduction.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_validation_checkpoints(self, plots_dir: Path) -> None:
        """Plot validation F1 scores at each checkpoint batch - shows stepped progression."""
        checkpoints_file = self.run_manager.get_run_dir() / "validation_checkpoints.json"

        if not checkpoints_file.exists():
            logger.info("No validation checkpoints file found, skipping checkpoint plot")
            return

        try:
            with open(checkpoints_file, 'r') as f:
                checkpoints = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load validation checkpoints: {e}")
            return

        if not checkpoints or len(checkpoints) < 2:
            logger.info("Not enough validation checkpoints for plotting")
            return

        # Extract data
        batch_indices = [cp.get('global_batch_index', 0) for cp in checkpoints]
        f1_scores = [cp.get('f1', 0.0) for cp in checkpoints]
        precisions = [cp.get('precision', 0.0) for cp in checkpoints]
        recalls = [cp.get('recall', 0.0) for cp in checkpoints]
        versions = [cp.get('constitution_version', 0) for cp in checkpoints]
        fn_counts = [cp.get('fn_count', 0) for cp in checkpoints]
        fp_counts = [cp.get('fp_count', 0) for cp in checkpoints]

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: F1 Score Progression (Main Plot)
        ax1 = axes[0, 0]
        ax1.plot(batch_indices, f1_scores, 'o-', color='#2E86AB', linewidth=2.5,
                 markersize=8, label='Validation F1')

        # Add stepped line to show when F1 actually changes
        ax1.step(batch_indices, f1_scores, where='post', color='#2E86AB',
                 linewidth=1, alpha=0.3, linestyle='--')

        # Mark version changes with annotations
        prev_version = -1
        for i, (batch, f1, version) in enumerate(zip(batch_indices, f1_scores, versions)):
            if version != prev_version:
                ax1.axvline(x=batch, color='green', linestyle=':', alpha=0.5)
                ax1.annotate(f'v{version}', (batch, f1), textcoords="offset points",
                            xytext=(5, 10), ha='left', fontsize=9, fontweight='bold',
                            color='darkgreen',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
                prev_version = version

        # Add epoch boundary markers
        num_batches = self.num_batches
        max_batch = max(batch_indices) if batch_indices else 60
        for epoch in range(1, (max_batch // num_batches) + 2):
            epoch_batch = epoch * num_batches
            if epoch_batch <= max_batch:
                ax1.axvline(x=epoch_batch, color='gray', linestyle='-', alpha=0.3, linewidth=2)
                ax1.text(epoch_batch, ax1.get_ylim()[0] + 0.02, f'E{epoch}',
                        ha='center', fontsize=10, fontweight='bold', alpha=0.6)

        ax1.set_xlabel('Global Batch Index', fontsize=11)
        ax1.set_ylabel('Validation F1 Score', fontsize=11)
        ax1.set_title('Validation F1 at Checkpoints (Stepped Progression)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 1)

        # Add improvement annotation
        if len(f1_scores) >= 2:
            improvement = f1_scores[-1] - f1_scores[0]
            ax1.text(0.02, 0.98, f'Improvement: {f1_scores[0]:.3f} → {f1_scores[-1]:.3f} ({improvement:+.3f})',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Plot 2: Precision & Recall
        ax2 = axes[0, 1]
        ax2.plot(batch_indices, precisions, 's-', color='#28A745', linewidth=2,
                 markersize=6, label='Precision', alpha=0.8)
        ax2.plot(batch_indices, recalls, '^-', color='#DC3545', linewidth=2,
                 markersize=6, label='Recall', alpha=0.8)

        ax2.set_xlabel('Global Batch Index', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Precision & Recall at Checkpoints', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 1)

        # Plot 3: FN/FP Counts
        ax3 = axes[1, 0]
        ax3.plot(batch_indices, fn_counts, 'o-', color='#DC3545', linewidth=2,
                 markersize=6, label='False Negatives')
        ax3.plot(batch_indices, fp_counts, 's-', color='#FFC107', linewidth=2,
                 markersize=6, label='False Positives')

        ax3.set_xlabel('Global Batch Index', fontsize=11)
        ax3.set_ylabel('Error Count', fontsize=11)
        ax3.set_title('FN/FP Counts at Checkpoints', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        # Plot 4: Constitution Version Timeline
        ax4 = axes[1, 1]
        ax4.step(batch_indices, versions, where='post', color='#6F42C1', linewidth=2.5)
        ax4.scatter(batch_indices, versions, color='#6F42C1', s=50, zorder=5)

        # Highlight version changes
        prev_v = -1
        for batch, version in zip(batch_indices, versions):
            if version != prev_v:
                ax4.annotate(f'v{version}', (batch, version), textcoords="offset points",
                            xytext=(5, 5), ha='left', fontsize=9)
                prev_v = version

        ax4.set_xlabel('Global Batch Index', fontsize=11)
        ax4.set_ylabel('Constitution Version', fontsize=11)
        ax4.set_title('Constitution Version at Checkpoints', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Summary stats
        total_checkpoints = len(checkpoints)
        version_changes = len(set(versions))
        final_f1 = f1_scores[-1] if f1_scores else 0

        plt.suptitle(f'VALIDATION CHECKPOINT ANALYSIS\n'
                    f'({total_checkpoints} checkpoints, {version_changes} versions, Final F1={final_f1:.3f})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'validation_checkpoints.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Created validation checkpoint plot with {total_checkpoints} data points")


def run_epoch_batch_constitutional_classifier(config_path: Path) -> TrainingResult:
    """
    Run epoch-batch constitutional classifier.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TrainingResult with training outcomes
    """
    pipeline = EpochBatchConstitutionalPipeline(config_path)
    return pipeline.run()
