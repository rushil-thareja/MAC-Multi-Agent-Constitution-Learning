"""
DSPy-style public API for MAC.

Users define Examples, a metric, and optionally an error analyzer.
MAC.compile() runs the epoch-batch pipeline and returns a trained constitution.

Usage:
    import mac

    train   = [mac.Example(input="John Smith went to...", output=["John Smith"])]
    holdout = [mac.Example(input="Dr. Brown called...",   output=["Dr. Brown"])]

    def metric(prediction, gold):
        pred_set = set(p.lower() for p in prediction)
        gold_set = set(g.lower() for g in gold)
        tp = len(pred_set & gold_set)
        return tp / len(gold_set) if gold_set else 1.0

    result = mac.MAC("configs/run.yaml").compile(train, holdout, metric=metric)
"""

import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Core data classes
# ============================================================================

@dataclass
class Example:
    """A single training/validation example.

    Mirrors dspy.Example: the user provides an input string and an
    expected output of any type (str, List[str], Dict, etc.).
    """
    input: str
    output: Any
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class ErrorItem:
    """A single error detected by the error analyzer."""
    error_type: str          # e.g. "MISSED", "WRONG", "SPURIOUS"
    description: str         # human-readable description
    context: str = ""        # surrounding text or extra info


@dataclass
class ErrorReport:
    """Aggregated error report for one example."""
    example_id: str
    score: float
    prediction: Any
    gold: Any
    errors: List[ErrorItem] = field(default_factory=list)
    input_text: str = ""
    reasoning: str = ""


# ============================================================================
# Error analyzers
# ============================================================================

class DefaultErrorAnalyzer:
    """Built-in error analyzer that handles common output types.

    - List[str]: set-difference (MISSED / SPURIOUS)
    - str: exact-match (WRONG)
    - Dict: per-field comparison (WRONG per key)
    """

    def analyze(self, prediction: Any, gold: Any, input_text: str = "",
                example_id: str = "") -> ErrorReport:
        score = 0.0
        errors: List[ErrorItem] = []

        try:
            if isinstance(gold, list):
                score, errors = self._analyze_list(prediction, gold, input_text)
            elif isinstance(gold, dict):
                score, errors = self._analyze_dict(prediction, gold, input_text)
            elif isinstance(gold, str):
                score, errors = self._analyze_str(prediction, gold, input_text)
            else:
                # Fallback: equality check
                score = 1.0 if prediction == gold else 0.0
                if score == 0.0:
                    errors.append(ErrorItem(
                        error_type="WRONG",
                        description=f"Predicted '{prediction}' but gold is '{gold}'",
                        context=input_text[:200] if input_text else ""
                    ))
        except Exception as e:
            logger.warning(f"Error analyzer failed for example {example_id}: {e}")
            errors.append(ErrorItem(
                error_type="ANALYZER_ERROR",
                description=str(e)
            ))

        return ErrorReport(
            example_id=example_id,
            score=score,
            prediction=prediction,
            gold=gold,
            errors=errors,
            input_text=input_text
        )

    # -- private helpers --

    def _analyze_list(self, prediction: Any, gold: list, input_text: str):
        if not isinstance(prediction, list):
            prediction = [prediction] if prediction else []

        pred_set = set(str(p).lower().strip() for p in prediction if p)
        gold_set = set(str(g).lower().strip() for g in gold if g)

        missed = gold_set - pred_set
        spurious = pred_set - gold_set
        tp = len(gold_set & pred_set)

        score = tp / len(gold_set) if gold_set else 1.0

        errors = []
        for m in missed:
            ctx = ""
            if input_text:
                idx = input_text.lower().find(m.lower())
                if idx >= 0:
                    start = max(0, idx - 40)
                    end = min(len(input_text), idx + len(m) + 40)
                    ctx = f"...{input_text[start:end]}..."
            errors.append(ErrorItem("MISSED", f"Gold '{m}' not in prediction", ctx))
        for s in spurious:
            errors.append(ErrorItem("SPURIOUS", f"Predicted '{s}' has no match in gold"))

        return score, errors

    def _analyze_dict(self, prediction: Any, gold: dict, input_text: str):
        if not isinstance(prediction, dict):
            prediction = {}

        correct = 0
        errors = []
        for key, gold_val in gold.items():
            pred_val = prediction.get(key)
            if str(pred_val).strip().lower() == str(gold_val).strip().lower():
                correct += 1
            else:
                errors.append(ErrorItem(
                    "WRONG",
                    f"Key '{key}': predicted '{pred_val}' but gold is '{gold_val}'",
                    input_text[:200] if input_text else ""
                ))

        score = correct / len(gold) if gold else 1.0
        return score, errors

    def _analyze_str(self, prediction: Any, gold: str, input_text: str):
        pred_s = str(prediction).strip()
        gold_s = gold.strip()
        if pred_s.lower() == gold_s.lower():
            return 1.0, []
        return 0.0, [ErrorItem(
            "WRONG",
            f"Predicted '{pred_s}' but gold is '{gold_s}'",
            input_text[:200] if input_text else ""
        )]


# ============================================================================
# Default config builder
# ============================================================================

def _build_config(
    model: str = "gpt-4o",
    mac_model: Optional[str] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 1,
    max_tokens: int = 2048,
    num_epochs: int = 1,
    batch_size: int = 4,
    output_key: str = "answer",
    task_description: str = "",
    task_prompt: str = "",
    rule_type: str = "",
    prompts_dir: str = "",
    # Three-tier model configuration
    mac_provider: Optional[str] = None,
    mac_base_url: Optional[str] = None,
    mac_temperature: Optional[float] = None,
    adapt_model: Optional[str] = None,
    adapt_provider: Optional[str] = None,
    adapt_base_url: Optional[str] = None,
    adapt_temperature: Optional[float] = None,
    **overrides,
) -> Dict:
    """Build a full config dict from Python kwargs — no YAML needed."""
    import os

    # Resolve prompts_dir: default to package-bundled prompts
    if not prompts_dir:
        prompts_dir = str(Path(__file__).resolve().parent / "prompts")

    # Resolve API key: explicit > env var
    resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "not-needed"

    model_block: Dict[str, Any] = {
        'provider': provider,
        'model_name': model,
        'temperature': temperature,
        'max_completion_tokens': max_tokens,
        'timeout': 120,
        'max_retries': 3,
    }
    if base_url:
        model_block['base_url'] = base_url

    cfg: Dict[str, Any] = {
        'model': model_block,
        'annotator': {'output_key': output_key},
        'task_prompt': task_prompt,
        'meta_model': {
            'enabled': bool(task_description),
            'provider': provider,
            'model_name': model,
            'temperature': 1,
            'max_completion_tokens': 8000,
            'timeout': 120,
            'max_retries': 2,
            'adapt_prompts': {
                'annotator': not bool(task_prompt),
                'decision_agent': True,
                'rule_proposer': True,
                'rule_editor': True,
            },
            'task': {
                'rule_type': rule_type,
                'description': task_description,
                'metric': 'exact match',
                'output_key': output_key,
            },
        },
        'data': {
            'active_dataset': 'none',
        },
        'algorithm': {
            'training_mode': 'epoch_batch_constitutional',
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'num_batches': 8,
            'training_docs': 32,
            'holdout_docs': 8,
            'holdout_batches': 1,
            'max_rules_per_epoch': 10,
            'max_total_rules': 20,
            'use_initial_constitution': False,
            'enable_rule_editing': True,
            'enable_rule_removal': True,
            'min_rule_age_for_edit': 1,
            'max_consecutive_removes': 1,
            'maintain_pool': False,
            'decision_strategy': 'adaptive',
            'balance_guidance': {
                'fn_weight': 1.0,
                'fp_weight': 0.7,
                'utility_focus': True,
            },
            'held_out_validation': {'enabled': False},
            'validation_checkpoints': {'enabled': False},
            'update_retries': {'enabled': True, 'max_attempts': 3, 'temperature': [0.7, 0.9, 1.2]},
        },
        'random_seed': 42,
        'output': {
            'runs_dir': 'runs',
            'log_level': 'INFO',
            'save_predictions': True,
            'save_deltas': True,
            'save_metrics': True,
        },
        'validation': {'max_rule_length': 500, 'max_validation_attempts': 3},
        'evaluation': {'strategy': 'direct-lower'},
        'tokenization': {'method': 'simple', 'preserve_spans': True, 'lowercase': False},
        'prompts': {'templates_dir': prompts_dir, 'max_phrase_examples': 5, 'context_chars': 500},
        'error_limits': {'max_fn_phrases': 10, 'max_fp_phrases': 10, 'phrase_selection': 'first'},
        'error_context': {
            'show_full_examples': True,
            'max_error_examples': 8,
            'max_input_chars': 1500,
            'max_reasoning_chars': 1000,
        },
    }

    # Three-tier model configuration:
    #   Tier 1 (Worker/Annotator): model, provider, base_url, temperature
    #   Tier 2 (MAC agents): mac_model, mac_provider, mac_base_url, mac_temperature
    #   Tier 3 (Prompt adaptation): adapt_model, adapt_provider, adapt_base_url, adapt_temperature
    #
    # Fallback cascade:
    #   adapt_model     → mac_model     → model
    #   adapt_provider  → mac_provider  → provider
    #   adapt_base_url  → mac_base_url  → (NO fallback to base_url)
    #   adapt_temperature → mac_temperature → 1
    #   mac_base_url does NOT fall back to base_url (VLLM worker ≠ MAC API)
    if mac_model:
        resolved_mac_provider = mac_provider or provider
        resolved_mac_temperature = mac_temperature if mac_temperature is not None else temperature
        mac_model_block: Dict[str, Any] = {
            'provider': resolved_mac_provider,
            'model_name': mac_model,
            'temperature': resolved_mac_temperature,
            'max_completion_tokens': max_tokens,
            'timeout': 120,
            'max_retries': 3,
        }
        # mac_base_url does NOT fall back to base_url — intentional
        if mac_base_url:
            mac_model_block['base_url'] = mac_base_url
        cfg['mac_model'] = mac_model_block

        # Tier 3: Prompt adaptation (meta_model) — falls back through mac, then worker
        resolved_adapt_model = adapt_model or mac_model
        resolved_adapt_provider = adapt_provider or mac_provider or provider
        # If adapt_model is explicitly set (different tier), don't inherit mac_base_url —
        # adapt may target a different API (e.g. OpenAI) while MAC agents use VLLM.
        # Only inherit mac_base_url when adapt falls back to mac_model (same server).
        resolved_adapt_base_url = adapt_base_url if adapt_base_url is not None else (
            None if adapt_model else mac_base_url
        )
        resolved_adapt_temp = adapt_temperature if adapt_temperature is not None else (
            mac_temperature if mac_temperature is not None else 1
        )
        cfg['meta_model']['model_name'] = resolved_adapt_model
        cfg['meta_model']['provider'] = resolved_adapt_provider
        cfg['meta_model']['temperature'] = resolved_adapt_temp
        if resolved_adapt_base_url:
            cfg['meta_model']['base_url'] = resolved_adapt_base_url
    elif adapt_model:
        # adapt_model without mac_model: adaptation uses adapt_model, agents use worker
        resolved_adapt_provider = adapt_provider or provider
        resolved_adapt_temp = adapt_temperature if adapt_temperature is not None else 1
        cfg['meta_model']['model_name'] = adapt_model
        cfg['meta_model']['provider'] = resolved_adapt_provider
        cfg['meta_model']['temperature'] = resolved_adapt_temp
        resolved_adapt_base_url = adapt_base_url
        if resolved_adapt_base_url:
            cfg['meta_model']['base_url'] = resolved_adapt_base_url

    # Inject api key into env so OpenAI client picks it up
    if resolved_key != "not-needed":
        os.environ.setdefault("OPENAI_API_KEY", resolved_key)

    # Apply flat overrides
    for k, v in overrides.items():
        if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v

    return cfg


# ============================================================================
# MAC entry point
# ============================================================================

class MAC:
    """DSPy-style optimizer for MAC constitution learning.

    Everything defined in Python — no config files.

    Usage:
        compiler = mac.MAC(
            model="gpt-4o",
            num_epochs=2,
            batch_size=4,
            task_description="Solve competition math problems",
            rule_type="math reasoning rules",
        )
        result = compiler.compile(trainset=train, holdout=holdout, metric=my_metric)
    """

    def __init__(self, model: str = "gpt-4o", mac_model: Optional[str] = None,
                 adapt_model: Optional[str] = None, **kwargs):
        """
        Args:
            model: Model name for the annotator/worker (e.g. "gpt-4o-mini").
            mac_model: Optional stronger model for MAC agents (decision,
                       proposer, editor). When None, all agents use `model`.
            adapt_model: Optional model for prompt adaptation. Defaults to
                         mac_model, then model (fallback cascade).
            **kwargs: Any of: provider, api_key, base_url, temperature,
                      max_tokens, num_epochs, batch_size, output_key,
                      task_description, rule_type, prompts_dir,
                      mac_provider, mac_base_url, mac_temperature,
                      adapt_provider, adapt_base_url, adapt_temperature.
        """
        self.model = model
        self.mac_model = mac_model
        self.adapt_model = adapt_model
        self.error_context_formatter = kwargs.pop('error_context_formatter', None)
        self.kwargs = kwargs

    def compile(
        self,
        trainset: List[Example],
        holdout: List[Example],
        metric: Callable[[Any, Any], float],
        error_analyzer: Optional[Any] = None,
    ):
        """Run MAC training on user-provided examples.

        Args:
            trainset: Training examples.
            holdout: Holdout examples for evaluation.
            metric: Callable(prediction, gold) -> float in [0, 1].
            error_analyzer: Optional custom error analyzer.

        Returns:
            TrainingResult from the epoch-batch pipeline.
        """
        from .epoch_pipeline import EpochBatchConstitutionalPipeline
        from .display import MACDisplay
        import time

        if error_analyzer is None:
            error_analyzer = DefaultErrorAnalyzer()

        config_dict = _build_config(model=self.model, mac_model=self.mac_model,
                                     adapt_model=self.adapt_model, **self.kwargs)

        # ── Inject data samples into task config for prompt adaptation ──
        if config_dict.get('meta_model', {}).get('enabled'):
            task_cfg = config_dict['meta_model']['task']
            # Sample train examples so the adapter sees real data
            samples = trainset[:min(3, len(trainset))]
            task_cfg['data_samples'] = [
                {'input': ex.input, 'output': repr(ex.output)}
                for ex in samples
            ]
            # Detect output shape: scalar vs list
            first_out = trainset[0].output if trainset else ""
            if isinstance(first_out, list):
                task_cfg['output_format'] = 'list'
                task_cfg['output_example'] = repr(first_out)
            elif isinstance(first_out, dict):
                task_cfg['output_format'] = 'dict'
                task_cfg['output_example'] = repr(first_out)
            else:
                task_cfg['output_format'] = 'scalar'
                task_cfg['output_example'] = repr(first_out)
            # Include metric source if possible
            try:
                import inspect
                task_cfg['metric_source'] = inspect.getsource(metric)
            except Exception:
                task_cfg['metric_source'] = ''

        # ── Display setup ─────────────────────────────────────
        n_train = len(trainset)
        n_holdout = len(holdout)
        epochs = config_dict['algorithm']['num_epochs']
        bs = config_dict['algorithm']['batch_size']
        n_batches = max(1, n_train // bs)
        task = self.kwargs.get('task_description', '')
        task_prompt = self.kwargs.get('task_prompt', '')
        mode = "Custom prompt" if task_prompt else "Auto-adapt"

        # Resolve adapt_model for display — show what will actually be used
        display_adapt = self.adapt_model or (self.mac_model if self.mac_model else None)
        # Only show adapt tier if it differs from mac_model
        if display_adapt and display_adapt == self.mac_model:
            display_adapt = None

        display = MACDisplay()
        display.banner(
            model=self.model, task=task or task_prompt,
            n_train=n_train, n_holdout=n_holdout,
            epochs=epochs, batches=n_batches, batch_size=bs,
            mode=mode, mac_model=self.mac_model,
            adapt_model=display_adapt,
        )

        # ── Run pipeline ──────────────────────────────────────
        start = time.time()

        try:
            pipeline = EpochBatchConstitutionalPipeline.init_from_examples(
                trainset=trainset,
                holdout=holdout,
                metric=metric,
                config_dict=config_dict,
                error_analyzer=error_analyzer,
                display=display,
                error_context_formatter=self.error_context_formatter,
            )
            result = pipeline.run()
        except ValueError as e:
            self._show_error(str(e), hint=self._config_hint(str(e)))
            raise SystemExit(1)
        except OSError as e:
            self._show_error(str(e), hint="Check that the model endpoint / base_url is reachable.")
            raise SystemExit(1)
        except Exception as e:
            self._show_error(str(e))
            raise SystemExit(1)

        elapsed = time.time() - start

        # ── Final summary ─────────────────────────────────────
        display.final_summary(result, elapsed, pipeline.constitution)

        # ── Build callable CompiledMAC ────────────────────────
        from .compiled import CompiledMAC

        annotator = pipeline.agents['annotator']
        constitution_text = pipeline.constitution.get_text()
        rules_list = list(pipeline.constitution.rules)
        baseline = display.state.scores[0] if display.state.scores else 0.0

        # Build training metadata for saved JSON
        training_metadata = {
            'worker_model': self.model,
            'mac_model': self.mac_model,
            'adapt_model': self.adapt_model or self.mac_model,
            'metric_name': getattr(metric, '__name__', str(metric)),
            'task_prompt': self.kwargs.get('task_prompt', ''),
            'task_description': self.kwargs.get('task_description', ''),
            'rule_type': self.kwargs.get('rule_type', ''),
            'num_epochs': config_dict['algorithm']['num_epochs'],
            'batch_size': config_dict['algorithm']['batch_size'],
            'holdout_score': result.best_constitution_info.get('f1_score', 0.0),
            'best_constitution_version': result.best_constitution_info.get('version', 0),
            'score_progression': [v.get('f1', 0.0) for v in result.holdout_results],
        }

        return CompiledMAC(
            annotator=annotator,
            constitution=constitution_text,
            rules=rules_list,
            result=result,
            baseline_score=baseline,
            model_config=dict(annotator.model_config),
            prompt_template_system=annotator.prompt_template.system,
            prompt_template_user=annotator.prompt_template.user,
            output_key=annotator.output_key,
            elapsed=elapsed,
            training_metadata=training_metadata,
        )

    # ── Error helpers ─────────────────────────────────────────

    @staticmethod
    def _config_hint(msg: str) -> str:
        m = msg.lower()
        if "openai_api_key" in m or "api_key" in m:
            return (
                "Set your API key in the environment:\n\n"
                "  export OPENAI_API_KEY='sk-...'\n\n"
                "Or pass it directly:\n\n"
                "  MAC(model='gpt-4o', api_key='sk-...')"
            )
        if "base_url" in m or "endpoint" in m:
            return "Check that base_url points to a running server, e.g. 'http://localhost:8001/v1'"
        if "model" in m:
            return "Check the model name — e.g. 'gpt-4o', 'gpt-4o-mini', or a vLLM model path."
        return ""

    @staticmethod
    def _show_error(message: str, hint: str = "") -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        console = Console(stderr=True)
        body = Text(message, style="red")
        if hint:
            body.append("\n\n")
            body.append(hint, style="yellow")
        console.print(Panel(body, title="[bold red]MAC Error[/bold red]", border_style="red"))
