"""
CompiledMAC — callable inference object returned by MAC.compile().

Wraps the learned constitution + annotator agent so users can do:
    optimized = compiler.compile(trainset, holdout, metric=m)
    answer = optimized("What is 2+2?")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class CompiledMAC:
    """Callable wrapper around a trained MAC constitution.

    Returned by ``MAC.compile()``. Supports direct inference, persistence,
    and backward-compatible access to ``TrainingResult`` fields.
    """

    def __init__(
        self,
        annotator,
        constitution: str,
        rules: List[str],
        result,
        baseline_score: float = 0.0,
        model_config: Optional[Dict] = None,
        prompt_template_system: str = "",
        prompt_template_user: str = "",
        output_key: str = "answer",
        elapsed: float = 0.0,
        training_metadata: Optional[Dict] = None,
    ):
        self._annotator = annotator
        self.constitution = constitution
        self.rules = list(rules)
        self.result = result
        self.baseline_score = baseline_score
        self._model_config = model_config or {}
        self._prompt_template_system = prompt_template_system
        self._prompt_template_user = prompt_template_user
        self._output_key = output_key
        self._elapsed = elapsed
        self._training_metadata = training_metadata or {}

    # ── Inference ──────────────────────────────────────────────

    def __call__(self, text: str) -> str:
        """Run inference on a single input using the learned constitution.

        Returns the extracted answer as a string.
        """
        raw = self._annotator.process(text, constitution=self.constitution)
        return self._unwrap(raw)

    def predict_batch(self, texts: List[str], max_concurrent: int = 8) -> List[str]:
        """Run inference on multiple inputs concurrently."""
        results = self._annotator.process_batch_concurrent(
            texts, constitution=self.constitution, max_concurrent=max_concurrent,
        )
        return [self._unwrap(r) for r in results]

    @staticmethod
    def _unwrap(raw) -> str:
        """Unwrap single-element lists to a plain string."""
        if isinstance(raw, list):
            if len(raw) == 1:
                return str(raw[0])
            return str(raw)
        return str(raw)

    # ── Display ────────────────────────────────────────────────

    def overview(self):
        """Print a Rich panel summarising compilation results."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        final = self.result.holdout_metrics.get("f1", 0.0)
        delta = final - self.baseline_score
        pct = (delta / self.baseline_score * 100) if self.baseline_score else 0.0
        sign = "+" if delta >= 0 else ""

        t = Text()
        t.append(f"  Baseline score :  {self.baseline_score:.3f}\n", style="dim green")
        t.append(f"  Final score    :  {final:.3f}", style="dim green")
        if delta:
            t.append(f"   ({sign}{delta:.3f}, {sign}{pct:.0f}%)", style="bold green" if delta > 0 else "red")
        t.append("\n")
        t.append(f"  Rules          :  {len(self.rules)}\n", style="dim green")
        t.append(f"  Time           :  {self._elapsed:.1f}s\n", style="dim green")
        t.append("\n")

        if self.rules:
            t.append("  Constitution:\n", style="green")
            for i, rule in enumerate(self.rules, 1):
                prefix = "└──" if i == len(self.rules) else "├──"
                t.append(f"  {prefix} {i}. {rule}\n", style="dim white")

        Console().print(Panel(
            t,
            title="[bold green]MAC Compilation Result[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

    def __repr__(self) -> str:
        final = self.result.holdout_metrics.get("f1", 0.0) if self.result else 0.0
        return f"<CompiledMAC rules={len(self.rules)} score={final:.3f}>"

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save to JSON (no API keys stored)."""
        data = {
            "constitution": self.constitution,
            "rules": self.rules,
            "baseline_score": self.baseline_score,
            "output_key": self._output_key,
            "model_config": {
                k: v for k, v in self._model_config.items()
                if k not in ("api_key",)
            },
            "prompt_template": {
                "system": self._prompt_template_system,
                "user": self._prompt_template_user,
            },
            "holdout_metrics": self.result.holdout_metrics if self.result else {},
            "elapsed": self._elapsed,
            "training_metadata": self._training_metadata,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "CompiledMAC":
        """Recreate a CompiledMAC from a saved JSON file.

        The annotator agent is reconstructed from the saved model config
        and prompt template. Requires the same provider/API key in the
        environment as was used during training.
        """
        from .agents import AnnotatorAgent, PromptTemplate

        data = json.loads(Path(path).read_text())
        model_config = data["model_config"]
        tmpl = PromptTemplate(
            system=data["prompt_template"]["system"],
            user=data["prompt_template"]["user"],
        )
        output_key = data.get("output_key", "answer")
        annotator = AnnotatorAgent(model_config, tmpl, output_key=output_key)

        # Build a lightweight result stand-in for .holdout_metrics access
        # Backward compatible: try holdout_metrics first, fall back to test_metrics
        result = _MinimalResult(data.get("holdout_metrics", data.get("test_metrics", {})))

        return cls(
            annotator=annotator,
            constitution=data["constitution"],
            rules=data["rules"],
            result=result,
            baseline_score=data.get("baseline_score", 0.0),
            model_config=model_config,
            prompt_template_system=data["prompt_template"]["system"],
            prompt_template_user=data["prompt_template"]["user"],
            output_key=output_key,
            elapsed=data.get("elapsed", 0.0),
            training_metadata=data.get("training_metadata", {}),
        )

    # ── Backward compatibility ─────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute lookups to self.result (TrainingResult)."""
        if name.startswith("_"):
            raise AttributeError(name)
        result = self.__dict__.get("result")
        if result is not None and hasattr(result, name):
            return getattr(result, name)
        raise AttributeError(f"'CompiledMAC' has no attribute '{name}'")


class _MinimalResult:
    """Lightweight stand-in for TrainingResult used after load()."""

    def __init__(self, holdout_metrics: Dict):
        self.holdout_metrics = holdout_metrics
