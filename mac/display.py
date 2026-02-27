"""
Rich CLI display for MAC training.

Green-themed terminal output with mountain logo, progress tables,
rule trees, and score tracking across all training stages.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.tree import Tree

# ── Theme ───────────────────────────────────────────────────────
THEME = {
    "primary": "green",
    "accent": "bold green",
    "bright": "bold bright_green",
    "dim": "dim green",
    "success": "bold green",
    "error": "bold red",
    "warning": "yellow",
    "muted": "dim white",
    "border": "green",
}


def _load_mountain_art() -> str:
    """Load art.txt from the project root. Returns empty string if not found."""
    here = Path(__file__).resolve().parent  # mac/
    for candidate in [here.parent / "art.txt", here / "art.txt", Path("art.txt")]:
        if candidate.exists():
            raw = candidate.read_text()
            return raw.replace("-", " ").replace(".", " ")
    return ""


_MAC_FONT = [
    "███   ███   █████    █████",
    "████ ████  ██   ██  ██    ",
    "██ ███ ██  ███████  ██    ",
    "██  █  ██  ██   ██  ██    ",
    "██     ██  ██   ██   █████",
]


def _render_with_shadow(lines, bright_style, shadow_style, indent=2, offset=1):
    """Render block-text lines with a drop shadow. Returns a Rich Text."""
    max_w = max(len(l) for l in lines) + indent + offset + 1
    h = len(lines) + offset
    t = Text()
    for y in range(h):
        for x in range(max_w):
            mx, my = x - indent, y
            sx, sy = x - indent - offset, y - offset
            has_main = (0 <= my < len(lines) and 0 <= mx < len(lines[my])
                        and lines[my][mx] != " ")
            has_shadow = (0 <= sy < len(lines) and 0 <= sx < len(lines[sy])
                          and lines[sy][sx] != " ")
            if has_main:
                t.append("█", style=bright_style)
            elif has_shadow:
                t.append("░", style=shadow_style)
            else:
                t.append(" ")
        t.append("\n")
    return t


def _build_mac_banner(model, task, n_train, n_holdout, epochs, batches, batch_size,
                      mode="", mac_model=None, adapt_model=None):
    """Build MAC block-text banner with drop shadow and config info."""
    t = _render_with_shadow(_MAC_FONT, THEME["bright"], "dim green")

    t.append("  Multi-Agent Constitution Learning\n", style=THEME["dim"])
    t.append("\n")

    # Config info
    total = epochs * batches
    if mac_model and adapt_model:
        # Three-tier display
        t.append(f"  Worker   : ", style=THEME["dim"])
        t.append(f"{model}\n", style=THEME["muted"])
        t.append(f"  MAC      : ", style=THEME["dim"])
        t.append(f"{mac_model}\n", style=THEME["muted"])
        t.append(f"  Adapt    : ", style=THEME["dim"])
        t.append(f"{adapt_model}\n", style=THEME["muted"])
    elif mac_model:
        # Two-tier display (current behavior)
        t.append(f"  Worker   : ", style=THEME["dim"])
        t.append(f"{model}\n", style=THEME["muted"])
        t.append(f"  MAC      : ", style=THEME["dim"])
        t.append(f"{mac_model}\n", style=THEME["muted"])
    else:
        t.append(f"  Model    : ", style=THEME["dim"])
        t.append(f"{model}\n", style=THEME["muted"])
    if mode:
        t.append(f"  Mode     : ", style=THEME["dim"])
        t.append(f"{mode}\n", style=THEME["muted"])
    if task:
        short_task = task if len(task) <= 55 else task[:52] + "..."
        t.append(f"  Task     : ", style=THEME["dim"])
        t.append(f"{short_task}\n", style=THEME["muted"])
    t.append(f"  Data     : ", style=THEME["dim"])
    t.append(f"{n_train} train / {n_holdout} holdout\n", style=THEME["muted"])
    t.append(f"  Schedule : ", style=THEME["dim"])
    t.append(f"{epochs} epoch{'s' if epochs > 1 else ''} × {batches} batches × {batch_size} examples", style=THEME["muted"])
    t.append(f"  ({total} iters)", style=THEME["dim"])

    return t


# ── State ───────────────────────────────────────────────────────
@dataclass
class DisplayState:
    scores: List[float] = field(default_factory=list)
    epoch_scores: Dict[int, List[float]] = field(default_factory=dict)
    best_score: float = 0.0
    best_version: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_no_change: int = 0
    start_time: float = 0.0


# ── Display ─────────────────────────────────────────────────────
class MACDisplay:
    """Rich-based CLI display for MAC training lifecycle."""

    def __init__(self, verbose: bool = True):
        self.console = Console()
        self.state = DisplayState()
        self.verbose = verbose

    # ── Prompt Adaptation Progress ─────────────────────────────

    def adaptation_start(self, task_desc: str, n_prompts: int):
        """Begin the adaptation phase."""
        self._adapt_n = n_prompts
        self._adapt_start = time.time()
        self.console.print()
        short = task_desc if len(task_desc) <= 60 else task_desc[:57] + "..."
        self.console.print(Panel(
            f"  [bold bright_green]Adapting {n_prompts} agent prompts[/bold bright_green]\n"
            f"  [dim white]Minor edits to conform to:[/dim white]  [green]{short}[/green]",
            border_style=THEME["border"],
            padding=(0, 1),
        ))
        self.console.print()

    def adaptation_agent_start(self, key: str, role: str, idx: int, total: int):
        """Show which agent is being adapted right now."""
        self.console.print(
            f"  [dim white]{idx}/{total}[/dim white]  [dim green]⟳[/dim green]  "
            f"[green]{key}[/green]  [dim white]adapting...[/dim white]"
        )

    def adaptation_agent_done(self, key: str, role: str, success: bool,
                               preview: str, idx: int, total: int,
                               errors: Optional[List[str]] = None):
        """Called after each agent prompt is adapted — show result card."""
        elapsed = time.time() - self._adapt_start

        if success:
            mark = "[bold green]✓[/bold green]"
            status_text = "[green]adapted[/green]"
        else:
            mark = "[bold red]✗[/bold red]"
            status_text = "[red]fallback (using original)[/red]"

        # Preview snippet — strip leading "System:" if already present
        snippet = preview.replace('\n', ' ').strip()
        if snippet.startswith("System:"):
            snippet = snippet[len("System:"):].strip()
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."

        self.console.print(
            f"  [dim white]{idx}/{total}[/dim white]  {mark}  "
            f"[bold green]{key}[/bold green]  {status_text}"
            f"  [dim white]({elapsed:.0f}s)[/dim white]"
        )
        self.console.print(
            f"       [dim]{snippet}[/dim]"
        )

        # Show failure reason when adaptation fails
        if not success and errors:
            last_error = errors[-1] if isinstance(errors[-1], str) else str(errors[-1])
            # Clean up the error string for display
            reason = last_error.replace('\n', ' ').strip()
            if len(reason) > 120:
                reason = reason[:117] + "..."
            self.console.print(
                f"       [bold red]Reason:[/bold red] [yellow]{reason}[/yellow]"
            )

    def adaptation_complete(self, succeeded: int, total: int):
        """Show adaptation summary."""
        elapsed = time.time() - self._adapt_start
        self.console.print()
        if succeeded == total:
            style = "bold bright_green"
            msg = f"{succeeded}/{total} adapted"
        elif succeeded > 0:
            style = "yellow"
            msg = (f"{succeeded}/{total} adapted"
                   f"  [dim white]({total - succeeded} using original prompts)[/dim white]")
        else:
            style = "bold red"
            msg = (f"0/{total} adapted — all prompts using originals"
                   f"\n  [yellow]Adapt model could not rewrite prompts for this task. "
                   f"Training continues with default prompts.[/yellow]")
        self.console.print(
            f"  [{style}]{msg}[/{style}]"
            f"  [dim white]in {elapsed:.1f}s[/dim white]"
        )
        self.console.print()

    # ── STAGE 0: Banner ─────────────────────────────────────────

    def banner(self, model: str, task: str, n_train: int, n_holdout: int,
               epochs: int, batches: int, batch_size: int,
               mode: str = "", mac_model: Optional[str] = None,
               adapt_model: Optional[str] = None):
        self.state.start_time = time.time()

        # ── Phase 1: Mountain art reveal ──────────────────────
        mountain_art = _load_mountain_art()
        if mountain_art:
            self.console.clear()
            self.console.print()
            mountain_text = Text(mountain_art, style=THEME["primary"])
            self.console.print(Panel(
                mountain_text, border_style=THEME["border"],
                padding=(0, 1),
                title="[bold bright_green]MAC: Multi-Agent Constitution Learning[/bold bright_green]",
            ))
            time.sleep(2.0)

        # ── Phase 2: Chunky loading bar ───────────────────────
        n_seg = 15
        self.console.print()
        with Live(console=self.console, transient=True, refresh_per_second=20) as live:
            for step in range(n_seg + 1):
                filled = "▰▰▰ " * step
                empty = "▱▱▱ " * (n_seg - step)
                pct = int(step / n_seg * 100)
                bar = Text()
                bar.append("  Initializing  ", style=THEME["dim"])
                bar.append(filled, style=THEME["bright"])
                bar.append(empty, style="dim green")
                bar.append(f" {pct:>3}%", style=THEME["dim"])
                live.update(bar)
                time.sleep(0.08)

        # ── Phase 3: Clear screen, show MAC config banner ─────
        self.console.clear()

        content = _build_mac_banner(model, task, n_train, n_holdout,
                                    epochs, batches, batch_size,
                                    mode=mode, mac_model=mac_model,
                                    adapt_model=adapt_model)

        self.console.print()
        self.console.print(Panel(content, border_style=THEME["border"],
                                 padding=(1, 2)))

    # ── STAGE 1: Epoch Start ────────────────────────────────────

    def epoch_start(self, epoch: int, total_epochs: int, n_rules: int):
        self.state.epoch_scores[epoch] = []
        self.console.print()
        self.console.print(Rule(
            f"EPOCH {epoch} / {total_epochs}  |  Rules: {n_rules}",
            style=THEME["primary"],
        ))

    # ── STAGE 2: Batch Results ──────────────────────────────────

    def batch_scored(self, epoch: int, batch_idx: int, n_batches: int,
                     batch_result: Any, constitution: Any,
                     total_tokens: int = 0, total_calls: int = 0):
        score = batch_result.batch_score
        self.state.scores.append(score)
        self.state.epoch_scores.setdefault(epoch, []).append(score)
        if score > self.state.best_score:
            self.state.best_score = score

        table = self._results_table_examples(batch_result)

        footer = Text()
        footer.append(f"\n  Batch: {score:.3f}", style=THEME["accent"])
        footer.append(f"          Best so far: {self.state.best_score:.3f}", style=THEME["muted"])
        if total_tokens:
            # Per-tier token breakdown
            from .agents import get_token_tracker
            tiers = get_token_tracker().get_per_tier()
            worker_t = tiers['Worker']
            mac_t = tiers['MAC']
            adapt_t = tiers['Adapt']
            parts = []
            if worker_t['tokens']:
                parts.append(f"Worker {worker_t['tokens']:,}")
            if mac_t['tokens']:
                parts.append(f"MAC {mac_t['tokens']:,}")
            if adapt_t['tokens']:
                parts.append(f"Adapt {adapt_t['tokens']:,}")
            tier_str = " | ".join(parts) if parts else f"{total_tokens:,}"
            footer.append(f"\n  Tokens : {tier_str}  ({total_calls} calls)", style=THEME["dim"])

        self.console.print(Panel(
            Group(table, footer),
            title=f"[bold green]Batch {batch_idx + 1}/{n_batches}[/bold green]",
            border_style=THEME["border"],
            padding=(0, 1),
        ))

    # ── STAGE 3: Decision ───────────────────────────────────────

    def decision(self, decision_dict: Dict, batch_result: Any):
        action = decision_dict.get('action', 'no_change')
        reasoning = decision_dict.get('reasoning', '')

        if action == 'no_change':
            self.state.n_no_change += 1
            symbol, label, style = "─", "NO CHANGE", THEME["muted"]
        elif action == 'add':
            n_errors = len(batch_result.aggregated_fn_phrases) + len(batch_result.aggregated_fp_phrases)
            symbol, label, style = "+", f"ADD rule  ({n_errors} errors)", THEME["accent"]
        elif action == 'edit':
            idx = decision_dict.get('rule_index', '?')
            symbol, label, style = "~", f"EDIT rule #{idx}", THEME["warning"]
        elif action == 'remove':
            idx = decision_dict.get('rule_index', '?')
            symbol, label, style = "-", f"REMOVE rule #{idx}", THEME["error"]
        else:
            symbol, label, style = "?", action.upper(), THEME["muted"]

        t = Text()
        t.append(f"  {symbol} Decision: ", style=style)
        t.append(label, style=style)
        self.console.print(t)

        if reasoning and self.verbose:
            self.console.print(f"    \"{self._truncate(reasoning, 80)}\"",
                               style=THEME["muted"])

    # ── STAGE 4a: Rule Change ───────────────────────────────────

    def rule_change(self, action: str, old_version: int, new_version: int,
                    rule_text: str, rule_index: Optional[int] = None):
        prefix = {"add": "[+]", "edit": "[~]", "remove": "[-]"}.get(action, "[?]")
        verb = "added" if action == "add" else action + "ed"

        t = Text()
        t.append(f"\n  {prefix} Rule {verb} ", style=THEME["accent"])
        t.append(f"(v{old_version} → v{new_version}):", style=THEME["muted"])
        self.console.print(t)

        if rule_text:
            self.console.print(f"      \"{self._truncate(rule_text, 100)}\"",
                               style=THEME["muted"])

    # ── STAGE 4b: Rule Validated ────────────────────────────────

    def rule_validated(self, accepted: bool, old_f1: float, new_f1: float,
                       delta: float, retry_attempt: Optional[int] = None,
                       constitution: Any = None):
        if accepted:
            self.state.n_accepted += 1
            mark, label, style = "✓", "ACCEPTED", THEME["success"]
        else:
            self.state.n_rejected += 1
            mark, label, style = "✗", "REJECTED", THEME["error"]

        retry_tag = f"  (retry #{retry_attempt})" if retry_attempt else ""
        t = Text()
        t.append(f"\n  {mark} Validated: ", style=style)
        t.append(f"{old_f1:.3f} → {new_f1:.3f} ({delta:+.3f})", style=style)
        t.append(f"  {label}{retry_tag}", style=style)
        self.console.print(t)

        # Show current constitution state after accepted changes
        if accepted and constitution:
            rules = getattr(constitution, 'rules', [])
            if rules:
                self.console.print(self._rule_tree(rules, constitution.version))

    # ── Per-step token display ─────────────────────────────────

    def step_tokens(self, step_name: str, tokens: int, calls: int):
        """Print a one-line token summary after an algorithm step."""
        self.console.print(
            f"    [dim green]⤷[/dim green] [dim white]{tokens:,} tok ({calls} call{'s' if calls != 1 else ''})[/dim white]"
        )

    # ── STAGE 5: Checkpoint ─────────────────────────────────────

    def checkpoint(self, global_batch: int, val_f1: float,
                   const_version: int, n_rules: int, is_new_best: bool):
        if is_new_best:
            self.state.best_score = val_f1
            self.state.best_version = const_version

        best_tag = "  ★ Best" if is_new_best else ""
        inner = Text()
        inner.append(f"  Val: {val_f1:.3f}  |  Constitution: v{const_version} ({n_rules} rules){best_tag}")

        style = THEME["accent"] if is_new_best else THEME["primary"]
        self.console.print()
        self.console.print(Panel(
            inner,
            title=f"[{style}]Checkpoint (batch {global_batch})[/{style}]",
            border_style=style,
            padding=(0, 1),
        ))

    # ── STAGE 6: Epoch End ──────────────────────────────────────

    def epoch_end(self, epoch_result: Any, constitution: Any):
        epoch = epoch_result.epoch
        duration = epoch_result.epoch_duration
        v_start = epoch_result.constitution_version_start
        v_end = epoch_result.constitution_version_end
        epoch_scores = self.state.epoch_scores.get(epoch, [])
        score_str = "  ".join(f"{s:.2f}" for s in epoch_scores)

        summary = Text()
        summary.append(f"  Batches: {len(epoch_result.batch_results)}", style=THEME["muted"])
        summary.append(f"  |  Time: {duration:.1f}s", style=THEME["muted"])
        summary.append(f"  |  v{v_start} → v{v_end}\n", style=THEME["muted"])
        summary.append(f"  Accepted: {self.state.n_accepted}", style=THEME["success"])
        summary.append(f"  |  Rejected: {self.state.n_rejected}", style=THEME["error"])
        summary.append(f"  |  No change: {self.state.n_no_change}\n", style=THEME["muted"])
        if score_str:
            summary.append(f"  Scores: {score_str}", style=THEME["dim"])

        self.console.print()
        self.console.print(Panel(
            summary,
            title=f"[bold green]Epoch {epoch} Summary[/bold green]",
            border_style=THEME["border"],
            padding=(0, 1),
        ))

        rules = getattr(constitution, 'rules', [])
        if rules:
            self.console.print(self._rule_tree(rules, constitution.version))

    # ── STAGE 7: Holdout Evaluation ──────────────────────────────

    def holdout_eval(self, holdout_metrics: Dict, individual_results: List[Dict],
                     const_version: int, error_reports: Optional[List] = None):
        n = len(individual_results)

        if error_reports:
            table = self._results_table_from_reports(error_reports)
        else:
            table = self._results_table_from_individual(individual_results)

        footer = Text()
        footer.append(f"\n  Holdout score: {holdout_metrics.get('f1', 0.0):.3f}",
                      style=THEME["accent"])

        self.console.print()
        self.console.print(Panel(
            Group(table, footer),
            title=f"[bold green]HOLDOUT EVALUATION ({n} examples, constitution v{const_version})[/bold green]",
            border_style=THEME["border"],
            padding=(0, 1),
        ))

    # ── STAGE 8: Final Summary ──────────────────────────────────

    def final_summary(self, result: Any, elapsed: float, constitution: Any):
        holdout_f1 = result.holdout_metrics.get('f1', 0.0)
        n_rules = len(getattr(constitution, 'rules', []))
        n_accepted = self.state.n_accepted
        n_total = n_accepted + self.state.n_rejected

        # Baseline holdout metrics (may be None if not computed)
        baseline_holdout_f1 = None
        if hasattr(result, 'baseline_holdout_metrics') and result.baseline_holdout_metrics:
            baseline_holdout_f1 = result.baseline_holdout_metrics.get('f1', 0.0)

        content = Text()
        content.append("              TRAINING COMPLETE\n\n", style=THEME["bright"])

        # Holdout baseline + final with delta
        if baseline_holdout_f1 is not None:
            holdout_delta = holdout_f1 - baseline_holdout_f1
            content.append(f"  Holdout baseline :  {baseline_holdout_f1:.3f}", style=THEME["muted"])
            content.append(f"     Holdout final :  {holdout_f1:.3f}", style=THEME["accent"])
            content.append(f"   Δ {holdout_delta:+.3f}\n", style=THEME["success"] if holdout_delta > 0 else THEME["error"])
        else:
            content.append(f"  Holdout score    :  {holdout_f1:.3f}\n", style=THEME["muted"])

        content.append(f"  Constitution     :  v{result.final_constitution_version} ", style=THEME["muted"])
        content.append(f"({n_rules} rules, {n_accepted}/{n_total} accepted)\n", style=THEME["muted"])
        content.append(f"  Time             :  {elapsed:.1f}s\n", style=THEME["muted"])

        if hasattr(result, 'token_usage') and result.token_usage:
            tu = result.token_usage
            content.append(f"  Tokens           :  {tu.get('total_tokens', 0):,} "
                           f"({tu.get('total_calls', 0)} LLM calls)\n", style=THEME["muted"])

            # Per-tier summary (Worker / MAC / Adapt)
            from .agents import get_token_tracker
            tiers = get_token_tracker().get_per_tier()
            has_tiers = any(t['tokens'] > 0 for t in tiers.values())
            if has_tiers:
                content.append("\n  Per-tier tokens:\n", style=THEME["dim"])
                for tier_name in ['Worker', 'MAC', 'Adapt']:
                    t = tiers[tier_name]
                    if t['tokens'] > 0:
                        content.append(
                            f"    {tier_name:<10} {t['tokens']:>9,} tok  ({t['calls']} calls)\n",
                            style=THEME["dim"],
                        )

            per_agent = tu.get('per_agent', {})
            if per_agent:
                content.append("\n  Per-agent breakdown:\n", style=THEME["dim"])
                for agent, stats in sorted(per_agent.items()):
                    content.append(
                        f"    {agent:<24} {stats['total']:>9,} tok  ({stats['calls']} calls)\n",
                        style=THEME["dim"],
                    )

        if self.state.epoch_scores:
            content.append("\n  Score progression:\n", style=THEME["dim"])
            for ep, scores in sorted(self.state.epoch_scores.items()):
                score_str = " ".join(f"{s:.2f}" for s in scores)
                content.append(f"  E{ep}: {score_str}\n", style=THEME["dim"])

        self.console.print()
        self.console.print(Panel(
            content,
            border_style=THEME["border"],
            padding=(1, 2),
        ))

    # ── Helpers ─────────────────────────────────────────────────

    def _results_table_examples(self, batch_result: Any) -> Table:
        table = Table(show_header=True, header_style=THEME["accent"],
                      box=None, padding=(0, 1))
        table.add_column("#", style=THEME["muted"], width=4)
        table.add_column("Input", style="white", max_width=45, no_wrap=True)
        table.add_column("Score", justify="right", width=8)

        for i, report in enumerate(batch_result.error_reports, 1):
            score = report.score
            s = THEME["success"] if score >= 0.8 else THEME["error"]
            mark = "✓" if score >= 0.8 else "✗"
            score_text = Text(f"{score:.2f}  {mark}", style=s)
            table.add_row(str(i), self._truncate(report.input_text or f"ex_{i}", 42), score_text)

        return table

    def _results_table_from_reports(self, error_reports: List) -> Table:
        table = Table(show_header=True, header_style=THEME["accent"],
                      box=None, padding=(0, 1))
        table.add_column("#", style=THEME["muted"], width=4)
        table.add_column("Input", style="white", max_width=45, no_wrap=True)
        table.add_column("Score", justify="right", width=8)

        for i, report in enumerate(error_reports, 1):
            score = report.score
            s = THEME["success"] if score >= 0.8 else THEME["error"]
            mark = "✓" if score >= 0.8 else "✗"
            score_text = Text(f"{score:.2f}  {mark}", style=s)
            table.add_row(str(i), self._truncate(report.input_text or f"ex_{i}", 42), score_text)

        return table

    def _results_table_from_individual(self, individual_results: List[Dict]) -> Table:
        table = Table(show_header=True, header_style=THEME["accent"],
                      box=None, padding=(0, 1))
        table.add_column("#", style=THEME["muted"], width=4)
        table.add_column("Doc", style="white", max_width=30, no_wrap=True)
        table.add_column("F1", justify="right", width=8)

        for i, res in enumerate(individual_results, 1):
            f1 = res.get('evaluation', {}).get('f1', 0.0)
            s = THEME["success"] if f1 >= 0.8 else THEME["error"]
            mark = "✓" if f1 >= 0.8 else "✗"
            score_text = Text(f"{f1:.2f}  {mark}", style=s)
            table.add_row(str(i), self._truncate(str(res.get('doc_id', f'doc_{i}')), 27), score_text)

        return table

    def _rule_tree(self, rules: List[str], version: int) -> Tree:
        tree = Tree(f"[{THEME['accent']}]Constitution (v{version})[/{THEME['accent']}]")
        for i, rule in enumerate(rules, 1):
            tree.add(f"[{THEME['muted']}]{i}. {self._truncate(rule, 70)}[/{THEME['muted']}]")
        return tree

    def _truncate(self, text: str, n: int) -> str:
        text = text.replace('\n', ' ').strip()
        return text[:n] + "..." if len(text) > n else text
