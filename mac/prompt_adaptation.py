"""
External model prompt adaptation for MAC.

Uses a stronger external model to rewrite agent prompts at run-time,
making MAC generalizable to any downstream task without code changes.
Generalization happens ONLY through prompt transformation.
"""

import json
import re
import shutil
import logging
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple, Optional
from datetime import datetime

from .agents import BaseAgent, PromptTemplate

logger = logging.getLogger(__name__)


# ── Prompt files that get adapted ────────────────────────────────────────────

TARGET_PROMPTS = {
    "annotator": {
        "file": "annotator.md",
        "role": "Processes input text and completes the task, optionally guided by constitution rules",
        "critical_vars": ["CONSTITUTION_BLOCK", "TEXT_CONTENT", "OUTPUT_KEY"],
        "is_annotator": True,
    },
    "decision_agent": {
        "file": "decision_agent.md",
        "role": "Analyzes errors (FN/FP) and decides ADD/EDIT/REMOVE action on constitution",
        "critical_vars": [
            "FN_COUNT", "FP_COUNT", "TOTAL_ERRORS", "FP_PER_FN",
            "RULE_COUNT", "TREND_INFO", "FN_SAMPLE_PHRASES",
            "FP_SAMPLE_PHRASES", "CONSTITUTION_RULES",
        ],
        "is_annotator": False,
    },
    "rule_proposer": {
        "file": "rule_proposer.md",
        "role": "Creates new constitution rules based on error patterns",
        "critical_vars": [
            "PREVIOUS_REASONING", "CONSTITUTION_TEXT",
            "ERROR_CONTEXT", "PREVIOUS_REJECTIONS",
        ],
        "is_annotator": False,
    },
    "rule_editor": {
        "file": "rule_editor.md",
        "role": "Edits existing constitution rules to fix error patterns",
        "critical_vars": [
            "RULE_NUMBER_TO_EDIT", "CONSTITUTION_TEXT",
            "PREVIOUS_REASONING", "ERROR_CONTEXT", "PREVIOUS_REJECTIONS",
        ],
        "is_annotator": False,
    },
}


# ── Helper functions ─────────────────────────────────────────────────────────

def extract_template_variables(text: str) -> Set[str]:
    """Extract all {{VARIABLE_NAME}} template variables from text."""
    return set(re.findall(r"\{\{([A-Z_]+)\}\}", text))


def validate_adapted_prompt(
    original: str, adapted: str
) -> Tuple[bool, List[str]]:
    """
    Validate an adapted prompt against the original.

    Checks:
      1. All original {{VAR}}s are preserved (no missing vars)
      2. System: / User: structure present
      3. JSON code-fence count matches
      4. Length within 0.3x–2.0x of original

    Returns:
        (passed, list_of_error_strings)
    """
    errors: List[str] = []

    # 1 — Variable check
    expected_vars = extract_template_variables(original)
    found_vars = extract_template_variables(adapted)
    missing = expected_vars - found_vars
    if missing:
        errors.append(f"MISSING template variables: {missing}")

    # 2 — Structure check (only require markers present in original)
    if "System:" in original and "System:" not in adapted:
        errors.append("Missing System: section marker")
    if "User:" in original and "User:" not in adapted:
        errors.append("Missing User: section marker")

    # 3 — JSON fence count
    orig_json_count = original.count("```json")
    adapt_json_count = adapted.count("```json")
    if orig_json_count != adapt_json_count:
        errors.append(
            f"JSON fence count mismatch: original={orig_json_count}, "
            f"adapted={adapt_json_count}"
        )

    # 4 — Length check
    orig_len = len(original)
    adapt_len = len(adapted)
    if orig_len > 0:
        ratio = adapt_len / orig_len
        if ratio < 0.3 or ratio > 2.0:
            errors.append(
                f"Length ratio {ratio:.2f} out of range [0.3, 2.0] "
                f"(original={orig_len}, adapted={adapt_len})"
            )

    return (len(errors) == 0, errors)


def _repair_json_fences(original: str, adapted: str) -> str:
    """Re-inject JSON fences that the adapt model dropped.

    LLMs frequently consume ```json ... ``` blocks during rewriting.
    If the original has fences the adapted doesn't, extract each
    fence block from the original and append it to the adapted text.
    """
    import re
    orig_fences = re.findall(r"```json\s*\n.*?\n```", original, re.DOTALL)
    adapt_fence_count = adapted.count("```json")

    if len(orig_fences) > adapt_fence_count:
        missing = orig_fences[adapt_fence_count:]  # fences not in adapted
        for fence in missing:
            adapted = adapted.rstrip() + "\n\n" + fence + "\n"
        logger.info(
            f"[prompt-adapt] Auto-repaired {len(missing)} missing JSON fence(s)"
        )
    return adapted


def _format_data_samples(task_config: Dict) -> str:
    """Format real data samples for the meta-prompt."""
    samples = task_config.get('data_samples', [])
    if not samples:
        return "  (no data samples provided)\n"
    lines = []
    for s in samples:
        lines.append(f"  Input:  {s['input']}")
        lines.append(f"  Expected output: {s['output']}")
        lines.append("")
    return "\n".join(lines)


def _format_output_format(task_config: Dict) -> str:
    """Format output format section for the meta-prompt."""
    output_key = task_config.get('output_key', 'answer')
    fmt = task_config.get('output_format', 'list')
    example = task_config.get('output_example', '')

    if fmt == 'scalar':
        return (
            f"  The output is a SINGLE VALUE (not a list).\n"
            f"  The JSON must use key \"{output_key}\" with format:\n"
            f"  {{\"{output_key}\": \"value\"}}\n"
            f"  Example gold output: {example}"
        )
    elif fmt == 'dict':
        return (
            f"  The output is a DICTIONARY of key-value pairs.\n"
            f"  The JSON must use key \"{output_key}\" with format:\n"
            f"  {{\"{output_key}\": {{\"key\": \"value\"}}}}\n"
            f"  Example gold output: {example}"
        )
    else:  # list (default, original behavior)
        return (
            f"  The output is a LIST of items.\n"
            f"  The JSON must use key \"{output_key}\" with format:\n"
            f"  {{\"{output_key}\": [\"item1\", \"item2\"]}}\n"
            f"  Example gold output: {example}"
        )


def build_meta_prompt(
    original_content: str,
    agent_role: str,
    task_config: Dict,
    variables: Set[str],
    is_annotator: bool = True,
) -> Tuple[str, str]:
    """
    Build the (system, user) prompt pair sent to the external model.

    Uses a holistic rewriting approach: shows the model the full task context
    (description, real data samples, metric source) and asks it to rewrite the
    agent prompt for the new task while retaining all skills and structure.

    Args:
        is_annotator: If True, rewrite JSON output key to output_key.
                      If False, keep original JSON keys (pipeline agents).

    Returns:
        (system_prompt, user_prompt)
    """
    output_key = task_config.get('output_key', 'answer')
    output_fmt = task_config.get('output_format', 'list')

    if output_fmt == 'scalar':
        json_constraint = f'Change the JSON output key to "{output_key}" with a scalar value.'
    elif output_fmt == 'dict':
        json_constraint = f'Change the JSON output key to "{output_key}" with a dict value.'
    else:
        json_constraint = f'Change the JSON output key to "{output_key}" with a list value.'

    if is_annotator:
        constraint_3 = (
            "3. [This is an annotator] The prompt already contains a JSON format instruction "
            "with {{OUTPUT_KEY}} — keep that template variable exactly as-is. "
            "Do NOT replace {{OUTPUT_KEY}} with an actual key name."
        )
    else:
        constraint_3 = (
            "3. [This is a pipeline agent] Keep all JSON keys exactly as-is "
            "(rule_text, action, rule_index, reasoning, etc.) — do NOT rename them."
        )

    system_prompt = (
        "You are an expert prompt updater for a multi-agent optimisation system.\n"
        "These agents are currently written for a PRIVACY DETECTION task.\n"
        "Your job: rewrite the given agent prompt so it works for a NEW TASK,\n"
        "retaining all the agent's skills, abilities, reasoning flow, and output structure.\n"
        "\n"
        "HARD CONSTRAINTS — never violate these:\n"
        "1. Keep every {{VARIABLE_NAME}} template variable exactly as written.\n"
        "2. Keep System: / User: section markers if present.\n"
        f"{constraint_3}\n"
        "4. Output only the rewritten prompt — no --- delimiters, no code fences, no commentary.\n"
    )

    task_desc = task_config.get('description', 'Identify relevant items in text')
    data_samples_text = _format_data_samples(task_config)
    output_format_text = _format_output_format(task_config)
    metric_source = task_config.get('metric_source', '')
    variable_list = "\n".join(f"  {{{{{{v}}}}}}" for v in sorted(variables))

    user_prompt = (
        f"NEW TASK: \"{task_desc}\"\n"
        f"RULE TYPE: \"{task_config.get('rule_type', 'rules')}\"\n"
        f"\n"
        f"DATA SAMPLES (real input → expected output for this task):\n"
        f"{data_samples_text}\n"
        f"OUTPUT FORMAT:\n"
        f"{output_format_text}\n"
        f"\n"
    )

    if metric_source:
        user_prompt += (
            f"METRIC (how outputs are scored):\n"
            f"{metric_source}\n"
            f"\n"
        )

    user_prompt += (
        f"AGENT ROLE: {agent_role}\n"
        f"\n"
        f"TEMPLATE VARIABLES TO PRESERVE (keep these exact strings unchanged):\n"
        f"{variable_list}\n"
        f"\n"
        f"ORIGINAL PROMPT (currently written for privacy detection — rewrite for the new task above):\n"
        f"---\n"
        f"{original_content}\n"
        f"---\n"
        f"\n"
        f"Replace all privacy-specific language with task-appropriate language.\n"
        f"Preserve all structural patterns, reasoning steps, and output format.\n"
        f"Output only the rewritten prompt, starting directly with 'System:' or the first line."
    )

    return system_prompt, user_prompt


# ── FreeformAgent (reused pattern) ───────────────────────────────────────────

class _FreeformAgent(BaseAgent):
    """Minimal agent for issuing freeform prompts via existing provider infra."""

    def __init__(self, model_config: Dict):
        super().__init__(model_config, PromptTemplate(system="", user=""))

    def call(self, user_prompt: str, system_prompt: str = "") -> str:
        return self._call_llm(system_prompt, user_prompt)

    def process(self, **kwargs):
        return None


# ── PromptAdapter ────────────────────────────────────────────────────────────

class PromptAdapter:
    """
    Adapts agent prompts to a new task using an external model.

    Usage::

        adapter = PromptAdapter(meta_model_config, task_config)
        output_dir = adapter.adapt_all_prompts(
            Path("prompts"), run_dir / "adapted_prompts"
        )
        # output_dir now contains adapted .md files
    """

    def __init__(self, meta_model_config: Dict, task_config: Dict):
        """
        Args:
            meta_model_config: Model settings (provider, model_name, temperature, …)
            task_config: Task description (rule_type, description, metric)
        """
        if not task_config:
            raise ValueError(
                "meta_model.task section is required when meta_model.enabled=true. "
                "Please set task.rule_type, task.description, and task.metric."
            )

        self.task_config = task_config
        self.max_retries = meta_model_config.get("max_retries", 2)

        # Build model config for the external model
        model_cfg = {
            "provider": meta_model_config.get("provider", "openai"),
            "model_name": meta_model_config.get("model_name", "gpt-4o"),
            "temperature": meta_model_config.get("temperature", 0.3),
            "max_completion_tokens": meta_model_config.get(
                "max_completion_tokens", 8000
            ),
            "timeout": meta_model_config.get("timeout", 120),
            "max_retries": 2,
            # Pass through base_url / api key env vars
            "base_url": meta_model_config.get("base_url"),
            "openrouter_base_url": meta_model_config.get(
                "openrouter_base_url", "https://openrouter.ai/api/v1"
            ),
        }

        self.agent = _FreeformAgent(model_cfg)

        # Which prompts to adapt (all true by default)
        self.adapt_flags = meta_model_config.get("adapt_prompts", {})

        # Quiet mode: suppress print() when Rich display handles output
        self.quiet = False

    # ── Single prompt ────────────────────────────────────────────────────

    def adapt_single_prompt(
        self,
        key: str,
        original_content: str,
        agent_role: str,
        critical_vars: List[str],
        is_annotator: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Adapt one prompt with retry + validation.

        Returns:
            (adapted_content, log_entry_dict)
        """
        variables = extract_template_variables(original_content)
        # Ensure critical vars are in the set (they should be, but belt-and-suspenders)
        variables.update(critical_vars)

        system_prompt, user_prompt = build_meta_prompt(
            original_content, agent_role, self.task_config, variables,
            is_annotator=is_annotator,
        )

        log_entry: Dict = {
            "prompt_name": key,
            "success": False,
            "retries": 0,
            "errors": [],
        }

        for attempt in range(self.max_retries + 1):
            try:
                # On retry, append error feedback to user prompt
                retry_suffix = ""
                if attempt > 0 and log_entry["errors"]:
                    last_errors = log_entry["errors"][-1]
                    retry_suffix = (
                        f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n"
                        f"{last_errors}\n"
                        f"Fix the issues and try again."
                    )

                adapted = self.agent.call(
                    user_prompt + retry_suffix, system_prompt
                )

                # Strip --- delimiters / markdown fencing the LLM may wrap
                adapted = adapted.strip()
                if adapted.startswith("---"):
                    adapted = adapted[3:].strip()
                if adapted.endswith("---"):
                    adapted = adapted[:-3].strip()
                # Strip markdown code fences (```markdown ... ``` or ``` ... ```)
                if adapted.startswith("```"):
                    first_nl = adapted.find("\n")
                    if first_nl != -1:
                        adapted = adapted[first_nl + 1:]
                    if adapted.rstrip().endswith("```"):
                        adapted = adapted.rstrip()[:-3].rstrip()

                # Auto-repair: if adapted is missing JSON fences that
                # exist in the original, inject them back.  LLMs often
                # consume code fences during rewriting.
                adapted = _repair_json_fences(original_content, adapted)

                passed, errors = validate_adapted_prompt(
                    original_content, adapted
                )

                if passed:
                    log_entry["success"] = True
                    log_entry["retries"] = attempt
                    logger.info(
                        f"[prompt-adapt] {key}: adapted successfully "
                        f"(attempt {attempt + 1})"
                    )
                    return adapted, log_entry

                log_entry["errors"].append(
                    f"Attempt {attempt + 1}: {errors}"
                )
                logger.warning(
                    f"[prompt-adapt] {key}: validation failed "
                    f"(attempt {attempt + 1}): {errors}"
                )

            except Exception as exc:
                log_entry["errors"].append(
                    f"Attempt {attempt + 1}: exception — {exc}"
                )
                logger.error(
                    f"[prompt-adapt] {key}: LLM call failed "
                    f"(attempt {attempt + 1}): {exc}"
                )

        # All retries exhausted — fall back to original
        log_entry["retries"] = self.max_retries
        logger.warning(
            f"[prompt-adapt] {key}: all retries failed, "
            f"falling back to original prompt"
        )
        if not self.quiet:
            print(
                f"[prompt-adapt] WARNING: {key} adaptation failed after "
                f"{self.max_retries + 1} attempts, using original prompt"
            )
        return original_content, log_entry

    # ── All prompts ──────────────────────────────────────────────────────

    def adapt_all_prompts(
        self, prompts_dir: Path, output_dir: Path,
        on_progress: Optional[Callable] = None,
        on_agent_start: Optional[Callable] = None,
    ) -> Path:
        """
        Adapt all target prompts, copy the rest unchanged.

        Args:
            prompts_dir: Source prompts directory
            output_dir:  Destination directory (e.g. run_dir/adapted_prompts)

        Returns:
            output_dir (same as input, for chaining)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        adaptation_log: List[Dict] = []

        if not self.quiet:
            print(f"[prompt-adapt] Adapting prompts for task: "
                  f"{self.task_config.get('description', '?')}")
            print(f"[prompt-adapt] Rule type: "
                  f"{self.task_config.get('rule_type', '?')}")
            print(f"[prompt-adapt] Source: {prompts_dir}")
            print(f"[prompt-adapt] Output: {output_dir}")

        # Collect names of files we'll adapt
        target_files = {
            info["file"] for info in TARGET_PROMPTS.values()
        }

        # Adapt target prompts
        for key, info in TARGET_PROMPTS.items():
            filename = info["file"]
            src_path = prompts_dir / filename

            # Check per-agent toggle
            if not self.adapt_flags.get(key, True):
                # User disabled adaptation for this agent — copy unchanged
                if src_path.exists():
                    shutil.copy2(src_path, output_dir / filename)
                    adaptation_log.append({
                        "prompt_name": key,
                        "success": True,
                        "retries": 0,
                        "errors": [],
                        "skipped": "disabled in config",
                    })
                    if on_progress:
                        on_progress(key=key, role=info["role"], success=True,
                                    preview="(skipped — disabled in config)",
                                    idx=len(adaptation_log), total=len(TARGET_PROMPTS))
                continue

            if not src_path.exists():
                logger.warning(
                    f"[prompt-adapt] {filename} not found in {prompts_dir}, skipping"
                )
                adaptation_log.append({
                    "prompt_name": key,
                    "success": False,
                    "retries": 0,
                    "errors": [f"File not found: {src_path}"],
                })
                if on_progress:
                    on_progress(key=key, role=info["role"], success=False,
                                preview=f"(file not found: {filename})",
                                idx=len(adaptation_log), total=len(TARGET_PROMPTS))
                continue

            original = src_path.read_text(encoding="utf-8")

            if on_agent_start:
                on_agent_start(key=key, role=info["role"],
                               idx=len(adaptation_log) + 1, total=len(TARGET_PROMPTS))

            adapted, log_entry = self.adapt_single_prompt(
                key=key,
                original_content=original,
                agent_role=info["role"],
                critical_vars=info["critical_vars"],
                is_annotator=info.get("is_annotator", True),
            )

            # Write adapted prompt
            (output_dir / filename).write_text(adapted, encoding="utf-8")
            adaptation_log.append(log_entry)

            if on_progress:
                on_progress(
                    key=key,
                    role=info["role"],
                    success=log_entry.get("success", False),
                    preview=adapted[:200],
                    errors=log_entry.get("errors", []),
                    idx=len(adaptation_log),
                    total=len(TARGET_PROMPTS),
                )

        # Copy all OTHER .md files unchanged
        for md_file in prompts_dir.glob("*.md"):
            if md_file.name not in target_files:
                dest = output_dir / md_file.name
                if not dest.exists():
                    shutil.copy2(md_file, dest)

        # Save adaptation log
        log_path = output_dir / "adaptation_log.json"
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "task_config": self.task_config,
            "results": adaptation_log,
            "summary": {
                "total": len(adaptation_log),
                "succeeded": sum(
                    1 for e in adaptation_log if e.get("success")
                ),
                "failed": sum(
                    1 for e in adaptation_log if not e.get("success")
                ),
            },
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        if not self.quiet:
            print(f"[prompt-adapt] Done: {log_data['summary']['succeeded']}/"
                  f"{log_data['summary']['total']} prompts adapted successfully")
            print(f"[prompt-adapt] Log saved to: {log_path}")

        return output_dir
