#!/usr/bin/env python3
"""
Mini smoke-test — run MAC through every config mode with tiny data.

No external datasets needed.  Each config runs 1 epoch on 4 train / 2
holdout examples so a full sweep finishes in minutes, not hours.

Configs tested (when all flags are supplied):

  1. API worker  + API MAC   — adapt style
  2. API worker  + API MAC   — custom style
  3. vLLM worker + API MAC   — adapt style   (needs --vllm-url + --vllm-model)
  4. vLLM worker + API MAC   — custom style   (needs --vllm-url + --vllm-model)
  5. Fully local (vLLM only) — adapt style   (needs --vllm-url + --vllm-model)
  6. Fully local (vLLM only) — custom style   (needs --vllm-url + --vllm-model)

Usage
-----
# API-only (configs 1-2):
python examples/mini_test.py

# All six configs:
python examples/mini_test.py \\
    --vllm-url http://localhost:8000/v1 \\
    --vllm-model Qwen/Qwen3-8B

# Override the cloud models:
python examples/mini_test.py \\
    --worker-model gpt-4o-mini --mac-model gpt-5.2
"""

import sys, os, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mac import Example, MAC

# ---------------------------------------------------------------------------
# Inline data — trivial arithmetic so even small models can score > 0
# ---------------------------------------------------------------------------

TRAIN = [
    Example(input="What is 12 + 7?", output="19"),
    Example(input="What is 25 - 8?", output="17"),
    Example(input="What is 6 * 9?", output="54"),
    Example(input="What is 100 / 4?", output="25"),
]

HOLDOUT = [
    Example(input="What is 15 + 23?", output="38"),
    Example(input="What is 9 * 7?", output="63"),
]

TASK_DESCRIPTION = (
    "Solve the arithmetic problem. Return ONLY the numeric answer, "
    "no explanation, no units."
)
RULE_TYPE = "arithmetic reasoning rules"

CUSTOM_PROMPT = """\
You are a calculator.  Solve the arithmetic problem step by step.

{{CONSTITUTION_BLOCK}}

Return your answer as JSON: {"answer": "<number>"}"""


def numeric_match(prediction, gold):
    """Metric: 1.0 if the numbers match, else 0.0."""
    try:
        return 1.0 if float(str(prediction).strip().rstrip(".")) == float(str(gold).strip()) else 0.0
    except (ValueError, TypeError):
        return 1.0 if str(prediction).strip() == str(gold).strip() else 0.0


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _base_kwargs(mac_model, epochs=1, batch_size=2):
    return dict(
        mac_model=mac_model,
        num_epochs=epochs,
        batch_size=batch_size,
        task_description=TASK_DESCRIPTION,
        rule_type=RULE_TYPE,
    )


def build_configs(args):
    """Return list of (name, MAC-constructor-kwargs) tuples."""
    configs = []

    # --- API-only configs ---------------------------------------------------
    base = _base_kwargs(args.mac_model, args.epochs, args.batch_size)

    # 1) API adapt
    configs.append((
        "API worker + API MAC (adapt)",
        dict(model=args.worker_model, **base),
    ))

    # 2) API custom
    configs.append((
        "API worker + API MAC (custom)",
        dict(model=args.worker_model, task_prompt=CUSTOM_PROMPT, **base),
    ))

    # --- vLLM configs (only when --vllm-url is given) -----------------------
    if args.vllm_url:
        vllm_model = args.vllm_model

        # 3) vLLM worker + cloud MAC — adapt
        configs.append((
            "vLLM worker + API MAC (adapt)",
            dict(model=vllm_model, base_url=args.vllm_url, **base),
        ))

        # 4) vLLM worker + cloud MAC — custom
        configs.append((
            "vLLM worker + API MAC (custom)",
            dict(model=vllm_model, base_url=args.vllm_url,
                 task_prompt=CUSTOM_PROMPT, **base),
        ))

        # 5) Fully local — adapt
        local_base = _base_kwargs(vllm_model, args.epochs, args.batch_size)
        local_base["mac_base_url"] = args.vllm_url
        configs.append((
            "Fully local (adapt)",
            dict(model=vllm_model, base_url=args.vllm_url, **local_base),
        ))

        # 6) Fully local — custom
        local_base2 = _base_kwargs(vllm_model, args.epochs, args.batch_size)
        local_base2["mac_base_url"] = args.vllm_url
        configs.append((
            "Fully local (custom)",
            dict(model=vllm_model, base_url=args.vllm_url,
                 task_prompt=CUSTOM_PROMPT, **local_base2),
        ))

    return configs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(name, mac_kwargs):
    """Run a single config.  Returns (name, status, score, n_rules, elapsed)."""
    print(f"\n{'='*60}")
    print(f"  CONFIG: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        compiler = MAC(**mac_kwargs)
        result = compiler.compile(trainset=TRAIN, holdout=HOLDOUT, metric=numeric_match)
        elapsed = time.time() - t0

        result.overview()
        n_rules = len(result.rules)
        score = result.result.holdout_metrics.get("f1", 0.0) if result.result else 0.0

        # Quick inference sanity check
        answer = result("What is 3 + 4?")
        print(f"\n  Inference check:  '3 + 4' -> '{answer}'")

        return (name, "PASS", score, n_rules, elapsed)

    except Exception:
        elapsed = time.time() - t0
        traceback.print_exc()
        return (name, "FAIL", 0.0, 0, elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Mini smoke-test — run MAC through every config mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--worker-model", default="gpt-4o-mini",
                        help="API worker model (default: gpt-4o-mini)")
    parser.add_argument("--mac-model", default="gpt-4o",
                        help="API MAC-agent model (default: gpt-4o)")
    parser.add_argument("--vllm-url", default=None,
                        help="vLLM base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-8B",
                        help="Model name served by vLLM (default: Qwen/Qwen3-8B)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs per config (default: 1)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per config (default: 2)")
    args = parser.parse_args()

    configs = build_configs(args)

    print(f"\nMini smoke-test: {len(configs)} configs, "
          f"{len(TRAIN)} train / {len(HOLDOUT)} holdout, "
          f"{args.epochs} epoch(s), batch_size={args.batch_size}")
    print(f"Worker model : {args.worker_model}")
    print(f"MAC model    : {args.mac_model}")
    if args.vllm_url:
        print(f"vLLM endpoint: {args.vllm_url}  ({args.vllm_model})")

    results = []
    for name, kw in configs:
        results.append(run_one(name, kw))

    # --- Summary table ------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<40} {'Status':>6}  {'Score':>6}  {'Rules':>5}  {'Time':>7}")
    print(f"  {'-'*40} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}")
    for name, status, score, n_rules, elapsed in results:
        tag = "PASS" if status == "PASS" else "FAIL"
        print(f"  {name:<40} {tag:>6}  {score:>5.1%}  {n_rules:>5}  {elapsed:>6.1f}s")

    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_fail = len(results) - n_pass
    print(f"\n  {n_pass} passed, {n_fail} failed out of {len(results)} configs")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
