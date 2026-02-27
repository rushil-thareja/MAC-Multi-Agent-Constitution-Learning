"""
GSM8K — adapt style (Mode 2: MAC auto-adapts prompt)

MAC learns constitution rules on GSM8K math problems.
Prompt is automatically adapted by the MAC agent.

Usage:
    # OpenAI models (Config C / D)
    python examples/gsm8k_adapt.py --worker-model gpt-4o-mini --mac-model gpt-5.2

    # vLLM worker + OpenAI MAC (Config A)
    python examples/gsm8k_adapt.py \\
        --worker-model Qwen/Qwen3-8B --worker-base-url http://localhost:8000/v1 \\
        --mac-model gpt-5.2

    # All vLLM (Config B)
    python examples/gsm8k_adapt.py \\
        --worker-model Qwen/Qwen3-8B --worker-base-url http://localhost:8000/v1 \\
        --mac-model Qwen/Qwen3-8B --mac-base-url http://localhost:8000/v1
"""

import sys, os, re, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mac import Example, MAC


def load_gsm8k(n_train=80, n_holdout=16, seed=42):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = list(ds)
    random.seed(seed)
    random.shuffle(items)
    items = items[:n_train + n_holdout]
    examples = []
    for row in items:
        match = re.search(r'####\s*(.+)', row["answer"])
        answer = match.group(1).strip().replace(",", "") if match else row["answer"].strip()
        examples.append(Example(input=row["question"], output=answer))
    return examples[:n_train], examples[n_train:]


def numeric_match(prediction, gold):
    try:
        return 1.0 if float(str(prediction).strip().rstrip('.')) == float(str(gold).strip()) else 0.0
    except (ValueError, TypeError):
        return 1.0 if str(prediction).strip() == str(gold).strip() else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K — adapt style")
    parser.add_argument("--worker-model", default="gpt-4o-mini",
                        help="Worker (student) model. e.g. gpt-4o-mini, Qwen/Qwen3-8B")
    parser.add_argument("--worker-base-url", default=None,
                        help="Base URL for worker (e.g. http://localhost:8000/v1 for vLLM)")
    parser.add_argument("--mac-model", default="gpt-5.2",
                        help="MAC controller model (decision/proposer/editor agents)")
    parser.add_argument("--mac-base-url", default=None,
                        help="Base URL for MAC model (if hosted via vLLM)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--holdout", type=int, default=16)
    parser.add_argument("--output", default="gsm8k_adapt_rules.json")
    args = parser.parse_args()

    train, holdout = load_gsm8k(args.train, args.holdout)
    print(f"GSM8K: {len(train)} train / {len(holdout)} holdout")

    mac_kwargs = dict(
        mac_model=args.mac_model,
        num_epochs=args.epochs,
        batch_size=4,
        provider="openai",
        task_description="Solve the math word problem step by step. Return only the final numeric answer.",
        rule_type="math reasoning rules",
    )
    if args.worker_base_url:
        mac_kwargs["base_url"] = args.worker_base_url
    if args.mac_base_url:
        mac_kwargs["mac_base_url"] = args.mac_base_url

    compiler = MAC(model=args.worker_model, **mac_kwargs)
    optimized = compiler.compile(trainset=train, holdout=holdout, metric=numeric_match)

    optimized.overview()
    print(f"\nRules learned: {len(optimized.rules)}")
    for i, r in enumerate(optimized.rules, 1):
        print(f"  {i}. {r}")

    optimized.save(args.output)
    print(f"\nSaved to {args.output}")
