"""
HotpotQA — custom style (Mode 1: user-supplied prompt with {{CONSTITUTION_BLOCK}})

MAC learns constitution rules on HotpotQA multi-hop questions.
You supply the full prompt; MAC injects learned rules at {{CONSTITUTION_BLOCK}}.

Usage:
    # OpenAI models (Config C / D)
    python examples/hotpotqa_custom.py --worker-model gpt-4o-mini --mac-model gpt-5.2

    # vLLM worker + OpenAI MAC (Config A)
    python examples/hotpotqa_custom.py \\
        --worker-model Qwen/Qwen3-8B --worker-base-url http://localhost:8000/v1 \\
        --mac-model gpt-5.2

    # All vLLM (Config B)
    python examples/hotpotqa_custom.py \\
        --worker-model Qwen/Qwen3-8B --worker-base-url http://localhost:8000/v1 \\
        --mac-model Qwen/Qwen3-8B --mac-base-url http://localhost:8000/v1
"""

import sys, os, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mac import Example, MAC

TASK_PROMPT = """\
You are an expert question-answering system specializing in multi-hop reasoning.
Break complex questions into sub-questions, reason through each hop, then combine.

{{CONSTITUTION_BLOCK}}

Return your answer as JSON: {"reasoning": "...", "answer": "<concise answer>"}"""


def load_hotpotqa(n_train=80, n_holdout=16, seed=42):
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    items = list(ds)
    random.seed(seed)
    random.shuffle(items)
    items = items[:n_train + n_holdout]
    examples = [Example(input=row["question"], output=row["answer"]) for row in items]
    return examples[:n_train], examples[n_train:]


def token_f1(prediction, gold):
    pred_tokens = set(str(prediction).lower().split())
    gold_tokens = set(str(gold).lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    prec = tp / len(pred_tokens)
    rec = tp / len(gold_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HotpotQA — custom prompt style")
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
    parser.add_argument("--output", default="hotpotqa_custom_rules.json")
    args = parser.parse_args()

    train, holdout = load_hotpotqa(args.train, args.holdout)
    print(f"HotpotQA: {len(train)} train / {len(holdout)} holdout")

    mac_kwargs = dict(
        mac_model=args.mac_model,
        num_epochs=args.epochs,
        batch_size=4,
        provider="openai",
        task_prompt=TASK_PROMPT,
        task_description="Answer multi-hop questions by reasoning through sub-questions.",
        rule_type="multi-hop QA reasoning rules",
    )
    if args.worker_base_url:
        mac_kwargs["base_url"] = args.worker_base_url
    if args.mac_base_url:
        mac_kwargs["mac_base_url"] = args.mac_base_url

    compiler = MAC(model=args.worker_model, **mac_kwargs)
    optimized = compiler.compile(trainset=train, holdout=holdout, metric=token_f1)

    optimized.overview()
    print(f"\nRules learned: {len(optimized.rules)}")
    for i, r in enumerate(optimized.rules, 1):
        print(f"  {i}. {r}")

    optimized.save(args.output)
    print(f"\nSaved to {args.output}")
