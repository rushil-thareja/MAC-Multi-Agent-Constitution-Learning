# Usage Modes

MAC supports two prompting modes for different levels of control.

---

## Mode 1: Custom Prompt

You write the full prompt and include a `{{CONSTITUTION_BLOCK}}` placeholder where the learned rules will be injected:

```python
compiler = MAC(
    model="gpt-4o",
    task_prompt=(
        "You are an expert math solver.\n\n"
        "{{CONSTITUTION_BLOCK}}\n\n"
        'Return JSON: {"reasoning": "...", "answer": 42}'
    ),
    task_description="Competition math",
    rule_type="math reasoning rules",
)
optimized = compiler.compile(trainset=train, holdout=holdout, metric=metric)
```

Use this when you need precise control over prompt structure, output format, or system instructions.

---

## Mode 2: Auto-Adapt

MAC builds the prompt automatically from `task_description`. A meta-model sees real data samples, output format, and metric source code, then makes structural adaptations (not just word swaps):

```python
compiler = MAC(
    model="gpt-4o",
    task_description="Extract named entities from legal text",
    rule_type="NER rules",
)
optimized = compiler.compile(trainset=train, holdout=holdout, metric=metric)
```

Use this when you want MAC to handle prompt engineering for you.

---

## Works with Any Task

MAC is task-agnostic. Beyond the benchmarks shown in the README, it works on:

- **Phrase extraction** - extract key phrases from documents
- **Classification** - sentiment, intent, topic classification
- **Structured output** - JSON extraction, table filling
- **Tool calling** - API call generation (see Tool-MAC in [paper results](paper-results.md))
