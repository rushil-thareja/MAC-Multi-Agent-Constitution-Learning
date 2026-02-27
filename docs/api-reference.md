# API Reference

## MAC constructor

| Param | Default | Description |
|-------|---------|-------------|
| `model` | `"gpt-4o"` | Tier 1: Worker model (annotator) |
| `provider` | `"openai"` | Provider (`openai`, `openrouter`, `local`, `cerebras`) |
| `base_url` | `None` | Custom endpoint for worker (vLLM, etc.) |
| `temperature` | `1` | LLM temperature for worker |
| `mac_model` | `None` | Tier 2: MAC agents. Falls back to `model`. |
| `mac_provider` | `None` | Provider for MAC agents. Falls back to `provider`. |
| `mac_base_url` | `None` | Endpoint for MAC agents. Does NOT fall back to `base_url`. |
| `mac_temperature` | `None` | Temperature for MAC agents. Falls back to `temperature`. |
| `adapt_model` | `None` | Tier 3: Prompt adaptation. Falls back to `mac_model` → `model`. |
| `adapt_provider` | `None` | Provider for adaptation. Falls back to `mac_provider` → `provider`. |
| `adapt_base_url` | `None` | Endpoint for adaptation. Falls back to `mac_base_url`. |
| `adapt_temperature` | `None` | Temperature for adaptation. Falls back to `mac_temperature`. |
| `api_key` | env `OPENAI_API_KEY` | API key |
| `num_epochs` | `1` | Training epochs |
| `batch_size` | `4` | Examples per batch |
| `task_prompt` | `""` | Mode 1: your prompt (must contain `{{CONSTITUTION_BLOCK}}`) |
| `task_description` | `""` | Mode 2: enables meta-model prompt adaptation |
| `rule_type` | `""` | e.g. "math rules", "NER rules" |
| `output_key` | `"answer"` | JSON key in annotator output |

## compile()

```python
compiler.compile(
    trainset: List[Example],   # Training examples
    holdout: List[Example],    # Held-out validation examples
    metric: Callable,          # (prediction, gold) -> float
) -> CompiledMAC
```

## CompiledMAC

```python
optimized = compiler.compile(trainset, holdout, metric)

optimized("input text")            # Single inference → str
optimized.predict_batch(texts)     # Batch inference → List[str]
optimized.overview()               # Rich panel: baseline → final, rules tree
optimized.rules                    # List[str] - the learned rules
optimized.constitution             # str - formatted constitution
optimized.baseline_score           # float - score before training

optimized.save("rules.json")
loaded = CompiledMAC.load("rules.json")
```
