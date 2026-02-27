# Model Configuration

MAC supports a three-tier model setup so you can independently configure the worker, MAC agents, and prompt adaptation.

## Three-Tier Setup

```python
compiler = MAC(
    # Tier 1: Worker (annotation) - can be cheap/local
    model="Qwen/Qwen3-8B",
    provider="openai",
    base_url="http://localhost:8000/v1",  # vLLM endpoint

    # Tier 2: MAC agents (decision/proposer/editor) - should be strong
    mac_model="gpt-5.2",
    # mac_base_url defaults to None (NOT inherited from base_url)

    # Tier 3: Prompt adaptation - defaults to mac_model
    # adapt_model="gpt-5.2",

    task_description="Solve AIME competition math problems.",
    rule_type="math reasoning rules",
)
```

## Fallback Cascade

Each tier falls back to the one above:

```
adapt_model     → mac_model     → model
adapt_provider  → mac_provider  → provider
adapt_base_url  → mac_base_url  → (NO fallback to base_url)
adapt_temperature → mac_temperature → temperature
```

`mac_base_url` does **NOT** fall back to `base_url`. If your worker is on vLLM, MAC agents should NOT default to that same local server.

## Provider Examples

**vLLM (local):**
```python
MAC(model="Qwen/Qwen3-8B", base_url="http://localhost:8000/v1")
```

**OpenAI:**
```python
MAC(model="gpt-4o")  # uses OPENAI_API_KEY from env
```

**OpenRouter:**
```python
MAC(model="meta-llama/llama-3-70b", provider="openrouter")
# uses OPENROUTER_API_KEY from env
```

**Cerebras:**
```python
MAC(model="llama3.1-8b", provider="cerebras")
```
