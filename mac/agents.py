"""
Agentic pipeline with role-based LLM agents for constitutional classifier.
"""

import json
import re
import os
import logging
import asyncio
from datetime import datetime
from string import Template
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import requests
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Import enhanced phrase matching evaluator
# from .enhanced_phrase_matching_evaluator import PhraseMatchingEvaluator


# Note: filter_meaningful_phrases was removed as it could hide important patterns
# Now we just show the first N phrases directly without filtering


# Global singleton for local LLM inference to avoid loading model multiple times
_GLOBAL_LOCAL_LLM_INFERENCE = None


# ============================================================================
# GLOBAL TOKEN TRACKER - Tracks all LLM token usage across the run
# ============================================================================
class TokenTracker:
    """Global singleton to track token usage across all LLM calls."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all token counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.per_agent_tokens = {}  # agent_name -> {prompt, completion, total, calls}
        self.per_batch_tokens = []  # List of token counts per batch
        self.current_batch_tokens = {'prompt': 0, 'completion': 0, 'total': 0, 'calls': 0}

    def add_tokens(self, prompt_tokens: int, completion_tokens: int, agent_name: str = "unknown"):
        """Add tokens from an LLM call."""
        total = prompt_tokens + completion_tokens

        # Global totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total
        self.call_count += 1

        # Per-agent tracking
        if agent_name not in self.per_agent_tokens:
            self.per_agent_tokens[agent_name] = {'prompt': 0, 'completion': 0, 'total': 0, 'calls': 0}
        self.per_agent_tokens[agent_name]['prompt'] += prompt_tokens
        self.per_agent_tokens[agent_name]['completion'] += completion_tokens
        self.per_agent_tokens[agent_name]['total'] += total
        self.per_agent_tokens[agent_name]['calls'] += 1

        # Current batch tracking
        self.current_batch_tokens['prompt'] += prompt_tokens
        self.current_batch_tokens['completion'] += completion_tokens
        self.current_batch_tokens['total'] += total
        self.current_batch_tokens['calls'] += 1

    def end_batch(self):
        """Mark end of batch, save batch tokens and reset batch counter."""
        self.per_batch_tokens.append(self.current_batch_tokens.copy())
        self.current_batch_tokens = {'prompt': 0, 'completion': 0, 'total': 0, 'calls': 0}

    def get_summary(self) -> Dict:
        """Get summary of all token usage."""
        return {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'total_calls': self.call_count,
            'per_agent': self.per_agent_tokens,
            'per_batch': self.per_batch_tokens,
            'current_batch': self.current_batch_tokens
        }

    def snapshot(self) -> Dict:
        """Return current totals for computing deltas between steps."""
        return {'tokens': self.total_tokens, 'calls': self.call_count}

    def get_totals(self) -> Tuple[int, int, int]:
        """Get (prompt_tokens, completion_tokens, total_tokens)."""
        return (self.total_prompt_tokens, self.total_completion_tokens, self.total_tokens)

    # Agent class names that belong to each tier
    _WORKER_AGENTS = {'AnnotatorAgent'}
    _ADAPT_AGENTS = {'_FreeformAgent'}  # prompt_adaptation.py's adapt agent
    # Everything else is MAC tier

    def get_per_tier(self) -> Dict[str, Dict[str, int]]:
        """Aggregate token usage into Worker / MAC / Adapt tiers."""
        tiers = {
            'Worker': {'tokens': 0, 'calls': 0},
            'MAC': {'tokens': 0, 'calls': 0},
            'Adapt': {'tokens': 0, 'calls': 0},
        }
        for agent_name, stats in self.per_agent_tokens.items():
            if agent_name in self._WORKER_AGENTS:
                tier = 'Worker'
            elif agent_name in self._ADAPT_AGENTS:
                tier = 'Adapt'
            else:
                tier = 'MAC'
            tiers[tier]['tokens'] += stats['total']
            tiers[tier]['calls'] += stats['calls']
        return tiers


# Global singleton instance
_GLOBAL_TOKEN_TRACKER = TokenTracker()


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    return _GLOBAL_TOKEN_TRACKER


def reset_token_tracker():
    """Reset the global token tracker."""
    _GLOBAL_TOKEN_TRACKER.reset()


def _get_or_create_local_llm_inference(model_config: Dict):
    """
    Get or create the global LocalLLMInference singleton.
    This ensures the model is loaded only ONCE across all agents.

    Args:
        model_config: Model configuration dict

    Returns:
        LocalLLMInference instance or None if not enabled
    """
    global _GLOBAL_LOCAL_LLM_INFERENCE

    # Check if local_llm is enabled in config
    local_llm_config = model_config.get('local_llm', {})
    if not local_llm_config.get('enabled', False):
        # Local LLM not enabled, return None (use existing providers)
        return None

    # If already initialized, return existing instance
    if _GLOBAL_LOCAL_LLM_INFERENCE is not None:
        return _GLOBAL_LOCAL_LLM_INFERENCE

    # Initialize for the first time
    try:
        from .local_llm_inference import LocalLLMInference

        model_path = local_llm_config.get('model_path')
        if not model_path:
            raise ValueError("local_llm.model_path is required when local_llm.enabled is true")

        print("[LocalLLM] Initializing model (ONCE for all agents)...")
        _GLOBAL_LOCAL_LLM_INFERENCE = LocalLLMInference(model_path, local_llm_config)
        print("[LocalLLM] Model loaded and ready for batched inference")

        return _GLOBAL_LOCAL_LLM_INFERENCE
    except Exception as e:
        print(f"[LocalLLM] WARNING: Failed to initialize local LLM: {e}")
        print("[LocalLLM] Falling back to existing provider...")
        return None


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    system: str
    user: str
    
    def format(self, **kwargs) -> Tuple[str, str]:
        """Format template with variables using $VAR syntax to avoid JSON brace conflicts."""
        # Convert {VAR} to $VAR in templates first
        system_template = self.system.replace('{{', '${').replace('}}', '}')
        user_template = self.user.replace('{{', '${').replace('}}', '}')
        
        # Use Template class for safe substitution
        system_tmpl = Template(system_template)
        user_tmpl = Template(user_template)
        
        return system_tmpl.substitute(**kwargs), user_tmpl.substitute(**kwargs)


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, model_config: Dict, prompt_template: PromptTemplate):
        """
        Initialize agent.

        Args:
            model_config: Model configuration
            prompt_template: Prompt template for this agent
        """
        self.model_config = model_config
        self.prompt_template = prompt_template

        # Agent memory — stores last N actions for cross-batch context
        self._memory: List[str] = []
        self._memory_max = 5

        # Initialize client based on provider
        provider = model_config.get('provider', 'openai').lower()

        if provider == 'local':
            self.client = None
            self.local_endpoint = model_config.get(
                'local_endpoint', 'http://localhost:8000/generate'
            )
            self.local_model_name = model_config.get(
                'local_model_name', 'local-llm'
            )
        elif provider == "cerebras":
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("Cerebras api key not found cant proceed with cerebras inferecing")
            from cerebras.cloud.sdk import Cerebras
            self.client = Cerebras(api_key = api_key)
            self.cerebras_endpoint = model_config.get(
                'cerebras_endpoint',
                'https://api.cerebras.ai/v1/chat/completions'
            )

        elif provider == 'openrouter':
            # Initialize OpenRouter client (compatible with OpenAI interface)
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is required when using OpenRouter provider")

            base_url = model_config.get('openrouter_base_url', 'https://openrouter.ai/api/v1')
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        elif provider == 'openai':
            # Initialize OpenAI client (supports custom base_url for vLLM, etc.)
            api_key = os.getenv('OPENAI_API_KEY')
            base_url = model_config.get('base_url')  # Optional: for vLLM or other OpenAI-compatible servers

            # For vLLM/local servers, API key may not be required
            if not api_key and not base_url:
                raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider (unless base_url is set for local server)")

            # Use dummy key for local servers if not provided
            if not api_key:
                api_key = "not-needed"  # vLLM doesn't require a real API key

            if base_url:
                logger.info(f"[OpenAI] Using custom base_url: {base_url}")
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: openai, openrouter, local, cerebras"
            )

        self.provider = provider

        # Initialize async client for concurrent batch processing (vLLM/OpenAI only)
        self.async_client = None
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY') or "not-needed"
            base_url = model_config.get('base_url')
            if base_url:
                self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            else:
                self.async_client = AsyncOpenAI(api_key=api_key)

        # Initialize local LLM inference (singleton, loaded once globally)
        # This is separate from provider='local' which uses HTTP endpoint
        self.local_llm_inference = None
        self.local_llm_enabled = False
        self.use_batched_local = False

        # Try to initialize local LLM if enabled in config
        try:
            # Merge parent model config with local_llm config (local_llm takes precedence)
            local_llm_config = model_config.get('local_llm', {})
            if local_llm_config.get('enabled', False):
                # Create merged config: parent model params + local_llm overrides
                merged_config = {
                    'temperature': model_config.get('temperature', 1.0),
                    'top_p': model_config.get('top_p', 0.9),
                    'top_k': model_config.get('top_k', 50),
                    'max_completion_tokens': model_config.get('max_completion_tokens'),
                    'max_tokens': model_config.get('max_tokens'),
                    **local_llm_config  # Local overrides
                }
                model_config_with_local = {**model_config, 'local_llm': merged_config}
            else:
                model_config_with_local = model_config

            self.local_llm_inference = _get_or_create_local_llm_inference(model_config_with_local)
            if self.local_llm_inference is not None:
                self.local_llm_enabled = True
                self.use_batched_local = local_llm_config.get('use_batched', True)
            else:
                self.local_llm_enabled = False
                self.use_batched_local = False
        except Exception as e:
            # If initialization fails, just continue without local LLM (backward compatible)
            self.logger.warning(f"Local LLM initialization failed: {e}, continuing with existing provider")
            self.local_llm_inference = None
            self.local_llm_enabled = False
            self.use_batched_local = False

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup LLM interaction logging (will be updated by run manager)
        self.log_dir = Path("logs/llm_interactions")
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def set_log_dir(self, log_dir: Path):
        """Set the logging directory (called by run manager)."""
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Agent Memory ──────────────────────────────────────────────
    def remember(self, action: str):
        """Store a brief note about what this agent just did."""
        self._memory.append(action)
        if len(self._memory) > self._memory_max:
            self._memory = self._memory[-self._memory_max:]

    def get_memory_text(self) -> str:
        """Format memory as a string for injection into prompts."""
        if not self._memory:
            return ""
        lines = [f"  {i+1}. {m}" for i, m in enumerate(self._memory)]
        return "YOUR RECENT ACTIONS (avoid repeating failed approaches):\n" + "\n".join(lines)

    def clear_memory(self):
        """Reset memory (e.g. between runs)."""
        self._memory.clear()

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """Process input and return results."""
        pass
    
    def _log_llm_interaction(self, agent_name: str, system_prompt: str, 
                           user_prompt: str, response: str, error: str = None):
        """
        Log LLM request and response to file.
        
        Args:
            agent_name: Name of the agent making the call
            system_prompt: System prompt sent
            user_prompt: User prompt sent
            response: LLM response received
            error: Error message if any
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self.log_dir / f"{agent_name}_{timestamp}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "provider": self.provider,
            "request": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model_config": self.model_config
            },
            "response": response,
            "error": error
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def _call_llm(self, system_prompt: str, user_prompt: str,
                  max_retries: int = None, temperature: float = None) -> str:
        """
        Call LLM with retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            max_retries: Maximum number of retries
            temperature: Optional temperature override (uses config default if None)

        Returns:
            LLM response text
        """
        max_retries = max_retries or self.model_config.get('max_retries', 3)
        agent_name = self.__class__.__name__

        # Check if local LLM direct inference is available (batched mode)
        if self.local_llm_enabled and self.local_llm_inference is not None:
            # Use local LLM direct inference (batched call with batch_size=1)
            try:
                response_texts, metadata = self.local_llm_inference.call_batched(
                    system_prompts=[system_prompt],
                    user_prompts=[user_prompt],
                    timeout=self.model_config.get('batch_timeout', 300.0),
                    verbose=False
                )
                response_text = response_texts[0]

                # Log successful interaction
                self._log_llm_interaction(
                    agent_name, system_prompt, user_prompt, response_text
                )

                return response_text
            except Exception as e:
                # Log failed attempt
                error_msg = f"Local LLM direct inference failed: {e}"
                self._log_llm_interaction(
                    agent_name, system_prompt, user_prompt, None, error_msg
                )
                raise Exception(f"Local LLM direct inference failed: {e}")

        if self.provider == 'local':
            return self._call_local_llm(agent_name, system_prompt, user_prompt, max_retries)

        for attempt in range(max_retries + 1):
            try:
                # Handle both max_completion_tokens (GPT-5) and max_tokens (older models)
                max_tokens_param = {}
                if 'max_completion_tokens' in self.model_config:
                    max_tokens_param['max_completion_tokens'] = self.model_config['max_completion_tokens']
                elif 'max_tokens' in self.model_config:
                    max_tokens_param['max_tokens'] = self.model_config['max_tokens']
                else:
                    # Default for newer models
                    max_tokens_param['max_completion_tokens'] = 2048

                # Use provided temperature or fall back to config (default 1.0)
                temp_value = temperature if temperature is not None else self.model_config.get('temperature', 1.0)

                response = self.client.chat.completions.create(
                    model=self.model_config.get('model_name', 'gpt-4o-mini'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp_value,
                    timeout=self.model_config.get('timeout', 30),
                    **max_tokens_param
                )
                response_text = response.choices[0].message.content

                # Track token usage
                if hasattr(response, 'usage') and response.usage is not None:
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                    _GLOBAL_TOKEN_TRACKER.add_tokens(prompt_tokens, completion_tokens, agent_name)

                # Log successful interaction
                self._log_llm_interaction(
                    agent_name, system_prompt, user_prompt, response_text
                )

                return response_text
                
            except openai.BadRequestError as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in ('temperature', 'unsupported_value', 'does not support')):
                    model_name = self.model_config.get('model_name', 'unknown')
                    raise RuntimeError(
                        f"Model '{model_name}' rejected temperature={temp_value}. "
                        f"This model likely only supports temperature=1. "
                        f"Set temperature=1 in your config or remove the temperature parameter. "
                        f"Original error: {e}"
                    ) from e
                # Other 400 errors — still raise immediately, don't waste retries
                raise

            except Exception as e:
                error_msg = f"LLM call attempt {attempt + 1} failed: {e}"

                # Log failed attempt
                self._log_llm_interaction(
                    agent_name, system_prompt, user_prompt, None, error_msg
                )

                if attempt == max_retries:
                    raise Exception(f"LLM call failed after {max_retries} retries: {e}")
                continue

    def _call_local_llm(self, agent_name: str, system_prompt: str, user_prompt: str,
                        max_retries: int) -> str:
        """Call local inference server with retry logic."""
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}" if system_prompt else user_prompt
        max_new_tokens = (
            self.model_config.get('max_completion_tokens')
            or self.model_config.get('max_tokens', 1024)
        )

        payload = {
            'prompts': [prompt],
            'max_new_tokens': max_new_tokens,
            'temperature': self.model_config.get('temperature', 0.7),
            'top_p': self.model_config.get('top_p', 0.9),
            'top_k': self.model_config.get('top_k', 0),
            'repetition_penalty': self.model_config.get('repetition_penalty', 1.0),
            'do_sample': self.model_config.get('do_sample', True),
        }

        headers = {'Content-Type': 'application/json'}

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.local_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.model_config.get('timeout', 30)
                )
                response.raise_for_status()
                data = response.json()
                responses = data.get('responses') or []
                if not responses:
                    raise ValueError('Local inference returned no responses')
                response_text = responses[0]
                self._log_llm_interaction(
                    agent_name, system_prompt, user_prompt, response_text
                )
                return response_text
            except Exception as e:
                error_msg = f"Local LLM call attempt {attempt + 1} failed: {e}"
                self._log_llm_interaction(
                    agent_name, system_prompt, user_prompt, None, error_msg
                )
                if attempt == max_retries:
                    raise Exception(f"Local LLM call failed after {max_retries} retries: {e}")
                continue

    async def _call_llm_async(self, system_prompt: str, user_prompt: str,
                               temperature: float = None) -> str:
        """
        Async version of _call_llm for concurrent batch processing.
        Only works with OpenAI/vLLM providers that have async_client.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Optional temperature override

        Returns:
            LLM response text
        """
        agent_name = self.__class__.__name__

        if self.async_client is None:
            # Fallback to sync call if no async client
            return self._call_llm(system_prompt, user_prompt, temperature=temperature)

        try:
            # Handle both max_completion_tokens and max_tokens
            max_tokens_param = {}
            if 'max_completion_tokens' in self.model_config:
                max_tokens_param['max_completion_tokens'] = self.model_config['max_completion_tokens']
            elif 'max_tokens' in self.model_config:
                max_tokens_param['max_tokens'] = self.model_config['max_tokens']
            else:
                max_tokens_param['max_completion_tokens'] = 2048

            temp_value = temperature if temperature is not None else self.model_config.get('temperature', 1.0)

            response = await self.async_client.chat.completions.create(
                model=self.model_config.get('model_name', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp_value,
                timeout=self.model_config.get('timeout', 300),
                **max_tokens_param
            )
            response_text = response.choices[0].message.content

            # Track token usage
            if hasattr(response, 'usage') and response.usage is not None:
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                _GLOBAL_TOKEN_TRACKER.add_tokens(prompt_tokens, completion_tokens, agent_name)

            return response_text

        except openai.BadRequestError as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ('temperature', 'unsupported_value', 'does not support')):
                model_name = self.model_config.get('model_name', 'unknown')
                raise RuntimeError(
                    f"Model '{model_name}' rejected temperature={temp_value}. "
                    f"This model likely only supports temperature=1. "
                    f"Set temperature=1 in your config or remove the temperature parameter. "
                    f"Original error: {e}"
                ) from e
            raise

        except Exception as e:
            self.logger.error(f"Async LLM call failed: {e}")
            raise

    def _extract_json_cerebras(self, response_text: str) -> Dict:
        """
        Extract JSON from Cerebras LLM response using robust parsing.
        Based on test_cerebras_agents.py parse_private_phrases() logic.
        """
        text = response_text.strip()
        original_text = text

        # Check if response contains a JSON code block
        if "```json" in text:
            start_idx = text.find("```json") + 7
            end_idx = text.find("```", start_idx)
            if end_idx != -1:
                text = text[start_idx:end_idx].strip()
            else:
                text = text[start_idx:].strip()
        elif "```" in text:
            start_idx = text.find("```") + 3
            end_idx = text.find("```", start_idx)
            if end_idx != -1:
                text = text[start_idx:end_idx].strip()
            else:
                text = text[start_idx:].strip()

        # If no code block found, try to find JSON by looking for { and }
        if not text.startswith("{"):
            json_start = text.find("{")
            if json_start != -1:
                text = text[json_start:]
            else:
                # No JSON object found - try to extract from explanatory text
                self.logger.warning(f"⚠️  Cerebras: No JSON object found, attempting text extraction")
                # Look for quoted phrases in the text
                quoted_phrases = re.findall(r'"([^"]+)"', original_text)
                # Filter out likely non-phrase quotes (short, meta-text, etc.)
                phrases = [p for p in quoted_phrases if len(p) > 2 and not p.startswith('private_')]
                if phrases:
                    self.logger.warning(f"⚠️  Cerebras: Extracted {len(phrases)} phrases from text")
                    return {'private_phrases': phrases[:100]}  # Limit to 100
                else:
                    self.logger.warning(f"⚠️  Cerebras: Response (first 500 chars): {original_text[:500]}")
                    return {'private_phrases': []}

        # Find the end of JSON object (matching closing brace)
        if text.startswith("{"):
            brace_count = 0
            json_end = -1
            for i, char in enumerate(text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end != -1:
                text = text[:json_end]

        # Strip JSON comments (// ...) that Cerebras sometimes adds
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove inline comments (e.g., "value", // comment)
            if '//' in line:
                # Check if // is inside a string
                parts = line.split('//')
                if len(parts) >= 2:
                    # Simple heuristic: keep part before // if it has content
                    before_comment = parts[0].rstrip()
                    if before_comment and not before_comment.strip().startswith('//'):
                        cleaned_lines.append(before_comment)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

        # Fix missing commas in JSON arrays (common Cerebras error)
        # Look for patterns like: "value"\n  "value" and add comma
        text = re.sub(r'"\s*\n\s*"', '",\n  "', text)

        # Parse JSON
        try:
            data = json.loads(text)
            # Ensure proper structure
            if not isinstance(data, dict):
                self.logger.warning(f"⚠️  Cerebras: Parsed JSON is not a dict, got {type(data)}")
                return {'private_phrases': []}
            # Ensure private_phrases key exists
            if 'private_phrases' not in data:
                self.logger.warning(f"⚠️  Cerebras: No 'private_phrases' key in parsed JSON")
                return {'private_phrases': []}
            return data
        except json.JSONDecodeError as e:
            self.logger.warning(f"⚠️  Cerebras: Failed to parse JSON: {e}")
            self.logger.warning(f"⚠️  Cerebras: Attempting fallback extraction from malformed JSON")
            # Try to extract phrases from malformed JSON text
            # Look for array content in "private_phrases": [...]
            match = re.search(r'"private_phrases"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if match:
                array_content = match.group(1)
                # Extract quoted strings
                phrases = re.findall(r'"([^"]+)"', array_content)
                if phrases:
                    self.logger.warning(f"⚠️  Cerebras: Extracted {len(phrases)} phrases from malformed JSON")
                    return {'private_phrases': phrases}
            self.logger.warning(f"⚠️  Cerebras: Cleaned text (first 500 chars): {text[:500]}")
            return {'private_phrases': []}

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response."""
        agent_name = self.__class__.__name__

        # Handle case where Cerebras returns a dict directly instead of string
        if isinstance(text, dict):
            self.logger.warning(f"⚠️  Response is already a dict (Cerebras format): {text}")
            return text

        # === CEREBRAS-SPECIFIC PARSING ===
        # If using Cerebras provider, use specialized parsing logic
        if self.provider == 'cerebras':
            self.logger.info(f"🔧 Using Cerebras-specific JSON parsing")
            result = self._extract_json_cerebras(text)
            # Validate result has expected keys based on agent type
            return result

        # === STANDARD PARSING (OpenAI, OpenRouter, Local) ===
        # Log the raw response for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_log_file = self.log_dir / f"{agent_name}_json_debug_{timestamp}.txt"

        with open(debug_log_file, 'w', encoding='utf-8') as f:
            f.write("=== RAW LLM RESPONSE ===\n")
            f.write(repr(text))
            f.write("\n\n=== CLEANED TEXT ===\n")
            f.write(text.strip() if isinstance(text, str) else str(text))
            f.write("\n\n=== JSON EXTRACTION ATTEMPTS ===\n")

        # Clean the text first
        text = text.strip()
        
        # Try to find JSON block with code fence (original OpenAI pattern)
        json_match = re.search(r'```(?:json)?\s*(\{[^}]*\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            with open(debug_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Method 1 (OpenAI code fence) found: {repr(json_str)}\n")
        else:
            # Try to find standalone JSON object (original OpenAI pattern)
            json_match = re.search(r'\{[^{}]*"private_phrases"[^{}]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                with open(debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Method 2 (OpenAI private_phrases) found: {repr(json_str)}\n")
            else:
                # Try Qwen-compatible patterns with nested JSON support
                # Method 3: Code fence with nested JSON (for Qwen responses in code blocks)
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    with open(debug_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Method 3 (Qwen code fence) found: {repr(json_str)}\n")
                else:
                    # Method 4: Plain nested JSON with private_phrases (for Qwen plain responses)
                    json_match = re.search(r'\{.*?"private_phrases".*?\}', text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        with open(debug_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Method 4 (Qwen private_phrases) found: {repr(json_str)}\n")
                    else:
                        # Last resort: look for any JSON-like structure
                        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            with open(debug_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"Method 5 (any JSON) found: {repr(json_str)}\n")
                        else:
                            # No JSON patterns matched — try plain-text fallback
                            with open(debug_log_file, 'a', encoding='utf-8') as f:
                                f.write("No JSON patterns matched, trying plain-text fallback\n")
                                f.write(f"Raw response: {repr(text[:500])}\n")

                            stripped = text.strip().strip('"').strip("'")
                            # Try json.loads on raw text (handles bare numbers, quoted strings)
                            try:
                                raw_val = json.loads(text.strip())
                                with open(debug_log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"Plain-text json.loads succeeded: {repr(raw_val)}\n")
                                return {"_raw_value": raw_val}
                            except (json.JSONDecodeError, ValueError):
                                pass

                            # Short plain text (< 200 chars) → treat as raw answer
                            if len(stripped) < 200 and stripped:
                                with open(debug_log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"Plain-text fallback (short answer): {repr(stripped)}\n")
                                return {"_raw_value": stripped}

                            # Truly unparseable — return empty
                            if "private_phrases" in text.lower():
                                return {"private_phrases": []}
                            else:
                                return {}
        
        # Clean common formatting issues
        json_str = json_str.strip()
        json_str = re.sub(r'^\s*```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```\s*$', '', json_str)
        
        with open(debug_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Final cleaned JSON string: {repr(json_str)}\n")
        
        try:
            result = json.loads(json_str)
            with open(debug_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Successfully parsed JSON: {result}\n")
            return result
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues and truncation
            try:
                # Fix trailing commas
                fixed_json = re.sub(r',\s*}', '}', json_str)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                result = json.loads(fixed_json)
                with open(debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Successfully parsed after fixing trailing commas: {result}\n")
                return result
            except json.JSONDecodeError as e2:
                # Try to repair truncated JSON
                try:
                    repaired_json = self._repair_truncated_json(json_str, debug_log_file)
                    result = json.loads(repaired_json)
                    with open(debug_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Successfully parsed after truncation repair: {result}\n")
                    return result
                except json.JSONDecodeError as e3:
                    # Skip this document - return empty result to continue training
                    with open(debug_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"SKIPPING DOCUMENT: JSON parsing completely failed\n")
                        f.write(f"Original error: {e}\n")
                        f.write(f"Truncation repair error: {e3}\n")
                        f.write(f"Failed JSON string: {repr(json_str)}\n")
                    
                    # Return empty result based on expected format
                    if "private_phrases" in json_str.lower():
                        return {"private_phrases": []}  # Empty annotation result
                    else:
                        return {}  # Empty general result
    
    def _repair_truncated_json(self, json_str: str, debug_log_file: Path) -> str:
        """
        Attempt to repair truncated JSON by adding missing closing characters.
        
        Args:
            json_str: The potentially truncated JSON string
            debug_log_file: Debug log file path
            
        Returns:
            Repaired JSON string
        """
        with open(debug_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Attempting JSON truncation repair...\n")
        
        repaired = json_str.strip()
        
        # Count opening and closing characters
        open_braces = repaired.count('{')
        close_braces = repaired.count('}')
        open_brackets = repaired.count('[')
        close_brackets = repaired.count(']')
        
        # Count unmatched quotes (simple heuristic)
        quote_count = repaired.count('"')
        # If odd number of quotes, we likely have an unclosed string
        if quote_count % 2 == 1:
            repaired += '"'
            with open(debug_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Added closing quote\n")
        
        # Add missing closing brackets
        missing_brackets = open_brackets - close_brackets
        if missing_brackets > 0:
            repaired += ']' * missing_brackets
            with open(debug_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Added {missing_brackets} closing bracket(s)\n")
        
        # Add missing closing braces
        missing_braces = open_braces - close_braces
        if missing_braces > 0:
            repaired += '}' * missing_braces
            with open(debug_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Added {missing_braces} closing brace(s)\n")
        
        with open(debug_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Repaired JSON: {repr(repaired)}\n")
        
        return repaired


class AnnotatorAgent(BaseAgent):
    """Agent that identifies private phrases in text."""

    def __init__(self, model_config: Dict, prompt_template: 'PromptTemplate',
                 output_key: str = 'private_phrases'):
        super().__init__(model_config, prompt_template)
        self.output_key = output_key

    def annotate_without_constitution(self, text: str) -> List[str]:
        """
        Identify private phrases without constitutional guidance.

        Args:
            text: Input text

        Returns:
            List of private phrases
        """
        system_prompt, user_prompt = self.prompt_template.format(
            CONSTITUTION_BLOCK="If no rules are provided, use your best judgement.",
            TEXT_CONTENT=text
        )
        
        response = self._call_llm(system_prompt, user_prompt)
        result = self._extract_json(response)

        return self._extract_output(result)

    def annotate_with_constitution(self, text: str, constitution: str) -> List[str]:
        """
        Identify private phrases using constitutional rules.

        Args:
            text: Input text
            constitution: Constitution text

        Returns:
            List of private phrases
        """
        const_block = (
            f"**CONSTITUTION RULES (follow these to guide your response):**\n{constitution}"
        )
        system_prompt, user_prompt = self.prompt_template.format(
            CONSTITUTION_BLOCK=const_block,
            TEXT_CONTENT=text
        )

        response = self._call_llm(system_prompt, user_prompt)
        result = self._extract_json(response)

        return self._extract_output(result)

    def _extract_output(self, result: Dict) -> List[str]:
        """Extract output from JSON, trying output_key then fallbacks."""
        # Try configured key first
        if self.output_key in result:
            val = result[self.output_key]
            return val if isinstance(val, list) else [str(val)]
        # Fallback: private_phrases (original default)
        if 'private_phrases' in result:
            return result['private_phrases']
        # Fallback: if JSON has exactly one key, use its value
        if len(result) == 1:
            val = next(iter(result.values()))
            return val if isinstance(val, list) else [str(val)]
        return []
    
    def process(self, text: str, constitution: Optional[str] = None) -> List[str]:
        """
        Process text and return output.

        Args:
            text: Input text
            constitution: Optional constitution text

        Returns:
            List of output items
        """
        if constitution:
            return self.annotate_with_constitution(text, constitution)
        else:
            return self.annotate_without_constitution(text)

    def process_with_reasoning(self, text: str, constitution: Optional[str] = None):
        """Process text and return (output, reasoning) tuple.

        Like process() but also extracts the 'reasoning' field from the model's
        JSON response so it can be passed to the error context for the rule proposer.
        """
        const_block = (
            f"**CONSTITUTION RULES (follow these to guide your response):**\n{constitution}"
            if constitution
            else "If no rules are provided, use your best judgement."
        )
        system_prompt, user_prompt = self.prompt_template.format(
            CONSTITUTION_BLOCK=const_block,
            TEXT_CONTENT=text
        )
        response = self._call_llm(system_prompt, user_prompt)
        result = self._extract_json(response)
        output = self._extract_output(result)
        reasoning = result.get('reasoning', '') if isinstance(result, dict) else ''
        return output, reasoning

    def process_batch(
        self,
        texts: List[str],
        constitution: Optional[str] = None,
        timeout: float = 300.0
    ) -> List[List[str]]:
        """
        Process multiple texts in batch using local LLM inference.

        This method is ONLY used when local_llm.enabled=true and local_llm.use_batched=true.
        Otherwise, it falls back to sequential processing via process().

        Args:
            texts: List of input texts to annotate
            constitution: Optional constitution text (same for all texts)
            timeout: Maximum time to wait for batch generation (seconds)

        Returns:
            List of private phrase lists (one per input text)
        """
        # Safety check: if local LLM not available, fall back to sequential
        if not self.local_llm_enabled or self.local_llm_inference is None:
            # Fallback to sequential processing (existing code path)
            return [self.process(text, constitution) for text in texts]

        # Safety check: if batched mode not enabled, fall back to sequential
        if not self.use_batched_local:
            return [self.process(text, constitution) for text in texts]

        # Use local LLM batched inference
        try:
            # Prepare prompts for all texts
            system_prompts = []
            user_prompts = []

            for text in texts:
                const_block = (
                    f"**CONSTITUTION RULES (follow these to guide your response):**\n{constitution}"
                    if constitution
                    else "If no rules are provided, use your best judgement."
                )
                system_prompt, user_prompt = self.prompt_template.format(
                    CONSTITUTION_BLOCK=const_block,
                    TEXT_CONTENT=text
                )
                system_prompts.append(system_prompt)
                user_prompts.append(user_prompt)

            # Call batched inference
            response_texts, metadata = self.local_llm_inference.call_batched(
                system_prompts=system_prompts,
                user_prompts=user_prompts,
                timeout=timeout,
                verbose=True
            )

            # Extract JSON from each response
            results = []
            for i, response_text in enumerate(response_texts):
                try:
                    # Use the same JSON extraction + output fallback as existing code
                    result = self.local_llm_inference._extract_json(response_text)
                    output = self._extract_output(result)
                    results.append(output)
                except Exception as e:
                    # If JSON extraction fails for one response, return empty list for that document
                    self.logger.warning(f"Failed to extract JSON from batch response {i}: {e}")
                    results.append([])

            return results

        except TimeoutError as e:
            # Timeout exceeded, retry once with sequential fallback
            self.logger.warning(f"Batched inference timeout ({timeout}s), retrying with sequential fallback...")
            return [self.process(text, constitution) for text in texts]

        except Exception as e:
            # Any other error, fall back to sequential processing (backward compatible)
            self.logger.warning(f"Batched inference failed: {e}, falling back to sequential processing...")
            return [self.process(text, constitution) for text in texts]

    async def _process_single_async(self, text: str, constitution: Optional[str] = None) -> List[str]:
        """
        Async version of processing a single text.

        Args:
            text: Input text
            constitution: Optional constitution text

        Returns:
            List of private phrases
        """
        const_block = (
            f"**CONSTITUTION RULES (follow these to guide your response):**\n{constitution}"
            if constitution
            else "If no rules are provided, use your best judgement."
        )
        system_prompt, user_prompt = self.prompt_template.format(
            CONSTITUTION_BLOCK=const_block,
            TEXT_CONTENT=text
        )

        response = await self._call_llm_async(system_prompt, user_prompt)
        result = self._extract_json(response)
        return self._extract_output(result)

    async def process_batch_async(
        self,
        texts: List[str],
        constitution: Optional[str] = None,
        max_concurrent: int = 8
    ) -> List[List[str]]:
        """
        Process multiple texts concurrently using async LLM calls.
        This leverages vLLM's internal batching by sending concurrent requests.

        Args:
            texts: List of input texts to annotate
            constitution: Optional constitution text (same for all texts)
            max_concurrent: Maximum concurrent requests (default 8)

        Returns:
            List of private phrase lists (one per input text)
        """
        if not texts:
            return []

        # If no async client available, fall back to sequential
        if self.async_client is None:
            self.logger.warning("No async client available, falling back to sequential processing")
            return [self.process(text, constitution) for text in texts]

        self.logger.info(f"🚀 Processing {len(texts)} texts concurrently (max_concurrent={max_concurrent})")

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(text: str) -> List[str]:
            async with semaphore:
                try:
                    return await self._process_single_async(text, constitution)
                except Exception as e:
                    self.logger.warning(f"Async processing failed for text: {e}")
                    return []

        # Create tasks for all texts
        tasks = [process_with_semaphore(text) for text in texts]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that were returned
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Text {i} failed: {result}")
                final_results.append([])
            else:
                final_results.append(result)

        return final_results

    def process_batch_concurrent(
        self,
        texts: List[str],
        constitution: Optional[str] = None,
        max_concurrent: int = 8
    ) -> List[List[str]]:
        """
        Synchronous wrapper for async batch processing.
        Call this from synchronous code to process texts concurrently.

        Args:
            texts: List of input texts to annotate
            constitution: Optional constitution text (same for all texts)
            max_concurrent: Maximum concurrent requests (default 8)

        Returns:
            List of private phrase lists (one per input text)
        """
        import concurrent.futures
        from functools import partial

        # Use ThreadPoolExecutor to run multiple sync requests truly in parallel
        # This bypasses async complexity and uses thread-level parallelism
        self.logger.info(f"🚀 Processing {len(texts)} texts in parallel threads (max_workers={max_concurrent})")

        def process_single_sync(text: str) -> List[str]:
            """Process a single text synchronously."""
            try:
                return self.process(text, constitution)
            except Exception as e:
                self.logger.warning(f"Thread processing failed: {e}")
                return []

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks at once
            futures = [executor.submit(process_single_sync, text) for text in texts]
            # Wait for all to complete and collect results in order
            for future in futures:
                try:
                    results.append(future.result(timeout=300))
                except Exception as e:
                    self.logger.warning(f"Future failed: {e}")
                    results.append([])

        return results


class DirectLowercaseEvaluator:
    """
    Simple direct lowercase exact matching evaluator.

    Uses deterministic greedy 1:1 matching with case-insensitive exact equality.
    Simpler and faster than bipartite containment matching.
    """

    def evaluate_document(self, predicted: List[str], gold: List[str]) -> Dict[str, Any]:
        """
        Evaluate predicted phrases against gold using lowercase exact matching.

        Args:
            predicted: List of predicted phrases
            gold: List of gold standard phrases

        Returns:
            Dictionary with same structure as PhraseMatchingEvaluator for compatibility
        """
        # Normalize to lowercase for matching
        pred_lower = [p.lower().strip() for p in predicted]
        gold_lower = [g.lower().strip() for g in gold]

        # Track which predictions have been matched
        pred_matched = [False] * len(predicted)
        gold_matched = [False] * len(gold)

        matched_pairs = []  # (pred_original, gold_original)

        # Greedy matching: for each gold, find first matching prediction
        for g_idx, g_norm in enumerate(gold_lower):
            for p_idx, p_norm in enumerate(pred_lower):
                if not pred_matched[p_idx] and p_norm == g_norm:
                    # Found a match
                    matched_pairs.append((predicted[p_idx], gold[g_idx]))
                    pred_matched[p_idx] = True
                    gold_matched[g_idx] = True
                    break

        # Collect unmatched phrases
        unmatched_pred = [predicted[i] for i in range(len(predicted)) if not pred_matched[i]]
        unmatched_gold = [gold[i] for i in range(len(gold)) if not gold_matched[i]]

        # Compute metrics
        tp = len(matched_pairs)
        fp = len(unmatched_pred)
        fn = len(unmatched_gold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Return same structure as PhraseMatchingEvaluator for compatibility
        return {
            'metrics': {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': 0,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'details': {
                'num_predicted': len(predicted),
                'num_gold': len(gold),
                'matched_pairs': matched_pairs,
                'unmatched_predicted': unmatched_pred,
                'unmatched_gold': unmatched_gold,
                'fp_phrases': unmatched_pred,
                'fn_phrases': unmatched_gold
            }
        }




# ============================================================================
# NEW UNIFIED AGENT SYSTEM (Replaces bifurcated ActionDecision + TargetSelection)
# ============================================================================

class NewDecisionAgent(BaseAgent):
    """
    Unified decision agent that outputs action AND rule_index in a single call.
    Replaces the bifurcated ActionDecisionAgent + TargetSelectionAgent pattern.
    Uses prompts/decision_agent.md
    """

    def decide(self, constitution_text: str, fn_count: int, fp_count: int,
               fn_phrases: List[str], fp_phrases: List[str],
               total_rules: int, trend_info: str = "") -> Dict:
        """
        Make unified decision on constitutional change.

        Args:
            constitution_text: Current constitution rules (indexed)
            fn_count: Total count of false negatives
            fp_count: Total count of false positives
            fn_phrases: Sample of FN phrases
            fp_phrases: Sample of FP phrases
            total_rules: Number of rules in constitution
            trend_info: Optional trend information

        Returns:
            Dict with {action: add|edit|remove|no_change, rule_index: int, reasoning: str}
        """
        # Calculate derived metrics
        total_errors = fn_count + fp_count
        fp_per_fn = round(fp_count / fn_count, 2) if fn_count > 0 else float('inf')

        # Format phrases for prompt
        fn_phrases_str = ", ".join(f'"{p}"' for p in fn_phrases) if fn_phrases else "None"
        fp_phrases_str = ", ".join(f'"{p}"' for p in fp_phrases) if fp_phrases else "None"

        # Inject agent memory into trend_info
        memory_text = self.get_memory_text()
        full_trend = (trend_info or "No trend data available")
        if memory_text:
            full_trend += f"\n\n{memory_text}"

        system_prompt, user_prompt = self.prompt_template.format(
            FN_COUNT=fn_count,
            FP_COUNT=fp_count,
            TOTAL_ERRORS=total_errors,
            FP_PER_FN=fp_per_fn,
            RULE_COUNT=total_rules,
            TREND_INFO=full_trend,
            FN_SAMPLE_PHRASES=fn_phrases_str,
            FP_SAMPLE_PHRASES=fp_phrases_str,
            CONSTITUTION_RULES=constitution_text or "No rules defined yet"
        )

        response = self._call_llm(system_prompt, user_prompt)
        result = self._extract_json(response)

        # Validate and normalize response
        if isinstance(result, dict):
            # Ensure action is lowercase
            if 'action' in result:
                result['action'] = str(result['action']).lower()

            # Convert rule_index to int (handle string "-1" or "1" etc.)
            if 'rule_index' in result:
                try:
                    result['rule_index'] = int(result['rule_index'])
                except (ValueError, TypeError):
                    result['rule_index'] = -1

            # Default rule_index for add and no_change actions
            if result.get('action') in ['add', 'no_change'] and result.get('rule_index', -1) == -1:
                result['rule_index'] = -1

        # Record to memory
        action = result.get('action', '?')
        reasoning = result.get('reasoning', '')[:80]
        self.remember(f"Decided {action.upper()} — {reasoning}")

        return result

    def process(self, **kwargs) -> Dict:
        """Process decision request."""
        return self.decide(**kwargs)


class NewRuleProposerAgent(BaseAgent):
    """
    Agent dedicated to creating NEW rules only.
    Uses prompts/rule_proposer.md
    Accepts temperature parameter for retry with escalation.
    """

    def propose_rule(self, constitution_text: str, error_context: str,
                     previous_reasoning: str = "", previous_rejections: List[Dict] = None,
                     temperature: float = None) -> Dict:
        """
        Propose a new rule to add to constitution.

        Args:
            constitution_text: Current constitution text
            error_context: Formatted error context (FN/FP with document context)
            previous_reasoning: Reasoning from decision agent
            previous_rejections: List of previously rejected rules
            temperature: Temperature for LLM call (for retry escalation)

        Returns:
            Dict with {rule_text: str}
        """
        # Format rejection history
        rejection_text = "No previous rejections"
        if previous_rejections:
            rejection_entries = []
            for rej in previous_rejections[-3:]:  # Show last 3 rejections
                rejection_entries.append(
                    f"- Rule '{rej.get('rule_text', 'unknown')[:50]}...' rejected: {rej.get('reason', 'unknown')}"
                )
            rejection_text = "\n".join(rejection_entries)

        # Inject agent memory into rejection context
        memory_text = self.get_memory_text()
        if memory_text:
            rejection_text += f"\n\n{memory_text}"

        system_prompt, user_prompt = self.prompt_template.format(
            PREVIOUS_REASONING=previous_reasoning or "No previous analysis available",
            CONSTITUTION_TEXT=constitution_text or "No rules defined yet",
            ERROR_CONTEXT=error_context or "No error context provided",
            PREVIOUS_REJECTIONS=rejection_text
        )

        response = self._call_llm(system_prompt, user_prompt, temperature=temperature)
        result = self._extract_json(response)

        # Ensure rule_text exists
        if isinstance(result, dict) and 'rule_text' not in result:
            for key in ['rule', 'new_rule', 'text']:
                if key in result:
                    result['rule_text'] = result[key]
                    break

        # Record to memory
        rule_text = result.get('rule_text', '')[:60]
        self.remember(f"Proposed: \"{rule_text}...\"")

        return result

    def process(self, **kwargs) -> Dict:
        """Process rule proposal request."""
        return self.propose_rule(**kwargs)


class NewRuleEditorAgent(BaseAgent):
    """
    Agent dedicated to EDITING existing rules only.
    Uses prompts/rule_editor.md
    Takes rule_index from decision agent and edits that specific rule.
    Accepts temperature parameter for retry with escalation.
    """

    def edit_rule(self, rule_index: int, constitution_text: str, error_context: str,
                  previous_reasoning: str = "", previous_rejections: List[Dict] = None,
                  temperature: float = None) -> Dict:
        """
        Edit an existing rule in the constitution.

        Args:
            rule_index: Index of rule to edit (from decision agent)
            constitution_text: Current constitution text
            error_context: Formatted error context (FN/FP with document context)
            previous_reasoning: Reasoning from decision agent
            previous_rejections: List of previously rejected edits
            temperature: Temperature for LLM call (for retry escalation)

        Returns:
            Dict with {rule_text: str}
        """
        # Format rejection history
        rejection_text = "No previous rejections"
        if previous_rejections:
            rejection_entries = []
            for rej in previous_rejections[-3:]:
                rejection_entries.append(
                    f"- Edit '{rej.get('rule_text', 'unknown')[:50]}...' rejected: {rej.get('reason', 'unknown')}"
                )
            rejection_text = "\n".join(rejection_entries)

        # Rule index is 0-based internally but 1-based in prompt for readability
        display_rule_index = rule_index + 1

        # Inject agent memory into rejection context
        memory_text = self.get_memory_text()
        if memory_text:
            rejection_text += f"\n\n{memory_text}"

        system_prompt, user_prompt = self.prompt_template.format(
            RULE_NUMBER_TO_EDIT=display_rule_index,
            CONSTITUTION_TEXT=constitution_text or "No rules defined yet",
            PREVIOUS_REASONING=previous_reasoning or "No previous analysis available",
            ERROR_CONTEXT=error_context or "No error context provided",
            PREVIOUS_REJECTIONS=rejection_text
        )

        response = self._call_llm(system_prompt, user_prompt, temperature=temperature)
        result = self._extract_json(response)

        # Ensure rule_text exists
        if isinstance(result, dict) and 'rule_text' not in result:
            for key in ['rule', 'edited_rule', 'new_rule', 'text']:
                if key in result:
                    result['rule_text'] = result[key]
                    break

        # Record to memory
        rule_text = result.get('rule_text', '')[:60]
        self.remember(f"Edited rule #{display_rule_index}: \"{rule_text}...\"")

        return result

    def process(self, **kwargs) -> Dict:
        """Process rule editing request."""
        return self.edit_rule(**kwargs)


# ============================================================================
# END OF NEW UNIFIED AGENT SYSTEM
# ============================================================================




def create_agents(config: Dict, prompts_dir: Path, quiet: bool = False) -> Dict[str, Any]:
    """
    Factory function to create all agents.

    Args:
        config: Configuration dictionary
        prompts_dir: Path to prompts directory
        quiet: Suppress print output (when Rich display active)

    Returns:
        Dict of agent_name -> agent instance
    """
    model_config = config['model']
    mac_model_config = config.get('mac_model', model_config)

    # LoRA agent configuration (for fine-tuned models)
    agent_models = config.get('agent_models', {})
    use_lora = agent_models.get('use_lora_agents', False)

    def get_agent_model_config(agent_class_name: str, base: Dict = None) -> Dict:
        """
        Get model config for a specific agent.

        Args:
            agent_class_name: Agent class for LoRA lookup.
            base: Base model config to use (default: model_config).
                  Pass mac_model_config for MAC agents.
        """
        base_cfg = base if base is not None else model_config
        if not use_lora:
            return base_cfg

        agent_override = agent_models.get(agent_class_name, {})
        if not agent_override or 'model_name' not in agent_override:
            return base_cfg

        merged = {**base_cfg}
        merged['model_name'] = agent_override['model_name']
        return merged

    # Load prompt templates
    prompt_files = {
        'annotator': 'annotator.md',
        'new_decision_agent': 'decision_agent.md',
        'new_rule_proposer': 'rule_proposer.md',
        'new_rule_editor': 'rule_editor.md',
    }

    templates = {}
    for name, filename in prompt_files.items():
        with open(prompts_dir / filename, 'r') as f:
            content = f.read()
            # Split system and user prompts
            if 'System:' in content and 'User:' in content:
                parts = content.split('User:', 1)
                system = parts[0].replace('System:', '').strip()
                user = parts[1].strip()
            else:
                system = ""
                user = content
            templates[name] = PromptTemplate(system=system, user=user)
    
    # Log LoRA status
    if use_lora:
        logger.info("[create_agents] LoRA agents ENABLED - using fine-tuned models")
    else:
        logger.info("[create_agents] LoRA agents disabled - using base model for all agents")

    # Configurable annotator output key (default: 'private_phrases' for backward compat)
    annotator_output_key = config.get('annotator', {}).get('output_key', 'private_phrases')

    # Single annotator template — constitution is injected via {{CONSTITUTION_BLOCK}}.
    # When no rules exist, CONSTITUTION_BLOCK = "". When rules exist, it contains them.
    # This ensures one prompt adaptation and clean signal isolation.
    task_prompt = config.get('task_prompt', '')
    if task_prompt:
        if '{{CONSTITUTION_BLOCK}}' not in task_prompt:
            raise ValueError(
                "task_prompt must contain {{CONSTITUTION_BLOCK}} placeholder.\n"
                "MAC injects learned constitution rules at this location.\n"
                "Example:\n"
                "  task_prompt='You are a math solver.\\n\\n{{CONSTITUTION_BLOCK}}\\n\\n"
                "Return JSON: {\"answer\": 42}'"
            )
        # Support User: section split if user included it
        if 'User:' in task_prompt:
            parts = task_prompt.split('User:', 1)
            annotator_tmpl = PromptTemplate(system=parts[0].strip(), user=parts[1].strip())
        else:
            annotator_tmpl = PromptTemplate(system=task_prompt, user="Text:\n{{TEXT_CONTENT}}")
    else:
        annotator_tmpl = templates['annotator']

    # Pre-substitute {{OUTPUT_KEY}} in annotator template
    annotator_tmpl.system = annotator_tmpl.system.replace('{{OUTPUT_KEY}}', annotator_output_key)
    annotator_tmpl.user = annotator_tmpl.user.replace('{{OUTPUT_KEY}}', annotator_output_key)

    # Create agents
    # For fine-tuned agents, use get_agent_model_config() which respects use_lora_agents toggle
    agents = {
        'annotator': AnnotatorAgent(get_agent_model_config('AnnotatorAgent'), annotator_tmpl, output_key=annotator_output_key),
        'new_decision_agent': NewDecisionAgent(get_agent_model_config('NewDecisionAgent', mac_model_config), templates['new_decision_agent']),
        'new_rule_proposer': NewRuleProposerAgent(get_agent_model_config('NewRuleProposerAgent', mac_model_config), templates['new_rule_proposer']),
        'new_rule_editor': NewRuleEditorAgent(get_agent_model_config('NewRuleEditorAgent', mac_model_config), templates['new_rule_editor']),
    }

    return agents
