import logging
from typing import Dict, Optional

from src.core.config import config

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config):
        self.config = config
        # Cache of model_id -> model info from the backend
        self._model_cache: Dict[str, dict] = {}
        self._cache_loaded = False

    async def load_models(self, openai_client) -> None:
        """Fetch the model list from the backend and cache capabilities."""
        if self._cache_loaded:
            return
        try:
            models = await openai_client.client.models.list()
            for model in models.data:
                info = model.model_dump() if hasattr(model, "model_dump") else {}
                self._model_cache[model.id] = info
            self._cache_loaded = True
            logger.info(
                f"Loaded {len(self._model_cache)} models from backend: "
                f"{', '.join(sorted(self._model_cache.keys())[:10])}"
                f"{'...' if len(self._model_cache) > 10 else ''}"
            )
        except Exception as e:
            logger.warning(f"Could not fetch models from backend: {e}. "
                           f"Will pass through max_tokens without clamping.")
            self._cache_loaded = True  # Don't retry on every request

    def get_max_output_tokens(self, openai_model: str) -> Optional[int]:
        """Return the max output tokens for a model, or None if unknown.

        Checks common fields that OpenAI-compatible backends expose:
        - max_completion_tokens  (OpenAI)
        - max_output_tokens      (some providers)
        - max_tokens             (fallback, often means context window)
        """
        info = self._model_cache.get(openai_model, {})
        if not info:
            return None

        # Try specific output token fields first
        for field in ("max_completion_tokens", "max_output_tokens"):
            val = info.get(field)
            if val and isinstance(val, int) and val > 0:
                return val

        # Some backends only expose a generic max_tokens —
        # don't use it as output limit since it's often context window size
        return None

    def resolve_max_tokens(self, requested: int, openai_model: str) -> int:
        """Determine the max_tokens to send to the backend.

        Priority:
        1. If the backend reported a model-specific limit, clamp to that.
        2. If the user set MAX_TOKENS_LIMIT env var, clamp to that.
        3. Otherwise, pass through whatever the client requested.
        """
        model_limit = self.get_max_output_tokens(openai_model)
        if model_limit:
            return min(requested, model_limit)

        # Fall back to config limit (user override or default)
        return min(requested, self.config.max_tokens_limit)

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to OpenAI model names based on BIG/MIDDLE/SMALL pattern.

        Recognises all current Claude model naming conventions:
        - claude-opus-4-6, claude-opus-4-5-20251101, claude-opus-4-20250514, ...
        - claude-sonnet-4-6, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, ...
        - claude-haiku-4-5-20251001, claude-3-haiku-20240307, ...
        """
        # If it's already an OpenAI-compatible model, return as-is
        if claude_model.startswith(("gpt-", "o1-", "o3-", "o4-")):
            return claude_model

        # OpenRouter namespaced models (e.g. openai/gpt-5.4, google/gemini-2.5-pro)
        if "/" in claude_model:
            return claude_model

        # xAI / Grok models
        if claude_model.startswith(("grok-", "xai-")):
            return claude_model

        # Other supported provider models, return as-is
        if claude_model.startswith(("ep-", "doubao-", "deepseek-", "gemini-",
                                     "mistral-", "llama-", "qwen-")):
            return claude_model

        # Map based on model naming patterns (keyword matching)
        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return self.config.small_model
        elif "sonnet" in model_lower:
            return self.config.middle_model
        elif "opus" in model_lower:
            return self.config.big_model
        else:
            # Default to big model for unknown models
            return self.config.big_model


model_manager = ModelManager(config)
