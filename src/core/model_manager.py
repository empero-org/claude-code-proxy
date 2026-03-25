from src.core.config import config


class ModelManager:
    def __init__(self, config):
        self.config = config

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

        # If it's other supported provider models, return as-is
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
