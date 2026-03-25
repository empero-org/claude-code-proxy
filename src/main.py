from fastapi import FastAPI
from src.api.endpoints import router as api_router
import uvicorn
import sys
from src.core.config import config

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="0.1.0")

app.include_router(api_router)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Claude-to-OpenAI API Proxy v0.1.0")
        print()
        print("Usage: python src/main.py")
        print()
        print("Required environment variables:")
        print("  OPENAI_API_KEY - Your OpenAI-compatible API key")
        print()
        print("Optional environment variables:")
        print("  ANTHROPIC_API_KEY - Expected API key for client validation")
        print("                      If set, clients must provide this exact key")
        print(
            f"  OPENAI_BASE_URL - Backend API base URL (default: https://api.openai.com/v1)"
        )
        print(f"  AZURE_API_VERSION - Azure OpenAI API version (enables Azure mode)")
        print(f"  BIG_MODEL - Model for opus requests (default: gpt-4o)")
        print(f"  MIDDLE_MODEL - Model for sonnet requests (default: same as BIG_MODEL)")
        print(f"  SMALL_MODEL - Model for haiku requests (default: gpt-4o-mini)")
        print(f"  HOST - Server host (default: 0.0.0.0)")
        print(f"  PORT - Server port (default: 8082)")
        print(f"  LOG_LEVEL - Logging level (default: INFO)")
        print(f"  MAX_TOKENS_LIMIT - Max output token limit (default: 128000)")
        print(f"  MIN_TOKENS_LIMIT - Min output token limit (default: 1024)")
        print(f"  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)")
        print()
        print("Model mapping:")
        print(f"  Claude haiku models  -> {config.small_model}")
        print(f"  Claude sonnet models -> {config.middle_model}")
        print(f"  Claude opus models   -> {config.big_model}")
        print()
        print("Supports: Claude Opus 4.6, Sonnet 4.6, Haiku 4.5, and all older variants.")
        print("Handles: extended thinking, document blocks, tool use, streaming,")
        print("         cache_control, output_config, and server tool passthrough.")
        sys.exit(0)

    # Configuration summary
    print("Claude-to-OpenAI API Proxy v0.1.0")
    print(f"Configuration loaded successfully")
    print(f"   OpenAI Base URL: {config.openai_base_url}")
    print(f"   Big Model (opus):   {config.big_model}")
    print(f"   Middle Model (sonnet): {config.middle_model}")
    print(f"   Small Model (haiku):  {config.small_model}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print(f"   Server: {config.host}:{config.port}")
    print(
        f"   Client API Key Validation: {'Enabled' if config.anthropic_api_key else 'Disabled'}"
    )
    print()

    # Parse log level
    log_level = config.log_level.split()[0].lower()
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if log_level not in valid_levels:
        log_level = "info"

    # Start server
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
