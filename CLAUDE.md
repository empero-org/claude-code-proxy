# Claude Code Proxy — Development Guide

## What this project is

A FastAPI proxy that accepts Anthropic Messages API requests and translates them to OpenAI chat completions format. It allows Claude Code (or any Anthropic SDK client) to use OpenAI-compatible backends.

## Architecture

```
Request flow:  Claude API request → endpoints.py → request_converter.py → OpenAI client
Response flow: OpenAI response → response_converter.py → Claude API response (SSE or JSON)
```

Key modules:
- `src/models/claude.py` — Pydantic models matching the Anthropic Messages API
- `src/conversion/request_converter.py` — Claude → OpenAI translation
- `src/conversion/response_converter.py` — OpenAI → Claude translation (including streaming)
- `src/core/model_manager.py` — Maps claude model names to backend models + auto-detects token limits
- `src/core/client.py` — Async OpenAI client with request cancellation

## Running locally

```bash
OPENAI_API_KEY=your-key python start_proxy.py
```

## Important patterns

- Thinking/redacted_thinking blocks in conversation history are **stripped** before forwarding to OpenAI — the backend doesn't understand them.
- Server tools (web_search, code_execution) are **accepted but not forwarded** — they're Anthropic-specific.
- Document blocks are **converted to text** for OpenAI.
- The proxy queries `/v1/models` on first request to detect max output token limits per model.
- All Pydantic models use `extra="allow"` for forward compatibility with new API fields.

## Commit messages

- Use conventional commits: `feat()`, `fix()`, `docs()`, etc.
- Do not include Co-Authored-By lines.
- Do not include Claude Code branding lines.
