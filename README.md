# Claude Code Proxy v0.1

A high-fidelity proxy that lets **Claude Code** (and any Anthropic API client) talk to OpenAI-compatible backends. Complete rewrite for the 2026 Anthropic API — supports extended thinking, document blocks, streaming tool use, cache control, and every current content type.

**Author:** L. Lehmann ([@kodee2k](https://github.com/kodee2k)) — [Empero AI](https://empero.ai)

> Originally inspired by [fuergaosi233/claude-code-proxy](https://github.com/fuergaosi233/claude-code-proxy).
> This is a ground-up rewrite with full support for the current Anthropic Messages API.

---

## What it does

```
Claude Code / Anthropic SDK          This Proxy               Any OpenAI-compatible backend
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  POST /v1/messages   │───▶│  Validate & convert  │───▶│  POST /chat/         │
│  (Anthropic format)  │◀───│  Claude ↔ OpenAI     │◀───│  completions         │
│  Streaming SSE       │    │  Auto model limits   │    │  (OpenAI format)     │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

## Features

- **Full 2026 Anthropic API support** — extended thinking (`enabled`/`adaptive`/`disabled`), `budget_tokens`, `display` modes
- **All content block types** — text, image (base64 + URL), document (PDF/plaintext), tool_use, tool_result, thinking, redacted_thinking, server_tool_use, search_result
- **Streaming** — real-time SSE with incremental `partial_json` tool argument deltas
- **Auto model limits** — queries the backend `/v1/models` endpoint on startup, clamps `max_tokens` to what the backend actually supports
- **Smart model mapping** — `opus` → BIG_MODEL, `sonnet` → MIDDLE_MODEL, `haiku` → SMALL_MODEL
- **Provider passthrough** — GPT, o1/o3/o4, DeepSeek, Gemini, Mistral, LLaMA, Qwen models forwarded as-is
- **Client auth** — optional API key gating via `ANTHROPIC_API_KEY`
- **Azure support** — set `AZURE_API_VERSION` to enable Azure OpenAI mode
- **Custom headers** — inject arbitrary headers via `CUSTOM_HEADER_*` env vars
- **Cache token tracking** — maps OpenAI `cached_tokens` → Claude `cache_read_input_tokens`
- **Request cancellation** — client disconnect cancels the upstream request
- **Forward-compatible** — `extra="allow"` on models, unknown fields pass through without 422

## Quick start

### 1. Install

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Run

```bash
python start_proxy.py

# Or with uv
uv run claude-code-proxy

# Or Docker
docker compose up -d
```

### 4. Connect Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

If you set `ANTHROPIC_API_KEY` on the proxy, pass the same key:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="your-key" claude
```

## Configuration

All config is via environment variables (auto-loaded from `.env`).

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for your backend provider |

### Model mapping

| Variable | Maps from | Default |
|----------|-----------|---------|
| `BIG_MODEL` | Claude opus models | `gpt-4o` |
| `MIDDLE_MODEL` | Claude sonnet models | same as `BIG_MODEL` |
| `SMALL_MODEL` | Claude haiku models | `gpt-4o-mini` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | _(none)_ | If set, clients must send this exact key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Backend API URL |
| `AZURE_API_VERSION` | _(none)_ | Set to enable Azure OpenAI mode |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8082` | Server port |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `MAX_TOKENS_LIMIT` | `128000` | Fallback max output tokens (used when backend doesn't report model limits) |
| `MIN_TOKENS_LIMIT` | `1024` | Floor for max_tokens (thinking requires ≥ 1024) |
| `REQUEST_TIMEOUT` | `90` | Request timeout in seconds |

### Provider examples

**OpenAI**
```bash
OPENAI_API_KEY="sk-..."
BIG_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
```

**Azure OpenAI**
```bash
OPENAI_API_KEY="your-azure-key"
OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
AZURE_API_VERSION="2024-12-01-preview"
BIG_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
```

**Ollama (local)**
```bash
OPENAI_API_KEY="dummy"
OPENAI_BASE_URL="http://localhost:11434/v1"
BIG_MODEL="llama3.1:70b"
SMALL_MODEL="llama3.1:8b"
```

**DeepSeek**
```bash
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.deepseek.com/v1"
BIG_MODEL="deepseek-chat"
SMALL_MODEL="deepseek-chat"
```

## Project structure

```
claude-code-proxy/
├── src/
│   ├── main.py                          # FastAPI app + uvicorn entry
│   ├── api/
│   │   └── endpoints.py                 # /v1/messages, /health, /test-connection
│   ├── core/
│   │   ├── config.py                    # Environment config
│   │   ├── client.py                    # Async OpenAI client with cancellation
│   │   ├── model_manager.py             # Model mapping + auto limit detection
│   │   ├── constants.py                 # All Anthropic API constants
│   │   └── logging.py                   # Logging setup
│   ├── conversion/
│   │   ├── request_converter.py         # Claude → OpenAI request translation
│   │   └── response_converter.py        # OpenAI → Claude response + streaming
│   └── models/
│       └── claude.py                    # Pydantic models for full Anthropic API
├── start_proxy.py                       # Convenience entry point
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## API compatibility

### Supported Anthropic API features

| Feature | Status |
|---------|--------|
| `/v1/messages` (POST) | Fully supported |
| `/v1/messages/count_tokens` (POST) | Estimation-based |
| Streaming (SSE) | Full support with incremental tool deltas |
| Extended thinking (`enabled`/`adaptive`/`disabled`) | Accepted (stripped for OpenAI backends) |
| Thinking/redacted_thinking in history | Stripped transparently |
| Document blocks (PDF, plaintext) | Converted to text |
| Image blocks (base64 + URL) | Converted to OpenAI image_url |
| Tool use + tool results | Full bidirectional conversion |
| Server tools (web_search, code_execution, etc.) | Accepted, not forwarded |
| `cache_control` | Accepted, cache tokens mapped |
| `service_tier`, `output_config`, `container` | Accepted passthrough |
| `tool_choice` (auto/any/none/tool) | Mapped to OpenAI equivalents |
| Client disconnection → request cancellation | Supported |

### Model routing

Any model name containing `opus`, `sonnet`, or `haiku` is routed to the corresponding backend model. All other model names (GPT, DeepSeek, Gemini, etc.) pass through unchanged.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Main messages endpoint (Claude API format) |
| `/v1/messages/count_tokens` | POST | Token count estimation |
| `/health` | GET | Health check |
| `/test-connection` | GET | Test backend API connectivity |
| `/` | GET | Proxy info and config summary |

## License

MIT License — see [LICENSE](LICENSE).

---

Built with FastAPI + Python. Maintained by [Empero AI](https://empero.ai).
