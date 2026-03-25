from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal


# --- Content block types ---

class ClaudeContentBlockText(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None
    citations: Optional[List[Dict[str, Any]]] = None


class ClaudeContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]  # base64 or url source
    cache_control: Optional[Dict[str, Any]] = None


class ClaudeContentBlockDocument(BaseModel):
    type: Literal["document"]
    source: Dict[str, Any]  # base64, url, text, or content source
    title: Optional[str] = None
    context: Optional[str] = None
    citations: Optional[Dict[str, Any]] = None
    cache_control: Optional[Dict[str, Any]] = None


class ClaudeContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: Optional[Dict[str, Any]] = None
    caller: Optional[Dict[str, Any]] = None


class ClaudeContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None
    is_error: Optional[bool] = None
    cache_control: Optional[Dict[str, Any]] = None


class ClaudeContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str


class ClaudeContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: str


class ClaudeContentBlockServerToolUse(BaseModel):
    type: Literal["server_tool_use"]
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: Optional[Dict[str, Any]] = None
    caller: Optional[Dict[str, Any]] = None


class ClaudeContentBlockSearchResult(BaseModel):
    type: Literal["search_result"]
    title: Optional[str] = None
    source: Optional[str] = None
    content: Optional[List[Dict[str, Any]]] = None
    citations: Optional[Dict[str, Any]] = None
    cache_control: Optional[Dict[str, Any]] = None


class ClaudeContentBlockContainerUpload(BaseModel):
    type: Literal["container_upload"]
    file_id: str
    cache_control: Optional[Dict[str, Any]] = None


# Union of all content block types that can appear in messages
ContentBlock = Union[
    ClaudeContentBlockText,
    ClaudeContentBlockImage,
    ClaudeContentBlockDocument,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockThinking,
    ClaudeContentBlockRedactedThinking,
    ClaudeContentBlockServerToolUse,
    ClaudeContentBlockSearchResult,
    ClaudeContentBlockContainerUpload,
]


# --- System content ---

class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


# --- Messages ---

class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Optional[Union[str, List[ContentBlock]]] = None


# --- Tools ---

class ClaudeTool(BaseModel):
    """Accepts both custom tools and server tools.
    Server tools (web_search_20260209, code_execution_20260120, etc.)
    use `type` to identify and have varying fields — we accept them
    via the catch-all `model_config` allowing extra fields.
    """
    type: Optional[str] = None  # "custom", "web_search_20260209", etc.
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    cache_control: Optional[Dict[str, Any]] = None
    defer_loading: Optional[bool] = None
    strict: Optional[bool] = None
    eager_input_streaming: Optional[bool] = None
    allowed_callers: Optional[List[str]] = None
    input_examples: Optional[List[Dict[str, Any]]] = None
    # Server tool specific fields (web_search, web_fetch, etc.)
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    max_uses: Optional[int] = None
    user_location: Optional[Dict[str, Any]] = None
    max_content_tokens: Optional[int] = None
    citations: Optional[Dict[str, Any]] = None
    use_cache: Optional[bool] = None
    max_characters: Optional[int] = None

    model_config = {"extra": "allow"}


# --- Thinking ---

class ClaudeThinkingConfig(BaseModel):
    """Extended thinking configuration.
    - type: "enabled" (manual), "adaptive" (Claude decides), "disabled"
    - budget_tokens: required when type="enabled", max tokens for thinking
    - display: "summarized" or "omitted"
    """
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None
    display: Optional[Literal["summarized", "omitted"]] = None


# --- Output config ---

class ClaudeOutputConfig(BaseModel):
    effort: Optional[Literal["low", "medium", "high", "max"]] = None
    format: Optional[Dict[str, Any]] = None  # json_schema format


# --- Main request ---

class ClaudeMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    # New API fields
    cache_control: Optional[Dict[str, Any]] = None
    service_tier: Optional[str] = None
    inference_geo: Optional[str] = None
    container: Optional[str] = None
    output_config: Optional[ClaudeOutputConfig] = None

    model_config = {"extra": "allow"}


# --- Token counting ---

class ClaudeTokenCountRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}
