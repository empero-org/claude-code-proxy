import json
import logging
from typing import Dict, Any, List, Optional

from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage
from src.core.config import config
from src.conversion.prompt_compressor import compact_system_prompt, summarize_system_prompt

logger = logging.getLogger(__name__)


async def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest,
    model_manager,
    openai_client=None,
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            system_text = system_text.strip()

            # Apply prompt compression if configured
            system_text = await _compress_system_prompt(
                system_text, openai_model, openai_client
            )

            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages):
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        hasattr(block, "type")
                        and block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                    )
                ):
                    i += 1  # Skip to tool result message

                    # Extract tool results as OpenAI tool messages
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)

                    # Also extract any non-tool-result text from the same
                    # message so context isn't silently dropped (e.g. the
                    # compact summary or user text mixed with tool results).
                    extra_text = _extract_non_tool_text(next_msg)
                    if extra_text:
                        openai_messages.append(
                            {"role": Constants.ROLE_USER, "content": extra_text}
                        )

        i += 1

    # For non-Claude models handling compaction requests, append a format
    # enforcer so the model actually produces the <analysis>/<summary> tags
    # that Claude Code expects to parse from the response.
    if _is_compaction_request(system_text if claude_request.system else "", openai_messages):
        if not openai_model.startswith("claude"):
            _append_compaction_enforcer(openai_messages)
            logger.info(f"Compaction request detected — added format enforcer for {openai_model}")

    # Build OpenAI request — resolve max_tokens from backend model limits
    resolved_max_tokens = model_manager.resolve_max_tokens(
        claude_request.max_tokens, openai_model
    )
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": resolved_max_tokens,
        "stream": claude_request.stream,
    }

    # Only set temperature if not using extended thinking
    # (some backends don't allow temperature with reasoning models)
    if claude_request.temperature is not None:
        openai_request["temperature"] = claude_request.temperature

    logger.debug(
        f"Converted Claude request to OpenAI format: model={openai_model}, "
        f"max_tokens={openai_request['max_tokens']}, stream={openai_request['stream']}"
    )

    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools (only custom tools are forwarded; server tools are Anthropic-specific)
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            # Only forward custom tools (or tools without a type, for backwards compat)
            tool_type = getattr(tool, "type", None)
            if tool_type is None or tool_type == "custom":
                if tool.name and tool.name.strip() and tool.input_schema:
                    openai_tools.append(
                        {
                            "type": Constants.TOOL_FUNCTION,
                            Constants.TOOL_FUNCTION: {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.input_schema,
                            },
                        }
                    )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "required"
        elif choice_type == "none":
            openai_request["tool_choice"] = "none"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        block_type = getattr(block, "type", None)

        if block_type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})

        elif block_type == Constants.CONTENT_IMAGE:
            source = block.source
            if isinstance(source, dict):
                if source.get("type") == "base64" and "media_type" in source and "data" in source:
                    openai_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source['media_type']};base64,{source['data']}"
                            },
                        }
                    )
                elif source.get("type") == "url" and "url" in source:
                    openai_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": source["url"]},
                        }
                    )

        elif block_type == Constants.CONTENT_DOCUMENT:
            # Convert document to text representation for OpenAI
            source = block.source
            doc_text = ""
            title = getattr(block, "title", None)
            context = getattr(block, "context", None)

            if isinstance(source, dict):
                if source.get("type") == "text":
                    doc_text = source.get("data", "")
                elif source.get("type") == "content":
                    doc_text = str(source.get("content", ""))
                elif source.get("type") in ("base64", "url"):
                    # Binary PDF / URL — we can't convert these for OpenAI
                    doc_text = f"[Document: {title or 'untitled'} — binary content not convertible]"

            parts = []
            if title:
                parts.append(f"Document: {title}")
            if context:
                parts.append(f"Context: {context}")
            if doc_text:
                parts.append(doc_text)

            if parts:
                openai_content.append({"type": "text", "text": "\n".join(parts)})

        elif block_type == Constants.CONTENT_TOOL_RESULT:
            # Tool results in user messages that aren't part of the assistant→tool flow
            # Convert to text
            content_str = parse_tool_result_content(getattr(block, "content", None))
            openai_content.append({"type": "text", "text": content_str})

        elif block_type == Constants.CONTENT_SEARCH_RESULT:
            # Convert search results to text
            parts = []
            title = getattr(block, "title", None)
            source = getattr(block, "source", None)
            if title:
                parts.append(f"Search Result: {title}")
            if source:
                parts.append(f"Source: {source}")
            content_list = getattr(block, "content", None)
            if content_list and isinstance(content_list, list):
                for item in content_list:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
            if parts:
                openai_content.append({"type": "text", "text": "\n".join(parts)})

        elif block_type in (Constants.CONTENT_CONTAINER_UPLOAD,):
            # Skip — not convertible to OpenAI
            pass

        # Thinking and redacted_thinking in user messages — skip silently
        elif block_type in (Constants.CONTENT_THINKING, Constants.CONTENT_REDACTED_THINKING):
            pass

    if not openai_content:
        return {"role": Constants.ROLE_USER, "content": ""}
    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format.
    Strips thinking/redacted_thinking blocks (OpenAI doesn't support them)."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        block_type = getattr(block, "type", None)

        if block_type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block_type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )
        elif block_type in (
            Constants.CONTENT_THINKING,
            Constants.CONTENT_REDACTED_THINKING,
        ):
            # Strip thinking blocks — not supported by OpenAI
            pass
        elif block_type == Constants.CONTENT_SERVER_TOOL_USE:
            # Strip server tool use — Anthropic-specific
            pass

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            block_type = getattr(block, "type", None)
            if block_type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(getattr(block, "content", None))
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except Exception:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    try:
        return str(content)
    except Exception:
        return "Unparseable content"


def _is_compaction_request(system_text: str, messages: list) -> bool:
    """Detect whether this request is a Claude Code compaction/summary call."""
    # Primary signal: the compaction system prompt
    if "summarizing conversations" in system_text.lower():
        return True
    # Fallback: check last user message for compaction keywords
    for msg in reversed(messages):
        if msg.get("role") == Constants.ROLE_USER:
            content = msg.get("content", "")
            if isinstance(content, str) and "<summary>" in content and "<analysis>" in content:
                return True
            break
    return False


_COMPACTION_ENFORCER = (
    "\n\n---\n"
    "IMPORTANT — OUTPUT FORMAT REQUIREMENTS:\n"
    "You MUST structure your entire response as two XML-tagged blocks and nothing else:\n\n"
    "1. <analysis> — Your detailed chronological walkthrough of the conversation.\n"
    "2. <summary> — The final structured summary with these numbered sections:\n"
    "   1. Primary Request and Intent\n"
    "   2. Key Technical Concepts\n"
    "   3. Files and Code Sections (include exact file paths and relevant snippets)\n"
    "   4. Errors and Fixes\n"
    "   5. Problem Solving\n"
    "   6. All User Messages (preserve every user instruction, not tool results)\n"
    "   7. Pending Tasks\n"
    "   8. Current Work (be very specific — file names, line numbers, code)\n"
    "   9. Optional Next Step\n\n"
    "Do NOT call any tools. Do NOT wrap your response in markdown code fences.\n"
    "Start your response with <analysis> immediately."
)


def _append_compaction_enforcer(messages: list) -> None:
    """Append format enforcement to the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == Constants.ROLE_USER:
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = content + _COMPACTION_ENFORCER
            elif isinstance(content, list):
                msg["content"].append({"type": "text", "text": _COMPACTION_ENFORCER})
            return


def _extract_non_tool_text(msg: ClaudeMessage) -> str:
    """Extract text/document content from a user message, ignoring tool_result blocks."""
    parts = []
    if not isinstance(msg.content, list):
        return ""
    for block in msg.content:
        block_type = getattr(block, "type", None)
        if block_type == Constants.CONTENT_TEXT:
            parts.append(block.text)
        elif block_type == Constants.CONTENT_DOCUMENT:
            source = getattr(block, "source", None) or {}
            if isinstance(source, dict) and source.get("type") == "text":
                parts.append(source.get("data", ""))
    return "\n".join(parts).strip()


async def _compress_system_prompt(
    system_text: str,
    openai_model: str,
    openai_client=None,
) -> str:
    """Apply prompt compression based on PROMPT_COMPRESSION config."""
    mode = config.prompt_compression
    if mode == "none":
        return system_text

    budget = config.prompt_max_system_tokens

    if mode == "compact":
        return compact_system_prompt(system_text, max_tokens=budget)

    if mode == "summarize":
        if openai_client is None:
            logger.warning("PROMPT_COMPRESSION=summarize but no client available, "
                           "falling back to compact")
            return compact_system_prompt(system_text, max_tokens=budget)
        return await summarize_system_prompt(
            system_text, openai_client, openai_model, max_tokens=budget
        )

    return system_text
