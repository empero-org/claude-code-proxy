import asyncio
import json
import uuid
from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add text content
    text_content = message.get("content")
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Build usage
    raw_usage = openai_response.get("usage", {})
    usage = {
        "input_tokens": raw_usage.get("prompt_tokens", 0),
        "output_tokens": raw_usage.get("completion_tokens", 0),
    }
    # Add cache tokens if available
    prompt_details = raw_usage.get("prompt_tokens_details", {})
    if prompt_details:
        cached = prompt_details.get("cached_tokens", 0)
        if cached:
            usage["cache_read_input_tokens"] = cached
            usage["cache_creation_input_tokens"] = 0

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield _sse(Constants.EVENT_MESSAGE_START, {
        "type": Constants.EVENT_MESSAGE_START,
        "message": {
            "id": message_id,
            "type": "message",
            "role": Constants.ROLE_ASSISTANT,
            "model": original_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    yield _sse(Constants.EVENT_CONTENT_BLOCK_START, {
        "type": Constants.EVENT_CONTENT_BLOCK_START,
        "index": 0,
        "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
    })

    yield _sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    has_text_content = False

    try:
        async for line in openai_stream:
            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse chunk: {chunk_data}, error: {e}")
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        has_text_content = True
                        yield _sse(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                            "index": text_block_index,
                            "delta": {"type": Constants.DELTA_TEXT, "text": delta["content"]},
                        })

                    # Handle tool call deltas
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            for event in _process_tool_call_delta(
                                tc_delta, current_tool_calls, text_block_index, tool_block_counter
                            ):
                                yield event
                            # Update counter if new tool was started
                            tc_index = tc_delta.get("index", 0)
                            if tc_index in current_tool_calls and current_tool_calls[tc_index].get("started"):
                                tool_block_counter = max(
                                    tool_block_counter,
                                    current_tool_calls[tc_index]["claude_index"] - text_block_index,
                                )

                    # Handle finish reason
                    if finish_reason:
                        final_stop_reason = _map_finish_reason(finish_reason)
                        break

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield _sse(Constants.EVENT_ERROR, {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        })
        return

    # Send final SSE events
    yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {
        "type": Constants.EVENT_CONTENT_BLOCK_STOP,
        "index": text_block_index,
    })

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {
                "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                "index": tool_data["claude_index"],
            })

    yield _sse(Constants.EVENT_MESSAGE_DELTA, {
        "type": Constants.EVENT_MESSAGE_DELTA,
        "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    yield _sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield _sse(Constants.EVENT_MESSAGE_START, {
        "type": Constants.EVENT_MESSAGE_START,
        "message": {
            "id": message_id,
            "type": "message",
            "role": Constants.ROLE_ASSISTANT,
            "model": original_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    yield _sse(Constants.EVENT_CONTENT_BLOCK_START, {
        "type": Constants.EVENT_CONTENT_BLOCK_START,
        "index": 0,
        "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
    })

    yield _sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {"input_tokens": 0, "output_tokens": 0}

    try:
        async for line in openai_stream:
            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)

                        # Extract usage from chunk if present
                        raw_usage = chunk.get("usage")
                        if raw_usage:
                            usage_data = _extract_usage(raw_usage)

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse chunk: {chunk_data}, error: {e}")
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        yield _sse(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                            "index": text_block_index,
                            "delta": {"type": Constants.DELTA_TEXT, "text": delta["content"]},
                        })

                    # Handle tool call deltas
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            for event in _process_tool_call_delta(
                                tc_delta, current_tool_calls, text_block_index, tool_block_counter
                            ):
                                yield event
                            # Update counter if new tool was started
                            tc_index = tc_delta.get("index", 0)
                            if tc_index in current_tool_calls and current_tool_calls[tc_index].get("started"):
                                tool_block_counter = max(
                                    tool_block_counter,
                                    current_tool_calls[tc_index]["claude_index"] - text_block_index,
                                )

                    # Handle finish reason
                    if finish_reason:
                        final_stop_reason = _map_finish_reason(finish_reason)

    except HTTPException as e:
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            yield _sse(Constants.EVENT_ERROR, {
                "type": "error",
                "error": {"type": "cancelled", "message": "Request was cancelled by client"},
            })
        else:
            # Can't send HTTP error responses once streaming has started —
            # emit an SSE error event instead of re-raising.
            logger.error(f"HTTP error during streaming (status={e.status_code}): {e.detail}")
            yield _sse(Constants.EVENT_ERROR, {
                "type": "error",
                "error": {"type": "api_error", "message": str(e.detail)},
            })
        return
    except asyncio.CancelledError:
        # Client disconnected — stop silently
        logger.info(f"Request {request_id} stream cancelled (client disconnected)")
        return
    except Exception as e:
        error_str = str(e).lower()
        # Client disconnection errors — stop silently
        if any(phrase in error_str for phrase in [
            "network connection lost", "connection reset", "broken pipe",
            "connection closed", "client disconnected", "eof occurred",
        ]):
            logger.info(f"Request {request_id} stream ended (client disconnected): {e}")
            return
        logger.error(f"Streaming error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield _sse(Constants.EVENT_ERROR, {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        })
        return

    # Send final SSE events
    yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {
        "type": Constants.EVENT_CONTENT_BLOCK_STOP,
        "index": text_block_index,
    })

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield _sse(Constants.EVENT_CONTENT_BLOCK_STOP, {
                "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                "index": tool_data["claude_index"],
            })

    yield _sse(Constants.EVENT_MESSAGE_DELTA, {
        "type": Constants.EVENT_MESSAGE_DELTA,
        "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
        "usage": usage_data,
    })
    yield _sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _map_finish_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to Claude stop_reason."""
    return {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)


def _extract_usage(raw_usage: dict) -> dict:
    """Extract usage from an OpenAI usage object into Claude format."""
    usage = {
        "input_tokens": raw_usage.get("prompt_tokens", 0),
        "output_tokens": raw_usage.get("completion_tokens", 0),
    }
    prompt_details = raw_usage.get("prompt_tokens_details", {})
    if prompt_details:
        cached = prompt_details.get("cached_tokens", 0)
        if cached:
            usage["cache_read_input_tokens"] = cached
            usage["cache_creation_input_tokens"] = 0
    return usage


def _process_tool_call_delta(tc_delta, current_tool_calls, text_block_index, tool_block_counter):
    """Process a single tool call delta and yield SSE events."""
    tc_index = tc_delta.get("index", 0)

    # Initialize tool call tracking by index if not exists
    if tc_index not in current_tool_calls:
        current_tool_calls[tc_index] = {
            "id": None,
            "name": None,
            "args_buffer": "",
            "claude_index": None,
            "started": False,
        }

    tool_call = current_tool_calls[tc_index]

    # Update tool call ID if provided
    if tc_delta.get("id"):
        tool_call["id"] = tc_delta["id"]

    # Update function name
    function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
    if function_data.get("name"):
        tool_call["name"] = function_data["name"]

    # Start content block when we have complete initial data
    if tool_call["id"] and tool_call["name"] and not tool_call["started"]:
        tool_block_counter += 1
        claude_index = text_block_index + tool_block_counter
        tool_call["claude_index"] = claude_index
        tool_call["started"] = True

        yield _sse(Constants.EVENT_CONTENT_BLOCK_START, {
            "type": Constants.EVENT_CONTENT_BLOCK_START,
            "index": claude_index,
            "content_block": {
                "type": Constants.CONTENT_TOOL_USE,
                "id": tool_call["id"],
                "name": tool_call["name"],
                "input": {},
            },
        })

    # Handle function arguments — stream as partial_json incrementally
    if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
        new_fragment = function_data["arguments"]
        tool_call["args_buffer"] += new_fragment

        # Send incremental partial_json delta for each fragment
        yield _sse(Constants.EVENT_CONTENT_BLOCK_DELTA, {
            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
            "index": tool_call["claude_index"],
            "delta": {
                "type": Constants.DELTA_INPUT_JSON,
                "partial_json": new_fragment,
            },
        })
