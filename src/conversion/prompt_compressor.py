"""Optional prompt compression for less capable backend models.

Modes:
  none      — pass through verbatim (default)
  compact   — rule-based: strip boilerplate, compress whitespace, cap length
  summarize — use the backend LLM to condense the system prompt (adds latency)
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known Claude Code boilerplate patterns (verbose instructions that small
# models don't benefit from).  Order matters — earlier patterns are tried first.
# ---------------------------------------------------------------------------
_BOILERPLATE_PATTERNS = [
    # Long behavioral instruction blocks
    re.compile(
        r"# Tone and style\n.*?(?=\n# |\Z)", re.DOTALL
    ),
    re.compile(
        r"# Output efficiency\n.*?(?=\n# |\Z)", re.DOTALL
    ),
    re.compile(
        r"# Doing tasks\n.*?(?=\n# |\Z)", re.DOTALL
    ),
    re.compile(
        r"# Executing actions with care\n.*?(?=\n# |\Z)", re.DOTALL
    ),
    re.compile(
        r"# Using your tools\n.*?(?=\n# |\Z)", re.DOTALL
    ),
    # Auto-memory instructions (very long, model-specific)
    re.compile(
        r"# auto memory\n.*?(?=\n# |\Z)", re.DOTALL
    ),
    # VSCode extension context boilerplate
    re.compile(
        r"# VSCode Extension Context\n.*?(?=\n# |\Z)", re.DOTALL
    ),
    # System reminder tags
    re.compile(
        r"<system-reminder>.*?</system-reminder>", re.DOTALL
    ),
]

# Sections to always preserve (even in compact mode)
_PRESERVE_KEYWORDS = [
    "# Environment",
    "# System",
    "# Tool",
    "CLAUDE.md",
]

# Condensed replacement for stripped behavioral instructions
_COMPACT_HEADER = """You are a helpful coding assistant. Follow these rules:
- Use the provided tools to complete tasks. Prefer dedicated tools over shell commands.
- Read files before editing. Make precise, minimal changes.
- Do not add unnecessary features, comments, or refactoring beyond what was asked.
- Be concise in responses. Lead with the answer, not the reasoning.
- Break complex tasks into steps and track progress.
"""


def compact_system_prompt(text: str, max_tokens: int = 4096) -> str:
    """Rule-based compression of the system prompt.

    Strategy:
    1. Replace known verbose boilerplate sections with a compact header
    2. Collapse excessive whitespace
    3. Truncate to max_tokens budget (estimated at ~4 chars/token)
    """
    original_len = len(text)

    # Check if any boilerplate sections exist
    has_boilerplate = any(p.search(text) for p in _BOILERPLATE_PATTERNS)

    if has_boilerplate:
        # Strip boilerplate sections
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)

        # Prepend compact behavioral header
        text = _COMPACT_HEADER + "\n" + text

    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces (but not newlines)
    text = re.sub(r"[ \t]{3,}", "  ", text)

    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    text = text.strip()

    # Truncate to budget (rough: 4 chars ≈ 1 token)
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        # Try to cut at a paragraph boundary
        cut_point = text.rfind("\n\n", 0, max_chars)
        if cut_point < max_chars * 0.5:
            # No good paragraph break — cut at last newline
            cut_point = text.rfind("\n", 0, max_chars)
        if cut_point < 0:
            cut_point = max_chars
        text = text[:cut_point].rstrip() + "\n\n[System prompt truncated for model compatibility]"

    compressed_len = len(text)
    if original_len > 0:
        ratio = (1 - compressed_len / original_len) * 100
        if ratio > 5:
            logger.info(
                f"Prompt compacted: {original_len} → {compressed_len} chars "
                f"({ratio:.0f}% reduction)"
            )

    return text


async def summarize_system_prompt(
    text: str,
    openai_client,
    model: str,
    max_tokens: int = 2048,
) -> str:
    """Use the backend LLM itself to summarize the system prompt.

    This adds one extra API call but can achieve much better compression
    for very large prompts while preserving semantic meaning.
    """
    # First do a compact pass to remove obvious boilerplate
    pre_compacted = compact_system_prompt(text, max_tokens=max_tokens * 3)

    # If already short enough, skip the LLM call
    if len(pre_compacted) <= max_tokens * 4:
        return pre_compacted

    summarize_prompt = (
        "You are a system prompt compressor. Condense the following system prompt "
        "into a shorter version that preserves:\n"
        "1. ALL tool names and their purposes (but not full JSON schemas)\n"
        "2. ALL project-specific context (file paths, architecture, conventions)\n"
        "3. Key behavioral rules (what to do and not do)\n"
        "4. Environment information (OS, shell, working directory)\n\n"
        "Remove: verbose examples, redundant explanations, formatting guidelines, "
        "style instructions that are obvious.\n\n"
        f"Target length: ~{max_tokens} tokens.\n\n"
        "SYSTEM PROMPT TO COMPRESS:\n\n"
        f"{pre_compacted}"
    )

    try:
        response = await openai_client.create_chat_completion({
            "model": model,
            "messages": [{"role": "user", "content": summarize_prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        })
        choices = response.get("choices", [])
        if choices:
            summary = choices[0].get("message", {}).get("content", "")
            if summary and len(summary) > 100:
                logger.info(
                    f"Prompt summarized: {len(pre_compacted)} → {len(summary)} chars "
                    f"via LLM ({model})"
                )
                return summary.strip()
    except Exception as e:
        logger.warning(f"Prompt summarization failed, falling back to compact: {e}")

    # Fallback to compact result
    return compact_system_prompt(text, max_tokens=max_tokens)
