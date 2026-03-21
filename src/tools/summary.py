"""Summary tools and context window management."""

import uuid
from src.config import MODEL_NAME, MODEL_TOKEN_LIMITS


# ====================  CONTEXT WINDOW MANAGEMENT ====================


def calculate_context_usage(context: str, model: str = MODEL_NAME) -> dict:
    """Calculate context window usage as percentage."""
    estimated_tokens = len(context) // 4  # ~4 chars per token
    max_tokens = MODEL_TOKEN_LIMITS.get(model, 128000)
    percentage = (estimated_tokens / max_tokens) * 100
    return {"tokens": estimated_tokens, "max": max_tokens, "percent": round(percentage, 1)}


def monitor_context_window(context: str, model: str = MODEL_NAME) -> dict:
    """
    Monitor the current context window and return capacity utilization.

    Args:
        context: The current context string to measure
        model: The model being used (to determine max tokens)

    Returns:
        dict with tokens, max, percent, and status ('ok', 'warning', 'critical')
    """
    result = calculate_context_usage(context, model)

    if result['percent'] < 50:
        result['status'] = 'ok'
    elif result['percent'] < 80:
        result['status'] = 'warning'
    else:
        result['status'] = 'critical'

    return result


def summarise_context_window(
        content: str,
        memory_manager,
        llm_client,
        model: str = MODEL_NAME,
        thread_id: str | None = None,
) -> dict:
    """
    Summarise content using LLM and store in summary memory.
    Captures technical details, emotional context, and entity information.
    """
    cleaned = (content or "").strip()
    if not cleaned:
        return {"status": "nothing_to_summarize"}

    def _message_text(resp) -> str:
        msg = resp.choices[0].message
        payload = getattr(msg, "content", None)
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
            return "\n".join(parts).strip()
        return ""

    def _fallback_description(summary_text: str, source_text: str) -> str:
        """Create a specific, non-generic summary label when model labeling is weak."""
        for raw_line in (summary_text or source_text).splitlines():
            line = raw_line.strip().strip("-•* ").strip()
            if not line or line.startswith("###"):
                continue
            words = [w for w in line.split() if w]
            if len(words) < 4:
                continue
            label = " ".join(words[:12]).strip(" ,.;:")
            if label:
                return label
        return "Recent thread context, decisions, and open actions"

    summary_prompt = f"""You are creating durable memory for an AI research assistant.
Summarize this conversation so it can be resumed accurately later.

Output with exactly these headings:
### Technical Information
### Emotional Context
### Entities & References
### Action Items & Decisions

Rules:
- Keep concrete details (names, dates, APIs, errors, decisions).
- Separate confirmed facts from open questions where relevant.
- Do not invent information.
- Keep it concise and useful for continuation.

Conversation:
{cleaned[:6000]}"""

    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        max_completion_tokens=4000
    )
    summary = _message_text(response)

    # Retry once with a simpler prompt if the model returns empty content.
    if not summary:
        retry_prompt = f"""Summarize this conversation in <= 180 words using these headings:
### Technical Information
### Emotional Context
### Entities & References
### Action Items & Decisions

Conversation:
{cleaned[:6000]}"""
        retry = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            max_completion_tokens=4000
        )
        summary = _message_text(retry)

    if not summary:
        # Deterministic fallback so downstream flow never breaks.
        excerpt = cleaned[:500].replace("\n", " ").strip()
        summary = (
            "### Technical Information\n"
            f"{excerpt or '(No content provided.)'}\n\n"
            "### Emotional Context\n"
            "Not available from model output.\n\n"
            "### Entities & References\n"
            "Not available from model output.\n\n"
            "### Action Items & Decisions\n"
            "Not available from model output."
        )

    desc_prompt = f"""Create a short 8-12 word label for this summary.
Return ONLY the label.

Requirements:
- Be specific about the topic/outcome.
- Include a concrete signal (entity, task, or issue).
- Do not use generic labels like "Conversation summary".

Summary:
{summary}"""

    desc_response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": desc_prompt}],
        max_completion_tokens=2000
    )
    description = _message_text(desc_response).strip().strip('"').strip("'")
    if not description or description.lower() in {
        "conversation summary",
        "summary",
        "chat summary",
        "thread summary",
    }:
        description = _fallback_description(summary, cleaned)

    summary_id = str(uuid.uuid4())[:8]
    memory_manager.write_summary(summary_id, cleaned, summary, description, thread_id=thread_id)

    return {"id": summary_id, "description": description, "summary": summary}


def offload_to_summary(context: str, memory_manager, llm_client, thread_id: str | None = None) -> tuple[
    str, list[dict]]:
    """
    Simple context compaction:
    - If thread_id is provided, summarize unsummarized conversation units for that thread.
    - Otherwise, summarize the provided context string.
    - Return a compact context with summary references that can be expanded on demand.
    """
    raw_context = (context or "").strip()

    if thread_id is not None:
        result = summarize_conversation(thread_id, memory_manager, llm_client)
    else:
        result = summarise_context_window(raw_context, memory_manager, llm_client, thread_id=thread_id)

    if result.get("status") == "nothing_to_summarize":
        return raw_context, []

    summary_ref = f"[Summary ID: {result['id']}] {result['description']}"
    conversation_stub = (
        "## Conversation Memory\n"
        "Older conversation content was summarized to reduce context size.\n"
        "Use Summary Memory references + expand_summary(id) for full detail."
    )

    # Replace only the conversation section, keep other memory segments intact.
    compact_context = raw_context
    if "## Conversation Memory" in compact_context:
        lines = compact_context.splitlines()
        rebuilt: list[str] = []
        in_conversation = False
        inserted_stub = False

        for line in lines:
            if line.startswith("## "):
                heading = line.strip()
                if heading == "## Conversation Memory":
                    in_conversation = True
                    if not inserted_stub:
                        if rebuilt and rebuilt[-1].strip():
                            rebuilt.append("")
                        rebuilt.extend(conversation_stub.splitlines())
                        rebuilt.append("")
                        inserted_stub = True
                    continue
                in_conversation = False

            if not in_conversation:
                rebuilt.append(line)

        compact_context = "\n".join(rebuilt).strip()
    else:
        compact_context = f"{conversation_stub}\n\n{compact_context}".strip()

    # Add summary reference(s).
    if "## Summary Memory" in compact_context:
        compact_context = f"{compact_context}\n{summary_ref}".strip()
    else:
        compact_context = (
            f"{compact_context}\n\n"
            "## Summary Memory\n"
            "Use expand_summary(id) to retrieve full underlying content.\n"
            f"{summary_ref}"
        ).strip()

    return compact_context, [result]


def summarize_conversation(thread_id: str, memory_manager, llm_client) -> dict:
    """
    Summarize all unsummarized messages in a thread and mark them.

    This function:
    1. Reads all unsummarized messages from the current thread
    2. Generates a comprehensive summary capturing technical, emotional, and entity info
    3. Stores the summary in the summary memory store
    4. Marks all conversation messages with the summary_id for later retrieval
    5. Returns the summary info that can be used to start a new context window
    """
    thread_id = str(thread_id)
    with memory_manager.conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, role, content, timestamp
            FROM {memory_manager.conversation_table}
            WHERE thread_id = :thread_id AND summary_id IS NULL
            ORDER BY timestamp ASC
        """, {"thread_id": thread_id})
        rows = cur.fetchall()

    if not rows:
        return {"status": "nothing_to_summarize"}

    message_ids: list[str] = []
    transcript_lines: list[str] = []
    for msg_id, role, content, timestamp in rows:
        message_ids.append(msg_id)
        ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
        transcript_lines.append(f"[{ts_str}] [{str(role).upper()}] {content}")

    transcript = "\n".join(transcript_lines).strip()
    if not transcript:
        return {"status": "nothing_to_summarize"}

    result = summarise_context_window(transcript, memory_manager, llm_client, thread_id=thread_id)
    if result.get("status") == "nothing_to_summarize":
        return result

    summary_id = result["id"]
    with memory_manager.conn.cursor() as cur:
        cur.executemany(f"""
            UPDATE {memory_manager.conversation_table}
            SET summary_id = :summary_id
            WHERE id = :id AND summary_id IS NULL
        """, [{"summary_id": summary_id, "id": msg_id} for msg_id in message_ids])
    memory_manager.conn.commit()

    result["num_messages_summarized"] = len(message_ids)

    print(f"✅ Conversation summarized: [Summary ID: {summary_id}]")
    print(f"   Description: {result['description']}")
    print(f"   Messages marked summarized: {len(message_ids)}")

    return result


# ====================  SUMMARY TOOLS ====================

def register_summary_tools(toolbox, memory_manager, llm_client):
    """
    Register summary-related tools with the toolbox.

    Args:
        toolbox: The Toolbox instance to register tools with
        memory_manager: MemoryManager instance for memory operations
        llm_client: OpenAI client for LLM calls

    Returns:
        dict of registered tool functions
    """

    def expand_summary(summary_id: str, thread_id: str = None) -> str:
        """
        Expand a summary reference to retrieve the original conversations.

        Use when you need more details from a [Summary ID: xxx] reference.
        Returns all original messages that were summarized, in chronological order with timestamps.
        """
        summary_text = memory_manager.read_summary_memory(summary_id, thread_id=thread_id)
        original_conversations = memory_manager.read_conversations_by_summary_id(summary_id)

        return f"""## Summary Context
{summary_text}

{original_conversations}"""

    def summarize_and_store(text: str = "", thread_id: str = None) -> str:
        """
        Summarize content and store it in Summary Memory.

        Preferred usage for agent continuity:
        - Provide `thread_id` to summarize unsummarized conversation units and mark them.

        Alternate usage:
        - Provide `text` to summarize arbitrary content.
        """
        if thread_id:
            result = summarize_conversation(thread_id, memory_manager, llm_client)
            if result.get("status") == "nothing_to_summarize":
                return f"No unsummarized messages found for thread {thread_id}."
            return f"Stored as [Summary ID: {result['id']}] {result['description']}"

        text = (text or "").strip()
        if not text:
            return "Provide `thread_id` (preferred) or non-empty `text` to summarize."

        result = summarise_context_window(text, memory_manager, llm_client)
        if result.get("status") == "nothing_to_summarize":
            return "No content to summarize."
        return f"Stored as [Summary ID: {result['id']}] {result['description']}"

    # Register with toolbox
    toolbox.register_tool(expand_summary, augment=True)
    toolbox.register_tool(summarize_and_store, augment=True)

    registered_tools = {
        "expand_summary": expand_summary,
        "summarize_and_store": summarize_and_store,
    }

    print(f"✅ Registered {len(registered_tools)} summary tools: {list(registered_tools.keys())}")

    return registered_tools
