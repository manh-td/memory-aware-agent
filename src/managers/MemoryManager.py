"""MemoryManager class for managing different types of memory stores."""

import json as json_lib
from datetime import datetime
from src.config import MODEL_NAME


class MemoryManager:
    """
    A simplified memory manager for AI agents using Oracle AI Database.

    Manages 7 types of memory:
    - Conversational: Chat history per thread (SQL table)
    - Tool Log: Raw tool execution outputs and metadata (SQL table)
    - Knowledge Base: Searchable documents (Vector store)
    - Workflow: Execution patterns (Vector store)
    - Toolbox: Available tools (Vector store)
    - Entity: People, places, systems (Vector store)
    - Summary: Storing compressed context window
    """

    def __init__(
            self,
            conn,
            conversation_table: str,
            knowledge_base_vs,
            workflow_vs,
            toolbox_vs,
            entity_vs,
            summary_vs,
            tool_log_table: str | None = None
    ):
        self.conn = conn
        self.conversation_table = conversation_table
        self.knowledge_base_vs = knowledge_base_vs
        self.workflow_vs = workflow_vs
        self.toolbox_vs = toolbox_vs
        self.entity_vs = entity_vs
        self.summary_vs = summary_vs
        self.tool_log_table = tool_log_table

    # ==================== CONVERSATIONAL MEMORY (SQL) ====================

    def write_conversational_memory(self, content: str, role: str, thread_id: str) -> str:
        """Store a message in conversation history."""
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            id_var = cur.var(str)
            cur.execute(f"""
                INSERT INTO {self.conversation_table} (thread_id, role, content, metadata, timestamp)
                VALUES (:thread_id, :role, :content, :metadata, CURRENT_TIMESTAMP)
                RETURNING id INTO :id
            """, {"thread_id": thread_id, "role": role, "content": content, "metadata": "{}", "id": id_var})
            record_id = id_var.getvalue()[0] if id_var.getvalue() else None
        self.conn.commit()
        return record_id

    def read_conversational_memory(self, thread_id: str, limit: int = 10) -> str:
        """Read conversation history for a thread (excludes summarized messages)."""
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT role, content, timestamp FROM {self.conversation_table}
                WHERE thread_id = :thread_id AND summary_id IS NULL
                ORDER BY timestamp ASC
                FETCH FIRST :limit ROWS ONLY
            """, {"thread_id": thread_id, "limit": limit})
            results = cur.fetchall()

        messages = [f"[{ts.strftime('%H:%M:%S')}] [{role}] {content}" for role, content, ts in results]
        messages_formatted = '\n'.join(messages)
        if not messages_formatted:
            messages_formatted = "(No unsummarized messages found for this thread.)"
        return f"""## Conversation Memory
### What this memory is
Chronological, unsummarized messages from the current thread. This memory captures user intent, constraints, and commitments made in recent turns.
### How you should leverage it
- Preserve continuity with prior decisions, terminology, and user preferences.
- Resolve references like "that", "previous step", or "the paper above" using earlier turns.
- If older context conflicts with newer user instructions, prioritize the latest user direction.
### Retrieved messages

{messages_formatted}"""

    def mark_as_summarized(self, thread_id: str, summary_id: str):
        """Mark all unsummarized messages in a thread as summarized."""
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {self.conversation_table}
                SET summary_id = :summary_id
                WHERE thread_id = :thread_id AND summary_id IS NULL
            """, {"summary_id": summary_id, "thread_id": thread_id})
        self.conn.commit()
        print(f"  📦 Marked messages as summarized (summary_id: {summary_id})")

    # ==================== TOOL LOG MEMORY (SQL) ====================

    def write_tool_log(
            self,
            thread_id: str,
            tool_name: str,
            tool_args,
            result: str,
            status: str = "success",
            tool_call_id: str | None = None,
            error_message: str | None = None,
            metadata: dict | None = None,
    ) -> str | None:
        """Persist raw tool execution logs for auditing and just-in-time retrieval."""
        if not self.tool_log_table:
            return None

        thread_id = str(thread_id)

        if isinstance(tool_args, (dict, list)):
            tool_args_str = json_lib.dumps(tool_args, ensure_ascii=False)
        else:
            tool_args_str = "" if tool_args is None else str(tool_args)

        result_str = "" if result is None else str(result)
        # Oracle VARCHAR2(2000) may be byte-limited; truncate preview by UTF-8 bytes.
        preview = result_str.encode("utf-8")[:2000].decode("utf-8", errors="ignore")

        metadata_str = json_lib.dumps(metadata, ensure_ascii=False) if metadata else "{}"

        with self.conn.cursor() as cur:
            id_var = cur.var(str)
            cur.execute(f"""
                INSERT INTO {self.tool_log_table}
                    (thread_id, tool_call_id, tool_name, tool_args, result, result_preview, status, error_message, metadata, timestamp)
                VALUES
                    (:thread_id, :tool_call_id, :tool_name, :tool_args, :result, :result_preview, :status, :error_message, :metadata, CURRENT_TIMESTAMP)
                RETURNING id INTO :id
            """, {
                "thread_id": thread_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args_str,
                "result": result_str,
                "result_preview": preview,
                "status": status,
                "error_message": error_message,
                "metadata": metadata_str,
                "id": id_var
            })
            log_id = id_var.getvalue()[0] if id_var.getvalue() else None

        self.conn.commit()
        return log_id

    def read_tool_logs(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Read recent tool logs for a thread, newest first."""
        if not self.tool_log_table:
            return []

        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, tool_call_id, tool_name, tool_args, result_preview, status, error_message, metadata, timestamp
                FROM {self.tool_log_table}
                WHERE thread_id = :thread_id
                ORDER BY timestamp DESC
                FETCH FIRST :limit ROWS ONLY
            """, {"thread_id": thread_id, "limit": limit})
            rows = cur.fetchall()

        logs = []
        for log_id, tool_call_id, tool_name, tool_args, result_preview, status, error_message, metadata, ts in rows:
            logs.append({
                "id": log_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result_preview": result_preview,
                "status": status,
                "error_message": error_message,
                "metadata": metadata,
                "timestamp": ts.isoformat() if ts else None,
            })
        return logs

    # ==================== KNOWLEDGE BASE (Vector Store) ====================

    def write_knowledge_base(self, text: str | list[str], metadata: dict | list[dict]):
        """
        Store knowledge-base content with metadata.

        Supports:
        - Single record: text=str, metadata=dict
        - Batch insert: text=list[str], metadata=list[dict]
        """
        if isinstance(text, list):
            texts = [str(t) for t in text]
            if isinstance(metadata, list):
                metadatas = metadata
            else:
                metadatas = [metadata for _ in texts]

            if len(texts) != len(metadatas):
                raise ValueError(
                    f"Knowledge-base batch length mismatch: {len(texts)} texts vs {len(metadatas)} metadata rows"
                )
            self.knowledge_base_vs.add_texts(texts, metadatas)
            return

        self.knowledge_base_vs.add_texts([str(text)], [metadata if isinstance(metadata, dict) else {}])

    def read_knowledge_base(self, query: str, k: int = 3) -> str:
        """Search knowledge base for relevant content."""
        results = self.knowledge_base_vs.similarity_search(query, k=k)
        content = "\n".join([doc.page_content for doc in results])
        if not content:
            content = "(No relevant knowledge base passages found.)"
        return f"""## Knowledge Base Memory
### What this memory is
Retrieved background documents and previously ingested reference material relevant to the current query.
### How you should leverage it
- Ground responses in these passages when making factual or technical claims.
- Prefer concrete details from this memory over unsupported assumptions.
- If evidence is missing or ambiguous, state uncertainty and request clarification or additional retrieval.
### Retrieved passages

{content}"""

    # ==================== WORKFLOW (Vector Store) ====================

    def write_workflow(self, query: str, steps: list, final_answer: str, success: bool = True):
        """Store a completed workflow pattern for future reference."""
        # Format steps as text
        steps_text = "\n".join([f"Step {i + 1}: {s}" for i, s in enumerate(steps)])
        text = f"Query: {query}\nSteps:\n{steps_text}\nAnswer: {final_answer[:200]}"

        metadata = {
            "query": query,
            "success": success,
            "num_steps": len(steps),
            "timestamp": datetime.now().isoformat()
        }
        self.workflow_vs.add_texts([text], [metadata])

    def read_workflow(self, query: str, k: int = 3) -> str:
        """Search for similar past workflows with at least 1 step."""
        # Filter to only include workflows that have steps (num_steps > 0)
        results = self.workflow_vs.similarity_search(
            query,
            k=k,
            filter={"num_steps": {"$gt": 0}}
        )
        if not results:
            return """## Workflow Memory
### What this memory is
Past task trajectories that include query context, ordered steps taken, and prior outcomes.
### How you should leverage it
- Use these workflows as reusable execution patterns for planning and tool orchestration.
- Adapt step sequences to the current task rather than copying blindly.
- Reuse successful patterns first, then adjust when task scope or constraints differ.
### Retrieved workflows
(No relevant workflows found.)"""
        content = "\n---\n".join([doc.page_content for doc in results])
        return f"""## Workflow Memory
### What this memory is
Past task trajectories that include query context, ordered steps taken, and prior outcomes.
### How you should leverage it
- Use these workflows as reusable execution patterns for planning and tool orchestration.
- Adapt step sequences to the current task rather than copying blindly.
- Reuse successful patterns first, then adjust when task scope or constraints differ.
### Retrieved workflows

{content}"""

    # ==================== TOOLBOX (Vector Store) ====================

    def write_toolbox(self, text: str, metadata: dict):
        """Store a tool definition in the toolbox."""
        self.toolbox_vs.add_texts([text], [metadata])

    def read_toolbox(self, query: str, k: int = 3) -> list[dict]:
        """Find relevant tools and return OpenAI-compatible schemas."""
        results = self.toolbox_vs.similarity_search(query, k=k)
        tools = []
        seen_tool_names: set[str] = set()
        for doc in results:
            meta = doc.metadata
            tool_name = meta.get("name", "tool")
            if tool_name in seen_tool_names:
                continue
            seen_tool_names.add(tool_name)
            # Extract parameters from metadata and convert to OpenAI format
            stored_params = meta.get("parameters", {})
            properties = {}
            required = []

            for param_name, param_info in stored_params.items():
                # Convert stored param info to OpenAI schema format
                param_type = param_info.get("type", "string")
                # Map Python types to JSON schema types
                type_mapping = {
                    "<class 'str'>": "string",
                    "<class 'int'>": "integer",
                    "<class 'float'>": "number",
                    "<class 'bool'>": "boolean",
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean"
                }
                json_type = type_mapping.get(param_type, "string")
                properties[param_name] = {"type": json_type}

                # If no default, it's required
                if "default" not in param_info:
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": meta.get("description", ""),
                    "parameters": {"type": "object", "properties": properties, "required": required}
                }
            })
        return tools

    # ==================== ENTITY (Vector Store) ====================

    def extract_entities(self, text: str, llm_client) -> list[dict]:
        """Use LLM to extract entities (people, places, systems) from text."""
        if not text or len(text.strip()) < 5:
            return []

        prompt = f'''Extract entities from: "{text[:500]}"
Return JSON: [{{"name": "X", "type": "PERSON|PLACE|SYSTEM", "description": "brief"}}]
If none: []'''

        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000
            )
            result = response.choices[0].message.content.strip()

            # Extract JSON array from response
            start, end = result.find("["), result.rfind("]")
            if start == -1 or end == -1:
                return []

            parsed = json_lib.loads(result[start:end + 1])
            return [{"name": e["name"], "type": e.get("type", "UNKNOWN"), "description": e.get("description", "")}
                    for e in parsed if isinstance(e, dict) and e.get("name")]
        except:
            return []

    def write_entity(self, name: str, entity_type: str, description: str, llm_client=None, text: str = None):
        """Store an entity OR extract and store entities from text."""
        if text and llm_client:
            # Extract and store entities from text
            entities = self.extract_entities(text, llm_client)
            for e in entities:
                self.entity_vs.add_texts(
                    [f"{e['name']} ({e['type']}): {e['description']}"],
                    [{"name": e['name'], "type": e['type'], "description": e['description']}]
                )
            return entities
        else:
            # Store single entity directly
            self.entity_vs.add_texts(
                [f"{name} ({entity_type}): {description}"],
                [{"name": name, "type": entity_type, "description": description}]
            )

    def read_entity(self, query: str, k: int = 5) -> str:
        """Search for relevant entities."""
        results = self.entity_vs.similarity_search(query, k=k)
        if not results:
            return """## Entity Memory
### What this memory is
Entity-level context such as people, organizations, systems, tools, and other named items previously identified in conversations or documents.
### How you should leverage it
- Use entities to disambiguate references and maintain consistent naming.
- Preserve important attributes (roles, relationships, descriptions) across turns.
- Personalize and contextualize responses using relevant known entities.
### Retrieved entities
(No entities found.)"""

        entities = [f"• {doc.metadata.get('name', '?')}: {doc.metadata.get('description', '')}"
                    for doc in results if hasattr(doc, 'metadata')]
        entities_formatted = '\n'.join(entities)
        return f"""## Entity Memory
### What this memory is
Entity-level context such as people, organizations, systems, tools, and other named items previously identified in conversations or documents.
### How you should leverage it
- Use entities to disambiguate references and maintain consistent naming.
- Preserve important attributes (roles, relationships, descriptions) across turns.
- Personalize and contextualize responses using relevant known entities.
### Retrieved entities

{entities_formatted}"""

    # ==================== SUMMARY (Vector Store) ====================

    def write_summary(
            self,
            summary_id: str,
            full_content: str,
            summary: str,
            description: str,
            thread_id: str | None = None,
    ):
        """Store a summary with its original content."""
        metadata = {
            "id": summary_id,
            "full_content": full_content,
            "summary": summary,
            "description": description,
        }
        if thread_id is not None:
            metadata["thread_id"] = str(thread_id)
        self.summary_vs.add_texts(
            [f"{summary_id}: {description}"],
            [metadata]
        )
        return summary_id

    def read_summary_memory(self, summary_id: str, thread_id: str | None = None) -> str:
        """Retrieve a specific summary by ID (just-in-time retrieval)."""
        filters = {"id": summary_id}
        if thread_id is not None:
            filters["thread_id"] = str(thread_id)

        results = self.summary_vs.similarity_search(
            summary_id,
            k=5,
            filter=filters
        )
        if not results:
            if thread_id is not None:
                return f"Summary {summary_id} not found for thread {thread_id}."
            return f"Summary {summary_id} not found."
        doc = results[0]
        return doc.metadata.get('summary', 'No summary content.')

    def read_summary_context(self, query: str = "", k: int = 10, thread_id: str | None = None) -> str:
        """Get available summaries for context window (IDs + descriptions only)."""
        filters = None
        if thread_id is not None:
            filters = {"thread_id": str(thread_id)}
        results = self.summary_vs.similarity_search(query or "summary", k=k, filter=filters)
        if not results:
            scope_note = (
                f"(No summaries available for thread {thread_id}.)"
                if thread_id is not None
                else "(No summaries available.)"
            )
            return f"""## Summary Memory
### What this memory is
Compressed snapshots of older conversation windows preserved to retain long-range context.
### How you should leverage it
- Use summaries to maintain continuity when full historical messages are not in the active context window.
- Call expand_summary(id) before depending on exact quotes, fine-grained details, or step-by-step chronology.
### Available summaries
{scope_note}"""

        lines = [
            "## Summary Memory",
            "### What this memory is",
            "Compressed snapshots of older conversation windows preserved to retain long-range context.",
            "### How you should leverage it",
            "- Use summaries to maintain continuity when full historical messages are not in the active context window.",
            "- Call expand_summary(id) before depending on exact quotes, fine-grained details, or step-by-step chronology.",
            "### Available summaries",
            "Use expand_summary(id) to retrieve the detailed underlying conversation."
        ]
        if thread_id is not None:
            lines.append(f"Scope: thread_id = {thread_id}")
        for doc in results:
            sid = doc.metadata.get('id', '?')
            desc = doc.metadata.get('description', 'No description')
            lines.append(f"  • [ID: {sid}] {desc}")
        return "\n".join(lines)

    def read_conversations_by_summary_id(self, summary_id: str) -> str:
        """
        Retrieve all original conversations that were summarized with a given summary_id.
        Returns conversations in order of occurrence with timestamps.

        Args:
            summary_id: The ID of the summary to expand

        Returns:
            Formatted string with original conversations and timestamps
        """
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, role, content, timestamp 
                FROM {self.conversation_table}
                WHERE summary_id = :summary_id
                ORDER BY timestamp ASC
            """, {"summary_id": summary_id})
            results = cur.fetchall()

        if not results:
            return f"No conversations found for summary_id: {summary_id}"

        # Format conversations with timestamps
        lines = [f"## Expanded Conversations for Summary ID: {summary_id}"]
        lines.append(f"Total messages: {len(results)}\n")

        for msg_id, role, content, timestamp in results:
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
            lines.append(f"[{ts_str}] [{role.upper()}]")
            lines.append(f"{content}")
            lines.append("")  # Empty line between messages

        return "\n".join(lines)
