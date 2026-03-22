"""LangChain-first memory-aware agent demo.

This script mirrors the repository's custom memory-aware agent idea, but leans on
LangChain/LangGraph memory primitives directly:

- Short-term memory: thread-scoped checkpoints via `MemorySaver`
- Long-term memory: namespace-scoped documents in a LangGraph store
- Tool calling: LangChain `@tool` with injected store/config

Reference:
https://docs.langchain.com/oss/python/concepts/memory
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from typing_extensions import Annotated

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.config import AGENT_SYSTEM_PROMPT, API_BASE, API_KEY, MODEL_NAME, TEMPERATURE
from src.utils.embedding_model import embedding_model


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _embed_texts_for_store(texts: list[str]) -> list[list[float]]:
    """Embedding callback expected by LangGraph `InMemoryStore`.

    Uses a lightweight deterministic fallback to keep the script runnable even if
    embedding infrastructure is unavailable in minimal environments.
    """
    if hasattr(embedding_model, "embed_documents"):
        return embedding_model.embed_documents(texts)

    if hasattr(embedding_model, "embed_query"):
        vectors = []
        for text in texts:
            vectors.append(embedding_model.embed_query(text))
        return vectors


def _build_memory_tools():
    """Create long-term memory tools using LangGraph store namespaces."""

    @tool
    def save_semantic_memory(
        fact: str,
        category: str = "user_preference",
        key: str | None = None,
        *,
        store: Annotated[BaseStore, InjectedStore()],
        config: RunnableConfig,
    ) -> str:
        """Save a factual memory (semantic memory) for this user across sessions."""
        user_id = str(config.get("configurable", {}).get("user_id", "default-user"))
        namespace = ("semantic", user_id, category)
        memory_key = key or f"sem-{datetime.now(timezone.utc).timestamp():.0f}"
        payload = {
            "fact": fact,
            "category": category,
            "created_at": _utc_now_iso(),
        }
        store.put(namespace, memory_key, payload)
        return f"Saved semantic memory under namespace={namespace}, key={memory_key}."

    @tool
    def search_semantic_memory(
        query: str,
        category: str = "user_preference",
        k: int = 5,
        *,
        store: Annotated[BaseStore, InjectedStore()],
        config: RunnableConfig,
    ) -> str:
        """Search semantic memories for relevant facts about this user."""
        user_id = str(config.get("configurable", {}).get("user_id", "default-user"))
        namespace = ("semantic", user_id, category)
        items = store.search(namespace, query=query, limit=max(1, min(k, 10)))
        records = [
            {
                "key": item.key,
                "namespace": item.namespace,
                "value": item.value,
                "score": getattr(item, "score", None),
            }
            for item in items
        ]
        return _json_dumps(records if records else {"message": "No semantic memory found."})

    @tool
    def save_episode(
        task: str,
        steps: list[str],
        outcome: str,
        success: bool = True,
        *,
        store: Annotated[BaseStore, InjectedStore()],
        config: RunnableConfig,
    ) -> str:
        """Save an execution trajectory (episodic memory) for later retrieval."""
        user_id = str(config.get("configurable", {}).get("user_id", "default-user"))
        namespace = ("episodic", user_id)
        key = f"ep-{datetime.now(timezone.utc).timestamp():.0f}"
        payload = {
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "success": success,
            "created_at": _utc_now_iso(),
        }
        store.put(namespace, key, payload)
        return f"Saved episode under namespace={namespace}, key={key}."

    @tool
    def search_episodes(
        query: str,
        k: int = 3,
        *,
        store: Annotated[BaseStore, InjectedStore()],
        config: RunnableConfig,
    ) -> str:
        """Retrieve similar past episodes to guide current planning."""
        user_id = str(config.get("configurable", {}).get("user_id", "default-user"))
        namespace = ("episodic", user_id)
        items = store.search(namespace, query=query, limit=max(1, min(k, 10)))
        records = [
            {
                "key": item.key,
                "value": item.value,
                "score": getattr(item, "score", None),
            }
            for item in items
        ]
        return _json_dumps(records if records else {"message": "No episodes found."})

    @tool
    def update_procedural_memory(
        instructions: str,
        *,
        store: Annotated[BaseStore, InjectedStore()],
        config: RunnableConfig,
    ) -> str:
        """Update the agent's persistent instruction profile (procedural memory)."""
        user_id = str(config.get("configurable", {}).get("user_id", "default-user"))
        namespace = ("procedural", user_id)
        store.put(
            namespace,
            "agent-instructions",
            {
                "instructions": instructions,
                "updated_at": _utc_now_iso(),
            },
        )
        return "Updated procedural memory for this user."

    @tool
    def get_procedural_memory(
        *,
        store: Annotated[BaseStore, InjectedStore()],
        config: RunnableConfig,
    ) -> str:
        """Return current procedural instructions for this user, if any."""
        user_id = str(config.get("configurable", {}).get("user_id", "default-user"))
        namespace = ("procedural", user_id)
        entry = store.get(namespace, "agent-instructions")
        if not entry:
            return "No procedural memory set yet."
        return _json_dumps({"namespace": entry.namespace, "key": entry.key, "value": entry.value})

    @tool
    def get_current_time() -> str:
        """Return the current UTC time."""
        return _utc_now_iso()

    return [
        save_semantic_memory,
        search_semantic_memory,
        save_episode,
        search_episodes,
        update_procedural_memory,
        get_procedural_memory,
        get_current_time,
    ]


def _build_system_prompt(
        system_prompt: str
    ) -> str:
    """Compose base prompt with memory-specific operating rules."""
    memory_rules = """

# Memory policy
- Use short-term thread memory for immediate conversation continuity.
- Use semantic memory tools for durable user facts and preferences.
- Use episodic memory tools to recall prior successful task trajectories.
- Use procedural memory to refine persistent instructions when feedback appears.
- Keep memory writes focused and specific; avoid writing transient details.
"""
    return f"{system_prompt.strip()}\n{memory_rules.strip()}"


class LangChainMemoryAwareAgent:
    """Memory-aware agent powered by LangChain/LangGraph primitives."""

    def __init__(
            self,
            system_prompt: str = AGENT_SYSTEM_PROMPT,
            tools: list[BaseTool] = []
        ) -> None:
        self.model = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=API_BASE,
            temperature=TEMPERATURE,
        )

        self.checkpointer = MemorySaver()
        self.store = InMemoryStore(index={"embed": _embed_texts_for_store, "dims": 64})
        self.tools = _build_memory_tools()
        self.tools += tools

        self.graph = create_agent(
            model=self.model,
            tools=self.tools,
            prompt=_build_system_prompt(system_prompt),
            checkpointer=self.checkpointer,
            store=self.store,
        )

    def ask(self, query: str, user_id: str = "demo-user", thread_id: str = "demo-thread") -> str:
        """Invoke agent with thread-scoped short-term + namespace long-term memory."""
        config: RunnableConfig = {
            "configurable": {
                "thread_id": str(thread_id),
                "user_id": str(user_id),
            }
        }
        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )
        messages = result.get("messages", [])
        if not messages:
            return ""
        return str(messages[-1].content)


def main() -> None:
    """Run a tiny multi-turn demo showing short- and long-term memory behavior."""
    agent = LangChainMemoryAwareAgent()

    demo_turns = [
        "Remember that I prefer concise answers and Python examples.",
        "Store an episode that we solved arXiv summarization via read -> chunk -> save.",
        "What do you remember about my preferences and prior workflows?",
    ]

    for i, turn in enumerate(demo_turns, start=1):
        response = agent.ask(turn, user_id="alice", thread_id="research-thread")
        print(f"\n--- Turn {i} ---")
        print(f"User: {turn}")
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
