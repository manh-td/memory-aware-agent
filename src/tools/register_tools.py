"""Tool registration helpers for MemoryAwareAgent."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from src.config import KNOWLEDGE_BASE_TABLE
from src.tools.common import register_common_tools
from src.tools.summary import register_summary_tools


def register_tools(
    toolbox,
    memory_manager,
    include_common_tools: bool = True,
    custom_tools: Iterable[Callable] | None = None,
    augment_custom_tools: bool = False,
    knowledge_base_table: str = KNOWLEDGE_BASE_TABLE,
    include_summary_tools: bool = True,
) -> dict[str, Callable]:
    """Register common tools, summary tools, and user-provided tools.

    Args:
        toolbox: Toolbox instance used to register and execute tools.
        memory_manager: MemoryManager instance required by common tools.
        include_common_tools: If True, registers the default shared tools.
        custom_tools: Optional iterable of user functions to register.
        augment_custom_tools: If True, apply LLM augmentation for custom tools.
        knowledge_base_table: Knowledge base table name for common tools.
        include_summary_tools: If True, include summary tools.

    Returns:
        Dict mapping tool function names to callables.
    """
    registered_tools: dict[str, Callable] = {}

    if include_common_tools:
        common = register_common_tools(
            toolbox=toolbox,
            memory_manager=memory_manager,
            knowledge_base_table=knowledge_base_table,
        )
        registered_tools.update(common)

    if include_summary_tools:
        summary = register_summary_tools(
            toolbox=toolbox,
            memory_manager=memory_manager,
            llm_client=toolbox.llm_client,
        )
        registered_tools.update(summary)

    for func in custom_tools or []:
        toolbox.register_tool(func, augment=augment_custom_tools)
        registered_tools[func.__name__] = func

    return registered_tools
