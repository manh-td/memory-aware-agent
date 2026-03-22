"""Memory-aware agent orchestration class."""

import json as json_lib
from json import JSONDecodeError
from collections.abc import Callable, Iterable

from langchain_community.vectorstores.utils import DistanceStrategy
from openai import OpenAI

from src.config import (
    AGENT_SYSTEM_PROMPT,
    API_BASE,
    API_KEY,
    CONVERSATIONAL_TABLE,
    ENTITY_TABLE,
    KNOWLEDGE_BASE_TABLE,
    MODEL_NAME,
    SUMMARY_TABLE,
    TOOLBOX_TABLE,
    TOOL_LOG_TABLE,
    WORKFLOW_TABLE,
)
from src.database.bootstrap import database_connection
from src.database.tables import (
    create_conversational_history_table,
    create_tool_log_table,
)
from src.database.vector_indexes import safe_create_index
from src.managers.MemoryManager import MemoryManager
from src.managers.StoreManager import StoreManager
from src.tools.register_tools import register_tools
from src.tools.summary import calculate_context_usage, offload_to_summary
from src.tools.toolbox import Toolbox
from src.utils.embedding_model import embedding_model


class MemoryAwareAgent:
    """Reusable memory-aware agent with dynamic tool calling and summarization."""

    def __init__(
        self,
        thread_id: str = "1",
        model: str = MODEL_NAME,
        max_iterations: int = 10,
        create_indexes: bool = True,
        agent_system_prompt: str = None,
        include_common_tools: bool = True,
        custom_tools: Iterable[Callable] | None = None,
        augment_custom_tools: bool = False,
        include_summary_tools: bool = True,
    ):
        self.thread_id = str(thread_id)
        self.model = model
        self.max_iterations = max_iterations
        self.agent_system_prompt = agent_system_prompt

        self.conversation_history_table = create_conversational_history_table(
            database_connection,
            CONVERSATIONAL_TABLE,
        )
        self.tool_log_history_table = create_tool_log_table(
            database_connection,
            TOOL_LOG_TABLE,
        )

        self.store_manager = StoreManager(
            client=database_connection,
            embedding_function=embedding_model,
            table_names={
                "knowledge_base": KNOWLEDGE_BASE_TABLE,
                "workflow": WORKFLOW_TABLE,
                "toolbox": TOOLBOX_TABLE,
                "entity": ENTITY_TABLE,
                "summary": SUMMARY_TABLE,
            },
            distance_strategy=DistanceStrategy.COSINE,
            conversational_table=self.conversation_history_table,
            tool_log_table=self.tool_log_history_table,
        )

        self.knowledge_base_vs = self.store_manager.get_knowledge_base_store()
        self.workflow_vs = self.store_manager.get_workflow_store()
        self.toolbox_vs = self.store_manager.get_toolbox_store()
        self.entity_vs = self.store_manager.get_entity_store()
        self.summary_vs = self.store_manager.get_summary_store()

        if create_indexes:
            print("Creating vector indexes...")
            safe_create_index(database_connection, self.knowledge_base_vs, "knowledge_base_vs_ivf")
            safe_create_index(database_connection, self.workflow_vs, "workflow_vs_ivf")
            safe_create_index(database_connection, self.toolbox_vs, "toolbox_vs_ivf")
            safe_create_index(database_connection, self.entity_vs, "entity_vs_ivf")
            safe_create_index(database_connection, self.summary_vs, "summary_vs_ivf")
            print("All indexes created!")

        self.memory_manager = MemoryManager(
            conn=database_connection,
            conversation_table=self.conversation_history_table,
            knowledge_base_vs=self.knowledge_base_vs,
            workflow_vs=self.workflow_vs,
            toolbox_vs=self.toolbox_vs,
            entity_vs=self.entity_vs,
            summary_vs=self.summary_vs,
            tool_log_table=self.tool_log_history_table,
        )

        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        self.toolbox = Toolbox(self.memory_manager, self.client, embedding_model)

        self.registered_tools = register_tools(
            self.toolbox,
            self.memory_manager,
            include_common_tools=include_common_tools,
            custom_tools=custom_tools,
            augment_custom_tools=augment_custom_tools,
            knowledge_base_table=KNOWLEDGE_BASE_TABLE,
            include_summary_tools=include_summary_tools,
        )

    def execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        current_thread_id: str | None = None,
    ) -> str:
        """Execute a tool by looking it up in the toolbox."""
        if tool_name not in self.toolbox._tools_by_name:
            return f"Error: Tool '{tool_name}' not found"

        args = dict(tool_args or {})

        # Keep summary source-row attribution pinned to active thread.
        if tool_name == "summarize_and_store" and "thread_id" not in args and current_thread_id is not None:
            args["thread_id"] = str(current_thread_id)

        return str(self.toolbox._tools_by_name[tool_name](**args) or "Done")

    def call_openai_chat(self, messages: list, tools: list | None = None):
        """Call OpenAI Chat Completions API with optional tools."""
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _parse_tool_args(raw_arguments) -> dict:
        """Parse tool-call arguments defensively for non-compliant model outputs."""
        if isinstance(raw_arguments, dict):
            return raw_arguments

        if raw_arguments is None:
            return {}

        text = str(raw_arguments).strip()
        if not text:
            return {}

        try:
            parsed = json_lib.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except JSONDecodeError:
            # Some models append extra text around JSON; attempt best-effort recovery.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end + 1]
                try:
                    parsed = json_lib.loads(candidate)
                    return parsed if isinstance(parsed, dict) else {}
                except JSONDecodeError:
                    return {}
            return {}

    def call_agent(
        self,
        query: str,
        thread_id: str | None = None,
        max_iterations: int | None = None,
    ) -> str:
        """Run the memory-aware tool-using agent loop and return final answer."""
        active_thread_id = str(thread_id) if thread_id is not None else self.thread_id
        loop_limit = max_iterations if max_iterations is not None else self.max_iterations
        steps = []

        print("\n" + "=" * 50)
        print("BUILDING CONTEXT...")

        memory_context = ""
        memory_context += self.memory_manager.read_conversational_memory(active_thread_id) + "\n\n"
        memory_context += self.memory_manager.read_knowledge_base(query) + "\n\n"
        memory_context += self.memory_manager.read_workflow(query) + "\n\n"
        memory_context += self.memory_manager.read_entity(query) + "\n\n"
        memory_context += self.memory_manager.read_summary_context(query, thread_id=active_thread_id) + "\n\n"

        usage = calculate_context_usage(memory_context, model=self.model)
        print(f"Context: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

        if usage["percent"] > 80:
            print("Context >80% - offloading conversation context to summary memory...")
            memory_context, summaries = offload_to_summary(
                memory_context,
                self.memory_manager,
                self.client,
                thread_id=active_thread_id,
            )
            if summaries:
                summary_ids = [s["id"] for s in summaries]
                print(f"Created {len(summary_ids)} summary reference(s): {summary_ids}")
            usage = calculate_context_usage(memory_context, model=self.model)
            print(f"After offload: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

        context = f"# Question\n{query}\n\n{memory_context}"

        dynamic_tools = self.memory_manager.read_toolbox(query, k=5)
        print(f"Tools: {[t['function']['name'] for t in dynamic_tools]}")

        self.memory_manager.write_conversational_memory(query, "user", active_thread_id)
        try:
            self.memory_manager.write_entity("", "", "", llm_client=self.client, text=query)
        except Exception:
            pass

        system_prompt = self.agent_system_prompt if self.agent_system_prompt else AGENT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
        final_answer = ""

        print("\nAGENT LOOP")
        for iteration in range(loop_limit):
            print(f"\n--- Iteration {iteration + 1} ---")
            response = self.call_openai_chat(messages, tools=dynamic_tools)
            try:
                msg = response.choices[0].message
            except:
                print("response: ", response)

            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_name = tool_name.split("<|channel|>")[0]
                    raw_args = tc.function.arguments
                    tool_args = self._parse_tool_args(raw_args)
                    if not tool_args and str(raw_args).strip() not in {"", "{}", "None"}:
                        print(f"Warning: Invalid tool args for {tool_name}; falling back to empty args.")
                    args_display = {
                        k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                        for k, v in tool_args.items()
                    }
                    print(f"Tool call: {tool_name}({args_display})")

                    try:
                        result = self.execute_tool(tool_name, tool_args, current_thread_id=active_thread_id)
                        status = "success"
                        error_message = None
                        steps.append(f"{tool_name}({args_display}) -> success")
                    except Exception as e:
                        result = f"Error: {e}"
                        status = "failed"
                        error_message = str(e)
                        steps.append(f"{tool_name}({args_display}) -> failed")

                    log_id = self.memory_manager.write_tool_log(
                        thread_id=active_thread_id,
                        tool_call_id=tc.id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result=result,
                        status=status,
                        error_message=error_message,
                        metadata={"iteration": iteration + 1},
                    )

                    if len(result) > 3000:
                        result_for_llm = (
                            result[:3000]
                            + f"\n\n[Truncated for context. Full output saved in TOOL_LOG_MEMORY as log_id: {log_id}]"
                        )
                    else:
                        result_for_llm = result

                    result_display = result_for_llm[:200] + "..." if len(result_for_llm) > 200 else result_for_llm
                    print(f"Result: {result_display}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_for_llm,
                        }
                    )
            else:
                final_answer = msg.content or ""
                print(f"DONE ({len(steps)} tool calls)")
                break
        else:
            print(f"WARNING: Max iterations ({loop_limit}) reached without final answer")
            final_answer = "I was unable to complete the request within the allowed iterations."

        if steps:
            self.memory_manager.write_workflow(query, steps, final_answer)
        try:
            self.memory_manager.write_entity("", "", "", llm_client=self.client, text=final_answer)
        except Exception:
            pass

        self.memory_manager.write_conversational_memory(final_answer, "assistant", active_thread_id)
        return final_answer