from langchain_community.vectorstores.utils import DistanceStrategy
from openai import OpenAI
import json as json_lib

from src.database.bootstrap import database_connection
from src.utils.embedding_model import embedding_model
from src.config import (
    KNOWLEDGE_BASE_TABLE,
    WORKFLOW_TABLE,
    TOOLBOX_TABLE,
    ENTITY_TABLE,
    SUMMARY_TABLE,
    CONVERSATIONAL_TABLE,
    TOOL_LOG_TABLE,
    MODEL_NAME,
    API_BASE,
    API_KEY,
    AGENT_SYSTEM_PROMPT
)
from src.database.vector_indexes import safe_create_index
from src.database.tables import (
    create_conversational_history_table,
    create_tool_log_table,
)
from src.managers.StoreManager import StoreManager
from src.managers.MemoryManager import MemoryManager
from src.tools.toolbox import Toolbox
from src.tools.common import register_common_tools
from src.tools.summary import (
    calculate_context_usage,
    offload_to_summary,
)


CONVERSATION_HISTORY_TABLE = create_conversational_history_table(database_connection, CONVERSATIONAL_TABLE)
TOOL_LOG_HISTORY_TABLE = create_tool_log_table(database_connection, TOOL_LOG_TABLE)


# Create StoreManager instance
store_manager = StoreManager(
    client=database_connection,
    embedding_function=embedding_model,
    table_names={
        'knowledge_base': KNOWLEDGE_BASE_TABLE,
        'workflow': WORKFLOW_TABLE,
        'toolbox': TOOLBOX_TABLE,
        'entity': ENTITY_TABLE,
        'summary': SUMMARY_TABLE,
    },
    distance_strategy=DistanceStrategy.COSINE,
    conversational_table=CONVERSATION_HISTORY_TABLE,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)


conversation_table = store_manager.get_conversational_table()
knowledge_base_vs = store_manager.get_knowledge_base_store()
workflow_vs = store_manager.get_workflow_store()
toolbox_vs = store_manager.get_toolbox_store()
entity_vs = store_manager.get_entity_store()
summary_vs = store_manager.get_summary_store()
tool_log_table = store_manager.get_tool_log_table()


print("Creating vector indexes...")
safe_create_index(database_connection, knowledge_base_vs, "knowledge_base_vs_ivf")
safe_create_index(database_connection, workflow_vs, "workflow_vs_ivf")
safe_create_index(database_connection, toolbox_vs, "toolbox_vs_ivf")
safe_create_index(database_connection, entity_vs, "entity_vs_ivf")
safe_create_index(database_connection, summary_vs, "summary_vs_ivf")
print("All indexes created!")


memory_manager = MemoryManager(
    conn=database_connection,
    conversation_table=CONVERSATION_HISTORY_TABLE, 
    knowledge_base_vs=knowledge_base_vs,
    workflow_vs=workflow_vs,
    toolbox_vs=toolbox_vs,
    entity_vs=entity_vs,
    summary_vs=summary_vs,
    tool_log_table=TOOL_LOG_HISTORY_TABLE
)


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)


toolbox = Toolbox(memory_manager, client, embedding_model)


common_tools = register_common_tools(toolbox, memory_manager, KNOWLEDGE_BASE_TABLE)


def execute_tool(tool_name: str, tool_args: dict, current_thread_id: str | None = None) -> str:
    """Execute a tool by looking it up in the toolbox."""

    if tool_name not in toolbox._tools_by_name:
        return f"Error: Tool '{tool_name}' not found"

    args = dict(tool_args or {})

    # Ensure conversation summarization marks source rows in the active thread.
    if tool_name == "summarize_and_store" and "thread_id" not in args and current_thread_id is not None:
        args["thread_id"] = str(current_thread_id)

    return str(toolbox._tools_by_name[tool_name](**args) or "Done")


# ==================== OPENAI CHAT FUNCTION ====================
def call_openai_chat(messages: list, tools: list = None, model: str = MODEL_NAME):
    """Call OpenAI Chat Completions API with tools."""
    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


# ==================== MAIN AGENT LOOP ====================
def call_agent(query: str, thread_id: str = "1", max_iterations: int = 10) -> str:
    """Agent loop with context window monitoring and summarization."""
    thread_id = str(thread_id)
    steps = []
    summaries = []  # Track created summaries
    
    # 1. Build context from memory
    print("\n" + "="*50)
    print("🧠 BUILDING CONTEXT...")
    
    # Build memory context (excluding query for now)
    memory_context = ""
    memory_context += memory_manager.read_conversational_memory(thread_id) + "\n\n"
    memory_context += memory_manager.read_knowledge_base(query) + "\n\n"
    memory_context += memory_manager.read_workflow(query) + "\n\n"
    memory_context += memory_manager.read_entity(query) + "\n\n"
    memory_context += memory_manager.read_summary_context(query, thread_id=thread_id) + "\n\n"  # Shows IDs + descriptions (thread-scoped when available)
    
    # 2. Check context usage - summarize if >80%
    usage = calculate_context_usage(memory_context)
    print(f"📊 Context: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")
    
    if usage['percent'] > 80:
        print("⚠️ Context >80% - offloading conversation context to summary memory...")
        memory_context, summaries = offload_to_summary(
            memory_context,
            memory_manager,
            client,
            thread_id=thread_id,
        )
        if summaries:
            print(f"🧾 Created {len(summaries)} summary reference(s): {[s['id'] for s in summaries]}")
        usage = calculate_context_usage(memory_context)
        print(f"📊 After offload: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")
    
    # Now prepend the query (always preserved, never summarized)
    context = f"# Question\n{query}\n\n{memory_context}"

    print("====CONTEXT WINDOW=====\n")
    print(context)
    
    # 3. Get tools
    dynamic_tools = memory_manager.read_toolbox(query, k=5)
    print(f"🔧 Tools: {[t['function']['name'] for t in dynamic_tools]}")
    
    # 4. Store user message & extract entities
    memory_manager.write_conversational_memory(query, "user", thread_id)
    try:
        memory_manager.write_entity("", "", "", llm_client=client, text=query)
    except Exception:
        pass
    
    # 5. Agent loop
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": context}]
    final_answer = ""
    
    print("\n🤖 AGENT LOOP")
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        response = call_openai_chat(messages, tools=dynamic_tools)
        msg = response.choices[0].message
        
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json_lib.loads(tc.function.arguments)
                # Format args for display (truncate long values)
                args_display = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) 
                               for k, v in tool_args.items()}
                print(f"🛠️ {tool_name}({args_display})")
                
                try:
                    result = execute_tool(tool_name, tool_args, current_thread_id=thread_id)
                    status = "success"
                    error_message = None
                    steps.append(f"{tool_name}({args_display}) → success")
                except Exception as e:
                    result = f"Error: {e}"
                    status = "failed"
                    error_message = str(e)
                    steps.append(f"{tool_name}({args_display}) → failed")

                # Persist full tool output to TOOL_LOG_MEMORY
                log_id = memory_manager.write_tool_log(
                    thread_id=thread_id,
                    tool_call_id=tc.id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    status=status,
                    error_message=error_message,
                    metadata={"iteration": iteration + 1},
                )

                # Next call gets only the immediate tool result (bounded for context control)
                if len(result) > 3000:
                    result_for_llm = result[:3000] + f"\n\n[Truncated for context. Full output saved in TOOL_LOG_MEMORY as log_id: {log_id}]"
                else:
                    result_for_llm = result

                result_display = result_for_llm[:200] + "..." if len(result_for_llm) > 200 else result_for_llm
                print(f"   → {result_display}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_for_llm})
        else:
            final_answer = msg.content or ""
            print(f"\n✅ DONE ({len(steps)} tool calls)")
            break
    else:
        # Max iterations reached without final answer
        print(f"\n⚠️ WARNING: Max iterations ({max_iterations}) reached without final answer")
        final_answer = "I was unable to complete the request within the allowed iterations."
    
    # 6. Save workflow & entities
    if steps:
        memory_manager.write_workflow(query, steps, final_answer)
    try:
        memory_manager.write_entity("", "", "", llm_client=client, text=final_answer)
    except Exception:
        pass
    memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)
    
    print("\n" + "="*50 + f"\n💬 ANSWER:\n{final_answer}\n" + "="*50)
    return final_answer


# ==================== TEST AGENT ====================
call_agent("Can you get me the paper MemGPT", thread_id="50000")
call_agent("Can you save the content of the paper", thread_id="50000")
call_agent("What are the main key takeaways from the paper", thread_id="50000")
call_agent("Summarize the converstation so far using your tool", thread_id="50000")
call_agent("What was my first question?", thread_id="50000")