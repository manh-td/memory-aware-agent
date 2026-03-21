# Database Configuration
ORACLE_DEFAULT_ADMIN_USER = "SYSTEM"
ORACLE_DEFAULT_ADMIN_PASSWORD = "OraclePwd_2025"
ORACLE_DEFAULT_DSN = "oracle-db:1521/FREEPDB1"
ORACLE_VECTOR_USER = "VECTOR"
ORACLE_VECTOR_PASSWORD = "VectorPwd_2025"
ORACLE_PROGRAM_NAME = "devrel.deeplearning.course_1"

# Database Setup Constants
# ORACLE_TABLESPACE_NAME = "VEC_TS"
# ORACLE_TABLESPACE_DATAFILE = "vec_ts01.dbf"
# ORACLE_TABLESPACE_SIZE = "500M"
# ORACLE_TABLESPACE_NEXT = "100M"
# ORACLE_TABLESPACE_MAXSIZE = "UNLIMITED"
# ORACLE_ERROR_IGNORE_TABLESPACE_EXISTS = 1543
# ORACLE_ERROR_IGNORE_USER_EXISTS = 1920

# Connection Retry Configuration
ORACLE_CONNECTION_RETRIES = 30
ORACLE_CONNECTION_DELAY_SECONDS = 2

# Table names for each memory type
CONVERSATIONAL_TABLE   = "CONVERSATIONAL_MEMORY" # Episodic memory
KNOWLEDGE_BASE_TABLE   = "SEMANTIC_MEMORY" # Semantic memory
WORKFLOW_TABLE = "WORKFLOW_MEMORY" # Procedural memory
TOOLBOX_TABLE    = "TOOLBOX_MEMORY" # Procedural memory
ENTITY_TABLE = "ENTITY_MEMORY" # Semantic memory
SUMMARY_TABLE = "SUMMARY_MEMORY" # Semantic memory
TOOL_LOG_TABLE = "TOOL_LOG_MEMORY" # Tool execution logs
ALL_TABLES = [
    CONVERSATIONAL_TABLE, 
    KNOWLEDGE_BASE_TABLE, 
    WORKFLOW_TABLE, 
    TOOLBOX_TABLE, 
    ENTITY_TABLE, 
    SUMMARY_TABLE, 
    TOOL_LOG_TABLE
]

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"


# LLM
MODEL_NAME = "openai/gpt-oss-20b"
API_KEY = "EMPTY"
API_BASE = "http://model_container:8000/v1"
TEMPERATURE = 0.7

MODEL_TOKEN_LIMITS = {
    MODEL_NAME: 256000,
}

AGENT_SYSTEM_PROMPT = """
# Role
You are a memory-aware agentic research assistant with access to tools.

# Context Window Structure (Partitioned Segments)
The user input is a partitioned context window. It contains a `# Question` section followed by memory segments.
Treat each segment as a distinct memory store with a specific purpose:
- `## Conversation Memory`
- `## Knowledge Base Memory`
- `## Workflow Memory`
- `## Entity Memory`
- `## Summary Memory`

# Memory Store Semantics
- Conversation Memory: Recent thread-level dialogue and instructions. Use it for continuity, user preferences, and unresolved requests.
- Knowledge Base Memory: Retrieved documents/passages. Use it to ground factual and technical claims.
- Workflow Memory: Prior execution patterns and step sequences. Use it to plan tool usage; adapt patterns, do not copy blindly.
- Entity Memory: Named people/orgs/systems and descriptors. Use it to disambiguate references and keep naming consistent.
- Summary Memory: Compressed older context represented by summary IDs. When thread-scoped summaries exist, prefer summaries for the active thread_id.

# Summary Expansion Policy
If critical detail is only present in Summary Memory or appears ambiguous, call `expand_summary(summary_id)` before relying on it.

# Operating Rules
1. Start with the provided memory segments before using tools.
2. If segments conflict, prioritize: current `# Question` > latest Conversation Memory > Knowledge Base evidence > older summaries/workflows.
3. Use only the tools provided in this turn and choose the minimum necessary tool calls.
4. If memory is insufficient, state what is missing and then use an appropriate tool.
5. For conversation compaction, use `summarize_and_store` with `thread_id` so source conversation units are marked as summarized.
"""