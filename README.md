# Memory-Aware Agent

This repository is a hands-on learning project for building a memory-aware agent.

The goal is to help you understand not only how to call an LLM, but how to give the agent durable memory, tool usage, context-window management, and retrieval-augmented behavior over long interactions.

## What You Will Learn

- How to structure an agent around multiple memory types
- How to persist conversation and tool logs in a database
- How to use vector stores for semantic retrieval
- How to compress long conversations into summaries when context grows
- How to register tools and let the agent decide when to call them
- How to run an end-to-end memory-aware agent stack with Docker

## Core Concepts in This Project

This project uses six memory stores:

- Conversation Memory: recent dialogue history per thread
- Knowledge Base Memory: ingested documents for factual grounding
- Workflow Memory: past execution trajectories and successful patterns
- Toolbox Memory: discoverable tool metadata
- Entity Memory: extracted entities (people, systems, topics)
- Summary Memory: compact summaries for long-term continuity

It also stores raw tool outputs in a dedicated tool log table to support observability and replay.

## Repository Structure

```text
.
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── scripts/
│   └── main.sh
└── src/
		├── config.py
		├── main.py
		├── database/
		│   ├── __init__.py
		│   ├── bootstrap.py
		│   ├── connection.py
		│   ├── tables.py
		│   └── vector_indexes.py
		├── managers/
		│   ├── MemoryManager.py
		│   └── StoreManager.py
		├── tools/
		│   ├── ToolMetadata.py
		│   ├── common.py
		│   ├── summary.py
		│   └── toolbox.py
		└── utils/
				├── embedding_model.py
				└── helpers.py
```

## High-Level Flow

1. Database bootstrap creates and validates Oracle user access.
2. SQL and vector memory stores are initialized.
3. Agent builds a context window from multiple memories.
4. Agent chooses tools dynamically via tool metadata retrieval.
5. Tool outputs are logged and optionally truncated for context safety.
6. When context usage is high, conversations are summarized and offloaded.
7. Final assistant output and workflow pattern are persisted.

## Prerequisites

- Docker + Docker Compose
- NVIDIA GPU runtime (recommended for local vLLM service)
- Internet access (for model/paper retrieval and package downloads)

## Quick Start

### 1) Build and start services

```bash
docker compose up --build
```

This starts:

- python-project service
- Oracle Free database
- vLLM OpenAI-compatible model server

### 2) Run the agent script

```bash
bash scripts/main.sh
```

This runs [scripts/main.sh](scripts/main.sh), which executes [src/main.py](src/main.py).

## Configuration

Main configuration lives in [src/config.py](src/config.py), including:

- Oracle connection and credentials
- Table names for memory stores
- Embedding model name
- LLM model/base URL/API key
- Agent system prompt
- Token limit settings for context monitoring