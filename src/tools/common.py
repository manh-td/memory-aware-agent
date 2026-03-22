"""Common tools available across all lessons."""

import json as json_lib
from datetime import datetime, timezone

from src.config import MODEL_NAME
from src.tools.toolbox import Toolbox
from src.managers.MemoryManager import MemoryManager

# ====================  COMMON TOOLS ====================

def register_common_tools(
        toolbox: Toolbox,
        memory_manager: MemoryManager,
        knowledge_base_table: str = "SEMANTIC_MEMORY",
):
    """
    Register common tools that should be available across all sessons.

    This function creates and registers tools for:
    - ArXiv paper search
    - ArXiv paper fetch and save to knowledge base
    - Get current time

    Args:
        toolbox: The Toolbox instance to register tools with
        memory_manager: The MemoryManager instance for storing data
        knowledge_base_table: Name of the knowledge base table

    Returns:
        dict: Dictionary of registered tool names to their functions
    """
    from langchain_community.retrievers import ArxivRetriever
    from langchain_community.document_loaders import ArxivLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from urllib.parse import urlparse

    # Create the ArXiv retriever
    arxiv_retriever = ArxivRetriever(
        load_max_docs=8,
        get_full_documents=False,
        doc_content_chars_max=4000
    )

    def _arxiv_id_from_entry_id(entry_id: str) -> str:
        """Convert 'http://arxiv.org/abs/2310.08560v2' -> '2310.08560v2'"""
        if not entry_id:
            return ""
        path = urlparse(entry_id).path
        return path.split("/abs/")[-1].strip("/")

    # Define the tools
    def arxiv_search_candidates(query: str, k: int = 5) -> str:
        """
        Search arXiv and return a JSON list of candidate papers with IDs + metadata.

        Output schema (JSON string):
        [
          {
            "arxiv_id": "2310.08560v2",
            "entry_id": "http://arxiv.org/abs/2310.08560v2",
            "title": "...",
            "authors": "...",
            "published": "2024-02-12",
            "abstract": "..."
          },
          ...
        ]
        """
        docs = arxiv_retriever.invoke(query)
        candidates = []
        for d in (docs or [])[:k]:
            meta = d.metadata or {}
            entry_id = meta.get("Entry ID", "")
            candidates.append({
                "arxiv_id": _arxiv_id_from_entry_id(entry_id),
                "entry_id": entry_id,
                "title": meta.get("Title", ""),
                "authors": meta.get("Authors", ""),
                "published": str(meta.get("Published", "")),
                "abstract": (d.page_content or "")[:2500],
            })
        return json_lib.dumps(candidates, ensure_ascii=False, indent=2)

    def fetch_and_save_paper_to_kb_db(
            arxiv_id: str,
            chunk_size: int = 1500,
            chunk_overlap: int = 200,
    ) -> str:
        """
        Fetch full arXiv paper text (PDF -> text) and store it into the OracleVS
        knowledge base table as chunked records.
        """
        loader = ArxivLoader(
            query=arxiv_id,
            load_max_docs=1,
            doc_content_chars_max=None,
        )
        docs = loader.load()
        if not docs:
            return f"No documents found for arXiv id: {arxiv_id}"

        doc = docs[0]
        title = (
                doc.metadata.get("Title")
                or doc.metadata.get("title")
                or f"arXiv {arxiv_id}"
        )

        entry_id = doc.metadata.get("Entry ID") or doc.metadata.get("entry_id") or ""
        published = doc.metadata.get("Published") or doc.metadata.get("published") or ""
        authors = doc.metadata.get("Authors") or doc.metadata.get("authors") or ""

        full_text = doc.page_content or ""
        if not full_text.strip():
            return f"Loaded arXiv {arxiv_id} but extracted empty text (PDF parsing issue)."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(full_text)

        ts_utc = datetime.now(timezone.utc).isoformat()
        metadatas = []
        for i in range(len(chunks)):
            metadatas.append({
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": title,
                "entry_id": entry_id,
                "published": str(published),
                "authors": str(authors),
                "chunk_id": i,
                "num_chunks": len(chunks),
                "ingested_ts_utc": ts_utc,
            })

        memory_manager.write_knowledge_base(chunks, metadatas)

        return (
            f"Saved arXiv {arxiv_id} to {knowledge_base_table}: "
            f"{len(chunks)} chunks (title: {title})."
        )

    def get_current_time(detailed: bool = False) -> str:
        """
        Returns the current time.

        Args:
            detailed: If True, returns detailed format with microseconds

        Returns:
            str: Current time as formatted string
        """
        if detailed:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        else:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Register the tools with the toolbox
    toolbox.register_tool(arxiv_search_candidates, augment=False)
    toolbox.register_tool(fetch_and_save_paper_to_kb_db, augment=True)
    toolbox.register_tool(get_current_time, augment=False)

    # Store references so they can be called directly
    registered_tools = {
        "arxiv_search_candidates": arxiv_search_candidates,
        "fetch_and_save_paper_to_kb_db": fetch_and_save_paper_to_kb_db,
        "get_current_time": get_current_time,
    }

    print(f"✅ Registered {len(registered_tools)} common tools: {list(registered_tools.keys())}")

    return registered_tools
