from src.MemoryAwareAgent import MemoryAwareAgent
import json as json_lib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    """Run a sample multi-turn workflow using MemoryAwareAgent."""
    agent = MemoryAwareAgent(
        thread_id="50000",
        max_iterations=20
    )

    workspace_root = Path("/app").resolve()

    def _resolve_target(path: str) -> Path | None:
        requested = Path(path)
        target = requested if requested.is_absolute() else (workspace_root / requested)
        target = target.resolve()
        if workspace_root not in target.parents and target != workspace_root:
            return None
        return target

    def _collect_files(path: str, recursive: bool, max_files: int) -> tuple[str | None, list[Path]]:
        target = _resolve_target(path)
        if target is None:
            return f"Refused path outside workspace: {path}", []
        if not target.exists():
            return f"Path does not exist: {target}", []

        if target.is_file():
            files = [target]
        else:
            pattern = "**/*" if recursive else "*"
            files = [p for p in target.glob(pattern) if p.is_file()]

        return None, files[:max_files]

    def read_filesystem_content(
        path: str,
        recursive: bool = False,
        max_files: int = 20,
        max_return_chars_per_file: int = 4000,
    ) -> str:
        """Read file contents from the local workspace so the agent can summarize them.

        Use this tool when the task requires understanding local code/docs.
        It returns JSON with file paths and text excerpts.
        """
        error, files = _collect_files(path=path, recursive=recursive, max_files=max_files)
        if error:
            return error
        if not files:
            return f"No files found at: {path}"

        payload = []
        skipped_files = 0
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                skipped_files += 1
                continue

            relative_path = str(file_path.relative_to(workspace_root))
            payload.append(
                {
                    "path": relative_path,
                    "content": text[:max_return_chars_per_file],
                    "content_truncated": len(text) > max_return_chars_per_file,
                }
            )

        response = {
            "status": "ok",
            "requested_path": path,
            "returned_files": len(payload),
            "skipped_files": skipped_files,
            "files": payload,
        }
        return json_lib.dumps(response, ensure_ascii=False, indent=2)

    def save_filesystem_to_kb(
        path: str,
        recursive: bool = False,
        max_files: int = 20,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ) -> str:
        """Save local file content into knowledge memory for future retrieval.

        This is analogous to arXiv fetch+save, but for workspace filesystem paths.
        """
        error, files = _collect_files(path=path, recursive=recursive, max_files=max_files)
        if error:
            return error
        if not files:
            return f"No files found at: {path}"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        ingested_files = 0
        ingested_chunks = 0
        skipped_files = 0

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                skipped_files += 1
                continue

            if not text.strip():
                skipped_files += 1
                continue

            chunks = splitter.split_text(text)
            if not chunks:
                skipped_files += 1
                continue

            relative_path = str(file_path.relative_to(workspace_root))
            metadatas = [
                {
                    "source": "filesystem",
                    "path": relative_path,
                    "chunk_id": idx,
                    "num_chunks": len(chunks),
                }
                for idx in range(len(chunks))
            ]

            agent.memory_manager.write_knowledge_base(chunks, metadatas)
            ingested_files += 1
            ingested_chunks += len(chunks)

        return (
            f"Filesystem KB save complete. Files ingested: {ingested_files}, "
            f"chunks stored: {ingested_chunks}, files skipped: {skipped_files}."
        )

    def read_filesystem_and_save_to_kb(
        path: str,
        recursive: bool = False,
        max_files: int = 20,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        return_content: bool = True,
        max_return_chars_per_file: int = 4000,
        max_return_files: int = 3,
    ) -> str:
        """Read file(s), return content, and store chunks in knowledge memory.

        This mirrors the arXiv pattern where content can be read first and then saved.
        Here both happen in one tool call for convenience.

        Args:
            path: Absolute or relative path to a file or folder.
            recursive: If path is a folder, traverse subfolders when True.
            max_files: Maximum number of files to ingest from a folder.
            chunk_size: Text chunk size before storing in knowledge memory.
            chunk_overlap: Overlap between consecutive chunks.
            return_content: If True, includes file text in the tool response.
            max_return_chars_per_file: Maximum returned characters per file.
            max_return_files: Maximum number of file contents to include in response.

        Returns:
            JSON string with ingestion summary and optional file content.
        """
        error, files = _collect_files(path=path, recursive=recursive, max_files=max_files)
        if error:
            return error
        if not files:
            return f"No files found at: {path}"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        ingested_files = 0
        ingested_chunks = 0
        skipped_files = 0
        content_payload = []

        for file_path in files[:max_files]:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                skipped_files += 1
                continue
            except Exception:
                skipped_files += 1
                continue

            if not text.strip():
                skipped_files += 1
                continue

            chunks = splitter.split_text(text)
            if not chunks:
                skipped_files += 1
                continue

            relative_path = str(file_path.relative_to(workspace_root))
            metadatas = [
                {
                    "source": "filesystem",
                    "path": relative_path,
                    "chunk_id": idx,
                    "num_chunks": len(chunks),
                }
                for idx in range(len(chunks))
            ]

            agent.memory_manager.write_knowledge_base(chunks, metadatas)
            ingested_files += 1
            ingested_chunks += len(chunks)

            if return_content and len(content_payload) < max_return_files:
                content_payload.append(
                    {
                        "path": relative_path,
                        "content": text[:max_return_chars_per_file],
                        "content_truncated": len(text) > max_return_chars_per_file,
                        "num_chunks": len(chunks),
                    }
                )

        response = {
            "status": "ok",
            "path": path,
            "ingested_files": ingested_files,
            "ingested_chunks": ingested_chunks,
            "skipped_files": skipped_files,
            "returned_files": len(content_payload),
            "files": content_payload,
        }
        return json_lib.dumps(response, ensure_ascii=False, indent=2)

    agent.toolbox.register_tool(read_filesystem_content, augment=True)
    agent.registered_tools["read_filesystem_content"] = read_filesystem_content
    agent.toolbox.register_tool(save_filesystem_to_kb, augment=True)
    agent.registered_tools["save_filesystem_to_kb"] = save_filesystem_to_kb
    agent.toolbox.register_tool(read_filesystem_and_save_to_kb, augment=True)
    agent.registered_tools["read_filesystem_and_save_to_kb"] = read_filesystem_and_save_to_kb

    queries = [
        "Read and summarize the content of files in src/tools. Use read_filesystem_content and save_filesystem_to_kb when useful.",
        # "Can you get me the paper MemGPT",
        # "Can you save the content of the paper",
        # "What are the main key takeaways from the paper",
        # "Summarize the converstation so far using your tool",
        # "What was my first question?",
    ]

    for query in queries:
        answer = agent.call_agent(query)
        print("\n" + "=" * 50 + f"\nANSWER:\n{answer}\n" + "=" * 50)


if __name__ == "__main__":
    main()