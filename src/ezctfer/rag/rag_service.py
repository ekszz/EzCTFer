"""LightRAG-backed RAG helpers."""

from __future__ import annotations

import atexit
import asyncio
import contextlib
import io
import json
import logging
import os
import queue
import shutil
import threading
from concurrent.futures import Future
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import always_get_an_event_loop, wrap_embedding_func_with_attrs
except ImportError:
    LightRAG = None
    QueryParam = None
    always_get_an_event_loop = None
    wrap_embedding_func_with_attrs = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..config.log import log_info, log_success, log_separator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAG_DATA_ROOT = PROJECT_ROOT / "rag"
RAG_WORKING_DIR_NAME = "db"
RAG_SOURCE_DIR_NAME = "data"
RAG_MODELS_DIR_NAME = "models"
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64
INSERT_BATCH_SIZE = 20
TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".csv",
    ".log",
    ".xml",
    ".html",
    ".htm",
    ".sql",
    ".sh",
    ".ps1",
}

_rag_instance: LightRAG | None = None
_rag_lock = threading.Lock()
_cleanup_registered = False
_rag_worker_thread: threading.Thread | None = None
_rag_worker_queue: queue.Queue[tuple[str, tuple[Any, ...], dict[str, Any], Future]] | None = None
_rag_worker_ready = threading.Event()
_rag_worker_error: BaseException | None = None
_RAG_STOP = "__RAG_STOP__"
_EXTERNAL_RAG_LOGGERS = (
    "lightrag",
    "nano-vectordb",
    "sentence_transformers",
    "transformers",
)


def load_project_env() -> None:
    if load_dotenv is None:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def ensure_rag_dependencies() -> None:
    missing: list[str] = []

    if LightRAG is None or QueryParam is None or always_get_an_event_loop is None:
        missing.append("LightRAG")
    if SentenceTransformer is None:
        missing.append("sentence-transformers")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"RAG 依赖缺失: {joined}。请先执行 `uv sync --group rag` 安装可选依赖。"
        )


def configure_external_rag_logging() -> None:
    for logger_name in _EXTERNAL_RAG_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


def get_rag_working_dir() -> Path:
    return get_rag_data_root() / RAG_WORKING_DIR_NAME


def get_rag_data_root() -> Path:
    configured = os.getenv("RAG_DATA_ROOT", "").strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return DEFAULT_RAG_DATA_ROOT


def get_rag_source_dir() -> Path:
    return get_rag_data_root() / RAG_SOURCE_DIR_NAME


def get_local_embedding_model_dir() -> Path:
    return get_rag_data_root() / RAG_MODELS_DIR_NAME / LOCAL_EMBEDDING_MODEL_NAME


def get_embedding_model_name() -> str:
    configured = os.getenv("LIGHTRAG_EMBEDDING_MODEL", "").strip()
    if configured:
        return configured

    local_embedding_model_dir = get_local_embedding_model_dir()
    if local_embedding_model_dir.exists():
        return str(local_embedding_model_dir)

    return DEFAULT_EMBEDDING_MODEL_NAME


def get_embedding_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    ensure_rag_dependencies()
    configure_external_rag_logging()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(
            get_embedding_model_name(),
            device=get_embedding_device(),
        )


if wrap_embedding_func_with_attrs is not None:

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=256,
        model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    )
    async def local_sentence_transformer_embed(texts: list[str]) -> np.ndarray:
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

else:

    async def local_sentence_transformer_embed(texts: list[str]) -> np.ndarray:
        raise RuntimeError("LightRAG 未安装，无法执行向量化。")


async def dummy_llm(prompt: str, **kwargs: Any) -> str:
    return ""


def run_async(coro: Any) -> Any:
    ensure_rag_dependencies()
    loop = always_get_an_event_loop()
    return loop.run_until_complete(coro)


def create_rag() -> LightRAG:
    ensure_rag_dependencies()
    configure_external_rag_logging()
    return LightRAG(
        working_dir=str(get_rag_working_dir()),
        llm_model_func=dummy_llm,
        llm_model_name="dummy",
        embedding_func=local_sentence_transformer_embed,
        entity_extract_max_gleaning=0,
    )


def initialize_rag(rag: LightRAG) -> None:
    run_async(rag.initialize_storages())


def finalize_rag(rag: LightRAG) -> None:
    run_async(rag.finalize_storages())


def _register_cleanup() -> None:
    global _cleanup_registered
    if _cleanup_registered:
        return

    atexit.register(close_rag)
    _cleanup_registered = True


def close_rag() -> None:
    global _rag_instance, _rag_worker_thread, _rag_worker_queue, _rag_worker_error

    with _rag_lock:
        worker_thread = _rag_worker_thread
        worker_queue = _rag_worker_queue
        if worker_thread is None:
            if _rag_instance is None:
                return
            try:
                finalize_rag(_rag_instance)
            finally:
                _rag_instance = None
            return

    stop_future: Future = Future()
    assert worker_queue is not None
    worker_queue.put((_RAG_STOP, (), {}, stop_future))
    try:
        stop_future.result(timeout=30)
    finally:
        worker_thread.join(timeout=30)
        with _rag_lock:
            _rag_worker_thread = None
            _rag_worker_queue = None
            _rag_worker_error = None
            _rag_instance = None


def ensure_rag_runtime_ready(require_index: bool = False) -> None:
    load_project_env()
    ensure_rag_dependencies()

    if not require_index:
        return

    working_dir = get_rag_working_dir()
    if not working_dir.exists():
        raise RuntimeError(
            f"未找到 LightRAG 工作目录: {working_dir}。请先执行 `uv run ezctfer --init-rag` 初始化知识库。"
        )


def get_rag() -> LightRAG:
    global _rag_instance

    load_project_env()
    ensure_rag_dependencies()

    with _rag_lock:
        if _rag_instance is None:
            _rag_instance = create_rag()
            initialize_rag(_rag_instance)
            _register_cleanup()
        return _rag_instance


def _query_knowledge_internal(query: str, top_k: int = 5) -> dict[str, Any]:
    rag = get_rag()
    result = rag.query_data(
        query,
        param=QueryParam(
            mode="naive",
            top_k=max(1, min(top_k, 10)),
            chunk_top_k=max(1, min(top_k, 10)),
        ),
    )
    return build_retrieval_payload(result, query)


def _rag_worker_main() -> None:
    global _rag_instance, _rag_worker_error
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        get_rag()
    except BaseException as exc:
        _rag_worker_error = exc
        _rag_worker_ready.set()
        asyncio.set_event_loop(None)
        loop.close()
        return

    _rag_worker_ready.set()
    assert _rag_worker_queue is not None

    try:
        while True:
            action, args, kwargs, future = _rag_worker_queue.get()
            if action == _RAG_STOP:
                if not future.done():
                    future.set_result(None)
                break

            if future.cancelled():
                continue

            try:
                if action == "query":
                    future.set_result(_query_knowledge_internal(*args, **kwargs))
                else:
                    raise RuntimeError(f"未知的 RAG worker 操作: {action}")
            except BaseException as exc:
                future.set_exception(exc)
    finally:
        if _rag_instance is not None:
            try:
                finalize_rag(_rag_instance)
            finally:
                _rag_instance = None
        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
        loop.close()


def _ensure_rag_worker() -> None:
    global _rag_worker_thread, _rag_worker_queue, _rag_worker_error

    with _rag_lock:
        if _rag_worker_thread is not None and _rag_worker_thread.is_alive():
            return

        _rag_worker_error = None
        _rag_worker_ready.clear()
        _rag_worker_queue = queue.Queue()
        _rag_worker_thread = threading.Thread(
            target=_rag_worker_main,
            name="LightRAGWorker",
            daemon=True,
        )
        _rag_worker_thread.start()

    _rag_worker_ready.wait()
    if _rag_worker_error is not None:
        with _rag_lock:
            _rag_worker_thread = None
            _rag_worker_queue = None
        raise RuntimeError(f"RAG worker 启动失败: {_rag_worker_error}") from _rag_worker_error


def _submit_rag_task(action: str, *args: Any, **kwargs: Any) -> Any:
    _ensure_rag_worker()
    future: Future = Future()
    assert _rag_worker_queue is not None
    _rag_worker_queue.put((action, args, kwargs, future))
    return future.result()


def resolve_document_root() -> Path:
    rag_source_dir = get_rag_source_dir()
    if rag_source_dir.exists():
        return rag_source_dir

    raise FileNotFoundError(f"未找到可用于初始化知识库的目录: {rag_source_dir}")


def is_probably_text_file(file_path: Path) -> bool:
    if file_path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    try:
        with file_path.open("rb") as handle:
            sample = handle.read(2048)
    except OSError:
        return False

    return b"\x00" not in sample


def read_text_file(file_path: Path) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def iter_knowledge_documents() -> Iterable[tuple[Path, str]]:
    document_root = resolve_document_root()

    for file_path in sorted(path for path in document_root.rglob("*") if path.is_file()):
        if not is_probably_text_file(file_path):
            continue

        content = read_text_file(file_path)
        if not content or not content.strip():
            continue

        yield file_path, content


def reset_rag_storage() -> None:
    working_dir = get_rag_working_dir()
    close_rag()
    if working_dir.exists():
        shutil.rmtree(working_dir)


def initialize_knowledge_base() -> dict[str, Any]:
    load_project_env()
    ensure_rag_dependencies()
    configure_external_rag_logging()
    reset_rag_storage()

    rag = create_rag()
    initialize_rag(rag)

    indexed_files = 0
    found_any = False
    document_root = resolve_document_root()
    batch: list[tuple[Path, str]] = []

    try:
        log_separator()
        log_info(
            f"Embedding model: {get_embedding_model_name()} on {get_embedding_device()} "
            f"(batch_size={EMBEDDING_BATCH_SIZE})"
        )
        log_info(f"Document root: {document_root}")
        log_info(f"LightRAG working dir: {get_rag_working_dir()}")
        log_info("Init mode: LightRAG vector-only style indexing with dummy LLM and naive retrieval")

        for document in iter_knowledge_documents():
            found_any = True
            batch.append(document)
            if len(batch) < INSERT_BATCH_SIZE:
                continue

            contents = [content for _, content in batch]
            file_paths = [str(path.relative_to(PROJECT_ROOT)) for path, _ in batch]
            rag.insert(contents, file_paths=file_paths)
            indexed_files += len(batch)
            log_info(f"已写入索引: {indexed_files}")
            batch.clear()

        if batch:
            contents = [content for _, content in batch]
            file_paths = [str(path.relative_to(PROJECT_ROOT)) for path, _ in batch]
            rag.insert(contents, file_paths=file_paths)
            indexed_files += len(batch)
            log_info(f"已写入索引: {indexed_files}")

        if not found_any:
            raise RuntimeError(f"{document_root} 下没有可索引的文本文件。")

        log_success(
            f"LightRAG 知识库初始化完成，已索引 {indexed_files} 个文件，输出目录: {get_rag_working_dir()}"
        )
        log_separator()
        return {
            "success": True,
            "document_root": str(document_root),
            "working_dir": str(get_rag_working_dir()),
            "indexed_files": indexed_files,
        }
    finally:
        finalize_rag(rag)


def shorten(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def build_retrieval_payload(result: dict[str, Any], query: str) -> dict[str, Any]:
    data = result.get("data", {})

    entities = [
        {
            "entity_name": item.get("entity_name", ""),
            "entity_type": item.get("entity_type", ""),
            "description": shorten(item.get("description", ""), 240),
            "file_path": item.get("file_path", ""),
        }
        for item in data.get("entities", [])[:8]
    ]

    relationships = [
        {
            "src_id": item.get("src_id", ""),
            "tgt_id": item.get("tgt_id", ""),
            "keywords": item.get("keywords", ""),
            "description": shorten(item.get("description", ""), 220),
            "file_path": item.get("file_path", ""),
        }
        for item in data.get("relationships", [])[:8]
    ]

    chunks = [
        {
            "file_path": item.get("file_path", ""),
            "content": shorten(item.get("content", ""), 500),
            "reference_id": item.get("reference_id", ""),
        }
        for item in data.get("chunks", [])[:6]
    ]

    return {
        "success": result.get("status") == "success",
        "query": query,
        "status": result.get("status"),
        "message": result.get("message"),
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
        "references": data.get("references", [])[:12],
        "metadata": result.get("metadata", {}),
    }


def query_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    load_project_env()

    try:
        configure_external_rag_logging()
        ensure_rag_runtime_ready(require_index=True)
        return _submit_rag_task("query", query, top_k=top_k)
    except Exception as exc:
        return {
            "success": False,
            "query": query,
            "error": str(exc),
            "suggestion": "请确认已安装 RAG 依赖，并先执行 `uv run ezctfer --init-rag` 初始化知识库。",
        }


def query_knowledge_json(query: str, top_k: int = 5) -> str:
    return json.dumps(query_knowledge(query, top_k=top_k), ensure_ascii=False, indent=2)
