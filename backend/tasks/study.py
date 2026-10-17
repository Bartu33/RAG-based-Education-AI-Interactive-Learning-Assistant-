"""Interactive Learning: a study buddy that only looks at the ONE document the student has open.

Why it is fast: the open document's chunks and vectors are read from the store once and kept in
memory (until the next write to the store). A question is then one local embedding plus a
matrix-vector product over those few chunks, with no search over the whole course and, by
default, no reranker and no faithfulness check. Only the LLM call remains, and it is streamed.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from backend.config import get_config, resolve_path
from backend.indexing.vectorstore import VectorStore, and_filter, get_store
from backend.llm.prompts import build_study_messages
from backend.llm.providers import chat_with_fallback, resolve_provider_name, stream_with_fallback
from backend.schemas import format_ms

MODES = ("ask", "summary", "explain", "concepts")
DEFAULTS = {
    "top_context": 4,            # excerpts sent with a question
    "page_boost": 0.05,          # bonus for chunks on the page / moment the student is looking at
    "use_reranker": False,       # true = rerank the top candidates (slower, sometimes better)
    "rerank_candidates": 10,
    "overview_chunks": 12,       # excerpts sent for summary / key concepts
    "max_chunk_chars": 1500,     # each overview excerpt is cut to this length
    "explain_window_ms": 60000,  # video: "explain this moment" uses +/- this window
    "history_turns": 3,          # earlier question/answer pairs sent for follow-up questions
    "max_tokens": 900,
    "max_tokens_long": 1600,     # summary / key concepts / explain
}
_CACHE: dict = {}


def study_cfg() -> dict:
    return {**DEFAULTS, **(get_config().get("study") or {})}


# --- the open document ------------------------------------------------------------------------
def _chunk_order(hits: list[dict], course: str, document: str) -> list[dict]:
    """Chunks in reading order. The store keeps no position, but chunk ids are
    md5(course|lecture|document|i) (see ChromaStore.add), so i can be recovered from the id."""
    position: dict[str, int] = {}
    for lecture in {h["metadata"].get("lecture", 0) for h in hits}:
        for i in range(len(hits)):
            key = hashlib.md5(f"{course}|{lecture}|{document}|{i}".encode()).hexdigest()
            position[key] = i
    far = len(hits)

    def key(h):
        m = h["metadata"]
        return (position.get(h["id"], far), m.get("page") or 0, m.get("timestamp_start_ms") or 0)
    return sorted(hits, key=key)


def document_chunks(course: str, document: str, store: Optional[VectorStore] = None) -> dict:
    """{"hits": [...] in reading order, "matrix": (n, dim) float32}. Cached per store version."""
    store = store or get_store()
    key = (id(store), course, document)
    cached = _CACHE.get(key)
    if cached and cached[0] == store.version:
        return cached[1]
    where = and_filter([{"course": {"$eq": course}}, {"document": {"$eq": document}}])
    hits = store.get(where=where, with_embeddings=True)
    if not hits:
        raise LookupError(f"'{document}' is not indexed for {course}.")
    hits = _chunk_order(hits, course, document)
    matrix = np.vstack([np.asarray(h.pop("embedding"), dtype=np.float32) for h in hits])
    entry = {"hits": hits, "matrix": matrix}
    _CACHE[key] = (store.version, entry)
    return entry


def upload_path(course: str, document: str) -> Path:
    """The original uploaded file (data/uploads/<course>/<document>). Rejects path tricks."""
    if not course or Path(course).name != course or course in (".", ".."):
        raise ValueError("Invalid course")
    if not document or Path(document).name != document or document in (".", ".."):
        raise ValueError("Invalid document name")
    path = resolve_path("data/uploads") / course / document
    if not path.is_file():
        raise FileNotFoundError(f"The original file of {document} is not on the server.")
    return path


def _location_label(meta: dict) -> str:
    kind = meta["source_type"]
    if kind == "video" and meta.get("timestamp_start_ms") is not None:
        return f"{format_ms(meta['timestamp_start_ms'])}–{format_ms(meta['timestamp_end_ms'])}"
    if kind == "pptx" and meta.get("page"):
        return f"Slide {meta['page']}"
    if kind == "pdf" and meta.get("page"):
        return f"Page {meta['page']}"
    return meta.get("section") or "Text"


def outline(course: str, document: str, store: Optional[VectorStore] = None) -> dict:
    """What the viewer needs: the document's units (pages / slides / sections / transcript parts)
    with their indexed text, and whether the original file can be shown."""
    hits = document_chunks(course, document, store)["hits"]
    meta0 = hits[0]["metadata"]
    kind = meta0["source_type"]
    units: list[dict] = []
    for h in hits:
        m = h["metadata"]
        if kind in ("pdf", "pptx"):
            same = units and units[-1]["page"] == m.get("page")
        elif kind == "docx":
            same = units and units[-1]["section"] == (m.get("section") or None)
        else:  # video: one unit per transcript chunk
            same = False
        if same:
            units[-1]["text"] += "\n\n" + h["text"]
            units[-1]["end_ms"] = m.get("timestamp_end_ms")
            continue
        units.append({"label": _location_label(m), "page": m.get("page"),
                      "section": m.get("section") or None,
                      "start_ms": m.get("timestamp_start_ms"), "end_ms": m.get("timestamp_end_ms"),
                      "text": h["text"]})
    if kind in ("pdf", "pptx"):
        units.sort(key=lambda u: u["page"] or 0)
    if kind == "docx":
        for i, u in enumerate(units, 1):
            if not u["section"]:
                u["label"] = f"Part {i}"

    try:
        path = upload_path(course, document)
    except (FileNotFoundError, ValueError):
        path = None
    pages = None
    if kind == "pdf":
        pages = max((u["page"] or 0) for u in units) or None
        if path is not None:
            try:
                import fitz  # PyMuPDF
                with fitz.open(path) as doc:
                    pages = doc.page_count
            except Exception:  # noqa: BLE001  (damaged file: fall back to indexed pages)
                pass
    return {"document": document, "course": course, "lecture": meta0.get("lecture"),
            "type": kind, "chunks": len(hits), "pages": pages, "has_file": path is not None,
            "units": units}


@lru_cache(maxsize=64)
def _render(path: str, mtime: float, page: int, zoom: float) -> bytes:
    import fitz  # PyMuPDF
    with fitz.open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise ValueError(f"Page {page} does not exist (the PDF has {doc.page_count} pages).")
        pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")


def render_page(course: str, document: str, page: int, zoom: float = 1.5) -> bytes:
    """One PDF page as PNG (browsers cannot be relied on to show a PDF inside the Streamlit page)."""
    if Path(document).suffix.lower() != ".pdf":
        raise ValueError("Only PDF pages can be rendered as images.")
    path = upload_path(course, document)
    zoom = round(min(max(float(zoom), 0.5), 3.0), 2)
    return _render(str(path), path.stat().st_mtime, int(page), zoom)


# --- choosing the excerpts ----------------------------------------------------------------------
def _at_location(meta: dict, page: Optional[int], time_ms: Optional[int],
                 section: Optional[str], window: int) -> bool:
    kind = meta["source_type"]
    if kind == "video":
        if time_ms is None or meta.get("timestamp_start_ms") is None:
            return False
        return meta["timestamp_start_ms"] <= time_ms + window and meta["timestamp_end_ms"] >= time_ms - window
    if kind == "docx":
        return section is not None and (meta.get("section") or None) == section
    return page is not None and meta.get("page") == page


def _spread(hits: list[dict], n: int) -> list[dict]:
    """n chunks evenly spread over the document (in reading order)."""
    if len(hits) <= n:
        return list(hits)
    step = len(hits) / n
    return [hits[int(i * step)] for i in range(n)]


def _describe(kind: str, page: Optional[int], time_ms: Optional[int], section: Optional[str]) -> str:
    if kind == "video" and time_ms is not None:
        return f"minute {format_ms(time_ms)} of the lecture video"
    if kind == "pptx" and page:
        return f"slide {page}"
    if kind == "pdf" and page:
        return f"page {page}"
    if kind == "docx" and section:
        return f"the section \"{section}\""
    return "this part of the document"


def select_chunks(course: str, document: str, mode: str, question: str = "",
                  page: Optional[int] = None, time_ms: Optional[int] = None,
                  section: Optional[str] = None, store: Optional[VectorStore] = None) -> dict:
    cfg = study_cfg()
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Use one of {', '.join(MODES)}.")
    t0 = perf_counter()
    doc = document_chunks(course, document, store)
    hits = doc["hits"]
    kind = hits[0]["metadata"]["source_type"]
    window = int(cfg["explain_window_ms"])
    location = _describe(kind, page, time_ms, section)

    if mode == "ask":
        if not question.strip():
            raise ValueError("Please type a question.")
        from backend.indexing.embedder import embed_query
        q = np.asarray(embed_query(question), dtype=np.float32)
        sims = doc["matrix"] @ q
        boost = float(cfg["page_boost"])
        scored = []
        for h, s in zip(hits, sims):
            bonus = boost if _at_location(h["metadata"], page, time_ms, section, window) else 0.0
            scored.append({**h, "score": float(s) + bonus})
        scored.sort(key=lambda h: h["score"], reverse=True)
        k = int(cfg["top_context"])
        if cfg["use_reranker"] and len(scored) > k:
            from backend.retrieval.reranker import rerank_scores
            pool = scored[:int(cfg["rerank_candidates"])]
            for h, s in zip(pool, rerank_scores(question, [h["text"] for h in pool])):
                h["score"] = float(s)
            pool.sort(key=lambda h: h["score"], reverse=True)
            scored = pool
        chosen = scored[:k]
    elif mode == "explain":
        chosen = [{**h, "score": 1.0} for h in hits
                  if _at_location(h["metadata"], page, time_ms, section, window)]
        if not chosen:
            raise ValueError(f"Nothing is indexed at {location}. Pick another page or ask a question.")
        chosen = chosen[:int(cfg["overview_chunks"])]
    else:  # summary / concepts: a view of the whole document
        limit = int(cfg["max_chunk_chars"])
        chosen = [{**h, "score": 1.0, "text": h["text"][:limit]}
                  for h in _spread(hits, int(cfg["overview_chunks"]))]
    return {"chunks": chosen, "location": location,
            "retrieval_ms": int((perf_counter() - t0) * 1000)}


def _sources(chunks: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        out.append({"n": i, "label": _location_label(m), "page": m.get("page"),
                    "section": m.get("section") or None, "start_ms": m.get("timestamp_start_ms"),
                    "end_ms": m.get("timestamp_end_ms"), "score": round(float(c["score"]), 3),
                    "text": c["text"][:300]})
    return out


def _history(history: Optional[list[dict]], turns: int) -> list[dict]:
    keep = [t for t in (history or []) if t.get("role") in ("user", "assistant") and t.get("content")]
    keep = keep[-2 * turns:] if turns > 0 else []
    return [{"role": t["role"], "content": str(t["content"])[:1500]} for t in keep]


def prepare(course: str, document: str, mode: str, question: str = "", language: str = "en",
            page: Optional[int] = None, time_ms: Optional[int] = None,
            section: Optional[str] = None, history: Optional[list[dict]] = None,
            store: Optional[VectorStore] = None) -> dict:
    """Pick excerpts and build the LLM messages. Shared by the streaming and the plain endpoint."""
    cfg = study_cfg()
    language = language if language in ("tr", "en") else "en"
    sel = select_chunks(course, document, mode, question, page, time_ms, section, store)
    messages = build_study_messages(document, mode, sel["chunks"], language, question,
                                    sel["location"], _history(history, int(cfg["history_turns"])))
    long = mode in ("summary", "concepts", "explain")
    return {"messages": messages, "sources": _sources(sel["chunks"]), "mode": mode,
            "language": language, "retrieval_ms": sel["retrieval_ms"],
            "max_tokens": int(cfg["max_tokens_long"] if long else cfg["max_tokens"])}


def study_answer(provider: Optional[str] = None, **kwargs) -> dict:
    """One full reply (no streaming)."""
    t0 = perf_counter()
    prep = prepare(**kwargs)
    answer, used = chat_with_fallback(prep["messages"], provider=resolve_provider_name(provider),
                                      max_tokens=prep["max_tokens"])
    return {"answer": answer, "sources": prep["sources"], "mode": prep["mode"],
            "language": prep["language"], "retrieval_ms": prep["retrieval_ms"],
            "total_s": round(perf_counter() - t0, 2), "provider": used}


def study_stream(provider: Optional[str] = None, **kwargs):
    """(meta, generator of text pieces). Errors in choosing excerpts are raised here, before the
    response starts, so the API can still answer them with a proper HTTP status."""
    prep = prepare(**kwargs)
    meta = {"sources": prep["sources"], "mode": prep["mode"], "language": prep["language"],
            "retrieval_ms": prep["retrieval_ms"]}
    pieces = stream_with_fallback(prep["messages"], provider=resolve_provider_name(provider),
                                  max_tokens=prep["max_tokens"])
    return meta, pieces
