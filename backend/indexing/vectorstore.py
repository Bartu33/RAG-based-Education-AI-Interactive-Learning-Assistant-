import hashlib
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

import numpy as np

from backend.config import get_config, resolve_path
from backend.schemas import Segment, metadata_to_public

log = logging.getLogger(__name__)
_BATCH = 4000  # upper bound; the client's own max batch size is used when smaller


class EmbeddingModelMismatch(RuntimeError):
    """The collection was built with another embedding model than config.yaml names."""


def chunk_id(course: str, lecture: int, document: str, i: int) -> str:
    """Id of the i-th chunk of a document (its reading position can be recovered from it)."""
    return hashlib.md5(f"{course}|{lecture}|{document}|{i}".encode()).hexdigest()


def and_filter(conditions: list[Optional[dict]]) -> Optional[dict]:
    """Build a Chroma `where` clause; returns None when there is nothing to filter."""
    conditions = [c for c in conditions if c]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class VectorStore(ABC):
    """Adapter interface so ChromaDB can be swapped for Qdrant / pgvector later."""

    # Incremented on every write; retrieval caches (e.g. the BM25 index) use it for invalidation.
    version: int = 0

    @abstractmethod
    def add(self, chunks: list[Segment], embeddings, ids: Optional[list[str]] = None) -> int: ...

    @abstractmethod
    def query(self, embedding: list[float], n: int, where: Optional[dict] = None) -> list[dict]: ...

    @abstractmethod
    def get(self, where: Optional[dict] = None, limit: Optional[int] = None,
            with_embeddings: bool = False) -> list[dict]: ...

    @abstractmethod
    def count(self, where: Optional[dict] = None) -> int: ...

    @abstractmethod
    def delete_document(self, document: str, course: Optional[str] = None) -> None: ...

    @abstractmethod
    def list_documents(self) -> list[dict]: ...

    @abstractmethod
    def drop(self) -> None: ...

    def set_search_ef(self, ef: int) -> None:
        """Optional: trade recall for latency at query time (HNSW ef_search)."""

    # --- management (used by the API / UI: delete buttons, course lists, transcript fixes) ---
    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> list[dict]: ...

    @abstractmethod
    def delete_where(self, where: dict) -> int: ...

    def delete_course(self, course: str, lecture: Optional[int] = None) -> int:
        """Remove every chunk of a course (or of one lecture). Returns the number removed."""
        return self.delete_where(and_filter([{"course": {"$eq": course}},
                                             {"lecture": {"$eq": int(lecture)}} if lecture else None]))

    @abstractmethod
    def clear(self) -> None:
        """Empty the collection but keep it (same HNSW settings)."""

    @abstractmethod
    def update_text(self, chunk_id: str, text: str, embedding, corrected: bool = True) -> None:
        """Replace one chunk's text + vector, keeping its metadata (instructor transcript fix)."""

    def low_confidence(self, threshold: float, course: Optional[str] = None,
                       limit: Optional[int] = None) -> list[dict]:
        """Video chunks whose STT confidence is below `threshold` (for the 'edit transcript' view)."""
        where = and_filter([{"source_type": {"$eq": "video"}},
                            {"confidence": {"$gte": 0.0}},  # -1 = no confidence (non-video)
                            {"confidence": {"$lt": float(threshold)}},
                            {"course": {"$eq": course}} if course else None])
        hits = self.get(where=where, limit=limit)
        return sorted(hits, key=lambda h: h["metadata"]["confidence"])

    def list_courses(self) -> list[dict]:
        """[{course, lectures: [..], documents, chunks}] aggregated from list_documents()."""
        agg: dict = {}
        for d in self.list_documents():
            c = agg.setdefault(d["course"], {"course": d["course"], "lectures": set(),
                                             "documents": 0, "chunks": 0})
            c["lectures"].add(d["lecture"])
            c["documents"] += 1
            c["chunks"] += d["chunks"]
        return [{**c, "lectures": sorted(c["lectures"])} for _, c in sorted(agg.items())]


class ChromaStore(VectorStore):
    def __init__(self, collection: Optional[str] = None, path: Optional[str] = None,
                 fresh: bool = False, hnsw: Optional[dict] = None,
                 allow_model_change: bool = False):
        """hnsw: optional overrides {"M", "construction_ef", "search_ef"} (used by experiments).
        allow_model_change: open a collection built with a different embedding model anyway
        (only to inspect or clear it; its vectors are not comparable with new queries)."""
        import chromadb
        cfg = get_config()["indexing"]
        self._name = collection or cfg["collection"]
        self.version = 0
        self._count_cache: dict = {}
        self._client = chromadb.PersistentClient(path=str(resolve_path(path or cfg["chroma_path"])))
        if fresh:
            try:
                self._client.delete_collection(self._name)
            except Exception:
                pass
        self._hnsw = {"M": cfg["hnsw_m"], "construction_ef": cfg["hnsw_construction_ef"],
                      "search_ef": cfg.get("hnsw_search_ef", 100), **(hnsw or {})}
        self.embedding_model = cfg["embedding_model"]
        self._col = self._open()
        self._check_model(allow_model_change)
        try:
            self._max_batch = min(_BATCH, int(self._client.get_max_batch_size()))
        except Exception:  # noqa: BLE001
            self._max_batch = _BATCH

    def _open(self):
        h = self._hnsw
        meta = {"embedding_model": self.embedding_model}
        try:  # chromadb >= 1.0: typed configuration (also exposes ef_search)
            return self._client.get_or_create_collection(
                self._name, embedding_function=None, metadata=meta,
                configuration={"hnsw": {"space": "cosine", "max_neighbors": h["M"],
                                        "ef_construction": h["construction_ef"],
                                        "ef_search": h["search_ef"]}})
        except (TypeError, ValueError):  # chromadb 0.5 / 0.6: legacy metadata keys
            return self._client.get_or_create_collection(
                self._name, embedding_function=None,
                metadata={**meta, "hnsw:space": "cosine", "hnsw:M": h["M"],
                          "hnsw:construction_ef": h["construction_ef"],
                          "hnsw:search_ef": h["search_ef"]})

    def _check_model(self, allow: bool) -> None:
        """Vectors from two different embedding models live in different spaces: mixing them
        returns plausible-looking but meaningless neighbours, with no error. Refuse that."""
        stored = (self._col.metadata or {}).get("embedding_model")
        if stored is None:  # collection created before this check existed: stamp it
            try:
                self._col.modify(metadata={**(self._col.metadata or {}),
                                           "embedding_model": self.embedding_model})
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not stamp embedding model on %s: %s", self._name, exc)
        elif stored != self.embedding_model and not allow:
            raise EmbeddingModelMismatch(
                f"Collection '{self._name}' was built with {stored}, but config.yaml uses "
                f"{self.embedding_model}. Re-index the materials (or use another collection name).")

    def _touch(self) -> None:
        self.version += 1
        self._count_cache.clear()

    def set_search_ef(self, ef: int) -> None:
        """Persists the new ef_search. chromadb keeps the loaded index cached per process, so the
        value is only used after a restart; for experiments create a new collection instead."""
        try:
            self._col.modify(configuration={"hnsw": {"ef_search": int(ef)}})
        except Exception as exc:  # noqa: BLE001  (older chromadb: fixed at creation)
            log.warning("Could not change ef_search: %s", exc)

    def add(self, chunks, embeddings, ids=None) -> int:
        if ids is None:
            counters: dict = {}
            ids = []
            for c in chunks:
                key = (c.course, c.lecture, c.document)
                i = counters.get(key, 0)
                counters[key] = i + 1
                ids.append(chunk_id(c.course, c.lecture, c.document, i))
        emb = np.asarray(embeddings, dtype=np.float32)
        for s in range(0, len(chunks), self._max_batch):
            e = s + self._max_batch
            self._col.upsert(ids=ids[s:e], embeddings=emb[s:e],
                             documents=[c.text for c in chunks[s:e]],
                             metadatas=[c.to_metadata() for c in chunks[s:e]])
        self._touch()
        return len(chunks)

    def query(self, embedding, n, where=None) -> list[dict]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=n, where=where,
                              include=["documents", "metadatas", "distances"])
        hits = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            hits.append({"id": id_, "text": doc, "metadata": metadata_to_public(meta),
                         "score": 1.0 - float(dist)})
        return hits

    def get(self, where=None, limit=None, with_embeddings=False) -> list[dict]:
        include = ["documents", "metadatas"] + (["embeddings"] if with_embeddings else [])
        res = self._col.get(where=where, limit=limit, include=include)
        hits = []
        for i, id_ in enumerate(res["ids"]):
            hit = {"id": id_, "text": res["documents"][i],
                   "metadata": metadata_to_public(res["metadatas"][i]), "score": 0.0}
            if with_embeddings:
                hit["embedding"] = np.asarray(res["embeddings"][i], dtype=np.float32)
            hits.append(hit)
        return hits

    def count(self, where=None) -> int:
        if where is None:
            return self._col.count()
        key = repr(where)  # filtered counts are cached until the next write (cheap per query)
        if key not in self._count_cache:
            self._count_cache[key] = len(self._col.get(where=where, include=[])["ids"])
        return self._count_cache[key]

    def delete_document(self, document, course=None) -> None:
        where = and_filter([{"document": {"$eq": document}},
                            {"course": {"$eq": course}} if course else None])
        self._col.delete(where=where)
        self._touch()

    def list_documents(self) -> list[dict]:
        res = self._col.get(include=["metadatas"])
        agg: dict = {}
        for m in res["metadatas"]:
            key = (m["document"], m["course"], m["lecture"], m["source_type"])
            agg[key] = agg.get(key, 0) + 1
        return [{"document": d, "course": c, "lecture": l, "type": t, "chunks": n}
                for (d, c, l, t), n in sorted(agg.items())]

    def get_by_ids(self, ids) -> list[dict]:
        res = self._col.get(ids=list(ids), include=["documents", "metadatas"])
        return [{"id": i, "text": d, "metadata": metadata_to_public(m), "score": 0.0}
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def delete_where(self, where) -> int:
        if not where:
            raise ValueError("delete_where needs a filter; use clear() to empty the collection")
        n = len(self._col.get(where=where, include=[])["ids"])
        if n:
            self._col.delete(where=where)
            self._touch()
        return n

    def clear(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._open()
        self._touch()

    def update_text(self, chunk_id, text, embedding, corrected=True) -> None:
        res = self._col.get(ids=[chunk_id], include=["metadatas"])
        if not res["ids"]:
            raise KeyError(f"No chunk with id {chunk_id}")
        meta = dict(res["metadatas"][0])
        if corrected:
            meta["corrected"] = True
            if meta.get("source_type") == "video":
                meta["confidence"] = 1.0  # verified by a person
        self._col.upsert(ids=[chunk_id], documents=[text], metadatas=[meta],
                         embeddings=np.asarray([embedding], dtype=np.float32))
        self._touch()

    def stats(self) -> dict:
        res = self._col.get(include=["metadatas"])
        by_type: dict = {}
        for m in res["metadatas"]:
            by_type[m["source_type"]] = by_type.get(m["source_type"], 0) + 1
        return {"collection": self._name, "chunks": len(res["ids"]), "by_type": by_type,
                "courses": len({m["course"] for m in res["metadatas"]}),
                "embedding_model": (self._col.metadata or {}).get("embedding_model"),
                "hnsw": dict(self._hnsw)}

    def drop(self) -> None:
        self._client.delete_collection(self._name)
        self._touch()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return ChromaStore()


def cleanup_collections(prefix: str = "bench_", path: Optional[str] = None) -> list[str]:
    """Delete leftover temporary collections (e.g. after an interrupted benchmark run)."""
    import chromadb
    cfg = get_config()["indexing"]
    client = chromadb.PersistentClient(path=str(resolve_path(path or cfg["chroma_path"])))
    removed = []
    for col in client.list_collections():
        name = getattr(col, "name", col)
        if name.startswith(prefix) and name != cfg["collection"]:
            client.delete_collection(name)
            removed.append(name)
    return removed
