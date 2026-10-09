import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

import numpy as np

from backend.config import get_config, resolve_path
from backend.schemas import Segment, metadata_to_public

_BATCH = 4000


def and_filter(conditions: list[Optional[dict]]) -> Optional[dict]:
    """Build a Chroma `where` clause; returns None when there is nothing to filter."""
    conditions = [c for c in conditions if c]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class VectorStore(ABC):
    """Adapter interface so ChromaDB can be swapped for Qdrant / pgvector later."""

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


class ChromaStore(VectorStore):
    def __init__(self, collection: Optional[str] = None, path: Optional[str] = None,
                 fresh: bool = False):
        import chromadb
        cfg = get_config()["indexing"]
        self._name = collection or cfg["collection"]
        self._client = chromadb.PersistentClient(path=str(resolve_path(path or cfg["chroma_path"])))
        if fresh:
            try:
                self._client.delete_collection(self._name)
            except Exception:
                pass
        self._col = self._client.get_or_create_collection(
            self._name,
            metadata={"hnsw:space": "cosine",
                      "hnsw:M": cfg["hnsw_m"],
                      "hnsw:construction_ef": cfg["hnsw_construction_ef"]},
        )

    def add(self, chunks, embeddings, ids=None) -> int:
        if ids is None:
            counters: dict = {}
            ids = []
            for c in chunks:
                key = (c.course, c.lecture, c.document)
                i = counters.get(key, 0)
                counters[key] = i + 1
                ids.append(hashlib.md5(f"{c.course}|{c.lecture}|{c.document}|{i}".encode()).hexdigest())
        emb = np.asarray(embeddings, dtype=np.float32)
        for s in range(0, len(chunks), _BATCH):
            e = s + _BATCH
            self._col.upsert(ids=ids[s:e], embeddings=emb[s:e].tolist(),
                             documents=[c.text for c in chunks[s:e]],
                             metadatas=[c.to_metadata() for c in chunks[s:e]])
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
        return len(self._col.get(where=where, include=[])["ids"])

    def delete_document(self, document, course=None) -> None:
        where = and_filter([{"document": {"$eq": document}},
                            {"course": {"$eq": course}} if course else None])
        self._col.delete(where=where)

    def list_documents(self) -> list[dict]:
        res = self._col.get(include=["metadatas"])
        agg: dict = {}
        for m in res["metadatas"]:
            key = (m["document"], m["course"], m["lecture"], m["source_type"])
            agg[key] = agg.get(key, 0) + 1
        return [{"document": d, "course": c, "lecture": l, "type": t, "chunks": n}
                for (d, c, l, t), n in sorted(agg.items())]

    def drop(self) -> None:
        self._client.delete_collection(self._name)


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return ChromaStore()