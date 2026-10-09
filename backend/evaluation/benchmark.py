import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from backend.config import get_config, resolve_path
from backend.evaluation.ragas_runner import classify, faithfulness_score, hallucination_type
from backend.indexing.vectorstore import ChromaStore, VectorStore, get_store
from backend.schemas import Segment
from backend.tasks.qa import answer_question

log = logging.getLogger(__name__)
_STEP = 10_000


def _to_segment(hit: dict, course: str, lecture: int) -> Segment:
    m = hit["metadata"]
    return Segment(text=hit["text"], source_type=m["source_type"], document=m["document"],
                   page=m.get("page"), section=m.get("section"),
                   timestamp_start_ms=m.get("timestamp_start_ms"),
                   timestamp_end_ms=m.get("timestamp_end_ms"),
                   confidence=m.get("confidence"), course=course, lecture=lecture)


def _build_store(size: int, courses: set, main: VectorStore, noise_std: float,
                 rng: np.random.Generator) -> VectorStore:
    """Real chunks of the benchmark courses + synthetic distractors (noisy copies of real chunks
    assigned to other courses) until the collection holds `size` chunks."""
    store = ChromaStore(collection=f"bench_{size}", fresh=True)
    real = []
    for course in courses:
        real += main.get(where={"course": {"$eq": course}}, with_embeddings=True)
    if not real:
        raise ValueError("No indexed chunks for the benchmark courses. Index materials first.")
    store.add([_to_segment(h, h["metadata"]["course"], h["metadata"]["lecture"]) for h in real],
              np.stack([h["embedding"] for h in real]), ids=[f"real-{i}" for i in range(len(real))])

    pool = main.get(limit=5000, with_embeddings=True)
    pool_emb = np.stack([h["embedding"] for h in pool])
    need = size - len(real)
    if need < 0:
        log.warning("Benchmark size %d is smaller than the real corpus (%d chunks).", size, len(real))
    for start in range(0, max(need, 0), _STEP):
        n = min(_STEP, need - start)
        idx = rng.integers(0, len(pool), n)
        emb = pool_emb[idx] + rng.normal(0, noise_std, (n, pool_emb.shape[1]))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        segs = [_to_segment(pool[i], f"__dist_{(start + k) % 20}", 0) for k, i in enumerate(idx)]
        store.add(segs, emb, ids=[f"dist-{start + k}" for k in range(n)])
    return store


def run_benchmark(sizes: Optional[list[int]] = None, questions: Optional[list[dict]] = None,
                  provider: Optional[str] = None) -> pd.DataFrame:
    """questions: [{"question": str, "course": str, "lecture": int | null}, ...]"""
    cfg = get_config()["benchmark"]
    sizes = sizes or cfg["sizes"]
    if questions is None:
        with open(resolve_path(cfg["questions_path"]), encoding="utf-8") as f:
            questions = json.load(f)
    courses = {q["course"] for q in questions}
    main, rng = get_store(), np.random.default_rng(42)

    rows = []
    for size in sizes:
        store = _build_store(size, courses, main, cfg["noise_std"], rng)
        try:
            ret_ms, total_s, faiths, labels, types = [], [], [], [], []
            for q in questions:
                res = answer_question(q["question"], q["course"], q.get("lecture"),
                                      cfg={"provider": provider} if provider else None, store=store)
                faith = res["faithfulness"]
                if faith is None:
                    faith = faithfulness_score(res["answer"], res["contexts"], provider)
                label = classify(faith)
                htype = hallucination_type(res["answer"], res["contexts"], provider) \
                    if label != "grounded" else None
                ret_ms.append(res["retrieval_ms"]); total_s.append(res["total_s"])
                faiths.append(faith); labels.append(label); types.append(htype)
            n = len(questions)
            rows.append({
                "Corpus (chunks)": size,
                "Retrieval (ms)": round(float(np.mean(ret_ms)), 1),
                "Answer (s)": round(float(np.mean(total_s)), 2),
                "RAGAs Faithfulness": round(float(np.mean(faiths)), 3),
                "Hallucination (%)": round(100 * labels.count("unsupported") / n, 1),
                "Intrinsic (%)": round(100 * types.count("intrinsic") / n, 1),
                "Extrinsic (%)": round(100 * types.count("extrinsic") / n, 1),
            })
        finally:
            store.drop()
    return pd.DataFrame(rows)