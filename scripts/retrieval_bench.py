"""Retrieval / indexing measurements for docs/retrieval_report.md (WP2-WP3).

Run from the repo root:

  python -m scripts.retrieval_bench store        # ChromaDB write/read/search speed + HNSW grid (no model needed)
  python -m scripts.retrieval_bench embed        # BGE-m3 throughput on this machine (CPU vs GPU)
  python -m scripts.retrieval_bench consistency  # same-lecture chunks more similar than other lectures? (needs an index)
  python -m scripts.retrieval_bench eval         # hit@k / MRR: dense vs dense+rerank vs hybrid, timestamp-boost sweep
  python -m scripts.retrieval_bench chunking --manifest data/sample/manifest.json   # chunk size / overlap grid

Every command prints a Markdown table and writes it (plus a CSV) to docs/retrieval_results/.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import get_config, resolve_path  # noqa: E402

OUT = ROOT / "docs" / "retrieval_results"


# --------------------------------------------------------------------------- helpers
def to_markdown(df: pd.DataFrame) -> str:
    """Small Markdown table writer (avoids the optional `tabulate` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save(df: pd.DataFrame, name: str, header: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    md = (f"{header}\n\n" if header else "") + to_markdown(df)
    (OUT / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    print(md, "\n")
    print(f"-> {OUT / name}.md / .csv")


def machine() -> str:
    info = f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, " \
           f"Python {platform.python_version()}"
    try:
        import chromadb
        info += f", chromadb {chromadb.__version__}"
    except ImportError:
        pass
    return info


def pct(xs, q) -> float:
    return round(float(np.percentile(xs, q)), 1)


# --------------------------------------------------------------------------- 1) store
def synthetic(n: int, dim: int, courses: int, lectures: int, rng: np.random.Generator):
    """Clustered unit vectors: course centroid + lecture offset + noise (mimics real topic structure)."""
    from backend.schemas import Segment
    c_cent = rng.normal(0, 1, (courses, dim))
    l_cent = c_cent[:, None, :] + rng.normal(0, 0.8, (courses, lectures, dim))
    ci = rng.integers(0, courses, n)
    li = rng.integers(0, lectures, n)
    emb = l_cent[ci, li] + rng.normal(0, 1.2, (n, dim))
    emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
    segs = [Segment(text=f"synthetic chunk {i}", source_type="pdf", document=f"doc_{c}_{l}.pdf",
                    page=int(i % 300) + 1, course=f"C{c}", lecture=int(l) + 1)
            for i, (c, l) in enumerate(zip(ci, li))]
    return segs, emb, ci, li


def cmd_store(a):
    from backend.indexing.vectorstore import ChromaStore
    rng = np.random.default_rng(0)
    rows = []
    for n in a.sizes:
        segs, emb, ci, li = synthetic(n, a.dim, a.courses, a.lectures, rng)
        qi = rng.integers(0, n, a.queries)
        queries = emb[qi] + rng.normal(0, 0.02, (a.queries, a.dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        # NOTE: chromadb applies ef_search when the collection is loaded; changing it at runtime
        # (collection.modify) only takes effect after a restart, so each setting gets its own build.
        for m in a.m:
            for efc in a.ef_construction:
                for ef in a.ef_search:
                    tmp = tempfile.mkdtemp(prefix="bench_chroma_")
                    store = None
                    try:
                        store = ChromaStore(collection="bench_store", path=tmp, fresh=True,
                                            hnsw={"M": m, "construction_ef": efc, "search_ef": ef})
                        t = time.perf_counter()
                        store.add(segs, emb, ids=[f"c{i}" for i in range(n)])
                        write_s = time.perf_counter() - t

                        t = time.perf_counter()
                        store.get(where={"course": {"$eq": "C0"}})
                        read_ms = (time.perf_counter() - t) * 1000

                        for label in ("none", "course", "course+lecture"):
                            lat, rec = [], []
                            for q, idx in zip(queries, qi):
                                c, l = ci[idx], li[idx]
                                if label == "none":
                                    w, mask = None, np.ones(n, bool)
                                elif label == "course":
                                    w, mask = {"course": {"$eq": f"C{c}"}}, ci == c
                                else:
                                    w = {"$and": [{"course": {"$eq": f"C{c}"}},
                                                  {"lecture": {"$eq": int(l) + 1}}]}
                                    mask = (ci == c) & (li == l)
                                t = time.perf_counter()
                                hits = store.query(q.tolist(), a.k, w)
                                lat.append((time.perf_counter() - t) * 1000)
                                cand = np.flatnonzero(mask)
                                exact = cand[np.argsort(-(emb[cand] @ q))[:a.k]]
                                got = {int(h["id"][1:]) for h in hits}
                                rec.append(len(got & set(exact.tolist())) / max(1, len(exact)))
                            rows.append({"chunks": n, "M": m, "ef_construction": efc, "ef_search": ef,
                                         "filter": label, "write (chunks/s)": int(n / write_s),
                                         "read course pool (ms)": round(read_ms, 1),
                                         "search p50 (ms)": pct(lat, 50), "search p95 (ms)": pct(lat, 95),
                                         f"recall@{a.k}": round(float(np.mean(rec)), 3)})
                            print(rows[-1])
                    finally:
                        del store
                        shutil.rmtree(tmp, ignore_errors=True)
    save(pd.DataFrame(rows), a.name,
         f"Synthetic {a.dim}-d clustered vectors, {a.courses} courses x {a.lectures} lectures, "
         f"{a.queries} queries. Machine: {machine()}")


# --------------------------------------------------------------------------- 2) embed
SAMPLE_TR = ("Örnekleme teoremine göre bant sınırlı bir sinyal, en yüksek frekans bileşeninin en az "
             "iki katı hızda örneklenirse örneklerinden kayıpsız biçimde geri elde edilebilir. ")


def cmd_embed(a):
    from backend.indexing import embedder
    from backend.indexing.chunker import count_tokens, get_tokenizer
    tok = get_tokenizer()
    text = SAMPLE_TR * 20
    words = text.split()
    while count_tokens(" ".join(words), tok) > a.tokens:
        words = words[:-5]
    chunk = " ".join(words)
    texts = [chunk] * a.n
    embedder.embed_texts(texts[:8])  # warm-up / model load
    rows = []
    for bs in a.batch_sizes:
        t = time.perf_counter()
        embedder.embed_texts(texts, batch_size=bs)
        dt = time.perf_counter() - t
        rows.append({**embedder.device_info(), "tokenizer": tok.name,
                     "tokens/chunk": count_tokens(chunk, tok), "batch": bs,
                     "chunks/s": round(a.n / dt, 1), "10K chunks (min)": round(10_000 / (a.n / dt) / 60, 1)})
        print(rows[-1])
    q = time.perf_counter()
    for _ in range(20):
        embedder.embed_query("Nyquist örnekleme teoremi nedir?")
    rows_q = round((time.perf_counter() - q) / 20 * 1000, 1)
    save(pd.DataFrame(rows), "embedding_speed",
         f"Query embedding latency: {rows_q} ms. Machine: {machine()}")


# --------------------------------------------------------------------------- 3) consistency
def consistency_table(meta: list[dict], emb: np.ndarray, min_chunks: int = 3) -> pd.DataFrame:
    """For every (course, lecture): mean cosine to other chunks of the same lecture vs. to chunks of
    other lectures of the same course. Pass = intra > inter."""
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    keys = np.array([f"{m['course']}|{m['lecture']}" for m in meta])
    courses = np.array([m["course"] for m in meta])
    rows = []
    for key in sorted(set(keys)):
        own = np.flatnonzero(keys == key)
        course = key.split("|")[0]
        other = np.flatnonzero((courses == course) & (keys != key))
        if len(own) < min_chunks or len(other) == 0:
            continue
        sims_own = emb[own] @ emb[own].T
        intra = (sims_own.sum() - len(own)) / (len(own) * (len(own) - 1))
        inter = float((emb[own] @ emb[other].T).mean())
        rows.append({"course": course, "lecture": key.split("|")[1], "chunks": len(own),
                     "intra cos": round(float(intra), 3), "inter cos": round(inter, 3),
                     "margin": round(float(intra) - inter, 3), "pass": bool(intra > inter)})
    return pd.DataFrame(rows)


def cmd_consistency(a):
    from backend.indexing.vectorstore import get_store
    where = {"course": {"$eq": a.course}} if a.course else None
    hits = get_store().get(where=where, with_embeddings=True)
    if not hits:
        sys.exit("Index is empty: index some course materials first (Materials page or /ingest).")
    df = consistency_table([h["metadata"] for h in hits], np.stack([h["embedding"] for h in hits]))
    if df.empty:
        sys.exit("Need at least two lectures with >= 3 chunks in the same course.")
    rate = df["pass"].mean()
    save(df, "consistency", f"Pass rate: {rate:.0%} of lectures ({int(df['pass'].sum())}/{len(df)}); "
                            f"mean margin {df['margin'].mean():.3f}. Machine: {machine()}")


# --------------------------------------------------------------------------- 4) eval
def _relevant(meta: dict, labels: list[dict]) -> bool:
    for lab in labels:
        if meta["document"] != lab["document"]:
            continue
        if lab.get("page") is not None and meta.get("page") != lab["page"]:
            continue
        if lab.get("start_s") is not None:
            s, e = meta.get("timestamp_start_ms"), meta.get("timestamp_end_ms")
            if s is None or s > lab["end_s"] * 1000 or e < lab["start_s"] * 1000:
                continue
        return True
    return False


def evaluate(items: list[dict], store=None, k: int = 4, **opts) -> dict:
    from backend.retrieval.retriever import retrieve
    hit1, hitk, rr, lat = [], [], [], []
    for it in items:
        ts = it.get("ts_range_min")
        window = (int(ts[0] * 60_000), int(ts[1] * 60_000)) if ts else None
        r = retrieve(it["question"], it["course"], it.get("lecture"), window,
                     top_context=k, store=store, **opts)
        flags = [_relevant(c["metadata"], it["relevant"]) for c in r["chunks"]]
        first = next((i for i, f in enumerate(flags) if f), None)
        hit1.append(first == 0)
        hitk.append(first is not None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        lat.append(r["retrieval_ms"])
    return {"n": len(items), "hit@1": round(float(np.mean(hit1)), 3),
            f"hit@{k}": round(float(np.mean(hitk)), 3), "MRR": round(float(np.mean(rr)), 3),
            "latency p50 (ms)": pct(lat, 50), "latency p95 (ms)": pct(lat, 95)}


def load_eval(path) -> list[dict]:
    p = resolve_path(path)
    if not p.exists():
        sys.exit(f"{p} not found. Copy data/eval_set.example.json to data/eval_set.json "
                 f"and label your own questions.")
    items = json.loads(p.read_text(encoding="utf-8"))
    # skip comment entries ("_readme" or a question starting with "_") and unlabeled items
    return [it for it in items
            if "question" in it and not str(it["question"]).startswith("_") and it.get("relevant")]


def cmd_eval(a):
    items = load_eval(a.eval)
    rows = []
    for name, opts in [("dense", dict(hybrid=False, use_reranker=False)),
                       ("dense + rerank", dict(hybrid=False, use_reranker=True)),
                       ("hybrid (BM25+dense, RRF)", dict(hybrid=True, use_reranker=False)),
                       ("hybrid + rerank", dict(hybrid=True, use_reranker=True))]:
        rows.append({"mode": name, **evaluate(items, k=a.k, **opts)})
        print(rows[-1])
    save(pd.DataFrame(rows), "retrieval_modes", f"Labelled questions: {a.eval}. Machine: {machine()}")

    ts_items = [it for it in items if it.get("ts_range_min")]
    if ts_items:
        rows = []
        for boost in a.boosts:
            rows.append({"timestamp_boost": boost, **evaluate(ts_items, k=a.k, timestamp_boost=boost)})
            print(rows[-1])
        save(pd.DataFrame(rows), "timestamp_boost",
             f"Questions asked while watching a video (ts_range_min given): {len(ts_items)}")


# --------------------------------------------------------------------------- 5) chunking
def cmd_chunking(a):
    from backend.indexing.chunker import chunk_segments, count_tokens
    from backend.indexing.embedder import embed_texts
    from backend.indexing.vectorstore import ChromaStore
    from backend.ingestion import load_file
    from backend.retrieval import bm25

    manifest = json.loads(resolve_path(a.manifest).read_text(encoding="utf-8"))
    items = load_eval(a.eval)
    segments = []
    for entry in manifest:
        segs = load_file(resolve_path(entry["path"]))
        for s in segs:
            s.course, s.lecture = entry["course"], int(entry["lecture"])
        segments += segs
    rows = []
    for size in a.sizes:
        for ov in a.overlaps:
            chunks = chunk_segments(segments, chunk_size=size, overlap=ov)
            tmp = tempfile.mkdtemp(prefix="bench_chunk_")
            try:
                store = ChromaStore(collection="bench_chunking", path=tmp, fresh=True)
                t = time.perf_counter()
                store.add(chunks, embed_texts([c.text for c in chunks]))
                index_s = time.perf_counter() - t
                bm25.clear_cache()
                res = evaluate(items, store=store, k=a.k)
                rows.append({"chunk_size": size, "overlap": ov, "chunks": len(chunks),
                             "mean tokens": round(float(np.mean([count_tokens(c.text) for c in chunks])), 1),
                             "index (s)": round(index_s, 1), **res})
                print(rows[-1])
            finally:
                del store
                shutil.rmtree(tmp, ignore_errors=True)
    save(pd.DataFrame(rows), "chunking", f"Materials: {a.manifest}. Machine: {machine()}")


# --------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("store")
    s.add_argument("--sizes", type=int, nargs="+", default=[10_000])
    s.add_argument("--dim", type=int, default=1024)
    s.add_argument("--courses", type=int, default=5)
    s.add_argument("--lectures", type=int, default=14)
    s.add_argument("--queries", type=int, default=100)
    s.add_argument("--k", type=int, default=20)
    s.add_argument("--m", type=int, nargs="+", default=[16])
    s.add_argument("--ef-construction", type=int, nargs="+", default=[100])
    s.add_argument("--ef-search", type=int, nargs="+", default=[20, 50, 100, 200])
    s.add_argument("--name", default="store_hnsw", help="output file name in docs/retrieval_results/")
    s.set_defaults(fn=cmd_store)

    s = sub.add_parser("embed")
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--tokens", type=int, default=500)
    s.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    s.set_defaults(fn=cmd_embed)

    s = sub.add_parser("consistency")
    s.add_argument("--course", default=None)
    s.set_defaults(fn=cmd_consistency)

    s = sub.add_parser("eval")
    s.add_argument("--eval", default="data/eval_set.json")
    s.add_argument("--k", type=int, default=get_config()["retrieval"]["top_context"])
    s.add_argument("--boosts", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("chunking")
    s.add_argument("--manifest", required=True)
    s.add_argument("--eval", default="data/eval_set.json")
    s.add_argument("--k", type=int, default=get_config()["retrieval"]["top_context"])
    s.add_argument("--sizes", type=int, nargs="+", default=[300, 400, 500, 600])
    s.add_argument("--overlaps", type=int, nargs="+", default=[50, 100])
    s.set_defaults(fn=cmd_chunking)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
