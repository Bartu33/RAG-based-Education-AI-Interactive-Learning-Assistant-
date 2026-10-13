"""Manual check of the real index (data/chroma) and of retrieval, without the UI or an LLM key.

  python -m scripts.check_index                                   # index summary
  python -m scripts.check_index --ask "Nyquist nedir?" --course CENG384 --lecture 5
  python -m scripts.check_index --ask "..." --course CENG384 --hybrid      # BM25 + dense
  python -m scripts.check_index --ask "..." --course CENG384 --ts 12 14    # watching minutes 12-14

Stop uvicorn first: ChromaDB must not be opened by two processes at the same time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_config  # noqa: E402
from backend.schemas import format_location  # noqa: E402


def line(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ask", help="question to retrieve for")
    p.add_argument("--course")
    p.add_argument("--lecture", type=int)
    p.add_argument("--ts", type=float, nargs=2, metavar=("START_MIN", "END_MIN"))
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    a = p.parse_args()

    from backend.indexing.vectorstore import get_store
    store = get_store()
    problems = []

    line("Index")
    st = store.stats()
    print(f"collection={st['collection']}  chunks={st['chunks']}  courses={st['courses']}  "
          f"model={st['embedding_model']}")
    print(f"by type: {st['by_type']}")
    if st["chunks"] == 0:
        print("Index is empty -> upload a file with POST /ingest first.")
        return

    line("Courses")
    for c in store.list_courses():
        print(f"{c['course']:<10} lectures={c['lectures']}  documents={c['documents']}  chunks={c['chunks']}")

    line("Documents")
    seen = set()
    for d in store.list_documents():
        key = (d["document"], d["course"])
        flag = "  <-- DUPLICATE (same file indexed under two lectures?)" if key in seen else ""
        seen.add(key)
        print(f"{d['course']:<10} L{d['lecture']:<3} {d['type']:<5} {d['chunks']:>5} chunks  {d['document']}{flag}")
        if flag:
            problems.append(f"duplicate document {d['document']}")

    line("Chunk sizes (real tokens)")
    from backend.indexing.chunker import count_tokens, get_tokenizer
    tok = get_tokenizer()
    sample = store.get(limit=200)
    sizes = sorted(count_tokens(h["text"], tok) for h in sample)
    limit = get_config()["indexing"]["chunk_size"]
    print(f"tokenizer={tok.name}  sample={len(sizes)}  min={sizes[0]}  median={sizes[len(sizes) // 2]}  "
          f"max={sizes[-1]}  (chunk_size={limit})")
    if tok.name == "regex-fallback":
        problems.append("real tokenizer not loaded (regex fallback in use)")
    if sizes[-1] > limit + 5:
        problems.append(f"a chunk has {sizes[-1]} tokens > chunk_size {limit}")

    thr = get_config()["ingestion"]["confidence_threshold"]
    low = store.low_confidence(thr, limit=5)
    if st["by_type"].get("video"):
        line(f"Low-confidence video chunks (< {thr})")
        for h in low:
            m = h["metadata"]
            print(f"conf={m['confidence']:.2f}  {format_location(m)}  {h['text'][:80]!r}")
        if not low:
            print("none")

    if a.ask:
        if not a.course:
            sys.exit("--course is required with --ask")
        from backend.retrieval.retriever import retrieve
        window = (int(a.ts[0] * 60_000), int(a.ts[1] * 60_000)) if a.ts else None
        r = retrieve(a.ask, a.course, a.lecture, window, hybrid=a.hybrid or None,
                     use_reranker=False if a.no_rerank else None)
        line(f"Retrieval  mode={r['mode']}")
        if r["mode"].endswith("(abstained)"):
            print("note: every reranker score was below rerank_min_score -> dense order kept")
        print(f"filtered_from={r['filtered_from']}  candidates={r['candidates']}  "
              f"retrieval_ms={r['retrieval_ms']}  timings={r['timings']}")
        for i, c in enumerate(r["chunks"], 1):
            m = c["metadata"]
            sim, rr = c.get("similarity"), c.get("rerank")
            win = "  [in window]" if c.get("in_window") else ""
            print(f"[{i}] score={c['score']:.3f}  cos={sim if sim is None else round(sim, 3)}  "
                  f"rerank={rr if rr is None else round(rr, 3)}  {format_location(m)}{win}")
            if c.get("duplicates"):
                print(f"    same content also at: {', '.join(c['duplicates'])}")
            print(f"    {c['text'][:160]!r}")
            if m["course"] != a.course or (a.lecture and m["lecture"] != a.lecture):
                problems.append(f"result [{i}] is from {m['course']} L{m['lecture']} (filter leak)")
            if not -1.01 <= c["score"] <= 1.0 + get_config()["retrieval"]["timestamp_boost"] + 0.01:
                problems.append(f"result [{i}] score {c['score']:.3f} outside 0..1(+boost)")

    line("Result")
    if problems:
        for pr in problems:
            print(f"PROBLEM: {pr}")
        sys.exit(1)
    print("OK - no problems found")


if __name__ == "__main__":
    main()
