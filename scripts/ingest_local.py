"""Index local files straight into data/chroma (no uvicorn, no Swagger needed).

  python -m scripts.ingest_local "C:\\Users\\me\\Desktop\\Chapter_1.pdf" --course CENG336 --lecture 1
  python -m scripts.ingest_local "C:\\Users\\me\\Desktop\\CENG336" --course CENG336 --lecture 1   # every supported file in a folder

Stop uvicorn first: ChromaDB must not be opened by two processes at the same time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion import EXTENSIONS  # noqa: E402


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            # Windows hides extensions: "Chapter_1" may really be "Chapter_1.pdf"
            guesses = [p.with_name(p.name + ext) for ext in EXTENSIONS if p.with_name(p.name + ext).exists()]
            if len(guesses) == 1:
                print(f"note: using {guesses[0].name}")
                p = guesses[0]
            else:
                sys.exit(f"Not found: {p}" + (f" (did you mean one of {[g.name for g in guesses]}?)" if guesses else ""))
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTENSIONS)
        elif p.suffix.lower() in EXTENSIONS:
            files.append(p)
        else:
            sys.exit(f"Unsupported file type: {p.name} (supported: {', '.join(sorted(EXTENSIONS))})")
    if not files:
        sys.exit("No supported files found.")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="files or folders")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", type=int, required=True)
    a = ap.parse_args()

    from backend.indexing.indexer import index_file
    for f in collect(a.paths):
        print(f"\n{f.name}")

        def progress(stage: str, frac: float) -> None:
            print(f"\r  {stage:<8} {frac:5.0%}", end="", flush=True)

        rec = index_file(f, a.course, a.lecture, on_progress=progress)
        t = rec["timings_ms"]
        print(f"\r  OK  {rec['chunks']} chunks  ({rec['type']})  "
              f"extract {t['extract']} ms · chunk {t['chunk']} ms · embed {t['embed']} ms · store {t['store']} ms"
              + (f"  · low-confidence segments: {rec['low_confidence_segments']}"
                 if rec["type"] == "video" else ""))
    print("\nDone. Check with:  python -m scripts.check_index")


if __name__ == "__main__":
    main()
