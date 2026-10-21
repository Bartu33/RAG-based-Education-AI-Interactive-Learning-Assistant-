"""Download the Piper voices used by the explanation video (once, ~60 MB each).

    python scripts/download_voices.py          # the voices in config.yaml (tr + en)
    python scripts/download_voices.py tr       # only Turkish

A dropped connection is retried and the download resumes where it stopped. If it still fails,
download the two files of the voice (.onnx and .onnx.json) in the browser from the links printed
below and put them in data/models/piper/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.media import tts  # noqa: E402


def main() -> int:
    langs = sys.argv[1:] or ["tr", "en"]
    failed = 0
    for lang in langs:
        name = tts.voice_name(lang)
        if tts.voice_ready(lang):
            print(f"[{lang}] {name}: already downloaded")
            continue
        print(f"[{lang}] {name}: downloading into {tts.voices_dir()}")
        shown = {}

        def progress(fname, done, total):
            pct = int(done * 100 / total) if total else 0
            if shown.get(fname) != pct:
                shown[fname] = pct
                print(f"\r  {fname}: {pct}%", end="", flush=True)
        try:
            tts.download(name, progress)
            print(f"\n[{lang}] done")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"\n[{lang}] FAILED: {exc}")
            print("  Download these two files in the browser and put them in", tts.voices_dir())
            for url in tts.voice_urls(name):
                print("   ", url)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
