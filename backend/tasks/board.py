"""Teaching board: the answer is explained the way a lecturer would do it in class.

- The answer comes from a lecture video -> the video plays small in the corner from that second
  (`video`), and the slide / PDF page being talked about at that moment is found automatically.
- The explanation is a short lesson (`lesson.beats`): the lecturer frames the question, walks
  through the relevant lines (of up to `max_pages` cited pages) one by one, explaining each in
  plain words with an example, and wraps up. Each beat points at one line (highlighted while it
  is explained) and carries a short key point; the lesson ends with a summary.
- Only a real calculation / derivation / proof question (math, physics, ...) opens the whiteboard
  instead: what is given and what is asked, then one line of the solution per step with why it is
  done (`write` + `say`), the result and a quick check. Formulas are always delimited ($...$).

Local: video choice, page matching and highlighting (BGE-m3 vectors, PyMuPDF), slide conversion
(LibreOffice). One LLM call writes the lesson (falls back to a plain walk-through without it).
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import get_config, resolve_path
from backend.indexing.vectorstore import VectorStore, and_filter, get_store
from backend.llm.providers import chat_with_fallback, parse_json, resolve_provider_name
from backend.media.mathfmt import fix_line

log = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": True,
    "math_mode": "auto",           # auto | always | never  (whiteboard for math questions)
    "match_min_similarity": 0.45,  # video chunk -> slide/page match must be at least this similar
    "highlight_lines": 3,          # at most this many lines are highlighted on the page
    "highlight_min_similarity": 0.30,
    "highlight_margin": 0.08,      # keep lines within this of the best line's similarity
    "narrate": True,               # lecturer-style lesson (one LLM call); false = highlights only
    "max_beats": 7,
    "max_steps": 10,
    "max_pages": 3,                # a lesson can walk through this many cited pages / slides
    "max_tokens": 2200,
    "libreoffice_path": None,      # null = find soffice on PATH / the default Windows location
    "zoom": 1.6,
}

# A whiteboard is only for questions that need working out: an explicit calculation / derivation /
# proof request, or numbers that are operated on (an expression, or quantities with units).
_SOLVE_WORDS = re.compile(
    r"(hesapla\w*|çöz\w*|coz\w*|türet\w*|turet\w*|ispatla\w*|ispat et\w*|kanıtla\w*|kanitla\w*|"
    r"sadeleştir\w*|integralini|türevini|limitini|"
    r"\bsolve\b|\bcalculate\b|\bcompute\b|\bderive\b|\bprove\b|\bevaluate\b|\bsimplify\b|"
    r"\bwork out\b|\bfind the value\b|\bstep[ -]by[ -]step\b)", re.IGNORECASE)
_EXPRESSION = re.compile(r"\d\s*[\+\-\*/\^×÷=]\s*\(?\s*[\dA-Za-z]|[∑∫√≤≥≠∞]|\\frac|\\sum|\\int|\$[^$]+\$")
_QUANTITY = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:[kKMG]?b/s|m/s²?|km/h|km|kg|g|m|cm|mm|s|ms|µs|ns|N|J|W|kW|V|A|Ω|Hz|kHz|MHz|GHz|"
    r"bps|kbps|Mbps|Gbps|bit|bits|byte|bytes|KB|MB|GB|°C|K|mol|Pa)"
    r"(?:['’]?(?:lık|lik|luk|lük|de|da|te|ta|den|dan|ten|tan|e|a|ye|ya))?(?![\wçğıöşü])")
_ASKS_VALUE = re.compile(r"(\bkaç\b|\bne kadar\b|\bnedir\b|\bbul\w*|how many|how much|what is the|find)",
                         re.IGNORECASE)


def board_cfg() -> dict:
    return {**DEFAULTS, **(get_config().get("board") or {})}


# --- what kind of answer is this -----------------------------------------------------------------
def is_math_question(question: str, answer: str = "") -> bool:
    """True only when the question needs working out (math, physics, any calculation):
    an explicit solve / calculate / derive / prove request, an expression with numbers, or at
    least two quantities with units together with a 'how much / find' question.
    Conceptual questions ("what is a protocol?") never open the whiteboard."""
    q = question or ""
    if _SOLVE_WORDS.search(q) or _EXPRESSION.search(q):
        return True
    return len(_QUANTITY.findall(q)) >= 2 and bool(_ASKS_VALUE.search(q))


def question_language(question: str) -> str:
    q = (question or "").lower()
    if re.search(r"[çğıöşü]", q) or re.search(r"\b(nedir|nasıl|neden|niçin|hangi|kaç|mı|mi|mu|mü|ile|ve)\b", q):
        return "tr"
    return "en"


_TR_WORDS = {"ve", "bir", "bu", "için", "ile", "da", "de", "olarak", "şu", "ne", "nasıl", "değil", "var", "yok",
             "gibi", "daha", "çok", "olan", "her", "ya", "veya", "ama", "sonra", "önce", "şimdi", "bakalım",
             "burada", "yani", "kadar", "eder", "olur", "olduğu", "biz", "siz"}
_EN_WORDS = {"the", "and", "is", "of", "to", "in", "that", "this", "we", "you", "are", "for", "with", "it",
             "as", "be", "on", "by", "an", "which", "here", "let", "now", "so", "can", "its", "what", "how"}


def text_language(text: str) -> str:
    """'tr' or 'en': which language a generated explanation is actually written in."""
    words = re.findall(r"[a-zçğıöşü]+", (text or "").lower())
    tr = sum(w in _TR_WORDS for w in words) + 2 * len(re.findall(r"[çğışöü]", " ".join(words)))
    en = sum(w in _EN_WORDS for w in words)
    return "tr" if tr > en else "en"


LANG_NAME = {"tr": "Turkish (Türkçe)", "en": "English"}


def _language_rule(language: str) -> str:
    name = LANG_NAME.get(language, "English")
    return (f"\n\nLANGUAGE: the student asked in {name}. Write EVERY text field of the JSON (title, given, "
            f"find, say, point, closing, summary, check, result words) in {name}, even when the course "
            f"material is in another language: explain its ideas in {name} (a technical term may follow "
            "in parentheses). Do not mix languages.")


def _ask_json(messages: list, language: str, texts_of, provider, temperature: float, max_tokens: int):
    """One LLM call for a JSON reply in `language`; if the reply came back in the other language,
    ask once more to rewrite it. Returns (data, provider used)."""
    text, used = chat_with_fallback(messages, provider=resolve_provider_name(provider),
                                    temperature=temperature, max_tokens=max_tokens, json_mode=True)
    data = parse_json(text)
    spoken = " ".join(texts_of(data))
    if spoken.strip() and text_language(spoken) != language:
        name = LANG_NAME.get(language, "English")
        retry = messages + [{"role": "assistant", "content": text},
                            {"role": "user", "content": f"That is not in {name}. Rewrite the same JSON (same "
                                                        f"keys, same line numbers) with every text in {name}."}]
        try:
            text2, used = chat_with_fallback(retry, provider=resolve_provider_name(provider),
                                             temperature=temperature, max_tokens=max_tokens, json_mode=True)
            data2 = parse_json(text2)
            if text_language(" ".join(texts_of(data2))) == language:
                data = data2
        except Exception as exc:  # noqa: BLE001  (keep the first reply)
            log.warning("Language retry failed: %s", exc)
    return data, used


def pick_video(sources: list[dict]) -> Optional[dict]:
    """The best-scored video source with a time range (sources arrive best first)."""
    for s in sources or []:
        if s.get("type") == "video" and s.get("start_ms") is not None:
            return {"document": s["document"], "start_ms": int(s["start_ms"]),
                    "end_ms": int(s.get("end_ms") or s["start_ms"]), "text": s.get("text", "")}
    return None


def page_sources(sources: list[dict]) -> list[dict]:
    """The PDF pages / slides among the answer's own sources, best first, each page once."""
    out, seen = [], set()
    for s in sources or []:
        if s.get("type") in ("pdf", "pptx") and s.get("page"):
            key = (s["document"], int(s["page"]))
            if key not in seen:
                seen.add(key)
                out.append({"document": s["document"], "type": s["type"], "page": int(s["page"]),
                            "reason": "source", "text": s.get("text", "")})
    return out


def pick_page_source(sources: list[dict]) -> Optional[dict]:
    """The best-scored PDF page or slide among the answer's own sources."""
    pages = page_sources(sources)
    return pages[0] if pages else None


def match_page_for_video(video: dict, course: str, lecture: Optional[int],
                         store: Optional[VectorStore] = None) -> Optional[dict]:
    """The slide / PDF page of the same course (and lecture) that is closest in meaning to what
    is said in the video at that moment."""
    from backend.indexing.embedder import embed_query
    cfg = board_cfg()
    store = store or get_store()
    if not video.get("text", "").strip():
        return None
    where = and_filter([{"course": {"$eq": course}},
                        {"lecture": {"$eq": int(lecture)}} if lecture else None,
                        {"source_type": {"$in": ["pdf", "pptx"]}}])
    hits = store.query(embed_query(video["text"]), 3, where)
    hits = [h for h in hits if h["metadata"].get("page")]
    if not hits or hits[0]["score"] < float(cfg["match_min_similarity"]):
        return None
    m = hits[0]["metadata"]
    return {"document": m["document"], "type": m["source_type"], "page": int(m["page"]),
            "reason": "video-match", "similarity": round(float(hits[0]["score"]), 3),
            "text": hits[0]["text"]}


# --- the page: file, text lines, highlights --------------------------------------------------------
def _soffice() -> Optional[str]:
    configured = board_cfg().get("libreoffice_path")
    for cand in [configured, shutil.which("soffice"), shutil.which("libreoffice"),
                 r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                 "/Applications/LibreOffice.app/Contents/MacOS/soffice"]:
        if cand and Path(cand).exists():
            return str(cand)
    return None


def pdf_for(course: str, document: str) -> Path:
    """A PDF to draw the page from: the upload itself, or the slides converted once with
    LibreOffice (cached under data/uploads/<course>/.rendered/)."""
    from backend.tasks.study import upload_path
    src = upload_path(course, document)
    if src.suffix.lower() == ".pdf":
        return src
    if src.suffix.lower() != ".pptx":
        raise ValueError(f"{document} has no pages to show.")
    out_dir = src.parent / ".rendered"
    out = out_dir / (src.stem + ".pdf")
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    exe = _soffice()
    if exe is None:
        raise FileNotFoundError("LibreOffice is not installed, so slides cannot be shown as images "
                                "(install it or set board.libreoffice_path in config.yaml).")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(src)],
                   check=True, timeout=180, capture_output=True)
    if not out.is_file():
        raise RuntimeError(f"LibreOffice did not produce {out.name}.")
    return out


def page_lines(pdf: Path, page: int) -> list[str]:
    import fitz  # PyMuPDF
    with fitz.open(pdf) as doc:
        if not 1 <= page <= doc.page_count:
            return []
        text = doc[page - 1].get_text("text")
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if len(re.findall(r"\w", ln)) >= 3 and len(ln) <= 220]


def choose_highlights(lines: list[str], question: str, answer: str = "") -> list[str]:
    """The page lines closest in meaning to the question (+ the start of the answer)."""
    cfg = board_cfg()
    if not lines:
        return []
    from backend.indexing.embedder import embed_texts
    vecs = embed_texts(lines + [f"{question}\n{(answer or '')[:500]}"])
    sims = vecs[:-1] @ vecs[-1]
    order = np.argsort(-sims)
    best = float(sims[order[0]])
    if best < float(cfg["highlight_min_similarity"]):
        return []
    keep = [i for i in order[:int(cfg["highlight_lines"])]
            if float(sims[i]) >= best - float(cfg["highlight_margin"])]
    return [lines[i] for i in sorted(keep)]  # in reading order


@lru_cache(maxsize=64)
def _render(pdf: str, mtime: float, page: int, highlights: tuple, zoom: float) -> bytes:
    import fitz  # PyMuPDF
    with fitz.open(pdf) as doc:
        if not 1 <= page <= doc.page_count:
            raise ValueError(f"Page {page} does not exist (the document has {doc.page_count}).")
        pg = doc[page - 1]
        for line in highlights:
            rects = pg.search_for(line) or (pg.search_for(line[:40]) if len(line) > 40 else [])
            if rects:
                annot = pg.add_highlight_annot(rects)
                annot.set_colors(stroke=(1.0, 0.85, 0.1))
                annot.update()
        return pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).tobytes("png")


@lru_cache(maxsize=128)
def _box(pdf: str, mtime: float, page: int, lines: tuple) -> Optional[tuple]:
    import fitz  # PyMuPDF
    with fitz.open(pdf) as doc:
        if not 1 <= page <= doc.page_count:
            return None
        pg = doc[page - 1]
        rects = []
        for line in lines:
            rects += pg.search_for(line) or (pg.search_for(line[:40]) if len(line) > 40 else [])
        if not rects:
            return None
        W, H = pg.rect.width, pg.rect.height
        return (min(r.x0 for r in rects) / W, min(r.y0 for r in rects) / H,
                max(r.x1 for r in rects) / W, max(r.y1 for r in rects) / H)


def highlight_box(course: str, document: str, page: int, lines: list[str]) -> Optional[tuple]:
    """Where the highlighted lines are on the page, as fractions (x0, y0, x1, y1), or None."""
    if not lines:
        return None
    pdf = pdf_for(course, document)
    return _box(str(pdf), pdf.stat().st_mtime, int(page), tuple(lines))


def render_page(course: str, document: str, page: int, highlights: Optional[list[str]] = None,
                zoom: Optional[float] = None) -> bytes:
    """PNG of a PDF page or slide with the given lines highlighted in yellow."""
    pdf = pdf_for(course, document)
    z = round(min(max(float(zoom or board_cfg()["zoom"]), 0.5), 3.0), 2)
    return _render(str(pdf), pdf.stat().st_mtime, int(page), tuple(highlights or ()), z)


# --- the whiteboard --------------------------------------------------------------------------------
WHITEBOARD_SYSTEM = (
    "You are the course lecturer, standing at the whiteboard and solving a student's question in "
    "class, the way a good teacher does it: first what is given and what is asked, then the "
    "solution one small step at a time, then the result and a quick check. Use the numbered course "
    "context for definitions, notation, values and units; standard math / physics reasoning between "
    "the steps is allowed.\n"
    "Each step has 'write' = the ONE short line you write on the board, and 'say' = what you say to "
    "the class while writing it: 1-3 natural spoken sentences that explain WHY this step is done "
    "(not just what), addressing the student. In 'say', read symbols in words (say 'times', never "
    "'\\times'); no LaTeX in 'say'.\n"
    "Formatting rules for everything written on the board ('given', 'find', 'write', 'result'): EVERY "
    "formula, number with an operator, symbol or set goes inside $...$ (e.g. '$x(n)=\\{1,2\\}$', "
    "'$d = \\frac{L}{R}$'); never write bare LaTeX outside $...$; no $$; write a product as 'a y_1[n]' "
    "or 'a \\cdot y_1[n]', never with a comma or LaTeX spacing commands; keep each line short enough "
    "for one board line and split a long computation over several steps.\n"
    "Write and speak in the language of the student's question. Respond with JSON only: "
    '{"title":"...","given":["..."],"find":"...","steps":[{"write":"...","say":"..."}],'
    '"result":"...","check":"...","closing":"..."} where \'given\' lists the known quantities, '
    "'find' is what is asked, 'check' is one short sanity check (units, size, a special case) and "
    "'closing' is one spoken sentence that wraps up."
)


def _context(sources: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {s.get('text', '')}" for i, s in enumerate(sources or [], 1))


def _wb_texts(data: dict) -> list[str]:
    steps = data.get("steps") or []
    return [str(s.get("say") or "") for s in steps if isinstance(s, dict)] + [str(data.get("closing") or "")]


def whiteboard(question: str, sources: list[dict], provider: Optional[str] = None,
               language: Optional[str] = None) -> Optional[dict]:
    cfg = board_cfg()
    language = language or question_language(question)
    messages = [{"role": "system", "content": WHITEBOARD_SYSTEM},
                {"role": "user", "content": f"COURSE CONTEXT:\n{_context(sources) or '(none)'}\n\n"
                                            f"QUESTION:\n{question}\n\n"
                                            f"Use 3-{cfg['max_steps']} steps." + _language_rule(language)}]
    data, used = _ask_json(messages, language, _wb_texts, provider, 0.2, int(cfg["max_tokens"]))
    steps = []
    for st in data.get("steps") or []:
        if isinstance(st, dict):
            write, say = str(st.get("write") or "").strip(), str(st.get("say") or "").strip()
        else:  # a plain string step
            write, say = str(st).strip(), ""
        if write:
            steps.append({"write": fix_line(write), "say": say})
    if not steps:
        return None
    given = data.get("given") or []
    given = [fix_line(str(g)) for g in (given if isinstance(given, list) else [given]) if str(g).strip()]
    return {"title": str(data.get("title") or "").strip(), "given": given[:6],
            "find": fix_line(str(data.get("find") or "")), "steps": steps[:int(cfg["max_steps"])],
            "result": fix_line(str(data.get("result") or "")), "check": str(data.get("check") or "").strip(),
            "closing": str(data.get("closing") or "").strip(), "provider": used}


# --- the lecturer's explanation over a page --------------------------------------------------------
LESSON_SYSTEM = (
    "You are the course lecturer. A student asked a question in class and you answer it with a short, "
    "complete mini-lesson, walking through the slides / pages on the screen and pointing at their "
    "lines one by one. Speak naturally, as in a live lecture, addressing the student ('Look at this "
    "line...', 'As I said in the video...').\n"
    "Structure: the first beat frames the question and says what we are going to look at; the middle "
    "beats go through the relevant lines in a logical order and, for each, explain in simple words "
    "what it means and why it matters for the question, with a short concrete example or analogy "
    "when it helps; the last beat ties the pieces together into the answer.\n"
    "Give 3-{max_beats} beats. Each beat: 'say' = 2-3 spoken sentences (formulas read in words, no "
    "LaTeX), 'line' = the number of the ONE line it explains (or null), 'point' = the key idea of the "
    "beat in at most 8 words (shown on screen as a bullet). Use ONLY the page lines, the video "
    "transcript and the given answer for facts; do not invent facts that are not there. Speak in the "
    "language of the student's question. Respond with JSON only: "
    '{{"beats":[{{"line":3,"say":"...","point":"..."}}],"closing":"...","summary":["..."]}} where '
    "closing is one wrap-up sentence and summary has 2-4 short take-away bullets."
)


def _fallback_lesson(question: str, highlights: list[str], video: Optional[dict]) -> dict:
    tr = question_language(question) == "tr"
    beats = []
    if video:
        beats.append({"line": None, "page": 0, "point": "",
                      "say": (f"Bu konuyu videoda {_mmss(video['start_ms'])} civarında anlatıyorum."
                              if tr else f"I explain this in the video around {_mmss(video['start_ms'])}.")})
    for i, line in enumerate(highlights):
        beats.append({"line": line, "page": 0, "point": line[:60],
                      "say": ("Şimdi şu satıra bakalım." if i == 0 else "Bir de şuraya dikkat edin.")
                      if tr else ("Let us look at this line." if i == 0 else "And note this part as well.")})
    return {"beats": beats, "closing": "", "summary": [], "narrated": False}


def _mmss(ms: int) -> str:
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _page_label(p: dict) -> str:
    kind = "slide" if p.get("type") == "pptx" else "page"
    return f"{p['document']}, {kind} {p['page']}"


def narrate_page(question: str, answer: str, lines, video: Optional[dict],
                 highlights: list[str], provider: Optional[str] = None,
                 pages: Optional[list[dict]] = None, language: Optional[str] = None) -> dict:
    """{beats: [{line: <exact page line or None>, page: <index into pages>, say, point}], closing,
    summary, narrated}. `lines` is one page's lines, or one list per page in `pages`. Lines are sent
    numbered (one numbering across the pages) and the model answers with numbers, so every
    highlighted line exists on its page."""
    cfg = board_cfg()
    per_page = lines if lines and isinstance(lines[0], list) else [lines or []]
    numbered, index = [], []   # index[n-1] = (page index, line)
    budget = 80
    for pi, plines in enumerate(per_page):
        shown = plines[:max(10, budget // len(per_page))]
        head = f"PAGE {pi + 1} ({_page_label(pages[pi])})" if pages and pi < len(pages) else f"PAGE {pi + 1}"
        numbered.append(head + ":")
        for ln in shown:
            index.append((pi, ln))
            numbered.append(f"{len(index)}. {ln}")
    video_part = (f"VIDEO TRANSCRIPT at {_mmss(video['start_ms'])}-{_mmss(video['end_ms'])}:\n"
                  f"{video.get('text', '')}\n\n" if video else "")
    body = "\n".join(numbered) if index else "(no text on this page)"
    messages = [{"role": "system", "content": LESSON_SYSTEM.format(max_beats=int(cfg["max_beats"]))},
                {"role": "user", "content": f"PAGE LINES:\n{body}\n\n"
                                            f"{video_part}ANSWER ALREADY GIVEN:\n{answer}\n\n"
                                            f"STUDENT QUESTION:\n{question}"
                                            + _language_rule(language or question_language(question))}]
    data, _ = _ask_json(messages, language or question_language(question),
                        lambda d: [str(b.get("say") or "") for b in (d.get("beats") or []) if isinstance(b, dict)]
                        + [str(d.get("closing") or "")], provider, 0.3, int(cfg["max_tokens"]))
    beats, page_now = [], 0
    for b in (data.get("beats") or [])[:int(cfg["max_beats"])]:
        if not isinstance(b, dict) or not str(b.get("say") or "").strip():
            continue
        n = b.get("line")
        if isinstance(n, str) and n.strip().isdigit():
            n = int(n.strip())
        line = None
        if isinstance(n, int) and 1 <= n <= len(index):
            page_now, line = index[n - 1]
        beats.append({"line": line, "page": page_now, "say": str(b["say"]).strip(),
                      "point": str(b.get("point") or "").strip()})
    if not beats:
        raise ValueError("The model returned no usable explanation steps.")
    summary = data.get("summary") or []
    summary = [str(x).strip() for x in (summary if isinstance(summary, list) else [summary]) if str(x).strip()]
    return {"beats": beats, "closing": str(data.get("closing") or "").strip(), "summary": summary[:4],
            "narrated": True}


# --- putting it together ---------------------------------------------------------------------------
def build_board(question: str, answer: str, course: str, lecture: Optional[int],
                sources: list[dict], provider: Optional[str] = None,
                store: Optional[VectorStore] = None) -> dict:
    """{view: whiteboard | page | none, video, page, whiteboard, is_math, notes}.
    Each part degrades on its own: no video file, no LibreOffice or an LLM error never stops
    the others."""
    cfg = board_cfg()
    out: dict = {"view": "none", "video": None, "page": None, "pages": [], "whiteboard": None, "lesson": None,
                 "is_math": False, "language": question_language(question), "notes": []}
    if not cfg["enabled"]:
        return out

    video = pick_video(sources)
    if video:
        from backend.tasks.study import upload_path
        try:
            upload_path(course, video["document"])
            out["video"] = {k: video[k] for k in ("document", "start_ms", "end_ms")}
        except (FileNotFoundError, ValueError):
            out["notes"].append(f"The video file {video['document']} is not on the server.")

    page = None
    if video:  # the slide being talked about at that moment
        try:
            page = match_page_for_video(video, course, lecture, store)
        except Exception as exc:  # noqa: BLE001
            log.warning("Video-to-page match failed: %s", exc)
    candidates = ([page] if page else []) + page_sources(sources)
    pages, all_lines, seen = [], [], set()
    for cand in candidates:   # the page on screen first, then the other cited pages
        key = (cand["document"], cand["page"])
        if key in seen or len(pages) >= max(1, int(cfg["max_pages"])):
            continue
        seen.add(key)
        try:
            pdf = pdf_for(course, cand["document"])
            plines = page_lines(pdf, cand["page"])
            cand["highlights"] = choose_highlights(plines, question, answer)
            cand["renderable"] = True
        except Exception as exc:  # noqa: BLE001  (file missing, no LibreOffice, ...)
            cand["highlights"], cand["renderable"], plines = [], False, []
            if not pages:
                out["notes"].append(str(exc))
        cand.pop("text", None)
        if pages and not cand["renderable"]:
            continue   # an extra page that cannot be drawn is simply left out
        pages.append(cand)
        all_lines.append(plines)
    if pages:
        out["page"], out["pages"] = pages[0], pages
    else:
        out["pages"] = []
    lines = all_lines[0] if all_lines else []

    mode = str(cfg["math_mode"]).lower()
    out["is_math"] = mode == "always" or (mode == "auto" and is_math_question(question, answer))
    if out["is_math"]:
        try:
            out["whiteboard"] = whiteboard(question, sources, provider, out["language"])
        except Exception as exc:  # noqa: BLE001
            out["notes"].append(f"Whiteboard unavailable: {exc}")

    if out["whiteboard"]:
        out["view"] = "whiteboard"
    elif out["page"] and out["page"]["renderable"]:
        out["view"] = "page"
        highlights = out["page"]["highlights"]
        lesson = None
        if cfg.get("narrate", True):
            try:
                lesson = narrate_page(question, answer, all_lines, video, highlights, provider,
                                      pages=out["pages"], language=out["language"])
            except Exception as exc:  # noqa: BLE001
                out["notes"].append(f"Lecturer explanation unavailable, showing the highlights: {exc}")
        lesson = lesson or _fallback_lesson(question, highlights, video)
        for pi, pg in enumerate(out["pages"]):   # each page's picture marks the lines pointed at on it
            pointed = [b["line"] for b in lesson["beats"] if b["line"] and b.get("page", 0) == pi]
            if pointed:
                pg["highlights"] = list(dict.fromkeys(pointed))
        used = {b.get("page", 0) for b in lesson["beats"]} | {0}
        if len(out["pages"]) > 1:   # keep only the pages the lesson actually visits
            keep = sorted(used)
            remap = {old: new for new, old in enumerate(keep)}
            out["pages"] = [out["pages"][i] for i in keep]
            for b in lesson["beats"]:
                b["page"] = remap.get(b.get("page", 0), 0)
            out["page"] = out["pages"][0]
        out["lesson"] = lesson
    out["language"] = question_language(question)
    said = (_wb_texts(out["whiteboard"]) if out["whiteboard"] else
            [b["say"] for b in (out["lesson"] or {}).get("beats") or []] if out["lesson"] and out["lesson"].get("narrated") else [])
    if " ".join(said).strip():   # labels and voice follow the language the explanation came back in
        out["language"] = text_language(" ".join(said))
    return out
