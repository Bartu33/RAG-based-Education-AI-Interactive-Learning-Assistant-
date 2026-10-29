"""Formulas written by the LLM, made safe to show.

The model does not always wrap its LaTeX in $...$ (e.g. "x(n)=\\{1,2\\}\\times 3"), and then the
board shows backslashes instead of symbols. Everything written on the whiteboard goes through
`fix_line` (Markdown/KaTeX in the UI), the video draws it with `for_mathtext` (matplotlib), and a
formula that has to be read aloud goes through `speakable`.
"""
from __future__ import annotations

import re

_CMD = re.compile(r"\\[A-Za-z]+|\\[{}|,;:!]|[\^_]\{")
# LaTeX spacing (\, \; \: \!): a plain space is just as readable, and a backslash lost on the way
# would turn "a\,y_1" into "a, y_1", which reads as a list instead of a product
_SPACING = re.compile(r"(?<!\\)\\[,;:!]")
_PROSE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]{3,}[.,:;!?]*")
_JOIN_WORDS = {"ve", "ile", "ise", "ya", "da", "de", "mi", "and", "or", "is", "of", "the", "for", "so",
               "to", "in", "at", "if"}
_MATH_WORDS = {"sin", "cos", "tan", "cot", "log", "ln", "exp", "lim", "max", "min", "arg", "det", "mod"}


def _is_prose(tok: str) -> bool:
    if tok.lower() in _JOIN_WORDS:
        return True
    bare = tok.rstrip(".,:;!?")
    return bool(_PROSE.fullmatch(tok)) and bare.lower() not in _MATH_WORDS and "\\" not in tok


def fix_line(text: str) -> str:
    """One whiteboard line with every formula inside $...$ (inline), prose left as text."""
    return _SPACING.sub(" ", _fix_line(text))


def _fix_line(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    s = re.sub(r"\\\[(.+?)\\\]", r"$\1$", s, flags=re.S)
    s = re.sub(r"\\\((.+?)\\\)", r"$\1$", s, flags=re.S)
    s = s.replace("$$", "$")
    if "$" in s:
        return s + "$" if s.count("$") % 2 else s
    if not _CMD.search(s):
        return s
    # bare LaTeX: wrap the runs of math tokens, keep the words of a sentence outside
    out, run = [], []
    for tok in s.split():
        if _is_prose(tok):
            if run:
                out.append("$" + " ".join(run) + "$")
                run = []
            out.append(tok)
        else:
            run.append(tok)
    if run:
        out.append("$" + " ".join(run) + "$")
    return " ".join(out)


def fix_markdown(text: str) -> str:
    r"""Free text (an answer) for Streamlit: \( \) and \[ \] become $ / $$, which it renders."""
    t = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n$$" + m.group(1).strip() + "$$\n", text or "", flags=re.S)
    return re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", t, flags=re.S)


# --- matplotlib mathtext (video) ----------------------------------------------------------------
def _brace_arg(s: str, i: int) -> tuple[str, int]:
    """The {...} group starting at s[i] (balanced) and the index after it."""
    if i >= len(s) or s[i] != "{":
        return "", i
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def _replace_cmd(s: str, name: str, fn) -> str:
    r"""Replace every \name{arg} by fn(arg) (balanced braces; \namex and a bare \name are kept)."""
    key, out, i = "\\" + name, [], 0
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        end = j + len(key)
        arg, k = _brace_arg(s, end)
        if (end < len(s) and s[end].isalpha()) or k == end:
            out.append(s[i:end])
            i = end
            continue
        out.append(s[i:j] + fn(arg))
        i = k


def _mt_math(m: str) -> str:
    """LaTeX inside $...$ -> what matplotlib's mathtext understands."""
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\displaystyle", ""), ("\\left.", ""),
                 ("\\right.", ""), ("\\cdotp", "\\cdot"), ("\\dots", "\\ldots"), ("\\lvert", "|"),
                 ("\\rvert", "|"), ("\\lVert", "\\Vert"), ("\\rVert", "\\Vert"), ("\\ne ", "\\neq "),
                 ("\\le ", "\\leq "), ("\\ge ", "\\geq "), ("\\implies", "\\Rightarrow"),
                 ("\\iff", "\\Leftrightarrow"), ("&", ""), ("\\\\", "\\;\\;")):
        m = m.replace(a, b)
    m = re.sub(r"\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}", "", m)
    m = re.sub(r"(^|[,(=\[;]|\\\{)\s*-", r"\1{-}", m)   # a leading minus is a sign, not "a - b"
    m = _replace_cmd(m, "boxed", lambda a: a)
    for name in ("text", "textrm", "mbox", "operatorname", "textbf", "mathrm"):
        bold = name == "textbf"
        m = _replace_cmd(m, name, lambda a, bold=bold:
                         ("\\mathbf{" if bold else "\\mathrm{") + re.sub(r"(?<!\\) ", r"\\ ", a) + "}")
    return m


def for_mathtext(text: str) -> str:
    """A fixed whiteboard line, its $...$ parts made mathtext-friendly; a lone $ is escaped."""
    s = fix_line(text)
    parts = s.split("$")
    if len(parts) % 2 == 0:  # unbalanced: show the dollar sign
        return s.replace("$", r"\$")
    return "".join(f"${_mt_math(p)}$" if i % 2 else p for i, p in enumerate(parts)).replace("$$", "")


# --- reading a formula aloud ----------------------------------------------------------------------
_SPOKEN = {
    "tr": [(r"\\times|\\cdot|×|·", " çarpı "), (r"\\div|÷", " bölü "), (r"\\pm", " artı eksi "),
           (r"\\approx|≈", " yaklaşık "), (r"\\neq|\\ne\b|≠", " eşit değil "), (r"\\leq?\b|≤", " küçük eşit "),
           (r"\\geq?\b|≥", " büyük eşit "), (r"\\sqrt", " karekök "), (r"\\infty|∞", " sonsuz "),
           (r"\\pi\b|π", " pi "), (r"\\to\b|\\rightarrow|→", " giderken "), (r"=", " eşittir "),
           (r"\+", " artı "), (r"(?<=[\w)\]}])\s*-\s*(?=[\w(\\])", " eksi "), (r"\^\{?2\}?", " kare "),
           (r"\^\{?3\}?", " küp "), (r"\^", " üssü ")],
    "en": [(r"\\times|\\cdot|×|·", " times "), (r"\\div|÷", " divided by "), (r"\\pm", " plus or minus "),
           (r"\\approx|≈", " approximately "), (r"\\neq|\\ne\b|≠", " is not equal to "),
           (r"\\leq?\b|≤", " is at most "), (r"\\geq?\b|≥", " is at least "), (r"\\sqrt", " square root of "),
           (r"\\infty|∞", " infinity "), (r"\\pi\b|π", " pi "), (r"\\to\b|\\rightarrow|→", " goes to "),
           (r"=", " equals "), (r"\+", " plus "), (r"(?<=[\w)\]}])\s*-\s*(?=[\w(\\])", " minus "),
           (r"\^\{?2\}?", " squared "), (r"\^\{?3\}?", " cubed "), (r"\^", " to the power ")],
}


_GREEK_EN = {"alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta", "Delta": "delta",
             "theta": "theta", "lambda": "lambda", "mu": "mu", "omega": "omega", "Omega": "omega",
             "phi": "phi", "Phi": "phi", "sigma": "sigma", "tau": "tau", "circ": "degrees", "sum": "sum of",
             "int": "integral of"}
_GREEK_TR = {**_GREEK_EN, "theta": "teta", "lambda": "lamda", "circ": "derece", "sum": "toplam",
             "int": "integral", "phi": "fi"}


def speakable(text: str, language: str = "en") -> str:
    """A formula as words for the voice / subtitle ('d = L/R' -> 'd eşittir L bölü R')."""
    t = re.sub(r"\^\{?\\circ\}?", r"\\circ", fix_line(text).replace("$", " "))
    t = re.sub(r"\\d?t?frac\{([^{}]*)\}\{([^{}]*)\}",
               r"\1 bölü \2" if language == "tr" else r"\1 over \2", t)
    t = t.replace("/", " bölü " if language == "tr" else " over ")
    for pat, rep in _SPOKEN.get(language, _SPOKEN["en"]):
        t = re.sub(pat, rep, t)
    words = _GREEK_TR if language == "tr" else _GREEK_EN
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: f" {words.get(m.group(1), m.group(1) if m.group(1) in _MATH_WORDS else '')} ", t)
    t = re.sub(r"\\.|[{}_]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --- formulas inside spoken sentences (subtitles / voice) -----------------------------------------
_SUB = str.maketrans("0123456789+-=()aeoxijnmkt", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓᵢⱼₙₘₖₜ")
_SUP = str.maketrans("0123456789+-=()nijk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱʲᵏ")
_UNICODE = {"times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "approx": "≈", "neq": "≠", "ne": "≠",
            "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "infty": "∞", "to": "→", "rightarrow": "→",
            "Rightarrow": "⇒", "leftarrow": "←", "sum": "∑", "prod": "∏", "int": "∫", "partial": "∂",
            "nabla": "∇", "in": "∈", "circ": "°", "degree": "°", "ldots": "…", "cdots": "⋯", "dots": "…",
            "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε", "theta": "θ",
            "lambda": "λ", "mu": "μ", "pi": "π", "Pi": "Π", "rho": "ρ", "sigma": "σ", "Sigma": "Σ", "tau": "τ",
            "phi": "φ", "Phi": "Φ", "omega": "ω", "Omega": "Ω", "quad": " ", "qquad": "  ", "{": "{", "}": "}",
            ",": " ", ";": " ", ":": " ", "!": "", " ": " ", "|": "‖", "angle": "∠", "sqrt": "√"}


def _script(s: str, table) -> str:
    s = s.strip()
    out = s.translate(table)
    return out if all(ch != c or not c.isalnum() for ch, c in zip(out, s)) or len(s) == 1 else "^(" + s + ")"


def pretty(math: str) -> str:
    """LaTeX -> readable plain text with Unicode symbols, for subtitles and panels
    ('x_2 \\times 3^2' -> 'x₂ × 3²', '\\frac{a}{b}' -> 'a/b')."""
    m = re.sub(r"\^\{?\\circ\}?", "°", math)
    for _ in range(3):   # nested fractions / roots
        m = re.sub(r"\\d?t?frac\{([^{}]*)\}\{([^{}]*)\}",
                   lambda g: (f"({g[1]})" if len(g[1]) > 1 else g[1]) + "/" + (f"({g[2]})" if len(g[2]) > 1 else g[2]), m)
        m = re.sub(r"\\sqrt\{([^{}]*)\}", lambda g: "√" + (g[1] if len(g[1]) == 1 else f"({g[1]})"), m)
        m = re.sub(r"\\(?:text|mathrm|textrm|mathbf|operatorname|boxed|mbox)\{([^{}]*)\}", r"\1", m)
    m = re.sub(r"\\left|\\right|\\displaystyle", "", m)
    m = re.sub(r"\\([A-Za-z]+|[{},;:! |])", lambda g: _UNICODE.get(g[1], g[1] if g[1] in _MATH_WORDS else ""), m)
    m = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda g: _script(g[1] or g[2], _SUB), m)
    m = re.sub(r"\^\{([^{}]*)\}|\^(\w|[+-])", lambda g: _script(g[1] or g[2], _SUP), m)
    m = m.replace("{", "").replace("}", "").replace("‖", "|")
    return re.sub(r"\s+", " ", m).strip()


_MASK = "\u2063M{}\u2063"


def mask_math(text: str) -> tuple[str, list[str]]:
    """Replace each formula of a sentence by a placeholder without spaces (so cutting the sentence
    into subtitle pieces never splits a formula). Returns (masked text, formulas)."""
    fixed = fix_line(text or "")
    parts = fixed.split("$")
    if len(parts) % 2 == 0:
        parts = fixed.replace("$", "").split("\0")
    formulas, out = [], []
    for i, p in enumerate(parts):
        if i % 2:
            formulas.append(p)
            out.append(_MASK.format(len(formulas) - 1))
        else:
            out.append(p)
    return "".join(out), formulas


def unmask(text: str, formulas: list[str], render) -> str:
    return re.sub("\u2063M(\\d+)\u2063", lambda g: render(formulas[int(g[1])]), text)


def plain_sentence(text: str) -> str:
    """A sentence for a subtitle / panel: formulas shown with Unicode symbols, no $ or LaTeX."""
    masked, formulas = mask_math(text)
    return unmask(masked, formulas, pretty)


def spoken_sentence(text: str, language: str = "en") -> str:
    """A sentence for the voice: formulas read out in words, no $ or LaTeX."""
    masked, formulas = mask_math(text)
    return unmask(masked, formulas, lambda f: speakable(f"${f}$", language))
