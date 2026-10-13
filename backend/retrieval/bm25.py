"""Lexical BM25 index for hybrid search (plan B for risk R2: dense search missing exact course terms,
formula names, codes such as "CENG384" or proper names such as "Nyquist").

Turkish-aware normalisation:
  * dotted/dotless I handled before lower-casing (İ->i, I->ı),
  * diacritics folded (ç ğ ı ö ş ü -> c g i o s u) so ASCII-typed questions still match,
  * prefix stemming: every token is truncated to its first `prefix_len` characters. Fixed-prefix
    truncation is a well-known, surprisingly strong stemmer for agglutinative Turkish
    (Can et al., 2008 report first-5-character truncation performing on par with full morphological
    stemming). Set prefix_len to 0 to disable.

The index is built per filtered pool (course / lecture) and cached until the vector store changes.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)
_FOLD = str.maketrans("çğıöşüâîûÇĞÖŞÜÂÎÛ", "cgiosuaiuCGOSUAIU")


_PREFIX_RE: dict[int, re.Pattern] = {}


def _prefix_re(n: int) -> re.Pattern:
    # one C-level pass: keeps the first n chars of every word (>= 2 chars, or a single digit)
    if n not in _PREFIX_RE:
        _PREFIX_RE[n] = re.compile(rf"(?<!\w)(\w{{2,{n}}}|\d)\w*", re.UNICODE)
    return _PREFIX_RE[n]


def normalize(text: str, prefix_len: int = 5) -> list[str]:
    text = text.replace("İ", "i").replace("I", "ı").lower().translate(_FOLD)
    if prefix_len and prefix_len > 1:
        return _prefix_re(prefix_len).findall(text)
    return [t for t in _TOKEN.findall(text) if len(t) > 1 or t.isdigit()]


class BM25Index:
    """Okapi BM25 on a sparse term matrix (scikit-learn CountVectorizer + numpy scoring)."""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75, prefix_len: int = 5):
        from sklearn.feature_extraction.text import CountVectorizer
        self.docs = docs
        self.k1, self.b, self.prefix_len = k1, b, prefix_len
        self._vocab: dict[str, int] = {}
        if not docs:
            return
        vec = CountVectorizer(analyzer=lambda t: normalize(t, prefix_len), dtype=np.float32)
        try:
            tf = vec.fit_transform([d["text"] for d in docs]).tocsc()
        except ValueError:  # empty vocabulary
            return
        self._vocab = vec.vocabulary_
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = max(float(dl.mean()), 1e-9)
        df = np.diff(tf.indptr)
        n = len(docs)
        self._idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._tf = tf
        self._norm = (k1 * (1 - b + b * dl / avg)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.docs)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not self._vocab:
            return out
        for t in set(normalize(query, self.prefix_len)):
            j = self._vocab.get(t)
            if j is None:
                continue
            lo, hi = self._tf.indptr[j], self._tf.indptr[j + 1]
            rows, tf = self._tf.indices[lo:hi], self._tf.data[lo:hi]
            out[rows] += self._idf[j] * tf * (self.k1 + 1) / (tf + self._norm[rows])
        return out

    def top(self, query: str, n: int) -> list[dict]:
        s = self.scores(query)
        if not len(s):
            return []
        n = min(n, len(s))
        idx = np.argpartition(-s, n - 1)[:n]
        idx = idx[np.argsort(-s[idx])]
        return [{**self.docs[i], "bm25": float(s[i])} for i in idx if s[i] > 0]


_CACHE: dict = {}
_CACHE_MAX = 32


def get_index(store, where: Optional[dict], prefix_len: int = 5) -> BM25Index:
    """Cached BM25 index for one filtered pool; rebuilt when the store's version changes."""
    key = (id(store), repr(where), prefix_len)
    hit = _CACHE.get(key)
    version = getattr(store, "version", 0)
    if hit is None or hit[0] != version:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        docs = [{"id": h["id"], "text": h["text"], "metadata": h["metadata"]}
                for h in store.get(where=where)]
        _CACHE[key] = (version, BM25Index(docs, prefix_len=prefix_len))
    return _CACHE[key][1]


def clear_cache() -> None:
    _CACHE.clear()
