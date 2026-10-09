import logging
from pathlib import Path

import fitz  # PyMuPDF

from backend.schemas import Segment

log = logging.getLogger(__name__)


def load_pdf(path) -> list[Segment]:
    """One segment per page. Scanned PDFs (no text layer) yield no text."""
    path = Path(path)
    segments: list[Segment] = []
    with fitz.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                segments.append(Segment(text=text, source_type="pdf",
                                        document=path.name, page=number))
    if not segments:
        log.warning("No text extracted from %s (scanned PDF? OCR is out of scope).", path.name)
    return segments