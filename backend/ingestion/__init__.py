from pathlib import Path

from backend.schemas import Segment

EXTENSIONS = {
    ".pdf": "pdf", ".pptx": "pptx", ".docx": "docx",
    ".mp4": "video", ".mkv": "video", ".avi": "video",
}


def source_type_of(path) -> str:
    ext = Path(path).suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    return EXTENSIONS[ext]


def load_file(path) -> list[Segment]:
    kind = source_type_of(path)
    if kind == "pdf":
        from backend.ingestion.pdf_loader import load_pdf
        return load_pdf(path)
    if kind == "pptx":
        from backend.ingestion.pptx_loader import load_pptx
        return load_pptx(path)
    if kind == "docx":
        from backend.ingestion.docx_loader import load_docx
        return load_docx(path)
    from backend.ingestion.video_stt import transcribe_video
    return transcribe_video(path)