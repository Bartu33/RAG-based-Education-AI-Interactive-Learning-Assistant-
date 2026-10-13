from backend.indexing.chunker import RegexTokenizer, chunk_segments, count_tokens
from backend.schemas import Segment

TOK = RegexTokenizer()  # deterministic, no model download needed in CI


def test_text_chunks_keep_page():
    seg = Segment(text=" ".join(f"w{i}" for i in range(1000)),
                  source_type="pdf", document="a.pdf", page=3)
    chunks = chunk_segments([seg], chunk_size=100, overlap=20, tokenizer=TOK)
    assert len(chunks) > 1
    assert all(c.page == 3 and c.document == "a.pdf" for c in chunks)
    assert all(count_tokens(c.text, TOK) <= 100 for c in chunks)


def test_overlap_is_respected():
    seg = Segment(text=" ".join(f"w{i}" for i in range(300)), source_type="pdf", document="a.pdf")
    chunks = chunk_segments([seg], chunk_size=100, overlap=20, tokenizer=TOK, sentence_aware=False)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == {f"w{i}" for i in range(300)}           # nothing lost


def test_sentence_aware_cut():
    sentence = " ".join(["kelime"] * 9) + "."                   # 10 tokens per sentence
    seg = Segment(text=" ".join([sentence] * 30), source_type="docx", document="d.docx")
    chunks = chunk_segments([seg], chunk_size=45, overlap=0, tokenizer=TOK)
    assert all(c.text.endswith(".") for c in chunks)
    assert all(count_tokens(c.text, TOK) <= 45 for c in chunks)


def test_short_segment_untouched():
    seg = Segment(text="Kısa slayt metni.", source_type="pptx", document="s.pptx", page=2)
    assert chunk_segments([seg], chunk_size=100, overlap=10, tokenizer=TOK) == [seg]


def test_video_chunks_keep_timestamps():
    segs = [Segment(text=" ".join(["word"] * 10), source_type="video", document="v.mp4",
                    timestamp_start_ms=i * 5000, timestamp_end_ms=i * 5000 + 4000, confidence=0.9)
            for i in range(20)]
    chunks = chunk_segments(segs, chunk_size=50, overlap=10, tokenizer=TOK)
    assert len(chunks) > 1
    assert all(c.timestamp_start_ms < c.timestamp_end_ms for c in chunks)
    assert chunks[0].timestamp_start_ms == 0
    assert chunks[-1].timestamp_end_ms == 19 * 5000 + 4000
    assert all(count_tokens(c.text, TOK) <= 50 for c in chunks)


def test_video_chunks_do_not_mix_documents():
    segs = [Segment(text="a b c", source_type="video", document=d,
                    timestamp_start_ms=i, timestamp_end_ms=i + 1) for i, d in enumerate("xxyy")]
    chunks = chunk_segments(segs, chunk_size=100, overlap=10, tokenizer=TOK)
    assert [c.document for c in chunks] == ["x", "y"]
