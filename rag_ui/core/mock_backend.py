# Added for testing; this will be replaced later.
import random, time
import pandas as pd

COURSES = {"CENG384": "Signals and Systems", "SENG449": "Software Engineering Project"}
RUBRIC = [("Factual accuracy", 0.40), ("Key concept coverage", 0.30),
          ("Course terminology alignment", 0.20), ("Completeness", 0.10)]
QTYPES = ["Multiple Choice", "True/False", "Short Answer", "Conceptual/Computational"]

def ingest(files, course, lecture):
    """Yield (stage, progress, records); the final step returns the record list."""
    stages = ["Text extraction / STT", "Chunking", "Local BGE-m3 embedding", "Writing to ChromaDB"]
    for i, s in enumerate(stages):
        time.sleep(0.4); yield s, (i + 1) / len(stages), None
    recs = []
    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        stype = ext if ext in ("pdf", "pptx", "docx") else "video"
        recs.append({"Document": f.name, "Course": course, "Week": lecture, "Type": stype,
                     "Chunks": random.randint(40, 400),
                     "Low-confidence segments": random.randint(0, 5) if stype == "video" else 0})
    yield "Complete", 1.0, recs

def ask(question, course, lecture, ts_range=None, cfg=None):
    time.sleep(0.8)
    f = round(random.uniform(0.45, 0.98), 2)
    srcs = [{"type": "video", "document": f"Lecture{lecture:02d}.mp4", "page": None,
             "start_ms": (ts_range[0] if ts_range else 12) * 60000,
             "end_ms": (ts_range[1] if ts_range else 13) * 60000,
             "score": 0.91, "text": "The Nyquist theorem states that the sampling frequency must be at least twice the highest frequency in the signal..."},
            {"type": "pdf", "document": "Oppenheim_Ch7.pdf", "page": 412, "start_ms": None,
             "end_ms": None, "score": 0.84, "text": "Sampling theorem: if x(t) is band-limited, ..."},
            {"type": "pptx", "document": f"Slides_L{lecture}.pptx", "page": 18, "start_ms": None,
             "end_ms": None, "score": 0.78, "text": "Aliasing occurs when fs < 2·fmax, causing the spectrum to overlap."}]
    return {"answer": "According to the Nyquist theorem, the sampling frequency must be at least twice the highest frequency in the signal; otherwise, aliasing occurs [1][3].",
            "faithfulness": f, "sources": srcs, "retrieval_ms": random.randint(40, 300),
            "total_s": round(random.uniform(1.2, 4.5), 1), "tokens_sent": random.randint(1200, 3000),
            "candidates": 20, "filtered_from": 8000}

def generate_questions(course, lecture, types, n, difficulty):
    time.sleep(1.0); out = []
    for i in range(n):
        t = types[i % len(types)]
        out.append({"id": f"q{random.randint(1000,9999)}_{i}", "type": t, "difficulty": difficulty,
               "question": f"[{t}] Sample question #{i+1} about the Nyquist sampling theorem",
               "options": ["fs ≥ 2·fmax", "fs ≥ fmax", "fs = fmax/2", "fs ≤ fmax"] if t == "Multiple Choice" else None,
               "correct_answer": "fs ≥ 2·fmax", "explanation": "To prevent aliasing, the sampling frequency must be at least twice the highest frequency.",
               "source": f"Lecture{lecture:02d}.mp4 · 12:20–13:17"})
    return out

def evaluate(question, student_answer, course):
    time.sleep(1.0)
    sc = [round(random.uniform(0.4, 1.0), 2) for _ in RUBRIC]
    total = round(sum(s * w for s, (_, w) in zip(sc, RUBRIC)) * 100)
    return {"total": total, "dimensions": [{"name": n, "weight": w, "score": s} for s, (n, w) in zip(sc, RUBRIC)],
            "feedback": ["✅ You correctly described the relationship between sampling and frequency.",
                 "⚠️ The answer does not mention aliasing (a key concept is missing).",
                 "⚠️ The course term 'sampling limit' was used instead of 'Nyquist rate'."],
            "source": "Slides_L5.pptx · Slide 18",
            "follow_up_question": "For a signal sampled at 8 kHz, what is the highest frequency that can be represented without aliasing?"}

def benchmark(sizes=(1000, 10000, 50000, 100000)):
    time.sleep(1.5)
    return pd.DataFrame({"Corpus (chunks)": sizes,
        "Retrieval (ms)": [45, 110, 260, 480], "Response (s)": [2.1, 2.6, 3.4, 4.7],
        "RAGAs Faithfulness": [0.91, 0.89, 0.87, 0.85], "Hallucination (%)": [4.0, 6.0, 8.5, 11.0]})