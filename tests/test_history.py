"""Accounts (backend/users.py) and per-test practice history (backend/evaluation/history.py),
through the HTTP endpoints, in a temporary SQLite file."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main  # noqa: E402
from backend import config  # noqa: E402
from backend.users import hash_password, verify_password  # noqa: E402

client = TestClient(main.app)
QS = [{"id": "m1", "type": "Multiple Choice", "difficulty": "Medium", "question": "Periodic if?",
       "options": ["x(n)=x(n+N)", "x(n)=0", "a", "b"], "correct_answer": "x(n)=x(n+N)",
       "explanation": "def", "source": "p. 92", "course": "CENG336", "test_id": "ignored"},
      {"id": "s1", "type": "Short Answer", "difficulty": "Medium", "question": "Define u(n)",
       "options": None, "correct_answer": "1 for n>=0, 0 for n<0", "explanation": "e", "source": "p. 66"},
      {"id": "t1", "type": "True/False", "difficulty": "Medium", "question": "a>1 compresses",
       "options": None, "correct_answer": "True", "explanation": "", "source": "p. 80"}]


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    monkeypatch.setitem(config.get_config(), "database", {"path": str(tmp_path / "app.db")})


def auth(student_no="220201032", password="secret1"):
    r = client.post("/auth/register", json={"student_no": student_no, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_password_hashing():
    stored = hash_password("secret1")
    assert stored.startswith("pbkdf2_sha256$") and "secret1" not in stored
    assert verify_password("secret1", stored) and not verify_password("secret2", stored)


def test_register_login_logout():
    h = auth()
    assert client.get("/auth/me", headers=h).json() == {"student_no": "220201032"}
    assert client.post("/auth/register", json={"student_no": "220201032", "password": "xxxxxx"}).status_code == 400
    assert client.post("/auth/login", json={"student_no": "220201032", "password": "wrong!"}).status_code == 401
    r = client.post("/auth/login", json={"student_no": " 220201032 ", "password": "secret1"})
    assert r.status_code == 200 and r.json()["student_no"] == "220201032"
    client.post("/auth/logout", headers=h)
    assert client.get("/auth/me", headers=h).status_code == 401          # token dropped
    assert client.get("/history").status_code == 401                     # no token


@pytest.mark.parametrize("no,pw", [("ab", "secret1"), ("2202 01", "secret1"), ("220201032", "123")])
def test_register_validation(no, pw):
    assert client.post("/auth/register", json={"student_no": no, "password": pw}).status_code == 400


def test_tests_group_answers_latest_wins_and_unanswered_shown():
    h = auth()
    first = client.post("/tests", json={"course": "CENG336", "lecture": 1, "difficulty": "Medium",
                                        "questions": QS}, headers=h).json()["id"]
    second = client.post("/tests", json={"course": "CENG336", "questions": QS[:1]}, headers=h).json()["id"]
    post = lambda body: client.post("/attempts", json=body, headers=h)  # noqa: E731
    assert post({"test_id": first, "question_id": "m1", "student_answer": "B) x(n)=0", "is_correct": False}).status_code == 200
    post({"test_id": first, "question_id": "m1", "student_answer": "A) x(n)=x(n+N)", "is_correct": True})
    post({"test_id": first, "question_id": "s1", "student_answer": "1 for n>=0", "score": 64,
          "feedback": ["add n<0"], "dims": [{"name": "Factual accuracy", "weight": 0.4, "score": 0.7}]})
    post({"test_id": second, "question_id": "m1", "student_answer": "B) x(n)=0", "is_correct": False})

    body = client.get("/history", headers=h).json()
    newest, oldest = body["tests"]
    assert (newest["id"], newest["number"], oldest["number"]) == (second, 2, 1)
    m1, s1, t1 = oldest["questions"]
    assert m1["attempt"]["is_correct"] is True and m1["attempt"]["tries"] == 2   # latest answer counts
    assert s1["attempt"]["feedback"] == ["add n<0"] and t1["attempt"] is None   # unanswered kept
    assert "test_id" not in m1                                                   # only question fields stored
    assert oldest["summary"] == {"questions": 3, "answered": 2, "choice": {"count": 1, "correct": 1},
                                 "open": {"count": 1, "average": 64.0}}
    assert body["summary"]["tests"] == 2 and body["summary"]["choice"] == {"count": 2, "correct": 1}


def test_students_cannot_touch_each_others_tests():
    ayse, mehmet = auth("ayse01"), auth("mehmet02")
    test = client.post("/tests", json={"course": "C", "questions": QS}, headers=ayse).json()["id"]
    r = client.post("/attempts", json={"test_id": test, "question_id": "m1", "student_answer": "x",
                                       "is_correct": True}, headers=mehmet)
    assert r.status_code == 404
    assert client.get("/history", headers=mehmet).json()["tests"] == []
    assert client.delete("/history", headers=mehmet).json() == {"deleted_tests": 0}
    assert client.delete("/history", headers=ayse).json() == {"deleted_tests": 1}
    assert client.get("/history", headers=ayse).json()["tests"] == []


def test_invalid_attempts_are_rejected():
    h = auth()
    test = client.post("/tests", json={"course": "C", "questions": QS}, headers=h).json()["id"]
    base = {"test_id": test, "question_id": "m1", "student_answer": "x", "is_correct": True}
    assert client.post("/attempts", json={**base, "student_answer": " "}, headers=h).status_code == 400
    assert client.post("/attempts", json={**base, "is_correct": None}, headers=h).status_code == 400
    assert client.post("/attempts", json={**base, "question_id": "zz"}, headers=h).status_code == 400
    assert client.post("/tests", json={"course": "C", "questions": []}, headers=h).status_code == 400
