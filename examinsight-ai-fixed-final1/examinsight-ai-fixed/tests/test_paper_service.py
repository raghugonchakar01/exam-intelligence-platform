import sqlite3

import pytest

from app import paper_service
from app.pyq_pipeline import initialize_schema


def seed_db(path, count=8):
    conn = sqlite3.connect(path)
    initialize_schema(conn)
    conn.execute("INSERT INTO pyq_papers (id,title,grade,subject,year,board) VALUES ('p','Science PYQ',10,'Science',2022,'KSEAB')")
    concepts = ["photosynthesis", "refraction", "oxidation", "heredity", "resistance", "ecosystem", "respiration", "magnetism"]
    for number in range(1, count + 1):
        conn.execute("INSERT INTO pyq_questions (id,paper_id,question_number,question_text,marks,grade,subject,year,board) VALUES (?,?,?,?,?,?,?,?,?)",
                     (f"q{number}", "p", number, f"Explain the scientific principle of {concepts[number - 1]} with a distinct example.", 2, 10, "Science", 2022, "KSEAB"))
    conn.commit()
    conn.close()


def test_pyq_paper_has_exact_marks_and_provenance(tmp_path, monkeypatch):
    db = tmp_path / "paper.db"
    seed_db(db, count=8)
    monkeypatch.setattr(paper_service, "DB_PATH", str(db))
    paper = paper_service.generate_question_paper(grade=10, subject="Science", board="KSEAB", total_marks=8,
                                                  difficulty="Moderate", question_pattern="Mixed")
    questions = [q for section in paper["sections"] for q in section["questions"]]
    assert sum(q["marks"] for q in questions) == 8
    assert all(q["is_pyq"] and q["source_question_id"] for q in questions)
    assert len({q["question"] for q in questions}) == len(questions)


def test_no_source_data_is_an_explicit_error_not_a_fabricated_paper(tmp_path, monkeypatch):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    initialize_schema(conn)
    conn.close()
    monkeypatch.setattr(paper_service, "DB_PATH", str(db))
    with pytest.raises(paper_service.GenerationUnavailable, match="No stored PYQ"):
        paper_service.generate_question_paper(grade=10, subject="Science", board="KSEAB", total_marks=5,
                                              difficulty="Moderate", question_pattern="Mixed")
