import hashlib
import json
import sqlite3
from pathlib import Path

import fitz

from app import pyq_pipeline


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def connection(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.row_factory = sqlite3.Row
    pyq_pipeline.initialize_schema(conn)
    return conn


def test_source_records_are_persisted_and_legacy_data_is_untouched(tmp_path):
    conn = connection(tmp_path)
    conn.execute("INSERT INTO pyq_papers (id,title,grade,subject,year,file_hash,extracted_text) VALUES (?,?,?,?,?,?,?)",
                 ("legacy", "Existing paper", 10, "Science", 2022, "a" * 64, "1. Explain photosynthesis in green plants."))
    conn.execute("INSERT INTO pyq_questions (id,paper_id,question_number,question_text,marks,grade,subject,year) VALUES (?,?,?,?,?,?,?,?)",
                 ("legacyq", "legacy", 1, "Explain photosynthesis in green plants.", 2, 10, "Science", 2022))
    conn.commit()
    pyq_pipeline.initialize_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM pyq_papers").fetchone()[0] == 1
    row = conn.execute("SELECT status,paper_id,questions_count FROM pyq_documents WHERE paper_id='legacy'").fetchone()
    assert dict(row) == {"status": "EXTRACTED_SUCCESSFULLY", "paper_id": "legacy", "questions_count": 1}


def test_extract_questions_keeps_numbering_marks_and_page_reference():
    text = "1. Define photosynthesis. (2 marks)\n2. Describe the role of chlorophyll. [3 Marks]\n--- PAGE BREAK ---\n3. Explain respiration. (5)"
    items = pyq_pipeline.extract_questions(text)
    assert [(item["num"], item["marks"], item["page"]) for item in items] == [(1, 2.0, 1), (2, 3.0, 1), (3, 5.0, 2)]


def test_pdf_validation_rejects_html_and_accepts_real_pdf(tmp_path):
    html = tmp_path / "error.pdf"
    html.write_text("<html>not a PDF</html>", encoding="utf-8")
    assert not pyq_pipeline.validate_pdf(html)[0]
    pdf = tmp_path / "real.pdf"
    make_pdf(pdf, "1. Explain a valid question paper item. (2 marks)")
    valid, _, pages = pyq_pipeline.validate_pdf(pdf)
    assert valid and pages == 1


def test_initialize_schema_rebases_moved_project_pdf_paths(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    monkeypatch.setattr(pyq_pipeline, "PROJECT_ROOT", project_root)
    conn = connection(tmp_path)

    relative_path = Path("pyq/class10/mathematics/2024/paper.pdf")
    local_pdf = project_root / relative_path
    local_pdf.parent.mkdir(parents=True)
    make_pdf(local_pdf, "1. Explain a valid question paper item. (2 marks)")
    digest = pyq_pipeline.sha256_file(local_pdf)
    old_path = tmp_path / "previous-project" / relative_path

    conn.execute(
        """INSERT INTO pyq_papers (id,title,grade,subject,file_path,file_hash,status)
           VALUES (?,?,?,?,?,?,?)""",
        ("moved-paper", "Moved paper", 10, "Mathematics", str(old_path), digest, "READY"),
    )
    document_id = pyq_pipeline._upsert_document(
        conn, "https://example.test/moved.pdf", "https://example.test/index", "fixture",
        {"subject": "Mathematics"},
    )
    conn.execute(
        "UPDATE pyq_documents SET file_path=?, sha256=? WHERE id=?",
        (str(old_path), digest, document_id),
    )
    conn.commit()
    conn.row_factory = None

    pyq_pipeline.initialize_schema(conn)

    assert conn.execute(
        "SELECT file_path FROM pyq_papers WHERE id='moved-paper'"
    ).fetchone()[0] == relative_path.as_posix()
    assert conn.execute(
        "SELECT file_path FROM pyq_documents WHERE id=?", (document_id,)
    ).fetchone()[0] == relative_path.as_posix()


def test_invalid_download_payloads_are_quarantined_and_preserved(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    monkeypatch.setattr(pyq_pipeline, "PROJECT_ROOT", project_root)
    conn = connection(tmp_path)
    relative_path = Path("class10/mathematics/2024/paper.pdf")
    destination = project_root / "pyq" / relative_path
    destination.parent.mkdir(parents=True)

    old_invalid = destination.with_suffix(".pdf.invalid")
    old_invalid.write_text("<html>previous invalid response</html>", encoding="utf-8")
    pyq_pipeline.quarantine_unregistered_invalid_payloads(conn)
    quarantined_old = project_root / "pyq" / "_quarantine" / relative_path.with_suffix(".pdf.invalid")
    assert not old_invalid.exists()
    assert quarantined_old.read_text(encoding="utf-8") == "<html>previous invalid response</html>"

    temporary_download = destination.with_suffix(".pdf.part")
    temporary_download.write_text("<html>new invalid response</html>", encoding="utf-8")
    stored_path = Path(pyq_pipeline._preserve_invalid_payload(temporary_download, destination))
    assert stored_path.parent == quarantined_old.parent
    assert stored_path.name.startswith("paper.pdf.")
    assert stored_path.name.endswith(".invalid")
    assert stored_path.read_text(encoding="utf-8") == "<html>new invalid response</html>"


def test_register_deduplicates_identical_documents_but_keeps_candidate_provenance(tmp_path, monkeypatch):
    conn = connection(tmp_path)
    monkeypatch.setattr(pyq_pipeline, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "paper.pdf"
    make_pdf(path, "1. Explain a valid question paper item. (2 marks)")
    first = pyq_pipeline._upsert_document(conn, "https://example.test/a.pdf", "https://example.test/list", "fixture",
                                           {"grade": 10, "subject": "Science", "year": 2024, "title": "A", "file_name": "a.pdf"})
    second = pyq_pipeline._upsert_document(conn, "https://example.test/b.pdf", "https://example.test/list", "fixture",
                                            {"grade": 10, "subject": "Science", "year": 2024, "title": "B", "file_name": "b.pdf"})
    for document_id in (first, second):
        conn.execute("UPDATE pyq_documents SET file_path=?, resolved_url=? WHERE id=?", (str(path), "https://example.test/final.pdf", document_id))
    conn.commit()
    assert pyq_pipeline.process_document(conn, first) == "EXTRACTED_SUCCESSFULLY"
    assert pyq_pipeline.process_document(conn, second) == "DUPLICATE_OF_ANOTHER_DOCUMENT"
    assert conn.execute("SELECT COUNT(*) FROM pyq_papers").fetchone()[0] == 1
    duplicate = conn.execute("SELECT duplicate_of FROM pyq_documents WHERE id=?", (second,)).fetchone()[0]
    assert duplicate == conn.execute("SELECT paper_id FROM pyq_documents WHERE id=?", (first,)).fetchone()[0]


def test_report_counts_distinct_documents_not_links(tmp_path):
    conn = connection(tmp_path)
    for suffix in ("one", "two"):
        pyq_pipeline._upsert_document(conn, f"https://example.test/{suffix}.pdf", "https://example.test/index", "fixture", {"subject": "Science"})
    report = pyq_pipeline.build_report(conn)
    assert report["documents_by_status"]["DISCOVERED"] == 2
    assert report["distinct_registered_papers"] == 0
    assert "SHA-256-distinct" in report["counting_rule"]
