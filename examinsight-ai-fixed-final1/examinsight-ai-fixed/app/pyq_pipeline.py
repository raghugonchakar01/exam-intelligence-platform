"""Resumable, auditable SSLC previous-question-paper collection pipeline.

This module deliberately keeps discovery, download validation, extraction and
database registration separate.  A failed URL is a first-class record rather
than an item which disappears from a sync result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests

from app.kseab_source import extract_text_from_pdf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "sslc_pyq_sources.json"
REQUEST_HEADERS = {
    # Google Drive returns its viewer HTML for the previous collector token;
    # use a normal browser-identification string while preserving ordinary,
    # rate-limited, unauthenticated public access only.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.5",
}
READY_STATUSES = {"EXTRACTED_SUCCESSFULLY", "NEEDS_REVIEW"}
RETRYABLE_STATUSES = {"DISCOVERED", "QUEUED", "DOWNLOAD_FAILED", "PDF_VALIDATION_FAILED", "OCR_FAILED", "TEXT_EXTRACTION_PENDING"}


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_url(url: str) -> str:
    """Canonicalise a URL without conflating distinct non-Google documents."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    query = parse_qs(parsed.query, keep_blank_values=True)
    if host.endswith("drive.google.com"):
        file_id = (query.get("id") or [None])[0]
        if not file_id:
            match = re.search(r"/d/([^/]+)", parsed.path)
            file_id = match.group(1) if match else None
        if file_id:
            return f"https://drive.google.com/uc?{urlencode({'id': file_id, 'export': 'download'})}"
    keep = [(key, value) for key, values in sorted(query.items()) if not key.lower().startswith("utm_") for value in sorted(values)]
    return urlunparse((parsed.scheme.lower() or "https", host, parsed.path, "", urlencode(keep), ""))


def safe_component(value: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._")
    return cleaned[:140] or fallback


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stored_project_path(path: Path) -> str:
    """Store project files relative to the project so the database can move with it."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def validate_pdf(path: Path) -> Tuple[bool, str, int]:
    """Reject HTML error pages, tiny payloads, encrypted/corrupt PDFs and empty files."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"file unavailable: {exc}", 0
    if size < 64:
        return False, f"payload is too small ({size} bytes)", 0
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            return False, "payload does not have a PDF signature", 0
    try:
        import fitz

        document = fitz.open(path)
        pages = document.page_count
        encrypted = document.is_encrypted
        document.close()
        if encrypted:
            return False, "PDF is encrypted", pages
        if pages < 1:
            return False, "PDF has no pages", pages
        return True, "", pages
    except Exception as exc:  # PyMuPDF has several exception classes across versions.
        return False, f"PDF integrity check failed: {exc}", 0


class LinkCollector(HTMLParser):
    """Small dependency-free extractor retaining nearby visible text for metadata."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible_text: List[str] = []
        self._anchor: Optional[Dict[str, str]] = None
        self.links: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self._anchor = {"href": href, "text": "", "before": " ".join(self.visible_text[-180:])[-1800:]}

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.visible_text.append(text)
            if self._anchor is not None:
                self._anchor["text"] += (" " if self._anchor["text"] else "") + text

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._anchor is not None:
            item = self._anchor
            item["context"] = (item["before"] + " " + item["text"])[-2000:]
            self.links.append(item)
            self._anchor = None


def extract_links(html: str, page_url: str) -> List[Dict[str, str]]:
    parser = LinkCollector()
    parser.feed(html)
    result = []
    for item in parser.links:
        href = urljoin(page_url, item["href"])
        if href.startswith(("http://", "https://")):
            result.append({**item, "url": href})
    return result


def is_document_link(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith(".pdf") or parsed.netloc.lower().endswith("drive.google.com") and (
        "/uc" in path or "/file/" in path or "id" in parse_qs(parsed.query)
    )


def infer_metadata(context: str, url: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Infer only what the source text makes reasonably clear; leave the rest unknown."""
    data = dict(fallback or {})
    normal = re.sub(r"\s+", " ", context).strip()
    years = re.findall(r"\b(20(?:1[0-9]|2[0-9]))\b", normal)
    if years and not data.get("year"):
        data["year"] = int(years[-1])
    lower = normal.lower()
    subject_rules = [
        ("Kannada First Language", r"kannada\s*\(?(?:first|1st)\s*language"),
        ("Kannada Second Language", r"kannada\s*\(?(?:second|2nd)\s*language"),
        ("English First Language", r"english\s*\(?(?:first|1st)\s*language"),
        ("English", r"english\s*\(?(?:second|2nd)\s*language"),
        ("Hindi", r"hindi"),
        ("Mathematics", r"mathematics|maths"),
        ("Science", r"science"),
        ("Social Science", r"social\s+science"),
    ]
    for subject, pattern in subject_rules:
        if re.search(pattern, lower) and not data.get("subject"):
            data["subject"] = subject
            break
    if "kannada medium" in lower:
        data["medium"] = "Kannada"
    elif "english medium" in lower:
        data["medium"] = "English"
    if "supplementary" in lower or "exam 2" in lower or "examination 2" in lower:
        data["exam"] = "SSLC Supplementary Examination"
    else:
        data.setdefault("exam", "SSLC Main Examination")
    data.setdefault("board", "KSEAB")
    data.setdefault("grade", 10)
    data.setdefault("subject", "Unknown")
    data.setdefault("medium", "Unknown")
    data.setdefault("paper_type", "Previous-year question paper")
    data.setdefault("title", f"KSEAB SSLC {data['subject']} {data.get('year', 'Unknown year')}")
    data.setdefault("file_name", safe_component(Path(urlparse(url).path).name, "question-paper.pdf"))
    return data


def classify_candidate(context: str) -> Tuple[str, str]:
    lower = context.lower()
    if "answer key" in lower or "key answer" in lower:
        return "CONFIRMED_IRRELEVANT", "Source labels this document as an answer key, not a question paper."
    if "question" in lower or "download pdf" in lower:
        return "DISCOVERED", ""
    return "NEEDS_REVIEW", "Document link has insufficient surrounding metadata."


def ensure_column(conn: sqlite3.Connection, table: str, definition: str) -> None:
    name = definition.split()[0]
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def quarantine_unregistered_invalid_payloads(conn: sqlite3.Connection) -> None:
    """Move unregistered invalid download sidecars out of the browsable paper folders."""
    pyq_root = (PROJECT_ROOT / "pyq").resolve()
    quarantine_root = pyq_root / "_quarantine"
    registered_paths = set()
    for table in ("pyq_papers", "pyq_documents"):
        for (stored_path,) in conn.execute(
            f"SELECT file_path FROM {table} WHERE file_path IS NOT NULL"
        ):
            path = Path(stored_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            registered_paths.add(os.path.normcase(str(path.resolve())))

    for invalid_path in pyq_root.rglob("*.invalid*"):
        if not invalid_path.is_file():
            continue
        try:
            relative_path = invalid_path.relative_to(pyq_root)
        except ValueError:
            continue
        if relative_path.parts[0].lower() == "_quarantine":
            continue
        if os.path.normcase(str(invalid_path.resolve())) in registered_paths:
            continue

        quarantined_path = quarantine_root / relative_path
        quarantined_path.parent.mkdir(parents=True, exist_ok=True)
        if quarantined_path.exists():
            quarantined_path = quarantined_path.with_name(
                f"{quarantined_path.stem}.{uuid.uuid4().hex[:8]}{quarantined_path.suffix}"
            )
        invalid_path.replace(quarantined_path)


def rebase_project_file_paths(conn: sqlite3.Connection) -> None:
    """Rebase copied-database paths onto matching, validated files in this project."""
    project_root = PROJECT_ROOT.resolve()
    for table, hash_column in (("pyq_papers", "file_hash"), ("pyq_documents", "sha256")):
        rows = conn.execute(
            f"SELECT id, file_path, {hash_column} FROM {table} WHERE file_path IS NOT NULL"
        ).fetchall()
        for row in rows:
            paper_id, stored_value, expected_hash = row
            stored_path = Path(stored_value)
            if not stored_path.is_absolute():
                continue

            try:
                relative_path = stored_path.resolve().relative_to(project_root)
            except ValueError:
                parts = stored_path.parts
                pyq_index = next(
                    (index for index, part in enumerate(parts) if part.lower() == "pyq"),
                    None,
                )
                if pyq_index is None:
                    continue
                relative_path = Path(*parts[pyq_index:])

            candidate = (project_root / relative_path).resolve()
            try:
                candidate.relative_to(project_root)
            except ValueError:
                continue
            if not candidate.is_file() or not validate_pdf(candidate)[0]:
                continue
            if expected_hash and sha256_file(candidate) != expected_hash:
                continue

            conn.execute(
                f"UPDATE {table} SET file_path = ? WHERE id = ?",
                (relative_path.as_posix(), paper_id),
            )


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Idempotent additive migration; no legacy paper or question is removed."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pyq_papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            grade INTEGER NOT NULL,
            subject TEXT NOT NULL,
            year INTEGER,
            board TEXT DEFAULT 'KSEAB',
            file_name TEXT,
            file_path TEXT,
            file_type TEXT,
            uploaded_by TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            download_count INTEGER DEFAULT 0,
            source_url TEXT,
            file_hash TEXT,
            extracted_text TEXT,
            catalog_id TEXT,
            exam TEXT,
            medium TEXT,
            paper_type TEXT,
            questions_json TEXT,
            status TEXT,
            resolved_url TEXT,
            source_page TEXT,
            extraction_quality REAL,
            review_status TEXT
        );

        CREATE TABLE IF NOT EXISTS pyq_documents (
            id TEXT PRIMARY KEY,
            catalog_id TEXT,
            source_url TEXT NOT NULL,
            normalized_url TEXT NOT NULL UNIQUE,
            resolved_url TEXT,
            source_page TEXT,
            source_name TEXT,
            status TEXT NOT NULL DEFAULT 'DISCOVERED',
            status_reason TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            http_status INTEGER,
            content_type TEXT,
            byte_size INTEGER,
            page_count INTEGER,
            sha256 TEXT,
            file_path TEXT,
            paper_id TEXT,
            extracted_text TEXT,
            extraction_quality REAL,
            questions_count INTEGER NOT NULL DEFAULT 0,
            duplicate_of TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            discovered_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            last_attempt_at TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pyq_documents_status ON pyq_documents(status);
        CREATE INDEX IF NOT EXISTS idx_pyq_documents_paper ON pyq_documents(paper_id);
        CREATE INDEX IF NOT EXISTS idx_pyq_documents_sha256 ON pyq_documents(sha256);

        CREATE TABLE IF NOT EXISTS pyq_questions (
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            question_number INTEGER,
            question_text TEXT NOT NULL,
            marks REAL,
            grade INTEGER,
            subject TEXT,
            year INTEGER,
            board TEXT DEFAULT 'KSEAB',
            created_at TEXT DEFAULT (datetime('now')),
            figure_base64 TEXT,
            figure_ext TEXT,
            FOREIGN KEY(paper_id) REFERENCES pyq_papers(id)
        );

        CREATE TABLE IF NOT EXISTS pyq_document_events (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES pyq_documents(id)
        );
        CREATE INDEX IF NOT EXISTS idx_pyq_document_events_document ON pyq_document_events(document_id, created_at);

        CREATE TABLE IF NOT EXISTS pyq_source_visits (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            source_url TEXT NOT NULL,
            status TEXT NOT NULL,
            http_status INTEGER,
            detail TEXT,
            visited_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pyq_source_visits_run ON pyq_source_visits(run_id);

        CREATE TABLE IF NOT EXISTS pyq_sync_runs (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            report_json TEXT
        );

        CREATE TABLE IF NOT EXISTS syllabus_sources (
            id TEXT PRIMARY KEY,
            grade INTEGER NOT NULL,
            subject TEXT NOT NULL,
            source_url TEXT NOT NULL,
            title TEXT NOT NULL,
            verified_at TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_syllabus_sources_subject ON syllabus_sources(grade, subject);

        CREATE TABLE IF NOT EXISTS generated_papers (
            id TEXT PRIMARY KEY,
            grade INTEGER NOT NULL,
            subject TEXT NOT NULL,
            total_marks INTEGER NOT NULL,
            provider TEXT NOT NULL,
            request_json TEXT NOT NULL,
            paper_json TEXT NOT NULL,
            source_question_ids_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    for column in (
        "questions_json TEXT",
        "status TEXT",
        "resolved_url TEXT",
        "source_page TEXT",
        "extraction_quality REAL",
        "review_status TEXT",
    ):
        ensure_column(conn, "pyq_papers", column)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pyq_papers_catalog_id ON pyq_papers(catalog_id) WHERE catalog_id IS NOT NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pyq_questions_lookup ON pyq_questions(grade, subject, year)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pyq_questions_unique_source ON pyq_questions(paper_id, question_number, question_text)")
    rebase_project_file_paths(conn)
    quarantine_unregistered_invalid_payloads(conn)
    reconcile_catalog_metadata(conn)
    backfill_legacy_papers(conn)
    quarantine_unverified_archive_discoveries(conn)
    conn.commit()


def _event(conn: sqlite3.Connection, document_id: str, event_type: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO pyq_document_events (id, document_id, event_type, detail, created_at) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), document_id, event_type, detail[:4000], utcnow()),
    )


def _set_status(conn: sqlite3.Connection, document_id: str, status: str, reason: str = "", **values: Any) -> None:
    columns = {"status": status, "status_reason": reason, "updated_at": utcnow(), **values}
    assignment = ", ".join(f"{name} = ?" for name in columns)
    conn.execute(f"UPDATE pyq_documents SET {assignment} WHERE id = ?", (*columns.values(), document_id))
    _event(conn, document_id, status, reason)


def _upsert_document(conn: sqlite3.Connection, source_url: str, source_page: str, source_name: str, metadata: Dict[str, Any], status: str = "DISCOVERED", reason: str = "") -> str:
    normalized = normalize_url(source_url)
    current = conn.execute("SELECT id, status, metadata_json FROM pyq_documents WHERE normalized_url = ?", (normalized,)).fetchone()
    now = utcnow()
    if current:
        old = json.loads(current[2] or "{}")
        old.update({key: value for key, value in metadata.items() if value not in (None, "", "Unknown")})
        conn.execute(
            "UPDATE pyq_documents SET source_url=?, source_page=COALESCE(?, source_page), source_name=COALESCE(?, source_name), metadata_json=?, last_seen_at=?, updated_at=? WHERE id=?",
            (source_url, source_page, source_name, json.dumps(old, ensure_ascii=False), now, now, current[0]),
        )
        return current[0]
    document_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO pyq_documents
        (id,catalog_id,source_url,normalized_url,source_page,source_name,status,status_reason,metadata_json,discovered_at,last_seen_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (document_id, metadata.get("catalog_id"), source_url, normalized, source_page, source_name, status, reason,
         json.dumps(metadata, ensure_ascii=False), now, now, now),
    )
    _event(conn, document_id, status, reason or "Candidate recorded")
    return document_id


def backfill_legacy_papers(conn: sqlite3.Connection) -> None:
    """Give already-imported papers an auditable document row without changing them."""
    cursor = conn.execute("SELECT * FROM pyq_papers")
    columns = [description[0] for description in cursor.description]
    for row in cursor.fetchall():
        item = dict(zip(columns, row))
        source_url = item.get("source_url") or f"existing://pyq_papers/{item['id']}"
        metadata = {
            "catalog_id": item.get("catalog_id"), "title": item.get("title"), "grade": item.get("grade"),
            "subject": item.get("subject"), "year": item.get("year"), "board": item.get("board"),
            "exam": item.get("exam"), "medium": item.get("medium"), "paper_type": item.get("paper_type"),
            "file_name": item.get("file_name"),
        }
        normalized = normalize_url(source_url)
        existing = conn.execute("SELECT id, paper_id FROM pyq_documents WHERE normalized_url = ?", (normalized,)).fetchone()
        question_count = conn.execute("SELECT COUNT(*) FROM pyq_questions WHERE paper_id = ?", (item["id"],)).fetchone()[0]
        status = "EXTRACTED_SUCCESSFULLY" if item.get("extracted_text") and question_count else "NEEDS_REVIEW"
        if existing:
            if existing[1]:
                continue
            conn.execute(
                """UPDATE pyq_documents SET paper_id=?, sha256=?, file_path=?, extracted_text=?, questions_count=?,
                   extraction_quality=?, resolved_url=?, status=?, status_reason=?, updated_at=? WHERE id=?""",
                (item["id"], item.get("file_hash"), item.get("file_path"), item.get("extracted_text"), question_count,
                 item.get("extraction_quality"), item.get("resolved_url"), status,
                 "Linked to an existing library paper during non-destructive backfill.", utcnow(), existing[0]),
            )
            continue
        document_id = _upsert_document(conn, source_url, item.get("source_page") or "", "Existing ExamInsight library", metadata, status,
                                       "Backfilled from the existing library; original paper and questions were preserved.")
        conn.execute(
            """UPDATE pyq_documents SET paper_id=?, sha256=?, file_path=?, extracted_text=?, questions_count=?,
               extraction_quality=?, resolved_url=?, updated_at=? WHERE id=?""",
            (item["id"], item.get("file_hash"), item.get("file_path"), item.get("extracted_text"), question_count,
             item.get("extraction_quality"), item.get("resolved_url"), utcnow(), document_id),
        )


def reconcile_catalog_metadata(conn: sqlite3.Connection) -> None:
    """Restore the configured identity if broad page context had overwritten it."""
    if not DEFAULT_CATALOG.exists():
        return
    for item in _load_catalog(DEFAULT_CATALOG).get("papers", []):
        catalog_id = item.get("catalog_id")
        if not catalog_id:
            continue
        row = conn.execute("SELECT id, paper_id FROM pyq_documents WHERE catalog_id=?", (catalog_id,)).fetchone()
        if not row:
            continue
        conn.execute(
            "UPDATE pyq_documents SET metadata_json=?, source_url=?, source_page=?, updated_at=? WHERE id=?",
            (json.dumps(item, ensure_ascii=False), item["source_url"], item.get("source_page", ""), utcnow(), row[0]),
        )
        if row[1]:
            conn.execute(
                """UPDATE pyq_papers SET title=?, grade=?, subject=?, year=?, board=?, catalog_id=?, exam=?, medium=?, paper_type=?
                   WHERE id=?""",
                (item.get("title"), item.get("grade", 10), item.get("subject"), item.get("year"), item.get("board", "KSEAB"),
                 catalog_id, item.get("exam"), item.get("medium"), item.get("paper_type"), row[1]),
            )


def quarantine_unverified_archive_discoveries(conn: sqlite3.Connection) -> None:
    """Keep unlabelled archive links, but never misrepresent them as ready papers.

    The archive can contain dozens of Drive links under a broad page context.
    If a link has no configured catalogue identity, its inferred subject/year is
    not authoritative.  Preserve all bytes and extraction work, but require
    metadata review before it is listed or retrieved by the generator.
    """
    rows = conn.execute(
        """SELECT id, paper_id FROM pyq_documents
           WHERE catalog_id IS NULL AND source_name = 'Karnataka SSLC public question-paper archive'
             AND paper_id IS NOT NULL"""
    ).fetchall()
    for document_id, paper_id in rows:
        conn.execute(
            "UPDATE pyq_documents SET status='NEEDS_REVIEW', status_reason=?, updated_at=? WHERE id=?",
            ("Downloaded from archive but its individual subject/year metadata was not independently verified.", utcnow(), document_id),
        )
        conn.execute("UPDATE pyq_papers SET status='NEEDS_REVIEW', review_status='NEEDS_REVIEW' WHERE id=?", (paper_id,))
    conn.execute(
        """UPDATE pyq_papers SET status='READY', review_status='READY'
           WHERE id IN (SELECT paper_id FROM pyq_documents WHERE status='EXTRACTED_SUCCESSFULLY' AND paper_id IS NOT NULL)"""
    )


def _load_catalog(catalog_path: Path) -> Dict[str, Any]:
    with catalog_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def discover_documents(conn: sqlite3.Connection, run_id: str, catalog_path: Path = DEFAULT_CATALOG, max_pages: int = 100) -> Dict[str, Any]:
    """Reconcile static known records with every reachable configured archive page.

    ``max_pages`` is a crawl-safety boundary, not a paper limit; every document
    link on each visited page is persisted.
    """
    catalog = _load_catalog(catalog_path)
    static = catalog.get("papers", [])
    static_by_url = {normalize_url(item["source_url"]): item for item in static if item.get("source_url")}
    discovered_before = conn.execute("SELECT COUNT(*) FROM pyq_documents").fetchone()[0]
    for item in static:
        if item.get("source_url"):
            _upsert_document(conn, item["source_url"], item.get("source_page", ""), item.get("source_provider", "Configured catalogue"), item)

    inventory = catalog.get("source_inventory") or [
        {"name": "Karnataka SSLC public archive", "url": catalog.get("archive_url"), "kind": "archive"},
        {"name": "KSEAB official board", "url": catalog.get("official_board_url"), "kind": "official"},
    ]
    queue: deque[Tuple[str, int, str]] = deque()
    allowed_hosts = set()
    for source in inventory:
        url = source.get("url")
        if url:
            queue.append((url, 0, source.get("name", "Configured source")))
            allowed_hosts.add(urlparse(url).netloc.lower())
    visited = set()
    source_failures: List[Dict[str, str]] = []
    candidates = 0
    while queue and len(visited) < max_pages:
        page_url, depth, source_name = queue.popleft()
        canonical_page = normalize_url(page_url)
        if canonical_page in visited:
            continue
        visited.add(canonical_page)
        try:
            # A source inventory may contain a temporarily slow official site.
            # Record that failure and continue with the remaining sources rather
            # than allowing one page to hold the whole resumable run hostage.
            response = requests.get(page_url, headers=REQUEST_HEADERS, timeout=(5, 12), allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                raise ValueError(f"expected HTML page but received {content_type or 'unknown content type'}")
            conn.execute(
                "INSERT INTO pyq_source_visits (id,run_id,source_url,status,http_status,detail,visited_at) VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), run_id, page_url, "VISITED", response.status_code, response.url, utcnow()),
            )
            for link in extract_links(response.text, response.url):
                link_url = link["url"]
                if is_document_link(link_url):
                    fallback = static_by_url.get(normalize_url(link_url))
                    metadata = infer_metadata(link["context"], link_url, fallback)
                    candidate_status, reason = classify_candidate(link["context"])
                    if not fallback:
                        # It is a valid discovery lead, but its metadata must
                        # be independently confirmed before automated import.
                        candidate_status, reason = "NEEDS_REVIEW", "Discovered outside the configured inventory; verify subject/year metadata before import."
                    _upsert_document(conn, link_url, response.url, source_name, metadata, candidate_status, reason)
                    candidates += 1
                    continue
                parsed = urlparse(link_url)
                label = (link.get("text", "") + " " + link.get("context", "")).lower()
                if depth < 2 and parsed.netloc.lower() in allowed_hosts and any(token in label for token in ("next", "older", "page", "question paper", "archive")):
                    queue.append((link_url, depth + 1, source_name))
        except Exception as exc:
            source_failures.append({"url": page_url, "error": str(exc)})
            conn.execute(
                "INSERT INTO pyq_source_visits (id,run_id,source_url,status,detail,visited_at) VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), run_id, page_url, "FAILED", str(exc)[:2000], utcnow()),
            )
    conn.commit()
    return {
        "configured_catalogue_records": len(static),
        "pages_visited": len(visited),
        "document_links_seen": candidates,
        "new_candidate_records": conn.execute("SELECT COUNT(*) FROM pyq_documents").fetchone()[0] - discovered_before,
        "source_failures": source_failures,
        "crawl_safety_limit_reached": bool(queue),
    }


def _document_path(metadata: Dict[str, Any], document_id: str) -> Path:
    grade = int(metadata.get("grade") or 10)
    subject = safe_component(str(metadata.get("subject") or "unknown").lower())
    year = safe_component(str(metadata.get("year") or "unknown"))
    base = PROJECT_ROOT / "pyq" / f"class{grade}" / subject / year
    base.mkdir(parents=True, exist_ok=True)
    source_name = metadata.get("file_name") or f"{document_id}.pdf"
    filename = safe_component(Path(str(source_name)).name, f"{document_id}.pdf")
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return base / filename


def _preserve_invalid_payload(temp_path: Path, destination: Path) -> str:
    pyq_root = (PROJECT_ROOT / "pyq").resolve()
    relative_path = destination.resolve().relative_to(pyq_root)
    invalid_path = pyq_root / "_quarantine" / relative_path
    invalid_path = invalid_path.with_suffix(invalid_path.suffix + ".invalid")
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    if invalid_path.exists():
        invalid_path = invalid_path.with_name(invalid_path.stem + f".{uuid.uuid4().hex[:8]}" + invalid_path.suffix)
    temp_path.replace(invalid_path)
    return str(invalid_path)


def _download(conn: sqlite3.Connection, row: sqlite3.Row, max_retries: int = 3, max_bytes: int = 50 * 1024 * 1024) -> Optional[Path]:
    metadata = json.loads(row["metadata_json"] or "{}")
    destination = _document_path(metadata, row["id"])
    if destination.exists():
        valid, reason, pages = validate_pdf(destination)
        if valid:
            _set_status(conn, row["id"], "DOWNLOADED", "Existing validated file reused", file_path=_stored_project_path(destination), byte_size=destination.stat().st_size, page_count=pages)
            return destination
        logger.warning("Existing candidate file is invalid and will be retained for audit: %s (%s)", destination, reason)
    last_error = ""
    for attempt in range(1, max_retries + 1):
        temp_path = destination.with_suffix(destination.suffix + ".part")
        try:
            _set_status(conn, row["id"], "QUEUED", "Downloading", attempts=row["attempts"] + attempt, last_attempt_at=utcnow())
            # Discovery pages often expose Google Drive's /file/.../view link.
            # The persisted normalized URL deliberately converts that shape to
            # Drive's documented download endpoint; fetch it rather than the
            # viewer HTML page.
            download_url = row["normalized_url"] or row["source_url"]
            with requests.get(download_url, headers=REQUEST_HEADERS, timeout=(10, 60), allow_redirects=True, stream=True) as response:
                status_code = response.status_code
                response.raise_for_status()
                size = 0
                with temp_path.open("wb") as stream:
                    for chunk in response.iter_content(128 * 1024):
                        if chunk:
                            size += len(chunk)
                            if size > max_bytes:
                                raise ValueError(f"document exceeds configured maximum of {max_bytes} bytes")
                            stream.write(chunk)
                valid, reason, pages = validate_pdf(temp_path)
                if not valid:
                    retained = _preserve_invalid_payload(temp_path, destination)
                    _set_status(conn, row["id"], "PDF_VALIDATION_FAILED", reason, http_status=status_code,
                                content_type=response.headers.get("content-type", ""), byte_size=size, file_path=_stored_project_path(Path(retained)),
                                resolved_url=response.url, page_count=pages)
                    conn.commit()
                    return None
                temp_path.replace(destination)
                _set_status(conn, row["id"], "DOWNLOADED", "PDF signature and integrity verified", http_status=status_code,
                            content_type=response.headers.get("content-type", ""), byte_size=size, file_path=_stored_project_path(destination),
                            resolved_url=response.url, page_count=pages)
                conn.commit()
                return destination
        except Exception as exc:
            last_error = str(exc)
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))
    _set_status(conn, row["id"], "DOWNLOAD_FAILED", last_error, last_attempt_at=utcnow())
    conn.commit()
    return None


def _ocr_pdf(path: Path, language_hint: str) -> Tuple[Optional[str], str]:
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        return None, f"OCR dependency unavailable: {exc}"
    try:
        languages = set(pytesseract.get_languages(config=""))
        wanted = "kan+eng" if "kannada" in language_hint.lower() and "kan" in languages else "eng"
        if "eng" not in languages:
            return None, "Tesseract English language data is not installed"
        document = fitz.open(path)
        pages = []
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            pages.append(pytesseract.image_to_string(image, lang=wanted).strip())
        document.close()
        text = "\n\n--- PAGE BREAK ---\n\n".join(part for part in pages if part)
        return (text or None), ""
    except Exception as exc:
        return None, f"OCR failed: {exc}"


def extract_questions(text: str) -> List[Dict[str, Any]]:
    """Conservative numbered-question parser retaining long question text and page references."""
    questions: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    page = 1
    for original_line in text.splitlines():
        line = original_line.strip()
        if "PAGE BREAK" in line:
            page += 1
            continue
        match = re.match(r"^(?:Q(?:uestion)?\s*)?(\d{1,3})\s*[.)-]\s*(.+)$", line, re.I)
        if match:
            if current:
                questions.append(current)
            current = {"num": int(match.group(1)), "text": match.group(2), "page": page}
        elif current and line and not re.match(r"^(?:SECTION|PART)\b", line, re.I):
            current["text"] += " " + line
    if current:
        questions.append(current)
    cleaned = []
    seen = set()
    for item in questions:
        content = re.sub(r"\s+", " ", item["text"]).strip()
        fingerprint = re.sub(r"\W+", "", content.lower())[:120]
        if len(content) < 15 or fingerprint in seen or "turn over" in content.lower():
            continue
        seen.add(fingerprint)
        marks_match = re.search(r"[\[(]\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)\s*[\])]|\(\s*(\d+(?:\.\d+)?)\s*\)", content, re.I)
        mark = next((group for group in marks_match.groups() if group), None) if marks_match else None
        cleaned.append({"num": item["num"], "text": content[:5000], "page": item["page"], "marks": float(mark) if mark else None})
    return cleaned


def _text_quality(text: str, questions: Sequence[Dict[str, Any]]) -> float:
    if not text:
        return 0.0
    printable = sum(char.isprintable() for char in text) / max(1, len(text))
    return round(min(1.0, (min(len(text), 5000) / 5000) * 0.55 + min(len(questions), 20) / 20 * 0.35 + printable * 0.10), 3)


def _register_paper(conn: sqlite3.Connection, row: sqlite3.Row, path: Path, text: str, questions: List[Dict[str, Any]], quality: float) -> Tuple[str, bool]:
    digest = sha256_file(path)
    duplicate = conn.execute("SELECT id FROM pyq_papers WHERE file_hash = ?", (digest,)).fetchone()
    if duplicate:
        _set_status(conn, row["id"], "DUPLICATE_OF_ANOTHER_DOCUMENT", "Identical SHA-256 document already registered", sha256=digest,
                    paper_id=duplicate[0], duplicate_of=duplicate[0], questions_count=0, extraction_quality=quality)
        return duplicate[0], True
    metadata = json.loads(row["metadata_json"] or "{}")
    paper_id = f"kseab_{uuid.uuid4().hex}"
    title = metadata.get("title") or f"KSEAB SSLC {metadata.get('subject', 'Unknown')} {metadata.get('year', '')}"
    conn.execute(
        """INSERT INTO pyq_papers
        (id,title,grade,subject,year,board,file_name,file_path,file_type,uploaded_by,source_url,file_hash,extracted_text,catalog_id,exam,medium,paper_type,questions_json,status,resolved_url,source_page,extraction_quality,review_status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (paper_id, title, int(metadata.get("grade") or 10), metadata.get("subject", "Unknown"), metadata.get("year"),
         metadata.get("board", "KSEAB"), path.name, _stored_project_path(path), "application/pdf", "PYQ collector", row["source_url"], digest,
         text, metadata.get("catalog_id"), metadata.get("exam"), metadata.get("medium"), metadata.get("paper_type"),
         json.dumps(questions, ensure_ascii=False), "READY" if questions else "NEEDS_REVIEW", row["resolved_url"], row["source_page"], quality,
         "READY" if questions else "NEEDS_REVIEW"),
    )
    for question in questions:
        conn.execute(
            """INSERT OR IGNORE INTO pyq_questions
            (id,paper_id,question_number,question_text,marks,grade,subject,year,board)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), paper_id, question["num"], question["text"], question["marks"], int(metadata.get("grade") or 10),
             metadata.get("subject", "Unknown"), metadata.get("year"), metadata.get("board", "KSEAB")),
        )
    _set_status(conn, row["id"], "EXTRACTED_SUCCESSFULLY" if questions else "NEEDS_REVIEW",
                "Text and numbered questions extracted" if questions else "Text extracted but no reliable numbered questions found",
                sha256=digest, paper_id=paper_id, extracted_text=text, extraction_quality=quality, questions_count=len(questions))
    return paper_id, False


def process_document(conn: sqlite3.Connection, document_id: str) -> str:
    row = conn.execute("SELECT * FROM pyq_documents WHERE id = ?", (document_id,)).fetchone()
    if not row:
        raise KeyError(document_id)
    if row["status"] == "CONFIRMED_IRRELEVANT":
        return row["status"]
    path = Path(row["file_path"]) if row["file_path"] else None
    if path and not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path or not path.exists() or not validate_pdf(path)[0]:
        path = _download(conn, row)
    if not path:
        return conn.execute("SELECT status FROM pyq_documents WHERE id=?", (document_id,)).fetchone()[0]
    _set_status(conn, document_id, "TEXT_EXTRACTION_PENDING", "Extracting digital PDF text")
    text = extract_text_from_pdf(str(path)) or ""
    questions = extract_questions(text)
    metadata = json.loads(row["metadata_json"] or "{}")
    if len(text.strip()) < 80 or not questions:
        _set_status(conn, document_id, "OCR_PENDING", "Digital text absent or insufficient; attempting OCR")
        ocr_text, ocr_error = _ocr_pdf(path, str(metadata.get("medium", "")))
        if ocr_text and len(ocr_text) > len(text):
            text = ocr_text
            questions = extract_questions(text)
        elif not text and ocr_error:
            _set_status(conn, document_id, "OCR_FAILED", ocr_error)
            conn.commit()
            return "OCR_FAILED"
    quality = _text_quality(text, questions)
    current = conn.execute("SELECT * FROM pyq_documents WHERE id = ?", (document_id,)).fetchone()
    _register_paper(conn, current, path, text, questions, quality)
    conn.commit()
    return conn.execute("SELECT status FROM pyq_documents WHERE id=?", (document_id,)).fetchone()[0]


def run_sync(conn: sqlite3.Connection, catalog_path: Path = DEFAULT_CATALOG, retry_failed: bool = False, max_pages: int = 100) -> Dict[str, Any]:
    initialize_schema(conn)
    run_id = str(uuid.uuid4())
    conn.execute("INSERT INTO pyq_sync_runs (id,started_at,status) VALUES (?,?,?)", (run_id, utcnow(), "RUNNING"))
    conn.commit()
    discovery = discover_documents(conn, run_id, catalog_path=catalog_path, max_pages=max_pages)
    status_filter = "status IN ('DISCOVERED','QUEUED','TEXT_EXTRACTION_PENDING','OCR_PENDING')"
    if retry_failed:
        status_filter = "status IN ('DISCOVERED','QUEUED','TEXT_EXTRACTION_PENDING','OCR_PENDING','DOWNLOAD_FAILED','PDF_VALIDATION_FAILED','OCR_FAILED')"
    rows = conn.execute(f"SELECT id FROM pyq_documents WHERE {status_filter} ORDER BY discovered_at, id").fetchall()
    outcomes = Counter()
    for row in rows:
        try:
            outcomes[process_document(conn, row[0])] += 1
        except Exception as exc:
            logger.exception("Unhandled processing failure for %s", row[0])
            _set_status(conn, row[0], "NEEDS_REVIEW", f"Unhandled processing error: {exc}")
            conn.commit()
            outcomes["NEEDS_REVIEW"] += 1
    report = build_report(conn, run_id, discovery, outcomes)
    conn.execute("UPDATE pyq_sync_runs SET finished_at=?, status=?, report_json=? WHERE id=?", (utcnow(), report["final_status"], json.dumps(report, ensure_ascii=False), run_id))
    conn.commit()
    return report


def build_report(conn: sqlite3.Connection, run_id: Optional[str] = None, discovery: Optional[Dict[str, Any]] = None, outcomes: Optional[Counter] = None) -> Dict[str, Any]:
    status_rows = conn.execute("SELECT status, COUNT(*) AS count FROM pyq_documents GROUP BY status ORDER BY status").fetchall()
    subject_year = conn.execute(
        """SELECT json_extract(metadata_json, '$.subject') AS subject, json_extract(metadata_json, '$.year') AS year,
                  json_extract(metadata_json, '$.exam') AS exam, json_extract(metadata_json, '$.medium') AS medium, status, COUNT(*) AS count
           FROM pyq_documents GROUP BY subject, year, exam, medium, status ORDER BY year DESC, subject"""
    ).fetchall()
    question_count = conn.execute("SELECT COUNT(*) FROM pyq_questions").fetchone()[0]
    paper_count = conn.execute("SELECT COUNT(*) FROM pyq_papers").fetchone()[0]
    unresolved = conn.execute("SELECT COUNT(*) FROM pyq_documents WHERE status NOT IN ('EXTRACTED_SUCCESSFULLY','DUPLICATE_OF_ANOTHER_DOCUMENT','CONFIRMED_IRRELEVANT')").fetchone()[0]
    completed = conn.execute("SELECT COUNT(*) FROM pyq_documents WHERE status IN ('EXTRACTED_SUCCESSFULLY','DUPLICATE_OF_ANOTHER_DOCUMENT')").fetchone()[0]
    final_status = "VERIFIED COMPLETE" if completed and unresolved == 0 and not (discovery or {}).get("crawl_safety_limit_reached") else "PARTIALLY COMPLETE"
    if not completed and unresolved:
        final_status = "INCOMPLETE"
    return {
        "run_id": run_id,
        "generated_at": utcnow(),
        "final_status": final_status,
        "counting_rule": "Distinct paper count is SHA-256-distinct registered documents; multiple links to identical bytes count once.",
        "discovery": discovery or {},
        "run_outcomes": dict(outcomes or {}),
        "documents_by_status": {row[0]: row[1] for row in status_rows},
        "distinct_registered_papers": paper_count,
        "stored_questions": question_count,
        "unresolved_documents": unresolved,
        "breakdown": [dict(row) for row in subject_year],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover, validate, extract and register KSEAB SSLC PYQs.")
    parser.add_argument("command", choices=("sync", "report"), nargs="?", default="sync")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "examinsight.db"))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        initialize_schema(connection)
        report = run_sync(connection, Path(args.catalog), args.retry_failed, args.max_pages) if args.command == "sync" else build_report(connection)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
