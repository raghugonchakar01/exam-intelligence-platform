"""Strict source-backed question-paper generation.

The old generator labelled hard-coded mathematics prompts as a PYQ fallback.
This module never does that: it either returns selected stored PYQs, asks a
configured AI provider to create a new practice paper from bounded verified
context, or reports exactly why generation is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.qp_engine import build_paper_blueprint, classify_question_topic, is_noise_question

logger = logging.getLogger(__name__)
DB_PATH = os.getenv("EXAMINSIGHT_DB_PATH", "examinsight.db")


def _safe_subject(subject: Optional[str]) -> str:
    if subject is None:
        return ""
    return str(subject).split("(", 1)[0].strip()


class GenerationUnavailable(ValueError):
    """Raised when verified context or an AI provider needed for the request is absent."""


def _normalise(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())[:500]


def retrieve_questions(subject: Optional[str], grade: int, chapters: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return deduplicated source questions, prioritising requested chapters without inventing a result."""
    subject_key = _safe_subject(subject)
    if not subject_key:
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT q.id, q.paper_id, q.question_number, q.question_text, q.marks, q.grade, q.subject, q.year, q.board
               FROM pyq_questions q JOIN pyq_papers p ON p.id = q.paper_id
               WHERE q.grade = ? AND lower(q.subject) LIKE lower(?) AND COALESCE(p.status, 'READY') = 'READY'
               ORDER BY q.year DESC, q.question_number ASC""",
            (grade, f"%{subject_key}%"),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    requested = " ".join(chapters or ()).lower()
    result: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        question = dict(row)
        text = (question.get("question_text") or "").strip()
        fingerprint = _normalise(text)
        if len(text) < 15 or is_noise_question(text) or fingerprint in seen:
            continue
        seen.add(fingerprint)
        question["topic"] = classify_question_topic(text, question["subject"])
        question["_chapter_score"] = sum(token in (text + " " + question["topic"]).lower() for token in requested.split() if len(token) > 2)
        result.append(question)
    result.sort(key=lambda item: (item["_chapter_score"], item.get("year") or 0), reverse=True)
    return result[:limit] if limit is not None else result


def retrieve_verified_syllabus(subject: Optional[str], grade: int, chapters: Optional[Sequence[str]] = None, limit_chars: int = 3000) -> Tuple[str, List[str]]:
    subject_key = _safe_subject(subject)
    if not subject_key:
        return "", []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, title, content FROM syllabus_sources
               WHERE grade = ? AND lower(subject) LIKE lower(?) ORDER BY verified_at DESC""",
            (grade, f"%{subject_key}%"),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    needles = [item.lower() for item in (chapters or ()) if item.strip()]
    pieces, ids = [], []
    remaining = limit_chars
    for row in rows:
        content = row["content"].strip()
        if needles and not any(term in content.lower() for term in needles):
            continue
        if remaining <= 0:
            break
        pieces.append(f"[{row['title']}] {content[:remaining]}")
        ids.append(row["id"])
        remaining -= len(pieces[-1])
    return "\n".join(pieces), ids


def _safe_blueprint(total_marks: int, subject: Optional[str], board: str, pattern: Optional[str]) -> List[Dict[str, Any]]:
    if not isinstance(total_marks, int) or total_marks < 1 or total_marks > 200:
        raise GenerationUnavailable("total_marks must be an integer between 1 and 200.")
    blueprint = build_paper_blueprint(total_marks, subject or "Mathematics", board, pattern or "Mixed")
    if sum(item["total_marks"] for item in blueprint) != total_marks:
        # The legacy board-pattern helper uses minimum section sizes.  For tiny
        # custom papers, a transparent objective-only structure is the only
        # truthful way to keep the requested marks exact.
        return [{"section": "Section A", "question_type": "Objective Questions", "marks_per_q": 1,
                 "count": total_marks, "total_marks": total_marks}]
    return blueprint


def _strict_validate(paper: Dict[str, Any], expected_marks: int) -> List[str]:
    errors: List[str] = []
    all_text: List[str] = []
    total = 0.0
    sections = paper.get("sections")
    if not isinstance(sections, list) or not sections:
        return ["Paper has no sections."]
    for section in sections:
        questions = section.get("questions")
        if not isinstance(questions, list) or not questions:
            errors.append(f"{section.get('name', 'A section')} has no questions.")
            continue
        for question in questions:
            text = str(question.get("question", "")).strip()
            marks = question.get("marks")
            if len(text) < 15:
                errors.append("Question text is blank or too short.")
                continue
            if not isinstance(marks, (int, float)) or marks <= 0:
                errors.append("Question has invalid marks.")
                continue
            # A source collection can legitimately contain questions with the
            # same command verb.  Only reject substantially identical wording;
            # topic-level similarity is not a duplicate.
            if any(SequenceMatcher(None, text.lower(), earlier.lower()).ratio() > 0.96 for earlier in all_text):
                errors.append("Paper contains duplicate or near-duplicate questions.")
            all_text.append(text)
            total += marks
    if total != expected_marks:
        errors.append(f"Total marks are {total:g}; expected exactly {expected_marks:g}.")
    return errors


def _select_stored_questions(questions: Sequence[Dict[str, Any]], blueprint: Sequence[Dict[str, Any]], subject: str, grade: int, board: str, difficulty: str) -> Dict[str, Any]:
    required = sum(int(section["count"]) for section in blueprint)
    if len(questions) < required:
        raise GenerationUnavailable(
            f"Only {len(questions)} distinct stored PYQ questions are available for {subject} Grade {grade}; "
            f"{required} are needed for this requested paper structure. Add more verified PYQs or configure an AI provider with verified syllabus context."
        )
    iterator = iter(questions)
    source_ids: List[str] = []
    sections = []
    for section in blueprint:
        generated = []
        for _ in range(int(section["count"])):
            source = next(iterator)
            source_ids.append(source["id"])
            generated.append({
                "question": source["question_text"],
                "marks": section["marks_per_q"],
                "source_marks": source.get("marks"),
                "topic": source["topic"],
                "difficulty": difficulty,
                "is_pyq": True,
                "source_question_id": source["id"],
                "source": f"Retrieved {source.get('board', 'KSEAB')} PYQ — {source.get('year', 'unknown year')}, Q{source.get('question_number', '?')}",
            })
        sections.append({
            "name": section["section"],
            "question_type": section["question_type"],
            "marks_per_question": section["marks_per_q"],
            "questions": generated,
        })
    paper = {
        "title": f"Retrieved PYQ Practice Paper — {subject} Grade {grade}",
        "subject": subject,
        "grade": str(grade),
        "board": board,
        "total_marks": sum(section["total_marks"] for section in blueprint),
        "duration_minutes": sum(section["total_marks"] for section in blueprint) * 2,
        "difficulty": difficulty,
        "generation_mode": "Retrieved previous-year questions",
        "provider": "ExamInsight source-backed assembler",
        "sections": sections,
        "_source_question_ids": source_ids,
        "_syllabus_source_ids": [],
        "_generation_source": "Retrieved PYQ Repository",
    }
    errors = _strict_validate(paper, paper["total_marks"])
    if errors:
        raise GenerationUnavailable("Could not assemble a valid source-backed paper: " + " ".join(errors))
    return paper


def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192}}).encode("utf-8")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        parts = data.get("candidates", [])[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "").strip() if parts else None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, IndexError) as exc:
        logger.warning("Gemini generation failed: %s", exc)
        return None


def _strip_json_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1]
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
    return value.strip()


def _ai_paper(
    api_key: str,
    grade: int,
    subject: str,
    board: str,
    total_marks: int,
    difficulty: str,
    question_pattern: str,
    chapters: Sequence[str],
    question_types: Sequence[str],
    blueprint: Sequence[Dict[str, Any]],
    pyqs: Sequence[Dict[str, Any]],
    syllabus_text: str,
    syllabus_ids: Sequence[str],
) -> Optional[Dict[str, Any]]:
    evidence = "\n".join(
        f"[PYQ id={item['id']} {item.get('year', 'unknown')}] {item['question_text']}" for item in pyqs[:25]
    )
    blueprint_json = [{"name": item["section"], "question_type": item["question_type"], "marks_per_question": item["marks_per_q"], "num_questions": item["count"]} for item in blueprint]
    prompt = f"""Create an original Class {grade} {subject} practice paper, not an official examination.
Board: {board}
Difficulty: {difficulty}
Requested chapters: {', '.join(chapters) or 'not specified'}
Requested question types: {', '.join(question_types) or question_pattern}
Total marks: {total_marks}

Use only the verified syllabus and the style/topics in the retrieved PYQs below. Do not claim invented questions are past papers. Do not reproduce a PYQ verbatim.

VERIFIED SYLLABUS:
{syllabus_text[:3000] or 'No verified syllabus record is available.'}

RETRIEVED PYQS:
{evidence[:6000] or 'No retrieved PYQs are available.'}

Return JSON only. It must contain sections matching this exact blueprint and every question must have question, marks, topic, difficulty, is_pyq=false.
Blueprint: {json.dumps(blueprint_json)}
"""
    raw = _call_gemini(prompt, api_key)
    if not raw:
        return None
    try:
        paper = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError:
        return None
    paper.update({
        "title": paper.get("title") or f"AI Practice Question Paper — {subject} Grade {grade}",
        "subject": subject,
        "grade": str(grade),
        "board": board,
        "total_marks": total_marks,
        "difficulty": difficulty,
        "duration_minutes": paper.get("duration_minutes") or total_marks * 2,
        "generation_mode": "AI-generated practice paper using verified context",
        "provider": "Google Gemini",
        "_source_question_ids": [item["id"] for item in pyqs],
        "_syllabus_source_ids": list(syllabus_ids),
        "_generation_source": "Google Gemini with verified PYQ/syllabus context",
    })
    for section in paper.get("sections", []):
        for question in section.get("questions", []):
            question["is_pyq"] = False
            question.pop("source", None)
    return paper if not _strict_validate(paper, total_marks) else None


def generate_question_paper(
    *, grade: int, subject: str, board: str, total_marks: int, difficulty: str,
    question_pattern: str, chapters: Optional[Sequence[str]] = None,
    question_types: Optional[Sequence[str]] = None, gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a validated paper or raise an actionable source/configuration error."""
    blueprint = _safe_blueprint(total_marks, subject, board, question_pattern)
    pyqs = retrieve_questions(subject, grade, chapters)
    syllabus_text, syllabus_ids = retrieve_verified_syllabus(subject, grade, chapters)
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if api_key and len(api_key.strip()) > 10:
        generated = _ai_paper(api_key.strip(), grade, subject, board, total_marks, difficulty, question_pattern,
                              chapters or (), question_types or (), blueprint, pyqs, syllabus_text, syllabus_ids)
        if generated:
            return generated
        if not pyqs:
            raise GenerationUnavailable("The configured AI provider did not return a valid paper and there are no stored PYQs for a source-backed fallback.")
    if not pyqs:
        raise GenerationUnavailable(
            f"No stored PYQ questions or verified syllabus context are available for {subject} Grade {grade}. "
            "Sync and review source papers first, or configure an AI provider after adding verified syllabus information."
        )
    return _select_stored_questions(pyqs, blueprint, subject, grade, board, difficulty)
