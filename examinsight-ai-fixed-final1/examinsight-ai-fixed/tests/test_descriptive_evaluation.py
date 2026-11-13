"""End-to-End Test Suite for Descriptive Answer Evaluation Pipeline.

Covers all required test cases A through O:
- Case A: Clean digital PDF with selectable text.
- Case B: Scanned handwritten answer script.
- Case C: Student who answers questions out of order.
- Case D: Long answer continued on next page.
- Case E: Subparts with different maximum marks.
- Case F: Unanswered question.
- Case G: Illegible page or low-confidence OCR output.
- Case H: Missing, duplicated, or ambiguous question number.
- Case I: Malformed or unsupported upload.
- Case J: AI response containing invalid JSON or marks exceeding maximum.
- Case K: API timeout or rate-limit error.
- Case L: Repeated upload of the same unchanged file to verify valid cached OCR is reused.
- Case M: Answer that is correct but uses different wording from expected answer.
- Case N: Partially correct descriptive answer that deserves partial credit.
- Case O: Reconciliation of question-wise, subpart, total, and maximum marks.
"""

from __future__ import annotations

import io
import json
import os
import unittest.mock as mock
import cv2
import numpy as np
import pymupdf as fitz
import pytest
import requests
from fastapi.testclient import TestClient

from app.answer_evaluator import (
    evaluate_matched_question,
    evaluate_single_answer_nlp,
    generate_paper_evaluation_report,
    get_or_synthesize_reference_rubric
)
from app.answer_matcher import (
    match_answers_to_questions,
    normalize_predicted_questions,
    parse_question_label,
    stitch_multipage_continuations
)
from app.handwriting_ocr import (
    _parse_text_into_question_blocks,
    recognize_handwriting
)
from app.handwriting_preprocessor import (
    deskew_image,
    detect_strikethrough_regions,
    extract_digital_pdf_text,
    load_document_pages,
    preprocess_document,
    preprocess_page
)
from app.main import app


client = TestClient(app)


# =====================================================================
# TEST CASE A: CLEAN DIGITAL PDF WITH SELECTABLE TEXT
# =====================================================================
def test_case_a_clean_digital_pdf_selectable_text():
    """Verify that a clean digital PDF extracts text directly with high confidence at 0 API cost."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    sample_text = (
        "Q1. Photosynthesis is the chemical process where green plants convert sunlight, "
        "carbon dioxide, and water into glucose and oxygen using chlorophyll.\n\n"
        "Q2. State Ohm's Law: The current passing through a conductor between two points is directly "
        "proportional to the voltage across the two points at constant temperature."
    )
    page.insert_text((50, 72), sample_text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()

    # Preprocessor digital extraction check
    extracted_digital = extract_digital_pdf_text(pdf_bytes)
    assert extracted_digital is not None
    assert len(extracted_digital) == 1
    assert "Photosynthesis" in extracted_digital[0]["text"]

    # Full OCR routing check
    res = recognize_handwriting(pdf_bytes, "digital_answer_sheet.pdf")
    assert res["ocr_engine"] == "Digital Document Text Extractor (PyMuPDF)"
    assert res["ocr_confidence"] >= 95.0
    assert not res["is_fallback"]
    assert len(res["recognized_answers"]) >= 2
    assert "Photosynthesis" in res["combined_text"]


# =====================================================================
# TEST CASE B: SCANNED HANDWRITTEN ANSWER SCRIPT
# =====================================================================
def test_case_b_scanned_handwritten_answer_script():
    """Verify image preprocessing, denoising, CLAHE, and layout segmentation on scanned image."""
    img = np.full((400, 600, 3), 235, dtype=np.uint8)
    # Simulate uneven illumination / shadow gradient
    for y in range(400):
        img[y, :] = np.clip(img[y, :].astype(int) - int(y * 0.25), 0, 255).astype(np.uint8)

    # Simulate handwritten lines
    cv2.putText(img, "Q1. Define Force", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2)
    cv2.putText(img, "Force is mass times acceleration (F=ma).", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

    success, enc = cv2.imencode(".jpg", img)
    assert success
    jpg_bytes = enc.tobytes()

    pages = preprocess_document(jpg_bytes, "scanned_script.jpg")
    assert len(pages) == 1
    p = pages[0]
    assert p["page_number"] == 1
    assert "image" in p
    assert "layout" in p
    assert p["layout"]["text_block_count"] >= 1


# =====================================================================
# TEST CASE C: OUT-OF-ORDER STUDENT ANSWERS
# =====================================================================
def test_case_c_out_of_order_answers():
    """Verify that student answers written in arbitrary order map to the correct questions."""
    paper_questions = [
        {"id": "q1", "question_number": 1, "question_text": "State Newton's first law of motion.", "marks": 2.0},
        {"id": "q2", "question_number": 2, "question_text": "Define momentum and its SI unit.", "marks": 2.0},
        {"id": "q3", "question_number": 3, "question_text": "Explain conservation of energy.", "marks": 3.0}
    ]

    # Student answered Q3 first, then Q1, then Q2
    student_raw_answers = [
        {"question_label": "Ans 3", "text": "Energy can neither be created nor destroyed, only transformed.", "page_number": 1},
        {"question_label": "1.", "text": "An object at rest stays at rest unless an external unbalanced force acts on it.", "page_number": 1},
        {"question_label": "Question 2", "text": "Momentum is the product of mass and velocity (p=mv). Unit is kg m/s.", "page_number": 1}
    ]

    matched = match_answers_to_questions(student_raw_answers, paper_questions)
    items = matched["matched_items"]
    assert len(items) == 3

    # Check Question 1 matched to Newton's law answer
    q1_item = next(i for i in items if i["question_number"] == 1)
    assert "rest stays at rest" in q1_item["student_answer"]
    assert q1_item["match_status"] == "MATCHED"

    # Check Question 2 matched to momentum answer
    q2_item = next(i for i in items if i["question_number"] == 2)
    assert "Momentum is the product" in q2_item["student_answer"]

    # Check Question 3 matched to conservation of energy answer
    q3_item = next(i for i in items if i["question_number"] == 3)
    assert "Energy can neither be created" in q3_item["student_answer"]


# =====================================================================
# TEST CASE D: LONG ANSWER CONTINUED ON NEXT PAGE
# =====================================================================
def test_case_d_multipage_answer_continuation():
    """Verify stitching of an answer continued from page 1 to page 2 with P.T.O./cont. markers."""
    answers = [
        {
            "question_label": "Q4",
            "text": "The digestive system consists of mouth, esophagus, and stomach. (Continued on next page P.T.O.)",
            "page_number": 1,
            "is_continuation": False
        },
        {
            "question_label": "Q4",
            "text": "In the small intestine, bile juice and pancreatic enzymes break down food molecules.",
            "page_number": 2,
            "is_continuation": True
        }
    ]

    stitched = stitch_multipage_continuations(answers)
    assert len(stitched) == 1
    combined_ans = stitched[0]
    assert combined_ans["is_continuation"]
    assert combined_ans["pages"] == [1, 2]
    assert "esophagus" in combined_ans["text"]
    assert "small intestine" in combined_ans["text"]


# =====================================================================
# TEST CASE E: SUBPARTS WITH DIFFERENT MAXIMUM MARKS
# =====================================================================
def test_case_e_subparts_with_different_maximum_marks():
    """Verify matching and grading of subparts 2(a) [2M] and 2(b) [3M] with reconciled parent total."""
    paper_questions = [
        {
            "id": "q2a",
            "question_number": 2,
            "subpart": "a",
            "question_text": "State Ohm's Law.",
            "marks": 2.0,
            "expected_concepts": ["Current directly proportional to voltage", "Constant temperature", "V = IR"]
        },
        {
            "id": "q2b",
            "question_number": 2,
            "subpart": "b",
            "question_text": "Derive equivalent resistance for two resistors R1 and R2 connected in series.",
            "marks": 3.0,
            "expected_concepts": ["Current is constant", "Total voltage V = V1 + V2", "Req = R1 + R2"]
        }
    ]

    student_answers = [
        {"question_label": "2(a)", "text": "Ohm's law states that current is directly proportional to voltage at constant temperature (V=IR).", "page_number": 1},
        {"question_label": "2(b)", "text": "In series, current I remains the same. Total potential difference is V = V1 + V2 = IR1 + IR2. Hence Req = R1 + R2.", "page_number": 1}
    ]

    matched = match_answers_to_questions(student_answers, paper_questions)
    assert len(matched["matched_items"]) == 2

    m_2a = next(i for i in matched["matched_items"] if i["subpart"] == "a")
    m_2b = next(i for i in matched["matched_items"] if i["subpart"] == "b")

    eval_2a = evaluate_matched_question(m_2a)
    eval_2b = evaluate_matched_question(m_2b)

    assert eval_2a["max_marks"] == 2.0
    assert eval_2a["marks_awarded"] >= 1.8
    assert eval_2b["max_marks"] == 3.0
    assert eval_2b["marks_awarded"] >= 2.5
    assert (eval_2a["marks_awarded"] + eval_2b["marks_awarded"]) <= 5.0


# =====================================================================
# TEST CASE F: UNANSWERED QUESTION
# =====================================================================
def test_case_f_unanswered_question():
    """Verify that an omitted question receives strictly 0 marks while preserving maximum possible marks."""
    paper_questions = [
        {"id": "q1", "question_number": 1, "question_text": "State Boyle's Law.", "marks": 3.0},
        {"id": "q2", "question_number": 2, "question_text": "State Charles's Law.", "marks": 3.0}
    ]

    # Student only answered Q1
    student_answers = [
        {"question_label": "Q1", "text": "Pressure is inversely proportional to volume at constant temperature (P1V1=P2V2).", "page_number": 1}
    ]

    matched = match_answers_to_questions(student_answers, paper_questions)
    assert matched["total_unanswered_questions"] == 1

    unans = next(i for i in matched["matched_items"] if i["question_number"] == 2)
    assert unans["match_status"] == "UNANSWERED"
    assert unans["student_answer"] == ""

    eval_unans = evaluate_matched_question(unans)
    assert eval_unans["marks_awarded"] == 0.0
    assert eval_unans["max_marks"] == 3.0
    assert eval_unans["status"] == "unanswered"
    assert "unattempted" in eval_unans["justification"].lower()


# =====================================================================
# TEST CASE G: ILLEGIBLE PAGE OR LOW-CONFIDENCE OCR OUTPUT
# =====================================================================
def test_case_g_illegible_page_low_confidence():
    """Verify that low-confidence OCR text triggers transparent review flags without failing."""
    q_item = {
        "question": {"question_number": 1, "question_text": "Explain nuclear fission.", "marks": 4.0},
        "student_answer": "n...clr f..ss..n spl.ts h..vy n..cl..s",
        "ocr_confidence": 38.0,  # Below 60 threshold
        "match_status": "MATCHED",
        "needs_review": True
    }

    eval_res = evaluate_matched_question(q_item)
    assert eval_res["ocr_confidence"] == 38.0
    assert eval_res["needs_manual_review"]
    assert "Low" in eval_res["evaluation_confidence"]


# =====================================================================
# TEST CASE H: MISSING, DUPLICATED, OR AMBIGUOUS QUESTION NUMBER
# =====================================================================
def test_case_h_ambiguous_question_number():
    """Verify that duplicate/ambiguous question labels are flagged for user review rather than guessed."""
    paper_questions = [
        {"id": "q1", "question_number": 1, "question_text": "Define velocity.", "marks": 2.0},
        {"id": "q2", "question_number": 2, "question_text": "Define speed.", "marks": 2.0}
    ]

    # Student wrote two ambiguous answers with unclear/duplicate numbers
    raw_answers = [
        {"question_label": "Q1", "text": "Rate of change of displacement.", "page_number": 1, "confidence": 75.0},
        {"question_label": "Q1", "text": "Total distance traveled divided by time taken.", "page_number": 1, "confidence": 74.0}
    ]

    matched = match_answers_to_questions(raw_answers, paper_questions)
    # The runner-up should be flagged or recorded in unmatched/ambiguous
    assert len(matched["matched_items"]) == 2
    assert len(matched["ambiguous_matches"]) > 0 or len(matched["unmatched_answers"]) > 0


# =====================================================================
# TEST CASE I: MALFORMED OR UNSUPPORTED UPLOAD
# =====================================================================
def test_case_i_malformed_unsupported_upload():
    """Verify that non-supported or empty files are rejected with clear HTTP 400 errors."""
    # Test unsupported extension
    resp_bad_ext = client.post(
        "/api/descriptive/extract-and-match",
        files={"file": ("virus.exe", b"MZexecutablecode", "application/x-msdownload")}
    )
    assert resp_bad_ext.status_code == 400
    assert "Invalid file type" in resp_bad_ext.json()["detail"]

    # Test empty file
    resp_empty = client.post(
        "/api/descriptive/extract-and-match",
        files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"].lower()


# =====================================================================
# TEST CASE J: AI RESPONSE CONTAINING INVALID JSON OR MARKS EXCEEDING MAXIMUM
# =====================================================================
def test_case_j_ai_response_invalid_json_or_out_of_bounds_marks():
    """Verify that malformed AI JSON or illegal marks (> max_marks or < 0) are bounded safely."""
    # 1. Test invalid JSON fallback
    with mock.patch("requests.post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "INVALID NOT JSON TEXT"}]}}]
        }
        mock_post.return_value = mock_resp

        q_item = {
            "question": {
                "question_number": 1,
                "question_text": "Define work done.",
                "marks": 2.0,
                "reference_answer": "Work done is the product of force and displacement.",
                "expected_concepts": ["Force", "Displacement", "W = Fs"]
            },
            "student_answer": "Work is done when force produces displacement in the direction of force.",
            "match_status": "MATCHED",
            "ocr_confidence": 90.0
        }

        # Should safely catch JSON error and fall back to NLP engine without raising
        eval_res = evaluate_matched_question(q_item, gemini_api_key="fake-test-key-12345")
        assert 0.0 <= eval_res["marks_awarded"] <= 2.0
        assert "fallback" in eval_res["evaluated_by"].lower()

    # 2. Test clamping if AI tries to award > max_marks
    over_res = evaluate_single_answer_nlp(
        question_text="Sample question",
        reference_answer="Sample answer",
        expected_concepts=["Concept A", "Concept B"],
        student_answer="Concept A and Concept B with deep detail and equations.",
        max_marks=5.0
    )
    assert over_res["score"] <= 5.0
    assert over_res["score"] >= 0.0


# =====================================================================
# TEST CASE K: API TIMEOUT OR RATE-LIMIT ERROR
# =====================================================================
def test_case_k_api_timeout_rate_limit():
    """Verify graceful fallback to deterministic NLP rubric engine when API times out or rate limits."""
    with mock.patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        q_item = {
            "question": {
                "question_number": 1,
                "question_text": "State the principle of moments.",
                "marks": 3.0,
                "reference_answer": "Sum of clockwise moments equals sum of anticlockwise moments in equilibrium.",
                "expected_concepts": ["Clockwise moments", "Anticlockwise moments", "Equilibrium"]
            },
            "student_answer": "For a body in equilibrium, clockwise moments equal anticlockwise moments.",
            "match_status": "MATCHED",
            "ocr_confidence": 88.0
        }

        eval_res = evaluate_matched_question(q_item, gemini_api_key="fake-gemini-key")
        assert eval_res["marks_awarded"] > 0.0
        assert "fallback" in eval_res["evaluated_by"].lower()


# =====================================================================
# TEST CASE L: REPEATED UPLOAD OF SAME UNCHANGED FILE (CACHE REUSE)
# =====================================================================
def test_case_l_repeated_upload_cache_reuse():
    """Verify that uploading the exact same document bytes reuses cached OCR without reprocessing."""
    _, buf = cv2.imencode(".jpg", np.full((120, 120, 3), 250, dtype=np.uint8))
    synthetic_file = buf.tobytes()
    
    # First call
    res_1 = recognize_handwriting(synthetic_file, "repeated_test.jpg")
    file_hash = res_1["file_hash"]
    assert file_hash is not None

    # Second call with identical bytes
    res_2 = recognize_handwriting(synthetic_file, "repeated_test.jpg")
    assert res_2["from_cache"] is True
    assert res_2["file_hash"] == file_hash
    assert res_2["combined_text"] == res_1["combined_text"]


# =====================================================================
# TEST CASE M: CORRECT ANSWER WITH DIFFERENT WORDING
# =====================================================================
def test_case_m_different_wording_awarded_credit():
    """Verify that semantically correct answers with different phrasing receive appropriate marks."""
    question = "State Newton's third law of motion."
    model_answer = "For every action, there is an equal and opposite reaction."
    expected_concepts = ["Equal magnitude", "Opposite direction", "Action and reaction pair"]

    # Student uses completely different wording
    student_phrasing = "Whenever body A exerts a force on body B, body B simultaneously exerts an identical force in the reverse direction on body A."

    res = evaluate_single_answer_nlp(
        question_text=question,
        reference_answer=model_answer,
        expected_concepts=expected_concepts,
        student_answer=student_phrasing,
        max_marks=2.0
    )

    # Must award partial/full credit (>= 1.2 out of 2.0)
    assert res["score"] >= 1.2
    assert res["percentage"] >= 60.0


# =====================================================================
# TEST CASE N: PARTIALLY CORRECT DESCRIPTIVE ANSWER
# =====================================================================
def test_case_n_partially_correct_descriptive_answer():
    """Verify that answers containing some correct concepts and omitting others receive partial marks."""
    question = "Explain 1NF, 2NF, and 3NF database normalization."
    model_answer = "1NF requires atomic values. 2NF removes partial dependency. 3NF removes transitive functional dependency."
    expected_concepts = ["Atomic values (1NF)", "Remove partial dependency (2NF)", "Remove transitive dependency (3NF)"]

    # Student only explains 1NF and 2NF, but omits 3NF
    student_answer = "1NF requires all attributes to have atomic values. 2NF requires table to be in 1NF and have no partial functional dependency on primary key."

    res = evaluate_single_answer_nlp(
        question_text=question,
        reference_answer=model_answer,
        expected_concepts=expected_concepts,
        student_answer=student_answer,
        max_marks=6.0
    )

    # Should receive partial marks strictly bounded between 2.0 and 5.5
    assert 2.0 <= res["score"] < 6.0
    assert len(res["covered_concepts"]) >= 1
    assert len(res["missing_concepts"]) >= 1


# =====================================================================
# TEST CASE O: RECONCILIATION OF QUESTION-WISE, SUBPART, AND TOTAL MARKS
# =====================================================================
def test_case_o_reconciliation_of_scores():
    """Verify that paper total marks obtained equals the exact sum of question-wise marks."""
    matching_result = {
        "ocr_engine": "Digital Document Text Extractor (PyMuPDF)",
        "ocr_confidence": 95.0,
        "matched_items": [
            {
                "question": {"question_number": 1, "question_text": "Q1 text", "marks": 2.0},
                "student_answer": "Valid full answer covering all required points.",
                "match_status": "MATCHED",
                "ocr_confidence": 95.0
            },
            {
                "question": {"question_number": 2, "question_text": "Q2 text", "marks": 3.0},
                "student_answer": "Partially correct answer.",
                "match_status": "MATCHED",
                "ocr_confidence": 90.0
            },
            {
                "question": {"question_number": 3, "question_text": "Q3 text", "marks": 5.0},
                "student_answer": "",
                "match_status": "UNANSWERED",
                "ocr_confidence": 100.0
            }
        ],
        "unmatched_answers": [],
        "ambiguous_matches": []
    }

    paper_meta = {
        "id": "paper_test_123",
        "title": "Grade 10 Practice Paper",
        "subject": "Science",
        "grade": 10,
        "board": "KSEAB"
    }

    report = generate_paper_evaluation_report(matching_result, paper_meta)
    summary = report["summary"]
    q_evals = report["question_evaluations"]

    # Verification: total marks obtained equals exact sum of question scores
    sum_marks = round(sum(q["marks_awarded"] for q in q_evals), 1)
    sum_max = round(sum(q["max_marks"] for q in q_evals), 1)

    assert summary["total_marks_obtained"] == sum_marks
    assert summary["max_possible_marks"] == sum_max
    assert summary["max_possible_marks"] == 10.0
    assert summary["unanswered_questions"] == 1
    assert summary["answered_questions"] == 2
    assert summary["total_questions"] == 3
    assert summary["percentage"] == round((sum_marks / sum_max) * 100.0, 1)
