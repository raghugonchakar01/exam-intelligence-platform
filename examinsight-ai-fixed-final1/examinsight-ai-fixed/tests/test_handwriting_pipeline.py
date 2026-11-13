import io
import json
import sqlite3
import numpy as np
import cv2
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.handwriting_preprocessor import (
    preprocess_page,
    preprocess_document,
    deskew_image,
    detect_strikethrough_regions,
    load_document_pages,
    segment_page_layout
)
from app.handwriting_ocr import (
    recognize_handwriting,
    parse_handwritten_answers,
    recognize_handwriting_local
)
from app.answer_matcher import (
    match_answers_to_questions,
    extract_question_label,
    calculate_content_similarity
)
from app.answer_evaluator import (
    evaluate_single_answer,
    generate_paper_evaluation_report
)
from app.main import app, get_db, DB_PATH


# ==========================================
# 1. PREPROCESSING TESTS
# ==========================================
class TestHandwritingPreprocessor:
    def test_preprocess_image_pipeline(self):
        # Create a synthetic image simulating uneven illumination & handwriting
        img = np.full((300, 400, 3), 220, dtype=np.uint8)
        # Add shadow gradient
        for y in range(300):
            img[y, :] = np.clip(img[y, :].astype(int) - int(y * 0.3), 0, 255).astype(np.uint8)
        # Add simulated text strokes
        cv2.putText(img, "Q1. State Newton's Third Law", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(img, "Fab = -Fba action and reaction", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

        result = preprocess_page(img, page_index=1)
        assert result is not None
        assert "image" in result
        assert "layout" in result
        assert "jpeg_bytes" in result
        assert result["image"].shape[:2] == (300, 400)

    def test_deskew_bounded(self):
        # Create image with text rotated by 8 degrees
        img = np.full((250, 400), 255, dtype=np.uint8)
        cv2.putText(img, "Sample Handwritten Paragraph", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        center = (200, 125)
        rot_mat = cv2.getRotationMatrix2D(center, 8.0, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, (400, 250), borderValue=255)

        deskewed, angle = deskew_image(rotated)
        assert abs(angle) <= 22.0
        assert deskewed.shape == rotated.shape

    def test_strikethrough_detection(self):
        # Create image with horizontal strikethrough line through text
        img = np.full((150, 300), 255, dtype=np.uint8)
        cv2.putText(img, "Incorrect answer here", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        # Add horizontal strikethrough line
        cv2.line(img, (15, 68), (280, 68), 0, 2)

        strikethroughs = detect_strikethrough_regions(img)
        assert isinstance(strikethroughs, list)
        assert len(strikethroughs) > 0

    def test_load_document_multipage_pdf(self):
        # Generate in-memory 2-page PDF
        doc = fitz.open()
        for i in range(2):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 72), f"Page {i + 1} of Student Answer Sheet", fontsize=16)
            page.insert_text((50, 120), f"Q{i + 1}. Answer text on page {i + 1}...", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        pages = load_document_pages(pdf_bytes, filename="multipage_test.pdf")
        assert len(pages) == 2
        assert all(isinstance(p, np.ndarray) for p in pages)


# ==========================================
# 2. HANDWRITING OCR & PARSING TESTS
# ==========================================
class TestHandwritingOCR:
    def test_local_offline_ocr_synthetic(self):
        # Create synthetic image with question label
        img = np.full((200, 500, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Q1 Normalization in DBMS reduces redundancy", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        is_success, buf = cv2.imencode(".png", img)
        assert is_success

        result = recognize_handwriting(buf.tobytes(), "test_answer.png")
        assert "ocr_engine" in result
        assert "ocr_confidence" in result
        assert isinstance(result["recognized_answers"], list)

    def test_parse_handwritten_answers_structured(self):
        raw_text = """
        Q1. What is Photosynthesis?
        Photosynthesis is the process where green plants convert light energy into chemical energy.
        Balanced equation: 6CO2 + 6H2O -> C6H12O6 + 6O2.
        
        Ans 2(a):
        Newton's third law states that for every action there is an equal and opposite reaction.
        Fab = -Fba. Forces act on two different bodies.
        
        3.
        Ohm's law states that current is directly proportional to potential difference (V = IR).
        """
        parsed = parse_handwritten_answers(raw_text)
        assert len(parsed) >= 3
        
        q_labels = [p.get("question_label") for p in parsed]
        assert any("1" in str(lbl) for lbl in q_labels)
        assert any("2" in str(lbl) for lbl in q_labels)
        assert any("3" in str(lbl) for lbl in q_labels)

        # Verify answer text content preserved
        q1 = next(p for p in parsed if "1" in str(p.get("question_label")))
        text_content = q1.get("text") or q1.get("answer_text") or ""
        assert "Photosynthesis" in text_content


# ==========================================
# 3. ANSWER MATCHER TESTS
# ==========================================
class TestAnswerMatcher:
    @pytest.fixture
    def sample_predicted_questions(self):
        return [
            {
                "id": "pred_q1",
                "question_number": 1,
                "section_name": "Section A",
                "question_text": "Explain Photosynthesis and write the balanced chemical equation.",
                "marks": 3.0,
                "topic": "Life Processes",
                "expected_concepts": ["Definition", "Balanced Equation", "Chlorophyll", "Stomata"],
                "reference_answer": "Photosynthesis is conversion of light into chemical energy: 6CO2 + 6H2O -> C6H12O6 + 6O2."
            },
            {
                "id": "pred_q2",
                "question_number": 2,
                "section_name": "Section A",
                "question_text": "State Newton's Third Law of Motion. Why do action-reaction forces not cancel?",
                "marks": 3.0,
                "topic": "Laws of Motion",
                "expected_concepts": ["Action-Reaction pair", "Equal and opposite", "Act on different bodies", "Fab = -Fba"],
                "reference_answer": "For every action there is an equal and opposite reaction. They act on two different bodies simultaneously."
            },
            {
                "id": "pred_q3",
                "question_number": 3,
                "section_name": "Section B",
                "question_text": "State Ohm's Law and calculate resistance when V=12V and I=3A.",
                "marks": 2.0,
                "topic": "Electricity",
                "expected_concepts": ["V = IR", "Direct proportionality", "R = V/I = 4 Ohms"],
                "reference_answer": "V = IR. At constant temperature, potential difference is proportional to current. R = 12/3 = 4 Ohms."
            },
            {
                "id": "pred_q4",
                "question_number": 4,
                "section_name": "Section B",
                "question_text": "Describe the function of the human heart and double circulation.",
                "marks": 4.0,
                "topic": "Circulation",
                "expected_concepts": ["Atria and ventricles", "Pulmonary circulation", "Systemic circulation", "Oxygenated blood"],
                "reference_answer": "Human heart pumps blood. Double circulation involves pulmonary circuit to lungs and systemic circuit to body."
            }
        ]

    def test_extract_question_label_variations(self):
        cases = [
            ("Q1. Photosynthesis is...", (1, None)),
            ("Question 2(a): Fab = -Fba", (2, "a")),
            ("Ans 3(b) - Ohm's law", (3, "b")),
            ("4. Human heart", (4, None)),
            ("No label here at all", (None, None))
        ]
        for text, expected in cases:
            num, sub = extract_question_label(text)
            assert num == expected[0], f"Failed on '{text}'"
            assert sub == expected[1], f"Failed sub on '{text}'"

    def test_match_out_of_order_answers(self, sample_predicted_questions):
        # Student wrote answers out of order: Q3, then Q1, then Q2
        recognized_answers = [
            {
                "question_label": "3",
                "subquestion_label": None,
                "answer_text": "Ohm's law: V = IR. When V=12V and I=3A, Resistance R = V/I = 12/3 = 4 Ohms.",
                "page_number": 1,
                "confidence": 0.90,
                "flags": []
            },
            {
                "question_label": "1",
                "subquestion_label": None,
                "answer_text": "Photosynthesis is the process where chlorophyll absorbs light. 6CO2 + 6H2O -> C6H12O6 + 6O2.",
                "page_number": 1,
                "confidence": 0.88,
                "flags": []
            },
            {
                "question_label": "2",
                "subquestion_label": None,
                "answer_text": "Every action has equal and opposite reaction. Fab = -Fba. They act on two different bodies.",
                "page_number": 1,
                "confidence": 0.85,
                "flags": []
            }
        ]

        result = match_answers_to_questions(recognized_answers, sample_predicted_questions)
        matched = result["matched_items"]

        # Check Q1 matched
        q1_item = next(m for m in matched if m["question_number"] == 1)
        assert q1_item["match_status"] == "MATCHED"
        assert "Photosynthesis" in q1_item["student_answer"]

        # Check Q2 matched
        q2_item = next(m for m in matched if m["question_number"] == 2)
        assert q2_item["match_status"] == "MATCHED"
        assert "equal and opposite" in q2_item["student_answer"]

        # Check Q3 matched
        q3_item = next(m for m in matched if m["question_number"] == 3)
        assert q3_item["match_status"] == "MATCHED"
        assert "4 Ohms" in q3_item["student_answer"]

        # Check Q4 is properly marked UNANSWERED
        q4_item = next(m for m in matched if m["question_number"] == 4)
        assert q4_item["match_status"] == "UNANSWERED"
        assert q4_item["student_answer"] == ""

        assert result["total_answered_questions"] == 3
        assert result["total_unanswered_questions"] == 1

    def test_multipage_continuation_stitching(self, sample_predicted_questions):
        # Answer to Q1 starts on Page 1 and continues on Page 2
        recognized_answers = [
            {
                "question_label": "1",
                "subquestion_label": None,
                "answer_text": "Photosynthesis is light-driven synthesis of glucose. Sunlight is absorbed by chlorophyll. (continued...)",
                "page_number": 1,
                "confidence": 0.90,
                "flags": []
            },
            {
                "question_label": None,  # Continued without re-writing question number
                "subquestion_label": None,
                "answer_text": "Continuing from page 1: The water molecules are split via photolysis. 6CO2 + 6H2O -> C6H12O6 + 6O2.",
                "page_number": 2,
                "confidence": 0.88,
                "flags": ["continuation_marker"]
            }
        ]

        result = match_answers_to_questions(recognized_answers, sample_predicted_questions)
        q1_item = next(m for m in result["matched_items"] if m["question_number"] == 1)

        assert q1_item["match_status"] == "MATCHED"
        assert q1_item["is_continuation"] is True
        assert "Sunlight is absorbed" in q1_item["student_answer"]
        assert "photolysis" in q1_item["student_answer"]

    def test_ambiguous_question_flagging(self, sample_predicted_questions):
        # Unclear or conflicting question label
        recognized_answers = [
            {
                "question_label": "?",  # Ambiguous question number
                "subquestion_label": None,
                "answer_text": "The energy currency of the cell is ATP produced by mitochondria.",
                "page_number": 1,
                "confidence": 0.55,
                "flags": ["ambiguous_question_number"]
            }
        ]

        result = match_answers_to_questions(recognized_answers, sample_predicted_questions)
        # Should be flagged as ambiguous or unmatched rather than silently assigned to an arbitrary question
        assert len(result["ambiguous_matches"]) > 0 or len(result["unmatched_answers"]) > 0


# ==========================================
# 4. ANSWER EVALUATOR TESTS
# ==========================================
class TestAnswerEvaluator:
    def test_evaluate_single_correct_answer(self):
        question = {
            "id": "q_photo",
            "question_number": 1,
            "question_text": "Explain Photosynthesis with balanced chemical equation.",
            "max_marks": 5.0,
            "topic": "Life Processes",
            "expected_concepts": ["Definition", "Balanced Equation (6CO2 + 6H2O -> C6H12O6 + 6O2)", "Chlorophyll", "Stomata", "Photolysis"],
            "reference_answer": "Photosynthesis is the process in green plants: 6CO2 + 6H2O -> C6H12O6 + 6O2 in the presence of chlorophyll and sunlight."
        }
        student_answer = "Photosynthesis is the process where autotrophs convert light into chemical energy. The balanced equation is 6CO2 + 6H2O -> C6H12O6 + 6O2. Chlorophyll in chloroplasts absorbs sunlight while stomata regulate gas exchange. Water photolysis splits water."

        eval_res = evaluate_single_answer(question, student_answer, ocr_confidence=92.0)
        assert eval_res["marks_awarded"] >= 4.0
        assert eval_res["marks_awarded"] <= 5.0
        assert eval_res["status"] in ("full", "partial")
        assert len(eval_res["covered_concepts"]) >= 3
        assert "rubric_breakdown" in eval_res

    def test_evaluate_partial_marks_bounded(self):
        question = {
            "id": "q_newton",
            "question_number": 2,
            "question_text": "State Newton's Third Law and explain Fab = -Fba.",
            "max_marks": 4.0,
            "topic": "Laws of Motion",
            "expected_concepts": ["Equal and opposite reaction", "Fab = -Fba", "Act on two different bodies", "Simultaneous action"],
            "reference_answer": "Newton's Third Law states every action has an equal and opposite reaction (Fab = -Fba). Forces act on different bodies."
        }
        # Student wrote only definition, missed vector equation and explanation of why they don't cancel
        partial_answer = "For every action there is an equal and opposite reaction."

        eval_res = evaluate_single_answer(question, partial_answer, ocr_confidence=85.0)
        assert 1.0 <= eval_res["marks_awarded"] <= 3.0
        assert eval_res["status"] == "partial"
        assert eval_res["marks_awarded"] <= question["max_marks"]

    def test_unanswered_question_strictly_zero_marks(self):
        question = {
            "id": "q_empty",
            "question_number": 3,
            "question_text": "Derive the quadratic formula.",
            "max_marks": 5.0,
            "topic": "Algebra",
            "expected_concepts": ["ax^2 + bx + c = 0", "Completing the square", "x = (-b +- sqrt(b^2 - 4ac)) / 2a"],
            "reference_answer": "Derivation using completing the square."
        }
        eval_res = evaluate_single_answer(question, "", ocr_confidence=0.0)
        assert eval_res["marks_awarded"] == 0.0
        assert eval_res["status"] == "unanswered"

    def test_generate_paper_evaluation_report_programmatic_caps(self):
        matching_res = {
            "matched_items": [
                {
                    "question_id": "q1",
                    "question_number": 1,
                    "section_name": "Section A",
                    "question_text": "Explain photosynthesis.",
                    "max_marks": 3.0,
                    "student_answer": "Photosynthesis produces glucose: 6CO2 + 6H2O -> C6H12O6 + 6O2.",
                    "reference_answer": "Photosynthesis formula and definition.",
                    "expected_concepts": ["Formula", "Definition"],
                    "topic": "Biology",
                    "ocr_confidence": 90.0,
                    "match_status": "MATCHED"
                },
                {
                    "question_id": "q2",
                    "question_number": 2,
                    "section_name": "Section A",
                    "question_text": "State Newton's third law.",
                    "max_marks": 3.0,
                    "student_answer": "Fab = -Fba. Action and reaction are equal and opposite.",
                    "reference_answer": "Newton third law statement.",
                    "expected_concepts": ["Fab = -Fba", "Equal and opposite"],
                    "topic": "Physics",
                    "ocr_confidence": 88.0,
                    "match_status": "MATCHED"
                },
                {
                    "question_id": "q3",
                    "question_number": 3,
                    "section_name": "Section B",
                    "question_text": "Describe the human heart.",
                    "max_marks": 4.0,
                    "student_answer": "",  # Unanswered
                    "reference_answer": "Human heart chambers and circulation.",
                    "expected_concepts": ["Chambers", "Circulation"],
                    "topic": "Biology",
                    "ocr_confidence": 0.0,
                    "match_status": "UNANSWERED"
                }
            ],
            "unmatched_answers": [],
            "ambiguous_matches": []
        }

        report = generate_paper_evaluation_report(
            matching_res,
            paper_metadata={"id": "paper_test", "title": "Model Paper", "subject": "Science", "grade": 10, "board": "KSEAB"},
            student_info={"id": "STU-999", "name": "Test Student"}
        )

        summary = report["summary"]
        assert summary["max_possible_marks"] == 10.0
        assert summary["total_marks_obtained"] <= summary["max_possible_marks"]
        assert summary["total_marks_obtained"] > 0.0
        assert summary["questions_evaluated"] == 3
        assert summary["questions_answered"] == 2
        assert summary["questions_unanswered"] == 1
        assert 0.0 <= summary["percentage"] <= 100.0


# ==========================================
# 5. END-TO-END FASTAPI ROUTE TESTS
# ==========================================
class TestDescriptiveAPIRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_get_predicted_papers_endpoint(self, client):
        response = client.get("/api/descriptive/predicted-papers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            p = data[0]
            assert "id" in p
            assert "title" in p
            assert "subject" in p
            assert "total_marks" in p

    def test_extract_and_match_endpoint(self, client):
        # Create small test image
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Q1. Test answer", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        _, img_encoded = cv2.imencode(".png", img)
        file_bytes = img_encoded.tobytes()

        response = client.post(
            "/api/descriptive/extract-and-match",
            files={"file": ("test_sheet.png", file_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "ocr_text" in data
        assert "ocr_confidence" in data
        assert "matched_items" in data
        assert "unanswered_questions" in data
        assert "total_predicted_questions" in data

    def test_evaluate_paper_and_history_lifecycle(self, client):
        payload = {
            "paper_id": "test_paper_cycle",
            "paper_title": "Class 10 Science Practice Paper",
            "subject": "Science",
            "grade": 10,
            "board": "KSEAB",
            "student_id": "STU-E2E-TEST",
            "student_name": "E2E Student",
            "matched_items": [
                {
                    "question_id": "q1",
                    "question_number": 1,
                    "section_name": "Section A",
                    "question_text": "State Newton's Third Law of Motion.",
                    "max_marks": 2.0,
                    "student_answer": "For every action there is an equal and opposite reaction.",
                    "reference_answer": "Fab = -Fba action reaction forces.",
                    "expected_concepts": ["Equal and opposite", "Fab = -Fba"],
                    "topic": "Physics",
                    "ocr_confidence": 90.0,
                    "match_status": "MATCHED"
                },
                {
                    "question_id": "q2",
                    "question_number": 2,
                    "section_name": "Section A",
                    "question_text": "Explain Photosynthesis.",
                    "max_marks": 3.0,
                    "student_answer": "",  # Unanswered
                    "reference_answer": "6CO2 + 6H2O -> C6H12O6 + 6O2",
                    "expected_concepts": ["Formula", "Chlorophyll"],
                    "topic": "Biology",
                    "ocr_confidence": 0.0,
                    "match_status": "UNANSWERED"
                }
            ],
            "unmatched_answers": [],
            "ambiguous_matches": [],
            "uploaded_file_name": "test_upload.png",
            "uploaded_file_path": "/tmp/test_upload.png",
            "ocr_confidence": 88.0,
            "ocr_engine": "Test OCR Engine"
        }

        # 1. Evaluate paper
        eval_resp = client.post("/api/descriptive/evaluate-paper", json=payload)
        assert eval_resp.status_code == 200
        report = eval_resp.json()

        assert "report_id" in report
        report_id = report["report_id"]
        summary = report["summary"]
        assert summary["max_possible_marks"] == 5.0
        assert summary["total_marks_obtained"] <= 5.0
        assert summary["questions_answered"] == 1
        assert summary["questions_unanswered"] == 1

        # 2. Check student reports list
        hist_resp = client.get("/api/descriptive/paper-reports/STU-E2E-TEST")
        assert hist_resp.status_code == 200
        reports_list = hist_resp.json()
        assert any(r["id"] == report_id for r in reports_list)

        # 3. Retrieve specific report
        single_resp = client.get(f"/api/descriptive/paper-report/{report_id}")
        assert single_resp.status_code == 200
        single_report = single_resp.json()
        assert single_report["report_id"] == report_id
        assert len(single_report["question_evaluations"]) == 2
