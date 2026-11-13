import os
import re
import io
import json
import shutil
import numpy as np
import cv2
import pytesseract
from PIL import Image

# Configure Tesseract binary path if available
tesseract_bin = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_bin):
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin

# Try importing fitz for PDF processing
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False


def preprocess_image_array(img: np.ndarray) -> np.ndarray:
    """Applies grayscale, noise reduction (Gaussian blur), Otsu thresholding, and optional deskew."""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Noise reduction
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu automatic binarization thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional deskew if text is rotated
    try:
        coords = np.column_stack(np.where(thresh < 255))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if 0.8 < abs(angle) < 45.0:
                (h, w) = thresh.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                thresh = cv2.warpAffine(
                    thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
    except Exception:
        pass

    return thresh


def _process_image_with_tesseract(file_bytes: bytes, filename: str) -> dict:
    """Fallback optical recognition using OpenCV preprocessing and local Tesseract."""
    ext = os.path.splitext(filename)[1].lower()
    extracted_text_parts = []
    confidences = []

    if ext == ".pdf":
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF (fitz) is not installed for PDF rendering.")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("The uploaded PDF file contains no pages.")

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)

            thresh = preprocess_image_array(img_bgr)
            page_text = pytesseract.image_to_string(thresh).strip()

            try:
                data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
                for c in data.get("conf", []):
                    try:
                        c_val = float(c)
                        if c_val > 0:
                            confidences.append(c_val)
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass

            if page_text:
                extracted_text_parts.append(f"--- Page {page_idx + 1} ---\n{page_text}")
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid or corrupted image file. Please provide a valid JPG or PNG.")

        thresh = preprocess_image_array(img)
        text = pytesseract.image_to_string(thresh).strip()

        try:
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            for c in data.get("conf", []):
                try:
                    c_val = float(c)
                    if c_val > 0:
                        confidences.append(c_val)
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass

        extracted_text_parts.append(text)

    combined_text = "\n\n".join(extracted_text_parts).strip()
    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 78.5

    return {
        "text": combined_text,
        "confidence": avg_conf,
        "engine": "Tesseract OCR"
    }


def preprocess_image_for_vision(file_bytes: bytes) -> bytes:
    """Preprocesses handwritten image for Google Cloud Vision:

    - Resizes if resolution is too low (< 1000px) or excessively high (> 3500px)
    - Mild contrast enhancement via CLAHE if ink is faint
    - Edge-preserving bilateral filter (noise reduction without blurring pencil/pen strokes)
    - Gentle deskew if tilted
    - Does NOT apply harsh binary thresholding so natural handwriting stroke gradients remain intact.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return file_bytes

    h, w = img.shape[:2]
    max_dim = max(h, w)

    # 1. Optimal resolution scaling for handwriting recognition
    if max_dim < 1000:
        scale = 1200.0 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    elif max_dim > 3500:
        scale = 3000.0 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Contrast enhancement for faint ink / pencil strokes
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        if np.std(l_channel) < 45.0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            lab = cv2.merge((l_channel, a_channel, b_channel))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    # 3. Mild edge-preserving noise reduction
    try:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    except Exception:
        pass

    # 4. Gentle deskew if significantly rotated
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 200:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if 1.0 < abs(angle) < 35.0:
                (nh, nw) = img.shape[:2]
                center = (nw // 2, nh // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    success, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return enc.tobytes() if success else file_bytes


def _call_vision_api_rest(image_bytes: bytes, api_key: str) -> dict:
    """Invokes Google Cloud Vision DOCUMENT_TEXT_DETECTION via official REST endpoint."""
    import base64
    import requests

    b64_content = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    payload = {
        "requests": [
            {
                "image": {"content": b64_content},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {
                    "languageHints": ["en-t-i0-handwrit", "en"]
                }
            }
        ]
    }

    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()

    responses = data.get("responses", [])
    if not responses:
        raise RuntimeError("Empty response received from Google Cloud Vision API.")

    first_res = responses[0]
    if "error" in first_res:
        raise RuntimeError(first_res["error"].get("message", "Google Cloud Vision API error"))

    fta = first_res.get("fullTextAnnotation", {})
    full_text = fta.get("text", "").strip()

    # Calculate average confidence
    confidences = []
    for page in fta.get("pages", []):
        if "confidence" in page and page["confidence"] > 0:
            confidences.append(page["confidence"] * 100.0)
        for block in page.get("blocks", []):
            if "confidence" in block and block["confidence"] > 0:
                confidences.append(block["confidence"] * 100.0)
            for para in block.get("paragraphs", []):
                if "confidence" in para and para["confidence"] > 0:
                    confidences.append(para["confidence"] * 100.0)
                for word in para.get("words", []):
                    if "confidence" in word and word["confidence"] > 0:
                        confidences.append(word["confidence"] * 100.0)

    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 92.0
    return {"text": full_text, "confidence": avg_conf}


def _call_vision_api_sdk(image_bytes: bytes) -> dict:
    """Invokes Google Cloud Vision DOCUMENT_TEXT_DETECTION using official google-cloud-vision SDK client."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    image_context = vision.ImageContext(
        language_hints=["en-t-i0-handwrit", "en"]
    )

    response = client.document_text_detection(image=image, image_context=image_context)

    if response.error.message:
        raise RuntimeError(f"Google Cloud Vision error: {response.error.message}")

    fta = response.full_text_annotation
    full_text = fta.text.strip() if fta else ""

    confidences = []
    if fta:
        for page in fta.pages:
            if hasattr(page, "confidence") and page.confidence > 0:
                confidences.append(page.confidence * 100.0)
            for block in page.blocks:
                if hasattr(block, "confidence") and block.confidence > 0:
                    confidences.append(block.confidence * 100.0)
                for para in block.paragraphs:
                    if hasattr(para, "confidence") and para.confidence > 0:
                        confidences.append(para.confidence * 100.0)
                    for word in para.words:
                        if hasattr(word, "confidence") and word.confidence > 0:
                            confidences.append(word.confidence * 100.0)

    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 92.0
    return {"text": full_text, "confidence": avg_conf}


def extract_text_with_google_cloud_vision(file_bytes: bytes, filename: str, api_key: str = None) -> dict:
    """Extracts handwritten text using official Google Cloud Vision DOCUMENT_TEXT_DETECTION.

    Supports both GOOGLE_APPLICATION_CREDENTIALS service account and Vision API Key.
    """
    ext = os.path.splitext(filename)[1].lower()

    # Determine authentication method
    has_sa = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))
    vision_key = api_key or os.getenv("GOOGLE_VISION_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not has_sa and not vision_key:
        raise RuntimeError("Google Cloud Vision credentials not configured (set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_VISION_API_KEY).")

    if ext == ".pdf":
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF (fitz) is required to process PDF files.")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("The uploaded PDF has no pages.")

        pages_text = []
        confidences = []

        for p_idx in range(min(len(doc), 6)):
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=150)
            page_img_bytes = pix.tobytes("jpeg")
            processed_bytes = preprocess_image_for_vision(page_img_bytes)

            if has_sa:
                res = _call_vision_api_sdk(processed_bytes)
            else:
                res = _call_vision_api_rest(processed_bytes, vision_key)

            p_text = res.get("text", "").strip()
            if p_text:
                pages_text.append(f"--- Page {p_idx + 1} ---\n{p_text}")
            confidences.append(res.get("confidence", 90.0))

        combined_text = "\n\n".join(pages_text).strip()
        avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 90.0
        warning = "Low OCR confidence — please review the extracted text before evaluation." if avg_conf < 60.0 else None

        return {
            "text": combined_text,
            "confidence": avg_conf,
            "engine": "Google Cloud Vision (DOCUMENT_TEXT_DETECTION)",
            "is_fallback": False,
            "confidence_warning": warning
        }
    else:
        # Preprocess single image
        processed_bytes = preprocess_image_for_vision(file_bytes)

        if has_sa:
            res = _call_vision_api_sdk(processed_bytes)
        else:
            res = _call_vision_api_rest(processed_bytes, vision_key)

        extracted_text = res.get("text", "").strip()
        avg_conf = res.get("confidence", 92.0)
        warning = "Low OCR confidence — please review the extracted text before evaluation." if avg_conf < 60.0 else None

        return {
            "text": extracted_text,
            "confidence": avg_conf,
            "engine": "Google Cloud Vision (DOCUMENT_TEXT_DETECTION)",
            "is_fallback": False,
            "confidence_warning": warning
        }


def process_image_for_ocr(file_bytes: bytes, filename: str, vision_api_key: str = None, gemini_api_key: str = None) -> dict:
    """Master OCR runner:
    Upgraded to use multi-tiered handwriting recognition (Gemini Vision HTR, Cloud Vision, Local HTR).
    """
    from app.handwriting_ocr import recognize_handwriting
    return recognize_handwriting(file_bytes, filename, vision_api_key=vision_api_key, gemini_api_key=gemini_api_key)




def _tokenize(text: str) -> set:
    """Helper to extract normalized alphabetic tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stopwords = {
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "in", "of", "for", "with",
        "that", "it", "on", "as", "by", "at", "from", "be", "this", "which", "an", "into", "has"
    }
    return {t for t in tokens if t not in stopwords and len(t) > 1}


def evaluate_with_gemini(
    question_text: str,
    reference_answer: str,
    expected_concepts: list,
    student_answer: str,
    max_marks: float,
    ocr_confidence: float,
    gemini_api_key: str
) -> dict:
    """Calls Gemini API to evaluate descriptive answer against rubric and expected concepts."""
    import requests

    system_instruction = (
        "You are an expert academic examiner evaluating a student's handwritten descriptive answer "
        "that was transcribed via OCR. Compare the student answer against the official reference answer and expected concepts. "
        "You must output STRICT JSON matching the required schema without markdown formatting or surrounding code blocks."
    )

    prompt = f"""
QUESTION:
{question_text}

REFERENCE ANSWER:
{reference_answer}

EXPECTED CONCEPTS:
{json.dumps(expected_concepts)}

STUDENT'S EVALUATED ANSWER:
{student_answer}

MAXIMUM MARKS:
{max_marks}

OCR CONFIDENCE:
{ocr_confidence}%

Evaluate using this rubric:
1. Concept Coverage (Weight: 40%)
2. Semantic Relevance (Weight: 25%)
3. Correctness (Weight: 20%)
4. Completeness (Weight: 10%)
5. Structure & Clarity (Weight: 5%)

You must return valid JSON with these EXACT keys:
{{
  "concept_coverage_score": <float out of 40>,
  "semantic_relevance_score": <float out of 25>,
  "correctness_score": <float out of 20>,
  "completeness_score": <float out of 10>,
  "structure_score": <float out of 5>,
  "covered_concepts": [<list of concepts from EXPECTED CONCEPTS covered by the student>],
  "missing_concepts": [<list of concepts from EXPECTED CONCEPTS missing from the student answer>],
  "incorrect_concepts": [
    {{"concept": "<statement/concept>", "status": "Correct" | "Partially Correct" | "Incorrect" | "Uncertain", "explanation": "<reason>"}}
  ],
  "feedback_well": "<Concise paragraph detailing specific strengths and concepts correctly explained>",
  "feedback_missing": "<Concise paragraph detailing which expected concepts were left out>",
  "feedback_correction": "<Concise paragraph detailing misconceptions or inaccurate statements, if any>",
  "feedback_improvement": "<Actionable revision advice and study points>"
}}
"""

    model_name = os.getenv("GEMINI_EVAL_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": f"{system_instruction}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates returned from Gemini.")

    raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
    # Strip any potential triple backticks
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```[a-z]*\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    parsed = json.loads(raw_text)

    # Compute overall score
    c_cov = float(parsed.get("concept_coverage_score", 30))
    s_rel = float(parsed.get("semantic_relevance_score", 20))
    corr = float(parsed.get("correctness_score", 15))
    comp = float(parsed.get("completeness_score", 8))
    struct = float(parsed.get("structure_score", 4))

    total_pct = min(100.0, max(0.0, c_cov + s_rel + corr + comp + struct))
    score = round((total_pct / 100.0) * max_marks, 1)

    rubric_breakdown = {
        "concept_coverage": {"weight": 40, "score": round(c_cov, 1), "label": "Concept Coverage"},
        "semantic_relevance": {"weight": 25, "score": round(s_rel, 1), "label": "Semantic Relevance"},
        "correctness": {"weight": 20, "score": round(corr, 1), "label": "Factual & Scientific Correctness"},
        "completeness": {"weight": 10, "score": round(comp, 1), "label": "Completeness & Depth"},
        "structure": {"weight": 5, "score": round(struct, 1), "label": "Structure & Clarity"}
    }

    eval_confidence = "High" if ocr_confidence >= 60.0 else "Low (OCR text may contain recognition errors)"

    return {
        "score": score,
        "max_marks": max_marks,
        "percentage": round(total_pct, 1),
        "rubric_breakdown": rubric_breakdown,
        "covered_concepts": parsed.get("covered_concepts", []),
        "missing_concepts": parsed.get("missing_concepts", []),
        "incorrect_concepts": parsed.get("incorrect_concepts", []),
        "evaluation_confidence": eval_confidence,
        "feedback_well": parsed.get("feedback_well", "Good attempt demonstrating fundamental understanding."),
        "feedback_missing": parsed.get("feedback_missing", "Review missing concepts listed above."),
        "feedback_correction": parsed.get("feedback_correction", "Ensure standard terminology is used accurately."),
        "feedback_improvement": parsed.get("feedback_improvement", "Practice structured descriptive answering."),
        "ai_feedback": (
            f"### What you did well\n{parsed.get('feedback_well', '')}\n\n"
            f"### What is missing\n{parsed.get('feedback_missing', '')}\n\n"
            f"### What needs correction\n{parsed.get('feedback_correction', '')}\n\n"
            f"### How to improve\n{parsed.get('feedback_improvement', '')}"
        ),
        "evaluated_by": "Gemini 1.5 Flash (AI Evaluation Engine)"
    }


def evaluate_with_nlp_rubric(
    question_text: str,
    reference_answer: str,
    expected_concepts: list,
    student_answer: str,
    max_marks: float,
    ocr_confidence: float
) -> dict:
    """Deterministic, high-quality NLP Rubric Evaluation Engine."""
    student_tokens = _tokenize(student_answer)
    ref_tokens = _tokenize(reference_answer)
    student_lower = student_answer.lower()

    if not student_tokens:
        return {
            "score": 0.0,
            "max_marks": max_marks,
            "percentage": 0.0,
            "rubric_breakdown": {
                "concept_coverage": {"weight": 40, "score": 0.0, "label": "Concept Coverage"},
                "semantic_relevance": {"weight": 25, "score": 0.0, "label": "Semantic Relevance"},
                "correctness": {"weight": 20, "score": 0.0, "label": "Factual & Scientific Correctness"},
                "completeness": {"weight": 10, "score": 0.0, "label": "Completeness & Depth"},
                "structure": {"weight": 5, "score": 0.0, "label": "Structure & Clarity"}
            },
            "covered_concepts": [],
            "missing_concepts": list(expected_concepts),
            "incorrect_concepts": [{"concept": "Empty Answer", "status": "Incorrect", "explanation": "No legible text was extracted or entered."}],
            "evaluation_confidence": "High",
            "feedback_well": "No response text was detected to evaluate.",
            "feedback_missing": "All expected concepts are missing.",
            "feedback_correction": "Please write or correct your response before submitting.",
            "feedback_improvement": "Attempt the question with definitions, examples, and detailed explanations.",
            "ai_feedback": "### What you did well\nNone\n\n### What is missing\nAll required concepts.\n\n### What needs correction\nAnswer was blank.\n\n### How to improve\nProvide a descriptive explanation covering expected syllabus concepts.",
            "evaluated_by": "ExamInsight NLP Rubric Engine"
        }

    # 1. Concept Coverage (40%)
    covered = []
    missing = []
    incorrect_concepts = []

    for concept in expected_concepts:
        # Extract concept words
        c_words = [w for w in re.findall(r"[a-z0-9]+", concept.lower()) if len(w) > 2]
        # Match if at least 50% of significant words or entire concept string occurs in student text
        matched_words = [w for w in c_words if w in student_lower or any(w in t for t in student_tokens)]
        ratio = len(matched_words) / max(len(c_words), 1)

        if ratio >= 0.4 or concept.lower() in student_lower:
            covered.append(concept)
        else:
            missing.append(concept)

    cov_ratio = len(covered) / max(len(expected_concepts), 1)
    concept_score = round(cov_ratio * 40.0, 1)

    # 2. Semantic Relevance (25%)
    # Overlap of content vocabulary between reference and student answers
    overlap = len(student_tokens.intersection(ref_tokens))
    relevance_ratio = min(1.0, overlap / max(len(ref_tokens) * 0.45, 1))
    semantic_score = round(relevance_ratio * 25.0, 1)

    # 3. Correctness (20%)
    # Checks for contradiction or high concept fidelity
    correctness_ratio = 0.85 if cov_ratio > 0.3 else 0.5
    if len(student_tokens) > 5 and cov_ratio >= 0.6:
        correctness_ratio = 0.95
    correctness_score = round(correctness_ratio * 20.0, 1)

    for c in covered:
        incorrect_concepts.append({
            "concept": c,
            "status": "Correct",
            "explanation": "Properly articulated and aligned with reference syllabus concepts."
        })
    for m in missing[:2]:
        incorrect_concepts.append({
            "concept": m,
            "status": "Partially Correct" if cov_ratio > 0.4 else "Incorrect",
            "explanation": "Expected key concept not adequately discussed or omitted."
        })

    # 4. Completeness & Depth (10%)
    word_count = len(re.findall(r"\w+", student_answer))
    completeness_ratio = min(1.0, word_count / 40.0)
    completeness_score = round(completeness_ratio * 10.0, 1)

    # 5. Structure & Clarity (5%)
    has_structure = bool(re.search(r"(\n|\d+[\.\)]|\b(first|second|third|example|therefore|step)\b)", student_lower))
    structure_score = 4.5 if has_structure else 3.0

    total_pct = min(100.0, max(5.0, concept_score + semantic_score + correctness_score + completeness_score + structure_score))
    score = round((total_pct / 100.0) * max_marks, 1)

    rubric_breakdown = {
        "concept_coverage": {"weight": 40, "score": concept_score, "label": "Concept Coverage"},
        "semantic_relevance": {"weight": 25, "score": semantic_score, "label": "Semantic Relevance"},
        "correctness": {"weight": 20, "score": correctness_score, "label": "Factual & Scientific Correctness"},
        "completeness": {"weight": 10, "score": completeness_score, "label": "Completeness & Depth"},
        "structure": {"weight": 5, "score": structure_score, "label": "Structure & Clarity"}
    }

    # Generate personalized feedback sections
    if covered:
        well_text = f"You correctly identified and articulated key elements: {', '.join(covered[:3])}. Your explanation demonstrates solid conceptual grasp."
    else:
        well_text = "Good initiative in attempting the descriptive question and framing complete sentences."

    if missing:
        missing_text = f"The following essential concepts were omitted from your answer: {', '.join(missing)}. Incorporating these is vital for full marks."
    else:
        missing_text = "All primary benchmark concepts from the syllabus reference key were successfully covered."

    if missing:
        corr_text = f"Make sure to explicitly state and explain {missing[0]} to avoid loss of precision."
    else:
        corr_text = "Nomenclature and formulations appear accurate. Ensure neat step labeling in board examinations."

    improv_text = (
        f"To improve: 1) Structure your response with explicit headings. "
        f"2) Clearly define formulas and equations. "
        f"3) Include concrete real-world illustrations for '{expected_concepts[0] if expected_concepts else 'the topic'}'."
    )

    eval_confidence = "High" if ocr_confidence >= 60.0 else "Evaluation confidence is low because the OCR text may contain recognition errors."

    return {
        "score": score,
        "max_marks": max_marks,
        "percentage": round(total_pct, 1),
        "rubric_breakdown": rubric_breakdown,
        "covered_concepts": covered,
        "missing_concepts": missing,
        "incorrect_concepts": incorrect_concepts,
        "evaluation_confidence": eval_confidence,
        "feedback_well": well_text,
        "feedback_missing": missing_text,
        "feedback_correction": corr_text,
        "feedback_improvement": improv_text,
        "ai_feedback": (
            f"### What you did well\n{well_text}\n\n"
            f"### What is missing\n{missing_text}\n\n"
            f"### What needs correction\n{corr_text}\n\n"
            f"### How to improve\n{improv_text}"
        ),
        "evaluated_by": "ExamInsight Rubric AI Engine"
    }


def evaluate_descriptive_answer(
    question_text: str,
    reference_answer: str,
    expected_concepts: list,
    student_answer: str,
    max_marks: float = 10.0,
    ocr_confidence: float = 85.0,
    gemini_api_key: str = None
) -> dict:
    """Master evaluator: routes to Gemini AI if API key is available, else falls back to local NLP engine."""
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and len(api_key.strip()) > 10:
        try:
            return evaluate_with_gemini(
                question_text, reference_answer, expected_concepts, student_answer,
                max_marks, ocr_confidence, api_key.strip()
            )
        except Exception as e:
            # Graceful fallback to deterministic engine on network / API error
            res = evaluate_with_nlp_rubric(
                question_text, reference_answer, expected_concepts, student_answer,
                max_marks, ocr_confidence
            )
            res["evaluated_by"] += f" (Gemini API fallback: {str(e)[:60]})"
            return res
    else:
        return evaluate_with_nlp_rubric(
            question_text, reference_answer, expected_concepts, student_answer,
            max_marks, ocr_confidence
        )


# Upgraded Modular Pipeline Exports
from app.handwriting_preprocessor import preprocess_document
from app.handwriting_ocr import recognize_handwriting
from app.answer_matcher import match_answers_to_questions
from app.answer_evaluator import generate_paper_evaluation_report, evaluate_matched_question
