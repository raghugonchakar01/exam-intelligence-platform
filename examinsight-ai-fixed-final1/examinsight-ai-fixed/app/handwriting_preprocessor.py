"""Advanced Image and PDF Preprocessor for Handwritten Answer Sheets.

Provides:
- Embedded digital text extraction from clean digital PDFs (PyMuPDF)
- High-resolution multi-page PDF rendering and multipage image extraction
- Edge-preserving denoising (Bilateral filtering) preserving pen/pencil strokes
- Bounded deskewing and orientation correction (prevents accidental 90-degree flips)
- Illumination flattening and local contrast enhancement (CLAHE) for camera shadows
- Layout analysis, margin detection, and paragraph/answer-block segmentation
- Strikethrough / crossed-out text pattern detection
"""

from __future__ import annotations

import io
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import pymupdf as fitz
    FITZ_AVAILABLE = True
except ImportError:
    try:
        import fitz
        FITZ_AVAILABLE = True
    except ImportError:
        FITZ_AVAILABLE = False

MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "20"))


def extract_digital_pdf_text(file_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
    """Attempts direct digital text extraction from PDF pages.
    
    If the PDF contains selectable embedded digital text (not a pure scan/photo),
    extracts the verbatim text directly with 100% fidelity without consuming vision tokens.
    Returns a list of per-page text dicts if valid text is found, or None if scanned/empty.
    """
    if not FITZ_AVAILABLE:
        return None

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if len(doc) == 0:
            return None

        if len(doc) > MAX_OCR_PAGES:
            raise ValueError(f"Document contains {len(doc)} pages, which exceeds the maximum limit of {MAX_OCR_PAGES} pages.")

        page_texts: List[Dict[str, Any]] = []
        total_len = 0

        for p_idx in range(len(doc)):
            page = doc[p_idx]
            txt = page.get_text().strip()
            page_texts.append({
                "page_number": p_idx + 1,
                "text": txt,
                "is_digital": True
            })
            total_len += len(txt)

        # If on average there are at least 30 characters per page, treat as digital PDF
        if total_len > (len(doc) * 30):
            return page_texts
    except ValueError:
        raise
    except Exception:
        pass

    return None


def load_document_pages(file_bytes: bytes, filename: str, target_dpi: int = 200) -> List[np.ndarray]:
    """Loads an uploaded PDF or image file into a list of BGR numpy image arrays (one per page).
    
    Preserves page ordering and renders PDFs at high resolution for optimal handwriting recognition.
    """
    ext = os.path.splitext(filename)[1].lower()
    pages: List[np.ndarray] = []

    if ext == ".pdf":
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF (fitz) is required to render PDF pages.")
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("The uploaded PDF file contains no pages.")

        if len(doc) > MAX_OCR_PAGES:
            raise ValueError(f"Uploaded PDF exceeds maximum page limit of {MAX_OCR_PAGES} pages (contains {len(doc)} pages).")

        for p_idx in range(len(doc)):
            page = doc[p_idx]
            zoom = target_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            
            if pix.n == 3:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
            pages.append(img_bgr)
    else:
        # Image file (JPG, PNG, TIFF, BMP, WEBP)
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback via Pillow
            try:
                pil_img = Image.open(io.BytesIO(file_bytes))
                n_frames = getattr(pil_img, "n_frames", 1)
                for frame_idx in range(n_frames):
                    pil_img.seek(frame_idx)
                    rgb = pil_img.convert("RGB")
                    arr = np.array(rgb)
                    pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            except Exception as exc:
                raise ValueError(f"Could not decode image file '{filename}': {exc}")
        else:
            pages.append(img)

    if not pages:
        raise ValueError(f"No valid image pages could be extracted from '{filename}'.")

    return pages


def enhance_illumination_and_contrast(img: np.ndarray) -> np.ndarray:
    """Corrects uneven phone lighting, shadows, and faint ink using CLAHE in LAB space."""
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Adaptive histogram equalization on lightness
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise_handwriting(img: np.ndarray) -> np.ndarray:
    """Removes paper grain and camera noise while strictly preserving handwritten ink strokes."""
    return cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=35)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimates skew angle of handwritten text lines, strictly bounded within [-20, 20] degrees."""
    h, w = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    margin_y, margin_x = int(h * 0.05), int(w * 0.05)
    core_thresh = thresh[margin_y:h - margin_y, margin_x:w - margin_x]

    coords = np.column_stack(np.where(core_thresh > 0))
    if len(coords) < 300:
        return 0.0

    try:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        if angle < -45.0:
            angle = -(90.0 + angle)
        elif angle > 45.0:
            angle = 90.0 - angle
        else:
            angle = -angle

        if 0.5 < abs(angle) <= 22.0:
            return float(angle)
    except Exception:
        pass

    return 0.0


def deskew_image(img: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Rotates image to correct page tilt while strictly avoiding 90-degree accidental flips."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    calculated_angle = angle if angle is not None else estimate_skew_angle(gray)

    # Safety clamp: if estimated angle is too high, do not perform rotation
    if abs(calculated_angle) < 0.5 or abs(calculated_angle) > 22.0:
        return img, 0.0

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, calculated_angle, 1.0)
    deskewed = cv2.warpAffine(
        img,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return deskewed, calculated_angle


def detect_strikethrough_regions(gray_img: np.ndarray) -> List[Dict[str, int]]:
    """Detects horizontal or cross-out strokes indicating crossed-out student answers."""
    h, w = gray_img.shape[:2]
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal line kernel to detect strikethroughs
    line_len = max(30, int(w * 0.05))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strikethroughs = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > w * 0.08 and ch < 15:  # Long horizontal stroke
            strikethroughs.append({"x": x, "y": y, "w": cw, "h": ch})

    return strikethroughs


def segment_page_layout(img: np.ndarray, page_index: int = 1) -> Dict[str, Any]:
    """Analyzes page layout: detects header area, margins, paragraphs, and text line blocks."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    header_h = int(h * 0.16) if page_index == 1 else int(h * 0.06)
    header_crop = img[0:header_h, :]
    body_crop = img[header_h:h, :]

    margin_w = int(w * 0.18)
    margin_crop = body_crop[:, 0:margin_w]
    content_crop = body_crop[:, margin_w:w]

    gray_body = gray[header_h:h, :]
    _, binary_body = cv2.threshold(gray_body, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.06), 3))
    dilated = cv2.dilate(binary_body, line_kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.08 and bh > 12:
            blocks.append({
                "y": int(by + header_h),
                "x": int(bx),
                "w": int(bw),
                "h": int(bh),
                "area": int(bw * bh)
            })

    blocks.sort(key=lambda b: b["y"])

    return {
        "page_number": page_index,
        "dimensions": {"width": w, "height": h},
        "header_height": header_h,
        "margin_width": margin_w,
        "text_block_count": len(blocks),
        "blocks": blocks,
        "strikethroughs": detect_strikethrough_regions(gray)
    }


def preprocess_page(img: np.ndarray, page_index: int = 1) -> Dict[str, Any]:
    """Complete preprocessing pipeline for a single page:
    
    1. Illumination flattening (CLAHE)
    2. Edge-preserving denoising
    3. Safe deskewing
    4. Layout analysis and page segmentation
    Returns preprocessed image (BGR), preprocessed JPEG bytes, and layout metadata.
    """
    enhanced = enhance_illumination_and_contrast(img)
    denoised = denoise_handwriting(enhanced)
    deskewed, skew_deg = deskew_image(denoised)

    layout = segment_page_layout(deskewed, page_index=page_index)
    layout["skew_angle_corrected"] = round(skew_deg, 2)

    success, enc = cv2.imencode(".jpg", deskewed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    jpg_bytes = enc.tobytes() if success else b""

    return {
        "image": deskewed,
        "jpeg_bytes": jpg_bytes,
        "layout": layout,
        "page_number": page_index
    }


def preprocess_document(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Master document preprocessor: handles multipage PDFs and images.
    
    Returns a list of preprocessed page dictionaries preserving page order.
    """
    raw_pages = load_document_pages(file_bytes, filename)
    processed_pages = []

    for idx, page_bgr in enumerate(raw_pages, start=1):
        p_res = preprocess_page(page_bgr, page_index=idx)
        processed_pages.append(p_res)

    return processed_pages
