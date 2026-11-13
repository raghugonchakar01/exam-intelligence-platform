"""
extract_question_figures.py
===========================
One-time script that:
1. Opens each KSEAB SSLC PDF
2. Extracts genuine content images (diagrams, figures, tables) — filters out small
   decorative logos/borders
3. Matches each image to the nearest question text on the same page
4. Stores the image as base64 in pyq_questions.figure_base64

Run once:  python extract_question_figures.py
"""

import fitz
import sqlite3
import base64
import re
import os

DB_PATH = "examinsight.db"

# Minimum dimensions/size to be considered a real diagram (not a border/logo)
MIN_WIDTH  = 150   # px
MIN_HEIGHT = 80    # px
MIN_BYTES  = 3000  # bytes

# All PDFs to process
PDF_MAP = [
    {
        "path": "pyq/class10/mathematics/2022/KSEAB_SSLC_Class10_Mathematics_2022_Main_81E.pdf",
        "paper_id_pattern": "kseab_sslc_10_math_2022_main",
        "subject": "Mathematics (Standard/Basic)",
        "year": 2022,
    },
    {
        "path": "pyq/class10/mathematics/2022/KSEAB_SSLC_Class10_Mathematics_2022_Supplementary_81E.pdf",
        "paper_id_pattern": "kseab_sslc_10_math_2022_sup",
        "subject": "Mathematics (Standard/Basic)",
        "year": 2022,
    },
    {
        "path": "pyq/class10/science/2022/KSEAB_SSLC_Class10_Science_2022_Main_83E.pdf",
        "paper_id_pattern": "kseab_sslc_10_sci_2022_main",
        "subject": "Science (Physics, Chemistry, Bio)",
        "year": 2022,
    },
    {
        "path": "pyq/class10/social_science/2022/KSEAB_SSLC_Class10_SocialScience_2022_Main_85E.pdf",
        "paper_id_pattern": "kseab_sslc_10_soc_2022_main",
        "subject": "Social Science",
        "year": 2022,
    },
    {
        "path": "pyq/class10/english_communicative/2022/KSEAB_SSLC_Class10_English_2022_Main_31E.pdf",
        "paper_id_pattern": "kseab_sslc_10_eng_2022_main",
        "subject": "English Communicative",
        "year": 2022,
    },
]


def migrate_db(conn):
    """Add figure columns to pyq_questions if they don't exist."""
    c = conn.cursor()
    cols = [row[1] for row in c.execute("PRAGMA table_info(pyq_questions)")]
    if "figure_base64" not in cols:
        c.execute("ALTER TABLE pyq_questions ADD COLUMN figure_base64 TEXT")
        print("  Added column: figure_base64")
    if "figure_ext" not in cols:
        c.execute("ALTER TABLE pyq_questions ADD COLUMN figure_ext TEXT")
        print("  Added column: figure_ext")
    conn.commit()


def get_content_images_by_page(doc):
    """
    Returns a dict: page_num (0-indexed) -> list of (rect, base64_str, ext)
    Only real content images (large enough, not repeated decorative logos).
    """
    # First pass: find xrefs that repeat on EVERY page (decorative borders/logos)
    xref_page_count = {}
    for page_num, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            xref_page_count[xref] = xref_page_count.get(xref, 0) + 1

    total_pages = len(doc)
    # If an image appears on more than 60% of pages it's decorative
    decorative_xrefs = {xref for xref, cnt in xref_page_count.items()
                        if cnt > max(2, total_pages * 0.6)}

    result = {}
    seen_xrefs = set()

    for page_num, page in enumerate(doc):
        content_imgs = []
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in decorative_xrefs:
                continue
            if xref in seen_xrefs:
                # Already processed on a previous page — still record position on this page
                info = doc.extract_image(xref)
                w, h, size = info["width"], info["height"], len(info["image"])
                if w < MIN_WIDTH and h < MIN_HEIGHT:
                    continue
                if size < MIN_BYTES:
                    continue
                rects = page.get_image_rects(xref)
                if rects:
                    b64 = base64.b64encode(info["image"]).decode()
                    content_imgs.append((rects[0], b64, info["ext"]))
                continue

            info = doc.extract_image(xref)
            w, h, size = info["width"], info["height"], len(info["image"])
            if w < MIN_WIDTH and h < MIN_HEIGHT:
                continue
            if size < MIN_BYTES:
                continue

            rects = page.get_image_rects(xref)
            if rects:
                b64 = base64.b64encode(info["image"]).decode()
                content_imgs.append((rects[0], b64, info["ext"]))
            seen_xrefs.add(xref)

        if content_imgs:
            result[page_num] = content_imgs

    return result


def extract_question_numbers_from_page(page):
    """
    Returns list of (question_number, y_position) from a PDF page.
    Looks for patterns like '5.', '15.', '23.' at the start of text blocks.
    """
    blocks = page.get_text("blocks")
    q_positions = []
    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block
        if block_type != 0:  # 0 = text block
            continue
        # Match question numbers like "5.", "23." at start of text
        m = re.match(r'^\s*(\d{1,2})\s*[\.\)]\s', text.strip())
        if m:
            qnum = int(m.group(1))
            q_positions.append((qnum, y0, y1))
    return q_positions


def link_images_to_questions(pdf_info, conn):
    """
    For a given PDF, finds content images on each page and links them to the
    nearest question in the DB (by question number and page position).
    """
    path = pdf_info["path"]
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        return

    print(f"\nProcessing: {os.path.basename(path)}")
    doc = fitz.open(path)
    images_by_page = get_content_images_by_page(doc)

    if not images_by_page:
        print("  No content images found.")
        doc.close()
        return

    c = conn.cursor()
    updates = 0

    for page_num, img_list in images_by_page.items():
        page = doc[page_num]
        q_positions = extract_question_numbers_from_page(page)

        for img_rect, img_b64, img_ext in img_list:
            img_top = img_rect.y0
            img_mid  = (img_rect.y0 + img_rect.y1) / 2

            # Find the question whose top-y is closest ABOVE the image
            best_qnum = None
            best_dist = float("inf")
            for qnum, q_y0, q_y1 in q_positions:
                # Image should be BELOW the question text (q_y0 <= img_top)
                if q_y0 <= img_top + 40:  # allow 40px tolerance
                    dist = img_top - q_y0
                    if dist < best_dist:
                        best_dist = dist
                        best_qnum = qnum

            if best_qnum is None:
                # Fallback: pick any q on the page closest in absolute y distance
                for qnum, q_y0, q_y1 in q_positions:
                    dist = abs(img_mid - (q_y0 + q_y1) / 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_qnum = qnum

            if best_qnum is None:
                print(f"  Page {page_num+1}: image at y={img_top:.0f} — no question found nearby")
                continue

            # Look up the question in DB
            rows = c.execute(
                """SELECT id FROM pyq_questions
                   WHERE question_number = ?
                   AND subject LIKE ?
                   AND year = ?
                   LIMIT 1""",
                (best_qnum, f"%{pdf_info['subject'].split('(')[0].strip()}%", pdf_info["year"])
            ).fetchall()

            if not rows:
                print(f"  Page {page_num+1}: Q{best_qnum} not found in DB")
                continue

            row_id = rows[0][0]
            c.execute(
                "UPDATE pyq_questions SET figure_base64 = ?, figure_ext = ? WHERE id = ?",
                (img_b64, img_ext, row_id)
            )
            updates += 1
            print(f"  Page {page_num+1}: attached diagram to Q{best_qnum} (id={row_id})")

    conn.commit()
    doc.close()
    print(f"  Done — {updates} questions updated with figures")


def main():
    conn = sqlite3.connect(DB_PATH)
    print("=== Migrating DB ===")
    migrate_db(conn)

    print("\n=== Extracting figures from PDFs ===")
    for pdf_info in PDF_MAP:
        link_images_to_questions(pdf_info, conn)

    conn.close()

    # Summary
    conn2 = sqlite3.connect(DB_PATH)
    c = conn2.cursor()
    total = c.execute("SELECT COUNT(*) FROM pyq_questions WHERE figure_base64 IS NOT NULL").fetchone()[0]
    print(f"\n=== DONE: {total} questions now have an associated figure ===")
    conn2.close()


if __name__ == "__main__":
    main()
