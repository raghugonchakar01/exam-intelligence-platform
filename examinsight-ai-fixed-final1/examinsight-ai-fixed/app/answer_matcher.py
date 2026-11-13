"""Intelligent Question Identification and Answer Matching Engine.

Associates recognized handwritten answers with predicted question paper questions using:
1. Priority 1: Explicit question numbers and subparts (e.g. Q1, 2(a), 2(b), Ans 3, 4(i))
2. Priority 2: Semantic and content relevance to actual question text and curriculum topic
3. Priority 3: Page order and sequential context as supporting evidence
4. Multipage answer continuation stitching (Page 1 -> Page 2, P.T.O., 'cont.')
5. Out-of-order student response resolution
6. Strikethrough / crossed-out answer handling
7. Ambiguity flagging (flags uncertain matches for manual review rather than guessing)
8. Clear identification of UNANSWERED questions and UNMATCHED answers
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def _tokenize(text: str) -> Set[str]:
    """Helper to extract normalized content tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stopwords = {
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "in", "of", "for", "with",
        "that", "it", "on", "as", "by", "at", "from", "be", "this", "which", "into", "has", "have",
        "had", "do", "does", "did", "question", "answer", "marks", "explain", "describe", "what",
        "write", "state", "find", "give", "show", "calculate"
    }
    return {t for t in tokens if t not in stopwords and len(t) > 1}


def parse_question_label(label_str: str) -> Tuple[Optional[int], Optional[str]]:
    """Extracts integer question number and subquestion part.
    
    Examples:
    - 'Q1' -> (1, None)
    - '2(a)' -> (2, 'a')
    - '2(b)' -> (2, 'b')
    - 'Ans 3b' -> (3, 'b')
    - 'Question IV' -> (4, None)
    - '5(i)' -> (5, 'i')
    - 'unlabeled_1' -> (None, None)
    """
    if not label_str or "unlabeled" in label_str.lower():
        return None, None

    clean = label_str.strip().lower()

    # Map roman numerals to integers
    roman_map = {
        "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10
    }

    # Match patterns like Q1, 1., 2(a), 3b, Ans 4(i), Part B Q4
    match = re.search(
        r"(?:(?:q(?:uestion)?|ans(?:wer)?|part\s+[a-z]\s+q)?\.?\s*)(\d+|[ivx]+)(?:\s*(?:\(\s*([a-z0-9ivx]{1,3})\s*\)|[\.\-]?\s*([a-d]|i{1,3}|iv|v)\b))?",
        clean
    )
    if match:
        num_raw = match.group(1)
        sub_raw = match.group(2) or match.group(3)

        if num_raw.isdigit():
            q_num = int(num_raw)
        elif num_raw in roman_map:
            q_num = roman_map[num_raw]
        else:
            q_num = None

        sub_part = sub_raw.lower().strip("()") if sub_raw else None
        return q_num, sub_part

    return None, None


def stitch_multipage_continuations(raw_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stitches answers that span across multiple pages into unified response blocks.
    
    Handles explicit continuation tags ('cont', 'p.t.o.'), identical question labels on consecutive pages,
    and trailing paragraphs.
    """
    if not raw_answers:
        return []

    stitched: List[Dict[str, Any]] = []

    for ans in raw_answers:
        label = ans.get("question_label", "")
        is_cont = ans.get("is_continuation", False)
        page_num = ans.get("page_number", 1)
        text = (ans.get("text") or ans.get("answer_text") or "").strip()

        # Check explicit continuation markers in the answer text
        has_cont_marker = bool(
            re.search(r"\b(cont(?:d|inued)?|p\.?t\.?o\.?|continued on next page)\b", text, re.IGNORECASE)
        )

        if stitched:
            prev = stitched[-1]
            prev_num, prev_sub = parse_question_label(prev.get("question_label", ""))
            curr_num, curr_sub = parse_question_label(label)

            # Continuation criteria:
            # 1. Explicit same question number and subpart across different pages
            # 2. Answer text explicitly marked with continuation marker
            # 3. Previous answer flagged continued and current has no new question number
            is_same_q = (
                curr_num is not None 
                and curr_num == prev_num 
                and curr_sub == prev_sub 
                and page_num > prev.get("last_page", 1)
            )
            is_implicit_cont = (
                is_cont
                or has_cont_marker
                or (curr_num is None and page_num >= prev.get("last_page", 1) and not str(label).lower().startswith("q"))
            )

            if is_same_q or is_implicit_cont:
                clean_add_text = re.sub(
                    r"^\s*(?:cont(?:d|inued)?|p\.?t\.?o\.?)\s*[:\-\.]?\s*", "", text, flags=re.IGNORECASE
                )
                prev["text"] = prev["text"].rstrip() + "\n\n" + clean_add_text
                if page_num not in prev["pages"]:
                    prev["pages"].append(page_num)
                prev["last_page"] = max(prev.get("last_page", 1), page_num)
                prev["confidence"] = round((prev["confidence"] + ans.get("confidence", 80.0)) / 2, 1)
                prev["is_continuation"] = True
                continue

        stitched.append({
            "question_label": label,
            "text": text,
            "first_page": page_num,
            "last_page": page_num,
            "pages": [page_num],
            "is_continuation": is_cont or has_cont_marker,
            "is_crossed_out": ans.get("is_crossed_out", False),
            "confidence": ans.get("confidence", 80.0),
            "notes": ans.get("notes", "")
        })

    return stitched


def compute_semantic_overlap(answer_text: str, question_context: str) -> float:
    """Computes normalized content token overlap between answer text and question context."""
    a_tokens = _tokenize(answer_text)
    q_tokens = _tokenize(question_context)

    if not a_tokens or not q_tokens:
        return 0.0

    overlap = len(a_tokens.intersection(q_tokens))
    score = overlap / max(len(q_tokens) * 0.35, 1)
    return min(1.0, score)


def normalize_predicted_questions(paper_or_questions: Any) -> List[Dict[str, Any]]:
    """Normalizes questions from either a generated paper JSON, sections list, or question list.
    
    Flattens sections while preserving question numbers, subparts, max marks, and reference criteria.
    """
    flattened: List[Dict[str, Any]] = []

    if isinstance(paper_or_questions, dict) and "sections" in paper_or_questions:
        q_counter = 1
        for sec in paper_or_questions.get("sections", []):
            sec_name = sec.get("name", "Section")
            sec_type = sec.get("question_type", "Descriptive")
            for q in sec.get("questions", []):
                q_num = q.get("question_number") or q_counter
                q_text = q.get("question") or q.get("question_text", "")
                subpart = q.get("subpart")
                
                # Detect subpart in text if not explicitly present (e.g. "2(a) State...")
                if not subpart:
                    sub_match = re.match(r"^\(?([a-z0-9ivx]{1,2})\)[\.\-\s]", q_text, re.IGNORECASE)
                    if sub_match:
                        subpart = sub_match.group(1).lower()

                disp_num = f"{q_num}({subpart})" if subpart else str(q_num)

                flattened.append({
                    "id": q.get("id") or q.get("source_question_id") or f"q_{q_counter}",
                    "question_number": int(q_num) if str(q_num).isdigit() else q_counter,
                    "subpart": subpart,
                    "display_number": disp_num,
                    "section_name": sec_name,
                    "question_type": sec_type,
                    "question_text": q_text,
                    "marks": float(q.get("marks", 2.0)),
                    "topic": q.get("topic", "General"),
                    "expected_concepts": q.get("expected_concepts", []),
                    "reference_answer": q.get("reference_answer", "")
                })
                q_counter += 1
    elif isinstance(paper_or_questions, list):
        for idx, item in enumerate(paper_or_questions, start=1):
            q_num = item.get("question_number", idx)
            q_text = item.get("question_text") or item.get("question") or item.get("title", "")
            subpart = item.get("subpart")
            if not subpart:
                sub_match = re.match(r"^\(?([a-z0-9ivx]{1,2})\)[\.\-\s]", q_text, re.IGNORECASE)
                if sub_match:
                    subpart = sub_match.group(1).lower()

            disp_num = f"{q_num}({subpart})" if subpart else str(q_num)

            flattened.append({
                "id": item.get("id") or f"q_{idx}",
                "question_number": int(q_num) if str(q_num).isdigit() else idx,
                "subpart": subpart,
                "display_number": disp_num,
                "section_name": item.get("section_name", "Section A"),
                "question_type": item.get("question_type", "Descriptive"),
                "question_text": q_text,
                "marks": float(item.get("marks") or item.get("max_marks", 10.0)),
                "topic": item.get("topic", "General"),
                "expected_concepts": item.get("expected_concepts", []),
                "reference_answer": item.get("reference_answer", "")
            })

    return flattened


def match_answers_to_questions(
    recognized_answers: List[Dict[str, Any]],
    predicted_questions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Associates recognized handwritten answers with predicted questions.
    
    Priority matching order:
    1. Explicit question number and subpart match (e.g. Q1, 2(a), 2(b))
    2. Semantic relevance and keyword overlap with question statement and topic
    3. Page order and sequence context as supporting evidence
    
    Handles:
    - Out of order student answers
    - Answers continued on next page
    - Subparts with distinct marks
    - Crossed out answers
    - Unanswered questions
    - Ambiguous matches flagged for user review
    """
    # 1. Stitch multipage continuations
    stitched_answers = stitch_multipage_continuations(recognized_answers)

    # 2. Normalize candidate questions
    norm_questions = normalize_predicted_questions(predicted_questions)
    if not norm_questions:
        return {
            "matched_items": [],
            "unanswered_questions": [],
            "unmatched_answers": stitched_answers,
            "ambiguous_matches": [],
            "total_predicted_questions": 0,
            "total_answered_questions": 0,
            "total_unanswered_questions": 0
        }

    matched_items: List[Dict[str, Any]] = []
    unanswered_questions: List[Dict[str, Any]] = []
    ambiguous_matches: List[Dict[str, Any]] = []
    assigned_answers: Set[int] = set()

    # Pass 1: For each question in paper, find best matching student answer
    for q_idx, q in enumerate(norm_questions):
        target_num = q["question_number"]
        target_sub = q.get("subpart")
        q_context = f"{q['question_text']} {q.get('topic', '')} {' '.join(q.get('expected_concepts', []))}"

        candidate_matches: List[Tuple[int, float, str]] = []

        for a_idx, ans in enumerate(stitched_answers):
            if a_idx in assigned_answers:
                continue

            a_num, a_sub = parse_question_label(ans.get("question_label", ""))
            sem_overlap = compute_semantic_overlap(ans.get("text", ""), q_context)
            is_crossed = ans.get("is_crossed_out", False)

            # Match score computation
            if a_num is not None and a_num == target_num:
                if target_sub and a_sub:
                    if target_sub == a_sub:
                        # Exact question number AND subpart match
                        score = 0.95 + (0.05 * sem_overlap)
                        reason = f"Explicit question {target_num}({target_sub}) label match."
                    else:
                        # Different subpart for same parent question: do not match
                        continue
                elif a_sub is None and target_sub is None:
                    # Exact parent question match
                    score = 0.85 + (0.15 * sem_overlap)
                    reason = f"Explicit question {target_num} label match."
                else:
                    # One has subpart and other doesn't: partial label match
                    score = 0.70 + (0.30 * sem_overlap)
                    reason = f"Partial question {target_num} label match."

                # Slight penalty if crossed out so that an uncrossed alternative is preferred
                if is_crossed:
                    score *= 0.7

                candidate_matches.append((a_idx, score, reason))

            elif a_num is None:
                # Unlabeled answer: evaluate purely by semantic overlap and position
                pos_bonus = 0.15 if abs(a_idx - q_idx) <= 1 else 0.0
                if sem_overlap >= 0.35:
                    score = (sem_overlap * 0.75) + pos_bonus
                    if is_crossed:
                        score *= 0.7
                    candidate_matches.append((a_idx, score, "Semantic content similarity match."))

        if candidate_matches:
            # Sort by score descending
            candidate_matches.sort(key=lambda x: x[1], reverse=True)
            top_a_idx, top_score, match_reason = candidate_matches[0]

            is_ambiguous = False
            ambiguity_reason = None

            # Ambiguity detection: duplicate question numbers, close score runner-up, or low absolute confidence
            duplicate_q_matches = [
                c for c in candidate_matches 
                if parse_question_label(stitched_answers[c[0]].get("question_label", ""))[0] == target_num
            ]
            if len(duplicate_q_matches) > 1:
                is_ambiguous = True
                ambiguity_reason = f"Multiple student answers claim duplicate question number Q{target_num}."
            elif len(candidate_matches) > 1 and (top_score - candidate_matches[1][1]) < 0.15 and top_score < 0.90:
                is_ambiguous = True
                ambiguity_reason = "Multiple student answers match this question with nearly identical confidence."
            elif top_score < 0.50:
                is_ambiguous = True
                ambiguity_reason = "Low question-matching confidence based on student answer text."

            matched_ans = stitched_answers[top_a_idx]
            assigned_answers.add(top_a_idx)

            ocr_conf = matched_ans.get("confidence", 80.0)
            needs_rev = is_ambiguous or (ocr_conf < 60.0) or matched_ans.get("is_crossed_out", False)

            record = {
                "question": q,
                "question_id": q.get("id"),
                "question_number": q.get("question_number"),
                "subpart": q.get("subpart"),
                "display_number": q.get("display_number", str(q.get("question_number"))),
                "section_name": q.get("section_name", "Section A"),
                "question_text": q.get("question_text", ""),
                "max_marks": float(q.get("marks", 2.0)),
                "topic": q.get("topic", "General"),
                "expected_concepts": q.get("expected_concepts", []),
                "reference_answer": q.get("reference_answer", ""),
                "student_answer": matched_ans.get("text", ""),
                "question_label_written": matched_ans.get("question_label"),
                "pages": matched_ans.get("pages", [1]),
                "ocr_confidence": ocr_conf,
                "matching_confidence": round(top_score * 100.0, 1),
                "match_status": "AMBIGUOUS" if is_ambiguous else "MATCHED",
                "is_continuation": matched_ans.get("is_continuation", False),
                "is_crossed_out": matched_ans.get("is_crossed_out", False),
                "ambiguity_reason": ambiguity_reason,
                "needs_review": needs_rev
            }
            matched_items.append(record)
            if is_ambiguous:
                ambiguous_matches.append(record)
        else:
            # Unanswered question
            unanswered_record = {
                "question": q,
                "question_id": q.get("id"),
                "question_number": q.get("question_number"),
                "subpart": q.get("subpart"),
                "display_number": q.get("display_number", str(q.get("question_number"))),
                "section_name": q.get("section_name", "Section A"),
                "question_text": q.get("question_text", ""),
                "max_marks": float(q.get("marks", 2.0)),
                "topic": q.get("topic", "General"),
                "expected_concepts": q.get("expected_concepts", []),
                "reference_answer": q.get("reference_answer", ""),
                "student_answer": "",
                "question_label_written": None,
                "pages": [],
                "ocr_confidence": 100.0,
                "matching_confidence": 100.0,
                "match_status": "UNANSWERED",
                "is_continuation": False,
                "is_crossed_out": False,
                "ambiguity_reason": None,
                "needs_review": False
            }
            matched_items.append(unanswered_record)
            unanswered_questions.append(unanswered_record)

    # Pass 2: Collect unmatched answers
    unmatched_answers = [
        stitched_answers[i] for i in range(len(stitched_answers)) if i not in assigned_answers
    ]

    total_pred = len(norm_questions)
    total_unans = len(unanswered_questions)
    total_ans = total_pred - total_unans

    return {
        "matched_items": matched_items,
        "unanswered_questions": unanswered_questions,
        "unmatched_answers": unmatched_answers,
        "ambiguous_matches": ambiguous_matches,
        "total_predicted_questions": total_pred,
        "total_answered_questions": total_ans,
        "total_unanswered_questions": total_unans
    }


# Public aliases
extract_question_label = parse_question_label
calculate_content_similarity = compute_semantic_overlap
