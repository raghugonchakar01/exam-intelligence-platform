from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import shutil
import sqlite3
import uuid
import json
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()
from app.descriptive_ai import evaluate_descriptive_answer, process_image_for_ocr
from app.kseab_source import (
    KSEAB_CATALOG, get_pyq_storage_path, download_pdf, extract_text_from_pdf, compute_file_hash,
    extract_questions_from_text
)
from app.pyq_pipeline import build_report, initialize_schema as initialize_pyq_schema, run_sync as run_pyq_sync

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SECRET_KEY = "examinsight-super-secret-key-2025-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("EXAMINSIGHT_DB_PATH", os.path.join(PROJECT_ROOT, "examinsight.db"))
UPLOAD_DIR = os.getenv("EXAMINSIGHT_UPLOAD_DIR", os.path.join(PROJECT_ROOT, "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="ExamInsight AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the SPA)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── DATABASE ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'STUDENT',
            grade INTEGER DEFAULT 10,
            student_id TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS pyq_papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            grade INTEGER NOT NULL,
            subject TEXT NOT NULL,
            year INTEGER,
            board TEXT DEFAULT 'CBSE',
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
            paper_type TEXT
        );

        CREATE TABLE IF NOT EXISTS mock_tests (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            grade INTEGER NOT NULL,
            subject TEXT NOT NULL,
            duration_minutes INTEGER DEFAULT 60,
            total_marks INTEGER DEFAULT 100,
            questions_json TEXT,
            created_by TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS test_attempts (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            test_id TEXT NOT NULL,
            score INTEGER DEFAULT 0,
            total_marks INTEGER DEFAULT 0,
            percentage REAL DEFAULT 0,
            answers_json TEXT,
            completed_at TEXT DEFAULT (datetime('now')),
            time_taken_seconds INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS teacher_notes (
            id TEXT PRIMARY KEY,
            from_teacher_id TEXT NOT NULL,
            to_student_id TEXT NOT NULL,
            note_text TEXT NOT NULL,
            subject TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            is_read INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS student_analytics (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            subject TEXT,
            score REAL,
            total REAL,
            activity_type TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS descriptive_questions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            grade INTEGER DEFAULT 10,
            max_marks REAL DEFAULT 10.0,
            question_text TEXT NOT NULL,
            reference_answer TEXT NOT NULL,
            expected_concepts_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS descriptive_evaluations (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            student_name TEXT,
            question_id TEXT,
            question_text TEXT NOT NULL,
            subject TEXT,
            topic TEXT,
            max_marks REAL DEFAULT 10.0,
            reference_answer TEXT,
            expected_concepts_json TEXT,
            uploaded_file_name TEXT,
            uploaded_file_path TEXT,
            ocr_text TEXT,
            corrected_text TEXT,
            ocr_confidence REAL DEFAULT 0.0,
            score REAL DEFAULT 0.0,
            percentage REAL DEFAULT 0.0,
            rubric_breakdown_json TEXT,
            covered_concepts_json TEXT,
            missing_concepts_json TEXT,
            incorrect_concepts_json TEXT,
            evaluation_confidence TEXT DEFAULT 'High',
            feedback_well TEXT,
            feedback_missing TEXT,
            feedback_correction TEXT,
            feedback_improvement TEXT,
            ai_feedback TEXT,
            teacher_override_score REAL,
            teacher_remarks TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS paper_evaluations (
            id TEXT PRIMARY KEY,
            paper_id TEXT,
            paper_title TEXT,
            subject TEXT,
            grade INTEGER,
            student_id TEXT,
            student_name TEXT,
            total_marks_obtained REAL,
            max_possible_marks REAL,
            percentage REAL,
            grade_label TEXT,
            summary_json TEXT,
            question_evaluations_json TEXT,
            unmatched_answers_json TEXT,
            ambiguous_matches_json TEXT,
            uploaded_file_name TEXT,
            uploaded_file_path TEXT,
            ocr_engine TEXT,
            ocr_confidence REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    # Additive migration for the durable PYQ collection pipeline.  It creates
    # missing tables/columns and indexes but never clears existing records.
    initialize_pyq_schema(conn)
    conn.commit()

    # Seed default users
    default_users = [
        (str(uuid.uuid4()), "Alex Student", "student@examinsight.ai",
         pwd_context.hash("StudentPass123!"), "STUDENT", 10, "STU-10291"),
        (str(uuid.uuid4()), "Ms. Teacher", "teacher@examinsight.ai",
         pwd_context.hash("TeacherPass123!"), "TEACHER", None, None),
        (str(uuid.uuid4()), "Admin User", "admin@examinsight.ai",
         pwd_context.hash("AdminPass123!"), "ADMIN", None, None),
    ]
    for u in default_users:
        try:
            c.execute(
                "INSERT INTO users (id,name,email,password_hash,role,grade,student_id) VALUES (?,?,?,?,?,?,?)",
                u
            )
        except sqlite3.IntegrityError:
            pass

    # Seed sample mock tests
    sample_tests = [
        {
            "id": str(uuid.uuid4()),
            "title": "Grade 10 Mathematics - Algebra Basics",
            "grade": 10,
            "subject": "Mathematics",
            "duration_minutes": 60,
            "total_marks": 40,
            "questions": [
                {"id": 1, "text": "What is the solution to 2x + 4 = 10?", "options": ["x = 2", "x = 3", "x = 4", "x = 5"], "correct": 1, "explanation": "2x = 10 - 4 = 6, so x = 3", "marks": 2, "topic": "Linear Equations"},
                {"id": 2, "text": "Which of the following is a quadratic equation?", "options": ["2x + 3 = 0", "x² + 2x + 1 = 0", "x³ - 1 = 0", "2/x = 4"], "correct": 1, "explanation": "A quadratic equation has degree 2. x² + 2x + 1 = 0 has degree 2.", "marks": 2, "topic": "Quadratic Equations"},
                {"id": 3, "text": "The roots of x² - 5x + 6 = 0 are:", "options": ["2 and 3", "1 and 6", "-2 and -3", "2 and -3"], "correct": 0, "explanation": "Factor: (x-2)(x-3) = 0, so x = 2 or x = 3", "marks": 2, "topic": "Quadratic Equations"},
                {"id": 4, "text": "If the HCF of 12 and 18 is p and LCM is q, then p × q equals:", "options": ["36", "72", "216", "108"], "correct": 2, "explanation": "HCF(12,18)=6, LCM(12,18)=36. p×q = 6×36 = 216. Also HCF×LCM = product of numbers = 12×18 = 216", "marks": 2, "topic": "Number Theory"},
                {"id": 5, "text": "The sum of first n natural numbers is:", "options": ["n(n+1)", "n(n+1)/2", "n²", "n(n-1)/2"], "correct": 1, "explanation": "The formula for sum of first n natural numbers is n(n+1)/2", "marks": 2, "topic": "Arithmetic Progressions"},
            ]
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Grade 10 Science - Physics Fundamentals",
            "grade": 10,
            "subject": "Science",
            "duration_minutes": 45,
            "total_marks": 30,
            "questions": [
                {"id": 1, "text": "The SI unit of force is:", "options": ["Joule", "Newton", "Watt", "Pascal"], "correct": 1, "explanation": "The SI unit of force is Newton (N), named after Sir Isaac Newton.", "marks": 2, "topic": "Forces"},
                {"id": 2, "text": "Which law states that for every action there is an equal and opposite reaction?", "options": ["Newton's First Law", "Newton's Second Law", "Newton's Third Law", "Law of Gravitation"], "correct": 2, "explanation": "Newton's Third Law of Motion states that for every action there is an equal and opposite reaction.", "marks": 2, "topic": "Newton's Laws"},
                {"id": 3, "text": "The speed of light in vacuum is approximately:", "options": ["3×10⁶ m/s", "3×10⁸ m/s", "3×10¹⁰ m/s", "3×10⁴ m/s"], "correct": 1, "explanation": "The speed of light in vacuum is approximately 3×10⁸ m/s (300,000 km/s).", "marks": 2, "topic": "Light"},
            ]
        }
    ]
    for test in sample_tests:
        try:
            questions_json = json.dumps(test["questions"])
            c.execute(
                "INSERT INTO mock_tests (id,title,grade,subject,duration_minutes,total_marks,questions_json,created_by) VALUES (?,?,?,?,?,?,?,?)",
                (test["id"], test["title"], test["grade"], test["subject"],
                 test["duration_minutes"], test["total_marks"], questions_json, "system")
            )
        except sqlite3.IntegrityError:
            pass

    # Seed sample descriptive questions
    sample_desc_questions = [
        (
            "desc_q1_dbms_norm",
            "Explain Normalization in DBMS",
            "Computer Science",
            "Database Management Systems (DBMS)",
            10,
            10.0,
            "Explain normalization in DBMS. Why is it needed, and describe 1NF, 2NF, and 3NF with examples.",
            "Normalization is the process of organizing database tables to reduce data redundancy and eliminate anomalies such as insertion, update, and deletion anomalies. 1NF ensures all attribute values are atomic without repeating groups. 2NF requires 1NF and ensures every non-prime attribute is fully functionally dependent on the candidate key with no partial dependencies. 3NF requires 2NF and ensures no transitive functional dependencies exist. Advantages include improved data integrity, storage efficiency, and consistency.",
            json.dumps(["Definition of Normalization", "Data Redundancy and Anomalies", "First Normal Form (1NF / Atomicity)", "Second Normal Form (2NF / Partial Dependency)", "Third Normal Form (3NF / Transitive Dependency)", "Functional Dependency", "Advantages & Real-world Example"])
        ),
        (
            "desc_q2_chem_comb",
            "Combination Reactions in Chemistry",
            "Science",
            "Chemical Reactions & Equations",
            10,
            10.0,
            "What is a combination reaction? Explain with balanced chemical equations, physical state notations, and examples from everyday life.",
            "A combination reaction is a reaction where two or more reactants combine to form a single product (A + B -> AB). An example is the burning of magnesium in air: 2Mg(s) + O2(g) -> 2MgO(s), forming magnesium oxide with a dazzling white flame. Another example is the slaking of quicklime: CaO(s) + H2O(l) -> Ca(OH)2(aq), which is an exothermic reaction releasing substantial heat. Key aspects are single product synthesis, balanced stoichiometry, and state symbols.",
            json.dumps(["Definition of Combination Reaction", "General representation (A + B -> AB)", "Balanced equation (2Mg + O2 -> 2MgO)", "Physical states notation (s, g, aq)", "Slaking of lime (CaO + H2O -> Ca(OH)2)", "Exothermic heat release"])
        ),
        (
            "desc_q3_phy_newton3",
            "Newton's Third Law of Motion",
            "Science",
            "Forces and Laws of Motion",
            10,
            10.0,
            "State and explain Newton's Third Law of Motion. Why do action and reaction forces not cancel each other out? Give practical examples.",
            "Newton's Third Law of Motion states that for every action, there is an equal and opposite reaction. If body A exerts a force on body B, body B simultaneously exerts an equal force in the opposite direction on body A (Fab = -Fba). Action and reaction forces never cancel each other because they act simultaneously on two different interacting bodies, not on the same body. Practical examples include rocket propulsion where exhausting gas pushes the rocket forward, recoil of a firearm, and swimming where water pushed backward propels the swimmer forward.",
            json.dumps(["Statement of Newton's Third Law", "Action-reaction force pair", "Equal magnitude and opposite direction (Fab = -Fba)", "Forces act on two different bodies (do not cancel)", "Simultaneous action", "Practical examples (rocket propulsion / recoil / swimming)"])
        ),
        (
            "desc_q4_bio_photo",
            "Photosynthesis Mechanism in Plants",
            "Science",
            "Life Processes",
            10,
            10.0,
            "Describe the process of photosynthesis in green plants. Include the balanced chemical equation, the role of stomata, chlorophyll, and key stages.",
            "Photosynthesis is the biochemical process in autotrophs that converts light energy into chemical energy stored in glucose carbohydrates. The balanced chemical equation is 6CO2 + 6H2O + Light + Chlorophyll -> C6H12O6 + 6O2. Chlorophyll in chloroplasts absorbs solar photons. Stomata with guard cells regulate carbon dioxide uptake and oxygen release. The process involves light absorption, water photolysis splitting water into hydrogen and oxygen, and reduction of carbon dioxide into glucose, stored as starch.",
            json.dumps(["Definition of Photosynthesis", "Balanced equation (6CO2 + 6H2O -> C6H12O6 + 6O2)", "Chlorophyll light absorption", "Stomatal gas regulation & guard cells", "Photolysis of water", "Reduction of CO2 to glucose / starch storage"])
        )
    ]
    for dq in sample_desc_questions:
        try:
            c.execute(
                "INSERT INTO descriptive_questions (id,title,subject,topic,grade,max_marks,question_text,reference_answer,expected_concepts_json) VALUES (?,?,?,?,?,?,?,?,?)",
                dq
            )
        except sqlite3.IntegrityError:
            pass


    conn.commit()
    conn.close()

init_db()

# ─── AUTH MODELS ───────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str
    remember_me: bool = False

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "STUDENT"
    grade: Optional[int] = 10

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

# ─── JWT HELPERS ───────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_current_user(authorization: str = None, db: sqlite3.Connection = None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.execute("SELECT * FROM users WHERE id = ?", (payload["sub"],)).fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

# ─── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: sqlite3.Connection = Depends(get_db)):
    user = db.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (req.email,)).fetchone()
    if not user or not pwd_context.verify(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user = dict(user)
    token = create_access_token({"sub": user["id"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "grade": user["grade"],
            "student_id": user["student_id"]
        }
    }

@app.post("/api/auth/register")
def register(req: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    existing = db.execute("SELECT id FROM users WHERE email = ?", (req.email,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    uid = str(uuid.uuid4())
    student_id = f"STU-{str(uuid.uuid4())[:5].upper()}" if req.role == "STUDENT" else None
    db.execute(
        "INSERT INTO users (id,name,email,password_hash,role,grade,student_id) VALUES (?,?,?,?,?,?,?)",
        (uid, req.name, req.email, pwd_context.hash(req.password), req.role, req.grade, student_id)
    )
    db.commit()
    return {"message": "Registration successful", "student_id": student_id}

@app.get("/api/auth/me")
def get_me(authorization: str = None, db: sqlite3.Connection = Depends(get_db)):
    from fastapi import Request
    # This is handled via header in actual requests
    raise HTTPException(status_code=401, detail="Use Authorization header")

# ─── USER ROUTES ───────────────────────────────────────────────────────────────
@app.get("/api/users")
def list_users(authorization: str = None, db: sqlite3.Connection = Depends(get_db)):
    from fastapi import Header
    users = db.execute("SELECT id,name,email,role,grade,student_id,created_at,is_active FROM users").fetchall()
    return [{"id": u["id"], "name": u["name"], "email": u["email"], "role": u["role"],
             "grade": u["grade"], "student_id": u["student_id"], "created_at": u["created_at"],
             "is_active": u["is_active"]} for u in users]

@app.delete("/api/users/{user_id}")
def delete_user(user_id: str, db: sqlite3.Connection = Depends(get_db)):
    db.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
    db.commit()
    return {"message": "User deactivated"}

@app.post("/api/users/{user_id}/reset-password")
def reset_password(user_id: str, db: sqlite3.Connection = Depends(get_db)):
    new_password = "TempPass123!"
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
               (pwd_context.hash(new_password), user_id))
    db.commit()
    return {"message": f"Password reset to: {new_password}"}

# ─── PYQ ROUTES ────────────────────────────────────────────────────────────────
def _safe_paper_path(stored_path: Optional[str]) -> str:
    """Allow only registered files inside this application's project directory."""
    if not stored_path:
        raise HTTPException(status_code=404, detail="Paper file is unavailable")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate = stored_path if os.path.isabs(stored_path) else os.path.join(project_root, stored_path)
    resolved = os.path.abspath(candidate)
    try:
        within_project = os.path.commonpath([project_root, resolved]) == project_root
    except ValueError:
        within_project = False
    if not within_project or not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="Paper file is unavailable")
    return resolved


@app.get("/api/pyq")
def list_pyq(grade: Optional[int] = None, subject: Optional[str] = None,
            db: sqlite3.Connection = Depends(get_db)):
    query = "SELECT * FROM pyq_papers WHERE COALESCE(status, 'READY') = 'READY'"
    params = []
    if grade:
        query += " AND grade = ?"
        params.append(grade)
    if subject and subject != "All":
        query += " AND subject = ?"
        params.append(subject)
    query += " ORDER BY created_at DESC"
    papers = db.execute(query, params).fetchall()
    result = []
    for p in papers:
        d = dict(p)
        result.append(d)
    return result

@app.post("/api/pyq")
async def upload_pyq(
    title: str = Form(...),
    grade: int = Form(...),
    subject: str = Form(...),
    year: Optional[int] = Form(None),
    board: str = Form("CBSE"),
    file: UploadFile = File(...),
    db: sqlite3.Connection = Depends(get_db)
):
    uid = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_name = f"{uid}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    db.execute(
        "INSERT INTO pyq_papers (id,title,grade,subject,year,board,file_name,file_path,file_type,uploaded_by) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (uid, title, grade, subject, year, board, file.filename, file_path,
         file.content_type, "teacher")
    )
    db.commit()
    return {"id": uid, "message": "PYQ paper uploaded successfully"}

@app.get("/api/pyq/{paper_id}/view")
def view_pyq(paper_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Serve PDF inline so browser renders it in an iframe/tab (no download prompt)."""
    paper = db.execute("SELECT * FROM pyq_papers WHERE id = ?", (paper_id,)).fetchone()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    file_path = _safe_paper_path(paper["file_path"])
    from fastapi.responses import Response
    with open(file_path, "rb") as f:
        content = f.read()
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{paper["file_name"]}"'}
    )

@app.get("/api/pyq/{paper_id}/download")
def download_pyq(paper_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Serve PDF as attachment (triggers browser Save dialog)."""
    paper = db.execute("SELECT * FROM pyq_papers WHERE id = ?", (paper_id,)).fetchone()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    file_path = _safe_paper_path(paper["file_path"])
    db.execute("UPDATE pyq_papers SET download_count = download_count + 1 WHERE id = ?", (paper_id,))
    db.commit()
    return FileResponse(file_path, filename=paper["file_name"])

@app.delete("/api/pyq/{paper_id}")
def delete_pyq(paper_id: str, db: sqlite3.Connection = Depends(get_db)):
    paper = db.execute("SELECT * FROM pyq_papers WHERE id = ?", (paper_id,)).fetchone()
    if paper and paper["file_path"]:
        try:
            os.remove(_safe_paper_path(paper["file_path"]))
        except HTTPException:
            pass
    db.execute("DELETE FROM pyq_papers WHERE id = ?", (paper_id,))
    db.commit()
    return {"message": "Paper deleted"}

# ─── AI QUESTION PAPER GENERATOR ─────────────────────────────────────────────

def _build_blueprint(total_marks: int) -> list:
    """
    Distributes total_marks into exam sections with realistic question patterns.
    Returns list of {section, type, marks_per_q, count}.
    """
    sections = []
    remaining = total_marks

    # Section A: 1-mark MCQs — 20% of marks (min 4, max 20)
    sec_a_count = max(4, min(20, round(total_marks * 0.20)))
    sec_a_count = round(sec_a_count / 1)  # 1 mark each
    if sec_a_count > 0:
        sections.append({"section": "Section A", "type": "MCQ / Very Short Answer", "marks_per_q": 1, "count": sec_a_count})
        remaining -= sec_a_count

    # Section B: 2-mark questions — 20% of marks
    sec_b_count = max(2, min(8, round(total_marks * 0.20 / 2)))
    if remaining > 0 and sec_b_count > 0:
        actual = min(sec_b_count, remaining // 2)
        if actual > 0:
            sections.append({"section": "Section B", "type": "Short Answer (2 marks)", "marks_per_q": 2, "count": actual})
            remaining -= actual * 2

    # Section C: 3-mark questions — 25% of marks
    sec_c_count = max(2, min(8, round(total_marks * 0.25 / 3)))
    if remaining > 0 and sec_c_count > 0:
        actual = min(sec_c_count, remaining // 3)
        if actual > 0:
            sections.append({"section": "Section C", "type": "Short Answer (3 marks)", "marks_per_q": 3, "count": actual})
            remaining -= actual * 3

    # Section D: 5-mark questions — consume what's left (split into 5-mark chunks)
    if remaining >= 5:
        sec_d_count = remaining // 5
        if sec_d_count > 0:
            sections.append({"section": "Section D", "type": "Long Answer (5 marks)", "marks_per_q": 5, "count": sec_d_count})
            remaining -= sec_d_count * 5

    # Section E: leftover marks absorbed as a final question
    if remaining > 0:
        sections.append({"section": "Section E", "type": "Case-Based / Analytical", "marks_per_q": remaining, "count": 1})

    return sections


def _validate_paper(paper: dict, expected_marks: int) -> list:
    """Returns list of error strings; empty list means valid."""
    errors = []
    if "sections" not in paper or not paper["sections"]:
        errors.append("No sections in response")
        return errors
    total = 0
    for sec in paper["sections"]:
        for q in sec.get("questions", []):
            m = q.get("marks", 0)
            if not m:
                errors.append(f"Question missing marks in {sec.get('name','?')}")
            if not q.get("question", "").strip():
                errors.append("Empty question text found")
            total += m
    if abs(total - expected_marks) > 1:
        errors.append(f"Total marks mismatch: got {total}, expected {expected_marks}")
    return errors


class GeneratePaperRequest(BaseModel):
    grade: int
    subject: str
    board: str = "KSEAB / State Board"
    total_marks: int = 80
    difficulty: str = "Moderate"
    question_pattern: str = "Mixed"
    gemini_api_key: Optional[str] = None
    pyq_context: Optional[str] = None   # Deprecated: server retrieves trusted context itself.
    study_material: Optional[str] = None  # Optional study material text
    chapters: List[str] = Field(default_factory=list)
    question_types: List[str] = Field(default_factory=list)
    paper_structure: Optional[str] = None


@app.post("/api/generate-paper")
def generate_paper(req: GeneratePaperRequest, db: sqlite3.Connection = Depends(get_db)):
    """Generate only from server-side verified PYQs/syllabus and persist provenance."""
    from app.paper_service import GenerationUnavailable, generate_question_paper

    try:
        paper = generate_question_paper(
            grade=req.grade, subject=req.subject, board=req.board, total_marks=req.total_marks,
            difficulty=req.difficulty, question_pattern=req.paper_structure or req.question_pattern,
            chapters=req.chapters, question_types=req.question_types, gemini_api_key=req.gemini_api_key,
        )
    except GenerationUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Question paper generation failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Question-paper generation is temporarily unavailable. No paper was fabricated.")

    db.execute(
        """INSERT INTO generated_papers
        (id,grade,subject,total_marks,provider,request_json,paper_json,source_question_ids_json,created_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (str(uuid.uuid4()), req.grade, req.subject, req.total_marks, paper.get("provider", "Unknown"),
         req.model_dump_json(), json.dumps(paper, ensure_ascii=False),
         json.dumps({"questions": paper.get("_source_question_ids", []), "syllabus": paper.get("_syllabus_source_ids", [])}),
         datetime.utcnow().isoformat()),
    )
    db.commit()
    return paper


# ─── KSEAB OFFICIAL PYQ SYNC ROUTE ───────────────────────────────────────────
@app.post("/api/admin/sync-kseab-pyqs")
def sync_kseab_pyqs(retry_failed: bool = False, db: sqlite3.Connection = Depends(get_db)):
    """Run the persisted discovery/download/extraction workflow.

    No candidate is dropped on failure: the response is an audit report rather
    than a misleading count of only successful downloads.
    """
    return run_pyq_sync(db, retry_failed=retry_failed)


@app.get("/api/admin/pyq-collection-report")
def pyq_collection_report(db: sqlite3.Connection = Depends(get_db)):
    return build_report(db)


@app.get("/api/admin/pyq-documents")
def list_pyq_documents(status_filter: Optional[str] = None, db: sqlite3.Connection = Depends(get_db)):
    query = "SELECT * FROM pyq_documents"
    params = []
    if status_filter:
        query += " WHERE status = ?"
        params.append(status_filter)
    query += " ORDER BY discovered_at DESC"
    return [dict(row) for row in db.execute(query, params).fetchall()]

@app.get("/api/pyq/questions")
def get_pyq_questions(grade: Optional[int] = None, subject: Optional[str] = None,
                      year: Optional[int] = None, limit: int = 50,
                      db: sqlite3.Connection = Depends(get_db)):
    """
    Returns authentic questions parsed from imported PYQs for the AI Question Paper Generator.
    """
    query = "SELECT * FROM pyq_questions WHERE 1=1"
    params = []
    if grade:
        query += " AND grade = ?"
        params.append(grade)
    if subject and subject != "All" and subject != "All Subjects":
        query += " AND subject LIKE ?"
        params.append(f"%{subject}%")
    if year:
        query += " AND year = ?"
        params.append(year)
    query += " ORDER BY year DESC, question_number ASC LIMIT ?"
    params.append(limit)

    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ─── MOCK TEST ROUTES ──────────────────────────────────────────────────────────
@app.get("/api/tests")
def list_tests(grade: Optional[int] = None, subject: Optional[str] = None,
              db: sqlite3.Connection = Depends(get_db)):
    query = "SELECT * FROM mock_tests WHERE is_active = 1"
    params = []
    if grade:
        query += " AND grade = ?"
        params.append(grade)
    if subject:
        query += " AND subject = ?"
        params.append(subject)
    tests = db.execute(query, params).fetchall()
    result = []
    for t in tests:
        d = dict(t)
        d["questions"] = json.loads(d["questions_json"]) if d["questions_json"] else []
        del d["questions_json"]
        result.append(d)
    return result

@app.post("/api/tests")
def create_test(test: dict, db: sqlite3.Connection = Depends(get_db)):
    uid = str(uuid.uuid4())
    db.execute(
        "INSERT INTO mock_tests (id,title,grade,subject,duration_minutes,total_marks,questions_json,created_by) VALUES (?,?,?,?,?,?,?,?)",
        (uid, test.get("title"), test.get("grade"), test.get("subject"),
         test.get("duration_minutes", 60), test.get("total_marks", 100),
         json.dumps(test.get("questions", [])), test.get("created_by", "teacher"))
    )
    db.commit()
    return {"id": uid, "message": "Test created"}

@app.post("/api/tests/{test_id}/submit")
def submit_test(test_id: str, attempt: dict, db: sqlite3.Connection = Depends(get_db)):
    uid = str(uuid.uuid4())
    percentage = (attempt.get("score", 0) / attempt.get("total_marks", 1)) * 100
    db.execute(
        "INSERT INTO test_attempts (id,student_id,test_id,score,total_marks,percentage,answers_json,time_taken_seconds) VALUES (?,?,?,?,?,?,?,?)",
        (uid, attempt.get("student_id"), test_id, attempt.get("score", 0),
         attempt.get("total_marks", 100), percentage,
         json.dumps(attempt.get("answers", [])), attempt.get("time_taken", 0))
    )
    db.commit()
    return {"id": uid, "percentage": percentage}

@app.get("/api/tests/{test_id}/attempts")
def get_test_attempts(test_id: str, student_id: Optional[str] = None,
                      db: sqlite3.Connection = Depends(get_db)):
    query = "SELECT * FROM test_attempts WHERE test_id = ?"
    params = [test_id]
    if student_id:
        query += " AND student_id = ?"
        params.append(student_id)
    attempts = db.execute(query, params).fetchall()
    return [dict(a) for a in attempts]

# ─── TEACHER NOTES ─────────────────────────────────────────────────────────────
@app.post("/api/notes")
def send_note(note: dict, db: sqlite3.Connection = Depends(get_db)):
    uid = str(uuid.uuid4())
    db.execute(
        "INSERT INTO teacher_notes (id,from_teacher_id,to_student_id,note_text,subject) VALUES (?,?,?,?,?)",
        (uid, note.get("from_teacher_id"), note.get("to_student_id"),
         note.get("note_text"), note.get("subject"))
    )
    db.commit()
    return {"id": uid, "message": "Note sent"}

@app.get("/api/notes/{student_id}")
def get_student_notes(student_id: str, db: sqlite3.Connection = Depends(get_db)):
    notes = db.execute(
        "SELECT * FROM teacher_notes WHERE to_student_id = ? ORDER BY created_at DESC",
        (student_id,)
    ).fetchall()
    return [dict(n) for n in notes]

@app.get("/api/students/search")
def search_student(student_id: str = None, db: sqlite3.Connection = Depends(get_db)):
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id required")
    user = db.execute(
        "SELECT id,name,email,grade,student_id FROM users WHERE student_id = ? OR id = ?",
        (student_id, student_id)
    ).fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="Student not found")
    attempts = db.execute(
        "SELECT ta.*, mt.title, mt.subject FROM test_attempts ta "
        "JOIN mock_tests mt ON ta.test_id = mt.id "
        "WHERE ta.student_id = ? ORDER BY ta.completed_at DESC LIMIT 10",
        (user["id"],)
    ).fetchall()
    notes = db.execute(
        "SELECT * FROM teacher_notes WHERE to_student_id = ? ORDER BY created_at DESC",
        (user["id"],)
    ).fetchall()
    return {
        "student": dict(user),
        "recent_attempts": [dict(a) for a in attempts],
        "notes": [dict(n) for n in notes]
    }

# ─── ANALYTICS ─────────────────────────────────────────────────────────────────
@app.get("/api/analytics/{student_id}")
def get_analytics(student_id: str, db: sqlite3.Connection = Depends(get_db)):
    attempts = db.execute(
        "SELECT ta.*, mt.subject, mt.title FROM test_attempts ta "
        "JOIN mock_tests mt ON ta.test_id = mt.id "
        "WHERE ta.student_id = ? ORDER BY ta.completed_at DESC",
        (student_id,)
    ).fetchall()
    total = len(attempts)
    avg_score = sum(a["percentage"] for a in attempts) / total if total else 0
    subject_breakdown = {}
    for a in attempts:
        s = a["subject"]
        if s not in subject_breakdown:
            subject_breakdown[s] = {"total": 0, "count": 0}
        subject_breakdown[s]["total"] += a["percentage"]
        subject_breakdown[s]["count"] += 1
    for s in subject_breakdown:
        subject_breakdown[s]["avg"] = subject_breakdown[s]["total"] / subject_breakdown[s]["count"]
    return {
        "total_attempts": total,
        "average_score": round(avg_score, 2),
        "subject_breakdown": subject_breakdown,
        "recent_attempts": [dict(a) for a in attempts[:10]]
    }

# ─── ADMIN STATS ───────────────────────────────────────────────────────────────
@app.get("/api/admin/stats")
def admin_stats(db: sqlite3.Connection = Depends(get_db)):
    total_users = db.execute("SELECT COUNT(*) as c FROM users WHERE is_active=1").fetchone()["c"]
    total_students = db.execute("SELECT COUNT(*) as c FROM users WHERE role='STUDENT' AND is_active=1").fetchone()["c"]
    total_teachers = db.execute("SELECT COUNT(*) as c FROM users WHERE role='TEACHER' AND is_active=1").fetchone()["c"]
    total_pyq = db.execute("SELECT COUNT(*) as c FROM pyq_papers").fetchone()["c"]
    total_tests = db.execute("SELECT COUNT(*) as c FROM mock_tests WHERE is_active=1").fetchone()["c"]
    total_attempts = db.execute("SELECT COUNT(*) as c FROM test_attempts").fetchone()["c"]
    return {
        "total_users": total_users,
        "total_students": total_students,
        "total_teachers": total_teachers,
        "total_pyq_papers": total_pyq,
        "total_mock_tests": total_tests,
        "total_attempts": total_attempts
    }

# ─── DESCRIPTIVE ANSWER & OCR ROUTES ───────────────────────────────────────────
class DescriptiveEvaluateRequest(BaseModel):
    student_id: str
    student_name: Optional[str] = "Student"
    question_id: Optional[str] = None
    question_text: str
    subject: Optional[str] = "General"
    topic: Optional[str] = "General"
    max_marks: Optional[float] = 10.0
    reference_answer: Optional[str] = ""
    expected_concepts: Optional[List[str]] = []
    ocr_text: Optional[str] = ""
    corrected_text: str
    ocr_confidence: Optional[float] = 85.0
    file_name: Optional[str] = ""
    file_path: Optional[str] = ""
    gemini_api_key: Optional[str] = None

class TeacherOverrideRequest(BaseModel):
    override_score: float
    teacher_remarks: Optional[str] = ""

@app.get("/api/descriptive/questions")
def get_descriptive_questions(subject: Optional[str] = None, db: sqlite3.Connection = Depends(get_db)):
    query = "SELECT * FROM descriptive_questions WHERE 1=1"
    params = []
    if subject and subject != "All":
        query += " AND subject LIKE ?"
        params.append(f"%{subject}%")
    query += " ORDER BY created_at ASC"
    rows = db.execute(query, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["expected_concepts"] = json.loads(d["expected_concepts_json"]) if d["expected_concepts_json"] else []
        del d["expected_concepts_json"]
        result.append(d)
    return result

@app.post("/api/descriptive/upload-ocr")
async def upload_and_ocr(
    file: UploadFile = File(...),
    vision_api_key: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_exts = {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".bmp", ".tiff", ".tif"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Supported formats: JPG, JPEG, PNG, PDF."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")

    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit (15 MB).")

    uid = str(uuid.uuid4())
    stored_filename = f"{uid}{ext}"
    stored_filepath = os.path.join(UPLOAD_DIR, stored_filename)
    with open(stored_filepath, "wb") as f:
        f.write(file_bytes)

    try:
        ocr_result = process_image_for_ocr(
            file_bytes, file.filename, vision_api_key=vision_api_key, gemini_api_key=gemini_api_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    return {
        "id": uid,
        "file_name": file.filename,
        "file_path": stored_filepath,
        "ocr_text": ocr_result.get("text", ""),
        "ocr_confidence": ocr_result.get("confidence", 80.0),
        "engine": ocr_result.get("engine", "Handwriting OCR Engine"),
        "is_fallback": ocr_result.get("is_fallback", False),
        "notice": ocr_result.get("notice"),
        "confidence_warning": ocr_result.get("confidence_warning"),
        "recognized_answers": ocr_result.get("recognized_answers", []),
        "pages": ocr_result.get("pages", []),
        "total_pages": ocr_result.get("total_pages", 1),
        "student_name": ocr_result.get("student_name"),
        "student_id": ocr_result.get("student_id")
    }


class EvaluatePaperRequest(BaseModel):
    paper_id: Optional[str] = None
    paper_title: Optional[str] = "Predicted Examination Paper"
    subject: Optional[str] = "General"
    grade: Optional[int] = 10
    board: Optional[str] = "KSEAB"
    student_id: Optional[str] = "STU-10291"
    student_name: Optional[str] = "Alex Student"
    matched_items: List[Dict[str, Any]]
    unmatched_answers: Optional[List[Dict[str, Any]]] = []
    ambiguous_matches: Optional[List[Dict[str, Any]]] = []
    uploaded_file_name: Optional[str] = ""
    uploaded_file_path: Optional[str] = ""
    ocr_confidence: Optional[float] = 85.0
    ocr_engine: Optional[str] = "Handwriting OCR Engine"
    gemini_api_key: Optional[str] = None


@app.get("/api/descriptive/predicted-papers")
def get_predicted_papers(db: sqlite3.Connection = Depends(get_db)):
    """Returns available predicted and generated papers from database."""
    rows = db.execute(
        "SELECT id, grade, subject, total_marks, provider, created_at, paper_json FROM generated_papers ORDER BY created_at DESC LIMIT 25"
    ).fetchall()
    results = []
    for r in rows:
        paper_data = {}
        try:
            paper_data = json.loads(r["paper_json"]) if r["paper_json"] else {}
        except Exception:
            pass
        sec_count = len(paper_data.get("sections", []))
        q_count = sum(len(s.get("questions", [])) for s in paper_data.get("sections", []))
        results.append({
            "id": r["id"],
            "title": paper_data.get("title", f"Class {r['grade']} {r['subject']} Practice Paper"),
            "subject": r["subject"],
            "grade": r["grade"],
            "total_marks": r["total_marks"],
            "provider": r["provider"],
            "created_at": r["created_at"],
            "section_count": sec_count,
            "question_count": q_count,
            "sections": paper_data.get("sections", [])
        })
    return results


@app.post("/api/descriptive/extract-and-match")
async def extract_and_match(
    file: UploadFile = File(...),
    paper_id: Optional[str] = Form(None),
    questions_json: Optional[str] = Form(None),
    vision_api_key: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    db: sqlite3.Connection = Depends(get_db)
):
    """End-to-end handwriting recognition & answer matching against predicted question paper:
    
    1. Preprocesses uploaded image/PDF (denoising, deskewing, CLAHE, layout analysis)
    2. Transcribes handwriting via multi-tier HTR
    3. Extracts question numbers and subquestion labels
    4. Matches answers to questions in the predicted paper (with continuation & out-of-order handling)
    5. Flags ambiguous matches and identifies unanswered questions
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_exts = {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".bmp", ".tiff", ".tif"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{ext}'. Supported: JPG, PNG, PDF.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 15 MB limit.")

    uid = str(uuid.uuid4())
    stored_filename = f"{uid}{ext}"
    stored_filepath = os.path.join(UPLOAD_DIR, stored_filename)
    with open(stored_filepath, "wb") as f:
        f.write(file_bytes)

    # 1. Resolve candidate questions from predicted paper or parameters
    from app.answer_matcher import match_answers_to_questions
    from app.handwriting_ocr import recognize_handwriting

    paper_meta = {"id": paper_id, "title": "Predicted Examination Paper", "subject": "General", "grade": 10, "board": "KSEAB"}
    questions = []

    if paper_id:
        row = db.execute("SELECT paper_json, subject, grade FROM generated_papers WHERE id = ?", (paper_id,)).fetchone()
        if row and row["paper_json"]:
            try:
                p_data = json.loads(row["paper_json"])
                paper_meta["title"] = p_data.get("title", f"Class {row['grade']} {row['subject']} Practice Paper")
                paper_meta["subject"] = row["subject"]
                paper_meta["grade"] = row["grade"]
                paper_meta["board"] = p_data.get("board", "KSEAB")
                for sec in p_data.get("sections", []):
                    for q in sec.get("questions", []):
                        questions.append({
                            "id": q.get("source_question_id") or q.get("id") or str(uuid.uuid4())[:8],
                            "question_number": len(questions) + 1,
                            "section_name": sec.get("name", "Section"),
                            "question_type": sec.get("question_type", "Descriptive"),
                            "question_text": q.get("question", ""),
                            "marks": float(q.get("marks", 2.0)),
                            "topic": q.get("topic", "General"),
                            "expected_concepts": q.get("expected_concepts", []),
                            "reference_answer": q.get("reference_answer", "")
                        })
            except Exception as e:
                logger.warning(f"Error parsing paper_json for paper {paper_id}: {e}")

    if not questions and questions_json:
        try:
            questions = json.loads(questions_json)
        except Exception:
            pass

    if not questions:
        # Fallback to database descriptive questions
        d_rows = db.execute("SELECT * FROM descriptive_questions LIMIT 10").fetchall()
        for idx, dr in enumerate(d_rows, start=1):
            exp = json.loads(dr["expected_concepts_json"]) if dr["expected_concepts_json"] else []
            questions.append({
                "id": dr["id"],
                "question_number": idx,
                "section_name": "Section A",
                "question_type": "Descriptive",
                "question_text": dr["question_text"],
                "marks": float(dr["max_marks"] or 10.0),
                "topic": dr["topic"],
                "expected_concepts": exp,
                "reference_answer": dr["reference_answer"]
            })

    # 2. Run Handwriting Recognition
    try:
        ocr_result = recognize_handwriting(
            file_bytes, file.filename, vision_api_key=vision_api_key, gemini_api_key=gemini_api_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Handwriting OCR processing failed: {str(e)}")

    # 3. Match Answers to Questions
    matching_result = match_answers_to_questions(ocr_result.get("recognized_answers", []), questions)

    return {
        "id": uid,
        "file_name": file.filename,
        "file_path": stored_filepath,
        "ocr_text": ocr_result.get("combined_text", ""),
        "ocr_confidence": ocr_result.get("ocr_confidence", 80.0),
        "engine": ocr_result.get("ocr_engine", "Handwriting OCR Engine"),
        "ocr_engine": ocr_result.get("ocr_engine", "Handwriting OCR Engine"),
        "is_fallback": ocr_result.get("is_fallback", False),
        "notice": ocr_result.get("notice"),
        "confidence_warning": ocr_result.get("confidence_warning"),
        "total_pages": ocr_result.get("total_pages", 1),
        "student_name": ocr_result.get("student_name"),
        "student_id": ocr_result.get("student_id"),
        "paper_metadata": paper_meta,
        "matched_items": matching_result["matched_items"],
        "unanswered_questions": matching_result["unanswered_questions"],
        "unmatched_answers": matching_result["unmatched_answers"],
        "ambiguous_matches": matching_result["ambiguous_matches"],
        "total_predicted_questions": matching_result["total_predicted_questions"],
        "total_answered_questions": matching_result["total_answered_questions"],
        "total_unanswered_questions": matching_result["total_unanswered_questions"]
    }


@app.post("/api/descriptive/evaluate-paper")
def evaluate_paper(
    req: EvaluatePaperRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    """Evaluates all questions from a reviewed matching result and produces a structured evaluation report."""
    from app.answer_evaluator import generate_paper_evaluation_report

    matching_res = {
        "matched_items": req.matched_items,
        "unmatched_answers": req.unmatched_answers or [],
        "ambiguous_matches": req.ambiguous_matches or [],
        "ocr_engine": req.ocr_engine,
        "ocr_confidence": req.ocr_confidence
    }
    paper_meta = {
        "id": req.paper_id,
        "title": req.paper_title,
        "subject": req.subject,
        "grade": req.grade,
        "board": req.board
    }
    student_info = {
        "id": req.student_id or "STU-10291",
        "name": req.student_name or "Alex Student"
    }

    report = generate_paper_evaluation_report(
        matching_res,
        paper_metadata=paper_meta,
        student_info=student_info,
        gemini_api_key=req.gemini_api_key
    )

    # Persist report in database
    summary = report["summary"]
    try:
        db.execute(
            """
            INSERT INTO paper_evaluations (
                id, paper_id, paper_title, subject, grade, student_id, student_name,
                total_marks_obtained, max_possible_marks, percentage, grade_label,
                summary_json, question_evaluations_json, unmatched_answers_json, ambiguous_matches_json,
                uploaded_file_name, uploaded_file_path, ocr_engine, ocr_confidence
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                report["report_id"], req.paper_id, req.paper_title, req.subject, req.grade,
                student_info["id"], student_info["name"], summary["total_marks_obtained"],
                summary["max_possible_marks"], summary["percentage"], summary["grade"],
                json.dumps(summary), json.dumps(report["question_evaluations"]),
                json.dumps(report["unmatched_answers"]), json.dumps(report["ambiguous_matches"]),
                req.uploaded_file_name, req.uploaded_file_path, req.ocr_engine, req.ocr_confidence
            )
        )

        # Also store individual question items in descriptive_evaluations for history views
        for qe in report["question_evaluations"]:
            sub_id = str(uuid.uuid4())
            db.execute(
                """
                INSERT INTO descriptive_evaluations (
                    id, student_id, student_name, question_id, question_text, subject, topic,
                    max_marks, reference_answer, expected_concepts_json, uploaded_file_name,
                    uploaded_file_path, ocr_text, corrected_text, ocr_confidence, score,
                    percentage, rubric_breakdown_json, covered_concepts_json, missing_concepts_json,
                    incorrect_concepts_json, evaluation_confidence, feedback_well, feedback_missing,
                    feedback_correction, feedback_improvement, ai_feedback
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sub_id, student_info["id"], student_info["name"], qe.get("question_id"),
                    qe.get("question_text"), req.subject, req.subject, qe.get("max_marks"),
                    qe.get("reference_answer"), json.dumps(qe.get("expected_concepts", [])),
                    req.uploaded_file_name, req.uploaded_file_path, qe.get("recognized_student_answer"),
                    qe.get("recognized_student_answer"), qe.get("ocr_confidence", 80.0),
                    qe.get("marks_awarded"), qe.get("percentage"), json.dumps(qe.get("rubric_breakdown", {})),
                    json.dumps(qe.get("covered_concepts", [])), json.dumps(qe.get("missing_concepts", [])),
                    json.dumps(qe.get("incorrect_concepts", [])), qe.get("evaluation_confidence"),
                    qe.get("feedback_well"), qe.get("feedback_missing"), qe.get("feedback_correction"),
                    qe.get("feedback_improvement"), qe.get("ai_feedback")
                )
            )

        db.commit()
    except Exception as dbe:
        logger.warning(f"Error persisting paper evaluation report: {dbe}")

    return report


@app.get("/api/descriptive/paper-reports/{student_id}")
def get_student_paper_reports(student_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Returns saved paper evaluation reports for student."""
    rows = db.execute(
        "SELECT * FROM paper_evaluations WHERE student_id = ? ORDER BY created_at DESC", (student_id,)
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["summary"] = json.loads(d["summary_json"]) if d.get("summary_json") else {}
        d["question_evaluations"] = json.loads(d["question_evaluations_json"]) if d.get("question_evaluations_json") else []
        result.append(d)
    return result


@app.get("/api/descriptive/paper-report/{report_id}")
def get_paper_report_by_id(report_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Retrieves full evaluation report by ID."""
    row = db.execute("SELECT * FROM paper_evaluations WHERE id = ?", (report_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Evaluation report not found.")
    d = dict(row)
    d["report_id"] = d["id"]
    d["summary"] = json.loads(d["summary_json"]) if d.get("summary_json") else {}
    d["question_evaluations"] = json.loads(d["question_evaluations_json"]) if d.get("question_evaluations_json") else []
    d["unmatched_answers"] = json.loads(d["unmatched_answers_json"]) if d.get("unmatched_answers_json") else []
    d["ambiguous_matches"] = json.loads(d["ambiguous_matches_json"]) if d.get("ambiguous_matches_json") else []
    return d


@app.get("/api/descriptive/paper/{paper_id}")
def get_predicted_paper_by_id(paper_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Retrieves full details of a specific generated question paper."""
    row = db.execute("SELECT id, grade, subject, total_marks, provider, created_at, paper_json FROM generated_papers WHERE id = ?", (paper_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Question paper not found.")
    p_data = json.loads(row["paper_json"]) if row["paper_json"] else {}
    return {
        "id": row["id"],
        "title": p_data.get("title", f"Class {row['grade']} {row['subject']} Practice Paper"),
        "subject": row["subject"],
        "grade": row["grade"],
        "total_marks": row["total_marks"],
        "provider": row["provider"],
        "created_at": row["created_at"],
        "sections": p_data.get("sections", []),
        "paper_json": p_data
    }


@app.get("/api/descriptive/paper-evaluations")
def get_all_paper_evaluations(student_id: Optional[str] = None, db: sqlite3.Connection = Depends(get_db)):
    """Lists saved paper evaluation reports."""
    if student_id:
        rows = db.execute("SELECT * FROM paper_evaluations WHERE student_id = ? ORDER BY created_at DESC", (student_id,)).fetchall()
    else:
        rows = db.execute("SELECT * FROM paper_evaluations ORDER BY created_at DESC LIMIT 50").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["report_id"] = d["id"]
        d["summary"] = json.loads(d["summary_json"]) if d.get("summary_json") else {}
        d["question_evaluations"] = json.loads(d["question_evaluations_json"]) if d.get("question_evaluations_json") else []
        result.append(d)
    return result


@app.post("/api/descriptive/paper-evaluations/{eval_id}/override")
def override_paper_evaluation(
    eval_id: str,
    req: TeacherOverrideRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    """Allows a teacher to adjust total score and leave remarks on an evaluation report."""
    row = db.execute("SELECT * FROM paper_evaluations WHERE id = ?", (eval_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Paper evaluation report not found.")

    summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
    max_marks = float(summary.get("max_possible_marks", row["max_possible_marks"]))
    new_score = min(max_marks, max(0.0, float(req.override_score)))
    new_pct = round((new_score / max_marks * 100.0), 1) if max_marks > 0 else 0.0

    summary["total_marks_obtained"] = new_score
    summary["percentage"] = new_pct
    summary["teacher_override"] = True
    summary["teacher_remarks"] = req.teacher_remarks

    db.execute(
        """
        UPDATE paper_evaluations
        SET total_marks_obtained = ?, percentage = ?, summary_json = ?
        WHERE id = ?
        """,
        (new_score, new_pct, json.dumps(summary), eval_id)
    )
    db.commit()
    return {"message": "Evaluation updated with teacher override.", "new_score": new_score, "percentage": new_pct}



@app.post("/api/descriptive/evaluate")
def evaluate_descriptive(
    req: DescriptiveEvaluateRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    if not req.corrected_text or not req.corrected_text.strip():
        raise HTTPException(status_code=400, detail="Answer text cannot be blank.")

    ref_answer = req.reference_answer or ""
    exp_concepts = req.expected_concepts or []
    if req.question_id and (not ref_answer or not exp_concepts):
        q_row = db.execute("SELECT * FROM descriptive_questions WHERE id = ?", (req.question_id,)).fetchone()
        if q_row:
            if not ref_answer:
                ref_answer = q_row["reference_answer"]
            if not exp_concepts and q_row["expected_concepts_json"]:
                exp_concepts = json.loads(q_row["expected_concepts_json"])

    eval_result = evaluate_descriptive_answer(
        question_text=req.question_text,
        reference_answer=ref_answer,
        expected_concepts=exp_concepts,
        student_answer=req.corrected_text,
        max_marks=req.max_marks or 10.0,
        ocr_confidence=req.ocr_confidence or 85.0,
        gemini_api_key=req.gemini_api_key
    )

    eval_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO descriptive_evaluations (
            id, student_id, student_name, question_id, question_text, subject, topic,
            max_marks, reference_answer, expected_concepts_json, uploaded_file_name,
            uploaded_file_path, ocr_text, corrected_text, ocr_confidence, score,
            percentage, rubric_breakdown_json, covered_concepts_json, missing_concepts_json,
            incorrect_concepts_json, evaluation_confidence, feedback_well, feedback_missing,
            feedback_correction, feedback_improvement, ai_feedback
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            eval_id, req.student_id, req.student_name, req.question_id, req.question_text,
            req.subject, req.topic, eval_result["max_marks"], ref_answer,
            json.dumps(exp_concepts), req.file_name, req.file_path, req.ocr_text,
            req.corrected_text, req.ocr_confidence, eval_result["score"], eval_result["percentage"],
            json.dumps(eval_result["rubric_breakdown"]), json.dumps(eval_result["covered_concepts"]),
            json.dumps(eval_result["missing_concepts"]), json.dumps(eval_result["incorrect_concepts"]),
            eval_result["evaluation_confidence"], eval_result["feedback_well"],
            eval_result["feedback_missing"], eval_result["feedback_correction"],
            eval_result["feedback_improvement"], eval_result["ai_feedback"]
        )
    )

    try:
        db.execute(
            """
            INSERT INTO student_analytics (id, student_id, date, subject, score, total, activity_type)
            VALUES (?, ?, date('now'), ?, ?, ?, 'DESCRIPTIVE_EVALUATION')
            """,
            (str(uuid.uuid4()), req.student_id, req.subject, eval_result["score"], eval_result["max_marks"])
        )
    except Exception:
        pass

    db.commit()

    saved_record = {
        "id": eval_id,
        "student_id": req.student_id,
        "student_name": req.student_name,
        "question_id": req.question_id,
        "question_text": req.question_text,
        "subject": req.subject,
        "topic": req.topic,
        "max_marks": eval_result["max_marks"],
        "reference_answer": ref_answer,
        "expected_concepts": exp_concepts,
        "uploaded_file_name": req.file_name,
        "uploaded_file_path": req.file_path,
        "ocr_text": req.ocr_text,
        "corrected_text": req.corrected_text,
        "ocr_confidence": req.ocr_confidence,
        "score": eval_result["score"],
        "percentage": eval_result["percentage"],
        "rubric_breakdown": eval_result["rubric_breakdown"],
        "covered_concepts": eval_result["covered_concepts"],
        "missing_concepts": eval_result["missing_concepts"],
        "incorrect_concepts": eval_result["incorrect_concepts"],
        "evaluation_confidence": eval_result["evaluation_confidence"],
        "feedback_well": eval_result["feedback_well"],
        "feedback_missing": eval_result["feedback_missing"],
        "feedback_correction": eval_result["feedback_correction"],
        "feedback_improvement": eval_result["feedback_improvement"],
        "ai_feedback": eval_result["ai_feedback"],
        "evaluated_by": eval_result.get("evaluated_by", "AI Engine"),
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    return saved_record

@app.get("/api/descriptive/history/{student_id}")
def get_student_descriptive_history(student_id: str, db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        "SELECT * FROM descriptive_evaluations WHERE student_id = ? ORDER BY created_at DESC",
        (student_id,)
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["expected_concepts"] = json.loads(d["expected_concepts_json"]) if d["expected_concepts_json"] else []
        d["rubric_breakdown"] = json.loads(d["rubric_breakdown_json"]) if d["rubric_breakdown_json"] else {}
        d["covered_concepts"] = json.loads(d["covered_concepts_json"]) if d["covered_concepts_json"] else []
        d["missing_concepts"] = json.loads(d["missing_concepts_json"]) if d["missing_concepts_json"] else []
        d["incorrect_concepts"] = json.loads(d["incorrect_concepts_json"]) if d["incorrect_concepts_json"] else []
        result.append(d)
    return result

@app.get("/api/descriptive/all")
def get_all_descriptive_evaluations(db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        "SELECT * FROM descriptive_evaluations ORDER BY created_at DESC"
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["expected_concepts"] = json.loads(d["expected_concepts_json"]) if d["expected_concepts_json"] else []
        d["rubric_breakdown"] = json.loads(d["rubric_breakdown_json"]) if d["rubric_breakdown_json"] else {}
        d["covered_concepts"] = json.loads(d["covered_concepts_json"]) if d["covered_concepts_json"] else []
        d["missing_concepts"] = json.loads(d["missing_concepts_json"]) if d["missing_concepts_json"] else []
        d["incorrect_concepts"] = json.loads(d["incorrect_concepts_json"]) if d["incorrect_concepts_json"] else []
        result.append(d)
    return result

@app.get("/api/descriptive/{eval_id}")
def get_descriptive_evaluation(eval_id: str, db: sqlite3.Connection = Depends(get_db)):
    r = db.execute(
        "SELECT * FROM descriptive_evaluations WHERE id = ?", (eval_id,)
    ).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Evaluation record not found.")
    d = dict(r)
    d["expected_concepts"] = json.loads(d["expected_concepts_json"]) if d["expected_concepts_json"] else []
    d["rubric_breakdown"] = json.loads(d["rubric_breakdown_json"]) if d["rubric_breakdown_json"] else {}
    d["covered_concepts"] = json.loads(d["covered_concepts_json"]) if d["covered_concepts_json"] else []
    d["missing_concepts"] = json.loads(d["missing_concepts_json"]) if d["missing_concepts_json"] else []
    d["incorrect_concepts"] = json.loads(d["incorrect_concepts_json"]) if d["incorrect_concepts_json"] else []
    return d

@app.post("/api/descriptive/{eval_id}/override")
def teacher_override(
    eval_id: str,
    req: TeacherOverrideRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    r = db.execute("SELECT * FROM descriptive_evaluations WHERE id = ?", (eval_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Evaluation record not found.")

    max_marks = r["max_marks"] or 10.0
    new_pct = round((req.override_score / max_marks) * 100, 1)

    db.execute(
        """
        UPDATE descriptive_evaluations
        SET teacher_override_score = ?, score = ?, percentage = ?, teacher_remarks = ?
        WHERE id = ?
        """,
        (req.override_score, req.override_score, new_pct, req.teacher_remarks, eval_id)
    )
    db.commit()
    return {
        "message": "Grade successfully updated with teacher override.",
        "id": eval_id,
        "score": req.override_score,
        "percentage": new_pct,
        "teacher_remarks": req.teacher_remarks
    }

# ─── ROOT ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    index_path = "index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "ExamInsight AI API v1.0", "docs": "/docs"})

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/download-zip")
def download_zip():
    zip_path = "examinsight-ai.zip"
    if os.path.exists(zip_path):
        return FileResponse(zip_path, media_type="application/zip", filename="examinsight-ai.zip")
    raise HTTPException(status_code=404, detail="Zip archive not found")
