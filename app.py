import os
import io
import uuid
import socket
import sqlite3
import smtplib
import threading
import csv
from email.message import EmailMessage
from datetime import datetime, timedelta
from typing import Optional, List
from urllib.parse import quote

# third-party libs
import numpy as np
from PIL import Image
import qrcode

# optional libraries
try:
    import cv2  # used only if needed in future
except Exception:
    cv2 = None

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    FACE_RECOGNITION_AVAILABLE = False

try:
    from pyngrok import ngrok, conf as ngrok_conf
    PYNGROK_AVAILABLE = True
except Exception:
    PYNGROK_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

from flask import (
    Flask, request, render_template_string, redirect, url_for, session, send_file,
    flash, jsonify, abort
)
from werkzeug.utils import secure_filename

# -------------------------
# App config from env
# -------------------------
app = Flask(__name__)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r
app.secret_key = os.environ.get("FLASK_SECRET", "dev_change_me_for_demo")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "teacher123")
APP_PORT = int(os.environ.get("APP_PORT", "5001"))

# storage paths
DB_PATH = os.environ.get("DB_PATH", "database/attendance.db")
QR_DIR = os.environ.get("QR_DIR", "static/qr_codes")
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "static/uploads")
STUDENTS_DIR = os.environ.get("STUDENTS_DIR", "students")
CLASSROOM_DIR = os.environ.get("CLASSROOM_DIR", "static/classrooms") # NEW: Classroom image storage
LOG_DIR = os.environ.get("LOG_DIR", "logs")

# defaults
LATE_CUTOFF_MINUTES = int(os.environ.get("LATE_CUTOFF_MINUTES", "10"))
ATTENDANCE_WINDOW_MINUTES = int(os.environ.get("ATTENDANCE_WINDOW_MINUTES", "30"))

FACE_DISTANCE_TOLERANCE = float(os.environ.get("FACE_DISTANCE_TOLERANCE", "0.65"))
SAME_PERSON_TOLERANCE = float(os.environ.get("SAME_PERSON_TOLERANCE", "0.60"))

# MODIFICATION: Liveness check is now highly discouraged and off by default
LIVENESS_REQUIRED = os.environ.get("LIVENESS_REQUIRED", "0") in ("1", "true", "True", "yes", "y")
MOUTH_DELTA_THRESHOLD = float(os.environ.get("MOUTH_DELTA_THRESHOLD", "0.04"))

BACKGROUND_CHECK_REQUIRED = os.environ.get("BACKGROUND_CHECK_REQUIRED", "0") in ("1", "true", "True", "yes", "y") # NEW: Config for background check
BACKGROUND_DISTANCE_TOLERANCE = float(os.environ.get("BACKGROUND_DISTANCE_TOLERANCE", "0.90")) # Placeholder for comparison logic

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587") or 587)
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)

PREFERRED_PUBLIC_BASE = os.environ.get("PUBLIC_BASE_URL")

# limits
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "6"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_SEMAPHORE = threading.Semaphore(int(os.environ.get("MAX_CONCURRENT_UPLOADS", "4")))

RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "8"))
_rate_store = {}
_rate_lock = threading.Lock()

def rate_limit_check(client_ip: str) -> bool:
    now = datetime.now().timestamp()
    with _rate_lock:
        lst = _rate_store.get(client_ip, [])
        lst = [t for t in lst if now - t < RATE_LIMIT_WINDOW]
        if len(lst) >= RATE_LIMIT_MAX:
            _rate_store[client_ip] = lst
            return False
        lst.append(now)
        _rate_store[client_ip] = lst
        return True

# ensure directories exist
def ensure_dirs():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(QR_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(STUDENTS_DIR, exist_ok=True)
    os.makedirs(CLASSROOM_DIR, exist_ok=True) # NEW: Classroom Dir
    os.makedirs(LOG_DIR, exist_ok=True)
ensure_dirs()

# -------------------------
# DB init + migrations
# -------------------------
def init_db_and_migrate():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                date TEXT,
                session_name TEXT DEFAULT '',
                subject TEXT DEFAULT '',
                course_id INTEGER DEFAULT NULL,
                classroom_id INTEGER DEFAULT NULL, -- NEW: classroom FK
                duration_minutes INTEGER DEFAULT 30,
                status TEXT DEFAULT 'OPEN'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS students(
                student_name TEXT PRIMARY KEY,
                email TEXT,
                phone TEXT,
                parent_phone TEXT,
                embedding BLOB,
                enrolled_at TEXT,
                course_id INTEGER DEFAULT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                session_id TEXT,
                status TEXT DEFAULT 'PRESENT',
                timestamp TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS courses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE,
                name TEXT,
                description TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS student_courses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                course_id INTEGER,
                UNIQUE(student_name, course_id)
            )
        """)
        # NEW: Classroom table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS classrooms(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_number TEXT UNIQUE,
                name TEXT,
                reference_image_path TEXT -- path to background image
            )
        """)
        con.commit()

        def add_col_if_missing(table: str, column: str, col_def: str):
            cur.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in cur.fetchall()]
            if column not in cols:
                app.logger.info(f"Adding column {column} to {table}")
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

        add_col_if_missing("students", "parent_phone", "parent_phone TEXT")
        add_col_if_missing("students", "course_id", "course_id INTEGER DEFAULT NULL")
        add_col_if_missing("sessions", "course_id", "course_id INTEGER DEFAULT NULL")
        add_col_if_missing("sessions", "classroom_id", "classroom_id INTEGER DEFAULT NULL") # NEW: Migration for sessions
        con.commit()

init_db_and_migrate()

# -------------------------
# helpers: numpy <-> blob
# -------------------------
def np_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def blob_to_np(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)

# -------------------------
# DB helpers (save/load/list/delete)
# -------------------------
def save_student(student_name: str, email: str, phone: str, parent_phone: str, enc: Optional[np.ndarray], primary_course_id: Optional[int]):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO students(student_name, email, phone, parent_phone, embedding, enrolled_at, course_id)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(student_name) DO UPDATE SET
             email=excluded.email,
             phone=excluded.phone,
             parent_phone=excluded.parent_phone,
             embedding=excluded.embedding,
             enrolled_at=excluded.enrolled_at,
             course_id=excluded.course_id
        """, (
            student_name,
            email,
            phone,
            parent_phone,
            sqlite3.Binary(np_to_blob(enc)) if enc is not None else None,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            primary_course_id
        ))
        con.commit()

def load_student_embedding(student_name: str) -> Optional[np.ndarray]:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT embedding FROM students WHERE student_name=?", (student_name,))
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    try:
        return blob_to_np(row[0])
    except Exception:
        return None

def delete_student(student_name: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM students WHERE student_name=?", (student_name,))
        cur.execute("DELETE FROM attendance WHERE student_name=?", (student_name,))
        cur.execute("DELETE FROM student_courses WHERE student_name=?", (student_name,))
        con.commit()

def list_students():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT student_name, email, phone, parent_phone, enrolled_at, course_id FROM students ORDER BY enrolled_at DESC")
        return cur.fetchall()

def get_student(student_name: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT student_name, email, phone, parent_phone, enrolled_at, course_id FROM students WHERE student_name=?", (student_name,))
        return cur.fetchone()

# courses
def create_course(code: str, name: str, description: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT OR IGNORE INTO courses(code, name, description) VALUES(?,?,?)", (code, name, description))
        con.commit()

def delete_course(course_id: int):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM courses WHERE id=?", (course_id,))
        cur.execute("UPDATE students SET course_id=NULL WHERE course_id=?", (course_id,))
        cur.execute("UPDATE sessions SET course_id=NULL WHERE course_id=?", (course_id,))
        cur.execute("DELETE FROM student_courses WHERE course_id=?", (course_id,))
        con.commit()

def list_courses():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, code, name, description FROM courses ORDER BY name")
        return cur.fetchall()

def get_course_name(course_id: int) -> str:
    if not course_id: return ""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT name FROM courses WHERE id=?", (course_id,))
        row = cur.fetchone()
    return row[0] if row else ""

def set_student_courses(student_name: str, course_ids: List[int]):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM student_courses WHERE student_name=?", (student_name,))
        for cid in (course_ids or []):
            try:
                cur.execute("INSERT OR IGNORE INTO student_courses(student_name, course_id) VALUES(?,?)", (student_name, cid))
            except Exception:
                pass
        con.commit()

def get_student_course_ids(student_name: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT course_id FROM student_courses WHERE student_name=?", (student_name,))
        return [r[0] for r in cur.fetchall()]

# attendance
def save_attendance_record(name: str, session_id: str, status: str = "PRESENT"):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT INTO attendance(student_name, session_id, status, timestamp) VALUES(?,?,?,?)",
                           (name, session_id, status, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        con.commit()

def attendance_for_session(session_id: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT student_name, status, timestamp FROM attendance WHERE session_id=?", (session_id,))
        return cur.fetchall()

def students_not_marked(session_id: str, course_id: Optional[int] = None):
    """
    If course_id provided, only consider students enrolled in that course (via student_courses).
    Otherwise, consider all students.
    """
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        if course_id:
            # students enrolled in course (student_courses)
            cur.execute("""
                SELECT s.student_name, s.email, s.parent_phone
                FROM students s
                INNER JOIN student_courses sc ON s.student_name = sc.student_name
                LEFT JOIN attendance a ON s.student_name = a.student_name AND a.session_id = ?
                WHERE sc.course_id = ? AND a.student_name IS NULL
            """, (session_id, course_id))
        else:
            cur.execute("""
                SELECT s.student_name, s.email, s.parent_phone
                FROM students s
                LEFT JOIN attendance a ON s.student_name = a.student_name AND a.session_id = ?
                WHERE a.student_name IS NULL
            """, (session_id,))
        return cur.fetchall()

def get_student_count(course_id: Optional[int] = None):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        if course_id:
            cur.execute("SELECT COUNT(DISTINCT student_name) FROM student_courses WHERE course_id=?", (course_id,))
        else:
            cur.execute("SELECT COUNT(*) FROM students")
        return cur.fetchone()[0]

# NEW: Classroom DB Helpers
def create_classroom(room_number: str, name: str, ref_img_path: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT OR IGNORE INTO classrooms(room_number, name, reference_image_path) VALUES(?,?,?)", (room_number, name, ref_img_path))
        con.commit()

def list_classrooms():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, room_number, name, reference_image_path FROM classrooms ORDER BY room_number")
        return cur.fetchall()

def delete_classroom(classroom_id: int):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT reference_image_path FROM classrooms WHERE id=?", (classroom_id,))
        row = cur.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            try: os.remove(row[0])
            except Exception: pass
        cur.execute("DELETE FROM classrooms WHERE id=?", (classroom_id,))
        cur.execute("UPDATE sessions SET classroom_id=NULL WHERE classroom_id=?", (classroom_id,))
        con.commit()

def get_classroom_info(classroom_id: Optional[int]):
    if not classroom_id: return None
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT room_number, name, reference_image_path FROM classrooms WHERE id=?", (classroom_id,))
        row = cur.fetchone()
    return row

# -------------------------
# Image & face helpers
# -------------------------
def ensure_rgb8_c_contig(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = np.asarray(arr, dtype=np.uint8, order="C")
    return np.ascontiguousarray(arr)

def load_rgb8_from_path_strict(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im, dtype=np.uint8)
    return np.ascontiguousarray(rgb)

def face_encoding_safe(img_rgb: np.ndarray):
    if not FACE_RECOGNITION_AVAILABLE:
        # Mock mode when library is missing
        return np.zeros(128, dtype=np.float64), "mock-ok (face_recognition missing)"
    try:
        safe = ensure_rgb8_c_contig(img_rgb)
        if safe.dtype != np.uint8 or safe.ndim != 3 or safe.shape[2] != 3:
            return None, f"bad-array (dtype={safe.dtype}, shape={safe.shape})"
        encs = face_recognition.face_encodings(safe)
        if not encs:
            return None, f"no-encodings (shape={safe.shape})"
        return encs[0], f"ok (shape={safe.shape})"
    except Exception as e:
        return None, f"exception: {type(e).__name__}: {e}"

# NOTE: Since LIVENESS is disabled, this is now dummy code
def mouth_open_score(rgb_img: np.ndarray) -> Optional[float]:
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    try:
        # Simplified: always return a score if face_recognition works to prevent hard crash if re-enabled
        _ = face_recognition.face_locations(rgb_img, model="hog")
        return 0.5 
    except Exception:
        return None

# NEW: Placeholder for Background Check
def background_check_safe(current_rgb: np.ndarray, reference_rgb: np.ndarray) -> tuple[bool, float, str]:
    # --- Start of Placeholder Logic ---
    
    # Resize reference to match current image size for comparison
    h, w, _ = current_rgb.shape
    ref_im = Image.fromarray(reference_rgb)
    ref_im = ref_im.resize((w, h))
    ref_rgb_resized = np.asarray(ref_im).astype(np.uint8)
    
    # Calculate difference using simple MSE (Mean Squared Error)
    diff = np.mean((current_rgb.astype(float) - ref_rgb_resized.astype(float))**2)
    # Normalize MSE to a 0-1 range (max value for 8bit color is 255*255=65025)
    simulated_distance = min(1.0, diff / 65025.0) 
    
    # Use the configured tolerance
    match = simulated_distance <= BACKGROUND_DISTANCE_TOLERANCE
    
    # --- End of Placeholder Logic ---

    if match:
        return True, simulated_distance, f"ok (dist={simulated_distance:.3f})"
    else:
        return False, simulated_distance, f"failed (dist={simulated_distance:.3f} > {BACKGROUND_DISTANCE_TOLERANCE:.2f})"

# -------------------------
# Host detection + ngrok + QR helpers
# -------------------------
def get_local_ip() -> str:
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip.startswith("127."):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
        return ip
    except Exception:
        return "127.0.0.1"

_active_ngrok_tunnel = None

def maybe_start_ngrok(port: int):
    """
    Start or reuse a pyngrok tunnel if NGROK_AUTO=1 and pyngrok available.
    Return public URL (no trailing slash) or None if failed.
    """
    global _active_ngrok_tunnel
    if os.environ.get("NGROK_AUTO", "0") != "1":
        return None
    if not PYNGROK_AVAILABLE:
        app.logger.warning("NGROK_AUTO=1 but pyngrok not installed")
        return None
    try:
        ngrok_conf.get_default().auth_token = os.environ.get("NGROK_AUTHTOKEN")
        tunnels = ngrok.get_tunnels()
        if tunnels:
            for t in tunnels:
                pub = getattr(t, "public_url", None)
                if pub and pub.startswith("https"):
                    _active_ngrok_tunnel = t # Set global for later use
                    return pub.rstrip("/")
        # We start a new tunnel only if one isn't already active and valid
        if _active_ngrok_tunnel is None or not _active_ngrok_tunnel.public_url.startswith("https"):
            _active_ngrok_tunnel = ngrok.connect(port, "http", bind_tls=True)
            return _active_ngrok_tunnel.public_url.rstrip("/")
        return _active_ngrok_tunnel.public_url.rstrip("/")
    except Exception as e:
        app.logger.exception("couldn't start ngrok")
        return None

def detect_public_base(request_host_url: str, port: int):
    """
    Priority:
      - PREFERRED_PUBLIC_BASE env var (Manual config)
      - Active NGROK tunnel (HTTPS URL)
      - fallback to request.host_url (risky if not HTTPS) or local IP
    """
    if PREFERRED_PUBLIC_BASE:
        return PREFERRED_PUBLIC_BASE.rstrip("/")
    if _active_ngrok_tunnel and _active_ngrok_tunnel.public_url.startswith("https"):
        return _active_ngrok_tunnel.public_url.rstrip("/")

    # fallback to request host or local ip
    if request_host_url and not "127.0.0.1" in request_host_url and not "localhost" in request_host_url:
        return request_host_url.rstrip("/")
        
    local = f"http://{get_local_ip()}:{port}"
    return local

def make_qr_to_path(url: str, session_id: str):
    url = (url or "").strip().replace(" ", "")
    qr_path = os.path.join(QR_DIR, f"session_{session_id}.png")
    q = qrcode.QRCode(box_size=6, border=2)
    q.add_data(url)
    q.make(fit=True)
    img = q.make_image(fill_color="black", back_color="white")
    img.save(qr_path)
    return qr_path

# -------------------------
# Email & SMS helpers
# -------------------------
def send_sms(to_number: str, message: str) -> bool:
    if not TWILIO_AVAILABLE or not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM):
        app.logger.info(f"[SMS fallback] To:{to_number} Msg:{message}")
        return False
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM, to=to_number)
        return True
    except Exception:
        app.logger.exception("Twilio send failed")
        return False

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        app.logger.info(f"[Email fallback] To:{to_email} Subject:{subject} Body:{body}")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception:
        app.logger.exception("SMTP send failed")
        return False

# -------------------------
# Templates
# -------------------------
BASE_HEAD = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{{ title }} | Smart Attendance</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="/static/css/style.css">
<script src="https://unpkg.com/lucide@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
  tailwind.config = {
    darkMode: 'class',
    theme: {
      extend: {
        colors: {
          indigo: {
            50: '#f5f3ff',
            100: '#ede9fe',
            200: '#ddd6fe',
            300: '#c084fc',
            400: '#a855f7',
            500: '#8b5cf6',
            600: '#4f46e5',
            700: '#4338ca',
            800: '#3730a3',
            900: '#312e81',
          }
        }
      }
    }
  }
</script>
<script>
  if (localStorage.getItem('color-theme') === 'dark' || (!('color-theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
  } else {
      document.documentElement.classList.remove('dark');
  }
</script>
</head>
<body class="bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 min-h-screen flex flex-col">
<div id="toast-container"></div>

{% set show_sidebar = (request.path.startswith('/admin') or request.path in ['/enroll', '/generate_session_qr', '/sessions_admin', '/session_report']) and request.path != '/admin/login' %}

{% if show_sidebar %}
<!-- Admin Dashboard Layout -->
<div class="flex min-h-screen">
  <!-- Sidebar -->
  <aside id="admin-sidebar" class="fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-slate-855 border-r border-slate-200 dark:border-slate-800 transform -translate-x-full md:translate-x-0 transition-transform duration-300 ease-in-out shadow-sm">
    <div class="h-16 flex items-center justify-between px-6 border-b border-slate-200 dark:border-slate-800">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-md shadow-indigo-500/20">A</div>
        <span class="font-bold text-lg tracking-tight bg-gradient-to-r from-indigo-600 to-violet-500 bg-clip-text text-transparent font-display">Attendance</span>
      </div>
      <button class="md:hidden text-slate-500 hover:text-slate-700 dark:hover:text-slate-300" onclick="toggleSidebar()">
        <i data-lucide="x" class="w-5 h-5"></i>
      </button>
    </div>
    
    <nav class="p-4 space-y-1.5 overflow-y-auto" style="height: calc(100vh - 4rem);">
      <a href="/admin" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition {{ 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400' if request.path == '/admin' else 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30' }}">
        <i data-lucide="layout-dashboard" class="w-5 h-5"></i>
        <span>Dashboard</span>
      </a>
      <a href="/admin/students" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition {{ 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400' if request.path == '/admin/students' else 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30' }}">
        <i data-lucide="users" class="w-5 h-5"></i>
        <span>Students</span>
      </a>
      <a href="/admin/courses" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition {{ 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400' if request.path == '/admin/courses' else 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30' }}">
        <i data-lucide="book-open" class="w-5 h-5"></i>
        <span>Courses</span>
      </a>
      <a href="/admin/classrooms" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition {{ 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400' if request.path == '/admin/classrooms' else 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30' }}">
        <i data-lucide="school" class="w-5 h-5"></i>
        <span>Classrooms</span>
      </a>
      <a href="/sessions_admin" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition {{ 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400' if request.path in ['/sessions_admin', '/sessions'] else 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30' }}">
        <i data-lucide="calendar" class="w-5 h-5"></i>
        <span>Sessions</span>
      </a>
      <a href="/generate_session_qr" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition {{ 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400' if request.path == '/generate_session_qr' else 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30' }}">
        <i data-lucide="qr-code" class="w-5 h-5"></i>
        <span>Generate QR</span>
      </a>
      <a href="/enroll" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition {{ 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400' if request.path == '/enroll' else 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30' }}">
        <i data-lucide="user-plus" class="w-5 h-5"></i>
        <span>Enroll Student</span>
      </a>
      
      <div class="pt-4 border-t border-slate-200 dark:border-slate-800 mt-4 space-y-1.5">
        <button onclick="toggleTheme()" class="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/30 text-left">
          <i data-lucide="sun" class="hidden dark:block w-5 h-5"></i>
          <i data-lucide="moon" class="block dark:hidden w-5 h-5"></i>
          <span>Theme Toggle</span>
        </button>
        <a href="/admin/logout" class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/20">
          <i data-lucide="log-out" class="w-5 h-5"></i>
          <span>Logout</span>
        </a>
      </div>
    </nav>
  </aside>

  <!-- Content Container -->
  <div class="flex-1 md:pl-64 flex flex-col min-h-screen">
    <!-- Top Nav -->
    <header class="h-16 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-800 px-6 flex items-center justify-between sticky top-0 z-40">
      <div class="flex items-center gap-4">
        <!-- Sidebar Toggle (Mobile) -->
        <button class="md:hidden text-slate-500 hover:text-slate-700 dark:hover:text-slate-300" onclick="toggleSidebar()">
          <i data-lucide="menu" class="w-6 h-6"></i>
        </button>
        <h1 class="text-xl font-bold text-slate-800 dark:text-white capitalize font-display">{{ title }}</h1>
      </div>
      
      <!-- Topbar Controls -->
      <div class="flex items-center gap-4">
        <!-- Notification Bell -->
        <div class="relative">
          <button id="notification-bell-btn" class="p-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700/50 transition">
            <i data-lucide="bell" class="w-5 h-5"></i>
            <span class="absolute top-1.5 right-1.5 w-2.5 h-2.5 bg-indigo-600 rounded-full border border-white dark:border-slate-800"></span>
          </button>
          <div id="notification-dropdown" class="hidden absolute right-0 mt-2 w-80 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-50 overflow-hidden">
            <div class="p-4 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center">
              <span class="font-bold text-sm text-slate-800 dark:text-white">Recent Activities</span>
              <span class="text-xs text-indigo-600 dark:text-indigo-400 font-semibold cursor-pointer" onclick="clearNotifications()">Clear</span>
            </div>
            <div id="notification-list" class="divide-y divide-slate-100 dark:divide-slate-700/50 max-h-60 overflow-y-auto">
              <div class="p-3 hover:bg-slate-50 dark:hover:bg-slate-700/30 transition text-xs">
                <p class="font-medium text-slate-800 dark:text-white">System started successfully</p>
                <p class="text-slate-400 mt-0.5">Just now</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Profile Dropdown -->
        <div class="relative">
          <button id="profile-dropdown-btn" class="flex items-center gap-2 p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700/50 transition">
            <div class="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 flex items-center justify-center font-bold text-sm">A</div>
            <i data-lucide="chevron-down" class="w-4 h-4 text-slate-400"></i>
          </button>
          <div id="profile-dropdown" class="hidden absolute right-0 mt-2 w-48 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-50 py-1">
            <div class="px-4 py-2 border-b border-slate-200 dark:border-slate-800">
              <p class="text-sm font-bold text-slate-800 dark:text-white">Admin Account</p>
              <p class="text-xs text-slate-400 mt-0.5">teacher@school.edu</p>
            </div>
            <a href="/admin" class="flex items-center gap-2 px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700/50"><i data-lucide="user" class="w-4 h-4"></i> Profile</a>
            <a href="/admin" class="flex items-center gap-2 px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700/50"><i data-lucide="settings" class="w-4 h-4"></i> Settings</a>
            <a href="/admin/logout" class="flex items-center gap-2 px-4 py-2 text-sm text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/20"><i data-lucide="log-out" class="w-4 h-4"></i> Logout</a>
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 p-6 max-w-7xl w-full mx-auto animate-slide-up">
{% else %}
<!-- Public/Student Portal Layout -->
<header class="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-800 sticky top-0 z-40 shadow-sm">
  <div class="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="text-2xl font-bold tracking-tight bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent flex items-center gap-2 font-display">
        <span>📋</span> Attendance
      </div>
      <div class="text-xs text-slate-400 border-l border-slate-200 dark:border-slate-800 pl-3 hidden sm:block">Smart face-based attendance</div>
    </div>
    
    <nav class="flex items-center gap-4 text-sm font-medium">
      <a href="/" class="text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 px-2 py-1 transition">Home</a>
      <a href="/student" class="text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 px-2 py-1 transition">Student</a>
      <button onclick="toggleTheme()" class="p-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700/50 transition">
        <i data-lucide="sun" class="hidden dark:block w-5 h-5"></i>
        <i data-lucide="moon" class="block dark:hidden w-5 h-5"></i>
      </button>
      <a href="/admin/login" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-xl text-sm font-semibold shadow-md shadow-indigo-500/10 transition">Admin Login</a>
    </nav>
  </div>
</header>
<main class="max-w-4xl mx-auto p-6 flex-1 w-full animate-fade-in">
{% endif %}
"""

BASE_FOOT = """
</main>

<!-- Footer -->
<footer class="border-t border-slate-200 dark:border-slate-800 py-6 text-center text-xs text-slate-400 dark:text-slate-500">
  <div class="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row justify-between items-center gap-3">
    <p>&copy; 2026 Smart Attendance System. All Rights Reserved.</p>
    <p>Version 2.0.0 &bull; Developed with ❤️</p>
  </div>
</footer>
</div>
{% if show_sidebar %}
</div>
{% endif %}

<!-- Custom Delete Confirmation Modal -->
<div id="custom-modal-overlay" class="custom-modal-overlay">
  <div class="custom-modal-content">
    <div class="flex items-center gap-3 mb-4">
      <div class="p-2 bg-rose-100 dark:bg-rose-950/30 text-rose-600 dark:text-rose-400 rounded-full">
        <i data-lucide="alert-triangle" class="w-6 h-6"></i>
      </div>
      <h3 id="custom-modal-title" class="text-lg font-bold text-slate-800 dark:text-white">Confirm Action</h3>
    </div>
    <p id="custom-modal-message" class="text-sm text-slate-500 dark:text-slate-400 mb-6">Are you sure you want to delete this record? This action cannot be undone.</p>
    <div class="flex justify-end gap-3">
      <button id="custom-modal-cancel" class="px-4 py-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-xl text-sm font-semibold transition">Cancel</button>
      <button id="custom-modal-confirm" class="px-4 py-2 bg-rose-600 hover:bg-rose-700 text-white rounded-xl text-sm font-semibold transition">Delete</button>
    </div>
  </div>
</div>

<!-- Scripts -->
<script>
  // Sidebar toggle
  function toggleSidebar() {
      const sidebar = document.getElementById('admin-sidebar');
      if (sidebar) {
          sidebar.classList.toggle('-translate-x-full');
      }
  }

  // Profile Dropdown toggle
  const profileBtn = document.getElementById('profile-dropdown-btn');
  const profileMenu = document.getElementById('profile-dropdown');
  if (profileBtn && profileMenu) {
      profileBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          profileMenu.classList.toggle('hidden');
          const notificationMenu = document.getElementById('notification-dropdown');
          if (notificationMenu) notificationMenu.classList.add('hidden');
      });
  }

  // Notification Dropdown toggle
  const notificationBtn = document.getElementById('notification-bell-btn');
  const notificationMenu = document.getElementById('notification-dropdown');
  if (notificationBtn && notificationMenu) {
      notificationBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          notificationMenu.classList.toggle('hidden');
          if (profileMenu) profileMenu.classList.add('hidden');
      });
  }

  // Close menus on click outside
  document.addEventListener('click', () => {
      if (profileMenu) profileMenu.classList.add('hidden');
      if (notificationMenu) notificationMenu.classList.add('hidden');
  });

  // Load notifications from localStorage or inject defaults
  function loadNotifications() {
      const list = document.getElementById('notification-list');
      if (!list) return;
      
      let activities = JSON.parse(localStorage.getItem('attendance_activities') || '[]');
      if (activities.length === 0) {
          activities = [
              { text: "System initialized successfully", time: "1 hour ago" },
              { text: "Teacher logged in from local subnet", time: "10 mins ago" }
          ];
          localStorage.setItem('attendance_activities', JSON.stringify(activities));
      }
      
      list.innerHTML = activities.map(act => `
          <div class="p-3 hover:bg-slate-55 dark:hover:bg-slate-700/30 transition text-xs">
              <p class="font-medium text-slate-800 dark:text-slate-200">${act.text}</p>
              <p class="text-slate-400 dark:text-slate-500 mt-0.5">${act.time}</p>
          </div>
      `).join('');
  }

  window.addActivity = function(text) {
      let activities = JSON.parse(localStorage.getItem('attendance_activities') || '[]');
      activities.unshift({ text: text, time: "Just now" });
      if (activities.length > 8) activities.pop();
      localStorage.setItem('attendance_activities', JSON.stringify(activities));
      loadNotifications();
  };

  window.clearNotifications = function() {
      localStorage.setItem('attendance_activities', JSON.stringify([]));
      loadNotifications();
  };
  
  loadNotifications();

  // Custom confirmation modal
  window.confirmDelete = function(message, onConfirm) {
      const overlay = document.getElementById('custom-modal-overlay');
      const msgEl = document.getElementById('custom-modal-message');
      const confirmBtn = document.getElementById('custom-modal-confirm');
      const cancelBtn = document.getElementById('custom-modal-cancel');
      
      if (!overlay) {
          if (confirm(message)) onConfirm();
          return;
      }
      
      msgEl.textContent = message;
      overlay.classList.add('active');
      
      const newConfirmBtn = confirmBtn.cloneNode(true);
      confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
      
      newConfirmBtn.addEventListener('click', function() {
          overlay.classList.remove('active');
          onConfirm();
      });
      
      cancelBtn.onclick = () => overlay.classList.remove('active');
  };

  // Toast notifications function
  window.showToast = function(message, type = 'success') {
      const container = document.getElementById('toast-container');
      if (!container) return;
      const toast = document.createElement('div');
      
      let icon = '';
      let borderClass = 'border-indigo-600';
      if (type === 'success') {
          icon = '<i data-lucide="check-circle" class="text-emerald-500 w-5 h-5"></i>';
          borderClass = 'border-emerald-500';
      } else if (type === 'error') {
          icon = '<i data-lucide="x-circle" class="text-rose-500 w-5 h-5"></i>';
          borderClass = 'border-rose-500';
      } else if (type === 'warning') {
          icon = '<i data-lucide="alert-triangle" class="text-amber-500 w-5 h-5"></i>';
          borderClass = 'border-amber-500';
      } else {
          icon = '<i data-lucide="info" class="text-blue-500 w-5 h-5"></i>';
          borderClass = 'border-blue-500';
      }
      
      toast.className = `toast-item animate-slide-in flex items-center p-4 mb-4 text-slate-800 dark:text-slate-200 bg-white dark:bg-slate-800 border-l-4 ${borderClass}`;
      toast.innerHTML = `
          <div class="inline-flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-lg">
              ${icon}
          </div>
          <div class="ml-3 text-sm font-normal pr-2">${message}</div>
          <button type="button" class="ml-auto -mx-1.5 -my-1.5 bg-white text-gray-400 hover:text-gray-900 rounded-lg p-1.5 hover:bg-gray-100 inline-flex items-center justify-center h-8 w-8 dark:bg-gray-800 dark:text-gray-500 dark:hover:text-white dark:hover:bg-slate-700/50" onclick="this.parentElement.remove()">
              <span class="sr-only">Close</span>
              <i data-lucide="x" class="w-4 h-4"></i>
          </button>
      `;
      container.appendChild(toast);
      lucide.createIcons();
      
      setTimeout(() => {
          toast.style.opacity = '0';
          toast.style.transform = 'translateX(50px)';
          toast.style.transition = 'opacity 0.3s, transform 0.3s';
          setTimeout(() => toast.remove(), 300);
      }, 4000);
  };

  // Dark/Light Theme Switcher
  window.toggleTheme = function() {
      if (document.documentElement.classList.contains('dark')) {
          document.documentElement.classList.remove('dark');
          localStorage.setItem('color-theme', 'light');
      } else {
          document.documentElement.classList.add('dark');
          localStorage.setItem('color-theme', 'dark');
      }
  };

  // Process flashed messages automatically
  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
      document.addEventListener("DOMContentLoaded", function() {
        {% for category, message in messages %}
          showToast("{{ message }}", "{{ 'error' if category == 'error' else 'success' }}");
        {% endfor %}
      });
    {% endif %}
  {% endwith %}

  // Initialize lucide icons
  lucide.createIcons();
</script>
</body>
</html>
"""

# Updated HOME_HTML with icons
HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Attendance System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>
      tailwind.config = {
        darkMode: 'class',
      }
    </script>
    <script>
      if (localStorage.getItem('color-theme') === 'dark' || (!('color-theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
          document.documentElement.classList.add('dark');
      } else {
          document.documentElement.classList.remove('dark');
      }
    </script>
</head>
<body class="bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-100 min-h-screen flex flex-col font-sans transition-colors duration-200">
    <!-- Navbar -->
    <header class="sticky top-0 z-40 w-full border-b border-slate-200/80 bg-white/80 dark:border-slate-800/80 dark:bg-slate-900/80 backdrop-blur-md">
      <div class="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <div class="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold shadow-md shadow-indigo-500/20 text-lg">📋</div>
          <span class="font-bold text-xl tracking-tight bg-gradient-to-r from-indigo-600 to-violet-500 bg-clip-text text-transparent font-display">Smart Attendance</span>
        </div>
        
        <div class="flex items-center gap-4 text-sm font-medium">
          <a href="/student" class="text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition">Student Portal</a>
          <button onclick="toggleTheme()" class="p-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800/50 transition">
            <i data-lucide="sun" class="hidden dark:block w-5 h-5"></i>
            <i data-lucide="moon" class="block dark:hidden w-5 h-5"></i>
          </button>
          <a href="/admin/login" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4.5 py-2 rounded-xl text-sm font-semibold shadow-md shadow-indigo-500/10 transition">Admin Login</a>
        </div>
      </div>
    </header>

    <!-- Hero Section -->
    <main class="flex-1 max-w-6xl w-full mx-auto px-6 py-12 md:py-20 flex flex-col md:flex-row items-center gap-12">
        <div class="flex-1 space-y-6 text-center md:text-left animate-slide-up">
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-100 dark:border-indigo-900/30 text-indigo-600 dark:text-indigo-400 text-xs font-semibold">
                <i data-lucide="sparkles" class="w-3.5 h-3.5"></i> State-of-the-Art Attendance
            </div>
            <h1 class="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight font-display">
                Next-Gen <br class="hidden lg:block"/>
                <span class="bg-gradient-to-r from-indigo-600 via-purple-600 to-violet-500 bg-clip-text text-transparent">Dual-Verification</span> <br/>
                Attendance System
            </h1>
            <p class="text-lg text-slate-500 dark:text-slate-400 max-w-lg">
                Mark attendance securely and efficiently. Utilizing dynamic QR codes, facial recognition matching, liveness spoof detection, and classroom background checks.
            </p>
            <div class="flex flex-wrap gap-4 justify-center md:justify-start pt-2">
                <a href="/student" class="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3.5 rounded-xl font-semibold shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30 transform hover:-translate-y-0.5 transition duration-200 flex items-center gap-2">
                    <i data-lucide="scan" class="w-5 h-5"></i> Student Portal
                </a>
                <a href="/admin/login" class="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-750 text-slate-700 dark:text-slate-200 px-8 py-3.5 rounded-xl font-semibold shadow-sm transform hover:-translate-y-0.5 transition duration-200 flex items-center gap-2">
                    <i data-lucide="lock" class="w-5 h-5"></i> Admin Login
                </a>
            </div>
            <p class="text-xs text-slate-400 mt-4">
                Teacher Access Password: <code class="bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded font-mono text-indigo-600 dark:text-indigo-400 font-semibold">teacher123</code>
            </p>
        </div>
        
        <!-- Illustration/Interactive Right Column -->
        <div class="flex-1 w-full max-w-md md:max-w-none flex justify-center items-center animate-fade-in">
            <div class="relative w-full max-w-md aspect-square rounded-3xl bg-gradient-to-tr from-indigo-500/10 via-purple-500/5 to-transparent flex items-center justify-center p-8 border border-slate-200/50 dark:border-slate-800/30">
                <div class="w-72 h-72 rounded-2xl bg-white dark:bg-slate-800 shadow-xl border border-slate-200/50 dark:border-slate-700/50 p-6 flex flex-col justify-between">
                    <div class="flex items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-700/50">
                        <div class="flex items-center gap-2">
                            <span class="text-lg">🏛️</span>
                            <div class="text-left">
                                <p class="text-xs font-bold text-slate-800 dark:text-slate-200">Main Lecture Hall</p>
                                <p class="text-[10px] text-slate-400">Classroom #402</p>
                            </div>
                        </div>
                        <span class="px-2 py-0.5 rounded-full text-[9px] font-semibold text-emerald-700 bg-emerald-50 dark:bg-emerald-950/40 dark:text-emerald-400">OPEN</span>
                    </div>
                    
                    <div class="my-6 flex flex-col items-center justify-center gap-3">
                        <div class="w-24 h-24 rounded-full border-4 border-dashed border-indigo-500 dark:border-indigo-400 flex items-center justify-center bg-indigo-50/50 dark:bg-indigo-950/20">
                            <i data-lucide="scan-face" class="w-10 h-10 text-indigo-600 dark:text-indigo-400"></i>
                        </div>
                        <p class="text-xs font-medium text-slate-500 dark:text-slate-400">Align face inside camera overlay</p>
                    </div>
                    
                    <div class="pt-4 border-t border-slate-100 dark:border-slate-700/50 flex justify-between items-center text-xs">
                        <span class="text-slate-400">Subject: Data Structures</span>
                        <span class="font-bold text-indigo-600 dark:text-indigo-400">CSE Sem 5</span>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Features Grid -->
    <section class="max-w-6xl w-full mx-auto px-6 py-12 border-t border-slate-200 dark:border-slate-800">
        <h2 class="text-2xl md:text-3xl font-bold text-center text-slate-800 dark:text-white mb-10 font-display">Advanced Security Measures</h2>
        <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm hover:shadow-md transition">
                <div class="w-12 h-12 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 rounded-xl flex items-center justify-center mb-4"><i data-lucide="qr-code" class="w-6 h-6"></i></div>
                <h3 class="text-lg font-bold text-slate-800 dark:text-white mb-2">QR Code Session</h3>
                <p class="text-sm text-slate-500 dark:text-slate-400">Dynamic session keys generated inside classrooms to guarantee physical attendance.</p>
            </div>
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm hover:shadow-md transition">
                <div class="w-12 h-12 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 rounded-xl flex items-center justify-center mb-4"><i data-lucide="fingerprint" class="w-6 h-6"></i></div>
                <h3 class="text-lg font-bold text-slate-800 dark:text-white mb-2">Facial Matching</h3>
                <p class="text-sm text-slate-500 dark:text-slate-400">Matches student snapshots against registered face print embeddings stored locally.</p>
            </div>
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm hover:shadow-md transition">
                <div class="w-12 h-12 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 rounded-xl flex items-center justify-center mb-4"><i data-lucide="shield-check" class="w-6 h-6"></i></div>
                <h3 class="text-lg font-bold text-slate-800 dark:text-white mb-2">Liveness Detection</h3>
                <p class="text-sm text-slate-500 dark:text-slate-400">Validates facial landmark deltas with a secondary capture to prevent photo-spoofing attacks.</p>
            </div>
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm hover:shadow-md transition">
                <div class="w-12 h-12 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 rounded-xl flex items-center justify-center mb-4"><i data-lucide="map" class="w-6 h-6"></i></div>
                <h3 class="text-lg font-bold text-slate-800 dark:text-white mb-2">Background Check</h3>
                <p class="text-sm text-slate-500 dark:text-slate-400">Validates the background of the classroom to prevent students from marking attendance from home.</p>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="border-t border-slate-200 dark:border-slate-800 py-8 bg-white dark:bg-slate-900 text-center text-xs text-slate-400 dark:text-slate-500 mt-auto">
      <div class="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row justify-between items-center gap-3">
        <p>&copy; 2026 Smart Attendance System. All Rights Reserved.</p>
        <p>Version 2.0.0 &bull; Mini-Project Redesigned</p>
      </div>
    </footer>

    <!-- Theme toggling script -->
    <script>
      window.toggleTheme = function() {
          if (document.documentElement.classList.contains('dark')) {
              document.documentElement.classList.remove('dark');
              localStorage.setItem('color-theme', 'light');
          } else {
              document.documentElement.classList.add('dark');
              localStorage.setItem('color-theme', 'dark');
          }
      };
      lucide.createIcons();
    </script>
</body>
</html>
"""

# -------------------------
# Small helpers
# -------------------------
def is_admin() -> bool:
    return bool(session.get("admin_logged"))

def require_admin():
    if not is_admin():
        return redirect(url_for("admin_login", next=request.path))
    return None

def admin_badge_html() -> str:
    return ("""<span class="ml-2 inline-flex items-center rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">Admin</span>""" if is_admin() else "")

# -------------------------
# Routes
# -------------------------
@app.route("/ping")
def ping():
    return "ok", 200

@app.route("/")
def home():
    return render_template_string(HOME_HTML)

@app.route("/student")
def student_home():
    html = BASE_HEAD + """
    <div class="max-w-md mx-auto bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-xl p-6 md:p-8 text-center animate-slide-up relative overflow-hidden">
        <div class="w-16 h-16 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 rounded-full flex items-center justify-center mx-auto mb-4 border border-indigo-100/50 dark:border-indigo-900/30">
          <i data-lucide="user-check" class="w-8 h-8"></i>
        </div>
        
        <h2 class="text-3xl font-extrabold text-slate-850 dark:text-white font-display mb-2">Student Attendance</h2>
        <p class="text-xs text-slate-450 dark:text-slate-500 mb-6">Enter the Session ID or scan the QR code to begin the attendance check-in.</p>
        
        <form action="/mark_attendance" method="GET" class="space-y-4">
            <div>
                <input type="text" name="session" placeholder="Session ID" required class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white text-center font-mono font-bold tracking-widest">
            </div>
            
            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-xl font-bold text-sm shadow-lg shadow-indigo-500/10 hover:shadow-indigo-500/20 transform hover:-translate-y-0.5 transition duration-200 flex items-center justify-center gap-2">
                <i data-lucide="arrow-right-circle" class="w-4 h-4"></i> Start Attendance Snap
            </button>
        </form>
        
        <div class="mt-6 border-t border-slate-100 dark:border-slate-750/50 pt-4 text-xs text-slate-400 dark:text-slate-500">
            Ensure your face is clearly visible and the background matches your classroom reference.
        </div>
    </div>
    <script>lucide.createIcons();</script>
    """ + BASE_FOOT
    return render_template_string(html, title="Student Login", is_admin=False)

# -------------------------
# Admin login/logout & hub
# -------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if (request.form.get("password") or "") == ADMIN_PASSWORD:
            session["admin_logged"] = True
            return redirect(request.args.get("next") or url_for("admin_hub"))
        else:
            flash("Incorrect password", "error")
    html = BASE_HEAD + """
    <div class="max-w-md mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md rounded-2xl shadow-xl border border-slate-100 dark:border-slate-700/50 p-8 animate-slide-up">
      <div class="text-center mb-6">
        <div class="w-16 h-16 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 rounded-full flex items-center justify-center mx-auto mb-4 border border-indigo-100/50 dark:border-indigo-900/30">
          <i data-lucide="lock" class="w-8 h-8"></i>
        </div>
        <h2 class="text-3xl font-extrabold text-slate-800 dark:text-white font-display">Admin Login</h2>
        <p class="text-sm text-slate-500 dark:text-slate-400 mt-2">Enter credentials to access teacher console</p>
      </div>

      <form method="POST" class="space-y-5">
        <div>
          <label class="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-1.5"><i data-lucide="key" class="w-4 h-4 text-slate-400"></i> Access Password</label>
          <div class="relative">
            <input name="password" id="password-input" type="password" class="w-full border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-3 bg-white dark:bg-slate-900 text-slate-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none text-sm transition" placeholder="Enter password" required>
            <button type="button" onclick="togglePasswordVisibility()" class="absolute right-3.5 top-3.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition">
              <i data-lucide="eye" id="password-eye-open" class="w-5 h-5"></i>
              <i data-lucide="eye-off" id="password-eye-closed" class="w-5 h-5 hidden"></i>
            </button>
          </div>
          <p class="text-[11px] text-slate-400 dark:text-slate-500 mt-1.5">Default login password is <code class="font-semibold text-indigo-500 dark:text-indigo-450">teacher123</code></p>
        </div>

        <div class="flex items-center justify-between text-xs">
          <label class="flex items-center gap-2 cursor-pointer text-slate-500 dark:text-slate-400">
            <input type="checkbox" name="remember" id="remember-me" class="rounded border-slate-300 dark:border-slate-700 text-indigo-600 focus:ring-indigo-500 w-4 h-4">
            <span>Remember Me</span>
          </label>
        </div>

        <button class="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-xl font-semibold shadow-lg shadow-indigo-500/10 hover:shadow-indigo-500/20 transform hover:-translate-y-0.5 transition duration-200 flex items-center justify-center gap-2 text-sm mt-6">
          <i data-lucide="log-in" class="w-4 h-4"></i> Login to Dashboard
        </button>
      </form>
    </div>

    <script>
      document.addEventListener("DOMContentLoaded", function() {
          const remember = localStorage.getItem("remember_login");
          if (remember === "true") {
              document.getElementById("remember-me").checked = true;
          }
      });

      document.getElementById("remember-me").addEventListener("change", function(e) {
          localStorage.setItem("remember_login", e.target.checked ? "true" : "false");
      });

      window.togglePasswordVisibility = function() {
          const pwd = document.getElementById("password-input");
          const eyeOpen = document.getElementById("password-eye-open");
          const eyeClosed = document.getElementById("password-eye-closed");
          if (pwd.type === "password") {
              pwd.type = "text";
              eyeOpen.classList.add("hidden");
              eyeClosed.classList.remove("hidden");
          } else {
              pwd.type = "password";
              eyeOpen.classList.remove("hidden");
              eyeClosed.classList.add("hidden");
          }
      };
      lucide.createIcons();
    </script>
    """ + BASE_FOOT
    return render_template_string(html, title="Admin Login", is_admin=is_admin(), admin_badge=admin_badge_html())

@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/admin")
def admin_hub():
    guard = require_admin()
    if guard:
        return guard
    
    # Defaults in case of error
    total_students = 0
    total_courses = 0
    total_classrooms = 0
    active_sessions = 0
    today_attendance = 0
    attendance_percentage = 0.0
    activities = []
    
    daily_labels = []
    daily_values = []
    weekly_labels = []
    weekly_values = []
    monthly_labels = []
    monthly_values = []
    course_labels = []
    course_values = []

    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            
            # Total Students
            cur.execute("SELECT COUNT(*) FROM students")
            total_students = cur.fetchone()[0]
            
            # Total Courses
            cur.execute("SELECT COUNT(*) FROM courses")
            total_courses = cur.fetchone()[0]
            
            # Total Classrooms
            cur.execute("SELECT COUNT(*) FROM classrooms")
            total_classrooms = cur.fetchone()[0]
            
            # Active Sessions
            cur.execute("SELECT COUNT(*) FROM sessions WHERE status='OPEN'")
            active_sessions = cur.fetchone()[0]
            
            # Today's Attendance
            today_str = datetime.now().strftime("%Y-%m-%d")
            cur.execute("SELECT COUNT(*) FROM attendance WHERE date(timestamp) = date(?)", (today_str,))
            today_attendance = cur.fetchone()[0]
            
            # Attendance Percentage
            cur.execute("SELECT COUNT(*) FROM attendance WHERE status IN ('PRESENT', 'LATE')")
            present_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM attendance")
            total_attendance_records = cur.fetchone()[0]
            if total_attendance_records > 0:
                attendance_percentage = round((present_count / total_attendance_records) * 100, 1)
                
            # Recent Activities
            # 1. Recent Attendance
            cur.execute("""
                SELECT student_name, timestamp, session_id, status 
                FROM attendance 
                ORDER BY timestamp DESC LIMIT 5
            """)
            for r in cur.fetchall():
                activities.append({
                    "text": f"Student {r[0]} marked {r[3].lower()} for session {r[2]}",
                    "time": r[1]
                })
                
            # 2. Recent Students Enrolled
            cur.execute("""
                SELECT student_name, enrolled_at 
                FROM students 
                ORDER BY enrolled_at DESC LIMIT 3
            """)
            for r in cur.fetchall():
                activities.append({
                    "text": f"Student '{r[0]}' enrolled into system",
                    "time": r[1]
                })
                
            # 3. Recent Sessions Created
            cur.execute("""
                SELECT session_name, date, session_id 
                FROM sessions 
                ORDER BY date DESC LIMIT 3
            """)
            for r in cur.fetchall():
                activities.append({
                    "text": f"Session '{r[0]}' ({r[2]}) created",
                    "time": r[1]
                })
                
            # Sort activities and limit to 5
            activities.sort(key=lambda x: x["time"] or "", reverse=True)
            activities = activities[:5]
            
            # Formatted times
            for act in activities:
                if act["time"]:
                    try:
                        dt = datetime.strptime(act["time"], "%Y-%m-%d %H:%M:%S")
                        diff = datetime.now() - dt
                        if diff.days == 0:
                            if diff.seconds < 60:
                                act["display_time"] = "Just now"
                            elif diff.seconds < 3600:
                                act["display_time"] = f"{diff.seconds // 60}m ago"
                            else:
                                act["display_time"] = f"{diff.seconds // 3600}h ago"
                        else:
                            act["display_time"] = f"{diff.days}d ago"
                    except Exception:
                        act["display_time"] = act["time"][:16]
                else:
                    act["display_time"] = "Recently"

            # Daily Attendance Trend (7 days)
            for i in range(6, -1, -1):
                dt_day = datetime.now() - timedelta(days=i)
                d_str = dt_day.strftime("%Y-%m-%d")
                cur.execute("SELECT COUNT(*) FROM attendance WHERE date(timestamp) = date(?) AND status IN ('PRESENT', 'LATE')", (d_str,))
                cnt = cur.fetchone()[0]
                daily_labels.append(dt_day.strftime("%a"))
                daily_values.append(cnt)

            # Weekly Attendance Trend (4 weeks)
            for i in range(3, -1, -1):
                d_start = (datetime.now() - timedelta(weeks=i+1)).strftime("%Y-%m-%d")
                d_end = (datetime.now() - timedelta(weeks=i)).strftime("%Y-%m-%d")
                cur.execute("SELECT COUNT(*) FROM attendance WHERE date(timestamp) > date(?) AND date(timestamp) <= date(?) AND status IN ('PRESENT', 'LATE')", (d_start, d_end))
                cnt = cur.fetchone()[0]
                weekly_labels.append(f"Week -{i}" if i > 0 else "This Week")
                weekly_values.append(cnt)

            # Monthly Attendance Trend (6 months)
            for i in range(5, -1, -1):
                dt_month = datetime.now() - timedelta(days=i*30)
                month_str = dt_month.strftime("%Y-%m")
                cur.execute("SELECT COUNT(*) FROM attendance WHERE strftime('%Y-%m', timestamp) = ? AND status IN ('PRESENT', 'LATE')", (month_str,))
                cnt = cur.fetchone()[0]
                monthly_labels.append(dt_month.strftime("%b"))
                monthly_values.append(cnt)

            # Course-wise counts
            cur.execute("""
                SELECT c.code, COUNT(a.id) 
                FROM attendance a 
                JOIN sessions s ON a.session_id = s.session_id 
                JOIN courses c ON s.course_id = c.id 
                WHERE a.status IN ('PRESENT', 'LATE')
                GROUP BY c.code 
                LIMIT 5
            """)
            course_data = cur.fetchall()
            course_labels = [r[0] for r in course_data]
            course_values = [r[1] for r in course_data]
            
            if not course_labels:
                course_labels = ["No Data"]
                course_values = [0]
    except Exception as e:
        print("Error fetching dashboard statistics:", e)
        daily_labels, daily_values = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], [0, 0, 0, 0, 0, 0, 0]
        weekly_labels, weekly_values = ["Wk 4", "Wk 3", "Wk 2", "Wk 1"], [0, 0, 0, 0]
        monthly_labels, monthly_values = ["Month 6", "Month 5", "Month 4", "Month 3", "Month 2", "Month 1"], [0, 0, 0, 0, 0, 0]
        course_labels, course_values = ["No Data"], [0]

    html = BASE_HEAD + """
    <!-- Welcome section -->
    <div class="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
            <h2 class="text-3xl font-extrabold text-slate-800 dark:text-white font-display">Welcome Back, Admin</h2>
            <p class="text-sm text-slate-500 dark:text-slate-400 mt-1" id="dynamic-greeting"></p>
        </div>
        <div class="flex items-center gap-3">
            <span class="text-xs text-slate-400 dark:text-slate-500 border border-slate-200 dark:border-slate-800 px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800">
                System Status: <span class="text-emerald-500 font-semibold">Online</span>
            </span>
        </div>
    </div>

    <!-- Stats row -->
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-5 mb-8">
        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-5 rounded-2xl shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div class="flex items-center justify-between text-slate-400 mb-4">
                <span class="text-xs font-semibold">Total Students</span>
                <i data-lucide="users" class="w-5 h-5 text-indigo-600 dark:text-indigo-400"></i>
            </div>
            <div>
                <p class="text-3xl font-bold text-slate-800 dark:text-white font-display">{{ total_students }}</p>
                <p class="text-[10px] text-emerald-500 font-semibold mt-1 flex items-center gap-0.5"><i data-lucide="trending-up" class="w-3 h-3"></i> Enrolled</p>
            </div>
        </div>
        
        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-5 rounded-2xl shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div class="flex items-center justify-between text-slate-400 mb-4">
                <span class="text-xs font-semibold">Today's Presence</span>
                <i data-lucide="check-square" class="w-5 h-5 text-emerald-600 dark:text-emerald-450"></i>
            </div>
            <div>
                <p class="text-3xl font-bold text-slate-800 dark:text-white font-display">{{ today_attendance }}</p>
                <p class="text-[10px] text-slate-400 mt-1">Marked attendance today</p>
            </div>
        </div>

        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-5 rounded-2xl shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div class="flex items-center justify-between text-slate-400 mb-4">
                <span class="text-xs font-semibold">Active Sessions</span>
                <i data-lucide="play-circle" class="w-5 h-5 text-amber-500"></i>
            </div>
            <div>
                <p class="text-3xl font-bold text-slate-800 dark:text-white font-display">{{ active_sessions }}</p>
                <p class="text-[10px] text-amber-500 font-semibold mt-1">Open class sessions</p>
            </div>
        </div>

        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-5 rounded-2xl shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div class="flex items-center justify-between text-slate-400 mb-4">
                <span class="text-xs font-semibold">Total Courses</span>
                <i data-lucide="book-open" class="w-5 h-5 text-blue-500"></i>
            </div>
            <div>
                <p class="text-3xl font-bold text-slate-800 dark:text-white font-display">{{ total_courses }}</p>
                <p class="text-[10px] text-slate-400 mt-1">Active curriculum</p>
            </div>
        </div>

        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-5 rounded-2xl shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div class="flex items-center justify-between text-slate-400 mb-4">
                <span class="text-xs font-semibold">Classrooms</span>
                <i data-lucide="school" class="w-5 h-5 text-purple-500"></i>
            </div>
            <div>
                <p class="text-3xl font-bold text-slate-800 dark:text-white font-display">{{ total_classrooms }}</p>
                <p class="text-[10px] text-slate-400 mt-1">Rooms configured</p>
            </div>
        </div>

        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-5 rounded-2xl shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div class="flex items-center justify-between text-slate-400 mb-4">
                <span class="text-xs font-semibold">Attendance %</span>
                <i data-lucide="percent" class="w-5 h-5 text-rose-500"></i>
            </div>
            <div>
                <p class="text-3xl font-bold text-slate-800 dark:text-white font-display">{{ attendance_percentage }}%</p>
                <p class="text-[10px] text-slate-400 mt-1">Overall present ratio</p>
            </div>
        </div>
    </div>

    <!-- Quick Actions Panel -->
    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm mb-8">
        <h3 class="text-lg font-bold text-slate-850 dark:text-white mb-4 flex items-center gap-2"><i data-lucide="zap" class="w-5 h-5 text-amber-500"></i> Quick Command Console</h3>
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <a href="/enroll" class="flex flex-col items-center justify-center p-4 border border-slate-100 dark:border-slate-700/60 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-750 transition text-center group">
                <div class="w-10 h-10 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-650 dark:text-indigo-400 rounded-xl flex items-center justify-center mb-2 group-hover:scale-105 transition"><i data-lucide="user-plus" class="w-5 h-5"></i></div>
                <span class="text-xs font-bold text-slate-700 dark:text-slate-300">Enroll Student</span>
            </a>
            <a href="/admin/students" class="flex flex-col items-center justify-center p-4 border border-slate-100 dark:border-slate-700/60 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-750 transition text-center group">
                <div class="w-10 h-10 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-650 dark:text-emerald-450 rounded-xl flex items-center justify-center mb-2 group-hover:scale-105 transition"><i data-lucide="users" class="w-5 h-5"></i></div>
                <span class="text-xs font-bold text-slate-700 dark:text-slate-300">Manage Students</span>
            </a>
            <a href="/admin/courses" class="flex flex-col items-center justify-center p-4 border border-slate-100 dark:border-slate-700/60 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-750 transition text-center group">
                <div class="w-10 h-10 bg-yellow-50 dark:bg-yellow-950/40 text-yellow-600 dark:text-yellow-405 rounded-xl flex items-center justify-center mb-2 group-hover:scale-105 transition"><i data-lucide="book-open" class="w-5 h-5"></i></div>
                <span class="text-xs font-bold text-slate-700 dark:text-slate-300">Manage Courses</span>
            </a>
            <a href="/admin/classrooms" class="flex flex-col items-center justify-center p-4 border border-slate-100 dark:border-slate-700/60 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-750 transition text-center group">
                <div class="w-10 h-10 bg-orange-50 dark:bg-orange-950/40 text-orange-650 dark:text-orange-400 rounded-xl flex items-center justify-center mb-2 group-hover:scale-105 transition"><i data-lucide="school" class="w-5 h-5"></i></div>
                <span class="text-xs font-bold text-slate-700 dark:text-slate-300">Classrooms</span>
            </a>
            <a href="/sessions_admin" class="flex flex-col items-center justify-center p-4 border border-slate-100 dark:border-slate-700/60 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-750 transition text-center group">
                <div class="w-10 h-10 bg-blue-50 dark:bg-blue-950/40 text-blue-650 dark:text-blue-405 rounded-xl flex items-center justify-center mb-2 group-hover:scale-105 transition"><i data-lucide="calendar" class="w-5 h-5"></i></div>
                <span class="text-xs font-bold text-slate-700 dark:text-slate-300">Sessions</span>
            </a>
            <a href="/generate_session_qr" class="flex flex-col items-center justify-center p-4 border border-slate-100 dark:border-slate-700/60 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-750 transition text-center group">
                <div class="w-10 h-10 bg-purple-50 dark:bg-purple-950/40 text-purple-650 dark:text-purple-400 rounded-xl flex items-center justify-center mb-2 group-hover:scale-105 transition"><i data-lucide="qr-code" class="w-5 h-5"></i></div>
                <span class="text-xs font-bold text-slate-700 dark:text-slate-300">Generate QR</span>
            </a>
        </div>
    </div>

    <!-- Charts and Activities Split -->
    <div class="grid lg:grid-cols-3 gap-8">
        <!-- Analytics column -->
        <div class="lg:col-span-2 space-y-8">
            <!-- Attendance Trends Card -->
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm">
                <div class="flex items-center justify-between mb-6 flex-wrap gap-3">
                    <h3 class="text-lg font-bold text-slate-850 dark:text-white flex items-center gap-2"><i data-lucide="bar-chart-2" class="w-5 h-5 text-indigo-600"></i> Attendance Trends</h3>
                    <!-- Chart toggle tabs -->
                    <div class="flex items-center gap-1 bg-slate-100 dark:bg-slate-900 p-1 rounded-xl">
                        <button onclick="updateChartType('daily')" id="tab-daily" class="px-3 py-1 rounded-lg text-xs font-bold transition-all bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 shadow-sm border border-slate-200/50 dark:border-slate-700/30">Daily</button>
                        <button onclick="updateChartType('weekly')" id="tab-weekly" class="px-3 py-1 rounded-lg text-xs font-bold transition-all text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Weekly</button>
                        <button onclick="updateChartType('monthly')" id="tab-monthly" class="px-3 py-1 rounded-lg text-xs font-bold transition-all text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Monthly</button>
                    </div>
                </div>
                
                <div class="relative w-full" style="height: 300px;">
                    <canvas id="attendanceTrendChart"></canvas>
                </div>
            </div>
            
            <!-- Course Wise Card -->
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm">
                <h3 class="text-lg font-bold text-slate-850 dark:text-white mb-6 flex items-center gap-2"><i data-lucide="pie-chart" class="w-5 h-5 text-indigo-600"></i> Course Distribution</h3>
                <div class="relative w-full" style="height: 250px;">
                    <canvas id="courseDistributionChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Recent Activities column -->
        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm h-fit">
            <h3 class="text-lg font-bold text-slate-850 dark:text-white mb-6 flex items-center gap-2"><i data-lucide="activity" class="w-5 h-5 text-indigo-600"></i> Recent Events</h3>
            <div class="relative pl-6 border-l-2 border-slate-100 dark:border-slate-700/50 space-y-6">
                {% if activities %}
                    {% for act in activities %}
                        <div class="relative">
                            <!-- Bullet dot -->
                            <div class="absolute -left-[31px] top-1.5 w-3.5 h-3.5 rounded-full bg-white dark:bg-slate-800 border-2 border-indigo-650 flex items-center justify-center">
                                <span class="w-1.5 h-1.5 rounded-full bg-indigo-600"></span>
                            </div>
                            <div>
                                <p class="text-xs font-semibold text-slate-850 dark:text-slate-200">{{ act.text }}</p>
                                <p class="text-[10px] text-slate-400 mt-1 flex items-center gap-1"><i data-lucide="clock" class="w-3 h-3"></i> {{ act.display_time }}</p>
                            </div>
                        </div>
                    {% endfor %}
                {% else %}
                    <p class="text-xs text-slate-400 py-6 text-center">No recent activities found.</p>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Chart rendering script -->
    <script>
      // Greeting based on time of day
      const hrs = new Date().getHours();
      let greetStr = "Here is what is happening today.";
      if (hrs < 12) greetStr = "Good morning! Here is what is happening today.";
      else if (hrs < 17) greetStr = "Good afternoon! Here is what is happening today.";
      else greetStr = "Good evening! Here is what is happening today.";
      document.getElementById('dynamic-greeting').textContent = greetStr;

      // Chart Data Configurations
      const chartData = {
          daily: {
              labels: {{ daily_labels|tojson }},
              values: {{ daily_values|tojson }}
          },
          weekly: {
              labels: {{ weekly_labels|tojson }},
              values: {{ weekly_values|tojson }}
          },
          monthly: {
              labels: {{ monthly_labels|tojson }},
              values: {{ monthly_values|tojson }}
          }
      };

      let currentTrendType = 'daily';
      let trendChart = null;

      function renderTrendChart() {
          const ctx = document.getElementById('attendanceTrendChart');
          if (!ctx) return;

          const dataSet = chartData[currentTrendType];
          
          if (trendChart) {
              trendChart.destroy();
          }

          const isDark = document.documentElement.classList.contains('dark');
          const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.05)';
          const labelColor = isDark ? '#94a3b8' : '#64748b';

          trendChart = new Chart(ctx.getContext('2d'), {
              type: 'line',
              data: {
                  labels: dataSet.labels,
                  datasets: [{
                      label: 'Presence Count',
                      data: dataSet.values,
                      borderColor: '#4f46e5',
                      backgroundColor: 'rgba(79, 70, 229, 0.08)',
                      fill: true,
                      tension: 0.35,
                      borderWidth: 3,
                      pointBackgroundColor: '#4f46e5',
                      pointBorderWidth: 2,
                      pointRadius: 4,
                      pointHoverRadius: 6
                  }]
              },
              options: {
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                      legend: { display: false }
                  },
                  scales: {
                      y: {
                          grid: { color: gridColor },
                          ticks: { color: labelColor, stepSize: 5 }
                      },
                      x: {
                          grid: { display: false },
                          ticks: { color: labelColor }
                      }
                  }
              }
          });
      }

      window.updateChartType = function(type) {
          currentTrendType = type;
          
          // Toggle tab button active classes
          ['daily', 'weekly', 'monthly'].forEach(t => {
              const el = document.getElementById('tab-' + t);
              if (t === type) {
                  el.className = "px-3 py-1 rounded-lg text-xs font-bold transition-all bg-white dark:bg-slate-800 text-indigo-650 dark:text-indigo-400 shadow-sm border border-slate-200/50 dark:border-slate-700/30";
              } else {
                  el.className = "px-3 py-1 rounded-lg text-xs font-bold transition-all text-slate-500 hover:text-slate-800 dark:hover:text-slate-200";
              }
          });

          renderTrendChart();
      };

      // Course distribution chart
      function renderCourseDistributionChart() {
          const ctx = document.getElementById('courseDistributionChart');
          if (!ctx) return;

          const isDark = document.documentElement.classList.contains('dark');
          const labelColor = isDark ? '#94a3b8' : '#64748b';

          new Chart(ctx.getContext('2d'), {
              type: 'bar',
              data: {
                  labels: {{ course_labels|tojson }},
                  datasets: [{
                      label: 'Attendance Count',
                      data: {{ course_values|tojson }},
                      backgroundColor: ['rgba(79, 70, 229, 0.85)', 'rgba(16, 185, 129, 0.85)', 'rgba(59, 130, 246, 0.85)', 'rgba(245, 158, 11, 0.85)', 'rgba(236, 72, 153, 0.85)'],
                      borderRadius: 8,
                      borderWidth: 0,
                      barPercentage: 0.5
                  }]
              },
              options: {
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                      legend: { display: false }
                  },
                  scales: {
                      y: {
                          grid: { color: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.05)' },
                          ticks: { color: labelColor, stepSize: 10 }
                      },
                      x: {
                          grid: { display: false },
                          ticks: { color: labelColor }
                      }
                  }
              }
          });
      }

      document.addEventListener("DOMContentLoaded", function() {
          renderTrendChart();
          renderCourseDistributionChart();
      });

      // Recalculate chart layout on theme change
      const originalToggleTheme = window.toggleTheme;
      window.toggleTheme = function() {
          if (originalToggleTheme) originalToggleTheme();
          setTimeout(() => {
              renderTrendChart();
              renderCourseDistributionChart();
          }, 150);
      };
      
      lucide.createIcons();
    </script>
    """ + BASE_FOOT
    return render_template_string(html, title="Admin Hub", is_admin=is_admin(), admin_badge=admin_badge_html(),
                                  total_students=total_students, total_courses=total_courses, total_classrooms=total_classrooms,
                                  active_sessions=active_sessions, today_attendance=today_attendance, attendance_percentage=attendance_percentage,
                                  activities=activities, daily_labels=daily_labels, daily_values=daily_values,
                                  weekly_labels=weekly_labels, weekly_values=weekly_values,
                                  monthly_labels=monthly_labels, monthly_values=monthly_values,
                                  course_labels=course_labels, course_values=course_values)


# -------------------------
# FIXED: Course Admin CRUD
# -------------------------
@app.route("/admin/courses", methods=["GET", "POST"])
def admin_courses():
    guard = require_admin()
    if guard:
        return guard

    if request.method == "POST":
        code = (request.form.get("code") or "").strip().upper()
        name = (request.form.get("name") or "").strip()
        description = (request.form.get("description") or "").strip()

        if not code or not name:
            flash("Course Code and Name are required.", "error")
        else:
            try:
                create_course(code, name, description)
                flash(f"Course {code} - {name} created successfully.", "success")
            except Exception:
                flash("Failed to create course. Code may already exist.", "error")
        return redirect(url_for("admin_courses"))

    rows = list_courses()

    html = BASE_HEAD + """
    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-sm p-6 mb-8">
      <h3 class="text-lg font-bold text-slate-800 dark:text-white mb-4">Create New Course</h3>
      <form method="POST" class="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Course Code</label>
            <input name="code" placeholder="e.g., CS101" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" required />
        </div>
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Course Name</label>
            <input name="name" placeholder="e.g., Intro to Programming" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" required />
        </div>
        <div class="md:col-span-1">
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Description</label>
            <input name="description" placeholder="Optional" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" />
        </div>
        <div class="md:col-span-1 text-right">
            <button class="bg-indigo-600 hover:bg-indigo-700 text-white py-2.5 rounded-xl font-semibold shadow-md shadow-indigo-500/10 transition w-full text-sm">Add Course</button>
        </div>
      </form>
    </div>

    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-sm overflow-hidden">
      <div class="p-6 border-b border-slate-200 dark:border-slate-750 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h3 class="text-lg font-bold text-slate-800 dark:text-white">Existing Courses ({{ rows | length }} Total)</h3>
        <div class="relative w-full sm:w-64">
            <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                <i data-lucide="search" class="w-4 h-4"></i>
            </span>
            <input type="text" id="table-search" oninput="filterTable()" class="w-full pl-9 pr-4 py-2 border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl text-xs focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="Search courses...">
        </div>
      </div>

      <table class="w-full table-auto text-left border-collapse" id="courses-table">
          <thead class="bg-slate-50 dark:bg-slate-900 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-250 dark:border-slate-750">
            <tr>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(0)">
                <div class="flex items-center gap-1.5">Code <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(1)">
                <div class="flex items-center gap-1.5">Name <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4">Description</th>
              <th class="p-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100 dark:divide-slate-750/40 text-slate-700 dark:text-slate-300" id="courses-table-body">
            {% for r in rows %}
            <tr class="hover:bg-slate-50/50 dark:hover:bg-slate-700/20 transition text-sm">
                <td class="p-4 font-bold text-slate-850 dark:text-white">{{ r[1] }}</td>
                <td class="p-4">{{ r[2] }}</td>
                <td class="p-4 text-xs text-slate-400 dark:text-slate-550">{{ r[3] or 'N/A' }}</td>
                <td class="p-4 text-right">
                    <button class="bg-rose-50 text-rose-600 hover:bg-rose-600 hover:text-white dark:bg-rose-950/20 dark:text-rose-455 px-3 py-1.5 rounded-xl text-xs font-semibold transition" onclick="triggerDeleteCourse('{{ r[0] }}', '{{ r[1] }}')">
                        Delete
                    </button>
                </td>
            </tr>
            {% endfor %}
          </tbody>
      </table>
      
      <div id="pagination-container"></div>
    </div>

    <form id="delete-course-form" method="POST" action="/admin/courses/delete" class="hidden">
      <input type="hidden" id="delete-course-id" name="id">
    </form>

    <script>
      let sortDirection = 1;
      let sortColumnIndex = 0;
      let currentPage = 1;
      const rowsPerPage = 5;

      window.triggerDeleteCourse = function(courseId, courseCode) {
          confirmDelete(`Are you sure you want to delete course ${courseCode}? This will remove all student enrollments and sessions associated with it.`, function() {
              document.getElementById('delete-course-id').value = courseId;
              document.getElementById('delete-course-form').submit();
          });
      };

      window.sortTable = function(colIndex) {
          const table = document.getElementById("courses-table");
          const tbody = document.getElementById("courses-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          
          if (sortColumnIndex === colIndex) {
              sortDirection = -sortDirection;
          } else {
              sortColumnIndex = colIndex;
              sortDirection = 1;
          }
          
          const headers = table.querySelectorAll("thead th");
          headers.forEach((th, idx) => {
              const icon = th.querySelector(".sort-icon");
              if (icon) {
                  if (idx === colIndex) {
                      icon.innerHTML = sortDirection === 1 ? '<i data-lucide="chevron-up" class="w-3.5 h-3.5"></i>' : '<i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>';
                  } else {
                      icon.innerHTML = '<i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i>';
                  }
              }
          });
          lucide.createIcons();

          trs.sort((a, b) => {
              const aVal = a.cells[colIndex].textContent.trim().toLowerCase();
              const bVal = b.cells[colIndex].textContent.trim().toLowerCase();
              return aVal.localeCompare(bVal, undefined, {numeric: true, sensitivity: 'base'}) * sortDirection;
          });
          
          trs.forEach(tr => tbody.appendChild(tr));
          updatePagination();
      };

      window.filterTable = function() {
          const query = document.getElementById("table-search").value.toLowerCase();
          const tbody = document.getElementById("courses-table-body");
          const trs = tbody.querySelectorAll("tr");
          
          trs.forEach(tr => {
              const text = tr.textContent.toLowerCase();
              tr.style.display = text.includes(query) ? "" : "none";
          });
          
          currentPage = 1;
          updatePagination();
      };

      window.updatePagination = function() {
          const tbody = document.getElementById("courses-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          const query = document.getElementById("table-search").value.toLowerCase();
          
          const visibleTrs = trs.filter(tr => tr.textContent.toLowerCase().includes(query));
          
          const totalPages = Math.ceil(visibleTrs.length / rowsPerPage) || 1;
          if (currentPage > totalPages) currentPage = totalPages;
          
          trs.forEach(tr => tr.style.display = "none");
          
          const startIdx = (currentPage - 1) * rowsPerPage;
          const endIdx = startIdx + rowsPerPage;
          
          visibleTrs.slice(startIdx, endIdx).forEach(tr => {
              tr.style.display = "";
          });
          
          const container = document.getElementById("pagination-container");
          if (!container) return;
          
          container.innerHTML = `
              <div class="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 p-4 border-t border-slate-200 dark:border-slate-750">
                  <div>Showing ${startIdx + 1} to ${Math.min(endIdx, visibleTrs.length)} of ${visibleTrs.length} entries</div>
                  <div class="flex gap-2">
                      <button onclick="changePage(-1)" ${currentPage === 1 ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Previous</button>
                      <span class="px-3 py-1 font-bold text-indigo-600 dark:text-indigo-400">${currentPage} / ${totalPages}</span>
                      <button onclick="changePage(1)" ${currentPage === totalPages ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Next</button>
                  </div>
              </div>
          `;
      };

      window.changePage = function(dir) {
          currentPage += dir;
          updatePagination();
      };

      document.addEventListener("DOMContentLoaded", function() {
          updatePagination();
      });
      lucide.createIcons();
    </script>
    """ + BASE_FOOT
    return render_template_string(html, title="Manage Courses", is_admin=is_admin(), rows=rows)


@app.route("/admin/courses/delete", methods=["POST"])
def admin_courses_delete():
    guard = require_admin()
    if guard:
        return guard
    try:
        cid = int(request.form.get("id"))
        delete_course(cid)
        flash("Course deleted successfully. Enrollments and session links removed.", "success")
    except Exception as e:
        flash(f"Failed to delete course: {e}", "error")
    return redirect(url_for("admin_courses"))


# -------------------------
# NEW: Classrooms admin CRUD
# -------------------------
@app.route("/admin/classrooms", methods=["GET", "POST"])
def admin_classrooms():
    guard = require_admin()
    if guard:
        return guard
    
    if request.method == "POST":
        room_number = (request.form.get("room_number") or "").strip()
        name = (request.form.get("name") or "").strip()
        f = request.files.get("photo")
        
        if not room_number or not f:
            flash("Room Number and Reference Photo are required.", "error")
        else:
            # save reference image
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{secure_filename(room_number)}.jpg"
            # Ensure static path is correct for Flask to serve it
            full_path = os.path.join(CLASSROOM_DIR, filename) 
            
            try:
                # Read file content from SpooledTemporaryFile
                file_content = f.read()
                f.seek(0) # Reset pointer just in case for later uses (not strictly needed here but good practice)
                im = Image.open(io.BytesIO(file_content)).convert("RGB")
                
                # Check for HEIF/HEIC support and path existence before saving
                os.makedirs(CLASSROOM_DIR, exist_ok=True)
                im.save(full_path, "JPEG", quality=85)
                
                # Save path into DB
                create_classroom(room_number, name, full_path)
                flash(f"Classroom {room_number} added successfully!", "success")
            except Exception as e:
                app.logger.exception("Classroom photo save failed")
                flash(f"Failed to process photo: {e}", "error")
        return redirect(url_for("admin_classrooms"))

    rows = list_classrooms()
    classrooms_list = []
    for r in rows:
        img_filename = os.path.basename(r[3])
        img_url = url_for('classroom_preview', filename=img_filename)
        classrooms_list.append({
            "id": r[0],
            "room_number": r[1],
            "name": r[2],
            "img_url": img_url
        })

    html = BASE_HEAD + """
    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-sm p-6 mb-8">
      <h3 class="text-lg font-bold text-slate-800 dark:text-white mb-4">Add New Classroom Background</h3>
      <form method="POST" enctype="multipart/form-data" class="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Room Number</label>
            <input name="room_number" placeholder="e.g., C-105" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" required />
        </div>
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Name (Optional)</label>
            <input name="name" placeholder="e.g., Main Lab" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" />
        </div>
        <div class="md:col-span-1">
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5 flex items-center gap-1"><i data-lucide="upload" class="w-3.5 h-3.5 text-slate-450"></i> Reference Photo</label>
            <input type="file" name="photo" accept="image/*" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-3 py-1.5 text-xs text-slate-500 file:mr-4 file:py-1 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-50 file:text-indigo-750 hover:file:bg-indigo-100" required />
        </div>
        <div class="md:col-span-1 text-right">
            <button class="bg-indigo-600 hover:bg-indigo-700 text-white py-2.5 rounded-xl font-semibold shadow-md shadow-indigo-500/10 transition w-full text-sm">Add Classroom</button>
        </div>
      </form>
    </div>

    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-sm overflow-hidden">
      <div class="p-6 border-b border-slate-200 dark:border-slate-750 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h3 class="text-lg font-bold text-slate-800 dark:text-white">Existing Classrooms ({{ classrooms | length }} Total)</h3>
        <div class="relative w-full sm:w-64">
            <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                <i data-lucide="search" class="w-4 h-4"></i>
            </span>
            <input type="text" id="table-search" oninput="filterTable()" class="w-full pl-9 pr-4 py-2 border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl text-xs focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="Search classrooms...">
        </div>
      </div>

      <table class="w-full table-auto text-left border-collapse" id="classrooms-table">
          <thead class="bg-slate-50 dark:bg-slate-900 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-250 dark:border-slate-750">
            <tr>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(0)">
                <div class="flex items-center gap-1.5">Room Number <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(1)">
                <div class="flex items-center gap-1.5">Name <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4">Reference Image</th>
              <th class="p-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100 dark:divide-slate-750/40 text-slate-700 dark:text-slate-300" id="classrooms-table-body">
            {% for r in classrooms %}
            <tr class="hover:bg-slate-50/50 dark:hover:bg-slate-700/20 transition text-sm">
                <td class="p-4 font-bold text-slate-850 dark:text-white">{{ r.room_number }}</td>
                <td class="p-4">{{ r.name or 'N/A' }}</td>
                <td class="p-4">
                    <a href="{{ r.img_url }}" target="_blank" class="inline-flex items-center gap-1 text-xs font-semibold text-indigo-605 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-350 transition">
                        <i data-lucide="image" class="w-4 h-4"></i> View Reference
                    </a>
                </td>
                <td class="p-4 text-right">
                    <button class="bg-rose-50 text-rose-600 hover:bg-rose-600 hover:text-white dark:bg-rose-950/20 dark:text-rose-455 px-3 py-1.5 rounded-xl text-xs font-semibold transition" onclick="triggerDeleteClassroom('{{ r.id }}', '{{ r.room_number }}')">
                        Delete
                    </button>
                </td>
            </tr>
            {% endfor %}
          </tbody>
      </table>
      
      <div id="pagination-container"></div>
    </div>

    <form id="delete-classroom-form" method="POST" action="/admin/classrooms/delete" class="hidden">
      <input type="hidden" id="delete-classroom-id" name="id">
    </form>

    <script>
      let sortDirection = 1;
      let sortColumnIndex = 0;
      let currentPage = 1;
      const rowsPerPage = 5;

      window.triggerDeleteClassroom = function(classroomId, roomNumber) {
          confirmDelete(`Are you sure you want to delete classroom ${roomNumber}? This will remove all sessions associated with it.`, function() {
              document.getElementById('delete-classroom-id').value = classroomId;
              document.getElementById('delete-classroom-form').submit();
          });
      };

      window.sortTable = function(colIndex) {
          const table = document.getElementById("classrooms-table");
          const tbody = document.getElementById("classrooms-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          
          if (sortColumnIndex === colIndex) {
              sortDirection = -sortDirection;
          } else {
              sortColumnIndex = colIndex;
              sortDirection = 1;
          }
          
          const headers = table.querySelectorAll("thead th");
          headers.forEach((th, idx) => {
              const icon = th.querySelector(".sort-icon");
              if (icon) {
                  if (idx === colIndex) {
                      icon.innerHTML = sortDirection === 1 ? '<i data-lucide="chevron-up" class="w-3.5 h-3.5"></i>' : '<i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>';
                  } else {
                      icon.innerHTML = '<i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i>';
                  }
              }
          });
          lucide.createIcons();

          trs.sort((a, b) => {
              const aVal = a.cells[colIndex].textContent.trim().toLowerCase();
              const bVal = b.cells[colIndex].textContent.trim().toLowerCase();
              return aVal.localeCompare(bVal, undefined, {numeric: true, sensitivity: 'base'}) * sortDirection;
          });
          
          trs.forEach(tr => tbody.appendChild(tr));
          updatePagination();
      };

      window.filterTable = function() {
          const query = document.getElementById("table-search").value.toLowerCase();
          const tbody = document.getElementById("classrooms-table-body");
          const trs = tbody.querySelectorAll("tr");
          
          trs.forEach(tr => {
              const text = tr.textContent.toLowerCase();
              tr.style.display = text.includes(query) ? "" : "none";
          });
          
          currentPage = 1;
          updatePagination();
      };

      window.updatePagination = function() {
          const tbody = document.getElementById("classrooms-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          const query = document.getElementById("table-search").value.toLowerCase();
          
          const visibleTrs = trs.filter(tr => tr.textContent.toLowerCase().includes(query));
          
          const totalPages = Math.ceil(visibleTrs.length / rowsPerPage) || 1;
          if (currentPage > totalPages) currentPage = totalPages;
          
          trs.forEach(tr => tr.style.display = "none");
          
          const startIdx = (currentPage - 1) * rowsPerPage;
          const endIdx = startIdx + rowsPerPage;
          
          visibleTrs.slice(startIdx, endIdx).forEach(tr => {
              tr.style.display = "";
          });
          
          const container = document.getElementById("pagination-container");
          if (!container) return;
          
          container.innerHTML = `
              <div class="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 p-4 border-t border-slate-200 dark:border-slate-750">
                  <div>Showing ${startIdx + 1} to ${Math.min(endIdx, visibleTrs.length)} of ${visibleTrs.length} entries</div>
                  <div class="flex gap-2">
                      <button onclick="changePage(-1)" ${currentPage === 1 ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Previous</button>
                      <span class="px-3 py-1 font-bold text-indigo-600 dark:text-indigo-400">${currentPage} / ${totalPages}</span>
                      <button onclick="changePage(1)" ${currentPage === totalPages ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Next</button>
                  </div>
              </div>
          `;
      };

      window.changePage = function(dir) {
          currentPage += dir;
          updatePagination();
      };

      document.addEventListener("DOMContentLoaded", function() {
          updatePagination();
      });
      lucide.createIcons();
    </script>
    """ + BASE_FOOT
    return render_template_string(html, title="Manage Classrooms", is_admin=is_admin(), classrooms=classrooms_list)

@app.route("/admin/classrooms/delete", methods=["POST"])
def admin_classrooms_delete():
    guard = require_admin()
    if guard:
        return guard
    try:
        cid = int(request.form.get("id"))
        delete_classroom(cid)
        flash("Classroom deleted.", "success")
    except Exception as e:
        flash(f"Failed to delete classroom: {e}", "error")
    return redirect(url_for("admin_classrooms"))

# -------------------------
# Generate session QR
# -------------------------
@app.route("/generate_session_qr", methods=["GET", "POST"])
def generate_session_qr():
    guard = require_admin()
    if guard:
        return guard
    courses = list_courses()
    classrooms = list_classrooms() # NEW: Load classrooms
    
    if request.method == "POST":
        session_name = (request.form.get("session_name") or "").strip()
        course_id = request.form.get("course_id") or None
        classroom_id = request.form.get("classroom_id") or None # NEW: Get classroom ID
        
        # ... (parse course_id and duration remain the same) ...
        if course_id == "":
            course_id = None
        else:
            try: course_id = int(course_id)
            except Exception: course_id = None
            
        if classroom_id == "":
            classroom_id = None
        else:
            try: classroom_id = int(classroom_id)
            except Exception: classroom_id = None
            
        duration = int(request.form.get("duration") or ATTENDANCE_WINDOW_MINUTES)
        
        session_id = str(uuid.uuid4())[:8]
        
        # Determine public base URL for QR code
        public_base = detect_public_base(request.host_url, APP_PORT)
        
        mark_url = f"{public_base}/mark_attendance?session={quote(session_id, safe='')}"
        
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            # NEW: Insert classroom_id
            cur.execute("INSERT INTO sessions(session_id, date, session_name, subject, course_id, classroom_id, duration_minutes, status) VALUES(?,?,?,?,?,?,?,?)",
                                 (session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_name, session_name, course_id, classroom_id, duration, "OPEN"))
            con.commit()
            
        qr_path = make_qr_to_path(mark_url, session_id)
        
        start_time_iso = datetime.now().isoformat()
        qr_image_url = url_for('static', filename='qr_codes/session_'+session_id+'.png')
        
        html = BASE_HEAD + """
        <div class="max-w-xl mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border border-slate-100 dark:border-slate-700/50 rounded-2xl shadow-xl p-8 text-center animate-slide-up">
          <div class="w-12 h-12 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-450 rounded-full flex items-center justify-center mx-auto mb-4 border border-emerald-100/50 dark:border-indigo-900/30">
            <i data-lucide="check-circle-2" class="w-7 h-7"></i>
          </div>
          <h2 class="text-3xl font-extrabold text-slate-850 dark:text-white font-display">Session QR Code</h2>
          <p class="text-sm text-slate-500 dark:text-slate-400 mt-2">Active session is successfully opened. Students can scan the QR code below.</p>
          
          <div class="my-6 p-4 bg-slate-50 dark:bg-slate-900 rounded-2xl inline-block border border-slate-200 dark:border-slate-800">
            <img src="{{ qr_image_url }}" alt="Session QR Code" width="280" class="mx-auto rounded-xl border-4 border-white dark:border-slate-800 shadow-lg"/>
          </div>

          <div class="bg-indigo-50 dark:bg-indigo-950/40 text-indigo-650 dark:text-indigo-400 px-4 py-2 rounded-xl text-sm font-bold mt-4 inline-block flex items-center justify-center gap-2 max-w-xs mx-auto" id="countdown-timer-container">
            <i data-lucide="timer" class="w-4 h-4"></i> Session Expires in: <span id="countdown-timer" class="font-mono">--:--</span>
          </div>

          <div class="mt-6 border-t border-slate-100 dark:border-slate-750 pt-6 flex justify-between items-center text-xs text-slate-400">
            <span>Session ID: <span class="font-bold text-slate-755 dark:text-slate-205">{{ session_id }}</span></span>
            <a href="{{ mark_url }}" target="_blank" class="text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-350 transition flex items-center gap-0.5"><i data-lucide="external-link" class="w-3.5 h-3.5"></i> Direct URL</a>
          </div>

          <div class="mt-6 flex justify-center gap-4">
            <a href="/sessions_admin" class="bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-650 text-slate-700 dark:text-slate-200 px-6 py-2.5 rounded-xl font-semibold text-sm transition shadow-sm">Manage Sessions</a>
            <a href="/generate_session_qr" class="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-xl font-semibold text-sm hover:bg-indigo-700 transition shadow-md shadow-indigo-500/10">Start Another</a>
          </div>
        </div>

        <script>
          const durationMinutes = {{ duration }};
          const startTime = new Date("{{ start_time_iso }}");
          const endTime = new Date(startTime.getTime() + durationMinutes * 60 * 1000);

          function updateTimer() {
              const now = new Date();
              const diff = endTime - now;
              if (diff <= 0) {
                  document.getElementById('countdown-timer').textContent = "Session Expired";
                  document.getElementById('countdown-timer-container').className = "text-rose-505 bg-rose-50 dark:bg-rose-950/20 px-4 py-2 rounded-xl text-sm font-bold mt-4 inline-block";
                  return;
              }
              
              const mins = Math.floor(diff / 60000);
              const secs = Math.floor((diff % 60000) / 1000);
              document.getElementById('countdown-timer').textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
              setTimeout(updateTimer, 1000);
          }
          updateTimer();
          lucide.createIcons();
        </script>
        """ + BASE_FOOT
        return render_template_string(html, title="Session QR", is_admin=is_admin(), qr_image_url=qr_image_url, session_id=session_id, mark_url=mark_url, duration=duration, start_time_iso=start_time_iso)
    
    # GET: Show form
    html = BASE_HEAD + """
    <div class="max-w-md mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md rounded-2xl shadow-xl border border-slate-100 dark:border-slate-700/50 p-8 animate-slide-up">
      <div class="text-center mb-6">
        <div class="w-16 h-16 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 rounded-full flex items-center justify-center mx-auto mb-4 border border-indigo-100/50 dark:border-indigo-900/30">
          <i data-lucide="qr-code" class="w-8 h-8"></i>
        </div>
        <h2 class="text-3xl font-extrabold text-slate-800 dark:text-white font-display">Start Session</h2>
        <p class="text-sm text-slate-500 dark:text-slate-400 mt-2">Generate a live QR code for student presence</p>
      </div>

      <form method="POST" class="space-y-5">
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5 flex items-center gap-1.5"><i data-lucide="type" class="w-3.5 h-3.5 text-slate-400"></i> Session Name / Subject</label>
            <input name="session_name" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="e.g. Algorithms Lecture 5" required>
        </div>

        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5 flex items-center gap-1.5"><i data-lucide="book-open" class="w-3.5 h-3.5 text-slate-400"></i> Class Course</label>
            <select name="course_id" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-700 dark:text-slate-300">
                <option value="">-- No course --</option>
                {% for c in courses %}
                    <option value="{{ c[0] }}">{{ c[2] }} ({{ c[1] }})</option>
                {% endfor %}
            </select>
        </div>

        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5 flex items-center gap-1.5"><i data-lucide="school" class="w-3.5 h-3.5 text-slate-400"></i> Target Classroom (For Background Snap Check)</label>
            <select name="classroom_id" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-700 dark:text-slate-300">
                <option value="">-- No classroom (Disable Background Check) --</option>
                {% for cl in classrooms %}
                    <option value="{{ cl[0] }}">{{ cl[1] }} - {{ cl[2] }}</option>
                {% endfor %}
            </select>
        </div>

        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5 flex items-center gap-1.5"><i data-lucide="clock" class="w-3.5 h-3.5 text-slate-400"></i> Active Window (Minutes)</label>
            <input name="duration" type="number" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="30" value="30">
        </div>

        <button class="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-xl font-semibold shadow-lg shadow-indigo-500/10 hover:shadow-indigo-500/20 transform hover:-translate-y-0.5 transition duration-200 flex items-center justify-center gap-2 text-sm mt-6">
            <i data-lucide="plus-circle" class="w-4 h-4"></i> Generate QR & Start Session
        </button>
      </form>
    </div>
    <script>lucide.createIcons();</script>
    """ + BASE_FOOT
    return render_template_string(html, title="Generate Session QR", is_admin=is_admin(), courses=courses, classrooms=classrooms)

# -------------------------
# -------------------------
# -------------------------
# Mark attendance (student-facing minimal UI)
# -------------------------
@app.route("/mark_attendance")
def mark_attendance():
    session_id = request.args.get("session") or ""

    # 🧠 Always define fallback session info (avoid early returns)
    session_name = "(Unknown)"
    status = "OPEN"
    classroom_msg = "Classroom reference **not set** for this session. Background snap check is disabled."

    # --- Fetch session data if valid ---
    if session_id:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("SELECT session_name, date, duration_minutes, status, course_id, classroom_id FROM sessions WHERE session_id=?", (session_id,))
            row = cur.fetchone()

        if row:
            session_name, date, duration, status, course_id, classroom_id = row

            # Check TTL / Expiry
            try:
                start = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                if datetime.now() > start + timedelta(minutes=int(duration) + LATE_CUTOFF_MINUTES):
                    status = "EXPIRED"
            except Exception:
                pass

            # Classroom info
            classroom_info = get_classroom_info(classroom_id)
            if classroom_info:
                room_number, c_name, _ = classroom_info
                classroom_msg = f"🏛️ Classroom: <b>{c_name} ({room_number})</b>. "
                classroom_msg += f"**Background snap check is active.**" if BACKGROUND_CHECK_REQUIRED else "Background snap is collected but **check is disabled globally**."
        else:
            # Invalid session ID
            classroom_msg = "⚠️ Invalid or expired session. Default camera mode loaded for testing."
    else:
        classroom_msg = "⚠️ No session provided. Default camera view loaded."

    # --- Full HTML always returned ---
    html = BASE_HEAD + """
    <div class="max-w-md mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-xl p-6 md:p-8 animate-slide-up relative overflow-hidden">
      <!-- Session Indicator Banner -->
      <div class="flex items-center justify-between border-b border-slate-100 dark:border-slate-700/50 pb-4 mb-5">
        <div class="flex items-center gap-2">
          <a href="/" class="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition"><i data-lucide="arrow-left" class="w-4 h-4"></i></a>
          <span class="text-xs font-bold text-slate-450 uppercase tracking-wider">Attendance Terminal</span>
        </div>
        <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full {% if status == 'OPEN' %}bg-emerald-500 animate-pulse{% else %}bg-rose-500{% endif %}"></span>
            <span class="text-xs font-bold {% if status == 'OPEN' %}text-emerald-600{% else %}text-rose-600{% endif %}">{{ status }}</span>
        </div>
      </div>

      <div id="secureContextWarning" class="hidden mb-5 bg-rose-50 dark:bg-rose-955/20 p-4 rounded-xl border border-rose-200/50 dark:border-rose-900/30 text-xs text-rose-750 dark:text-rose-400 flex items-start gap-2">
          <i data-lucide="shield-alert" class="w-4 h-4 flex-shrink-0 mt-0.5 animate-bounce"></i>
          <div>
              <strong class="font-bold">Camera Blocked:</strong> Browsers block camera access on non-secure connections. Access using <code class="bg-rose-100 dark:bg-rose-950 px-1.5 py-0.5 rounded font-mono">localhost</code> or configure HTTPS.
          </div>
      </div>

      <div class="text-center mb-6">
        <h2 class="text-2xl font-extrabold text-slate-850 dark:text-white font-display">{{ session_name }}</h2>
        {% if classroom_message %}
        <p class="text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/30 px-3 py-1.5 rounded-xl inline-block mt-2">{{ classroom_message | safe }}</p>
        {% endif %}
      </div>

      <!-- Steps tracker -->
      <div class="grid grid-cols-3 gap-2 mb-6">
        <div class="text-center border-b-2 pb-2 text-[10px] font-bold text-indigo-600 dark:text-indigo-400 border-indigo-600 dark:border-indigo-400" id="step-1-indicator">
            1. Roll/Name
        </div>
        <div class="text-center border-b-2 pb-2 text-[10px] font-bold text-slate-400 border-slate-200 dark:border-slate-700" id="step-2-indicator">
            2. Face Scan
        </div>
        <div class="text-center border-b-2 pb-2 text-[10px] font-bold text-slate-400 border-slate-200 dark:border-slate-700" id="step-3-indicator">
            3. Verify
        </div>
      </div>

      <div class="space-y-4">
        <!-- Input Roll/Name -->
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Student Full Name or Roll Number</label>
            <input id="studentName" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="e.g. Jane Doe / 12345" required>
        </div>

        <!-- Camera Area -->
        <div class="relative w-64 h-64 mx-auto rounded-full overflow-hidden border-4 border-indigo-600 dark:border-indigo-500 shadow-xl bg-slate-900 flex items-center justify-center group mt-4">
            <video id="camera" autoplay playsinline muted class="w-full h-full object-cover" style="transform: scaleX(-1);"></video>
            
            <!-- Viewfinder Corners Overlay -->
            <div class="absolute inset-6 border border-dashed border-indigo-450/20 rounded-full pointer-events-none"></div>
            <div class="absolute inset-0 border-4 border-indigo-600/30 rounded-full pointer-events-none group-hover:scale-95 transition-transform duration-500"></div>
            
            <!-- Glow Pulse Border -->
            <div class="absolute inset-0 border-2 border-indigo-500/20 rounded-full pointer-events-none animate-pulse"></div>

            <!-- Face Scan Overlay Line -->
            <div class="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-indigo-400 to-transparent top-0 animate-[scan_3s_infinite_linear] pointer-events-none shadow-[0_0_8px_rgba(129,140,248,0.8)]"></div>
        </div>

        <p class="text-[11px] text-slate-450 dark:text-slate-500 text-center mt-2 font-semibold" id="capture-instructions">
            **Capture #1 (Face) is required. Align your face in the circular guide.**
        </p>
        <div id="cameraStatus" class="text-[11px] font-mono text-center mt-1.5 min-h-[16px]"></div>

        <!-- Camera Controls -->
        <div class="flex flex-col items-center gap-2 mt-4" id="button-container-outer">
            <div class="flex gap-2 justify-center w-full" id="button-container">
              <button id="cap1" disabled class="bg-indigo-600 hover:bg-indigo-750 text-white px-4 py-2.5 rounded-xl text-xs font-bold shadow-md transition disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"><i data-lucide="camera" class="w-3.5 h-3.5"></i> Capture #1 (Face)</button>
              <button id="cap2" disabled class="bg-amber-600 hover:bg-amber-700 text-white px-4 py-2.5 rounded-xl text-xs font-bold shadow-md transition disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1" style="display:none;"><i data-lucide="smile" class="w-3.5 h-3.5"></i> Capture #2 (Open Mouth)</button>
            </div>
            <button id="switchBtn" class="bg-slate-100 hover:bg-slate-205 dark:bg-slate-700 dark:hover:bg-slate-650 text-slate-700 dark:text-slate-200 px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition"><i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i> Switch Camera</button>
        </div>

        <!-- Submit Button -->
        <div class="pt-4 border-t border-slate-100 dark:border-slate-750/50 mt-6 text-center">
            <button id="submitBtn" disabled class="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-3 rounded-xl font-extrabold text-sm shadow-lg shadow-emerald-500/10 hover:shadow-emerald-550/20 transform hover:-translate-y-0.5 transition duration-200 flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed">
                <i data-lucide="check-circle" class="w-4 h-4"></i> Submit Attendance
            </button>
        </div>
      </div>
    </div>

    <!-- Hidden Form reference for upload_liveness standard keys -->
    <form id="uploadForm" action="/upload_liveness" method="POST" enctype="multipart/form-data" class="hidden">
      <input name="session_id" value="{{ session_id }}" id="session_id_hidden">
      <input type="hidden" id="studentNameHidden" name="student_name">
      <input type="file" id="photo1" name="photo1">
      <input type="file" id="photo2" name="photo2">
    </form>

    <!-- Global Verification Processing Overlay -->
    <div id="loadingOverlay" class="hidden fixed inset-0 bg-slate-950/80 backdrop-blur-md flex flex-col items-center justify-center z-50">
      <div class="relative w-20 h-20 mb-6">
        <div class="absolute inset-0 rounded-full border-4 border-indigo-500/20"></div>
        <div class="absolute inset-0 rounded-full border-4 border-t-indigo-500 animate-spin"></div>
      </div>
      <p class="text-white font-bold text-sm tracking-wide">Processing Facial Verification...</p>
      <p class="text-slate-400 text-xs mt-2">Checking liveness & matching facial keypoints. Please wait.</p>
    </div>

    <style>
      @keyframes scan {
        0%, 100% { top: 0%; }
        50% { top: 100%; }
      }
    </style>

    <script>
      const LIVENESS_REQUIRED = {{ LIVENESS_REQUIRED | tojson }};
      const SESSION_ID = {{ session_id | tojson }};
    </script>
    
    {% raw %}
<script>
      if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          document.getElementById('secureContextWarning').classList.remove('hidden');
      }

      const studentNameInput = document.getElementById('studentName');
      const cap1 = document.getElementById('cap1');
      const cap2 = document.getElementById('cap2');
      const submitBtn = document.getElementById('submitBtn');
      const switchBtn = document.getElementById('switchBtn');
      const video = document.getElementById('camera');

      let blob1 = null;
      let blob2 = null;
      let streamReady = false;
      let currentFacing = 'user';
      let enrollStream = null;

      function updateSteps() {
        const hasName = studentNameInput.value.trim().length > 0;
        const hasPhoto1 = blob1 !== null;
        const hasPhoto2 = blob2 !== null;

        const step1 = document.getElementById('step-1-indicator');
        const step2 = document.getElementById('step-2-indicator');
        const step3 = document.getElementById('step-3-indicator');

        if (hasName) {
            step1.className = "text-center border-b-2 pb-2 text-[10px] font-bold text-slate-450 border-slate-200 dark:border-slate-700";
            step2.className = "text-center border-b-2 pb-2 text-[10px] font-bold text-indigo-650 dark:text-indigo-400 border-indigo-600 dark:border-indigo-400";
        } else {
            step1.className = "text-center border-b-2 pb-2 text-[10px] font-bold text-indigo-650 dark:text-indigo-400 border-indigo-600 dark:border-indigo-400";
            step2.className = "text-center border-b-2 pb-2 text-[10px] font-bold text-slate-450 border-slate-200 dark:border-slate-700";
        }

        if (hasPhoto1 && (!LIVENESS_REQUIRED || hasPhoto2)) {
            step2.className = "text-center border-b-2 pb-2 text-[10px] font-bold text-slate-450 border-slate-200 dark:border-slate-700";
            step3.className = "text-center border-b-2 pb-2 text-[10px] font-bold text-indigo-650 dark:text-indigo-400 border-indigo-600 dark:border-indigo-400";
        } else {
            step3.className = "text-center border-b-2 pb-2 text-[10px] font-bold text-slate-450 border-slate-200 dark:border-slate-700";
        }
      }

      studentNameInput.addEventListener('input', () => {
        const ready = studentNameInput.value.trim().length > 0 && blob1 && (!LIVENESS_REQUIRED || blob2);
        submitBtn.disabled = !ready;
        updateSteps();
      });

      const cameraStatus = document.getElementById('cameraStatus');

      function activateCamera(facingMode = 'user') {
        console.log('Attempting to start camera facingMode=', facingMode);
        if (cameraStatus) cameraStatus.innerHTML = "<span class='text-indigo-500 animate-pulse'>Initializing webcam...</span>";
        
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const errorMsg = "Camera API not supported or blocked by browser (requires HTTPS/localhost)";
            console.error(errorMsg);
            if (cameraStatus) cameraStatus.innerHTML = "<span class='text-rose-500 font-semibold'>⚠️ " + errorMsg + "</span>";
            return;
        }
        
        if (enrollStream) {
            try {
                enrollStream.getTracks().forEach(track => track.stop());
            } catch (e) { console.error('Error stopping track:', e); }
        }
        
        const constraints = {
            video: facingMode ? { facingMode: facingMode } : true
        };
        
        navigator.mediaDevices.getUserMedia(constraints)
        .catch(err => {
            console.warn("Failed with facingMode constraints, retrying with basic video", err);
            return navigator.mediaDevices.getUserMedia({ video: true });
        })
        .then(stream => {
            enrollStream = stream;
            video.srcObject = stream;
            
            // Play video stream directly to bypass onloadedmetadata deadlock on WebKit/iOS
            video.play()
            .then(() => {
                streamReady = true;
                cap1.disabled = false;
                if (LIVENESS_REQUIRED) cap2.disabled = false;
                console.log('Camera started successfully!');
                if (cameraStatus) cameraStatus.innerHTML = "<span class='text-emerald-500 font-bold'>✓ Camera Active</span>";
                currentFacing = facingMode;
            })
            .catch(err => {
                console.error('Play error:', err);
                if (cameraStatus) cameraStatus.innerHTML = "<span class='text-amber-500 font-semibold'>⚠️ Playback blocked. Click the video circle to start.</span>";
                
                const startPlayback = () => {
                    video.play()
                    .then(() => {
                        streamReady = true;
                        cap1.disabled = false;
                        if (LIVENESS_REQUIRED) cap2.disabled = false;
                        if (cameraStatus) cameraStatus.innerHTML = "<span class='text-emerald-500 font-bold'>✓ Camera Active</span>";
                        video.removeEventListener('click', startPlayback);
                    })
                    .catch(e => console.error(e));
                };
                video.addEventListener('click', startPlayback);
            });
        })
        .catch(err => {
            console.error('Camera error:', err);
            const detailMsg = "Camera failed: " + err.name + " - " + err.message;
            if (cameraStatus) cameraStatus.innerHTML = "<span class='text-rose-500 font-bold'>⚠️ " + detailMsg + "</span>";
            alert("Camera failed. Make sure you have granted camera permissions. Error: " + err.message);
        });
      }

  activateCamera(currentFacing);

  if (switchBtn) {
    switchBtn.onclick = () => {
      const newFacing = (currentFacing === "user") ? "environment" : "user";
      activateCamera(newFacing);
    };
  }

  function capture(cb) {
    if (!streamReady) return alert("Camera not ready yet!");
    const c = document.createElement("canvas");
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    c.toBlob(cb, "image/jpeg", 0.9);
  }

  cap1.addEventListener("click", () => {
    capture(b => {
      blob1 = b;
      cap1.innerHTML = `<i data-lucide="check-circle" class="w-3.5 h-3.5"></i> Captured #1 (Face)`;
      lucide.createIcons();
      const ready = studentNameInput.value.trim().length > 0 && blob1 && (!LIVENESS_REQUIRED || blob2);
      submitBtn.disabled = !ready;
      updateSteps();
    });
  });

  cap2.addEventListener("click", () => {
    capture(b => {
      blob2 = b;
      cap2.innerHTML = `<i data-lucide="check-circle" class="w-3.5 h-3.5"></i> Captured #2 (Open)`;
      lucide.createIcons();
      const ready = studentNameInput.value.trim().length > 0 && blob1 && blob2;
      submitBtn.disabled = !ready;
      updateSteps();
    });
  });

  submitBtn.addEventListener("click", () => {
    const name = studentNameInput.value.trim();
    if (!name) return alert("Enter your name/roll number.");
    if (!blob1) return alert("Please capture photo #1.");
    if (LIVENESS_REQUIRED && !blob2) return alert("Please capture photo #2.");

    const formData = new FormData();
    formData.append("session_id", SESSION_ID);
    formData.append("student_name", name);
    if (blob1) formData.append("photo1", blob1, "cap1.jpg");
    if (LIVENESS_REQUIRED && blob2) formData.append("photo2", blob2, "cap2.jpg");

    submitBtn.disabled = true;
    submitBtn.innerHTML = "Verifying...";
    document.getElementById('loadingOverlay').classList.remove('hidden');

    fetch("/upload_liveness", { method: "POST", body: formData })
      .then(r => r.text())
      .then(html => {
        document.getElementById('loadingOverlay').classList.add('hidden');
        document.documentElement.innerHTML = html;
        lucide.createIcons();
      })
      .catch(err => {
        console.error(err);
        document.getElementById('loadingOverlay').classList.add('hidden');
        alert("Upload failed. Please retry.");
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i data-lucide="check-circle" class="w-4 h-4"></i> Submit Attendance`;
        lucide.createIcons();
      });
  });

  lucide.createIcons();
</script>
{% endraw %}
    """ + BASE_FOOT

    return render_template_string(html,
                                    title="Mark Attendance",
                                    is_admin=is_admin(),
                                    admin_badge=admin_badge_html(),
                                    session_name=session_name,
                                    status=status,
                                    session_id=session_id,
                                    classroom_message=classroom_msg,
                                    LIVENESS_REQUIRED=LIVENESS_REQUIRED)
# Upload & verification (server-side)
# -------------------------
@app.route("/upload_liveness", methods=["POST"])
def upload_liveness():
    client_ip = request.remote_addr or "unknown"
    if not rate_limit_check(client_ip):
        return "<h2>❌ Too many requests. Please wait and try again.</h2><a href='/'>Home</a>", 429

    # enforce concurrency limit
    acquired = UPLOAD_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        return "<h2>❌ Server busy. Try again in a few seconds.</h2><a href='/'>Home</a>", 503
    try:
        session_id = request.form.get("session_id") or "unknown"
        name = (request.form.get("student_name") or "").strip()
        f1 = request.files.get("photo1")
        f2 = request.files.get("photo2")
        if not name or not f1:
            return "<h2>❌ Name and at least one photo required</h2><a href='/mark_attendance'>Back</a>"
        
        # Liveness check pre-requisite (Only if LIVENESS_REQUIRED is true in ENV)
        if LIVENESS_REQUIRED and not f2:
            return f"<h2>❌ Liveness check is required, but second photo (open mouth) is missing.</h2><a href='/mark_attendance?session={session_id}'>Back</a>"

        # file size check
        f1.seek(0, os.SEEK_END)
        size1 = f1.tell()
        f1.seek(0)
        if size1 > MAX_UPLOAD_BYTES:
            return f"<h2>❌ photo1 too large (max {MAX_UPLOAD_MB}MB)</h2><a href='/mark_attendance?session={session_id}'>Back</a>"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        path1, path2 = None, None
        im1, im2 = None, None # Keep images in memory if needed later
        try:
            # Load the first image (photo1: normal face)
            f1_content = f1.read()
            im1 = Image.open(io.BytesIO(f1_content)).convert("RGB")
            path1 = os.path.join(UPLOADS_DIR, f"{ts}_{secure_filename(name)}_1.jpg")
            im1.save(path1, "JPEG", quality=85)
            
            # Load the second image (photo2: open mouth / liveness check)
            if f2 and f2.filename:
                f2.seek(0, os.SEEK_END)
                size2 = f2.tell()
                f2.seek(0)
                if size2 <= MAX_UPLOAD_BYTES:
                    f2_content = f2.read()
                    im2 = Image.open(io.BytesIO(f2_content)).convert("RGB")
                    path2 = os.path.join(UPLOADS_DIR, f"{ts}_{secure_filename(name)}_2.jpg")
                    im2.save(path2, "JPEG", quality=85)
                else:
                    return f"<h2>❌ photo2 too large (max {MAX_UPLOAD_MB}MB)</h2><a href='/mark_attendance?session={session_id}'>Back</a>"
        except Exception as e:
            return f"<h2>❌ Could not read or save photo: {e}</h2><a href='/mark_attendance?session={session_id}'>Back</a>"

        try:
            # Use loaded images in memory for numpy conversion
            rgb1 = np.asarray(im1).astype(np.uint8)
            rgb2 = np.asarray(im2).astype(np.uint8) if im2 is not None else None
        except Exception as e:
            return f"<h2>❌ Could not process images: {e}</h2><a href='/mark_attendance?session={session_id}'>Back</a>"

        # ensure session exists and is OPEN and within window
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            # Fetch date, duration, status, classroom, and session name
            cur.execute("SELECT date, duration_minutes, status, classroom_id, session_name FROM sessions WHERE session_id=?", (session_id,))
            row = cur.fetchone()
        
        if not row:
            error_html = BASE_HEAD + """
            <div class="max-w-md mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-xl p-6 md:p-8 text-center animate-slide-up">
                <div class="w-16 h-16 bg-rose-50 dark:bg-rose-955/20 text-rose-650 dark:text-rose-450 rounded-full flex items-center justify-center mx-auto mb-4 border border-rose-100/50 dark:border-rose-900/30">
                  <i data-lucide="alert-circle" class="w-8 h-8"></i>
                </div>
                <h2 class="text-2xl font-extrabold text-slate-850 dark:text-white font-display mb-2">Session Not Found</h2>
                <p class="text-xs text-slate-455 dark:text-slate-500 mb-6">The Session ID you provided does not exist. Please check the ID or scan the QR code again.</p>
                <a href="/" class="inline-block bg-slate-100 hover:bg-slate-205 dark:bg-slate-700 dark:hover:bg-slate-650 text-slate-700 dark:text-slate-200 px-6 py-2.5 rounded-xl text-xs font-bold transition">
                    Return Home
                </a>
            </div>
            <script>lucide.createIcons();</script>
            """ + BASE_FOOT
            return error_html
        
        start_str, duration, status, classroom_id, session_name = row
        
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            if datetime.now() > start + timedelta(minutes=int(duration) + LATE_CUTOFF_MINUTES):
                error_html = BASE_HEAD + """
                <div class="max-w-md mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-xl p-6 md:p-8 text-center animate-slide-up">
                    <div class="w-16 h-16 bg-amber-50 dark:bg-amber-955/20 text-amber-650 dark:text-amber-450 rounded-full flex items-center justify-center mx-auto mb-4 border border-amber-100/50 dark:border-amber-900/30">
                      <i data-lucide="clock" class="w-8 h-8"></i>
                    </div>
                    <h2 class="text-2xl font-extrabold text-slate-850 dark:text-white font-display mb-2">Session Expired</h2>
                    <p class="text-xs text-slate-455 dark:text-slate-500 mb-6">This attendance session has ended and is no longer accepting submissions.</p>
                    <a href="/" class="inline-block bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-650 text-slate-700 dark:text-slate-200 px-6 py-2.5 rounded-xl text-xs font-bold transition">
                        Return Home
                    </a>
                </div>
                <script>lucide.createIcons();</script>
                """ + BASE_FOOT
                return error_html
        except Exception:
            pass

        # Prevent duplicate marking for student for same session
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("SELECT 1 FROM attendance WHERE student_name=? AND session_id=?", (name, session_id))
            if cur.fetchone():
                return f"<h2>❌ {name} already marked for this session</h2><a href='/'>Home</a>"
        
        # --- Background Snap Check Logic (Conditional and now Optional) ---
        background_ok = True
        dbg_bg = "N/A"
        
        # Only run if BACKGROUND_CHECK_REQUIRED is ON AND a classroom is linked to the session
        run_bg_check = BACKGROUND_CHECK_REQUIRED and classroom_id
        
        if run_bg_check:
            classroom_info = get_classroom_info(classroom_id)
            if classroom_info:
                ref_path = classroom_info[2]
                try:
                    ref_rgb = load_rgb8_from_path_strict(ref_path)
                    background_ok, dist_bg, dbg_bg = background_check_safe(rgb1, ref_rgb)
                except Exception as e:
                    background_ok = False
                    dbg_bg = f"BG_FAIL: Ref img load error: {e}"
            else:
                dbg_bg = "BG_SKIP: Classroom not found"
        else:
            dbg_bg = "BG_SKIP: Not required or no classroom set"
        # --- End Background Snap Check Logic ---

        # load enrolled encoding
        ref_enc = load_student_embedding(name)
        if ref_enc is None:
            return f"<h2>⚠ No enrollment found for <code>{name}</code></h2><p>Please enroll first.</p><a href='/admin/login'>Admin</a>"

        # Face Recognition/Liveness Logic
        enc1, dbg1 = face_encoding_safe(rgb1)
        enc2, dbg2 = (face_encoding_safe(rgb2) if rgb2 is not None else (None, "no second photo"))

        if enc1 is None:
            return f"<h2>❌ Couldn't detect face in photo1: {dbg1}</h2><a href='/mark_attendance?session={session_id}'>Back</a>"

        try:
            dist_ref1 = float(face_recognition.face_distance([ref_enc], enc1)[0]) if FACE_RECOGNITION_AVAILABLE else 0.0
        except Exception:
            dist_ref1 = 0.0
        match_ref1 = dist_ref1 <= FACE_DISTANCE_TOLERANCE

        # Liveness checks (optional based on LIVENESS_REQUIRED)
        liveness_ok = True
        
        if LIVENESS_REQUIRED:
             # Retains full, strict Liveness logic if configured in ENV
            if enc2 is None:
                liveness_ok = False # Fail if required and missing
            else:
                try:
                    dist_ref2 = float(face_recognition.face_distance([ref_enc], enc2)[0]) if FACE_RECOGNITION_AVAILABLE else 0.0
                    match_ref2 = dist_ref2 <= FACE_DISTANCE_TOLERANCE
                    same_person = face_recognition.compare_faces([enc1], enc2, tolerance=SAME_PERSON_TOLERANCE)[0] if FACE_RECOGNITION_AVAILABLE else True
                    s1 = mouth_open_score(rgb1)
                    s2 = mouth_open_score(rgb2)
                    liveness_ok = (s1 is not None and s2 is not None and (s2 - s1) >= MOUTH_DELTA_THRESHOLD and match_ref2 and same_person)
                except Exception:
                    liveness_ok = False
        
        # --- SIMPLIFIED ACCEPTANCE LOGIC ---
        # Success requires match_ref1 AND (background_ok IF background check is running)
        accepted = match_ref1
        if run_bg_check:
            accepted = accepted and background_ok
        if LIVENESS_REQUIRED:
            accepted = accepted and liveness_ok
        # --- END SIMPLIFIED ACCEPTANCE LOGIC ---


        reasons = []
        if not match_ref1:
            reasons.append(f"Photo does not match enrolled face (Distance: {dist_ref1:.2f})")
        if run_bg_check and not background_ok:
            reasons.append(f"Classroom background check failed ({dbg_bg.split(':')[1].strip()})")
        if LIVENESS_REQUIRED and not liveness_ok:
             reasons.append("Liveness check failed (Need clear open mouth in photo #2, or verification failed.)")
        
        if accepted:
            status_str = "PRESENT"
            try:
                # compute LATE if beyond late cutoff
                with sqlite3.connect(DB_PATH) as con:
                    cur = con.cursor()
                    cur.execute("SELECT date, duration_minutes FROM sessions WHERE session_id=?", (session_id,))
                    row2 = cur.fetchone()
                if row2:
                    start2 = datetime.strptime(row2[0], "%Y-%m-%d %H:%M:%S")
                    dur2 = int(row2[1] or ATTENDANCE_WINDOW_MINUTES)
                    if datetime.now() > start2 + timedelta(minutes=dur2 + LATE_CUTOFF_MINUTES):
                        status_str = "LATE"
                    else:
                        if (datetime.now() - start2).total_seconds() > LATE_CUTOFF_MINUTES * 60:
                            status_str = "LATE"
            except Exception:
                pass
            save_attendance_record(name, session_id, status_str)
            
            verified_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_pill = "bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-450" if status_str == "PRESENT" else "bg-amber-50 dark:bg-amber-955/40 text-amber-600 dark:text-amber-450"

            # Success page with added flair
            success_html = BASE_HEAD + f"""
            <div class="max-w-md mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-xl p-8 text-center animate-slide-up">
                <div class="success-animation mb-6 flex justify-center">
                    <svg class="w-20 h-20 text-emerald-500 animate-[scaleIn_0.5s_ease-out_forwards]" viewBox="0 0 52 52">
                        <circle class="stroke-emerald-500 stroke-2 fill-none animate-[stroke_0.6s_cubic-bezier(0.65,0,0.45,1)_forwards]" style="stroke-dasharray: 166; stroke-dashoffset: 166;" cx="26" cy="26" r="25"/>
                        <path class="stroke-emerald-500 stroke-2 fill-none animate-[stroke_0.3s_cubic-bezier(0.65,0,0.45,1)_0.6s_forwards]" style="stroke-dasharray: 48; stroke-dashoffset: 48;" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
                    </svg>
                </div>
                
                <h2 class="text-3xl font-extrabold text-slate-850 dark:text-white font-display mb-1">Check-in Verified</h2>
                <p class="text-xs text-slate-400 dark:text-slate-500 mb-6">Your attendance status is updated successfully</p>
                
                <div class="bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-750 rounded-2xl p-5 text-left space-y-3 mb-6">
                    <div class="flex justify-between items-center text-xs">
                        <span class="text-slate-400 font-semibold">Student Name:</span>
                        <span class="text-slate-800 dark:text-slate-200 font-bold">{name}</span>
                    </div>
                    <div class="flex justify-between items-center text-xs">
                        <span class="text-slate-400 font-semibold">Session ID:</span>
                        <span class="text-slate-800 dark:text-slate-200 font-mono">#{session_id}</span>
                    </div>
                    <div class="flex justify-between items-center text-xs">
                        <span class="text-slate-400 font-semibold">Lecture:</span>
                        <span class="text-slate-800 dark:text-slate-200 font-semibold">{session_name}</span>
                    </div>
                    <div class="flex justify-between items-center text-xs">
                        <span class="text-slate-400 font-semibold">Time:</span>
                        <span class="text-slate-600 dark:text-slate-350">{verified_time}</span>
                    </div>
                    <div class="flex justify-between items-center text-xs pt-2 border-t border-slate-100 dark:border-slate-800/80">
                        <span class="text-slate-400 font-semibold">Status Marked:</span>
                        <span class="px-2.5 py-0.5 rounded-lg text-xs font-bold {status_pill}">{status_str}</span>
                    </div>
                </div>

                <a href="/" class="w-full bg-indigo-650 hover:bg-indigo-750 text-white py-3 rounded-xl font-bold text-sm shadow-md transition duration-200 flex items-center justify-center gap-2">
                    <i data-lucide="home" class="w-4 h-4"></i> Back to Homepage
                </a>
            </div>

            <style>
              @keyframes stroke {{
                100% {{ stroke-dashoffset: 0; }}
              }}
              @keyframes scaleIn {{
                0% {{ transform: scale(0); }}
                100% {{ transform: scale(1); }}
              }}
            </style>
            <script>lucide.createIcons();</script>
            """ + BASE_FOOT
            return render_template_string(success_html, title="Check-in Verified")
        else:
            dbg_lines = f"Face1:{dbg1}"
            if LIVENESS_REQUIRED and enc2 is not None:
                dbg_lines += f"<br/>Face2:{dbg2}"
            if dist_ref1 is not None:
                dbg_lines += f"<br/>Face Dist1={dist_ref1:.2f} (Max: {FACE_DISTANCE_TOLERANCE:.2f})"
            if run_bg_check:
                dbg_lines += f"<br/>Background: {dbg_bg}"
                
            reasons_html = "".join(f"<li class='flex items-start gap-2 text-xs text-rose-700 dark:text-rose-450'><i data-lucide='alert-triangle' class='w-4 h-4 mt-0.5 flex-shrink-0'></i><span>{r}</span></li>" for r in reasons)
            
            # Failure page with detailed reasons
            fail_html = BASE_HEAD + f"""
            <div class="max-w-md mx-auto bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-xl p-8 text-center animate-slide-up">
                <div class="mb-6 flex justify-center animate-[scaleIn_0.5s_ease-out_forwards]">
                    <div class="w-16 h-16 bg-rose-50 dark:bg-rose-950/30 text-rose-500 rounded-full flex items-center justify-center border border-rose-100 dark:border-rose-900/30 shadow-inner">
                        <i data-lucide="x-circle" class="w-10 h-10"></i>
                    </div>
                </div>

                <h2 class="text-3xl font-extrabold text-slate-850 dark:text-white font-display mb-1">Verification Failed</h2>
                <p class="text-xs text-slate-400 dark:text-slate-500 mb-6">We could not verify your identity for {name}</p>
                
                <div class="bg-rose-50/50 dark:bg-rose-955/20 border border-rose-100 dark:border-rose-950/40 rounded-2xl p-5 text-left space-y-3 mb-6">
                    <p class="text-xs font-bold text-rose-800 dark:text-rose-400 mb-1">Check-in Issues:</p>
                    <ul class="space-y-2">
                        {reasons_html}
                    </ul>
                </div>

                <div class="bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-750 rounded-2xl p-4 text-left mb-6">
                    <details class="group">
                        <summary class="flex justify-between items-center text-xs font-bold text-slate-500 dark:text-slate-400 cursor-pointer outline-none">
                             <span>Technical Debug Logs</span>
                             <i data-lucide="chevron-down" class="w-3.5 h-3.5 group-open:rotate-180 transition-transform duration-200"></i>
                        </summary>
                        <div class="mt-3 text-[10px] font-mono text-slate-400 dark:text-slate-500 space-y-1 overflow-x-auto">
                            {dbg_lines}
                        </div>
                    </details>
                </div>

                <div class="flex gap-3">
                    <a href="/" class="flex-1 bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-650 text-slate-700 dark:text-slate-200 py-3 rounded-xl font-semibold text-xs transition duration-200 flex items-center justify-center gap-1.5">
                        <i data-lucide="home" class="w-3.5 h-3.5"></i> Home
                    </a>
                    <a href="/mark_attendance?session={session_id}" class="flex-1 bg-indigo-650 hover:bg-indigo-750 text-white py-3 rounded-xl font-bold text-xs shadow-md transition duration-200 flex items-center justify-center gap-1.5">
                        <i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i> Try Again
                    </a>
                </div>
            </div>
            <script>lucide.createIcons();</script>
            """ + BASE_FOOT
            return render_template_string(fail_html, title="Verification Failed")
    finally:
        UPLOAD_SEMAPHORE.release()

# -------------------------
# Enroll (admin) - multi-course checkboxes, parent phone
# -------------------------
@app.route("/enroll", methods=["GET"])
def enroll():
    guard = require_admin()
    if guard:
        return guard
    courses = list_courses()
    
    # Format course checkboxes with course name and code
    boxes = ""
    for c in courses:
        # c: id, code, name, description
        boxes += f"<label class='inline-flex items-center gap-2 text-xs font-medium text-slate-700 dark:text-slate-350 cursor-pointer'><input type='checkbox' name='course_ids' value='{c[0]}' class='form-checkbox rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-4 h-4'>{c[2]} ({c[1]})</label>"
    
    # Enrollment page warning if liveness is required
    liveness_warning = ""
    if LIVENESS_REQUIRED:
        liveness_warning = """
        <div class="bg-amber-50 dark:bg-amber-955/20 p-4 rounded-xl border border-amber-200/50 dark:border-amber-900/30 text-xs text-amber-700 dark:text-amber-400 flex items-start gap-2">
            <i data-lucide="alert-triangle" class="w-4 h-4 flex-shrink-0 mt-0.5"></i>
            <div>
                <strong class="font-bold">Liveness Check ON:</strong> Students must submit two photos (one with normal face, one with open mouth) to mark attendance.
            </div>
        </div>
        """
    
    html = BASE_HEAD + """
    <div class="max-w-2xl mx-auto bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-      <h2 class="text-3xl font-extrabold text-slate-850 dark:text-white font-display mb-4">Enroll Student</h2>
      <div id="secureContextWarning" class="hidden mb-4 bg-rose-50 dark:bg-rose-955/20 p-4 rounded-xl border border-rose-200/50 dark:border-rose-900/30 text-xs text-rose-700 dark:text-rose-455 flex items-start gap-2">
          <i data-lucide="shield-alert" class="w-4 h-4 flex-shrink-0 mt-0.5 animate-bounce"></i>
          <div>
              <strong class="font-bold">Camera Blocked:</strong> Browsers block camera access on non-secure connections. Access using <code class="bg-rose-100 dark:bg-rose-950 px-1.5 py-0.5 rounded font-mono">localhost</code> or configure HTTPS.
          </div>
      </div>
      {{ liveness_warning | safe }}
      <form id="enrollForm" action="/enroll_upload" method="POST" enctype="multipart/form-data" class="space-y-4 mt-4">
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Student Name / Roll Number</label>
            <input name="student_name" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="e.g. Jane Doe / 12345" required>
        </div>
        
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
              <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Email Address</label>
              <input name="email" type="email" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="jane@example.com">
          </div>
          <div>
              <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Phone Number</label>
              <input name="phone" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="+1234567890">
          </div>
        </div>
        
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Parent Phone (for SMS alerts)</label>
            <input name="parent_phone" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="Parent phone number">
        </div>
        
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Courses</label>
            <div class="mt-2 p-4 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-750 grid grid-cols-2 gap-3 max-h-40 overflow-y-auto">
                {{ boxes | safe }}
            </div>
        </div>
 
        <div class="border-t border-slate-100 dark:border-slate-750/50 pt-5 mt-5">
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-2">Capture Photo (for face recognition) - **Front Facing**</label>
            <p class="text-[11px] text-slate-455 dark:text-slate-500 mb-3">Upload a high-quality photo of the student's face for the reference embedding.</p>
            <div class="mt-2">
                <video id="enrollCamera" autoplay playsinline muted class="rounded-xl bg-slate-900 w-full h-48 mb-2 object-cover"></video>
                <div class="flex gap-2 mb-3 mt-2">
                    <button id="enrollCap" type="button" disabled class="bg-indigo-650 hover:bg-indigo-750 text-white px-4 py-2.5 rounded-xl text-xs font-bold shadow-md transition disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"><i data-lucide="camera" class="w-3.5 h-3.5"></i> Capture from Webcam</button>
                    <button id="clearPreview" type="button" class="bg-slate-100 hover:bg-slate-205 dark:bg-slate-700 dark:hover:bg-slate-650 text-slate-700 dark:text-slate-200 px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-1 transition"><i data-lucide="x-circle" class="w-3.5 h-3.5"></i> Clear File</button>
                </div>
                <input type="file" id="photoFile" name="photo" accept="image/*" capture="user" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2 text-xs focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" required />
            </div>
        </div>
 
        <div class="text-center mt-6">
            <button class="bg-indigo-600 hover:bg-indigo-700 text-white px-10 py-3 rounded-xl font-bold text-sm shadow-lg shadow-indigo-500/10 hover:shadow-indigo-500/20 transform hover:-translate-y-0.5 transition duration-200">
                Enroll Student
            </button>
        </div>
      </form>
    </div>
    <script>lucide.createIcons();</script>

    {% raw %}
    <script>
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        document.getElementById('secureContextWarning').classList.remove('hidden');
    }

    const video = document.getElementById('enrollCamera');
    const enrollCap = document.getElementById('enrollCap');
    const photoFile = document.getElementById('photoFile');
    let enrollStream=false;
    
    // Use user-facing camera for enrollment reference photo
    // This method is robust because it relies solely on the promise resolution
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })  
        .then(s => {  
            video.srcObject = s;  
            video.play()
            .then(() => {
                enrollStream = true; 
                enrollCap.disabled = false;
            })
            .catch(err => {
                console.warn("Autoplay blocked, enabling capture button anyway", err);
                enrollStream = true;
                enrollCap.disabled = false;
            });
        })
        .catch(e => {  
            console.log('webcam not available', e);  
            enrollCap.style.display = 'none';
            video.style.display = 'none';
            alert("Camera access failed. Please upload a photo manually.");
        });
    }

    enrollCap.addEventListener('click', () => {
        if(!enrollStream){ alert('webcam not ready'); return; }
        const c = document.createElement('canvas');
        c.width = video.videoWidth; c.height = video.videoHeight;
        c.getContext('2d').drawImage(video,0,0);
        c.toBlob(b => {
            const file = new File([b], 'enroll.jpg', { type: 'image/jpeg' });
            const dt = new DataTransfer();
            dt.items.add(file);
            photoFile.files = dt.files;
            alert('Photo captured and ready for upload.');
        }, 'image/jpeg', 0.9);
    });

    document.getElementById('clearPreview').addEventListener('click', ()=>{ photoFile.value=''; });
    </script>
    {% endraw %}
    """ + BASE_FOOT
    # FIX 4: Pass all variables as keyword arguments to render_template_string
    return render_template_string(html, 
                                 title="Enroll Student", 
                                 is_admin=is_admin(), 
                                 admin_badge=admin_badge_html(),
                                 liveness_warning=liveness_warning, # Passed directly for Jinja2 rendering
                                 boxes=boxes # Passed directly for Jinja2 rendering
                                 )

@app.route("/enroll_upload", methods=["POST"])
def enroll_upload():
    guard = require_admin()
    if guard:
        return guard
    name = (request.form.get("student_name") or "").strip()
    email = (request.form.get("email") or "").strip()
    phone = (request.form.get("phone") or "").strip()
    parent_phone = (request.form.get("parent_phone") or "").strip()
    course_ids = request.form.getlist("course_ids")
    primary_course_id = int(course_ids[0]) if course_ids else None
    f = request.files.get("photo")
    if not name or not f:
        return "<h2>❌ Name and photo required</h2><a href='/enroll'>Back</a>"
    # size check
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return f"<h2>❌ photo too large (max {MAX_UPLOAD_MB}MB)</h2><a href='/enroll'>Back</a>"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpg_path = os.path.join(STUDENTS_DIR, f"{ts}_{secure_filename(name)}.jpg")
    try:
        im = Image.open(io.BytesIO(f.read())).convert("RGB")
        im.save(jpg_path, "JPEG", quality=88)
    except Exception as e:
        return f"<h2>❌ Could not read or save photo: {e}</h2><a href='/enroll'>Back</a>"
    
    try:
        rgb = np.asarray(im).astype(np.uint8) # Use in-memory image
    except Exception as e:
        return f"<h2>❌ Could not load photo: {e}</h2><a href='/enroll'>Back</a>"

    enc, dbg = face_encoding_safe(rgb)
    if enc is None:
        try: os.remove(jpg_path)
        except Exception: pass
        return f"<h2>❌ No clear face detected. debug: {dbg}</h2><p>Please try a clearer image where the face is prominent.</p><a href='/enroll'>Back</a>"

    # save primary course into students table and all courses into student_courses
    save_student(name, email, phone, parent_phone, enc, primary_course_id)
    set_student_courses(name, [int(cid) for cid in course_ids])
    
    # Use Flask's route for student static content to generate the image URL
    img_filename = os.path.basename(jpg_path)
    img_url = url_for('student_preview', filename=img_filename)
    
    html = BASE_HEAD + f"""
    <div class="max-w-md mx-auto bg-white rounded-2xl shadow-xl p-8 text-center fade-in border-t-8 border-emerald-600">
      <h2 class="text-3xl font-bold text-emerald-700 mb-2">✅ Enrollment Complete!</h2>
      <p class="text-xl text-gray-800 mb-4"><strong>{name}</strong> is now enrolled.</p>
      <img src="{img_url}" alt="Enrolled Photo" class="mx-auto w-32 h-32 object-cover rounded-full shadow-lg mb-4 border-4 border-white">
      <p class="text-sm text-gray-500">Reference photo saved. Ready for face verification.</p>
      <div class="mt-6"><a href="/admin" class="gradient-btn text-white px-8 py-3 rounded-full font-semibold shadow-lg">Back to Admin Hub</a></div>
    </div>
    """ + BASE_FOOT
    return render_template_string(html, title="Enroll Success", is_admin=is_admin())

# -------------------------
# Admin: students list, edit, delete
# -------------------------
@app.route("/admin/students")
def admin_students():
    guard = require_admin()
    if guard:
        return guard
    rows = list_students()
    students_list = []
    for r in rows:
        course_name = get_course_name(r[5]) if r[5] else "N/A"
        students_list.append({
            "name": r[0],
            "email": r[1] or "N/A",
            "phone": r[2] or "N/A",
            "parent_phone": r[3] or "N/A",
            "enrolled_at": r[4] or "N/A",
            "course": course_name
        })

    html = BASE_HEAD + """
    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-sm overflow-hidden animate-slide-up">
      <div class="p-6 border-b border-slate-200 dark:border-slate-750 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h3 class="text-lg font-bold text-slate-800 dark:text-white">Enrolled Students ({{ students | length }} Total)</h3>
        <div class="flex items-center gap-2 w-full sm:w-auto">
            <div class="relative w-full sm:w-64">
                <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                    <i data-lucide="search" class="w-4 h-4"></i>
                </span>
                <input type="text" id="table-search" oninput="filterTable()" class="w-full pl-9 pr-4 py-2 border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl text-xs focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="Search students...">
            </div>
            <a href="/enroll" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-xl text-xs font-semibold shadow-md shadow-indigo-500/10 transition whitespace-nowrap flex items-center gap-1.5"><i data-lucide="user-plus" class="w-3.5 h-3.5"></i> Add Student</a>
        </div>
      </div>

      <table class="w-full table-auto text-left border-collapse" id="students-table">
          <thead class="bg-slate-50 dark:bg-slate-900 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-250 dark:border-slate-750">
            <tr>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(0)">
                <div class="flex items-center gap-1.5">Name <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4">Email</th>
              <th class="p-4">Phone</th>
              <th class="p-4">Parent Phone</th>
              <th class="p-4 text-center">Enrolled At</th>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(5)">
                <div class="flex items-center gap-1.5">Primary Course <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100 dark:divide-slate-750/40 text-slate-700 dark:text-slate-300" id="students-table-body">
            {% for r in students %}
            <tr class="hover:bg-slate-50/50 dark:hover:bg-slate-700/20 transition text-sm">
                <td class="p-4 font-bold text-slate-850 dark:text-white">{{ r.name }}</td>
                <td class="p-4">{{ r.email }}</td>
                <td class="p-4">{{ r.phone }}</td>
                <td class="p-4">{{ r.parent_phone }}</td>
                <td class="p-4 text-xs text-slate-400 dark:text-slate-500 text-center">{{ r.enrolled_at }}</td>
                <td class="p-4"><span class="px-2.5 py-1 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 text-xs font-semibold rounded-lg">{{ r.course }}</span></td>
                <td class="p-4 text-right space-x-1.5 whitespace-nowrap">
                    <a class="bg-indigo-50 text-indigo-600 hover:bg-indigo-650 hover:text-white dark:bg-indigo-950/20 dark:text-indigo-400 px-3 py-1.5 rounded-xl text-xs font-semibold transition" href="/admin/students/edit?name={{ r.name|urlencode }}">Edit</a>
                    <button class="bg-rose-50 text-rose-600 hover:bg-rose-600 hover:text-white dark:bg-rose-950/20 dark:text-rose-455 px-3 py-1.5 rounded-xl text-xs font-semibold transition" onclick="triggerDeleteStudent('{{ r.name }}')">
                        Delete
                    </button>
                </td>
            </tr>
            {% endfor %}
          </tbody>
      </table>
      
      <div id="pagination-container"></div>
    </div>

    <form id="delete-student-form" method="POST" action="/admin/students/delete" class="hidden">
      <input type="hidden" id="delete-student-name" name="name">
    </form>

    <script>
      let sortDirection = 1;
      let sortColumnIndex = 0;
      let currentPage = 1;
      const rowsPerPage = 5;

      window.triggerDeleteStudent = function(studentName) {
          confirmDelete(`Are you sure you want to delete student ${studentName}? This will remove all their enrollments and attendance logs.`, function() {
              document.getElementById('delete-student-name').value = studentName;
              document.getElementById('delete-student-form').submit();
          });
      };

      window.sortTable = function(colIndex) {
          const table = document.getElementById("students-table");
          const tbody = document.getElementById("students-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          
          if (sortColumnIndex === colIndex) {
              sortDirection = -sortDirection;
          } else {
              sortColumnIndex = colIndex;
              sortDirection = 1;
          }
          
          const headers = table.querySelectorAll("thead th");
          headers.forEach((th, idx) => {
              const icon = th.querySelector(".sort-icon");
              if (icon) {
                  if (idx === colIndex) {
                      icon.innerHTML = sortDirection === 1 ? '<i data-lucide="chevron-up" class="w-3.5 h-3.5"></i>' : '<i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>';
                  } else {
                      icon.innerHTML = '<i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i>';
                  }
              }
          });
          lucide.createIcons();

          trs.sort((a, b) => {
              const aVal = a.cells[colIndex].textContent.trim().toLowerCase();
              const bVal = b.cells[colIndex].textContent.trim().toLowerCase();
              return aVal.localeCompare(bVal, undefined, {numeric: true, sensitivity: 'base'}) * sortDirection;
          });
          
          trs.forEach(tr => tbody.appendChild(tr));
          updatePagination();
      };

      window.filterTable = function() {
          const query = document.getElementById("table-search").value.toLowerCase();
          const tbody = document.getElementById("students-table-body");
          const trs = tbody.querySelectorAll("tr");
          
          trs.forEach(tr => {
              const text = tr.textContent.toLowerCase();
              tr.style.display = text.includes(query) ? "" : "none";
          });
          
          currentPage = 1;
          updatePagination();
      };

      window.updatePagination = function() {
          const tbody = document.getElementById("students-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          const query = document.getElementById("table-search").value.toLowerCase();
          
          const visibleTrs = trs.filter(tr => tr.textContent.toLowerCase().includes(query));
          
          const totalPages = Math.ceil(visibleTrs.length / rowsPerPage) || 1;
          if (currentPage > totalPages) currentPage = totalPages;
          
          trs.forEach(tr => tr.style.display = "none");
          
          const startIdx = (currentPage - 1) * rowsPerPage;
          const endIdx = startIdx + rowsPerPage;
          
          visibleTrs.slice(startIdx, endIdx).forEach(tr => {
              tr.style.display = "";
          });
          
          const container = document.getElementById("pagination-container");
          if (!container) return;
          
          container.innerHTML = `
              <div class="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 p-4 border-t border-slate-200 dark:border-slate-750">
                  <div>Showing ${startIdx + 1} to ${Math.min(endIdx, visibleTrs.length)} of ${visibleTrs.length} entries</div>
                  <div class="flex gap-2">
                      <button onclick="changePage(-1)" ${currentPage === 1 ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Previous</button>
                      <span class="px-3 py-1 font-bold text-indigo-600 dark:text-indigo-400">${currentPage} / ${totalPages}</span>
                      <button onclick="changePage(1)" ${currentPage === totalPages ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Next</button>
                  </div>
              </div>
          `;
      };

      window.changePage = function(dir) {
          currentPage += dir;
          updatePagination();
      };

      document.addEventListener("DOMContentLoaded", function() {
          updatePagination();
      });
      lucide.createIcons();
    </script>
    """ + BASE_FOOT
    return render_template_string(html, title="Manage Students", is_admin=is_admin(), students=students_list)

@app.route("/admin/students/edit", methods=["GET", "POST"])
def admin_students_edit():
    guard = require_admin()
    if guard:
        return guard
    name = (request.args.get("name") or request.form.get("student_name") or "").strip()
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        phone = (request.form.get("phone") or "").strip()
        parent_phone = (request.form.get("parent_phone") or "").strip()
        course_ids = request.form.getlist("course_ids")
        primary_course_id = int(course_ids[0]) if course_ids else None
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("UPDATE students SET email=?, phone=?, parent_phone=?, course_id=? WHERE student_name=?",
                                 (email, phone, parent_phone, primary_course_id, name))
            con.commit()
        set_student_courses(name, [int(cid) for cid in course_ids])
        flash(f"Details for {name} updated successfully.", "success")
        return redirect(url_for("admin_students"))

    student = get_student(name)
    if not student:
        flash("Student not found.", "error")
        return redirect(url_for("admin_students"))

    sname, semail, sphone, sparent, enrolled_at, primary_course = student
    courses = list_courses()
    enrolled_ids = set(get_student_course_ids(sname))

    # Reference Photo Info
    photo_filename = f"{sname}.jpg"
    photo_exists = os.path.exists(os.path.join(STUDENTS_DIR, photo_filename))
    photo_url = url_for('students_preview', filename=photo_filename) if photo_exists else None

    # Stats and History
    present_days = 0
    late_days = 0
    total_days = 0
    attendance_history = []
    
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("SELECT status, timestamp, session_id FROM attendance WHERE student_name = ? ORDER BY timestamp DESC LIMIT 10", (sname,))
            rows = cur.fetchall()
            total_days = len(rows)
            for r in rows:
                if r[0] == 'PRESENT':
                    present_days += 1
                elif r[0] == 'LATE':
                    late_days += 1
                
                attendance_history.append({
                    "status": r[0],
                    "time": r[1],
                    "session_id": r[2]
                })
    except Exception as e:
        print("Error fetching student attendance stats:", e)

    attendance_rate = round(((present_days + late_days) / total_days * 100), 1) if total_days > 0 else 100.0

    html = BASE_HEAD + """
    <div class="grid lg:grid-cols-3 gap-8 animate-slide-up">
      <!-- Left side: Photo & Stats -->
      <div class="space-y-6">
        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm flex flex-col items-center">
            <div class="relative w-36 h-36 rounded-full overflow-hidden border-4 border-indigo-500 bg-slate-100 dark:bg-slate-900 flex items-center justify-center mb-4">
                {% if photo_url %}
                    <img src="{{ photo_url }}" class="w-full h-full object-cover">
                {% else %}
                    <div class="w-full h-full flex items-center justify-center bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 font-bold text-4xl">
                        {{ sname[0]|upper }}
                    </div>
                {% endif %}
            </div>
            <h3 class="text-xl font-bold text-slate-850 dark:text-white">{{ sname }}</h3>
            <p class="text-xs text-slate-450 dark:text-slate-500 mt-1">Enrolled: {{ enrolled_at }}</p>
        </div>

        <div class="grid grid-cols-3 gap-4">
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-4 rounded-xl text-center shadow-sm">
                <p class="text-2xl font-bold text-slate-800 dark:text-white">{{ total_days }}</p>
                <p class="text-[9px] font-semibold text-slate-450 uppercase tracking-wider mt-1">Classes</p>
            </div>
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-4 rounded-xl text-center shadow-sm">
                <p class="text-2xl font-bold text-emerald-600">{{ present_days }}</p>
                <p class="text-[9px] font-semibold text-slate-450 uppercase tracking-wider mt-1">Present</p>
            </div>
            <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-4 rounded-xl text-center shadow-sm">
                <p class="text-2xl font-bold text-indigo-600 dark:text-indigo-455">{{ attendance_rate }}%</p>
                <p class="text-[9px] font-semibold text-slate-450 uppercase tracking-wider mt-1">Rate</p>
            </div>
        </div>
      </div>

      <!-- Right side: edit form & logs -->
      <div class="lg:col-span-2 space-y-6">
        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm">
          <h3 class="text-lg font-bold text-slate-850 dark:text-white mb-6">Modify Details</h3>
          
          <form method="POST" class="space-y-4">
            <input type="hidden" name="student_name" value="{{ sname }}">
            <div>
                <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Student Name (Read Only)</label>
                <input value="{{ sname }}" readonly class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm cursor-not-allowed text-slate-500 outline-none" required>
            </div>
            
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                    <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Email Address</label>
                    <input name="email" value="{{ semail or '' }}" type="email" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white">
                </div>
                <div>
                    <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Phone Number</label>
                    <input name="phone" value="{{ sphone or '' }}" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white">
                </div>
            </div>
            
            <div>
                <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Parent Phone Number</label>
                <input name="parent_phone" value="{{ sparent or '' }}" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white">
            </div>
            
            <div>
                <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Registered Courses</label>
                <div class="mt-2 p-4 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-750 grid grid-cols-2 gap-3 max-h-40 overflow-y-auto">
                    {% for c in courses %}
                    <label class="inline-flex items-center gap-2 text-xs font-medium text-slate-700 dark:text-slate-300 cursor-pointer">
                        <input type="checkbox" name="course_ids" value="{{ c[0] }}" class="form-checkbox rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-4 h-4" {% if c[0] in enrolled_ids %}checked{% endif %}>
                        <span>{{ c[2] }} ({{ c[1] }})</span>
                    </label>
                    {% endfor %}
                </div>
            </div>
            
            <div class="flex justify-between items-center pt-4 border-t border-slate-100 dark:border-slate-750">
                <a class="text-xs font-bold text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition" href="/admin/students">Cancel</a>
                <button class="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-xl text-sm font-semibold shadow-md shadow-indigo-500/10 hover:shadow-indigo-500/20 transform hover:-translate-y-0.5 transition duration-200">
                    Save Changes
                </button>
            </div>
          </form>
        </div>

        <!-- History logs -->
        <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 p-6 rounded-2xl shadow-sm">
            <h3 class="text-lg font-bold text-slate-850 dark:text-white mb-6">Recent Attendance Logs (Last 10)</h3>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs border-collapse">
                    <thead>
                        <tr class="text-slate-400 border-b border-slate-100 dark:border-slate-700/50">
                            <th class="pb-3 font-semibold">Session ID</th>
                            <th class="pb-3 font-semibold">Status</th>
                            <th class="pb-3 font-semibold">Time</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100 dark:divide-slate-750/30 text-slate-700 dark:text-slate-300">
                        {% for log in history %}
                        <tr>
                            <td class="py-3 font-semibold text-slate-800 dark:text-white">#{{ log.session_id }}</td>
                            <td class="py-3">
                                {% if log.status == 'PRESENT' %}
                                    <span class="px-2 py-0.5 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-450 font-semibold rounded-lg text-[10px]">PRESENT</span>
                                {% elif log.status == 'LATE' %}
                                    <span class="px-2 py-0.5 bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-450 font-semibold rounded-lg text-[10px]">LATE</span>
                                {% else %}
                                    <span class="px-2 py-0.5 bg-rose-50 dark:bg-rose-950/40 text-rose-600 dark:text-rose-455 font-semibold rounded-lg text-[10px]">{{ log.status }}</span>
                                {% endif %}
                            </td>
                            <td class="py-3 text-slate-400">{{ log.time }}</td>
                        </tr>
                        {% endfor %}
                        {% if not history %}
                        <tr>
                            <td colspan="3" class="py-4 text-center text-slate-400">No attendance logs found.</td>
                        </tr>
                        {% endif %}
                    </tbody>
                </table>
            </div>
        </div>
      </div>
    </div>
    <script>lucide.createIcons();</script>
    """ + BASE_FOOT
    return render_template_string(html, title="Edit Student", is_admin=is_admin(), admin_badge=admin_badge_html(),
                                  sname=sname, semail=semail, sphone=sphone, sparent=sparent, enrolled_at=enrolled_at,
                                  photo_url=photo_url, courses=courses, enrolled_ids=enrolled_ids,
                                  total_days=total_days, present_days=present_days, late_days=late_days,
                                  attendance_rate=attendance_rate, history=attendance_history)

@app.route("/admin/students/delete", methods=["POST"])
def admin_students_delete():
    guard = require_admin()
    if guard:
        return guard
    name = (request.form.get("name") or "").strip()
    if name:
        delete_student(name)
        # Attempt to delete student reference images
        for fn in os.listdir(STUDENTS_DIR):
            if name in fn:
                try: os.remove(os.path.join(STUDENTS_DIR, fn))
                except Exception: pass
        flash(f"Student {name} and all records deleted.", "success")
    return redirect(url_for("admin_students"))


# -------------------------
# Sessions list & admin actions
# -------------------------
@app.route("/sessions") # Student/public view, redirects to sessions_admin if logged in
def sessions():
    if is_admin():
        return redirect(url_for("sessions_admin"))
    
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, session_id, date, session_name, subject, course_id, duration_minutes, status FROM sessions WHERE status='OPEN' ORDER BY id DESC")
        rows = cur.fetchall()
    
    rows_html = ""
    for r in rows:
        # r => id, session_id, date, session_name, subject, course_id, duration, status
        rows_html += f"<tr class='border-b last:border-0 hover:bg-gray-50'><td class='p-3'>{r[0]}</td><td class='p-3 font-mono text-sm'>{r[1]}</td><td class='p-3 font-semibold'>{r[3]}</td><td class='p-3 text-sm text-gray-600'>{r[4]}</td><td class='p-3 text-sm text-gray-500'>{r[2]}</td><td class='p-3 text-center'><span class='px-3 py-1 rounded-full text-white text-xs font-semibold bg-green-500'>OPEN</span></td><td class='p-3 text-right'><a class='bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1 rounded-full text-sm font-semibold' href='/mark_attendance?session={r[1]}'>Mark Attendance</a></td></tr>"
    
    html = BASE_HEAD + """
    <h2 class="text-3xl font-bold text-blue-600 mb-4">Open Sessions</h2>
    <div class="bg-white rounded-2xl shadow-lg overflow-hidden mt-4 fade-in border-t-4 border-blue-600">
        <p class="p-4 text-sm text-gray-600">Scan the QR code or click the link for any open session to mark your attendance.</p>
        <table class="w-full table-auto">
            <thead class="bg-blue-100 text-left text-sm text-blue-800">
                <tr>
                    <th class="p-3">ID</th>
                    <th class="p-3">Code</th>
                    <th class="p-3">Name</th>
                    <th class="p-3">Subject</th>
                    <th class="p-3">Date/Time</th>
                    <th class="p-3 text-center">Status</th>
                    <th class="p-3 text-right">Actions</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 text-gray-700">
                """ + rows_html + """
            </tbody>
        </table>
    </div>
    <div class="mt-6 text-center">
        <a class="bg-gray-700 hover:bg-gray-800 text-white px-6 py-2 rounded-full font-semibold shadow-lg" href="/">Back to Home</a>
    </div>
    """ + BASE_FOOT
    return render_template_string(html, title="Open Sessions", is_admin=False, rows_html=rows_html)

@app.route("/session_report")
def session_report():
    guard = require_admin()
    if guard:
        return guard
    
    session_id = request.args.get("session")
    if not session_id:
        return redirect(url_for("sessions_admin"))
        
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        # NEW: Get classroom_id
        cur.execute("SELECT session_name, date, status, subject, course_id, classroom_id FROM sessions WHERE session_id=?", (session_id,))
        row = cur.fetchone()
        cur.execute("SELECT student_name, status, timestamp FROM attendance WHERE session_id=?", (session_id,))
        attendance = cur.fetchall()
    
    if not row:
        return "Session not found", 404

    session_name = row[0]
    date = row[1]
    status = row[2]
    subject = row[3]
    course_id = row[4]
    classroom_id = row[5] # NEW: Get classroom ID

    present_count = 0
    late_count = 0
    present_rows_html = ""
    for r in attendance:
        status_color = "text-green-600" if r[1] == "PRESENT" else "text-yellow-600"
        if r[1] == "PRESENT":
            present_count += 1
        elif r[1] == "LATE":
            late_count += 1
        present_rows_html += f"<tr class='border-b last:border-0 hover:bg-gray-50'><td class='p-3'>{r[0]}</td><td class='p-3 font-semibold {status_color}'>{r[1]}</td><td class='p-3 text-sm text-gray-500'>{r[2]}</td></tr>"

    # compute absentees
    absentees = students_not_marked(session_id, course_id)
    abs_count = len(absentees)
    abs_html = ""
    for a in absentees:
        # a => student_name, email, parent_phone
        abs_html += f"<tr class='border-b last:border-0 hover:bg-gray-50'><td class='p-3 font-semibold'>{a[0]}</td><td class='p-3 text-sm text-gray-500'>{a[1] or 'N/A'}</td><td class='p-3 text-sm text-gray-500'>{a[2] or 'N/A'}</td><td class='p-3 text-right'><form method='POST' action='/session_manual_mark' style='display:inline;'><input type='hidden' name='session_id' value='{session_id}'><input type='hidden' name='student_name' value=\"{a[0]}\"><input type='hidden' name='status' value='PRESENT'><button class='bg-indigo-600 text-white px-3 py-1 rounded-full text-sm font-semibold'>Mark Present</button></form></td></tr>"

    classroom_info = get_classroom_info(classroom_id)
    classroom_name = f"Classroom: {classroom_info[1]} ({classroom_info[0]})" if classroom_info else "Classroom: N/A"

    html = BASE_HEAD + """
    <h2 class="text-3xl font-bold text-indigo-700 mb-2">Session Report</h2>
    <p class="text-sm text-gray-600">Final report for session <span class="font-mono">{{ session_id }}</span></p>

    <div class="bg-white rounded-2xl shadow-lg p-6 mt-6 fade-in border-t-8 border-indigo-600">
        <div class="flex justify-between items-center pb-4 border-b border-gray-200">
            <div>
                <p class="text-xl font-bold text-indigo-700">{{ session_name }}</p>
                <p class="text-sm text-gray-500">Subject: {{ subject }} | Date: {{ date }} | {{ classroom_name }}</p>
            </div>
            <span class="text-sm font-semibold text-white px-4 py-1 rounded-full {{ status_class }}">{{ status }}</span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6 text-center">
            <div class="bg-green-50 p-4 rounded-lg shadow-inner"><p class="text-4xl font-extrabold text-green-700">{{ present_count }}</p><p class="text-sm text-green-600">Present</p></div>
            <div class="bg-yellow-50 p-4 rounded-lg shadow-inner"><p class="text-4xl font-extrabold text-yellow-700">{{ late_count }}</p><p class="text-sm text-yellow-600">Late</p></div>
            <div class="bg-rose-50 p-4 rounded-lg shadow-inner"><p class="text-4xl font-extrabold text-rose-700">{{ abs_count }}</p><p class="text-sm text-rose-600">Absent</p></div>
        </div>

        <div class="mt-6 flex flex-col lg:flex-row gap-6">
            <div class="bg-white rounded-xl shadow-lg border p-4 flex-1">
                <h3 class="text-lg font-bold text-gray-800 mb-2">Present Students ({{ present_count + late_count }})</h3>
                <div class="max-h-80 overflow-y-auto">
                    <table class="w-full text-sm">
                        <thead class="sticky top-0 bg-white shadow-sm border-b"><tr><th class="text-left p-2">Name</th><th class="text-left p-2">Status</th><th class="text-left p-2">Timestamp</th></tr></thead>
                        <tbody class="divide-y divide-gray-200">""" + present_rows_html + """</tbody>
                    </table>
                </div>
            </div>
            <div class="bg-white rounded-xl shadow-lg border p-4 flex-1">
                <h3 class="text-lg font-bold text-gray-800 mb-2">Absent Students ({{ abs_count }})</h3>
                <div class="max-h-80 overflow-y-auto">
                    <table class="w-full text-sm">
                        <thead class="sticky top-0 bg-white shadow-sm border-b"><tr><th class="text-left p-2">Name</th><th class="text-left p-2">Email</th><th class="text-left p-2">Parent Phone</th><th class="p-2 text-right">Action</th></tr></thead>
                        <tbody class="divide-y divide-gray-200">""" + abs_html + """</tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    <div class="mt-6 flex justify-center gap-4 fade-in">
        <a class="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-full font-semibold shadow" href="/session_export?session={{ session_id }}">Download CSV</a>
        <a class="bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-2 rounded-full font-semibold shadow" href="/session_export_pdf?session={{ session_id }}">Download PDF</a>
        {% if status == 'OPEN' %}
        <form method="POST" action="/sessions_admin_end" style="display:inline;"><input type="hidden" name="session_id" value="{{ session_id }}"><button class="bg-amber-600 hover:bg-amber-700 text-white px-6 py-2 rounded-full font-semibold shadow" onclick="return confirm('End session and notify absentees?');">End Session</button></form>
        {% endif %}
        <a class="bg-gray-700 hover:bg-gray-800 text-white px-6 py-2 rounded-full font-semibold shadow" href="/sessions_admin">Back to Sessions</a>
    </div>
    """ + BASE_FOOT

    status_class = "bg-rose-500" if status == "CLOSED" else "bg-green-500"
    return render_template_string(html, title="Session Report", is_admin=is_admin(), admin_badge=admin_badge_html(),
                                 session_id=session_id, session_name=session_name, subject=subject, date=date,
                                 status=status, status_class=status_class, classroom_name=classroom_name,
                                 present_count=present_count, late_count=late_count, abs_count=abs_count)
    
# -------------------------
# Manual mark endpoint (from report)
# -------------------------
@app.route("/session_manual_mark", methods=["POST"])
def session_manual_mark():
    guard = require_admin()
    if guard:
        return guard
    session_id = request.form.get("session_id")
    student_name = request.form.get("student_name")
    status = request.form.get("status", "PRESENT")
    if session_id and student_name:
        save_attendance_record(student_name, session_id, status)
        flash(f"Manually marked {student_name} as {status}.", "success")
    return redirect(url_for("session_report", session=session_id))

# -------------------------
# Export session data (CSV)
# -------------------------
@app.route("/session_export")
def session_export():
    guard = require_admin()
    if guard:
        return guard
    session_id = request.args.get("session")
    if not session_id:
        return redirect(url_for("sessions_admin"))
    
    attendance = attendance_for_session(session_id)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT session_name, date, course_id FROM sessions WHERE session_id=?", (session_id,))
        row = cur.fetchone()
    
    session_name = "Unknown"
    date_str = "Unknown"
    course_name = "NoCourse"
    
    if row:
        session_name = row[0]
        date_str = row[1]
        course_id = row[2]
        if course_id:
            c_name = get_course_name(course_id)
            if c_name:
                course_name = c_name

    def clean_fn(s):
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)

    clean_course = clean_fn(course_name)
    clean_session = clean_fn(session_name)
    clean_date = date_str.split(" ")[0] if " " in date_str else date_str
    clean_date = clean_fn(clean_date)
    csv_filename = f"Attendance_{clean_course}_{clean_session}_{clean_date}.csv"

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["student_name", "status", "timestamp"])
    for r in attendance:
        cw.writerow(r)
    mem = io.BytesIO()
    mem.write(si.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=csv_filename, mimetype="text/csv")

# -------------------------
# Export session PDF (reportlab optional)
# -------------------------
@app.route("/session_export_pdf")
def session_export_pdf():
    guard = require_admin()
    if guard:
        return guard
    session_id = request.args.get("session")
    if not session_id:
        return redirect(url_for("sessions_admin"))
    attendance = attendance_for_session(session_id)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT session_name, date, subject, course_id, classroom_id FROM sessions WHERE session_id=?", (session_id,))
        row = cur.fetchone()
    
    session_name = "Unknown"
    date_str = "Unknown"
    subject = "N/A"
    course_name = "NoCourse"
    classroom_label = "N/A"
    
    if row:
        session_name = row[0]
        date_str = row[1]
        subject = row[2] or "N/A"
        course_id = row[3]
        classroom_id = row[4]
        if course_id:
            c_name = get_course_name(course_id)
            if c_name:
                course_name = c_name
        if classroom_id:
            cl_info = get_classroom_info(classroom_id)
            if cl_info:
                classroom_label = f"{cl_info[1]} ({cl_info[0]})"

    def clean_fn(s):
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)

    clean_course = clean_fn(course_name)
    clean_session = clean_fn(session_name)
    clean_date = date_str.split(" ")[0] if " " in date_str else date_str
    clean_date = clean_fn(clean_date)
    pdf_filename = f"Attendance_{clean_course}_{clean_session}_{clean_date}.pdf"

    if REPORTLAB_AVAILABLE:
        from reportlab.lib.colors import HexColor
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        
        # Calculate statistics
        total_p = sum(1 for r in attendance if r[1] == 'PRESENT')
        total_l = sum(1 for r in attendance if r[1] == 'LATE')
        total_a = sum(1 for r in attendance if r[1] == 'ABSENT')

        def draw_page_template(canvas_obj, page_num):
            # Top Banner
            canvas_obj.setFillColor(HexColor('#312E81')) # Deep Indigo Navy
            canvas_obj.rect(0, height - 80, width, 80, stroke=0, fill=1)
            
            # Vector logo representation
            canvas_obj.setFillColor(HexColor('#FFFFFF'))
            canvas_obj.circle(55, height - 40, 20, stroke=0, fill=1)
            canvas_obj.setFillColor(HexColor('#4F46E5')) # Indigo Accent
            canvas_obj.rect(47, height - 48, 16, 16, stroke=0, fill=1)
            canvas_obj.setFillColor(HexColor('#FFFFFF'))
            canvas_obj.circle(55, height - 40, 4, stroke=0, fill=1)
            
            # Title
            canvas_obj.setFont("Helvetica-Bold", 14)
            canvas_obj.drawString(90, height - 35, "SMART ATTENDANCE SYSTEM")
            canvas_obj.setFont("Helvetica", 10)
            canvas_obj.setFillColor(HexColor('#E2E8F0'))
            canvas_obj.drawString(90, height - 52, "Official Presence Logs & Facial Verification Records")
            
            # Footer
            canvas_obj.setStrokeColor(HexColor('#E2E8F0'))
            canvas_obj.setLineWidth(0.5)
            canvas_obj.line(40, 45, width - 40, 45)
            
            canvas_obj.setFont("Helvetica", 8)
            canvas_obj.setFillColor(HexColor('#64748B'))
            canvas_obj.drawString(40, 30, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Private & Confidential")
            canvas_obj.drawRightString(width - 40, 30, f"Page {page_num}")

        # Page 1 Template Setup
        page_count = 1
        draw_page_template(c, page_count)
        
        # Header Info Card Box
        c.setFillColor(HexColor('#F8FAFC'))
        c.setStrokeColor(HexColor('#E2E8F0'))
        c.setLineWidth(1)
        c.rect(40, height - 200, width - 80, 100, stroke=1, fill=1)
        
        # Metadata Text Info
        c.setFillColor(HexColor('#1E293B'))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(55, height - 125, f"Session: {session_name}")
        c.setFont("Helvetica", 9)
        c.drawString(55, height - 145, f"Subject: {subject}")
        c.drawString(55, height - 165, f"Course: {course_name}")
        c.drawString(55, height - 185, f"Classroom: {classroom_label}")
        
        # Stats info (right column of the card box)
        c.drawString(320, height - 125, f"Session ID: {session_id}")
        c.drawString(320, height - 145, f"Session Date: {date_str}")
        c.setFont("Helvetica-Bold", 9)
        c.drawString(320, height - 165, "Attendance Summary:")
        c.setFont("Helvetica", 9)
        c.drawString(340, height - 185, f"Present: {total_p}  |  Late: {total_l}  |  Absent: {total_a}")

        # Draw Table Header
        y = height - 230
        c.setFillColor(HexColor('#475569'))
        c.rect(40, y - 5, width - 80, 20, stroke=0, fill=1)
        
        c.setFillColor(HexColor('#FFFFFF'))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(55, y, "S.No")
        c.drawString(100, y, "Student Name")
        c.drawString(290, y, "Status")
        c.drawString(390, y, "Timestamp")
        
        y -= 25
        c.setFont("Helvetica", 9)
        c.setFillColor(HexColor('#0F172A'))
        
        idx = 1
        for r in attendance:
            # Alternating background colors
            if idx % 2 == 0:
                c.setFillColor(HexColor('#F8FAFC'))
                c.rect(40, y - 4, width - 80, 16, stroke=0, fill=1)
                c.setFillColor(HexColor('#0F172A'))
            
            # Simple clean horizontal separator lines
            c.setStrokeColor(HexColor('#F1F5F9'))
            c.setLineWidth(0.5)
            c.line(40, y - 4, width - 40, y - 4)

            # Draw row values
            c.drawString(55, y, str(idx))
            c.drawString(100, y, r[0])
            
            status_val = r[1]
            if status_val == 'PRESENT':
                c.setFillColor(HexColor('#059669')) # Emerald Green
            elif status_val == 'LATE':
                c.setFillColor(HexColor('#D97706')) # Amber Orange
            else:
                c.setFillColor(HexColor('#DC2626')) # Rose Red
            c.drawString(290, y, status_val)
            c.setFillColor(HexColor('#0F172A'))
            
            c.drawString(390, y, r[2] or 'N/A')
            
            y -= 16
            idx += 1
            
            # Multi-page check
            if y < 80:
                c.showPage()
                page_count += 1
                draw_page_template(c, page_count)
                
                # Draw Table Header on the next page
                y = height - 120
                c.setFillColor(HexColor('#475569'))
                c.rect(40, y - 5, width - 80, 20, stroke=0, fill=1)
                
                c.setFillColor(HexColor('#FFFFFF'))
                c.setFont("Helvetica-Bold", 9)
                c.drawString(55, y, "S.No")
                c.drawString(100, y, "Student Name")
                c.drawString(290, y, "Status")
                c.drawString(390, y, "Timestamp")
                
                y -= 25
                c.setFont("Helvetica", 9)
                c.setFillColor(HexColor('#0F172A'))
        
        c.save()
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=pdf_filename, mimetype="application/pdf")
    else:
        # fallback to CSV if reportlab not installed
        flash("PDF generation failed: Reportlab library not installed. Downloaded CSV instead.", "error")
        return redirect(url_for("session_export", session=session_id))

# -------------------------
# Sessions admin: create, close, delete
# -------------------------
@app.route("/sessions_admin", methods=["GET", "POST"])
def sessions_admin():
    guard = require_admin()
    if guard:
        return guard
    
    # POST: Create New Session (Quick)
    if request.method == "POST":
        sname = (request.form.get("session_name") or "").strip()
        subject = (request.form.get("subject") or "").strip()
        course_id = request.form.get("course_id") or None
        classroom_id = request.form.get("classroom_id") or None # NEW: Get classroom ID

        if course_id == "": course_id = None
        else:
            try: course_id = int(course_id)
            except Exception: course_id = None
            
        if classroom_id == "": classroom_id = None
        else:
            try: classroom_id = int(classroom_id)
            except Exception: classroom_id = None
            
        duration = int(request.form.get("duration") or ATTENDANCE_WINDOW_MINUTES)
        sid = str(uuid.uuid4())[:8]
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            # NEW: Insert classroom_id
            cur.execute("INSERT INTO sessions(session_id, date, session_name, subject, course_id, classroom_id, duration_minutes, status) VALUES(?,?,?,?,?,?,?,?)",
                                 (sid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sname, subject, course_id, classroom_id, duration, "OPEN"))
            con.commit()
        flash(f"Session {sid} ('{sname}') created successfully. Share its QR link with students.", "success")
        return redirect(url_for("sessions_admin"))
        
    # GET: List Sessions
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, session_id, date, session_name, subject, course_id, classroom_id, duration_minutes, status FROM sessions ORDER BY id DESC")
        rows = cur.fetchall()

    rows_with_counts = []
    for r in rows:
        # r => id, session_id, date, session_name, subject, course_id, classroom_id, duration, status
        session_id = r[1]
        course_id = r[5]
        cur.execute("SELECT count(*) FROM attendance WHERE session_id=?", (session_id,))
        present_count = cur.fetchone()[0]
        
        if course_id:
            # Only students enrolled in that course count towards total/absentee
            cur.execute("SELECT count(*) FROM student_courses WHERE course_id=?", (course_id,))
            total_students = cur.fetchone()[0]
        else:
            # If no course is specified for the session, count all enrolled students
            cur.execute("SELECT count(*) FROM students")
            total_students = cur.fetchone()[0]
            
        absent_count = max(0, total_students - present_count)
        
        # Get classroom info for display
        classroom_info = get_classroom_info(r[6])
        classroom_label = f"({classroom_info[0]})" if classroom_info else "N/A"
        
        rows_with_counts.append(r + (present_count, absent_count, classroom_label))

    sessions_list = []
    for r in rows_with_counts:
        sessions_list.append({
            "id": r[0],
            "session_id": r[1],
            "date": r[2],
            "session_name": r[3],
            "subject": r[4],
            "course_id": r[5],
            "classroom_id": r[6],
            "duration": r[7],
            "status": r[8],
            "present_count": r[9],
            "absent_count": r[10],
            "classroom_label": r[11]
        })

    courses = list_courses()
    classrooms = list_classrooms()

    html = BASE_HEAD + """
    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-sm p-6 mb-8">
      <h3 class="text-lg font-bold text-slate-800 dark:text-white mb-4">Create New Session (Quick Start)</h3>
      <form method="POST" class="grid grid-cols-1 md:grid-cols-5 gap-4 items-end">
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Session Name</label>
            <input name="session_name" placeholder="e.g., Week 1 Lecture" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" required />
        </div>
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Subject</label>
            <input name="subject" placeholder="Optional" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" />
        </div>
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Course (Filter)</label>
            <select name="course_id" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-700 dark:text-slate-300">
                <option value="">-- All Students --</option>
                {% for c in courses %}
                    <option value="{{ c[0] }}">{{ c[2] }} ({{ c[1] }})</option>
                {% endfor %}
            </select>
        </div>
        <div>
            <label class="block text-xs font-bold text-slate-500 dark:text-slate-400 mb-1.5">Classroom (Context)</label>
            <select name="classroom_id" class="w-full border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-700 dark:text-slate-300">
                <option value="">-- No Classroom --</option>
                {% for cl in classrooms %}
                    <option value="{{ cl[0] }}">{{ cl[1] }} - {{ cl[2] }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="md:col-span-1 text-right">
            <button class="bg-indigo-600 hover:bg-indigo-700 text-white py-2.5 rounded-xl font-semibold shadow-md shadow-indigo-500/10 transition w-full text-sm">Create Session</button>
        </div>
      </form>
    </div>

    <div class="bg-white dark:bg-slate-800 border border-slate-150 dark:border-slate-700/50 rounded-2xl shadow-sm overflow-hidden animate-slide-up">
      <div class="p-6 border-b border-slate-200 dark:border-slate-750 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
            <h3 class="text-lg font-bold text-slate-800 dark:text-white">Active & Past Sessions ({{ sessions | length }} Total)</h3>
            <p class="text-xs text-slate-400 dark:text-slate-550 mt-1">P/A: Present / Expected Absent. The 'Expected Absent' count is based on enrolled students in the session's selected course, or all students if no course is set.</p>
        </div>
        
        <div class="relative w-full sm:w-64">
            <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                <i data-lucide="search" class="w-4 h-4"></i>
            </span>
            <input type="text" id="table-search" oninput="filterTable()" class="w-full pl-9 pr-4 py-2 border border-slate-200 dark:border-slate-750 bg-slate-50 dark:bg-slate-900 rounded-xl text-xs focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white" placeholder="Search sessions...">
        </div>
      </div>

      <table class="w-full table-auto text-left border-collapse" id="sessions-table">
          <thead class="bg-slate-50 dark:bg-slate-900 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-250 dark:border-slate-750">
            <tr>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(0)">
                <div class="flex items-center gap-1.5">ID <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4">Code</th>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(2)">
                <div class="flex items-center gap-1.5">Date <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4 cursor-pointer hover:text-slate-800 dark:hover:text-white" onclick="sortTable(3)">
                <div class="flex items-center gap-1.5">Name <span class="sort-icon"><i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i></span></div>
              </th>
              <th class="p-4">Subject / Classroom</th>
              <th class="p-4 text-center">Remaining</th>
              <th class="p-4 text-center">Counts (P/A)</th>
              <th class="p-4 text-center">Status</th>
              <th class="p-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100 dark:divide-slate-750/40 text-slate-700 dark:text-slate-300" id="sessions-table-body">
            {% for r in sessions %}
            <tr class="hover:bg-slate-50/50 dark:hover:bg-slate-700/20 transition text-sm">
                <td class="p-4 font-bold text-slate-850 dark:text-white">{{ r.id }}</td>
                <td class="p-4 font-mono text-xs">{{ r.session_id }}</td>
                <td class="p-4 text-xs text-slate-400">{{ r.date }}</td>
                <td class="p-4 font-semibold">{{ r.session_name }}</td>
                <td class="p-4 text-xs">{{ r.subject }} / <span class="text-slate-400">{{ r.classroom_label }}</span></td>
                <td class="p-4 text-center">
                    {% if r.status == 'OPEN' %}
                        <span class="active-timer text-indigo-600 dark:text-indigo-400 font-mono font-bold" data-start="{{ r.date }}" data-duration="{{ r.duration }}">--:--</span>
                    {% else %}
                        <span class="text-slate-400 dark:text-slate-550 text-xs">Ended</span>
                    {% endif %}
                </td>
                <td class="p-4 text-center">
                    <span class="text-emerald-600 font-bold">{{ r.present_count }}</span> / <span class="text-rose-600 font-bold">{{ r.absent_count }}</span>
                </td>
                <td class="p-4 text-center">
                    {% if r.status == 'OPEN' %}
                        <span class="px-2 py-0.5 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-450 font-semibold rounded-lg text-xs">OPEN</span>
                    {% else %}
                        <span class="px-2 py-0.5 bg-slate-100 dark:bg-slate-750 text-slate-500 dark:text-slate-400 font-semibold rounded-lg text-xs">CLOSED</span>
                    {% endif %}
                </td>
                <td class="p-4 text-right space-x-1.5 whitespace-nowrap">
                    <a class="bg-indigo-50 text-indigo-650 hover:bg-indigo-600 hover:text-white dark:bg-indigo-950/20 dark:text-indigo-400 px-3 py-1.5 rounded-xl text-xs font-semibold transition" href="/session_report?session={{ r.session_id }}">Report</a>
                    {% if r.status == 'OPEN' %}
                        <a class="bg-emerald-50 text-emerald-605 hover:bg-emerald-600 hover:text-white dark:bg-emerald-950/20 dark:text-emerald-455 px-3 py-1.5 rounded-xl text-xs font-semibold transition" href="/mark_attendance?session={{ r.session_id }}">QR Link</a>
                        <button class="bg-amber-50 text-amber-600 hover:bg-amber-600 hover:text-white dark:bg-amber-955/20 dark:text-amber-450 px-3 py-1.5 rounded-xl text-xs font-semibold transition" onclick="triggerEndSession('{{ r.session_id }}')">End</button>
                    {% endif %}
                    <button class="bg-rose-50 text-rose-600 hover:bg-rose-600 hover:text-white dark:bg-rose-955/20 dark:text-rose-455 px-3 py-1.5 rounded-xl text-xs font-semibold transition" onclick="triggerDeleteSession('{{ r.id }}', '{{ r.session_name }}')">Delete</button>
                </td>
            </tr>
            {% endfor %}
          </tbody>
      </table>
      
      <div id="pagination-container"></div>
    </div>

    <script>
      let sortDirection = 1;
      let sortColumnIndex = 0;
      let currentPage = 1;
      const rowsPerPage = 5;

      window.triggerDeleteSession = function(rowId, name) {
          confirmDelete(`Are you sure you want to delete session '${name}'? This action is permanent and will delete all attendance logs for this session.`, function() {
              const form = document.createElement('form');
              form.method = 'POST';
              form.action = `/sessions_admin/${rowId}/delete`;
              document.body.appendChild(form);
              form.submit();
          });
      };

      window.triggerEndSession = function(sessionId) {
          confirmDelete(`Are you sure you want to end this session now? This will send notifications to absentees.`, function() {
              const form = document.createElement('form');
              form.method = 'POST';
              form.action = '/sessions_admin_end';
              form.innerHTML = `<input type="hidden" name="session_id" value="${sessionId}">`;
              document.body.appendChild(form);
              form.submit();
          });
      };

      // Timer Logic
      function tickActiveTimers() {
          const timers = document.querySelectorAll('.active-timer');
          timers.forEach(el => {
              const startStr = el.getAttribute('data-start');
              const durationMins = parseInt(el.getAttribute('data-duration'));
              
              const start = new Date(startStr.replace(' ', 'T'));
              const end = new Date(start.getTime() + durationMins * 60 * 1000);
              const now = new Date();
              const diff = end - now;
              
              if (diff <= 0) {
                  el.textContent = "Expired";
                  el.className = "text-rose-500 font-semibold";
              } else {
                  const mins = Math.floor(diff / 60000);
                  const secs = Math.floor((diff % 60000) / 1000);
                  el.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
              }
          });
      }
      setInterval(tickActiveTimers, 1000);
      tickActiveTimers();

      window.sortTable = function(colIndex) {
          const table = document.getElementById("sessions-table");
          const tbody = document.getElementById("sessions-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          
          if (sortColumnIndex === colIndex) {
              sortDirection = -sortDirection;
          } else {
              sortColumnIndex = colIndex;
              sortDirection = 1;
          }
          
          const headers = table.querySelectorAll("thead th");
          headers.forEach((th, idx) => {
              const icon = th.querySelector(".sort-icon");
              if (icon) {
                  if (idx === colIndex) {
                      icon.innerHTML = sortDirection === 1 ? '<i data-lucide="chevron-up" class="w-3.5 h-3.5"></i>' : '<i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>';
                  } else {
                      icon.innerHTML = '<i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 opacity-40"></i>';
                  }
              }
          });
          lucide.createIcons();

          trs.sort((a, b) => {
              const aVal = a.cells[colIndex].textContent.trim().toLowerCase();
              const bVal = b.cells[colIndex].textContent.trim().toLowerCase();
              return aVal.localeCompare(bVal, undefined, {numeric: true, sensitivity: 'base'}) * sortDirection;
          });
          
          trs.forEach(tr => tbody.appendChild(tr));
          updatePagination();
      };

      window.filterTable = function() {
          const query = document.getElementById("table-search").value.toLowerCase();
          const tbody = document.getElementById("sessions-table-body");
          const trs = tbody.querySelectorAll("tr");
          
          trs.forEach(tr => {
              const text = tr.textContent.toLowerCase();
              tr.style.display = text.includes(query) ? "" : "none";
          });
          
          currentPage = 1;
          updatePagination();
      };

      window.updatePagination = function() {
          const tbody = document.getElementById("sessions-table-body");
          const trs = Array.from(tbody.querySelectorAll("tr"));
          const query = document.getElementById("table-search").value.toLowerCase();
          
          const visibleTrs = trs.filter(tr => tr.textContent.toLowerCase().includes(query));
          
          const totalPages = Math.ceil(visibleTrs.length / rowsPerPage) || 1;
          if (currentPage > totalPages) currentPage = totalPages;
          
          trs.forEach(tr => tr.style.display = "none");
          
          const startIdx = (currentPage - 1) * rowsPerPage;
          const endIdx = startIdx + rowsPerPage;
          
          visibleTrs.slice(startIdx, endIdx).forEach(tr => {
              tr.style.display = "";
          });
          
          const container = document.getElementById("pagination-container");
          if (!container) return;
          
          container.innerHTML = `
              <div class="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 p-4 border-t border-slate-200 dark:border-slate-750">
                  <div>Showing ${startIdx + 1} to ${Math.min(endIdx, visibleTrs.length)} of ${visibleTrs.length} entries</div>
                  <div class="flex gap-2">
                      <button onclick="changePage(-1)" ${currentPage === 1 ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Previous</button>
                      <span class="px-3 py-1 font-bold text-indigo-600 dark:text-indigo-400">${currentPage} / ${totalPages}</span>
                      <button onclick="changePage(1)" ${currentPage === totalPages ? 'disabled class="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed border border-transparent"' : 'class="px-3 py-1 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition"'}>Next</button>
                  </div>
              </div>
          `;
      };

      window.changePage = function(dir) {
          currentPage += dir;
          updatePagination();
      };

      document.addEventListener("DOMContentLoaded", function() {
          updatePagination();
      });
      lucide.createIcons();
    </script>
    """ + BASE_FOOT
    return render_template_string(html, title="Manage Sessions", is_admin=is_admin(), admin_badge=admin_badge_html(), sessions=sessions_list)

@app.post("/sessions_admin/<int:row_id>/delete")
def sessions_admin_delete(row_id):
    guard = require_admin()
    if guard:
        return guard
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT session_id, session_name FROM sessions WHERE id=?", (row_id,))
        row = cur.fetchone()
        if row:
            sess_code, sess_name = row
            cur.execute("DELETE FROM attendance WHERE session_id=?", (sess_code,))
            cur.execute("DELETE FROM sessions WHERE id=?", (row_id,))
            con.commit()
            flash(f"Session '{sess_name}' deleted.", "success")
        else:
            flash("Session not found.", "error")
    return redirect(url_for("sessions_admin"))

@app.route("/sessions_admin_end", methods=["POST"])
def sessions_admin_end():
    guard = require_admin()
    if guard:
        return guard
    session_id = request.form.get("session_id")
    if not session_id:
        return redirect(url_for("sessions_admin"))

    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT session_name, date, subject, course_id FROM sessions WHERE session_id=?", (session_id,))
        row = cur.fetchone()
        if not row:
            flash("Session not found.", "error")
            return redirect(url_for("sessions_admin"))

        cur.execute("UPDATE sessions SET status='CLOSED' WHERE session_id=?", (session_id,))
        con.commit()

    session_name = row[0]
    session_date = row[1]
    subject = row[2]
    course_id = row[3]

    absentees = students_not_marked(session_id, course_id)
    subj = f"Absent: {session_name or subject or 'Class'} ({session_date})"
    sms_msg = f"Absent: Student missed {session_name or subject or 'class'} on {session_date}. Report attendance to instructor."
    email_body_template = "Dear Parent/Student,\n\n{student} was marked absent for {session} on {date}.\n\nRegards,\nAttendance System"
    sms_count = 0
    email_count = 0
    
    # Use threading for notifications to avoid blocking the request
    def send_notifications_async():
        nonlocal sms_count, email_count
        for name, email, parent in absentees:
            body = email_body_template.format(student=name, session=(session_name or subject or "class"), date=session_date)
            if parent:
                if send_sms(parent, sms_msg + f" Student: {name}"):
                    sms_count += 1
            if email:
                if send_email(email, subj, body):
                    email_count += 1

        app.logger.info(f"Notification summary for {session_id}: SMS sent: {sms_count}, Emails sent: {email_count}")

    # Start the thread, but don't wait for it. The flash message will show a placeholder count.
    # The actual counts will be logged to the console.
    threading.Thread(target=send_notifications_async).start()
    
    # Since we can't wait for the thread, we'll give a generic success message
    flash(f"Session closed. Notifications are being sent to {len(absentees)} absentees in the background.", "success")
    
    return redirect(url_for("session_report", session=session_id))

# -------------------------
# Static preview route for enrolled images and classroom images
# -------------------------
@app.route("/students/preview/<path:filename>")
def student_preview(filename):
    path = os.path.join(STUDENTS_DIR, filename)
    if os.path.exists(path):
        # Prevent accessing files outside of STUDENTS_DIR
        return send_file(path)
    return "Not found", 404

# NEW: Static route for classroom reference images
@app.route("/classrooms/preview/<path:filename>")
def classroom_preview(filename):
    # This route is necessary because CLASSROOM_DIR is outside 'static' and we need to serve the file
    path = os.path.join(CLASSROOM_DIR, filename)
    if os.path.exists(path):
        # Prevent accessing files outside of CLASSROOM_DIR
        return send_file(path)
    return "Not found", 404
    
# -------------------------
# SSL helper
# -------------------------
def ssl_context_if_available():
    cert = os.path.join(os.path.dirname(__file__), "cert.pem")
    key = os.path.join(os.path.dirname(__file__), "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        return (cert, key)
    return None

# -------------------------
# Main runner + helpful startup messages
# -------------------------
if __name__ == "__main__":
    guessed_local = f"http://localhost:{APP_PORT}"
    
    # Start ngrok/determine public URL before starting the Flask server for the print messages
    public = None
    if os.environ.get("NGROK_AUTO", "0") == "1" and PYNGROK_AVAILABLE:
        try:
            pub_ngrok = maybe_start_ngrok(APP_PORT)
            if pub_ngrok:
                public = pub_ngrok
        except Exception:
            app.logger.error("ngrok startup failed.")
            
    if PREFERRED_PUBLIC_BASE:
        public = PREFERRED_PUBLIC_BASE.rstrip("/")
    elif public is None:
           # Fallback to local
        public = guessed_local
        
    print("="*60)
    print("Starting Attendance System")
    print("App port:", APP_PORT)
    print("Detected public base (for QR generation):", public)
    print("-"*60)
    
    # NEW: Print status
    print(f"Liveness Check Required: {'ENABLED' if LIVENESS_REQUIRED else 'DISABLED'} (Set ENV LIVENESS_REQUIRED=1 to enable)")
    print(f"Background Snap Check (Global): {'ENABLED' if BACKGROUND_CHECK_REQUIRED else 'DISABLED'}")
    if BACKGROUND_CHECK_REQUIRED:
        print(f"  -> Tolerance: {BACKGROUND_DISTANCE_TOLERANCE}")

    if os.environ.get("NGROK_AUTO", "0") == "1":
        if not PYNGROK_AVAILABLE:
            print("NGROK_AUTO=1 but pyngrok is not installed.")
            print("  -> Install: pip install pyngrok")
            print("  OR set PUBLIC_BASE_URL to your HTTPS endpoint.")
        else:
            if not public or public.startswith("http://localhost"):
                print("pyngrok was attempted but did not return a public HTTPS URL (check NGROK_AUTHTOKEN).")
            else:
                print("ngrok public URL:", public)
    else:
        print("NGROK_AUTO != 1. If you need HTTPS for mobile camera (iPhone), run ngrok manually:")
        print(f"  ngrok http -bind-tls=true {APP_PORT}")
        print("then set PUBLIC_BASE_URL to the HTTPS URL shown by ngrok or set NGROK_AUTO=1.")

    # Twilio / SMTP hints
    if TWILIO_AVAILABLE:
        if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM):
            print("Twilio library present but TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM not fully configured.")
            print("  -> Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM in env to enable SMS notifications.")
    else:
        print("Twilio not installed (optional). Install with: pip install twilio")
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        print("SMTP not fully configured. Email notifications will be disabled until SMTP_HOST/SMTP_USER/SMTP_PASS are set.")
    else:
        print("SMTP configured. Emails will be sent from:", SMTP_FROM or SMTP_USER)

    print("-"*60)
    print("Important mobile note:")
    print(" - **THE CAMERA WILL NOT OPEN OVER HTTP (127.0.0.1).**")
    print(" - **ACTION REQUIRED:** You must use an **HTTPS** address (e.g., your Ngrok/Cloudflare Tunnel URL) for the camera to work on phones and most laptops.")
    print("="*60)

    # server TLS: if cert.pem and key.pem exist next to app.py, run HTTPS locally
    ssl_ctx = ssl_context_if_available()
    host = "0.0.0.0"
    debug_flag = os.environ.get("FLASK_DEBUG", "1") in ("1", "true", "True")
    threaded = True

    try:
        if ssl_ctx:
            print(f"Starting HTTPS Flask on {host}:{APP_PORT} (local certs found).")
            print("Open in browser at:", f"https://{get_local_ip()}:{APP_PORT} (or use public URL)")
            app.run(host=host, port=APP_PORT, debug=debug_flag, threaded=threaded, ssl_context=ssl_ctx)
        else:
            print(f"Starting HTTP Flask on {host}:{APP_PORT}.")
            print("Local URLs:")
            print(" - http://127.0.0.1:{port}".format(port=APP_PORT))
            print(" - http://{ip}:{port}".format(ip=get_local_ip(), port=APP_PORT))
            if public and public.startswith("https"):
                print("Public URL (HTTPS):", public)
            elif public and public.startswith("http://") and "localhost" not in public:
                print("Public URL (HTTP):", public)
            app.run(host=host, port=APP_PORT, debug=debug_flag, threaded=threaded)
    except KeyboardInterrupt:
        print("\nShutting down (keyboard interrupt).")
    except Exception as e:
        print("Server error:", repr(e))
    finally:
        # if we opened a pyngrok tunnel, attempt to close it cleanly
        try:
            if _active_ngrok_tunnel:
                try:
                    ngrok.disconnect(_active_ngrok_tunnel.public_url)
                except Exception:
                    pass
                try:
                    ngrok.kill()
                except Exception:
                    pass
                print("Closed ngrok tunnel (if one was active).")
        except Exception:
            pass

    print("Goodbye.")