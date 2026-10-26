import os, sqlite3
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()
DB_PATH = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "history.db"))

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with get_db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sms_messages(id INTEGER PRIMARY KEY, sender_id TEXT, message TEXT NOT NULL,
          prediction TEXT NOT NULL, category TEXT NOT NULL, risk_score REAL NOT NULL, model_name TEXT,
          severity TEXT NOT NULL, reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS model_metrics(id INTEGER PRIMARY KEY, model_name TEXT, accuracy REAL, precision REAL,
          recall REAL, f1 REAL, confusion_matrix TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('alert_threshold','0.75')")

def create_message(sender, message, prediction, category, score, model, severity, reason):
    with get_db() as c:
        cur = c.execute("""INSERT INTO sms_messages(sender_id,message,prediction,category,risk_score,model_name,severity,reason)
          VALUES(?,?,?,?,?,?,?,?)""", (sender, message, prediction, category, score, model, severity, reason))
        return cur.lastrowid

def get_settings():
    with get_db() as c:
        return {r["key"]: r["value"] for r in c.execute("SELECT key,value FROM settings").fetchall()}

def set_setting(key, value):
    with get_db() as c:
        c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value))
