"""三层长期记忆（SQLite）：L1 会话消息 / L2 长期事实 / L3 检测档案。

设计见 agent.md §6。表结构自动建库，线程安全（Gradio 事件并发）。
"""
import json
import sqlite3
import threading
from pathlib import Path

from agent.config import CFG

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS facts (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS detections (
  result_id TEXT PRIMARY KEY,
  session_id TEXT,
  image_path TEXT NOT NULL,
  image_hash TEXT NOT NULL,
  conf REAL, iou REAL,
  num_objects INTEGER,
  summary_json TEXT,
  vis_path TEXT,
  report_paths TEXT,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id, created_at);
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(CFG.store_dir).mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(Path(CFG.store_dir) / "memory.db"),
                                check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


# ---------- L1 会话记忆 ----------

def save_message(session_id: str, role: str, content: str, max_len: int = 6000) -> None:
    with _lock:
        _db().execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?,?,?)",
            (session_id, role, str(content)[:max_len]))
        _db().commit()


def recent_messages(session_id: str, limit: int = 20) -> list[dict]:
    """按时间正序返回最近 limit 条（最早->最新），供上下文注入。"""
    rows = _db().execute(
        "SELECT role, content FROM (SELECT id, role, content FROM messages "
        "WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
        (session_id, limit)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ---------- L2 长期事实 ----------

def save_fact(key: str, value: str) -> None:
    with _lock:
        _db().execute(
            "INSERT INTO facts(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=datetime('now','localtime')", (key, str(value)))
        _db().commit()


def load_facts() -> dict[str, str]:
    rows = _db().execute("SELECT key, value FROM facts").fetchall()
    return {r["key"]: r["value"] for r in rows}


# ---------- L3 检测档案 ----------

def save_detection(rec: dict) -> None:
    with _lock:
        _db().execute(
            "INSERT OR REPLACE INTO detections(result_id, session_id, image_path, "
            "image_hash, conf, iou, num_objects, summary_json, vis_path, report_paths) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rec["result_id"], rec.get("session_id"), rec["image_path"],
             rec["image_hash"], rec.get("conf"), rec.get("iou"),
             rec.get("num_objects", 0), json.dumps(rec.get("summary", {}), ensure_ascii=False),
             rec.get("vis_path"), json.dumps(rec.get("report_paths", {}), ensure_ascii=False)))
        _db().commit()


def get_detection(result_id: str) -> dict | None:
    row = _db().execute("SELECT * FROM detections WHERE result_id=?",
                        (result_id,)).fetchone()
    return _row_to_rec(row) if row else None


def list_detections(keyword: str | None = None, limit: int = 10,
                    session_id: str | None = None) -> list[dict]:
    sql = "SELECT * FROM detections WHERE 1=1"
    args: list = []
    if keyword:
        sql += " AND image_path LIKE ?"
        args.append(f"%{keyword}%")
    if session_id:
        sql += " AND session_id=?"
        args.append(session_id)
    sql += " ORDER BY created_at DESC, result_id DESC LIMIT ?"
    args.append(limit)
    return [_row_to_rec(r) for r in _db().execute(sql, args).fetchall()]


def latest_detection(session_id: str | None = None) -> dict | None:
    recs = list_detections(limit=1, session_id=session_id)
    return recs[0] if recs else None


def save_report_paths(result_id: str, fmt: str, path: str) -> None:
    rec = get_detection(result_id)
    if not rec:
        return
    paths = rec.get("report_paths", {})
    paths[fmt] = path
    with _lock:
        _db().execute("UPDATE detections SET report_paths=? WHERE result_id=?",
                      (json.dumps(paths, ensure_ascii=False), result_id))
        _db().commit()


def _row_to_rec(row: sqlite3.Row) -> dict:
    rec = dict(row)
    rec["summary"] = json.loads(rec.pop("summary_json") or "{}")
    rec["report_paths"] = json.loads(rec.pop("report_paths") or "{}")
    return rec
