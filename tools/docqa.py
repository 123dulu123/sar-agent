"""轻量文档问答（RAG-lite）：SAR 判读资料导入 + FTS5 trigram 检索。

设计取舍：用 SQLite FTS5（trigram 分词）做关键词检索而非向量检索——
零新依赖（embedding API 不稳定时也能用）、中文可用；升级到向量 RAG 的
路径见 agent.md §10。支持 txt / md / docx / pdf。
"""
import re
import sqlite3
import threading
from pathlib import Path

from agent.config import CFG

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

MAX_CHUNK_CHARS = 500
SUPPORTED = {".txt", ".md", ".docx", ".pdf"}


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(CFG.store_dir).mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(Path(CFG.store_dir) / "docs.db"),
                                check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        try:
            _conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
                    text, doc_id UNINDEXED, chunk_no UNINDEXED, tokenize='trigram');
                CREATE TABLE IF NOT EXISTS docs(
                    doc_id TEXT PRIMARY KEY, name TEXT, chunks INTEGER,
                    created_at TEXT DEFAULT (datetime('now','localtime')));
            """)
        except sqlite3.OperationalError as e:  # 老 sqlite 无 trigram 时降级
            _conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
                    text, doc_id UNINDEXED, chunk_no UNINDEXED);
                CREATE TABLE IF NOT EXISTS docs(
                    doc_id TEXT PRIMARY KEY, name TEXT, chunks INTEGER,
                    created_at TEXT DEFAULT (datetime('now','localtime')));
            """)
            print(f"FTS5 trigram 不可用（{e}），使用默认分词")
        _conn.commit()
    return _conn


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in
                         PdfReader(str(path)).pages)
    raise ValueError(f"不支持的文档类型：{suffix}（支持 {sorted(SUPPORTED)}）")


def _chunk(text: str) -> list[str]:
    """按段落聚合到约 MAX_CHUNK_CHARS 的块，保留上下文完整性。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 > MAX_CHUNK_CHARS and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def ingest(path: str | Path) -> dict:
    """导入一份文档并建索引；同名文档重复导入会先清旧索引。"""
    path = Path(path)
    if not path.exists():
        return {"ok": False, "error": f"文件不存在：{path}"}
    if path.suffix.lower() not in SUPPORTED:
        return {"ok": False, "error": f"不支持的类型 {path.suffix}（支持 {sorted(SUPPORTED)}）"}
    try:
        text = _read_text(path)
    except Exception as e:
        return {"ok": False, "error": f"文档解析失败：{e}"}
    chunks = _chunk(text)
    if not chunks:
        return {"ok": False, "error": "未解析出有效文本（可能是扫描版 PDF）"}

    import uuid
    doc_id = uuid.uuid4().hex[:10]
    name = path.name
    with _lock:
        old = _db().execute("SELECT doc_id FROM docs WHERE name=?", (name,)).fetchone()
        if old:
            _db().execute("DELETE FROM doc_fts WHERE doc_id=?", (old["doc_id"],))
            _db().execute("DELETE FROM docs WHERE doc_id=?", (old["doc_id"],))
        for i, c in enumerate(chunks):
            _db().execute("INSERT INTO doc_fts(text, doc_id, chunk_no) VALUES (?,?,?)",
                          (c, doc_id, i))
        _db().execute("INSERT INTO docs(doc_id, name, chunks) VALUES (?,?,?)",
                      (doc_id, name, len(chunks)))
        _db().commit()
    return {"ok": True, "doc_id": doc_id, "name": name, "chunks": len(chunks)}


def search(query: str, k: int = 5) -> dict:
    """检索资料片段；trigram 需查询词 ≥3 字符，过短时降级为 LIKE。"""
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "查询词为空"}
    k = min(max(int(k), 1), 10)
    with _lock:
        try:
            rows = _db().execute(
                "SELECT text, doc_id, chunk_no, bm25(doc_fts) AS score "
                "FROM doc_fts WHERE doc_fts MATCH ? ORDER BY score LIMIT ?",
                (query, k)).fetchall()
        except sqlite3.OperationalError:
            rows = _db().execute(
                "SELECT text, doc_id, chunk_no, 1.0 AS score FROM doc_fts "
                "WHERE text LIKE ? LIMIT ?", (f"%{query}%", k)).fetchall()
        if not rows and len(query) < 3:
            rows = _db().execute(
                "SELECT text, doc_id, chunk_no, 1.0 AS score FROM doc_fts "
                "WHERE text LIKE ? LIMIT ?", (f"%{query}%", k)).fetchall()
    docs = {r["doc_id"]: r2 for r, r2 in
            ((r, _db().execute("SELECT name FROM docs WHERE doc_id=?",
                               (r["doc_id"],)).fetchone()) for r in rows)}
    hits = [{"doc": docs[r["doc_id"]]["name"] if docs[r["doc_id"]] else r["doc_id"],
             "chunk_no": r["chunk_no"], "text": r["text"][:600]} for r in rows]
    return {"ok": True, "query": query, "hits": hits, "count": len(hits)}


def list_docs() -> dict:
    with _lock:
        rows = _db().execute(
            "SELECT doc_id, name, chunks, created_at FROM docs ORDER BY created_at").fetchall()
    return {"ok": True, "docs": [dict(r) for r in rows]}


def store_doc_path(name: str) -> Path:
    """参考文档归档位置（把上传的资料留档，重启后可重新导入）。"""
    d = CFG.store_dir / "reference_docs"
    d.mkdir(parents=True, exist_ok=True)
    return d / name
