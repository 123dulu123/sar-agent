"""文档问答单测：导入 -> 检索 -> 引用，全部在临时库上执行。"""
import pytest

from agent.config import CFG
from tools import docqa

DOC = """SAR 舰船判读要点

舰船在 SAR 图像上通常表现为亮斑，伴有明显的尾迹特征。锚地中的舰船排列较为均匀，
间距稳定，朝向多样；港口泊位上的舰船则紧密排列且朝向一致。

虚警的常见来源包括波浪亮点、浮标、岛礁边缘等。低置信度检测框需要结合斑点噪声
水平与海杂波背景综合判断，必要时降低阈值进行复核。
"""


@pytest.fixture()
def tmp_docs(monkeypatch, tmp_path):
    monkeypatch.setattr(CFG, "store_dir", tmp_path)
    monkeypatch.setattr(docqa, "_conn", None)
    p = tmp_path / "manual.md"
    p.write_text(DOC, encoding="utf-8")
    r = docqa.ingest(p)
    assert r["ok"], r
    return r, p


def test_ingest_and_list(tmp_docs):
    info, path = tmp_docs
    assert info["chunks"] >= 1          # 小文档聚为 1 块也合法
    lst = docqa.list_docs()
    assert lst["docs"][0]["name"] == "manual.md"
    # 同名重复导入：覆盖旧索引
    r2 = docqa.ingest(path)
    assert r2["ok"]
    assert docqa.list_docs()["docs"][0]["doc_id"] == r2["doc_id"]


def test_search_chinese(tmp_docs):
    r = docqa.search("锚地")
    assert r["ok"] and r["count"] >= 1
    assert any("锚地" in h["text"] for h in r["hits"])
    assert all(h["doc"] == "manual.md" for h in r["hits"])


def test_search_no_hit(tmp_docs):
    r = docqa.search("量子纠缠")
    assert r["ok"] and r["count"] == 0


def test_ingest_rejects_bad_type(tmp_docs):
    bad = CFG.store_dir / "x.exe"
    bad.write_bytes(b"MZ")
    r = docqa.ingest(bad)
    assert not r["ok"] and "不支持" in r["error"]
