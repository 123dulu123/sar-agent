"""记忆层单测：在临时目录建独立 memory.db，不污染真实数据。"""
import pytest

from agent import memory
from agent.config import CFG


@pytest.fixture()
def tmp_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(CFG, "store_dir", tmp_path)
    monkeypatch.setattr(memory, "_conn", None)
    yield


def test_fact_roundtrip(tmp_memory):
    memory.save_fact("conf_threshold", "0.3")
    memory.save_fact("conf_threshold", "0.35")  # 覆盖更新
    memory.save_fact("report_format", "docx")
    facts = memory.load_facts()
    assert facts["conf_threshold"] == "0.35"
    assert facts["report_format"] == "docx"


def test_message_order(tmp_memory):
    for i in range(5):
        memory.save_message("s1", "user", f"msg{i}")
    msgs = memory.recent_messages("s1", 3)
    assert [m["content"] for m in msgs] == ["msg2", "msg3", "msg4"]
    assert memory.recent_messages("s2", 3) == []  # 会话隔离


def test_detection_roundtrip(tmp_memory):
    rec = {"result_id": "r_test_1", "session_id": "s1",
           "image_path": "x/a.jpg", "image_hash": "h1",
           "conf": 0.25, "iou": 0.45, "num_objects": 3,
           "summary": {"cls_counts": {"ship": 3}}, "vis_path": "v.jpg"}
    memory.save_detection(rec)
    got = memory.get_detection("r_test_1")
    assert got["num_objects"] == 3
    assert got["summary"]["cls_counts"] == {"ship": 3}
    assert memory.latest_detection("s1")["result_id"] == "r_test_1"
    assert memory.latest_detection("s9") is None

    memory.save_report_paths("r_test_1", "markdown", "out.md")
    assert memory.get_detection("r_test_1")["report_paths"]["markdown"] == "out.md"

    assert memory.list_detections(keyword="a.jpg")[0]["result_id"] == "r_test_1"
    assert memory.list_detections(keyword="zzz") == []
