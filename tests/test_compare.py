"""对比功能单测：构造两次已知检测结果，验证差分与 IoU 匹配逻辑。"""
import json

from agent import memory
from agent.config import CFG
from tools.compare import compare


def _mk_result(rid, boxes):
    data = {"result_id": rid, "image_path": f"img_{rid}.jpg",
            "weights": "w.pt", "weights_source": "单测",
            "params": {"conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "cpu"},
            "boxes": boxes,
            "summary": {"num_objects": len(boxes), "cls_counts": {"ship": len(boxes)},
                        "conf_mean": round(sum(b["conf"] for b in boxes) / len(boxes), 4),
                        "conf_max": max(b["conf"] for b in boxes),
                        "conf_min": min(b["conf"] for b in boxes),
                        "elapsed_sec": 1.0}}
    (CFG.results_dir / f"{rid}.json").write_text(json.dumps(data, ensure_ascii=False),
                                                 encoding="utf-8")
    memory.save_detection({"result_id": rid, "session_id": "ut",
                           "image_path": data["image_path"], "image_hash": rid,
                           "conf": 0.25, "iou": 0.45, "num_objects": len(boxes),
                           "summary": data["summary"], "vis_path": None})


def test_compare_match_added_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(CFG, "results_dir", tmp_path)
    # A：两个目标（一个与 B 重叠，一个将"消失"）
    _mk_result("r_a", [
        {"cls_id": 0, "cls_name": "ship", "conf": 0.9, "xyxy": [10, 10, 40, 30]},
        {"cls_id": 0, "cls_name": "ship", "conf": 0.8, "xyxy": [200, 200, 230, 220]},
    ])
    # B：保留第一个（IoU≈1），新增两个
    _mk_result("r_b", [
        {"cls_id": 0, "cls_name": "ship", "conf": 0.95, "xyxy": [11, 10, 41, 30]},
        {"cls_id": 0, "cls_name": "ship", "conf": 0.7, "xyxy": [300, 50, 330, 70]},
        {"cls_id": 0, "cls_name": "ship", "conf": 0.6, "xyxy": [400, 50, 430, 70]},
    ])
    r = compare("r_a", "r_b")
    assert r["ok"]
    assert r["delta"]["num_objects"] == 1          # 2 -> 3
    assert r["delta"]["added"] == 2
    assert r["delta"]["removed"] == 1              # (200,200) 处的目标消失
    assert r["baseline"]["num_objects"] == 2 and r["current"]["num_objects"] == 3


def test_compare_missing_result(tmp_path, monkeypatch):
    monkeypatch.setattr(CFG, "results_dir", tmp_path)
    try:
        compare("r_none1", "r_none2")
        raised = False
    except KeyError:
        raised = True
    assert raised
