"""可视化与报告单测：不依赖 torch / 真实数据集。"""
import json
from pathlib import Path

import numpy as np

from agent.config import CFG
from tools import report as report_mod
from tools import visualizer as viz

BOXES = [
    {"cls_id": 0, "cls_name": "ship", "conf": 0.91, "xyxy": [10, 10, 40, 25]},
    {"cls_id": 0, "cls_name": "ship", "conf": 0.42, "xyxy": [100, 80, 128, 96]},
]


def test_draw_and_charts(tmp_path):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = viz.draw_boxes(img, BOXES)
    assert out.shape == img.shape and out.sum() > 0  # 画过框

    charts = viz.make_stats_charts(BOXES, img.shape, tmp_path, "t")
    for p in charts.values():
        p = Path(p)
        assert p.exists() and p.stat().st_size > 0


def test_report_markdown(tmp_path, monkeypatch):
    result_id = "r_ut_1"
    monkeypatch.setattr(CFG, "reports_dir", tmp_path)
    monkeypatch.setattr(CFG, "results_dir", tmp_path)
    monkeypatch.setattr(report_mod, "CFG", CFG)

    # 伪造检测产物
    data = {"result_id": result_id, "image_path": "fake.jpg",
            "weights": "fake.pt", "weights_source": "单测",
            "params": {"conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "cpu"},
            "boxes": BOXES,
            "summary": {"num_objects": 2, "cls_counts": {"ship": 2},
                        "conf_mean": 0.665, "conf_max": 0.91, "conf_min": 0.42,
                        "elapsed_sec": 0.5},
            "charts": {}}
    (tmp_path / f"{result_id}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")

    class _FakeMem:  # 隔离 memory 依赖
        @staticmethod
        def get_detection(rid):
            return {"result_id": rid, "created_at": "2026-09-28", "vis_path": None}
        @staticmethod
        def save_report_paths(*a, **k):
            pass
    monkeypatch.setattr(report_mod, "memory", _FakeMem)

    r = report_mod.generate(result_id, fmt="markdown", interpretation="测试解读。")
    assert r["ok"], r
    text = Path(r["path"]).read_text(encoding="utf-8")
    assert "检出目标数" in text
    assert "测试解读" in text and "ship" in text
