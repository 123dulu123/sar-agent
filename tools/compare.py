"""检测结果对比：两次检测的统计差分 + 基于 IoU 的空间目标匹配（新增/消失）。

语义：以 A 为基准、B 为新状态——added = B 中在 A 里找不到匹配的目标，
removed = A 中在 B 里找不到匹配的目标。匹配规则：同类别且 IoU ≥ 0.5，贪心配对。
"""
import json
from pathlib import Path

from agent import memory
from agent.config import CFG


def _iou(b1: list[float], b2: list[float]) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def _load(result_id: str) -> dict:
    rec = memory.get_detection(result_id)
    if rec is None:
        raise KeyError(f"检测结果不存在：{result_id}")
    p = CFG.results_dir / f"{result_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"结果文件缺失：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def compare(result_id_a: str, result_id_b: str, iou_match: float = 0.5) -> dict:
    """对比两次检测，返回统计差分与新增/消失目标明细。"""
    a, b = _load(result_id_a), _load(result_id_b)
    boxes_a, boxes_b = a.get("boxes", []), b.get("boxes", [])

    # 贪心匹配：置信度高的 B 框优先找同类别的最大 IoU A 框
    unmatched_a = list(range(len(boxes_a)))
    added = []
    for bb in sorted(boxes_b, key=lambda x: -x["conf"]):
        best_i, best_v = -1, iou_match
        for i in unmatched_a:
            if boxes_a[i]["cls_id"] != bb["cls_id"]:
                continue
            v = _iou(bb["xyxy"], boxes_a[i]["xyxy"])
            if v >= best_v:
                best_i, best_v = i, v
        if best_i >= 0:
            unmatched_a.remove(best_i)
        else:
            added.append(bb)
    removed = [boxes_a[i] for i in unmatched_a]

    def _cls_counts(boxes):
        out: dict[str, int] = {}
        for x in boxes:
            out[x["cls_name"]] = out.get(x["cls_name"], 0) + 1
        return out

    def _conf_mean(boxes):
        return round(sum(x["conf"] for x in boxes) / len(boxes), 4) if boxes else 0.0

    sa, sb = a.get("summary", {}), b.get("summary", {})
    return {
        "ok": True,
        "baseline": {"result_id": result_id_a, "image": Path(a["image_path"]).name,
                     "num_objects": sa.get("num_objects"),
                     "conf_mean": sa.get("conf_mean")},
        "current": {"result_id": result_id_b, "image": Path(b["image_path"]).name,
                    "num_objects": sb.get("num_objects"),
                    "conf_mean": sb.get("conf_mean")},
        "delta": {
            "num_objects": sb.get("num_objects", 0) - sa.get("num_objects", 0),
            "cls_counts_a": _cls_counts(boxes_a),
            "cls_counts_b": _cls_counts(boxes_b),
            "conf_mean": round(_conf_mean(boxes_b) - _conf_mean(boxes_a), 4),
            "added": len(added),
            "removed": len(removed),
        },
        "added_detail": [{"conf": x["conf"], "xyxy": x["xyxy"]} for x in added[:20]],
        "removed_detail": [{"conf": x["conf"], "xyxy": x["xyxy"]} for x in removed[:20]],
        "iou_match": iou_match,
        "vis_paths": {"baseline": a.get("vis_path"), "current": b.get("vis_path")},
    }
