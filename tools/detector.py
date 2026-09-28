"""YOLOv8 检测封装：系统内唯一的检测入口（界面按钮与 Agent 工具共用）。

职责：加载权重 → 推理 → 结构化结果落盘（JSON + 标注图 + 统计图）→ 写入 L3 检测档案。
"""
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

import cv2

from agent import memory
from agent.config import CFG, ensure_dirs
from tools import visualizer as viz

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_device() -> str:
    if CFG.device != "auto":
        return CFG.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class ShipDetector:
    """惰性加载的 YOLOv8 单例；权重优先级见 prd.md §10.2。"""

    _instance: "ShipDetector | None" = None

    def __init__(self, weights: str | Path | None = None):
        from ultralytics import YOLO  # 延迟导入，加快 UI 启动
        weights = Path(weights) if weights else None
        if weights and weights.exists():
            chosen = weights
            source = "指定权重"
        elif CFG.default_weights.exists():
            chosen, source = CFG.default_weights, "SAR 舰船权重"
        else:
            chosen, source = CFG.fallback_weights, "兜底通用权重（管线演示模式，检测无意义）"
            log.warning("未找到 %s，使用 %s。请按 prd.md §10.2 放置 SAR 权重。",
                        CFG.default_weights, CFG.fallback_weights)
        self.device = _resolve_device()
        self.model = YOLO(str(chosen))
        self.weights_path = str(chosen)
        self.weights_source = source
        self.names = dict(self.model.names or {})
        log.info("YOLOv8 已加载：%s（%s，device=%s）", chosen, source, self.device)

    @classmethod
    def get(cls) -> "ShipDetector":
        if cls._instance is None:
            ensure_dirs()
            cls._instance = cls()
        return cls._instance

    def detect(self, image_path: str | Path, conf: float | None = None,
               iou: float | None = None, session_id: str | None = None) -> dict:
        """执行检测并落盘；返回完整结果 dict（含 result_id）。"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在：{image_path}")
        if not is_image_path(image_path):
            raise ValueError(f"不支持的文件类型：{image_path.suffix}")

        conf = CFG.conf if conf is None else float(conf)
        iou = CFG.iou if iou is None else float(iou)

        img = viz.imread_unicode(image_path)
        if img is None:
            raise ValueError(f"图像解码失败（文件损坏或格式不支持）：{image_path}")

        # 大图压缩到 max_side 长边内，防内存与耗时失控
        h, w = img.shape[:2]
        scale = CFG.max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        t0 = time.time()
        results = self.model.predict(source=img, conf=conf, iou=iou,
                                     imgsz=CFG.imgsz, device=self.device,
                                     verbose=False)
        elapsed = time.time() - t0
        r = results[0]

        boxes = []
        for b in r.boxes:
            cls_id = int(b.cls.item())
            boxes.append({
                "cls_id": cls_id,
                "cls_name": self.names.get(cls_id, str(cls_id)),
                "conf": round(float(b.conf.item()), 4),
                "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
            })
        boxes.sort(key=lambda b: -b["conf"])

        image_hash = file_md5(image_path)
        result_id = f"r_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        confs = [b["conf"] for b in boxes]
        summary = {
            "num_objects": len(boxes),
            "cls_counts": _count_by(boxes, "cls_name"),
            "conf_mean": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "conf_max": round(max(confs), 4) if confs else 0.0,
            "conf_min": round(min(confs), 4) if confs else 0.0,
            "elapsed_sec": round(elapsed, 3),
        }

        result = {
            "result_id": result_id,
            "session_id": session_id,
            "image_path": str(image_path),
            "image_hash": image_hash,
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "weights": self.weights_path,
            "weights_source": self.weights_source,
            "params": {"conf": conf, "iou": iou, "imgsz": CFG.imgsz,
                       "device": self.device},
            "boxes": boxes,
            "summary": summary,
        }

        result_json = CFG.results_dir / f"{result_id}.json"
        result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        annotated = viz.draw_boxes(img, boxes)
        vis_path = viz.imwrite_unicode(CFG.vis_dir / f"{result_id}_annotated.jpg", annotated)
        charts = viz.make_stats_charts(boxes, img.shape, CFG.vis_dir, result_id)

        rec = {**{k: result[k] for k in
                  ("result_id", "session_id", "image_path", "image_hash",
                   "summary")},
               "conf": conf, "iou": iou, "num_objects": len(boxes),
               "vis_path": str(vis_path)}
        memory.save_detection(rec)
        result["charts"] = {k: str(v) for k, v in charts.items()}
        result["vis_path"] = str(vis_path)
        result["result_json"] = str(result_json)
        return result


def _count_by(boxes: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for b in boxes:
        out[str(b[key])] = out.get(str(b[key]), 0) + 1
    return out
