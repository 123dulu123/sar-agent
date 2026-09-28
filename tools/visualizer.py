"""可视化：检测框绘制、置信度直方图、目标位置分布图；含中文路径安全的图像读写。"""
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 每个类别一个稳定颜色（BGR）
_PALETTE = [(60, 200, 60), (80, 80, 230), (230, 160, 40), (200, 60, 180),
            (240, 240, 80), (90, 200, 200)]


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """cv2.imread 不支持 Windows 中文路径，用 fromfile+imdecode 兜底。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_unicode(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise OSError(f"图像编码失败：{path}")
    buf.tofile(str(path))
    return path


def _color(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def draw_boxes(img_bgr: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """在图上画框与标签。boxes 元素：{cls_id, cls_name, conf, xyxy}。"""
    canvas = img_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
        color = _color(b.get("cls_id", 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{b.get('cls_name', 'obj')} {b.get('conf', 0):.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(canvas, (x1, y_text - th - bl), (x1 + tw + 2, y_text + bl),
                      color, -1)
        cv2.putText(canvas, label, (x1 + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def confidence_histogram(boxes: list[dict], save_path: str | Path) -> Path:
    confs = [b["conf"] for b in boxes] or [0]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=120)
    ax.hist(confs, bins=10, range=(0, 1), color="#3a7bd5", edgecolor="white")
    ax.set_title("置信度分布")
    ax.set_xlabel("confidence")
    ax.set_ylabel("目标数")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return Path(save_path)


def position_scatter(boxes: list[dict], image_shape: tuple[int, int],
                     save_path: str | Path) -> Path:
    """目标中心位置分布（点面积∝框面积），反映空间聚集特征。"""
    h, w = image_shape[:2]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=120)
    if boxes:
        xs = [(b["xyxy"][0] + b["xyxy"][2]) / 2 for b in boxes]
        ys = [(b["xyxy"][1] + b["xyxy"][3]) / 2 for b in boxes]
        areas = [max((b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]), 4)
                 for b in boxes]
        ax.scatter(xs, ys, s=[min(a * 0.6, 300) for a in areas],
                   c="#3a7bd5", alpha=0.65, edgecolors="white", linewidths=0.5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # 图像坐标系：y 向下
    ax.set_aspect("equal")
    ax.set_title("目标中心位置分布（点面积∝框面积）")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return Path(save_path)


def make_stats_charts(boxes: list[dict], image_shape: tuple[int, int],
                      out_dir: str | Path, stem: str) -> dict:
    """一次生成两张统计图，返回 {"hist": path, "position": path}。"""
    out_dir = Path(out_dir)
    return {
        "hist": str(confidence_histogram(boxes, out_dir / f"{stem}_conf_hist.png")),
        "position": str(position_scatter(boxes, image_shape, out_dir / f"{stem}_position.png")),
    }
