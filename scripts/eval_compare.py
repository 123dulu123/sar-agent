"""Phase 2 实验脚本：在同一个 test 集上评估两个权重并输出 mAP 对比。

用法：python scripts/eval_compare.py
前提：scripts/train_quick.py 已完成（weights/yolo_dataset/data.yaml 存在），
      待对比权重位于 weights/ 下（默认对比 best.pt 与 best_trained.pt）。
产出：控制台对比表 + docs/experiments.md + PR 曲线图路径。
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import CFG  # noqa: E402


def evaluate(weights: Path, data_yaml: Path) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(weights))
    print(f"\n=== 评估 {weights.name} ===", flush=True)
    r = m.val(data=str(data_yaml), split="test", imgsz=640, device=0
              if _cuda() else "cpu", verbose=False)
    box = r.box
    return {"weights": str(weights),
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4)}


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(CFG.weights_dir / "best_demo.pt"))
    ap.add_argument("--new", default=str(CFG.weights_dir / "best.pt"))
    args = ap.parse_args()

    data_yaml = CFG.weights_dir / "yolo_dataset" / "data.yaml"
    if not data_yaml.exists():
        sys.exit("缺少 weights/yolo_dataset/data.yaml —— 请先运行 scripts/train_quick.py")

    rows = []
    for w in (args.old, args.new):
        if not Path(w).exists():
            print(f"[跳过] 权重不存在：{w}")
            continue
        rows.append(evaluate(Path(w), data_yaml))

    if len(rows) < 2:
        sys.exit("有效权重不足两个，无法对比")

    lines = ["# Phase 2 训练实验记录", "",
             f"评估集：HRSID_OPENSSDD v2 test（672 图 / 2014 框），imgsz=640",
             "", "| 权重 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {Path(r['weights']).name} | {r['mAP50']} | {r['mAP50_95']} "
                     f"| {r['precision']} | {r['recall']} |")
    d50 = round(rows[-1]["mAP50"] - rows[0]["mAP50"], 4)
    d5095 = round(rows[-1]["mAP50_95"] - rows[0]["mAP50_95"], 4)
    lines += ["", f"- mAP@0.5 变化：{d50:+.4f}；mAP@0.5:0.95 变化：{d5095:+.4f}",
              f"- 说明：demo 权重来源为开源社区（训练配置未知）；trained 权重为 "
              f"YOLOv8n 在本数据集 train 子集训练所得，二者仅作工程闭环验证。"]

    out = ROOT / "docs" / "experiments.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
