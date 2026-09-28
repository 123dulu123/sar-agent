"""Phase 2 快速训练脚本（有 GPU 后使用）：
1) COCO(MMDetection) 标注 -> YOLO txt 转换；2) 生成 Ultralytics data yaml；
3) 训练 YOLOv8（默认按 RTX 3050 4GB 配置）。

用法示例：
  python scripts/train_quick.py --subset 800 --epochs 30 --imgsz 640 --batch 8
  python scripts/train_quick.py --full --epochs 60 --device 0
产物：weights/train/…/best.pt（训练后复制为 weights/best.pt 即可接入系统）
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import CFG  # noqa: E402


def coco_to_yolo(split: str, out_root: Path) -> int:
    """转换一个子集；返回转换的标注框数。"""
    import shutil
    src = CFG.dataset_root / split
    data = json.loads((src / "_annotations.coco.json").read_text(encoding="utf-8"))
    labels_dir = out_root / split / "labels"
    (out_root / split / "images").mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    id2img = {im["id"]: im for im in data["images"]}
    for im in data["images"]:  # 复制图像（Windows 下不用软链）
        img_dst = out_root / split / "images" / im["file_name"]
        if not img_dst.exists():
            shutil.copyfile(src / im["file_name"], img_dst)

    stems = {im_id: Path(im["file_name"]).stem for im_id, im in id2img.items()}
    count = 0
    for a in data["annotations"]:
        im = id2img[a["image_id"]]
        x, y, w, h = a["bbox"]
        cx, cy = (x + w / 2) / im["width"], (y + h / 2) / im["height"]
        nw, nh = w / im["width"], h / im["height"]
        with open(labels_dir / f"{stems[a['image_id']]}.txt", "a", encoding="utf-8") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        count += 1
    return count


def build_data_yaml(out_root: Path, n_classes: int = 1) -> Path:
    import yaml
    cfg = {"path": str(out_root),
           "train": "train/images", "val": "valid/images", "test": "test/images",
           "names": {0: "ship"}}
    p = out_root / "data.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser(description="Phase 2: YOLOv8 快速训练（含数据转换）")
    ap.add_argument("--out", default=str(ROOT / "weights" / "yolo_dataset"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="GPU 编号；CPU 传 cpu（很慢）")
    ap.add_argument("--subset", type=int, default=0,
                    help="只用 train 前 N 张图快速验证管线；0=全量")
    ap.add_argument("--full", action="store_true", help="全量训练（忽略 --subset）")
    args = ap.parse_args()

    out_root = Path(args.out)
    if not (out_root / "data.yaml").exists():
        print("转换数据集（仅首次）…")
        for split in ("train", "valid", "test"):
            n = coco_to_yolo(split, out_root)
            print(f"  {split}: {n} 框")
        if args.subset and not args.full:  # 子集模式：截断 train 图像与标签
            imgs = sorted((out_root / "train" / "images").glob("*.jpg"))
            for p in imgs[args.subset:]:
                p.unlink()
                (out_root / "train" / "labels" / f"{p.stem}.txt").unlink(missing_ok=True)
            print(f"  子集模式：train 截断为 {args.subset} 张")
    data_yaml = build_data_yaml(out_root)
    print(f"data yaml: {data_yaml}")

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(ROOT / "weights"),
                name="train", exist_ok=True)
    print("完成。将 weights/train/weights/best.pt（看控制台输出路径）复制为 weights/best.pt "
          "后重启应用即接入系统。")


if __name__ == "__main__":
    main()
