"""生成数据集画像：统计三个子集的图像/标注规模、类别、bbox 尺寸分布，
并从 test 集挑选目标最密集的图像复制为内置示例图。
产出：data/dataset_profile.yaml（供 Agent 工具 get_dataset_profile 使用）。
"""
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import CFG, ensure_dirs  # noqa: E402


def _subset_stats(split_dir: Path) -> dict:
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"缺少标注文件：{ann_file}")
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    areas = sorted(a["bbox"][2] * a["bbox"][3] for a in data["annotations"])
    img_sizes = Counter((im["width"], im["height"]) for im in data["images"])
    per_img = Counter(a["image_id"] for a in data["annotations"])
    empty = len(data["images"]) - len(per_img)
    return {
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": [c["name"] for c in data["categories"]],
        "image_sizes": [f"{w}x{h} x{n}" for (w, h), n in img_sizes.most_common(3)],
        "images_without_targets": empty,
        "bbox_area_px2": {
            "median": round(statistics.median(areas)) if areas else 0,
            "p90": round(areas[int(len(areas) * 0.9)]) if areas else 0,
            "max": round(areas[-1]) if areas else 0,
        },
        "max_targets_in_one_image": max(per_img.values()) if per_img else 0,
    }


def build_profile() -> dict:
    root = CFG.dataset_root
    profile = {"dataset_root": str(root),
               "name": "HRSID_OPENSSDD v2（Roboflow 导出，COCO for MMDetection）",
               "note": "本地目录名为 SARscope，实际内容为 HRSID 与 SSDD/OpenSSDD "
                       "合并的舰船检测数据集；图像已预处理为 640x640，无增强。"}
    total_img = total_ann = 0
    for split in ("train", "valid", "test"):
        s = _subset_stats(root / split)
        profile[split] = s
        total_img += s["images"]
        total_ann += s["annotations"]
    profile["total"] = {"images": total_img, "annotations": total_ann}
    return profile


def pick_samples(n: int = 8) -> list[str]:
    """从 test 集选标注框最多的 n 张图作为内置示例（密集场景演示效果最好）。"""
    test_dir = CFG.dataset_root / "test"
    data = json.loads((test_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    per_img = Counter(a["image_id"] for a in data["annotations"])
    id2name = {im["id"]: im["file_name"] for im in data["images"]}
    ranked = sorted(per_img.items(), key=lambda kv: -kv[1])[:n]
    CFG.samples_dir.mkdir(parents=True, exist_ok=True)
    picked = []
    for img_id, cnt in ranked:
        src = test_dir / id2name[img_id]
        dst = CFG.samples_dir / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
        picked.append({"file": dst.name, "targets": cnt})
    return picked


def build_and_save() -> dict:
    profile = build_profile()
    profile["samples"] = pick_samples()
    CFG.profile_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    CFG.profile_path.write_text(yaml.safe_dump(profile, allow_unicode=True,
                                               sort_keys=False), encoding="utf-8")
    return profile


def main() -> None:
    ensure_dirs()
    p = build_and_save()
    print(f"画像已写入 {CFG.profile_path}")
    for k in ("name", "total", "train", "valid", "test"):
        print(f"{k}: {p.get(k)}")
    print("示例图:", [s["file"] for s in p.get("samples", [])])


if __name__ == "__main__":
    main()
