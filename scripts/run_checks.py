"""验收自检脚本：逐项检查环境、数据集、模型、记忆与报告链路（prd.md §12）。
用法：python scripts/run_checks.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import memory  # noqa: E402
from agent.config import CFG, ensure_dirs  # noqa: E402
from agent.llm import is_configured  # noqa: E402

PASS, FAIL, WARN = "[通过]", "[失败]", "[警告]"
results: list[tuple[str, str, bool]] = []


def check(name: str, fn, warn_only=False):
    try:
        detail = fn()
        results.append((PASS, name, True))
        print(f"{PASS} {name}" + (f" —— {detail}" if detail else ""))
    except Exception as e:
        if warn_only:
            results.append((WARN, name, False))
            print(f"{WARN} {name} —— {e}")
        else:
            results.append((FAIL, name, False))
            print(f"{FAIL} {name} —— {e}")


def c_dirs():
    ensure_dirs()
    return "目录齐备"


def c_dataset():
    for s in ("train", "valid", "test"):
        p = CFG.dataset_root / s / "_annotations.coco.json"
        if not p.exists():
            raise FileNotFoundError(f"缺 {p}")
    return str(CFG.dataset_root)


def c_profile():
    if not CFG.profile_path.exists():
        from scripts.make_dataset_profile import build_and_save
        build_and_save()
        return "画像缺失，已现场生成"
    return "画像存在"


def c_samples():
    n = len(list(CFG.samples_dir.glob("*.jpg")))
    if n == 0:
        raise FileNotFoundError("无示例图，运行 scripts/make_dataset_profile.py")
    return f"{n} 张示例图"


def c_weights():
    if CFG.default_weights.exists():
        return f"SAR 权重：{CFG.default_weights.name}"
    return f"未找到 {CFG.default_weights}，将退回 {CFG.fallback_weights.name}（管线演示模式）"


def c_torch():
    import torch
    if not torch.cuda.is_available() and CFG.device == "cuda":
        raise RuntimeError("device=cuda 但 CUDA 不可用")
    return f"torch {torch.__version__}（device={CFG.device}）"


def c_detect():
    from tools.detector import ShipDetector
    samples = sorted(CFG.samples_dir.glob("*.jpg"))
    if not samples:
        raise FileNotFoundError("无示例图可测")
    det = ShipDetector.get()
    r = det.detect(samples[0], session_id="run_checks")
    if r["result_json"] and Path(r["result_json"]).exists() and Path(r["vis_path"]).exists():
        return (f"检出 {r['summary']['num_objects']} 目标 / {r['summary']['elapsed_sec']}s "
                f"[{r['weights_source']}]")
    raise RuntimeError("产物落盘不完整")


def c_memory():
    memory.save_fact("_check", "ok")
    assert memory.load_facts().get("_check") == "ok"
    memory.save_message("run_checks", "user", "test")
    assert memory.recent_messages("run_checks", 1)[-1]["content"] == "test"
    recs = memory.list_detections(limit=1)
    return f"读写正常；检测档案 {len(recs) if recs else 0} 条（最近）"


def c_report():
    recs = memory.list_detections(limit=1)
    if not recs:
        raise FileNotFoundError("无检测记录，先跑检测")
    from tools import report as rm
    r = rm.generate(recs[0]["result_id"], fmt="markdown")
    if not r.get("ok"):
        raise RuntimeError(r.get("error"))
    return Path(r["path"]).name


def c_llm():
    if not is_configured():
        raise RuntimeError("未配置 ZHIPUAI_API_KEY（对话功能停用，检测不受影响）")
    from agent import llm
    r = llm.chat([{"role": "user", "content": "回复：连通"}])
    return f"模型 {CFG.llm_model} 返回 {len(r['content'] or '')} 字"


if __name__ == "__main__":
    print("=" * 60)
    check("目录与配置", c_dirs)
    check("数据集可访问", c_dataset, warn_only=True)  # 独立演示模式下允许缺数据集
    check("数据集画像", c_profile)
    check("内置示例图", c_samples)
    check("检测权重", c_weights, warn_only=True)
    check("torch / 设备", c_torch)
    check("YOLOv8 检测冒烟", c_detect)
    check("三层记忆读写", c_memory)
    check("报告生成(markdown)", c_report)
    check("GLM API 连通", c_llm, warn_only=True)
    print("=" * 60)
    failed = [n for s, n, ok in results if not ok]
    print(f"结果：{len(results) - len(failed)}/{len(results)} 项通过" +
          (f"；未通过：{', '.join(failed)}" if failed else ""))
    sys.exit(1 if any(not ok for s, n, ok in results if s == FAIL) else 0)
