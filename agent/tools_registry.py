"""工具注册表：Function Calling schema（agent.md §4）与 Python 实现的映射。

dispatch() 是 Agent 调工具的唯一入口：参数校验 → 执行 → 统一 JSON 返回。
所有工具失败都返回 {"ok": false, "error": ...}，由模型自纠或如实转述用户。
"""
import json
import logging
import re
from pathlib import Path

from agent import memory
from agent.config import CFG
from tools import report as report_mod
from tools.detector import IMAGE_EXTS, ShipDetector, is_image_path

log = logging.getLogger(__name__)

TOOLS = [
    {"type": "function", "function": {
        "name": "detect_ships",
        "description": "对指定 SAR 图像执行 YOLOv8 舰船检测。返回 result_id 与结构化统计"
                       "（目标数、逐框坐标、置信度）。检测前必须确认路径有效。",
        "parameters": {"type": "object", "properties": {
            "image_path": {"type": "string", "description": "图像路径或 uploads 中的文件名"},
            "conf_threshold": {"type": "number", "description": "置信度阈值，默认 0.25；用户有偏好时优先用偏好值"},
            "iou_threshold": {"type": "number", "description": "NMS IoU 阈值，默认 0.45"}},
            "required": ["image_path"]}}},
    {"type": "function", "function": {
        "name": "get_detection_detail",
        "description": "查询某次检测结果的详细信息：逐目标明细（类别、置信度、bbox）、统计汇总、可视化图路径。",
        "parameters": {"type": "object", "properties": {
            "result_id": {"type": "string", "description": "检测结果 ID"},
            "top_k": {"type": "integer", "description": "只返回置信度前 K 个目标（可选）"}},
            "required": ["result_id"]}}},
    {"type": "function", "function": {
        "name": "list_available_images",
        "description": "列出当前可分析的图像：用户本次会话上传的图像与系统内置示例图，含路径与尺寸。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "generate_report",
        "description": "依据某次检测结果生成分析报告文件（Markdown/PDF/Word），包含统计表、"
                       "可视化图和文字解读，返回可下载路径。",
        "parameters": {"type": "object", "properties": {
            "result_id": {"type": "string", "description": "检测结果 ID"},
            "format": {"type": "string", "enum": ["markdown", "pdf", "docx"],
                       "description": "报告格式，默认 markdown；用户有格式偏好时优先用偏好值"},
            "interpretation": {"type": "string",
                               "description": "（可选）你撰写的文字解读，将写入报告；省略则用内置兜底解读"}},
            "required": ["result_id"]}}},
    {"type": "function", "function": {
        "name": "query_history",
        "description": "查询历史：历次检测档案（图像名、时间、目标数）和/或历史对话摘要，支持按图像名关键词过滤。",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "图像名关键词，可省略"},
            "limit": {"type": "integer", "description": "最多返回条数，默认 10"}}}}},
    {"type": "function", "function": {
        "name": "save_memory",
        "description": "把用户的长期偏好或约定写入持久记忆（如置信度阈值习惯、报告语言、关注区域）。",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "偏好名，如 conf_threshold / report_format / language"},
            "value": {"type": "string", "description": "偏好值"}},
            "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "recall_memory",
        "description": "读取持久记忆中的偏好与事实；key 省略时返回全部。",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "get_dataset_profile",
        "description": "返回数据集画像：HRSID+SSDD 合并舰船数据集的子集规模、类别、图像尺寸与小目标特性统计。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "compare_detections",
        "description": "对比两次检测结果：目标数/置信度差分，以及按 IoU 匹配的新增与消失目标明细。"
                       "baseline 为基准，current 为新状态。",
        "parameters": {"type": "object", "properties": {
            "result_id_a": {"type": "string", "description": "基准检测结果 ID"},
            "result_id_b": {"type": "string", "description": "待比较检测结果 ID"},
            "iou_match": {"type": "number", "description": "同一目标的判定 IoU 阈值，默认 0.5"}},
            "required": ["result_id_a", "result_id_b"]}}},
    {"type": "function", "function": {
        "name": "search_reference_docs",
        "description": "在已导入的 SAR 参考资料（判读手册、术语表等）中检索相关段落，"
                       "用于回答领域知识类问题时引用原文依据。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "检索词，如'虚警''锚地特征'"},
            "k": {"type": "integer", "description": "返回段落数，默认 5"}},
            "required": ["query"]}}},
]

_FACT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


# ---------- 参数校验 ----------

def _validate_image_path(raw: str) -> Path:
    if not raw or not isinstance(raw, str):
        raise ValueError("image_path 不能为空")
    p = Path(raw)
    if not p.is_absolute():
        p = CFG.uploads_dir / p.name if (CFG.uploads_dir / p.name).exists() \
            else CFG.samples_dir / p.name
    p = p.resolve()
    allowed = [CFG.uploads_dir.resolve(), CFG.samples_dir.resolve(),
               CFG.dataset_root / "test", CFG.dataset_root / "train",
               CFG.dataset_root / "valid"]
    if not any(str(p).startswith(str(a.resolve())) for a in allowed):
        raise PermissionError("路径不在允许目录内（uploads / 示例图 / 数据集）")
    if not p.exists():
        raise FileNotFoundError(f"图像不存在：{p}")
    if not is_image_path(p):
        raise ValueError(f"不支持的文件类型：{p.suffix}（支持 {sorted(IMAGE_EXTS)}）")
    return p


def _clamp(x, lo, hi, default):
    try:
        return min(max(float(x), lo), hi)
    except (TypeError, ValueError):
        return default


# ---------- 工具实现 ----------

def _detect_ships(args: dict, session_id: str | None) -> dict:
    image_path = _validate_image_path(args["image_path"])
    facts = memory.load_facts()
    conf_default = float(facts["conf_threshold"]) if facts.get("conf_threshold") else CFG.conf
    conf = _clamp(args.get("conf_threshold"), 0.01, 0.95, conf_default)
    iou = _clamp(args.get("iou_threshold"), 0.1, 0.9, CFG.iou)
    r = ShipDetector.get().detect(image_path, conf=conf, iou=iou, session_id=session_id)
    r.pop("boxes", None)  # 上下文经济性：全量框在 get_detection_detail 里取
    return {"ok": True, **r}


def _get_detection_detail(args: dict, session_id: str | None) -> dict:
    rid = args.get("result_id")
    rec = memory.get_detection(rid) if rid else memory.latest_detection(session_id)
    if not rec:
        return {"ok": False, "error": f"检测结果不存在：{rid or '（会话内无检测记录）'}"}
    full_path = CFG.results_dir / f"{rec['result_id']}.json"
    detail = json.loads(full_path.read_text(encoding="utf-8")) if full_path.exists() else {}
    boxes = detail.get("boxes", [])
    try:
        top_k = int(args.get("top_k") or 0)
        if top_k > 0:
            boxes = boxes[:top_k]
    except (TypeError, ValueError):
        pass
    return {"ok": True, "result_id": rec["result_id"], "image_path": rec["image_path"],
            "summary": detail.get("summary", rec.get("summary", {})),
            "boxes": boxes,
            "vis_path": rec.get("vis_path"), "charts": detail.get("charts", {})}


def _list_available_images(args: dict, session_id: str | None) -> dict:
    def _scan(d: Path):
        return [{"path": str(p), "size_kb": round(p.stat().st_size / 1024)}
                for p in sorted(d.glob("*")) if is_image_path(p)][:50]
    return {"ok": True,
            "uploads": _scan(CFG.uploads_dir),
            "samples": _scan(CFG.samples_dir)}


def _generate_report(args: dict, session_id: str | None) -> dict:
    rid = args.get("result_id")
    if not rid:
        latest = memory.latest_detection(session_id)
        if not latest:
            return {"ok": False, "error": "没有可用的检测结果，请先执行检测"}
        rid = latest["result_id"]
    fmt = args.get("format") or memory.load_facts().get("report_format") or "markdown"
    r = report_mod.generate(rid, fmt=fmt, interpretation=args.get("interpretation"))
    return r


def _query_history(args: dict, session_id: str | None) -> dict:
    limit = min(int(args.get("limit") or 10), 50)
    recs = memory.list_detections(keyword=args.get("keyword"), limit=limit,
                                  session_id=session_id)
    return {"ok": True, "count": len(recs), "records": [{
        "result_id": r["result_id"], "image_path": r["image_path"],
        "created_at": r["created_at"], "num_objects": r["num_objects"],
        "conf": r["conf"], "report_paths": r.get("report_paths", {}),
    } for r in recs]}


def _save_memory(args: dict, session_id: str | None) -> dict:
    key = str(args.get("key", "")).strip()
    value = str(args.get("value", "")).strip()
    if not _FACT_KEY_RE.match(key):
        return {"ok": False, "error": "key 需为下划线/字母开头的短标识符，如 conf_threshold"}
    if not value or len(value) > 200:
        return {"ok": False, "error": "value 需为 1~200 字符"}
    memory.save_fact(key, value)
    return {"ok": True, "saved": {key: value}}


def _recall_memory(args: dict, session_id: str | None) -> dict:
    key = args.get("key")
    facts = memory.load_facts()
    if key:
        return {"ok": True, key: facts.get(key), "found": key in facts}
    return {"ok": True, "facts": facts}


def _get_dataset_profile(args: dict, session_id: str | None) -> dict:
    if CFG.profile_path.exists():
        import yaml
        return {"ok": True, **yaml.safe_load(CFG.profile_path.read_text(encoding="utf-8"))}
    try:  # 画像缺失时现场生成并缓存
        from scripts.make_dataset_profile import build_and_save
        return {"ok": True, **build_and_save()}
    except Exception as e:
        return {"ok": False, "error": f"数据集画像不可用（{e}），请运行 scripts/make_dataset_profile.py"}


def _compare_detections(args: dict, session_id: str | None) -> dict:
    from tools.compare import compare
    rid_a, rid_b = args.get("result_id_a"), args.get("result_id_b")
    if not rid_a or not rid_b:
        recs = memory.list_detections(limit=2, session_id=session_id)
        if len(recs) < 2:
            return {"ok": False, "error": "会话内可用检测结果不足两次，请提供两个 result_id"}
        rid_a = rid_a or recs[1]["result_id"]
        rid_b = rid_b or recs[0]["result_id"]
    iou = _clamp(args.get("iou_match"), 0.1, 0.95, 0.5)
    return compare(rid_a, rid_b, iou_match=iou)


def _search_reference_docs(args: dict, session_id: str | None) -> dict:
    from tools import docqa
    return docqa.search(args.get("query", ""), k=args.get("k") or 5)


_HANDLERS = {
    "detect_ships": _detect_ships,
    "get_detection_detail": _get_detection_detail,
    "list_available_images": _list_available_images,
    "generate_report": _generate_report,
    "query_history": _query_history,
    "save_memory": _save_memory,
    "recall_memory": _recall_memory,
    "get_dataset_profile": _get_dataset_profile,
    "compare_detections": _compare_detections,
    "search_reference_docs": _search_reference_docs,
}


def dispatch(name: str, args: dict, session_id: str | None = None) -> dict:
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"未知工具：{name}"}
    try:
        return handler(args or {}, session_id)
    except (ValueError, FileNotFoundError, PermissionError) as e:
        return {"ok": False, "error": str(e)}
    except Exception:
        log.exception("工具 %s 执行异常", name)
        return {"ok": False, "error": f"工具 {name} 执行异常，请稍后重试"}
