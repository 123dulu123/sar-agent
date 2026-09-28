"""全局配置：加载 configs/settings.yaml 与 configs/.env，暴露 CFG 单例。"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "configs" / ".env")

with open(ROOT / "configs" / "settings.yaml", encoding="utf-8") as f:
    _raw = yaml.safe_load(f)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


class _Cfg:
    root = ROOT
    dataset_root = Path(_raw["paths"]["dataset_root"])
    weights_dir = _resolve(_raw["paths"]["weights_dir"])
    outputs_dir = _resolve(_raw["paths"]["outputs_dir"])
    uploads_dir = _resolve(_raw["paths"]["uploads_dir"])
    store_dir = _resolve(_raw["paths"]["store_dir"])
    samples_dir = _resolve(_raw["paths"]["samples_dir"])
    profile_path = _resolve(_raw["paths"]["profile_path"])

    default_weights = _resolve(_raw["weights"]["default"])
    fallback_weights = _resolve(_raw["weights"]["fallback"])
    device = _raw["weights"]["device"]

    conf = float(_raw["detect"]["conf"])
    iou = float(_raw["detect"]["iou"])
    imgsz = int(_raw["detect"]["imgsz"])
    max_side = int(_raw["detect"]["max_side"])

    llm_base_url = os.getenv("LLM_BASE_URL", _raw["llm"]["base_url"])
    llm_model = os.getenv("LLM_MODEL", _raw["llm"]["model"])
    llm_api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
    llm_temperature = float(_raw["llm"]["temperature"])
    llm_max_steps = int(_raw["llm"]["max_steps"])
    llm_history_window = int(_raw["llm"]["history_window"])
    llm_timeout = int(_raw["llm"]["timeout"])
    llm_max_retries = int(_raw["llm"]["max_retries"])

    report_formats = list(_raw["report"]["formats"])

    server_host = os.getenv("SERVER_HOST", _raw["server"]["host"])
    server_port = int(os.getenv("SERVER_PORT", str(_raw["server"]["port"])))

    results_dir = outputs_dir / "results"    # 检测结果 JSON
    vis_dir = outputs_dir / "vis"            # 标注图与统计图
    reports_dir = outputs_dir / "reports"    # 报告产物


def ensure_dirs() -> None:
    dirs = (_Cfg.weights_dir, _Cfg.outputs_dir, _Cfg.uploads_dir, _Cfg.store_dir,
            _Cfg.samples_dir, _Cfg.results_dir, _Cfg.vis_dir, _Cfg.reports_dir,
            _Cfg.profile_path.parent)
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


CFG = _Cfg()
