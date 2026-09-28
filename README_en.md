# SAR-Agent

[中文](README.md) | **English**

SAR image ship detection and intelligent analysis powered by **YOLOv8** (perception) + **GLM-5.3-Flash** (cognition, native Function-Calling Agent), delivered as a Gradio web application.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c) ![Gradio](https://img.shields.io/badge/Gradio-6.x-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

| Module | Description |
|---|---|
| Detection Workbench | Upload or pick a built-in sample → YOLOv8 detection → annotated image, per-target table, confidence histogram, spatial distribution; thresholds adjustable live |
| Batch & Compare | Batch detection of multiple images; diff two detections (IoU-matched added/removed targets) |
| Agent Chat | Drive the whole pipeline in natural language: detect, interpret, compare, report, remember preferences — native Function Calling with a hand-rolled ReAct loop and 10 tools |
| Document Q&A | Import SAR reference material (txt/md/docx/pdf); the Agent retrieves and cites passages via FTS5 (lightweight RAG) |
| History & Reports | Browse past detections; export Markdown / PDF / Word reports |
| Long-term Memory | Three-tier memory (session buffer / long-term facts / detection archive) on SQLite, survives restarts |

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

copy configs\.env.example configs\.env # fill in ZHIPUAI_API_KEY (chat only; detection works without it)

python scripts\run_checks.py           # environment self-check
python app.py                          # http://127.0.0.1:7860
```

The repo is **self-contained**: demo weights, sample images and the dataset profile are bundled, so everything works without a local dataset. Plug in `HRSID_OPENSSDD v2` (path in `configs/settings.yaml`) to unlock training and full dataset profiling.

### GPU Training (Phase 2)

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python scripts\train_quick.py --full --epochs 40 --batch 8 --device 0   # auto COCO->YOLO conversion
python scripts\eval_compare.py                                          # mAP comparison vs demo weights
```

### A note on torch versions

`requirements.txt` pins torch **2.5.1**: the 2.8/2.9 Windows wheels hit `WinError 1114` (c10.dll initialization failure) on the development machine; verify `python -c "import torch"` before upgrading.

## Architecture

```
Gradio UI (six tabs)
        │
Agent layer (ReAct-style loop + system prompt + tool registry + three-tier memory)
        │  GLM-5.3-Flash native Function Calling
        ▼
Tools (10: detect / detail / list-images / report / compare / history / memory×2 / dataset-profile / doc-search)
        ▼
Data (YOLOv8 weights / dataset / SQLite memory.db + docs.db / outputs)
```

- Design docs: [prd.md](prd.md) (product & engineering plan, in Chinese) and [agent.md](agent.md) (Agent design: prompts, tool protocol, memory, measured evaluation).
- Dataset: HRSID_OPENSSDD v2 — 6,735 images (640×640), 19,435 annotations, single class `ship`, small-target-heavy (median bbox area ≈ 600 px²).
- Demo weight: `weights/best.pt` from [sethubolt7/YOLOV8_SAR_SHIP_DETECTION](https://github.com/sethubolt7/YOLOV8_SAR_SHIP_DETECTION).

## Testing & Quality

```bash
pytest tests/                  # 11 unit tests
ruff check .                   # lint
python scripts/run_checks.py   # 10-item environment self-check
python scripts/agent_suite.py  # 20-instruction Agent regression (needs API key, bar ≥90%)
```

CI runs lint + unit tests + a UI build smoke test via GitHub Actions.

## License & Acknowledgements

Code released under [MIT](LICENSE). Bundled sample images come from the public HRSID/SSDD academic datasets (for demonstration only); the demo weight originates from the open-source project above. For academic and civil use only.
