# SAR-Agent

**中文** | [English](README_en.md)

基于 **GLM-5.3-Flash**（原生 Function Calling Agent）+ **YOLOv8** + **Gradio** 的 SAR 图像舰船检测与智能分析系统。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c) ![Gradio](https://img.shields.io/badge/Gradio-6.x-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## 功能总览

| 模块 | 说明 |
|---|---|
| 检测工作台 | 上传/示例图 → YOLOv8 检测 → 标注图、明细表、置信度直方图、位置分布图，阈值实时可调 |
| 批量与对比 | 多图批量检测；两次检测差分对比（IoU 匹配新增/消失目标） |
| 智能对话 | 自然语言指挥 Agent：检测、解读、对比、报告、记忆偏好（GLM 原生 Function Calling + 自研 ReAct 循环，10 个工具） |
| 文档问答 | 导入 SAR 判读资料（txt/md/docx/pdf），Agent 检索引用原文回答领域问题（FTS5 RAG-lite） |
| 历史与报告 | 检测档案回看；Markdown / PDF / Word 三格式报告下载 |
| 长期记忆 | SQLite 三层记忆（会话缓冲 / 长期事实 / 检测档案），重启不丢 |

## 快速开始

```bash
# 1) 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2) 配置 GLM API Key（不配置也能用检测工作台，仅 Agent 对话停用）
copy configs\.env.example configs\.env    # 然后填入 ZHIPUAI_API_KEY

# 3) 环境自检
python scripts\run_checks.py

# 4) 启动
python app.py                     # http://127.0.0.1:7860
```

项目**开箱即用**：演示权重、示例图、数据集画像均已内置（见 `weights/`、`data/`），不依赖本地数据集即可完整体验；接入 `G:\数据集\SARscope\SARscope`（HRSID_OPENSSDD v2）后可解锁训练与全量画像。

### GPU 训练（Phase 2）

```bash
# 1) 安装 CUDA 版 torch（按驱动选择 cu121/cu124）
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2) 全量训练（COCO->YOLO 自动转换；4GB 显存建议 batch 6~8）
python scripts\train_quick.py --full --epochs 40 --batch 8 --device 0

# 3) 与演示权重在同一 test 集上做 mAP 对比
python scripts\eval_compare.py
```

### 关于 torch 版本

`requirements.txt` 将 torch 锁定在 **2.5.1**：2.8/2.9 的 Windows 轮子在本机（Win11 + 腾讯电脑管家实时防护）触发 `WinError 1114`（c10.dll 初始化失败），2.5.1 正常。升级 torch 前请先验证 `python -c "import torch"`。

## 架构

```
Gradio UI（六个 Tab：工作台/批量对比/对话/文档问答/历史报告/关于）
        │
Agent 编排层（自研 ReAct 循环 + 系统提示词 + 工具注册表 + 三层记忆）
        │  GLM-5.3-Flash 原生 Function Calling
        ▼
工具层（10 个工具：检测 / 明细 / 列图 / 报告 / 对比 / 历史 / 记忆×2 / 画像 / 资料检索）
        ▼
数据层（YOLOv8 权重 / 数据集 / SQLite memory.db+docs.db / outputs 产物）
```

- 设计与实现依据：[prd.md](prd.md)（产品需求与工程方案）、[agent.md](agent.md)（Agent 详细设计：提示词/工具协议/记忆/评估实测）
- 演示话术：[docs/demo_script.md](docs/demo_script.md)；面试复习：[docs/interview_notes.md](docs/interview_notes.md)；简历表述：[docs/resume.md](docs/resume.md)

## 数据集与权重

- **数据集**：HRSID_OPENSSDD v2（HRSID 与 SSDD/OpenSARShip 合并舰船检测数据集，Roboflow 导出，COCO for MMDetection 格式）：6735 张 640×640 图像（train 4717 / valid 1346 / test 672）、19435 个标注框、单类别 `ship`、典型小目标（框面积中位数约 600 px²）。画像见 `data/dataset_profile.yaml`。
- **演示权重**：`weights/best.pt`（6.2MB，YOLOv8n 规模）来自开源项目 [sethubolt7/YOLOV8_SAR_SHIP_DETECTION](https://github.com/sethubolt7/YOLOV8_SAR_SHIP_DETECTION)，CPU 单图推理约 2~3s。
- **自训练**：`scripts/train_quick.py` 自动完成 COCO→YOLO 标注转换与训练；`scripts/eval_compare.py` 输出与演示权重的 mAP 对比。

## 测试与质量

```bash
python -m pytest tests/          # 11 项单元测试
ruff check .                     # 代码检查（配置见 pyproject.toml）
python scripts/run_checks.py     # 运行环境自检（10 项）
python scripts/agent_suite.py    # Agent 20 条指令回归（需 API Key，验收线 ≥90%）
```

CI：GitHub Actions 自动执行 lint + 单测 + 界面构建冒烟（见 `.github/workflows/ci.yml`）。

## 许可与致谢

代码以 [MIT](LICENSE) 发布。示例图取自 HRSID/SSDD 公开学术数据集（仅作演示，版权归原作者）；演示权重来自上述开源项目。系统仅用于学术与民用场景分析。
