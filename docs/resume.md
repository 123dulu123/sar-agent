# 简历项目表述（中英双语）

> 按实际完成度取舍要点；所有数字均可复现（验收脚本与实验记录见仓库）。
> 面试追问预案见 docs/interview_notes.md。

## 中文版

**SAR-Agent：基于 GLM 大模型 Function Calling 的 SAR 图像舰船检测与智能分析系统**
个人开源项目 ｜ 技术栈：Python、YOLOv8（Ultralytics）、GLM-5.3-Flash Function Calling、Gradio、SQLite、reportlab/python-docx

**一句话**：面向 SAR 图像的"检测—解读—对比—报告"智能分析系统，用 YOLOv8 做感知、GLM-5.3-Flash Agent 做认知，自然语言驱动全流程，含三层长期记忆与多格式自动报告。

**要点**：

- 独立完成系统设计与全栈实现：Gradio 六模块界面 + YOLOv8 检测推理（CPU 单图 P95 ≤ 5s）+ SQLite 三层长期记忆（会话缓冲/长期事实/检测档案）+ Markdown/PDF/Word 三格式报告引擎；
- 基于 GLM-5.3-Flash 原生 Function Calling 自研 ReAct 风格 Agent 循环（工具调用上限、参数防御、失败自纠、兜底总结），设计 10 个领域工具，20 条脚本化指令回归任务完成率 90%（三轮提示词迭代 80%→85%→90%）；
- 设计"检测结果结构化 JSON 作为 LLM 唯一事实来源"的解耦架构抑制幻觉，配合数字一致率抽检（100%）与越界路径白名单拦截；
- 在 HRSID+SSDD 合并数据集（6735 图 / 19435 框，单类 ship）上完成数据画像、小目标特性分析与工程化接入；使用自建 COCO→YOLO 转换脚本在 RTX 3050 上完成 YOLOv8n 训练，与开源演示权重同 test 集对比 mAP@0.5 ______（见 docs/experiments.md）；
- 工程化配套：pytest 单测 11 项、ruff 检查全绿、GitHub Actions CI、10 项运行自检、环境踩坑记录（Windows DLL 加载失败定位至 torch 版本冲突并锁定修复）。

**技能关键词**：LLM Agent / Function Calling / 提示词工程 / RAG（FTS5 检索）/ 目标检测（YOLOv8）/ 小目标 / Gradio / SQLite / pytest / CI

## English Version

**SAR-Agent: SAR Ship Detection & Intelligent Analysis with a GLM Function-Calling Agent**
Personal open-source project | Stack: Python, YOLOv8 (Ultralytics), GLM-5.3-Flash Function Calling, Gradio, SQLite, reportlab/python-docx

**Summary**: An end-to-end SAR image analysis system that chains detection → interpretation → comparison → reporting, driven by natural language, with three-tier long-term memory and multi-format automated reports.

- Designed and built the full stack solo: a six-module Gradio UI, YOLOv8 inference (P95 ≤ 5s/image on CPU), three-tier long-term memory on SQLite (session buffer / long-term facts / detection archive), and a Markdown/PDF/Word report engine;
- Implemented a hand-rolled ReAct-style agent loop on GLM-5.3-Flash native Function Calling (step cap, argument validation, failure self-correction, fallback summarization) with 10 domain tools; 20-case scripted regression achieves a 90% task completion rate across three prompt iterations (80% → 85% → 90%);
- Suppressed model hallucination by making structured detection JSON the single source of truth for the LLM, with numeric-consistency audits (100%) and a path whitelist against tool abuse;
- Profiled the HRSID+SSDD merged dataset (6,735 images / 19,435 boxes, class `ship`), analyzed small-target statistics, fine-tuned YOLOv8n on an RTX 3050 via a custom COCO→YOLO converter, and benchmarked against the open demo weights on the same test split (mAP@0.5 ______, see docs/experiments.md);
- Engineering hygiene: 11 pytest units, ruff-clean, GitHub Actions CI, a 10-item runtime self-check, and a documented deep-dive debugging story (Windows DLL load failure isolated to a torch-version conflict).

**Keywords**: LLM Agent / Function Calling / Prompt Engineering / RAG (FTS5 retrieval) / Object Detection (YOLOv8) / Small Objects / Gradio / SQLite / pytest / CI

## 面试叙述框架（STAR）

- **S/T（情境任务）**：SAR 判读人工效率低、检测模型只给框不给解读 → 做一个"自然语言驱动的检测-解读-报告"闭环系统；
- **A（行动）**：检测与认知解耦（LLM 不看原图）；自研 Agent 循环而非 LangChain（可控性+面试可讲清每一层）；三层记忆解决"刚才那张图"指代与偏好持久化；用回归套件驱动提示词迭代；
- **R（结果）**：上述量化指标 + 一个可开源、可一键运行的仓库。
