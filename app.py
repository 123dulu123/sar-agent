"""Gradio Web 界面：检测工作台 / 批量与对比 / 智能对话 / 文档问答 / 历史与报告 / 关于。

适配 Gradio 6.x（Chatbot 默认 messages 格式，更新用组件实例返回）。
启动：python app.py
"""
import logging
import os
import shutil
import uuid
from pathlib import Path

import gradio as gr

from agent import memory
from agent.config import CFG, ensure_dirs
from agent.core import run_agent
from agent.llm import is_configured
from agent.tools_registry import dispatch
from tools import docqa
from tools.detector import IMAGE_EXTS, ShipDetector

# 本机若有系统代理，须豁免 localhost，否则 Gradio 自检会经代理收到 502
_no_proxy = "127.0.0.1,localhost,::1"
for k in ("NO_PROXY", "no_proxy"):
    existing = os.environ.get(k, "")
    os.environ[k] = ",".join({*filter(None, existing.split(",")), *_no_proxy.split(",")})

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ensure_dirs()

_CUSTOM_CSS = """
.gradio-container {max-width: 1320px; margin: auto;}
#app-header {background: linear-gradient(135deg,#0b2545 0%,#13315c 55%,#1d4e89 100%);
             padding:18px 26px; border-radius:14px; margin-bottom:10px;}
#app-header h1 {color:#ffffff; margin:0; font-size:22px; letter-spacing:.5px;}
#app-header p {color:#c9ddff; margin:6px 0 0; font-size:13px;}
footer {display:none !important;}
"""


# ---------- 上传归档 ----------

def _stage_upload(image_path: str | None) -> str | None:
    """Gradio 把上传文件存在其临时目录（不在白名单内）——先归档进 outputs/uploads：
    ① 通过路径白名单；② Agent 可经 list_available_images 发现；③ 历史回看不失效。"""
    if not image_path:
        return None
    p = Path(image_path).resolve()
    inside = [CFG.uploads_dir.resolve(), CFG.samples_dir.resolve()]
    if any(str(p).startswith(str(a)) for a in inside):
        return str(p)
    if not p.exists() or p.suffix.lower() not in IMAGE_EXTS:
        return str(p)  # 非法文件交给工具层给出具体报错
    CFG.uploads_dir.mkdir(parents=True, exist_ok=True)
    dst = CFG.uploads_dir / f"{p.stem[:40]}_{uuid.uuid4().hex[:6]}{p.suffix.lower()}"
    shutil.copyfile(p, dst)
    return str(dst)


# ---------- Tab1 检测工作台 ----------

def _detect_and_render(image_path, conf, iou, session_id):
    session_id = session_id or uuid.uuid4().hex
    image_path = _stage_upload(image_path)
    if not image_path:
        return None, None, None, None, "请先上传图像或选择示例图", "", session_id
    r = dispatch("detect_ships", {"image_path": image_path,
                                  "conf_threshold": conf, "iou_threshold": iou},
                 session_id)
    if not r.get("ok"):
        return (None, None, None, None, f"检测失败：{r.get('error')}", "", session_id)
    rows = [[i + 1, b["cls_name"], b["conf"], ", ".join(str(v) for v in b["xyxy"])]
            for i, b in enumerate(_full_boxes(r["result_id"]))] or \
           [["-", "-", "-", "（无检出目标）"]]
    s = r["summary"]
    status = (f"完成：{s['num_objects']} 个目标 | 平均置信度 {s['conf_mean']} | "
              f"耗时 {s['elapsed_sec']}s | result_id={r['result_id']} | {r['weights_source']}")
    return (r.get("vis_path"), r["charts"].get("hist"), r["charts"].get("position"),
            rows, status, r["result_id"], session_id)


def _full_boxes(result_id):
    import json
    p = CFG.results_dir / f"{result_id}.json"
    return json.loads(p.read_text(encoding="utf-8")).get("boxes", []) if p.exists() else []


def _use_sample(name):
    return str(CFG.samples_dir / name) if name else None


# ---------- Tab2 批量与对比 ----------

def _batch_detect(files, conf, iou, session_id):
    session_id = session_id or uuid.uuid4().hex
    if not files:
        return None, "请先选择要批量检测的图像文件", session_id
    rows = []
    for f in files:
        path = _stage_upload(str(f))
        r = dispatch("detect_ships", {"image_path": path,
                                      "conf_threshold": conf, "iou_threshold": iou},
                     session_id)
        name = Path(str(f)).name
        if r.get("ok"):
            rows.append([name, r["summary"]["num_objects"],
                         r["summary"]["conf_mean"], r["result_id"]])
        else:
            rows.append([name, "-", "-", f"失败：{r.get('error')}"])
    _, choices = _history_rows()
    return (rows, f"批量检测完成：{len(files)} 张图像", session_id,
            gr.Dropdown(choices=choices, value=choices[0] if choices else None),
            gr.Dropdown(choices=choices, value=choices[1] if len(choices) > 1
                        else (choices[0] if choices else None)))


def _history_rows():
    recs = memory.list_detections(limit=30)
    rows = [[r["result_id"], Path(r["image_path"]).name, r["created_at"],
             r["num_objects"], "、".join(r.get("report_paths", {}).keys()) or "-"]
            for r in recs]
    choices = [f"{r['result_id']}  {Path(r['image_path']).name}" for r in recs]
    return rows, choices


def _run_compare(sel_a, sel_b):
    if not sel_a or not sel_b:
        return None, None, None, "请选择两个检测记录（基准 A 与待比较 B）"
    r = dispatch("compare_detections", {"result_id_a": sel_a.split()[0],
                                        "result_id_b": sel_b.split()[0]})
    if not r.get("ok"):
        return None, None, None, f"对比失败：{r.get('error')}"
    d = r["delta"]
    df = [["目标数", r["baseline"]["num_objects"], r["current"]["num_objects"],
           d["num_objects"]],
          ["平均置信度", r["baseline"]["conf_mean"], r["current"]["conf_mean"],
           d["conf_mean"]],
          ["新增目标（IoU 匹配）", "-", "-", d["added"]],
          ["消失目标（IoU 匹配）", "-", "-", d["removed"]]]
    trend = "增加" if d["num_objects"] >= 0 else "减少"
    summary = (f"以 {r['baseline']['image']} 为基准 → {r['current']['image']}："
               f"目标数{trend} {abs(d['num_objects'])} 个；新增 {d['added']}、"
               f"消失 {d['removed']}（IoU≥{r['iou_match']} 视为同一目标）。")
    return (r["vis_paths"]["baseline"], r["vis_paths"]["current"], df, summary)


# ---------- Tab3 智能对话 ----------

def _chat(message, history, session_id):
    session_id = session_id or uuid.uuid4().hex
    if not message or not message.strip():
        return history, "", session_id
    history = list(history) + [{"role": "user", "content": message}]
    yield history, "思考中…", session_id
    reply = run_agent(message, session_id)
    history = history + [{"role": "assistant", "content": reply}]
    yield history, "", session_id


# ---------- Tab4 文档问答 ----------

def _ingest_docs(files):
    if not files:
        return "请先选择资料文件（txt / md / docx / pdf）", None
    lines = []
    for f in files:
        src = Path(str(f))
        dst = docqa.store_doc_path(src.name)
        shutil.copyfile(src, dst)
        r = docqa.ingest(dst)
        lines.append(f"{src.name}: {'已索引 ' + str(r['chunks']) + ' 段' if r.get('ok') else r.get('error')}")
    lst = docqa.list_docs()
    table = [[d["name"], d["chunks"], d["created_at"]] for d in lst.get("docs", [])]
    tip = "已可在「智能对话」中提问领域知识，Agent 会检索资料引用原文作答。"
    return "\n".join(lines) + f"\n\n{tip}", table


# ---------- Tab5 历史与报告 ----------

def _on_load():
    rows, choices = _history_rows()
    return (rows, gr.Dropdown(choices=choices, value=choices[0] if choices else None),
            uuid.uuid4().hex)


def _refresh_history():
    rows, choices = _history_rows()
    return rows, gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def _show_selected(selection):
    if not selection:
        return None
    rid = selection.split()[0]
    rec = memory.get_detection(rid)
    return rec.get("vis_path") if rec else None


def _make_report(selection, fmt, session_id):
    session_id = session_id or uuid.uuid4().hex
    if not selection:
        return None, "请先点击刷新并选择一条检测记录"
    rid = selection.split()[0]
    r = dispatch("generate_report", {"result_id": rid, "format": fmt}, session_id)
    if not r.get("ok"):
        return None, f"报告生成失败：{r.get('error')}"
    return r["path"], f"已生成 {fmt} 报告：{r['path']}"


# ---------- Tab6 关于 ----------

def _about_md():
    try:
        import yaml
        p = yaml.safe_load(CFG.profile_path.read_text(encoding="utf-8")) \
            if CFG.profile_path.exists() else {}
        ds = (f"数据集：{p.get('name', '未生成画像（运行 scripts/make_dataset_profile.py）')}；"
              f"总计 {p.get('total', {}).get('images', '-')} 图 / "
              f"{p.get('total', {}).get('annotations', '-')} 框")
    except Exception as e:
        ds = f"数据集画像读取失败：{e}"
    det = ShipDetector.get()
    chat_state = "可用" if is_configured() else "未配置 API Key（对话功能停用）"
    return (f"**SAR-Agent** —— 基于 GLM-5.3-Flash 的 SAR 图像舰船检测与智能分析系统\n\n"
            f"- {ds}\n"
            f"- 检测模型：{det.weights_path}（{det.weights_source}，device={det.device}）\n"
            f"- 智能对话（GLM-5.3-Flash Agent，10 个工具）：{chat_state}\n"
            f"- 方案文档：prd.md（工程方案）/ agent.md（Agent 设计）/ docs/（演示与讲解）\n"
            f"- 仅用于学术与民用场景分析")


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate", radius_size="lg")
    with gr.Blocks(title="SAR-Agent", theme=theme, css=_CUSTOM_CSS) as demo:
        session_id = gr.State("")
        with gr.Row(elem_id="app-header"):
            gr.Markdown(
                "# SAR-Agent —— SAR 图像舰船检测与智能分析\n"
                "YOLOv8 感知 × GLM-5.3-Flash Agent 认知 ｜ 自然语言指挥 · 检测 · "
                "解读 · 对比 · 报告 · 记忆", elem_id="app-header-md")
        with gr.Tab("检测工作台"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="filepath", label="上传 SAR 图像",
                                        sources=["upload", "clipboard"], height=280)
                    sample_dd = gr.Dropdown(
                        choices=sorted(p.name for p in CFG.samples_dir.glob("*.jpg")),
                        label="或选择内置示例图（test 集精选）", interactive=True)
                    with gr.Row():
                        conf_s = gr.Slider(0.01, 0.95, value=CFG.conf, step=0.01,
                                           label="置信度阈值 conf")
                        iou_s = gr.Slider(0.1, 0.9, value=CFG.iou, step=0.05,
                                          label="NMS IoU")
                    btn = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    vis_img = gr.Image(label="标注结果", height=280)
                    status_tb = gr.Textbox(label="状态", interactive=False)
                    with gr.Row():
                        hist_img = gr.Image(label="置信度分布", height=200)
                        pos_img = gr.Image(label="目标位置分布", height=200)
            detail_df = gr.Dataframe(headers=["#", "类别", "置信度", "bbox"],
                                     label="检测明细", interactive=False)
            sample_dd.change(_use_sample, sample_dd, image_in)
            btn.click(_detect_and_render,
                      [image_in, conf_s, iou_s, session_id],
                      [vis_img, hist_img, pos_img, detail_df, status_tb, session_id])
        with gr.Tab("批量与对比"):
            with gr.Tab("批量检测"):
                batch_files = gr.Files(label="批量选择图像（JPG/PNG）", file_count="multiple")
                with gr.Row():
                    bconf_s = gr.Slider(0.01, 0.95, value=CFG.conf, step=0.01,
                                        label="置信度阈值")
                    biou_s = gr.Slider(0.1, 0.9, value=CFG.iou, step=0.05, label="NMS IoU")
                    batch_btn = gr.Button("批量检测", variant="primary")
                batch_df = gr.Dataframe(headers=["文件", "目标数", "平均置信度", "result_id"],
                                        label="批量结果", interactive=False)
                batch_status = gr.Textbox(label="状态", interactive=False)
            with gr.Tab("差分对比"):
                with gr.Row():
                    cmp_sel_a = gr.Dropdown(label="基准 A（较早）", interactive=True)
                    cmp_sel_b = gr.Dropdown(label="待比较 B（较新）", interactive=True)
                    cmp_btn = gr.Button("开始对比", variant="primary")
                with gr.Row():
                    cmp_img_a = gr.Image(label="基准 A 标注图", height=250)
                    cmp_img_b = gr.Image(label="B 标注图", height=250)
                cmp_df = gr.Dataframe(headers=["指标", "A", "B", "差值"],
                                      label="对比结果", interactive=False)
                cmp_status = gr.Textbox(label="结论", interactive=False)
            # 事件绑定放在两个子 Tab 组件全部定义之后
            batch_btn.click(_batch_detect,
                            [batch_files, bconf_s, biou_s, session_id],
                            [batch_df, batch_status, session_id,
                             cmp_sel_a, cmp_sel_b])
            cmp_btn.click(_run_compare, [cmp_sel_a, cmp_sel_b],
                          [cmp_img_a, cmp_img_b, cmp_df, cmp_status])
        with gr.Tab("智能对话"):
            llm_hint = "" if is_configured() else \
                "（未配置 ZHIPUAI_API_KEY，请先完成 configs/.env 配置）"
            gr.Markdown(f"用自然语言指挥 Agent：检测、解读、对比、生成报告、检索资料、"
                        f"记忆偏好。{llm_hint}")
            chatbot = gr.Chatbot(label="SAR-Agent", height=430)
            msg_tb = gr.Textbox(placeholder="例如：检测第二张示例图，然后和之前的结果对比一下",
                                label="输入指令（回车发送）")
            busy_md = gr.Markdown()
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")
            submit = msg_tb.submit(_chat, [msg_tb, chatbot, session_id],
                                   [chatbot, busy_md, session_id])
            submit.then(lambda: "", None, msg_tb)
            send = send_btn.click(_chat, [msg_tb, chatbot, session_id],
                                  [chatbot, busy_md, session_id])
            send.then(lambda: "", None, msg_tb)
            clear_btn.click(lambda: ([], ""), None, [chatbot, busy_md])
        with gr.Tab("文档问答"):
            gr.Markdown("导入 SAR 参考资料（判读手册/术语表等），之后在「智能对话」中"
                        "提问领域知识，Agent 会调用 `search_reference_docs` 引用原文作答。")
            doc_files = gr.Files(label="上传资料（txt / md / docx / pdf）",
                                 file_count="multiple")
            doc_btn = gr.Button("导入并建立索引", variant="primary")
            doc_status = gr.Textbox(label="导入结果", interactive=False)
            doc_df = gr.Dataframe(headers=["文档", "分段数", "导入时间"],
                                  label="已导入资料", interactive=False)
            doc_btn.click(_ingest_docs, doc_files, [doc_status, doc_df])
        with gr.Tab("历史与报告"):
            hist_df = gr.Dataframe(headers=["result_id", "图像", "时间", "目标数", "已有报告"],
                                   label="检测历史", interactive=False)
            with gr.Row():
                sel_dd = gr.Dropdown(label="选择检测记录", interactive=True)
                refresh_btn = gr.Button("刷新历史")
            vis_prev = gr.Image(label="结果回看", height=260)
            with gr.Row():
                fmt_radio = gr.Radio(["markdown", "pdf", "docx"], value="markdown",
                                     label="报告格式")
                report_btn = gr.Button("生成报告并下载", variant="primary")
            report_file = gr.File(label="报告下载")
            report_status = gr.Textbox(label="状态", interactive=False)
            refresh_btn.click(_refresh_history, None, [hist_df, sel_dd])
            demo.load(_on_load, None, [hist_df, sel_dd, session_id])
            sel_dd.change(_show_selected, sel_dd, vis_prev)
            report_btn.click(_make_report, [sel_dd, fmt_radio, session_id],
                             [report_file, report_status])
        with gr.Tab("关于"):
            gr.Markdown(_about_md)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name=CFG.server_host, server_port=CFG.server_port,
                inbrowser=False, show_error=True)
