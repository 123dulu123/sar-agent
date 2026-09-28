"""Agent 主循环（ReAct 风格，见 agent.md §5）：

组装上下文（系统提示词 + L2 事实 + L1 历史）→ GLM 决策 → Function Calling
→ 观察回填 → 迭代（至多 MAX_STEPS）→ 最终回答写回记忆。
"""
import json
import logging

from agent import llm, memory
from agent.config import CFG
from agent.prompts import build_system_prompt
from agent.tools_registry import TOOLS, dispatch

log = logging.getLogger(__name__)

TOOL_RESULT_MAX_CHARS = 4000  # 回填给模型的工具结果上限，防上下文膨胀

_NOT_CONFIGURED_REPLY = (
    "尚未配置 GLM API Key，对话分析功能不可用（检测工作台不受影响）。\n"
    "配置方法：把 configs/.env.example 复制为 configs/.env，填入 ZHIPUAI_API_KEY，"
    "然后重启应用。")


def _session_hint() -> dict | None:
    return memory.latest_detection()


def _trim(result: dict) -> str:
    text = json.dumps(result, ensure_ascii=False)
    return text if len(text) <= TOOL_RESULT_MAX_CHARS else \
        text[:TOOL_RESULT_MAX_CHARS] + f'...（截断，原文 {len(text)} 字符）'


def run_agent(user_input: str, session_id: str) -> str:
    """执行一轮完整的 Agent 任务，返回最终中文回答。"""
    if not llm.is_configured():
        return _NOT_CONFIGURED_REPLY

    facts = memory.load_facts()
    history = memory.recent_messages(session_id, CFG.llm_history_window)
    messages = [{"role": "system", "content": build_system_prompt(facts, _session_hint())}]
    messages += [m for m in history if m["role"] in ("user", "assistant")]
    messages.append({"role": "user", "content": user_input})
    memory.save_message(session_id, "user", user_input)

    for step in range(CFG.llm_max_steps):
        try:
            resp = llm.chat(messages, tools=TOOLS)
        except llm.LLMError as e:
            reply = f"模型调用失败：{e}。检测功能不受影响，可稍后在对话中重试。"
            memory.save_message(session_id, "assistant", reply)
            return reply

        if not resp["tool_calls"]:
            reply = resp["content"] or "（模型未返回内容，请重试）"
            memory.save_message(session_id, "assistant", reply)
            return reply

        messages.append({
            "role": "assistant",
            "content": resp["content"] or "",
            "tool_calls": [{"id": tc["id"], "type": "function",
                            "function": {"name": tc["name"],
                                         "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                           for tc in resp["tool_calls"]],
        })
        for tc in resp["tool_calls"]:
            result = dispatch(tc["name"], tc["arguments"], session_id)
            log.info("agent 工具调用 step=%d tool=%s ok=%s",
                     step + 1, tc["name"], result.get("ok"))
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": _trim(result)})

    # 达到 MAX_STEPS 仍未收敛：带观察摘要做一次兜底总结（不再提供工具）
    summary = "\n".join(m["content"][:600] for m in messages if m["role"] == "tool")[-3000:]
    try:
        resp = llm.chat(messages + [{"role": "user", "content":
              "工具调用轮次已达上限。请基于以上工具观察结果，直接给出对用户问题的最终回答，"
              "不要表示要继续调用工具。"}], tools=None)
        reply = resp["content"] or "已完成部分工具调用，但未能生成总结，请重试或简化问题。"
    except llm.LLMError:
        reply = f"工具调用轮次达到上限（{CFG.llm_max_steps}），且总结阶段模型调用失败。已有观察摘要：\n{summary}"
    memory.save_message(session_id, "assistant", reply)
    return reply
