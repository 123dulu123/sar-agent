"""GLM-5.3-Flash 客户端：OpenAI 兼容协议封装，统一重试与超时。

返回结构（与 openai 对象解耦，便于 core.py 构造消息）：
    {"content": str | None,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}],
     "usage": {"prompt_tokens": int, "completion_tokens": int}}
"""
import json
import logging
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from agent.config import CFG

log = logging.getLogger(__name__)

RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)


class LLMError(Exception):
    """LLM 调用失败（网络/接口/配额等）。"""


class LLMNotConfigured(LLMError):
    """未配置 API Key。"""


def is_configured() -> bool:
    return bool(CFG.llm_api_key)


def _client() -> OpenAI:
    if not is_configured():
        raise LLMNotConfigured("未配置 ZHIPUAI_API_KEY")
    return OpenAI(api_key=CFG.llm_api_key, base_url=CFG.llm_base_url,
                  timeout=CFG.llm_timeout, max_retries=0)


def chat(messages: list[dict], tools: list[dict] | None = None,
         temperature: float | None = None) -> dict:
    """调用 GLM-5.3-Flash；瞬时错误按指数退避重试。"""
    kwargs = dict(model=CFG.llm_model, messages=messages,
                  temperature=CFG.llm_temperature if temperature is None else temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_err: Exception | None = None
    for attempt in range(CFG.llm_max_retries):
        try:
            resp = _client().chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            tool_calls = []
            for tc in (msg.tool_calls or []):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name,
                                   "arguments": args if isinstance(args, dict) else {}})
            usage = getattr(resp, "usage", None)
            return {"content": msg.content,
                    "tool_calls": tool_calls,
                    "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", 0),
                              "completion_tokens": getattr(usage, "completion_tokens", 0)}}
        except RETRYABLE as e:
            last_err = e
            wait = 2 ** attempt
            log.warning("LLM 瞬时错误（第%d次）：%s，%ds 后重试", attempt + 1, e, wait)
            time.sleep(wait)
        except APIStatusError as e:
            raise LLMError(f"接口返回错误（HTTP {e.status_code}）：请检查 API Key、模型名与账户额度") from e

    raise LLMError(f"LLM 调用重试 {CFG.llm_max_retries} 次后仍失败：{last_err}")
