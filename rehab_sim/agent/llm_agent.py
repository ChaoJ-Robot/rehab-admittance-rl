"""LLM-powered interaction layer for the Phase 8 Agent.

The rule-based agent keeps real-time event detection; this module adds an
optional, fully asynchronous LLM layer (DeepSeek / OpenAI-compatible
``/chat/completions``) that personalises event messages, generates training
summaries, and answers therapist questions. Every method degrades to ``None``
on failure so the control path and telemetry are never affected.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from rehab_sim.agent.rule_based import AgentEvent, AgentSummary
from rehab_sim.config import load_yaml

LOGGER = logging.getLogger("rehab.agent.llm")

SYSTEM_ENRICH_PROMPT = (
    "你是上肢康复训练系统的交互教练。规则系统检测到了一个训练事件，请基于事件类型、"
    "严重级别和量化上下文，生成一句简短、温暖、专业的中文口头反馈（不超过 60 字），"
    "直接指导患者如何调整动作。安全准则：绝不鼓励患者过度用力；疲劳或交互力异常时建议"
    "休息或告知治疗师；不得给出医疗诊断。只输出 JSON：{\"message\": \"...\"}。"
)

SYSTEM_SUMMARY_PROMPT = (
    "你是上肢康复训练系统的训练总结助手。请基于训练报告数据和事件列表，为治疗师生成"
    "结构化总结。安全准则：不得给出医疗诊断，只描述训练表现。只输出 JSON："
    "{\"title\": \"...\", \"message\": \"...\", \"highlights\": [\"...\", \"...\"], "
    "\"recommendation\": \"...\"}，highlights 不超过 3 条，全部使用中文。"
)

SYSTEM_CHAT_PROMPT = (
    "你是上肢康复训练系统的交互教练助手。患者或治疗师会向你提问，请基于当前训练数据"
    "（训练任务、患者类型、指标、事件）用中文回答，简洁专业。安全准则：绝不鼓励患者过度"
    "用力；疲劳或交互力异常时建议休息或告知治疗师；不得给出医疗诊断；若问题超出训练数据"
    "范围，礼貌说明并建议咨询治疗师。"
)


@dataclass(frozen=True)
class LLMConfig:
    """Validated LLM integration settings loaded from ``agent.yaml``."""

    enabled: bool
    mode: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    timeout_s: float
    event_enrichment_enabled: bool
    event_enrichment_cooldown_s: float
    summary_enabled: bool
    chat_enabled: bool
    max_history_messages: int


def load_llm_config(agent_config_path: str | Path) -> LLMConfig:
    """Load the ``agent.llm`` section from ``configs/agent.yaml``."""

    config = load_yaml(agent_config_path)
    agent_section = config.get("agent")
    if not isinstance(agent_section, Mapping):
        raise ValueError("agent config must contain an agent mapping")
    section = agent_section.get("llm")
    if not isinstance(section, Mapping):
        raise ValueError("agent config must contain an agent.llm mapping")
    timeout_s = float(section.get("timeout_s", 20.0))
    cooldown_s = float(section.get("event_enrichment_cooldown_s", 15.0))
    max_history = int(section.get("max_history_messages", 20))
    if timeout_s <= 0.0 or cooldown_s < 0.0 or max_history <= 0:
        raise ValueError("llm timeout/cooldown/max_history must be positive")
    return LLMConfig(
        enabled=bool(section.get("enabled", False)),
        mode=str(agent_section.get("mode", "rules_only")),
        provider=str(section.get("provider", "deepseek")),
        base_url=str(section.get("base_url", "https://api.deepseek.com")).rstrip("/"),
        model=str(section.get("model", "deepseek-chat")),
        api_key_env=str(section.get("api_key_env", "DEEPSEEK_API_KEY")),
        timeout_s=timeout_s,
        event_enrichment_enabled=bool(section.get("event_enrichment_enabled", True)),
        event_enrichment_cooldown_s=cooldown_s,
        summary_enabled=bool(section.get("summary_enabled", True)),
        chat_enabled=bool(section.get("chat_enabled", True)),
        max_history_messages=max_history,
    )


def _parse_json_content(content: str) -> dict[str, Any] | None:
    """Parse a model reply as JSON, tolerating ```json fences."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class LLMClient:
    """Thin asynchronous client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: LLMConfig,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key if api_key is not None else os.environ.get(config.api_key_env, "")
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        """Whether the key is present; the endpoint itself is checked per call."""

        return bool(self.api_key)

    def _client_for(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout_s,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str | None:
        """Send one chat request and return the assistant text, or None on failure."""

        if not self.available:
            LOGGER.warning("llm_api_key_missing env=%s", self.config.api_key_env)
            return None
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = await self._client_for().post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return str(content).strip() if content else None
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            LOGGER.warning("llm_request_failed error=%s", error)
            return None


class LLMAgent:
    """Asynchronous LLM layer; every method returns None when unavailable."""

    def __init__(self, config: LLMConfig, client: LLMClient | None = None) -> None:
        self.config = config
        self.client = client if client is not None else LLMClient(config)
        self._last_enrich_at = 0.0

    @property
    def enabled(self) -> bool:
        """LLM layer is active only in hybrid mode with a configured key."""

        return (
            self.config.enabled
            and self.config.mode != "rules_only"
            and self.client.available
        )

    @property
    def event_enrichment_enabled(self) -> bool:
        return self.enabled and self.config.event_enrichment_enabled

    @property
    def summary_enabled(self) -> bool:
        return self.enabled and self.config.summary_enabled

    @property
    def chat_enabled(self) -> bool:
        return self.enabled and self.config.chat_enabled

    def _enrichment_allowed(self) -> bool:
        now = time.monotonic()
        if now - self._last_enrich_at < self.config.event_enrichment_cooldown_s:
            return False
        self._last_enrich_at = now
        return True

    async def enrich_event(self, event: AgentEvent) -> str | None:
        """Personalise one rule-detected event into a natural feedback message."""

        if not self.event_enrichment_enabled or not self._enrichment_allowed():
            return None
        context = ", ".join(f"{key}={value}" for key, value in sorted(event.context.items()))
        messages = [
            {"role": "system", "content": SYSTEM_ENRICH_PROMPT},
            {
                "role": "user",
                "content": (
                    f"事件类型: {event.event}，严重级别: {event.severity}，"
                    f"事件上下文: {context}"
                ),
            },
        ]
        content = await self.client.chat(messages, temperature=0.8, json_mode=True)
        if content is None:
            return None
        parsed = _parse_json_content(content)
        if parsed is not None and isinstance(parsed.get("message"), str):
            return str(parsed["message"]).strip()
        return content[:200]

    async def generate_summary(
        self,
        report: Mapping[str, Any],
        events: Sequence[AgentEvent],
    ) -> AgentSummary | None:
        """Generate a structured training summary, or None to keep the template."""

        if not self.summary_enabled:
            return None
        report_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(report.items()))
        event_lines = "\n".join(
            f"- [{event.timestamp_s:.1f}s] {event.event} ({event.severity}): {event.message}"
            for event in events
        )
        messages = [
            {"role": "system", "content": SYSTEM_SUMMARY_PROMPT},
            {
                "role": "user",
                "content": f"训练报告:\n{report_lines}\n\n事件列表:\n{event_lines}",
            },
        ]
        content = await self.client.chat(messages, temperature=0.7, json_mode=True)
        if content is None:
            return None
        parsed = _parse_json_content(content)
        if parsed is None:
            LOGGER.warning("llm_summary_invalid_json content=%s", content[:200])
            return None
        try:
            highlights = [str(item) for item in parsed.get("highlights", [])][:3]
            return AgentSummary(
                title=str(parsed.get("title", "训练总结"))[:60],
                message=str(parsed.get("message", ""))[:300],
                highlights=highlights,
                recommendation=str(parsed.get("recommendation", ""))[:200],
                event_count=len(events),
            )
        except (TypeError, ValueError) as error:
            LOGGER.warning("llm_summary_parse_failed error=%s", error)
            return None

    async def answer(
        self,
        question: str,
        context: Mapping[str, Any],
        history: Sequence[Mapping[str, str]] = (),
    ) -> str | None:
        """Answer a patient/therapist question grounded in training data."""

        if not self.chat_enabled:
            return None
        context_lines = "\n".join(
            f"- {key}: {value}" for key, value in sorted(context.items())
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_CHAT_PROMPT}]
        messages.extend(
            {
                # OpenAI-compatible APIs only accept system/user/assistant roles.
                "role": (
                    str(item["role"]) if str(item["role"]) in ("user", "system") else "assistant"
                ),
                "content": str(item["content"]),
            }
            for item in history[-self.config.max_history_messages :]
        )
        messages.append(
            {
                "role": "user",
                "content": f"当前训练数据:\n{context_lines}\n\n问题: {question}",
            }
        )
        # Free-text mode: DeepSeek JSON mode requires the literal "json" in the
        # prompt, and a natural chat reply should not be forced into a schema.
        content = await self.client.chat(messages, temperature=0.7, json_mode=False)
        if content is None:
            return None
        parsed = _parse_json_content(content)
        if parsed is not None:
            for key in ("message", "answer", "reply"):
                if isinstance(parsed.get(key), str) and parsed[key].strip():
                    return str(parsed[key]).strip()
        return content[:500]
