"""Unit tests for the LLM interaction layer (mocked HTTP transport)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from rehab_sim.agent import (
    AgentEvent,
    LLMAgent,
    LLMClient,
    LLMConfig,
    load_llm_config,
)
from rehab_sim.agent.llm_agent import _parse_json_content

ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides: object) -> LLMConfig:
    values: dict[str, object] = dict(
        enabled=True,
        mode="hybrid",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_TEST_KEY",
        timeout_s=5.0,
        event_enrichment_enabled=True,
        event_enrichment_cooldown_s=0.0,
        summary_enabled=True,
        chat_enabled=True,
        max_history_messages=10,
    )
    values.update(overrides)
    return LLMConfig(**values)  # type: ignore[arg-type]


def _client(content: str, status: int = 200) -> LLMClient:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})

    return LLMClient(_config(), api_key="test-key", transport=httpx.MockTransport(handler))


def _event() -> AgentEvent:
    return AgentEvent(
        event="force_too_high",
        message="模板消息",
        severity="warning",
        timestamp_s=1.0,
        context={"force_level": "high"},
    )


def test_load_llm_config_reads_yaml_section() -> None:
    config = load_llm_config(ROOT / "configs" / "agent.yaml")
    assert config.enabled is True
    assert config.mode == "hybrid"
    assert config.model == "deepseek-chat"
    assert config.api_key_env == "DEEPSEEK_API_KEY"


def test_load_llm_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="agent"):
        load_llm_config(ROOT / "configs" / "safety.yaml")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"message": "ok"}', {"message": "ok"}),
        ('```json\n{"message": "ok"}\n```', {"message": "ok"}),
        ("not json", None),
    ],
)
def test_parse_json_content(content: str, expected: object) -> None:
    assert _parse_json_content(content) == expected


def test_client_chat_returns_content() -> None:
    result = asyncio.run(
        _client('{"message": "加油"}').chat([{"role": "user", "content": "hi"}])
    )
    assert result == '{"message": "加油"}'


def test_client_chat_http_error_returns_none() -> None:
    result = asyncio.run(
        _client("", status=500).chat([{"role": "user", "content": "hi"}])
    )
    assert result is None


def test_client_chat_missing_key_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_TEST_KEY", raising=False)
    client = LLMClient(_config(), api_key=None)
    assert client.available is False
    assert asyncio.run(client.chat([{"role": "user", "content": "hi"}])) is None


def test_client_chat_malformed_response_returns_none() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = LLMClient(_config(), api_key="test-key", transport=httpx.MockTransport(handler))
    assert asyncio.run(client.chat([{"role": "user", "content": "hi"}])) is None


def test_llm_agent_enabled_requires_hybrid_mode_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_TEST_KEY", raising=False)
    assert LLMAgent(_config()).enabled is False
    rules_only = LLMAgent(_config(mode="rules_only"), client=_client(""))
    assert rules_only.enabled is False


def test_enrich_event_returns_personalised_message() -> None:
    agent = LLMAgent(_config(), client=_client('{"message": "请稍微放松握把，保持平稳呼吸。"}'))
    message = asyncio.run(agent.enrich_event(_event()))
    assert message == "请稍微放松握把，保持平稳呼吸。"


def test_enrich_event_respects_cooldown() -> None:
    agent = LLMAgent(
        _config(event_enrichment_cooldown_s=1000.0),
        client=_client('{"message": "第一条"}'),
    )
    assert asyncio.run(agent.enrich_event(_event())) is not None
    assert asyncio.run(agent.enrich_event(_event())) is None


def test_enrich_event_failure_returns_none() -> None:
    agent = LLMAgent(_config(), client=_client("", status=500))
    assert asyncio.run(agent.enrich_event(_event())) is None


def test_enrich_event_plain_text_fallback() -> None:
    agent = LLMAgent(_config(), client=_client("直接返回的自然语言反馈"))
    message = asyncio.run(agent.enrich_event(_event()))
    assert message == "直接返回的自然语言反馈"


def test_generate_summary_builds_structured_summary() -> None:
    agent = LLMAgent(
        _config(),
        client=_client(
            '{"title": "训练总结", "message": "表现良好", '
            '"highlights": ["误差低", "主动做功高"], "recommendation": "继续保持"}'
        ),
    )
    summary = asyncio.run(
        agent.generate_summary({"average_tracking_error": 0.01}, [_event()])
    )
    assert summary is not None
    assert summary.title == "训练总结"
    assert summary.highlights == ["误差低", "主动做功高"]
    assert summary.event_count == 1


def test_generate_summary_invalid_json_returns_none() -> None:
    agent = LLMAgent(_config(), client=_client("这不是 JSON"))
    summary = asyncio.run(agent.generate_summary({"completed": True}, []))
    assert summary is None


def test_answer_returns_json_message() -> None:
    agent = LLMAgent(_config(), client=_client('{"message": "今天表现不错，继续保持。"}'))
    reply = asyncio.run(agent.answer("今天练得怎么样？", {"task_progress": 0.9}))
    assert reply == "今天表现不错，继续保持。"


def test_answer_falls_back_to_plain_text() -> None:
    agent = LLMAgent(_config(), client=_client("基于数据来看，建议休息十分钟。"))
    reply = asyncio.run(agent.answer("我累了吗？", {"fatigue": 0.8}))
    assert reply == "基于数据来看，建议休息十分钟。"


def test_answer_disabled_returns_none() -> None:
    agent = LLMAgent(_config(chat_enabled=False), client=_client(""))
    assert asyncio.run(agent.answer("你好", {})) is None


def test_answer_never_sends_invalid_history_roles() -> None:
    """DeepSeek rejects roles other than system/user/assistant with a 400."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        roles = {message["role"] for message in body["messages"]}
        if not roles <= {"system", "user", "assistant"}:
            return httpx.Response(400, json={"error": "invalid role"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "回答正常"}}]}
        )

    client = LLMClient(_config(), api_key="test-key", transport=httpx.MockTransport(handler))
    agent = LLMAgent(_config(), client=client)
    reply = asyncio.run(
        agent.answer("问题", {}, history=[{"role": "agent", "content": "历史回复"}])
    )
    assert reply == "回答正常"
