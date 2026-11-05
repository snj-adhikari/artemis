import asyncio
import base64
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel
import pytest

from artemis.llm.claude_code import ClaudeCodeChatModel, _sniff_media_type
from artemis.llm.router import ModelEndpoint, ModelFactory, ModelProvider

JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"0" * 20).decode()
PNG_B64 = base64.b64encode(b"\x89PNG" + b"0" * 20).decode()
TOOL = {
    "type": "function",
    "function": {"name": "tap", "description": "tap", "parameters": {"type": "object"}},
}


def _stdout(text, is_error=False, usage=None):
    result = {
        "type": "result",
        "is_error": is_error,
        "result": text,
        "usage": usage or {"input_tokens": 3, "output_tokens": 2},
    }
    return (json.dumps({"type": "system"}) + "\n" + json.dumps(result) + "\n").encode()


class FakeProc:
    def __init__(self, stdout=b"", stderr=b"", hang=False):
        self.stdout, self.stderr, self.hang = stdout, stderr, hang
        self.returncode = None
        self.stdin_data = None
        self.kill = MagicMock()

    async def communicate(self, data=None):
        self.stdin_data = data
        if self.hang:
            await asyncio.sleep(10)
        self.returncode = 0
        return self.stdout, self.stderr


def _patch_exec(procs):
    calls = []

    async def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return procs[len(calls) - 1]

    return patch("asyncio.create_subprocess_exec", fake), calls


def test_sniff_media_type():
    assert _sniff_media_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert _sniff_media_type(b"GIF89a") == "image/gif"
    assert _sniff_media_type(b"RIFF\x00\x00\x00\x00WEBPx") == "image/webp"
    assert _sniff_media_type(b"junk") == "image/png"


@pytest.mark.asyncio
async def test_argv_env_and_conversion(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    proc = FakeProc(
        _stdout(
            "hi",
            usage={
                "input_tokens": 1,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 5,
                "output_tokens": 2,
            },
        )
    )
    ctx, calls = _patch_exec([proc])
    messages = [
        SystemMessage("be brief"),
        HumanMessage(
            [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{JPEG_B64}"}},
            ]
        ),
        AIMessage("", tool_calls=[{"name": "tap", "args": {"x": 1}, "id": "c1"}]),
        ToolMessage(
            [{"type": "image", "base64": PNG_B64, "mime_type": "image/jpeg"}],
            tool_call_id="c1",
            name="tap",
        ),
    ]
    with ctx:
        out = await ClaudeCodeChatModel().ainvoke(messages)

    argv, kwargs = calls[0]
    assert argv[0] == "claude" and "--bare" not in argv
    for flag in (
        "-p", "--input-format", "--output-format", "--verbose", "--max-turns",
        "--tools", "--setting-sources", "--strict-mcp-config",
        "--no-session-persistence", "--disable-slash-commands",
    ):  # fmt: skip
        assert flag in argv
    assert argv[argv.index("--system-prompt") + 1] == "be brief"
    assert argv[argv.index("--model") + 1] == "haiku"
    assert "CLAUDECODE" not in kwargs["env"] and "ANTHROPIC_API_KEY" not in kwargs["env"]

    line = json.loads(proc.stdin_data)
    blocks = line["message"]["content"]
    images = [b for b in blocks if b["type"] == "image"]
    assert [i["source"]["media_type"] for i in images] == ["image/jpeg", "image/png"]
    assert any("tool_calls" in b.get("text", "") for b in blocks)
    assert any("[tool result tap c1]" in b.get("text", "") for b in blocks)
    assert out.content == "hi"
    assert out.usage_metadata == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


@pytest.mark.asyncio
async def test_default_system_prompt_when_none():
    ctx, calls = _patch_exec([FakeProc(_stdout("x"))])
    with ctx:
        await ClaudeCodeChatModel().ainvoke([HumanMessage("hi")])
    argv = calls[0][0]
    assert argv[argv.index("--system-prompt") + 1]


@pytest.mark.parametrize(
    "reply",
    [
        '{"tool_calls":[{"name":"tap","args":{"x":1}},{"name":"tap","args":{"x":2}}]}',
        '```json\n{"tool_calls":[{"name":"tap","args":{"x":1}},{"name":"tap","args":{"x":2}}]}\n```',
    ],
)
@pytest.mark.asyncio
async def test_tool_calls_parsed_with_unique_ids(reply):
    ctx, _ = _patch_exec([FakeProc(_stdout(reply))])
    model = ClaudeCodeChatModel().bind_tools([TOOL])
    with ctx:
        out = await model.ainvoke([HumanMessage("go")])
    assert [c["args"] for c in out.tool_calls] == [{"x": 1}, {"x": 2}]
    assert len({c["id"] for c in out.tool_calls}) == 2


@pytest.mark.asyncio
async def test_content_reply_with_tools():
    ctx, _ = _patch_exec([FakeProc(_stdout('{"content":"done"}'))])
    with ctx:
        out = await ClaudeCodeChatModel().bind_tools([TOOL]).ainvoke([HumanMessage("go")])
    assert out.content == "done" and not out.tool_calls


@pytest.mark.asyncio
async def test_bad_json_retries_once():
    ctx, calls = _patch_exec(
        [
            FakeProc(_stdout("not json")),
            FakeProc(_stdout('{"tool_calls":[{"name":"tap","args":{}}]}')),
        ]
    )
    with ctx:
        out = await ClaudeCodeChatModel().bind_tools([TOOL]).ainvoke([HumanMessage("go")])
    assert len(calls) == 2 and out.tool_calls[0]["name"] == "tap"


@pytest.mark.asyncio
async def test_forced_choice_fails_after_retry():
    ctx, calls = _patch_exec([FakeProc(_stdout("nope")), FakeProc(_stdout("nope"))])
    model = ClaudeCodeChatModel().bind_tools([TOOL], tool_choice="tap")
    with ctx, pytest.raises(ValueError):
        await model.ainvoke([HumanMessage("go")])
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_is_error_preserved():
    ctx, _ = _patch_exec([FakeProc(_stdout("429 rate limit", is_error=True))])
    with ctx, pytest.raises(RuntimeError, match="429 rate limit"):
        await ClaudeCodeChatModel().ainvoke([HumanMessage("hi")])


@pytest.mark.asyncio
async def test_timeout_kills_process():
    proc = FakeProc(hang=True)
    ctx, _ = _patch_exec([proc])
    with ctx, pytest.raises(asyncio.TimeoutError):
        await ClaudeCodeChatModel(timeout_s=0.05).ainvoke([HumanMessage("hi")])
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_with_structured_output():
    class Point(BaseModel):
        x: int

    ctx, _ = _patch_exec([FakeProc(_stdout('{"tool_calls":[{"name":"Point","args":{"x":7}}]}'))])
    with ctx:
        out = (
            await ClaudeCodeChatModel().with_structured_output(Point).ainvoke([HumanMessage("go")])
        )
    assert out == Point(x=7)


def test_router():
    for name in ("claude-code", "claude_code", "claudecode"):
        assert ModelProvider.from_string(name) == ModelProvider.CLAUDE_CODE
    assert ModelProvider.from_string("claude") == ModelProvider.ANTHROPIC
    model = ModelFactory.create_model(
        ModelEndpoint(provider=ModelProvider.CLAUDE_CODE, model_name="haiku", timeout_seconds=12)
    )
    assert isinstance(model, ClaudeCodeChatModel)
    assert model.model_name == "haiku" and model.timeout_s == 12
