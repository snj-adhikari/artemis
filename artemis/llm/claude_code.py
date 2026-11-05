# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chat model backed by the local Claude Code CLI (``claude -p``).

Uses the user's Claude Code login; ANTHROPIC_API_KEY is stripped from the
subprocess env so the plan login is never bypassed.
"""

import asyncio
import base64
import json
import os
import tempfile
from typing import Any
import uuid

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from artemis.llm.structured import ParseFailure, content_to_text, parse_structured

_DEFAULT_SYSTEM = "You are a helpful assistant. Follow the user's instructions exactly."
_REPAIR = (
    "Your previous reply was not valid. Reply again with only the required JSON object, "
    "no prose or fences."
)


def _sniff_media_type(b: bytes) -> str:
    if b.startswith(b"\x89PNG"):
        return "image/png"
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if b.startswith(b"GIF8"):
        return "image/gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _image_block(b64: str) -> dict[str, Any]:
    head = b64[:64]
    media_type = _sniff_media_type(base64.b64decode(head + "=" * (-len(head) % 4)))
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}


def _content_blocks(content: Any, prefix: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": f"{prefix}\n{content}"}]
    blocks: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
    for part in content or []:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            url = part["image_url"]
            url = url.get("url", "") if isinstance(url, dict) else url
            blocks.append(_image_block(url.split(",", 1)[1]))
        elif part.get("type") == "image":
            blocks.append(_image_block(part.get("base64") or part.get("data")))
    return blocks


def _tool_contract(tools: list[dict], tool_choice: str | None) -> str:
    text = (
        "Available tools (JSON schemas):\n"
        + json.dumps(tools)
        + '\nReply with only JSON: either {"tool_calls":[{"name":"<tool>","args":{...}}]} '
        'to call tools, or {"content":"<text>"} to answer in text.'
    )
    if tool_choice in ("any", "required"):
        text += "\nYou must call at least one tool; do not reply with content."
    elif tool_choice:
        text += f"\nYou must call the tool named {tool_choice}."
    return text


def _to_stream_json(
    messages: list[BaseMessage], tools: list[dict], tool_choice: str | None
) -> tuple[str, str]:
    system = [content_to_text(m.content) for m in messages if isinstance(m, SystemMessage)]
    if tools:
        system.append(_tool_contract(tools, tool_choice))
    blocks: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            blocks += _content_blocks(m.content, "[user]")
        elif isinstance(m, AIMessage):
            blocks += _content_blocks(m.content, "[assistant]")
            if m.tool_calls:
                calls = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
                blocks.append({"type": "text", "text": json.dumps({"tool_calls": calls})})
        elif isinstance(m, ToolMessage):
            prefix = f"[tool result {m.name or ''} {m.tool_call_id}]"
            blocks += _content_blocks(m.content, prefix)
    line = {"type": "user", "message": {"role": "user", "content": blocks}}
    return "\n\n".join(s for s in system if s) or _DEFAULT_SYSTEM, json.dumps(line)


class ClaudeCodeChatModel(BaseChatModel):
    model_name: str = "haiku"
    timeout_s: float = 60.0
    cli_path: str = "claude"
    tools: list[dict] = []
    tool_choice: str | None = None

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    def bind_tools(self, tools: Any, tool_choice: str | None = None, **kwargs: Any):
        return self.model_copy(
            update={
                "tools": [convert_to_openai_tool(t) for t in tools],
                "tool_choice": tool_choice,
            }
        )

    async def _run(self, system: str, line: str) -> dict[str, Any]:
        env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "ANTHROPIC_API_KEY")}
        proc = await asyncio.create_subprocess_exec(
            self.cli_path,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model_name,
            "--max-turns",
            "1",
            "--tools",
            "",
            "--system-prompt",
            system,
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--disable-slash-commands",
            cwd=tempfile.gettempdir(),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate((line + "\n").encode()), self.timeout_s
            )
        finally:
            if proc.returncode is None:
                proc.kill()
        result = None
        for raw in out.decode(errors="replace").splitlines():
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "result":
                result = obj
        if result is None:
            tail = err.decode(errors="replace")[-500:]
            raise RuntimeError(f"claude CLI returned no result: {tail}")
        if result.get("is_error"):
            raise RuntimeError(str(result.get("result")))
        return result

    @staticmethod
    def _usage(result: dict[str, Any]) -> dict[str, int]:
        u = result.get("usage") or {}
        inp = (
            u.get("input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
        )
        out = u.get("output_tokens", 0)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    def _to_message(self, result: dict[str, Any]) -> AIMessage | ParseFailure:
        text = str(result.get("result", ""))
        usage = self._usage(result)
        if not self.tools:
            return AIMessage(content=text, usage_metadata=usage)
        data = parse_structured(text)
        if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
            calls = [
                {
                    "name": c["name"],
                    "args": c.get("args") or {},
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                }
                for c in data["tool_calls"]
            ]
            return AIMessage(content="", tool_calls=calls, usage_metadata=usage)
        if isinstance(data, dict) and "content" in data and not self.tool_choice:
            return AIMessage(content=str(data["content"]), usage_metadata=usage)
        if isinstance(data, ParseFailure):
            return data
        return ParseFailure(raw=text, error="reply is not a valid tool call")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, line = _to_stream_json(messages, self.tools, self.tool_choice)
        result = await self._run(system, line)
        msg = self._to_message(result)
        if isinstance(msg, ParseFailure):
            retry = json.loads(line)
            retry["message"]["content"] += [
                {"type": "text", "text": f"[assistant]\n{msg.raw}"},
                {"type": "text", "text": f"[user]\n{_REPAIR}"},
            ]
            result = await self._run(system, json.dumps(retry))
            msg = self._to_message(result)
            if isinstance(msg, ParseFailure):
                if self.tool_choice:
                    raise ValueError(f"claude CLI reply was not a valid tool call: {msg.error}")
                msg = AIMessage(content=msg.raw, usage_metadata=self._usage(result))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop, **kwargs))
