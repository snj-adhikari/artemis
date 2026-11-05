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

"""LLM & Multimodal Vision Credentials Readiness Probe."""

"""Claude Code CLI readiness, used by the LLM credentials probe."""

import asyncio
import os
import shutil
from typing import Any

from artemis.core.diagnostics.schema import ProbeAction, ProbeResult, ProbeStatus


def _contains_claude_code(node: Any) -> bool:
    if isinstance(node, dict):
        return node.get("provider") == "claude-code" or any(
            _contains_claude_code(v) for v in node.values()
        )
    if isinstance(node, (list, tuple)):
        return any(_contains_claude_code(v) for v in node)
    return False


def config_uses_claude_code() -> bool:
    try:
        from artemis.config.llm import get_default_llm_config

        return _contains_claude_code(get_default_llm_config().model_dump())
    except Exception:
        return False


async def _claude_cli_ready() -> tuple[bool, str]:
    path = shutil.which("claude")
    if not path:
        return False, "claude CLI not found on PATH"
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    proc = await asyncio.create_subprocess_exec(
        path,
        "auth",
        "status",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except TimeoutError:
        proc.kill()
        return False, "claude auth status timed out"
    if proc.returncode != 0:
        return False, "claude CLI is not logged in"
    return True, ""


async def claude_code_result(
    probe: Any, metadata: dict[str, Any], configured_providers: list[dict[str, Any]]
) -> ProbeResult | None:
    """Return the probe result when the config routes to claude-code, else None."""
    if not config_uses_claude_code():
        return None
    ready, reason = await _claude_cli_ready()
    metadata["providers"] = [
        *configured_providers,
        {"provider": "claude-code", "label": "Claude Code"},
    ]
    if ready:
        return ProbeResult(
            id=probe.probe_id,
            category=probe.category,
            title="Multimodal LLM API Key",
            status=ProbeStatus.PASS,
            is_blocker=probe.is_blocker,
            summary="Active (Claude Code)",
            description="Claude Code CLI is logged in and used as the LLM provider.",
            metadata=metadata,
            actions=[
                ProbeAction(
                    action_type="hint",
                    label="Provider Active",
                    payload="Claude Code CLI login is active.",
                )
            ],
        )
    missing = "not found" in reason
    return ProbeResult(
        id=probe.probe_id,
        category=probe.category,
        title="Multimodal LLM API Key",
        status=ProbeStatus.FAIL,
        is_blocker=probe.is_blocker,
        summary="Claude Code Not Ready",
        description=reason,
        metadata=metadata,
        actions=[
            ProbeAction(
                action_type="link" if missing else "command",
                label="Install Claude Code" if missing else "Log in to Claude Code",
                payload="https://docs.claude.com/en/docs/claude-code"
                if missing
                else "claude auth login",
            )
        ],
    )
