from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from artemis.core.diagnostics.probes import credentials_probe as cp
from artemis.core.diagnostics.schema import ProbeStatus

MOD = "artemis.core.diagnostics.probes.credentials_probe"
KEY = SecretStr("AIzaSyFAKEKEYFAKEKEY1234")


def _proc(code):
    proc = MagicMock()
    proc.returncode = code
    proc.wait = AsyncMock(return_value=code)
    return proc


@pytest.mark.asyncio
async def test_cli_ready_when_auth_status_exits_zero():
    with (
        patch(f"{MOD}.shutil.which", return_value="/bin/claude"),
        patch(f"{MOD}.asyncio.create_subprocess_exec", AsyncMock(return_value=_proc(0))) as spawn,
        patch.dict("os.environ", {"CLAUDECODE": "1"}),
    ):
        assert await cp._claude_cli_ready() == (True, "")
    assert "CLAUDECODE" not in spawn.call_args.kwargs["env"]


@pytest.mark.asyncio
async def test_cli_not_on_path():
    with patch(f"{MOD}.shutil.which", return_value=None):
        ready, reason = await cp._claude_cli_ready()
    assert not ready and "not found" in reason


@pytest.mark.asyncio
async def test_cli_logged_out():
    with (
        patch(f"{MOD}.shutil.which", return_value="/bin/claude"),
        patch(f"{MOD}.asyncio.create_subprocess_exec", AsyncMock(return_value=_proc(1))),
    ):
        ready, _ = await cp._claude_cli_ready()
    assert not ready


@pytest.mark.asyncio
async def test_probe_passes_with_claude_code_despite_google_key():
    with (
        patch(f"{MOD}._config_uses_claude_code", return_value=True),
        patch(f"{MOD}._claude_cli_ready", AsyncMock(return_value=(True, ""))),
        patch("artemis.config.settings.Settings.get_api_key", return_value=KEY),
    ):
        result = await cp.LLMCredentialsProbe().probe()
    assert result.status == ProbeStatus.PASS
    assert result.summary == "Active (Claude Code)"


@pytest.mark.asyncio
async def test_probe_fails_with_login_action_when_not_ready():
    with (
        patch(f"{MOD}._config_uses_claude_code", return_value=True),
        patch(
            f"{MOD}._claude_cli_ready",
            AsyncMock(return_value=(False, "claude CLI is not logged in")),
        ),
    ):
        result = await cp.LLMCredentialsProbe().probe()
    assert result.status == ProbeStatus.FAIL
    assert result.actions[0].payload == "claude auth login"


@pytest.mark.asyncio
async def test_probe_gemini_path_unchanged_without_claude_code():
    with (
        patch(f"{MOD}._config_uses_claude_code", return_value=False),
        patch("artemis.config.settings.Settings.get_api_key", return_value=KEY),
    ):
        result = await cp.LLMCredentialsProbe().probe()
    assert result.summary == "Active (Gemini)"


def test_config_scan():
    assert cp._contains_claude_code({"a": [{"provider": "claude-code"}]})
    assert not cp._contains_claude_code({"a": {"provider": "google"}})
