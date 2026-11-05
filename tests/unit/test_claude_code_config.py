from types import SimpleNamespace
from unittest.mock import patch

import pytest

from artemis.config import paths
from artemis.config.llm import LLMConfig, _expand_default_into_nodes
from artemis.llm.router import ModelEndpoint, ModelProvider
from artemis.services import llm as llm_service
from third_party.mobile_use.utils.file import load_jsonc

EXAMPLE = paths.ROOT_DIR / "config" / "artemis.claude-code.example.jsonc"


def _config_dir(monkeypatch, tmp_path, with_local):
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("ARTEMIS_ARTEMIS_JSONC", raising=False)
    (tmp_path / "artemis.jsonc").write_text("{}")
    if with_local:
        (tmp_path / "artemis.local.jsonc").write_text("{}")


def test_local_override_preferred(monkeypatch, tmp_path):
    _config_dir(monkeypatch, tmp_path, True)
    assert paths.get_config_path("artemis.jsonc") == tmp_path / "artemis.local.jsonc"


def test_default_without_local(monkeypatch, tmp_path):
    _config_dir(monkeypatch, tmp_path, False)
    assert paths.get_config_path("artemis.jsonc") == tmp_path / "artemis.jsonc"


def test_env_var_beats_local(monkeypatch, tmp_path):
    _config_dir(monkeypatch, tmp_path, True)
    env_file = tmp_path / "env.jsonc"
    env_file.write_text("{}")
    monkeypatch.setenv("ARTEMIS_ARTEMIS_JSONC", str(env_file))
    assert paths.get_config_path("artemis.jsonc") == env_file


def test_example_config_all_claude_code():
    try:
        ModelProvider.from_string("claude-code")
    except Exception:
        pytest.skip("ModelProvider.CLAUDE_CODE not present (needs T1)")
    with open(EXAMPLE, encoding="utf-8") as f:
        cfg = LLMConfig.model_validate(_expand_default_into_nodes(load_jsonc(f)))
    dumped = cfg.model_dump()

    def providers(node):
        if isinstance(node, dict):
            if "provider" in node:
                yield node["provider"]
            for v in node.values():
                yield from providers(v)

    found = list(providers(dumped))
    assert found and set(found) == {"claude-code"}
    assert cfg.validator_pixel_safety_net and cfg.planner_validation


@pytest.mark.parametrize(
    "provider,expect_none",
    [
        (ModelProvider.GOOGLE, True),
        (ModelProvider.VERTEX_AI, True),
        (ModelProvider.ANTHROPIC, False),
    ],
)
def test_get_non_google_llm(provider, expect_none):
    ep = ModelEndpoint(provider=provider, model_name="m")
    with (
        patch.object(llm_service, "_resolve_endpoint", return_value=ep),
        patch.object(llm_service, "get_llm", return_value="LLM") as get_llm,
    ):
        result = llm_service.get_non_google_llm(SimpleNamespace(), "summarizer")
    assert (result is None) is expect_none
    assert get_llm.called is (not expect_none)


def test_get_non_google_llm_resolve_failure():
    with patch.object(llm_service, "_resolve_endpoint", side_effect=RuntimeError):
        assert llm_service.get_non_google_llm(SimpleNamespace(), "summarizer") is None
