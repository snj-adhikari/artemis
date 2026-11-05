from types import SimpleNamespace
from unittest.mock import patch

import pytest

from artemis.config import paths
from artemis.config.local_override import apply_local_override
from artemis.config.llm import LLMConfig, _expand_default_into_nodes
from artemis.llm.router import ModelEndpoint, ModelProvider
from artemis.services import llm as llm_service
from third_party.mobile_use.utils.file import load_jsonc

EXAMPLE = paths.ROOT_DIR / "config" / "artemis.claude-code.example.jsonc"


def test_local_override_merges_partial(monkeypatch, tmp_path):
    local = tmp_path / "local.jsonc"
    local.write_text('{"planner": {"model": "sonnet"}}')
    monkeypatch.setenv("ARTEMIS_LOCAL_JSONC", str(local))
    base = {"planner": {"provider": "google", "model": "g"}, "operator": {"model": "o"}}
    assert apply_local_override(base) == {
        "planner": {"provider": "google", "model": "sonnet"},
        "operator": {"model": "o"},
    }


def test_no_local_override_returns_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTEMIS_LOCAL_JSONC", str(tmp_path / "missing.jsonc"))
    assert apply_local_override({"a": 1}) == {"a": 1}


def test_example_routes_every_role_to_claude_code(monkeypatch):
    monkeypatch.setenv("ARTEMIS_LOCAL_JSONC", str(EXAMPLE))
    with open(paths.ROOT_DIR / "config" / "artemis.jsonc", encoding="utf-8") as f:
        expanded = apply_local_override(_expand_default_into_nodes(load_jsonc(f)))
    cfg = LLMConfig.model_validate(expanded)

    def providers(node):
        if isinstance(node, dict):
            if "provider" in node:
                yield node["provider"]
            for v in node.values():
                yield from providers(v)

    found = set(providers(cfg.model_dump()))
    assert "claude-code" in found
    assert cfg.planner.provider == "claude-code"
    assert cfg.validator_pixel_safety_net and cfg.planner_validation
    return found


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
