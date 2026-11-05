# Claude Code provider (no API keys)

Runs every LLM role through your logged-in `claude` CLI.

```bash
cp config/artemis.claude-code.example.jsonc config/artemis.local.jsonc   # or ARTEMIS_LOCAL_JSONC=/path/to/file.jsonc
uv run artemis doctor                       # should report Claude Code active
export OBJECT_DETECTOR_TIMEOUT=30
uv run python scripts/smoke_claude_code.py  # optional screenshot path argument
```

The local file is merged over `config/artemis.jsonc`, so it only lists what changes and survives upstream config updates.

* One `claude -p` process per call: expect 2-3s+ extra latency per step.
* Consumes Claude Code plan limits.
* No temperature, thinking or video input; video analysis is disabled in the example.
* Set `OBJECT_DETECTOR_TIMEOUT=30`; the 10s default is too short.
