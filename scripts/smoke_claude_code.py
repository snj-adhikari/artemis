#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Smoke test for the Claude Code provider: plain vision reply and object-detector grounding."""

import asyncio
import base64
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from artemis.agents.object_detector.object_detector import _detect_single_label
from artemis.llm.router import ModelEndpoint, ModelFactory, ModelProvider
from artemis.tools.object_detection_tool import _load_prompt_templates

DEFAULT_IMAGE = "live_screenshot_adb-0008714A9000292-y8s4N0__adb-tls-connect__tcp.jpg"
IMAGE = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE)
LABEL = "close (X) button at the top right"


async def main() -> None:
    llm = ModelFactory.create_model(
        ModelEndpoint(provider=ModelProvider.CLAUDE_CODE, model_name="haiku", timeout_seconds=120)
    )
    data = IMAGE.read_bytes()
    b64 = base64.b64encode(data).decode()

    start = time.monotonic()
    reply = await llm.ainvoke(
        [
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "What is the title at the top of this screen?"},
                ]
            )
        ]
    )
    print(f"vision reply ({time.monotonic() - start:.1f}s): {reply.content!r}")
    assert reply.content, "empty vision reply"

    start = time.monotonic()
    results = await _detect_single_label(
        llm, None, data, "image/jpeg", LABEL, _load_prompt_templates(), timeout_val=120
    )
    print(f"grounding ({time.monotonic() - start:.1f}s): {results}")
    assert results, "no detections"
    for item in results:
        point = item.get("point")
        assert isinstance(point, list) and len(point) == 2, item
        assert all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in point), item
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
