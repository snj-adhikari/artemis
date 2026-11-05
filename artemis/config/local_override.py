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

"""Gitignored partial LLM config merged over the tracked artemis.jsonc."""

import os
from pathlib import Path
from typing import Any

from artemis.config import paths
from third_party.mobile_use.utils.file import load_jsonc


def local_override_path() -> Path:
    return Path(os.getenv("ARTEMIS_LOCAL_JSONC") or paths.CONFIG_DIR / "artemis.local.jsonc")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_local_override(config: dict[str, Any]) -> dict[str, Any]:
    path = local_override_path()
    if not path.is_file():
        return config
    with open(path, encoding="utf-8") as f:
        return _merge(config, load_jsonc(f))
