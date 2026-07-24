from __future__ import annotations

import importlib


def test_app_import_does_not_require_a_model_file() -> None:
    module = importlib.import_module("app")
    assert callable(module.live_payload)
    assert module.MODEL is None
