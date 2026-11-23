# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The export page's MoE probe must not follow a Hub adapter to a Windows network share (NTLM)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from core.export import reap as reap_mod
from utils.datasets import remote_images


def _route(monkeypatch, base):
    from routes import export as export_route
    import routes.models as models_route
    from utils.models import model_config

    probed = []

    async def require(model, token):
        return None

    monkeypatch.setattr(models_route, "_require_model_access_or_caller_token", require)
    monkeypatch.setattr(model_config, "get_base_model_from_lora_identifier", lambda m, t = None: base)

    def probe(model, token, local, **kwargs):
        probed.append(model)
        return None if model == "someone/lora" else {"num_experts": 8, "top_k": 2}

    monkeypatch.setattr(reap_mod, "moe_export_info", probe)
    monkeypatch.setattr(reap_mod, "hub_file_exists", lambda repo, name, token = None: False)
    # Windows path rules on this host; no Windows runner here.
    monkeypatch.setattr(remote_images, "_WINDOWS", True)

    def call():
        return asyncio.run(
            export_route.get_moe_info(
                model = "someone/lora", hf_token = None, current_subject = "t", allow_ambient = False
            )
        )

    return call, probed


@pytest.mark.parametrize(
    "base", [r"\\evil.example\share\model", "//evil.example/share/model", r"\\?\UNC\evil\s\m"]
)
def test_hub_adapter_with_unc_base_is_not_probed(monkeypatch, base):
    call, probed = _route(monkeypatch, base)
    with pytest.raises(HTTPException) as info:
        call()
    assert info.value.status_code == 400
    assert "network share" in info.value.detail
    assert probed == ["someone/lora"]


def test_hub_adapter_with_hub_base_still_probes_the_base(monkeypatch):
    call, probed = _route(monkeypatch, "org/moe-base")
    res = call()
    assert (res.num_experts, res.top_k) == (8, 2)
    assert probed == ["someone/lora", "org/moe-base"]
