# SPDX-License-Identifier: AGPL-3.0-only

import inspect

import pytest


@pytest.mark.parametrize("method", ["find_prefix_length", "update"])
def test_vlm_prompt_cache_session_takes_every_upstream_keyword(method):
    # mlx-vlm's stream_generate calls these with whatever keywords its own PromptCacheState takes (0.8.1 added video_timing).
    common = pytest.importorskip("mlx_vlm.generate.common")
    from core.inference.mlx_inference import VLMPromptCacheSession

    upstream = inspect.signature(getattr(common.PromptCacheState, method)).parameters.values()
    ours = inspect.signature(getattr(VLMPromptCacheSession, method))
    keywords = {p.name: p.default for p in upstream if p.kind is inspect.Parameter.KEYWORD_ONLY}
    positional = [None] * sum(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in upstream)
    ours.bind(*positional, **keywords)
