# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The load contract the two image engines share, and the native engine's offline half.

``POST /images/load`` calls ``begin_load`` through ONE call site for whichever engine was
activated, so every keyword it passes has to be accepted by both. That is not a style rule: the
native engine is what a CPU-only host, an opted-in MPS host and ``UNSLOTH_DIFFUSION_ENGINE=sd_cpp``
select, so a keyword only the diffusers engine declares TypeErrors every single load on those
hosts -- including the ordinary user-initiated ones from the Images page, which pass the flag's
default. ``local_files_only`` shipped exactly that way.

The engine doubles here are ``create_autospec`` mocks on purpose. A hand-written fake with
``**kwargs`` accepts anything, which is why the existing route tests passed against an engine that
could not be called at all; autospec binds against the real signature and raises the TypeError the
user would have seen.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
import threading
import types
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from core.inference.diffusion import DiffusionBackend
from core.inference.sd_cpp_backend import SdCppDiffusionBackend


# ── What the route actually passes ─────────────────────────────────────────


def _route_begin_load_keywords() -> list[str]:
    """The keyword names ``_start_engine_load`` hands ``engine.begin_load``, read off the route.

    Parsed rather than duplicated so this test cannot drift: the next keyword added to that call
    is covered the moment it is added, which is the whole failure mode here.
    """
    import routes.inference as route_module

    source = textwrap.dedent(inspect.getsource(route_module.load_diffusion_model_gated))
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_start_engine_load"):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "begin_load"
            ):
                # ``**kwargs`` would arrive as a None-named keyword; the route spells every one out.
                return [kw.arg for kw in call.keywords if kw.arg is not None]
    raise AssertionError("_start_engine_load no longer calls engine.begin_load")


def test_the_route_still_passes_local_files_only():
    # Guards the guard: if the route stopped passing it, every assertion below would still hold
    # while the offline promise had quietly gone.
    assert "local_files_only" in _route_begin_load_keywords()


@pytest.mark.parametrize("engine", [DiffusionBackend, SdCppDiffusionBackend])
def test_both_engines_accept_every_keyword_the_route_passes(engine):
    """``inspect.signature().bind`` is the exact check the interpreter makes at call time."""
    keywords = _route_begin_load_keywords()
    signature = inspect.signature(engine.begin_load)
    # Bound against the UNBOUND function, so ``self`` is just the first positional and no engine
    # has to be constructed. bind checks names and arity, never values.
    signature.bind(
        None,
        "unsloth/FLUX.1-dev-GGUF",
        **{name: None for name in keywords},
    )


def test_the_two_begin_load_signatures_declare_local_files_only_alike():
    """Same name, same keyword-only kind, same default on both engines.

    A native ``**kwargs`` catch-all would satisfy the bind test above while silently DROPPING the
    flag, so the shape is asserted, not just the acceptance.
    """
    params = {
        engine: inspect.signature(engine.begin_load).parameters
        for engine in (DiffusionBackend, SdCppDiffusionBackend)
    }
    for engine, parameters in params.items():
        assert "local_files_only" in parameters, engine
        declared = parameters["local_files_only"]
        assert declared.kind is inspect.Parameter.KEYWORD_ONLY, engine
        assert declared.default is False, engine
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params[SdCppDiffusionBackend].values()
    ), "a **kwargs catch-all would accept the flag and ignore it"


# ── The route, driven onto the native engine ───────────────────────────────


def _drive_the_images_load(monkeypatch, *, user_initiated: bool):
    """Run ``POST /images/load``'s body with the NATIVE engine selected; return the mock engine.

    Autospec'd off the real class, so the call the route makes is bound against the real
    ``begin_load`` signature: this is what turns the shipped TypeError into a test failure.
    """
    import core.inference.diffusion_device as device_module
    import core.inference.diffusion_engine_router as router_module
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP
    from models.inference import DiffusionLoadRequest
    from routes.inference import load_diffusion_model_gated

    engine = create_autospec(SdCppDiffusionBackend, instance = True)
    engine.begin_load.return_value = {"loaded": False, "repo_id": None}
    engine.preflight_base_access.return_value = None

    monkeypatch.setattr(router_module, "predict_engine", lambda *a, **k: ENGINE_SD_CPP)
    monkeypatch.setattr(router_module, "active_engine_name", lambda: ENGINE_SD_CPP)
    monkeypatch.setattr(router_module, "engine_for", lambda name: engine)
    monkeypatch.setattr(router_module, "select_and_activate_engine", lambda *a, **k: engine)
    monkeypatch.setattr(router_module, "begin_load_on", lambda _engine, start: start())
    monkeypatch.setattr(router_module, "annotate_status", lambda status: status)
    # A CPU-only host is where the native engine is selected in the first place.
    monkeypatch.setattr(
        device_module,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(device = "cpu"),
    )
    monkeypatch.setattr("routes.inference._guard_diffusion_load_against_training", lambda: None)

    async def _no_ordinal(_gpu_ids):
        return None

    monkeypatch.setattr("routes.inference._selected_gpu_ordinal", _no_ordinal)

    asyncio.run(
        load_diffusion_model_gated(
            DiffusionLoadRequest(
                model_path = "unsloth/FLUX.1-dev-GGUF",
                gguf_filename = "flux1-dev-Q4_K_M.gguf",
            ),
            "test-user",
            user_initiated = user_initiated,
        )
    )
    return engine


@pytest.mark.parametrize("user_initiated", [True, False])
def test_the_images_page_can_load_on_the_native_engine(monkeypatch, user_initiated):
    # The regression: this raised TypeError for BOTH values, so the Images page could not load a
    # model at all on any host that selects sd.cpp. The parametrisation keeps the user-initiated
    # case explicit, because that is the one nobody expects an offline flag to break.
    engine = _drive_the_images_load(monkeypatch, user_initiated = user_initiated)

    engine.begin_load.assert_called_once()
    assert engine.begin_load.call_args.kwargs["local_files_only"] is (not user_initiated)


# ── The native loader honours it ───────────────────────────────────────────


def _no_hub(monkeypatch):
    """Make every huggingface_hub API call this load could reach an outright failure."""
    import huggingface_hub

    def _forbidden(*_a, **_k):
        raise AssertionError("a cache-only load reached the Hub")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _forbidden)
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", _forbidden)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", _forbidden)


def test_a_cache_only_native_load_makes_no_hub_call(monkeypatch):
    """The size probe and the companion preflight are both pure network; neither may run.

    Their failure mode is quiet -- ``_set_expected_bytes`` swallows everything and the preflight
    fails open -- so an unguarded call would not fail the load, it would just download.
    """
    from core.inference.diffusion_families import detect_family
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend as Native

    _no_hub(monkeypatch)
    backend = Native(engine = None)
    monkeypatch.setattr(
        Native,
        "_resolve_backend",
        lambda self: ("oneshot", None, types.SimpleNamespace(version = lambda: "master")),
    )
    fetched: list = []

    def _fetch(
        self,
        assets,
        token,
        cancel_event = None,
        local_files_only = False,
    ):
        fetched.append(local_files_only)
        raise RuntimeError("stop here; the Hub calls under test all precede the fetch")

    monkeypatch.setattr(Native, "_fetch_assets", _fetch)

    repo = "unsloth/FLUX.1-dev-GGUF"
    Native._run_load(
        backend,
        repo_id = repo,
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        base = "black-forest-labs/FLUX.1-dev",
        fam = detect_family(repo),
        hf_token = None,
        local_files_only = True,
        _load_token = 1,
    )

    # Reached the fetch (so the probe and preflight were skipped, not merely tolerated) and the
    # flag arrived there, which is the only call that can still pull bytes.
    assert fetched == [True]


def test_the_native_fetch_resolves_from_cache_only(monkeypatch, tmp_path):
    """``local_files_only`` reaches huggingface_hub, where it is the only thing that stops a pull."""
    import utils.hf_xet_fallback as xet
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend as Native

    seen: list = []
    cached = tmp_path / "flux1-dev-Q4_K_M.gguf"
    cached.write_bytes(b"")

    def _download(repo_id, filename, token, **kwargs):
        seen.append((repo_id, filename, kwargs.get("local_files_only")))
        return str(cached)

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)

    Native(engine = None)._fetch_assets(
        [("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", "diffusion_model")],
        None,
        local_files_only = True,
    )

    assert seen == [("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", True)]


def test_an_uncached_asset_fails_with_a_local_error_naming_it(monkeypatch):
    """The miss must READ as a miss. huggingface_hub's own text names neither repo nor file, and
    this string is what /images/load-progress puts in front of the user."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    import utils.hf_xet_fallback as xet
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend as Native

    def _download(*_a, **_k):
        raise LocalEntryNotFoundError("Cannot find the requested files in the disk cache")

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)

    with pytest.raises(RuntimeError) as caught:
        Native(engine = None)._fetch_assets(
            [("black-forest-labs/FLUX.1-dev", "ae.safetensors", "vae")],
            None,
            local_files_only = True,
        )
    message = str(caught.value)
    assert "ae.safetensors" in message
    # The FETCH repo, which is where the bytes were looked for: the gated vendor base is swapped
    # to its ungated mirror before the lookup, so naming the upstream id would misdirect.
    assert "unsloth/FLUX.1-dev" in message


def test_the_default_still_takes_the_xet_fallback_ladder(monkeypatch, tmp_path):
    """Nothing changes with the flag off: the shared Xet -> HTTP path is still the one used, and
    ``local_files_only`` is not forwarded to a shared layer that may predate it."""
    import utils.hf_xet_fallback as xet

    seen: list = []

    def _shared(repo_id, filename, token, **kwargs):
        seen.append(kwargs)
        return str(tmp_path / filename)

    monkeypatch.setattr(xet, "_shared_hf_hub_download_with_xet_fallback", _shared)

    xet.hf_hub_download_with_xet_fallback(
        "unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", None, cache_dir = str(tmp_path)
    )

    assert len(seen) == 1
    assert "local_files_only" not in seen[0]


def test_the_offline_download_never_reaches_the_shared_ladder(monkeypatch, tmp_path):
    """And with the flag on it goes straight to huggingface_hub.

    Deliberately NOT forwarded to unsloth_zoo: ``start_watchdog`` already showed that an older
    installed zoo silently drops kwargs it does not declare, and a dropped ``local_files_only``
    downloads -- the one outcome the flag exists to prevent.
    """
    import huggingface_hub

    import utils.hf_xet_fallback as xet

    def _forbidden(*_a, **_k):
        raise AssertionError("the shared Xet ladder must not run for a cache-only download")

    monkeypatch.setattr(xet, "_shared_hf_hub_download_with_xet_fallback", _forbidden)
    seen: list = []

    def _hub(**kwargs):
        seen.append(kwargs)
        return str(tmp_path / "flux1-dev-Q4_K_M.gguf")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hub)

    xet.hf_hub_download_with_xet_fallback(
        "unsloth/FLUX.1-dev-GGUF",
        "flux1-dev-Q4_K_M.gguf",
        None,
        cache_dir = str(tmp_path),
        local_files_only = True,
    )

    assert seen and seen[0]["local_files_only"] is True


def test_a_cancelled_offline_download_still_stops(monkeypatch, tmp_path):
    """The cancellation contract is the ladder's, not huggingface_hub's, so the bypass keeps it."""
    import utils.hf_xet_fallback as xet

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError):
        xet.hf_hub_download_with_xet_fallback(
            "unsloth/FLUX.1-dev-GGUF",
            "flux1-dev-Q4_K_M.gguf",
            None,
            cache_dir = str(tmp_path),
            local_files_only = True,
            cancel_event = cancel,
        )


def test_the_binary_install_is_not_covered_by_the_flag():
    """Stated as a test so the boundary is not re-litigated by accident.

    ``local_files_only`` is about MODEL ASSETS. The sd-cli / sd-server binary lives in a separate
    managed tree with its own install policy, and ``_run_load`` resolves it before any asset is
    fetched; a background load may still install one, exactly as before. If that ever needs to
    change it is a deliberate decision, not a side effect of this flag.
    """
    source = inspect.getsource(SdCppDiffusionBackend._run_load)
    resolve = source.index("self._resolve_backend()")
    fetch = source.index("self._fetch_assets(")
    assert resolve < fetch, "the binary is resolved before the assets; the comment above assumes it"
    assert Path(inspect.getsourcefile(SdCppDiffusionBackend)).name == "sd_cpp_backend.py"
