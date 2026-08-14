from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.inference import llama_server_catalog as catalog_module
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_server_catalog import (
    _PROBE_CACHE_MAX_ENTRIES,
    _probe_cache,
    _probe_retry_after,
    capability_policy_gaps,
    clear_llama_server_help_cache,
    get_llama_server_argument_catalog,
    parse_llama_server_help,
    probe_llama_server_help,
)
from core.inference.llama_server_args import (
    declared_exact_aliases,
    flag_policy,
    resolve_flag_alias,
    safe_flag_policy,
    scrub_llama_server_env,
    validate_extra_args,
)


def test_arguments_endpoint_is_studio_only():
    from auth.authentication import get_current_subject
    from routes import inference as inference_routes

    path = "/llama-server/arguments"
    assert path in [route.path for route in inference_routes.studio_router.routes]
    assert path not in [route.path for route in inference_routes.router.routes]
    route = next(
        route for route in inference_routes.studio_router.routes if route.path == path
    )
    assert get_current_subject in [dependency.call for dependency in route.dependant.dependencies]


HELP_FIXTURE = """\
----- common params -----
-h, --help                             show this help
-c, --ctx-size N                       context size in tokens (default: 4096)
                                       env: LLAMA_ARG_CTX_SIZE
--flash-attn [on|off|auto]             flash attention mode
--spec-type TYPE                       speculative decoding type
                                       (none,draft-mtp,ngram-mod)
--legacy-mode VALUE                    deprecated compatibility selector
--removed-mode VALUE                   argument has been removed
----- server params -----
--host HOST                            bind address (default: 127.0.0.1)
--fit-target {model,context}            fitting target, default: model (env: LLAMA_ARG_FIT_TARGET)
"""


# Exact declaration blocks captured from the installed b-build used during the
# review. Fixtures make parser tests independent of that binary after capture.
INSTALLED_DECLARATIONS_FIXTURE = """\
-fitt, --fit-target MiB0,MiB1,MiB2,...
                                      target margin per device for --fit, comma-separated list of values,
                                      single value is broadcast across all devices, default: 1024
--control-vector-scaled FNAME:SCALE,...
                                      add a control vector with user defined scaling SCALE
-ot, --override-tensor <tensor name pattern>=<buffer type>,...
                                      override tensor buffer type
-dev,  --device <dev1,dev2,..>       comma-separated list of devices to use for offloading (none = don't offload)
--spec-draft-device, -devd, --device-draft <dev1,dev2,..>
                                      comma-separated list of devices to use for offloading the draft model
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
                                      comma-separated list of types of speculative decoding to use (default: none)
--ui-config, --webui-config JSON      JSON that provides default UI settings
--ui-config-file, --webui-config-file PATH
                                      JSON file that provides default UI settings
--ui-mcp-proxy, --webui-mcp-proxy, --no-ui-mcp-proxy, --no-webui-mcp-proxy
                                      whether to enable MCP CORS proxy
-h,    --help, --usage                print usage and exit
--version                             show version and build info
--list-devices                        print list of available devices and exit
-cl,   --cache-list                   show list of models in cache
--completion-bash                     print source-able bash completion script
"""


@pytest.fixture(autouse = True)
def _clear_probe_caches():
    clear_llama_server_help_cache()
    with LlamaCppBackend._capability_cache_lock:
        LlamaCppBackend._capability_cache.clear()
        LlamaCppBackend._capability_retry_after.clear()
    yield
    clear_llama_server_help_cache()


def _argument(arguments, name):
    return next(argument for argument in arguments if argument.name == name)


def test_multiline_catalog_parser_extracts_metadata_and_omits_removed_stubs():
    arguments = parse_llama_server_help(HELP_FIXTURE)

    ctx = _argument(arguments, "--ctx-size")
    assert ctx.aliases == ("-c",)
    assert ctx.value_hint == "N"
    assert ctx.group == "common params"
    assert ctx.default_value == "4096"
    assert ctx.env_var == "LLAMA_ARG_CTX_SIZE"
    assert "context size in tokens" in ctx.description

    flash = _argument(arguments, "--flash-attn")
    assert flash.value_hint == "[on|off|auto]"
    assert flash.choices == ("on", "off", "auto")

    spec = _argument(arguments, "--spec-type")
    assert spec.choices == ("none", "draft-mtp", "ngram-mod")
    assert _argument(arguments, "--legacy-mode").deprecated is True
    assert _argument(arguments, "--host").group == "server params"
    fit_target = _argument(arguments, "--fit-target")
    assert fit_target.choices == ("model", "context")
    assert fit_target.default_value == "model"
    assert fit_target.env_var == "LLAMA_ARG_FIT_TARGET"
    assert not any(argument.name == "--removed-mode" for argument in arguments)


def test_installed_declaration_shapes_are_value_aware_without_fake_device_choices():
    arguments = parse_llama_server_help(INSTALLED_DECLARATIONS_FIXTURE)

    fit_target = _argument(arguments, "--fit-target")
    assert fit_target.value_hint == "MiB0,MiB1,MiB2,..."
    assert fit_target.choices == ()

    control = _argument(arguments, "--control-vector-scaled")
    assert control.value_hint == "FNAME:SCALE,..."
    assert control.choices == ()

    override_tensor = _argument(arguments, "--override-tensor")
    assert override_tensor.aliases == ("-ot",)
    assert override_tensor.value_hint == "<tensor name pattern>=<buffer type>,..."
    assert override_tensor.choices == ()
    assert override_tensor.description == "override tensor buffer type"

    device = _argument(arguments, "--device")
    assert device.value_hint == "<dev1,dev2,..>"
    assert device.choices == ()
    assert _argument(arguments, "--spec-draft-device").choices == ()

    spec = _argument(arguments, "--spec-type")
    assert spec.value_hint.startswith("none,draft-simple")
    assert spec.choices == (
        "none",
        "draft-simple",
        "draft-eagle3",
        "draft-mtp",
        "draft-dflash",
        "draft-dspark",
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
        "ngram-cache",
    )


def test_public_catalog_adds_authoritative_managed_and_overlap_classification(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")

    payload = get_llama_server_argument_catalog(
        str(binary),
        run = lambda *_args, **_kwargs: SimpleNamespace(
            returncode = 0, stdout = HELP_FIXTURE, stderr = ""
        ),
    )

    assert payload["available"] is True
    assert payload["authoritative"] is True
    assert payload["error_code"] is None
    host = next(item for item in payload["arguments"] if item["name"] == "--host")
    ctx = next(item for item in payload["arguments"] if item["name"] == "--ctx-size")
    assert host["managed_by_studio"] is True
    assert host["overlaps_studio_control"] is False
    assert ctx["managed_by_studio"] is False
    assert ctx["overlaps_studio_control"] is True
    assert host["policy_category"] == "Routing/listening"
    assert ctx["policy_category"] == "Unclassified"
    assert host["value_arity"] == 1


def test_capability_audit_detects_a_new_unclassified_surface():
    arguments = parse_llama_server_help(
        "--tools-future-runtime TARGET  run tools on another process\n"
    )
    assert capability_policy_gaps(arguments) == ("--tools-future-runtime",)


def test_capability_audit_covers_representative_installed_declarations():
    arguments = parse_llama_server_help(
        """\
--rpc SERVERS                         connect to RPC servers
--lora FNAME                          path to LoRA adapter
--grammar-file FNAME                  file to read grammar from
--lookup-cache-dynamic FNAME          path to dynamic lookup cache
--tools TOOL1,TOOL2                   enable built-in tools
--props                               enable POST /props
--log-prompts-dir PATH                log prompts to a directory
--fim-qwen-1.5b-default               use a default model (note: can download weights from the internet)
"""
    )
    assert capability_policy_gaps(arguments) == ()
    assert _argument(arguments, "--fim-qwen-1.5b-default").name == "--fim-qwen-1.5b-default"


def test_installed_llama_server_help_has_no_current_capability_policy_gap():
    binary = Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    if not binary.is_file():
        pytest.skip("installed llama-server is unavailable")
    probe = probe_llama_server_help(str(binary))
    assert probe.available is True
    assert capability_policy_gaps(probe.arguments) == ()

    installed_multi_aliases = {
        spelling
        for argument in probe.arguments
        for spelling in (argument.name, *argument.aliases)
        if spelling.startswith("-") and not spelling.startswith("--") and len(spelling) > 2
    }
    required_exact_aliases = {"-cmoe", "-cmoed", "-ndio", "-no-mmap", "-nkvo", "-no-kvu"}
    # This installed revision spells no-mmap as ``--no-mmap``; the historical
    # single-dash spelling remains declared so exact-first parsing cannot let
    # ``-m`` steal it when older binaries/settings are encountered.
    assert required_exact_aliases - {"-no-mmap"} <= installed_multi_aliases
    assert required_exact_aliases <= set(declared_exact_aliases())
    assert installed_multi_aliases <= set(declared_exact_aliases())
    assert all(resolve_flag_alias(alias) == alias for alias in installed_multi_aliases)
    assert resolve_flag_alias("-c4096") == "-c"
    assert resolve_flag_alias("-mg0") == "-mg"

    # Required-value syntax is checked in the same central policy as blocked
    # aliases. Bracketed help values are optional switches, not arity gaps.
    missing_safe_arity = [
        argument.name
        for argument in probe.arguments
        if flag_policy(argument.name) is None
        and argument.value_hint
        and not argument.value_hint.startswith("[")
        and safe_flag_policy(argument.name) is None
    ]
    assert missing_safe_arity == []
    for argument in probe.arguments:
        blocked = flag_policy(argument.name)
        if blocked is not None:
            assert all(
                flag_policy(spelling) == blocked
                for spelling in (argument.name, *argument.aliases)
            )
        safe = safe_flag_policy(argument.name)
        if safe is not None:
            assert all(
                safe_flag_policy(spelling) == safe
                for spelling in (argument.name, *argument.aliases)
            )

    documented = {argument.name for argument in probe.arguments}
    assert {"--top-k", "--rpc", "--lora", "--log-prompts-dir"} <= documented
    assert validate_extra_args(["--top-k", "20"]) == ["--top-k", "20"]
    smoke_env = dict(os.environ)
    scrub_llama_server_env(smoke_env)
    safe_smoke = subprocess.run(
        [str(binary), "--top-k", "20", "--version"],
        capture_output = True,
        text = True,
        timeout = 20,
        check = False,
        env = smoke_env,
    )
    assert safe_smoke.returncode == 0
    assert "version:" in (safe_smoke.stdout + safe_smoke.stderr).lower()
    for args in (
        ["--rpc", "127.0.0.1:5000"],
        ["--lora", "private.gguf"],
        ["--log-prompts-dir", "private-dir"],
    ):
        with pytest.raises(ValueError):
            validate_extra_args(args)


def test_probe_reports_installed_tag_without_exposing_failure_details(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    monkeypatch.setattr(
        "core.inference.llama_server_catalog._installed_tag", lambda _path: "b12345-mix-deadbee"
    )

    probe = probe_llama_server_help(
        str(binary),
        run = lambda *_args, **_kwargs: SimpleNamespace(
            returncode = 7, stdout = "", stderr = "secret local path"
        ),
    )

    assert probe.available is False
    assert probe.error_code == "probe_failed"
    assert probe.installed_tag == "b12345-mix-deadbee"
    public = probe.as_public_catalog()
    assert public["available"] is False
    assert public["authoritative"] is False
    assert public["installed_tag"] == "b12345-mix-deadbee"
    assert public["error_code"] == "probe_failed"
    assert public["arguments"] == []
    assert "--host" in public["managed_flags"]
    assert ["--host"] in public["managed_flag_groups"]
    assert "secret" not in repr(public)


def test_missing_binary_and_timeout_have_sanitized_categories(tmp_path):
    missing = probe_llama_server_help(str(tmp_path / "missing"))
    assert missing.error_code == "not_installed"

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("llama-server", 10, output = "private")

    timed_out = probe_llama_server_help(str(binary), run = timeout)
    assert timed_out.available is False
    assert timed_out.error_code == "probe_timeout"
    assert timed_out.as_public_catalog()["arguments"] == []


def test_help_probe_scrubs_ambient_llama_args_and_auth(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    monkeypatch.setenv("LLAMA_ARG_FUTURE_CAPABILITY", "on")
    monkeypatch.setenv("LLAMA_API_KEY", "secret")
    monkeypatch.setenv("PATH", "kept-path")
    captured = {}

    def run(*_args, **kwargs):
        captured.update(kwargs["env"])
        return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")

    assert probe_llama_server_help(str(binary), run = run).available is True
    assert "LLAMA_ARG_FUTURE_CAPABILITY" not in captured
    assert "LLAMA_API_KEY" not in captured
    assert captured["PATH"] == "kept-path"


def test_unrecognized_help_and_failed_probe_are_cached_briefly(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    calls = []

    def run(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(returncode = 0, stdout = "llama-server usage", stderr = "")

    first = probe_llama_server_help(str(binary), run = run)
    second = probe_llama_server_help(str(binary), run = run)
    assert first.error_code == "unrecognized_help"
    assert second is first
    assert len(calls) == 1


def test_probe_cache_is_shared_and_binary_replacement_invalidates_it(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"old")
    calls = []

    def run(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")

    first = probe_llama_server_help(str(binary), run = run)
    second = probe_llama_server_help(str(binary), run = run)
    assert first is second
    assert len(calls) == 1

    previous_ns = binary.stat().st_mtime_ns
    binary.write_bytes(b"new binary revision")
    os.utime(binary, ns = (previous_ns + 1_000_000, previous_ns + 1_000_000))
    replaced = probe_llama_server_help(str(binary), run = run)
    assert replaced.fingerprint != first.fingerprint
    assert len(calls) == 2


def test_same_size_timestamp_preserving_replacement_invalidates_content_identity(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"AAAA")
    metadata = binary.stat()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")

    first = probe_llama_server_help(str(binary), run = run)
    binary.write_bytes(b"BBBB")
    os.utime(binary, ns = (metadata.st_atime_ns, metadata.st_mtime_ns))
    second = probe_llama_server_help(str(binary), run = run)

    assert first.fingerprint[:3] == second.fingerprint[:3]
    assert first.fingerprint[3] != second.fingerprint[3]
    assert len(calls) == 2


def test_nonzero_process_with_structural_help_remains_available(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    probe = probe_llama_server_help(
        str(binary),
        run = lambda *_args, **_kwargs: SimpleNamespace(
            returncode = 9, stdout = INSTALLED_DECLARATIONS_FIXTURE, stderr = "device warning"
        ),
    )
    assert probe.available is True
    assert probe.returncode == 9
    assert probe.error_code == "probe_nonzero"
    assert probe.authoritative is False
    assert probe.as_public_catalog()["authoritative"] is False
    assert _argument(probe.arguments, "--spec-type").choices


def test_partial_help_is_retried_until_the_probe_becomes_authoritative(
    tmp_path, monkeypatch
):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    now = [100.0]
    calls = []
    monkeypatch.setattr(catalog_module.time, "monotonic", lambda: now[0])

    def run(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(
            returncode = 9 if len(calls) == 1 else 0,
            stdout = INSTALLED_DECLARATIONS_FIXTURE,
            stderr = "device warning" if len(calls) == 1 else "",
        )

    first = probe_llama_server_help(str(binary), run = run)
    assert first.available is True
    assert first.authoritative is False
    assert probe_llama_server_help(str(binary), run = run) is first
    assert len(calls) == 1

    now[0] += catalog_module._PROBE_RETRY_SECONDS + 1
    refreshed = probe_llama_server_help(str(binary), run = run)
    assert refreshed.authoritative is True
    assert len(calls) == 2


def test_same_fingerprint_concurrent_callers_share_one_subprocess(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        first = pool.submit(probe_llama_server_help, str(binary), run = run)
        assert entered.wait(5)
        second = pool.submit(probe_llama_server_help, str(binary), run = run)
        release.set()
        assert first.result(timeout = 5).available is True
        assert second.result(timeout = 5).available is True
    assert len(calls) == 1


def test_different_fingerprints_probe_without_global_subprocess_lock(tmp_path):
    binaries = [tmp_path / "llama-server-a", tmp_path / "llama-server-b"]
    for index, binary in enumerate(binaries):
        binary.write_bytes(f"fake-{index}".encode())
    barrier = threading.Barrier(2)

    def run(*_args, **_kwargs):
        barrier.wait(timeout = 5)
        return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        results = list(
            pool.map(lambda path: probe_llama_server_help(str(path), run = run), binaries)
        )
    assert all(result.available for result in results)


def test_cache_and_retry_retention_stay_bounded_across_revisions(tmp_path):
    binary = tmp_path / "llama-server"
    for revision in range(_PROBE_CACHE_MAX_ENTRIES + 3):
        binary.write_bytes(bytes([revision]) * 4)
        probe_llama_server_help(
            str(binary),
            run = lambda *_args, **_kwargs: SimpleNamespace(
                returncode = 0, stdout = HELP_FIXTURE, stderr = ""
            ),
        )
    assert len(_probe_cache) <= _PROBE_CACHE_MAX_ENTRIES
    assert set(_probe_retry_after).issubset(_probe_cache)


def test_failed_single_flight_cleans_up_and_later_retry_can_succeed(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    now = [100.0]
    monkeypatch.setattr(catalog_module.time, "monotonic", lambda: now[0])
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            raise RuntimeError("synthetic probe failure")
        return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        first = pool.submit(probe_llama_server_help, str(binary), run = run)
        assert entered.wait(5)
        second = pool.submit(probe_llama_server_help, str(binary), run = run)
        release.set()
        assert first.result(timeout = 5).available is False
        assert second.result(timeout = 5).available is False
    assert len(calls) == 1

    now[0] += catalog_module._PROBE_RETRY_SECONDS + 1
    assert probe_llama_server_help(str(binary), run = run).available is True
    assert len(calls) == 2


def test_capability_cache_uses_content_identity_and_stays_bounded(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"AAAA")
    metadata = binary.stat()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(1)
        content = binary.read_bytes()
        spec = "none,draft-mtp" if content == b"AAAA" else "none,draft-mtp,draft-dflash"
        return SimpleNamespace(
            returncode = 0,
            stdout = f"--spec-type {spec}\n  speculative decoding type\n",
            stderr = "",
        )

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", run)
    monkeypatch.setattr(
        LlamaCppBackend, "_llama_server_env_for_binary", classmethod(lambda cls, _binary: {})
    )
    first = LlamaCppBackend.probe_server_capabilities(str(binary))
    assert first["supports_dflash"] is False

    for revision in range(LlamaCppBackend._CAPABILITY_CACHE_MAX_ENTRIES + 2):
        content = b"BBBB" if revision == 0 else bytes([revision + 1]) * 4
        binary.write_bytes(content)
        os.utime(binary, ns = (metadata.st_atime_ns, metadata.st_mtime_ns))
        latest = LlamaCppBackend.probe_server_capabilities(str(binary))

    assert len(calls) == LlamaCppBackend._CAPABILITY_CACHE_MAX_ENTRIES + 3
    assert len(LlamaCppBackend._capability_cache) <= LlamaCppBackend._CAPABILITY_CACHE_MAX_ENTRIES
    assert set(LlamaCppBackend._capability_retry_after).issubset(
        LlamaCppBackend._capability_cache
    )
    # The first same-metadata replacement carried DFlash and was freshly parsed.
    binary.write_bytes(b"BBBB")
    os.utime(binary, ns = (metadata.st_atime_ns, metadata.st_mtime_ns))
    assert LlamaCppBackend.probe_server_capabilities(str(binary))["supports_dflash"] is True


def test_stale_inflight_revision_cannot_replace_newer_current_revision(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"AAAA")
    metadata = binary.stat()
    old_entered = threading.Event()
    release_old = threading.Event()
    calls = []
    fresh_help = HELP_FIXTURE + "\n--fresh-flag VALUE  newest revision\n"

    def run(*_args, **_kwargs):
        call_id = len(calls)
        calls.append(call_id)
        if call_id == 0:
            old_entered.set()
            assert release_old.wait(5)
            return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")
        return SimpleNamespace(returncode = 0, stdout = fresh_help, stderr = "")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        stale_future = pool.submit(probe_llama_server_help, str(binary), run = run)
        assert old_entered.wait(5)
        binary.write_bytes(b"BBBB")
        os.utime(binary, ns = (metadata.st_atime_ns, metadata.st_mtime_ns))
        fresh = probe_llama_server_help(str(binary), run = run)
        release_old.set()
        stale = stale_future.result(timeout = 5)

    assert "--fresh-flag" not in {argument.name for argument in stale.arguments}
    assert "--fresh-flag" in {argument.name for argument in fresh.arguments}
    current = probe_llama_server_help(str(binary), run = run)
    assert current is fresh
    assert len(calls) == 2


def test_capabilities_and_catalog_share_one_help_execution(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    calls = []

    def run(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(returncode = 0, stdout = HELP_FIXTURE, stderr = "")

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", run)
    monkeypatch.setattr(
        LlamaCppBackend, "_llama_server_env_for_binary", classmethod(lambda cls, _binary: {})
    )

    catalog = LlamaCppBackend.get_server_argument_catalog(str(binary))
    capabilities = LlamaCppBackend.probe_server_capabilities(str(binary))

    assert catalog["available"] is True
    assert capabilities["found"] is True
    assert capabilities["supports_mtp"] is True
    assert len(calls) == 1
    capability_policy_gaps,
