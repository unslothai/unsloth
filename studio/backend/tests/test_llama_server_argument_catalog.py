from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference.llama_server_catalog import (
    capability_policy_gaps,
    clear_llama_server_help_cache,
    parse_llama_server_help,
    probe_llama_server_help,
)
from core.inference.llama_server_args import (
    declared_exact_aliases,
    resolve_flag_alias,
    safe_flag_category,
    validate_extra_args,
)


HELP = """\
----- common params -----
-h, --help                             show help
-c, --ctx-size N                       context size (default: 4096)
                                       env: LLAMA_ARG_CTX_SIZE
-ot, --override-tensor PATTERN=TYPE    override tensor buffer type
--flash-attn [on|off|auto]             flash attention mode
--direct-io, -dio, -ndio, --no-direct-io
                                       direct model I/O
--legacy-mode VALUE                    deprecated selector
--removed-mode VALUE                   argument has been removed
----- server params -----
--host HOST                            bind address
--rpc SERVERS                          connect to RPC servers
--lora FNAME                           load a LoRA adapter
--tools TOOL1,TOOL2                    enable built-in tools
"""


def _argument(arguments, name):
    return next(argument for argument in arguments if argument.name == name)


def test_catalog_parser_and_public_policy_contract():
    arguments = parse_llama_server_help(HELP)
    assert not any(argument.name == "--removed-mode" for argument in arguments)
    ctx = _argument(arguments, "--ctx-size")
    assert (ctx.aliases, ctx.value_hint, ctx.default_value, ctx.env_var, ctx.group) == (
        ("-c",),
        "N",
        "4096",
        "LLAMA_ARG_CTX_SIZE",
        "common params",
    )
    override = _argument(arguments, "--override-tensor").as_public_dict()
    direct = _argument(arguments, "--direct-io").as_public_dict()
    assert override["aliases"] == ["-ot"]
    assert override["policy_category"] == "Compute/placement"
    assert direct["policy_category"] == "Compute/placement"

    assert _argument(arguments, "--host").as_public_dict()["policy_category"] == (
        "Routing/listening"
    )


def test_policy_audit_blocks_capabilities_but_keeps_unknown_pass_through():
    assert capability_policy_gaps(parse_llama_server_help(HELP)) == ()
    future = parse_llama_server_help("--future-safe VALUE  future compute knob\n")[0]
    assert future.as_public_dict()["policy_category"] == "Unclassified"
    assert validate_extra_args(["--future-safe", "value"]) == ["--future-safe", "value"]
    assert capability_policy_gaps(
        parse_llama_server_help("--tools-future-runtime TARGET  run another process\n")
    ) == ("--tools-future-runtime",)
    for args in (["--rpc", "host:1"], ["--lora", "private.gguf"], ["--tools", "shell"]):
        with pytest.raises(ValueError):
            validate_extra_args(args)


def test_installed_help_alias_and_category_matrix():
    binary = (
        Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    )
    if not binary.is_file():
        pytest.skip("installed llama-server is unavailable")
    probe = probe_llama_server_help(str(binary))
    assert probe.available is True
    assert capability_policy_gaps(probe.arguments) == ()
    exact = set(declared_exact_aliases())
    installed_multi = {
        spelling
        for argument in probe.arguments
        for spelling in (argument.name, *argument.aliases)
        if spelling.startswith("-") and not spelling.startswith("--") and len(spelling) > 2
    }
    assert installed_multi <= exact
    assert all(resolve_flag_alias(alias) == alias for alias in installed_multi)
    assert resolve_flag_alias("-c4096") == "-c"
    assert resolve_flag_alias("-mg0") == "-mg"
    assert safe_flag_category("-ot") == "Compute/placement"
    assert safe_flag_category("--direct-io") == "Compute/placement"
    assert {"--top-k", "--rpc", "--lora"} <= {item.name for item in probe.arguments}


def test_probe_scrubs_env(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake")
    monkeypatch.setenv("LLAMA_ARG_FUTURE", "secret")
    monkeypatch.setenv("LLAMA_API_KEY", "secret")
    monkeypatch.setenv("PATH", "kept")
    seen = {}

    def capture(*_args, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode = 0, stdout = HELP, stderr = "")

    clear_llama_server_help_cache()
    assert probe_llama_server_help(str(binary), run = capture).available is True
    assert seen["PATH"] == "kept"
    assert not any(key.startswith("LLAMA_ARG_") for key in seen)
    assert "LLAMA_API_KEY" not in seen
