"""The llama.cpp source-build fallback ships ggml-rpc-server (setup.sh / setup.ps1).

The prebuilt bundles install the RPC server (install_llama_prebuilt.py
runtime_patterns_for_choice), but when the installer falls back to building llama.cpp
from source the scripts configured without GGML_RPC and never built the target, so a
source-built install had no RPC server and the two-Spark layer split
(studio/spark_cluster.py rpc_server_binary()) could not run on it.

Both scripts now pass -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF on every configure and build
the RPC server best-effort after llama-server and llama-quantize: "ggml-rpc-server"
upstream, "rpc-server" on older trees, read from the tree, and a tree without either
never fails the build. RDMA is off on every platform: that is what every shipped prebuilt
is built with, and it avoids the hard runtime dependency on libibverbs and libnl that
ggml-rpc otherwise picks up whenever libibverbs happens to be installed on the build
host (it auto-enables the transport when it finds a verbs library: libibverbs on every
DGX Spark, librdma on Apple). macOS additionally checks after the build that the cache
kept it off and nothing links librdma, the gate the fork's unsloth-prebuilt-macos.yml
runs, because a Mac with the RDMA framework would otherwise link /usr/lib/librdma.dylib
into libggml-rpc and the whole install then fails to load on a Mac without it.

llama-server and llama-quantize stay the required targets; the RPC server is neither a
health requirement nor a validation gate.
"""

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from unsloth_pwsh_runner import run_pwsh

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"

BASH = "/bin/bash"
PWSH = "/usr/bin/pwsh"
PWSH_AVAILABLE = os.path.isfile(PWSH) and os.access(PWSH, os.X_OK)
requires_pwsh = pytest.mark.skipif(not PWSH_AVAILABLE, reason = "pwsh not available")

SH_FUNCTIONS = ("_llama_rpc_server_target", "_llama_macos_rdma_gate_ok", "_llama_build_rpc_server")
CPU_FALLBACK_COPY = 'CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"'


def _sh() -> str:
    return SETUP_SH.read_text(encoding = "utf-8")


def _ps1() -> str:
    return SETUP_PS1.read_text(encoding = "utf-8")


def _bash_function(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert match is not None, f"{name} is not defined in setup.sh"
    return match.group(0)


def _source_build_block(text: str) -> str:
    start = text.index("# ── 9. Build llama.cpp binaries")
    end = text.index("fi  # end _SKIP_GGUF_BUILD check", start)
    return text[start:end]


def _ps1_step_f(text: str) -> str:
    start = text.index("# -- Step F: Build the RPC server")
    end = text.index("# Swap temp build dir into final location", start)
    return text[start:end]


# ── setup.sh: configure ──


class TestSetupShConfigure:
    def test_rpc_on_and_rdma_off_are_set_once_before_the_cpu_fallback_copy(self):
        """Both flags on the shared CMAKE_ARGS, before CPU_FALLBACK_CMAKE_ARGS copies
        it: every configure (CUDA, ROCm, Metal, CPU, and each CPU fallback) then
        carries them. RDMA off everywhere, not only on macOS: it is what every shipped
        prebuilt is built with, and it avoids the hard runtime dependency on libibverbs
        and libnl that ggml-rpc otherwise picks up whenever libibverbs happens to be
        installed on the build host (it is on every DGX Spark)."""
        block = _source_build_block(_sh())
        assert block.count("-DGGML_RPC=ON") == 1
        assert block.count("-DGGML_RPC_RDMA=OFF") == 1
        flags = 'CMAKE_ARGS="$CMAKE_ARGS -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF"'
        assert block.index(flags) < block.index(CPU_FALLBACK_COPY)
        # Not inside any platform branch: the flags sit between the base line and the
        # Darwin deployment-target block.
        base = block.index('CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release')
        darwin = block.index('if [ "$_HOST_SYSTEM" = "Darwin" ]; then')
        assert base < block.index(flags) < darwin

    def test_backend_selection_is_untouched(self):
        """CUDA, HIP, Metal and the Metal CPU fallback lines exactly as before."""
        block = _source_build_block(_sh())
        assert (
            'CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"'
            in block
        )
        assert 'CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"' in block
        assert (
            'CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON '
            '-DGGML_METAL_USE_BF16=ON -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"'
        ) in block
        assert 'CPU_FALLBACK_CMAKE_ARGS="$CPU_FALLBACK_CMAKE_ARGS -DGGML_METAL=OFF"' in block


# ── setup.sh: targets ──


class TestSetupShTargets:
    def test_rpc_server_follows_every_visual_server_build(self):
        """Both extra-target sites (main build, smoke-test CPU fallback) build the RPC
        server right after the visual server, with the matching label."""
        lines = _source_build_block(_sh()).splitlines()
        visual = [
            i
            for i, line in enumerate(lines)
            if "--target llama-diffusion-gemma-visual-server" in line
        ]
        assert len(visual) == 2
        assert lines[visual[0] + 1].strip() == '_llama_build_rpc_server ""'
        assert "(cpu fallback)" in lines[visual[1]]
        assert lines[visual[1] + 1].strip() == '_llama_build_rpc_server " (cpu fallback)"'

    def test_required_targets_are_exactly_as_before(self):
        lines = _source_build_block(_sh()).splitlines()
        server = [line for line in lines if "--target llama-server" in line]
        assert len(server) == 3
        for line in server:
            assert (
                "|| BUILD_OK=false" in line or 'if ! run_quiet_no_exit "build llama-server"' in line
            )
        quantize = [line for line in lines if "--target llama-quantize" in line]
        assert len(quantize) == 2
        assert all(line.rstrip().endswith("|| true") for line in quantize)

    def test_rpc_target_is_optional(self):
        """The RPC build line is best-effort; BUILD_OK=false in the helper is only the
        macOS no-RPC rebuild of llama-server, i.e. a failed required target."""
        helper = _bash_function(_sh(), "_llama_build_rpc_server")
        rpc = [line for line in helper.splitlines() if '--target "$_target"' in line]
        assert len(rpc) == 1
        assert rpc[0].rstrip().endswith("|| true")
        for line in helper.splitlines():
            if "BUILD_OK=false" in line:
                assert "--target llama-server" in line or line.strip() == "BUILD_OK=false"
        assert "-DGGML_RPC=OFF" in helper

    def test_no_root_level_link_for_the_rpc_server(self):
        """rpc_server_binary() searches build/bin first; the prebuilt path made the same
        call (install_from_archives links nothing at the root either)."""
        for line in _source_build_block(_sh()).splitlines():
            if "ln -sf" in line:
                assert "rpc-server" not in line

    def test_gate_reads_the_cache_and_otool(self):
        gate = _bash_function(_sh(), "_llama_macos_rdma_gate_ok")
        assert "GGML_RPC_RDMA" in gate and "CMakeCache.txt" in gate
        assert "otool -L" in gate and "librdma" in gate
        # pipefail: `grep -q` exiting early would turn a librdma hit into a pass.
        assert "grep -qi 'librdma'" not in gate


# ── setup.sh: the helpers, run ──

_PRELUDE = textwrap.dedent(
    """
    set -euo pipefail
    C_WARN=; C_ERR=; C_DIM=; C_OK=; C_RST=
    step() { printf 'step %s: %s\\n' "$1" "$2"; }
    substep() { printf 'substep %s\\n' "$1"; }
    verbose_substep() { printf 'verbose %s\\n' "$1"; }
    run_quiet_no_exit() { shift; "$@"; }
    """
)

_STUB_CMAKE = textwrap.dedent(
    """\
    #!/bin/bash
    printf '%s\\n' "$*" >> "$CMAKE_LOG"
    case " $* " in
        *" --target ggml-rpc-server "*|*" --target rpc-server "*)
            [ "${FAIL_RPC:-0}" = 1 ] && exit 1 ;;
        *" --target llama-server "*)
            [ "${FAIL_SERVER:-0}" = 1 ] && exit 1 ;;
        *" -S "*)
            mkdir -p "$STUB_BUILD_DIR"
            printf 'GGML_RPC_RDMA:BOOL=OFF\\n' > "$STUB_BUILD_DIR/CMakeCache.txt" ;;
    esac
    exit 0
    """
)

_STUB_OTOOL = "#!/bin/bash\nprintf '%s\\n' \"${OTOOL_OUT:-}\"\n"


@pytest.fixture
def sh_env(tmp_path):
    """A llama.cpp tree with the current RPC tool, a build dir with an RDMA=OFF cache,
    and stub cmake/otool on PATH that log their calls."""
    tree = tmp_path / "tree"
    (tree / "tools" / "rpc").mkdir(parents = True)
    (tree / "tools" / "rpc" / "CMakeLists.txt").write_text("set(TARGET ggml-rpc-server)\n")
    (tree / "build").mkdir()
    (tree / "build" / "CMakeCache.txt").write_text("GGML_RPC_RDMA:BOOL=OFF\n")
    (tree / "build" / "bin").mkdir()
    (tree / "build" / "bin" / "libggml-rpc.dylib").write_bytes(b"")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name, body in (("cmake", _STUB_CMAKE), ("otool", _STUB_OTOOL)):
        stub = stub_bin / name
        stub.write_text(body)
        stub.chmod(0o755)
    log = tmp_path / "cmake.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{stub_bin}:{env['PATH']}",
            "CMAKE_LOG": str(log),
            "STUB_BUILD_DIR": str(tree / "build"),
        }
    )
    return tree, log, env


def _run_helpers(env: dict, script: str, **extra_env) -> subprocess.CompletedProcess:
    text = _sh()
    functions = "".join(_bash_function(text, name) for name in SH_FUNCTIONS)
    run_env = dict(env)
    run_env.update(extra_env)
    return subprocess.run(
        [BASH, "-c", _PRELUDE + functions + textwrap.dedent(script)],
        capture_output = True,
        text = True,
        timeout = 60,
        env = run_env,
    )


def _build_script(
    tree: Path,
    host: str,
    metal: str = "false",
) -> str:
    return f"""
        _BUILD_TMP='{tree}'; NCPU=3; CMAKE_GENERATOR_ARGS=""
        CMAKE_ARGS="-DGGML_METAL=ON -DGGML_RPC=ON"; CPU_FALLBACK_CMAKE_ARGS="-DGGML_METAL=OFF -DGGML_RPC=ON"
        _TRY_METAL_CPU_FALLBACK={metal}; _HOST_SYSTEM={host}; BUILD_OK=true
        _llama_build_rpc_server " (label)"
        echo "BUILD_OK=$BUILD_OK"
    """


class TestSetupShHelpersRun:
    @pytest.mark.parametrize(
        ("layout", "expected"),
        [
            ("tools/rpc:set(TARGET ggml-rpc-server)", "ggml-rpc-server"),
            ("tools/rpc:set(TARGET rpc-server)", "rpc-server"),
            ("examples/rpc:add_executable(rpc-server rpc-server.cpp)", "rpc-server"),
            ("", ""),
        ],
        ids = ["current", "older-tools", "legacy-examples", "no-rpc-tool"],
    )
    def test_target_name_follows_the_tree(self, tmp_path, layout, expected):
        tree = tmp_path / "tree"
        tree.mkdir()
        if layout:
            sub, content = layout.split(":", 1)
            (tree / sub).mkdir(parents = True)
            (tree / sub / "CMakeLists.txt").write_text(content + "\n")
        result = _run_helpers(os.environ.copy(), f"_llama_rpc_server_target '{tree}'; echo")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected

    def test_builds_the_resolved_target_on_linux(self, sh_env):
        tree, log, env = sh_env
        result = _run_helpers(env, _build_script(tree, "Linux"))
        assert result.returncode == 0, result.stderr
        calls = log.read_text().splitlines()
        assert calls == [f"--build {tree}/build --config Release --target ggml-rpc-server -j3"]
        assert "BUILD_OK=true" in result.stdout

    def test_a_failed_rpc_build_keeps_the_build_ok(self, sh_env):
        tree, log, env = sh_env
        result = _run_helpers(env, _build_script(tree, "Linux"), FAIL_RPC = "1")
        assert result.returncode == 0, result.stderr
        assert "BUILD_OK=true" in result.stdout
        assert len(log.read_text().splitlines()) == 1

    def test_a_tree_without_the_tool_builds_nothing_and_stays_ok(self, sh_env):
        tree, log, env = sh_env
        (tree / "tools" / "rpc" / "CMakeLists.txt").unlink()
        result = _run_helpers(env, _build_script(tree, "Linux"))
        assert result.returncode == 0, result.stderr
        assert not log.exists()
        assert "BUILD_OK=true" in result.stdout

    def test_macos_with_rdma_off_and_nothing_linked_passes(self, sh_env):
        tree, log, env = sh_env
        result = _run_helpers(
            env, _build_script(tree, "Darwin"), OTOOL_OUT = "\t@rpath/libggml-base.0.dylib"
        )
        assert result.returncode == 0, result.stderr
        assert len(log.read_text().splitlines()) == 1
        assert "rebuilding without GGML_RPC" not in result.stdout
        assert "BUILD_OK=true" in result.stdout

    @pytest.mark.parametrize("cache", ["GGML_RPC_RDMA:BOOL=OFF", "GGML_RPC_RDMA:UNINITIALIZED=OFF"])
    def test_gate_accepts_an_off_cache_of_either_type(self, sh_env, cache):
        """An older tree without the option keeps the -D value as UNINITIALIZED."""
        tree, _log, env = sh_env
        (tree / "build" / "CMakeCache.txt").write_text(cache + "\n")
        result = _run_helpers(env, f"_llama_macos_rdma_gate_ok '{tree}/build'; echo gate=$?")
        assert result.stdout.strip() == "gate=0", result.stderr

    @pytest.mark.parametrize(
        ("cache", "otool_out"),
        [
            ("GGML_RPC_RDMA:BOOL=ON", ""),
            ("GGML_RPC_RDMA:BOOL=OFF", "\t/usr/lib/librdma.dylib (compatibility version 1.0.0)"),
        ],
        ids = ["cache-on", "librdma-linked"],
    )
    def test_gate_rejects_rdma(self, sh_env, cache, otool_out):
        tree, _log, env = sh_env
        (tree / "build" / "CMakeCache.txt").write_text(cache + "\n")
        result = _run_helpers(
            env,
            f"_llama_macos_rdma_gate_ok '{tree}/build' && echo gate=0 || echo gate=1",
            OTOOL_OUT = otool_out,
        )
        assert result.stdout.strip() == "gate=1", result.stderr

    @pytest.mark.parametrize(
        "metal", ["true", "false"], ids = ["metal-config", "cpu-fallback-config"]
    )
    def test_macos_rdma_leak_rebuilds_without_rpc_from_the_active_args(self, sh_env, metal):
        tree, log, env = sh_env
        result = _run_helpers(
            env, _build_script(tree, "Darwin", metal = metal), OTOOL_OUT = "\t/usr/lib/librdma.dylib"
        )
        assert result.returncode == 0, result.stderr
        assert "rebuilding without GGML_RPC" in result.stdout
        calls = log.read_text().splitlines()
        expected_args = (
            "-DGGML_METAL=ON -DGGML_RPC=ON" if metal == "true" else "-DGGML_METAL=OFF -DGGML_RPC=ON"
        )
        assert calls == [
            f"--build {tree}/build --config Release --target ggml-rpc-server -j3",
            f"-S {tree} -B {tree}/build {expected_args} -DGGML_RPC=OFF",
            f"--build {tree}/build --config Release --target llama-server -j3",
            f"--build {tree}/build --config Release --target llama-quantize -j3",
            f"--build {tree}/build --config Release --target llama-diffusion-gemma-visual-server -j3",
        ]
        assert "BUILD_OK=true" in result.stdout

    def test_only_a_failed_llama_server_rebuild_fails_the_build(self, sh_env):
        tree, _log, env = sh_env
        result = _run_helpers(
            env,
            _build_script(tree, "Darwin"),
            OTOOL_OUT = "\t/usr/lib/librdma.dylib",
            FAIL_SERVER = "1",
        )
        assert result.returncode == 0, result.stderr
        assert "BUILD_OK=false" in result.stdout


# ── setup.ps1 ──


class TestSetupPs1:
    def test_rpc_on_and_rdma_off_are_common_flags(self):
        """Once each, between the shared flags and the CUDA selection, so both the CUDA
        and the CPU configure carry them."""
        text = _ps1()
        native = text.index("$CmakeArgs += '-DGGML_NATIVE=ON'")
        cuda = text.index("if ($HasNvidiaSmi -and $NvccPath) {", native)
        for flag in ("$CmakeArgs += '-DGGML_RPC=ON'", "$CmakeArgs += '-DGGML_RPC_RDMA=OFF'"):
            assert text.count(flag) == 1
            assert native < text.index(flag) < cuda
        assert text.count("GGML_RPC_RDMA") == 1
        assert text.count("$CmakeArgs += '-DGGML_CUDA=ON'") == 1
        assert text.count("$CmakeArgs += '-DGGML_CUDA=OFF'") == 2

    def test_step_f_builds_the_resolved_target_best_effort(self):
        step_f = _ps1_step_f(_ps1())
        assert "Get-LlamaRpcServerTarget -SourceDir $LlamaCppDir" in step_f
        assert (
            "cmake --build $BuildDir --config Release --target $RpcServerTarget -j $NumCpu"
            in step_f
        )
        assert "$BuildOk = $false" not in step_f
        assert "$FailedStep" not in step_f
        assert "Copy-Item" not in step_f and "New-Item" not in step_f

    def test_step_f_follows_the_visual_server(self):
        text = _ps1()
        assert text.index("--target llama-diffusion-gemma-visual-server") < text.index(
            "# -- Step F: Build the RPC server"
        )

    def test_required_targets_are_exactly_as_before(self):
        text = _ps1()
        assert (
            text.count("cmake --build $BuildDir --config Release --target llama-server -j $NumCpu")
            == 1
        )
        assert (
            text.count(
                "cmake --build $BuildDir --config Release --target llama-quantize -j $NumCpu"
            )
            == 1
        )

    def test_target_resolver_reads_both_tree_layouts(self):
        text = _ps1()
        start = text.index("function Get-LlamaRpcServerTarget")
        body = text[start : text.index("\n}\n", start)]
        assert "tools\\rpc\\CMakeLists.txt" in body and "examples\\rpc\\CMakeLists.txt" in body
        assert "return 'ggml-rpc-server'" in body and "return 'rpc-server'" in body

    def test_summary_looks_in_the_release_dir(self):
        """build\\bin\\Release is where rpc_server_binary() looks on Windows."""
        text = _ps1()
        assert 'Join-Path $BuildDir "bin\\Release\\$rpcName.exe"' in text
        assert "@('ggml-rpc-server', 'rpc-server')" in text

    @requires_pwsh
    @pytest.mark.parametrize(
        ("layout", "expected"),
        [
            ("tools/rpc:set(TARGET ggml-rpc-server)", "ggml-rpc-server"),
            ("tools/rpc:set(TARGET rpc-server)", "rpc-server"),
            ("examples/rpc:add_executable(rpc-server rpc-server.cpp)", "rpc-server"),
            ("", ""),
        ],
        ids = ["current", "older-tools", "legacy-examples", "no-rpc-tool"],
    )
    def test_target_resolver_runs(self, tmp_path, layout, expected):
        tree = tmp_path / "tree"
        tree.mkdir()
        if layout:
            sub, content = layout.split(":", 1)
            (tree / sub).mkdir(parents = True)
            (tree / sub / "CMakeLists.txt").write_text(content + "\n")
        text = _ps1()
        start = text.index("function Get-LlamaRpcServerTarget")
        function = text[start : text.index("\n}\n", start) + 3]
        script = (
            function
            + f"\nWrite-Output ('<' + (Get-LlamaRpcServerTarget -SourceDir '{tree}') + '>')\n"
        )
        result = run_pwsh(
            [PWSH, "-NoProfile", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert result.returncode == 0, result.stderr
        assert f"<{expected}>" in result.stdout


# ── the two scripts agree, and the RPC server is never required ──


def test_both_scripts_resolve_the_same_names():
    sh_helper = _bash_function(_sh(), "_llama_rpc_server_target")
    text = _ps1()
    start = text.index("function Get-LlamaRpcServerTarget")
    ps1_helper = text[start : text.index("\n}\n", start)]
    for name in ("ggml-rpc-server", "rpc-server"):
        assert name in sh_helper and name in ps1_helper
    for path in ("tools/rpc/CMakeLists.txt", "examples/rpc/CMakeLists.txt"):
        assert path in sh_helper and path.replace("/", "\\") in ps1_helper


def test_rpc_server_is_not_a_health_requirement():
    """runtime_payload_health_groups drives the staged validation and the existing
    install health check; a source or prebuilt tree without the RPC server must not
    start failing (or re-installing) because of it."""
    spec = importlib.util.spec_from_file_location(
        "studio_install_llama_prebuilt_rpc_source_build", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kinds = (
        "linux-cpu",
        "linux-cuda",
        "linux-arm64-cuda",
        "linux-rocm",
        "linux-arm64",
        "linux-vulkan",
        "macos-arm64",
        "macos-x64",
        "windows-cpu",
        "windows-cuda",
        "windows-hip",
        "windows-vulkan",
        "windows-rocm",
        "windows-arm64",
    )
    names = set(module.RPC_SERVER_NAMES) | {name + ".exe" for name in module.RPC_SERVER_NAMES}
    for kind in kinds:
        for source_label in (None, "published", "upstream"):
            groups = module.runtime_payload_health_groups(kind, source_label = source_label)
            required = {entry for group in groups for entry in group}
            assert not (required & names), (kind, source_label, required & names)
