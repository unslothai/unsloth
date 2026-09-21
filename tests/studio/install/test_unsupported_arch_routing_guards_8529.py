# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one safety property behind the #8529 unsupported-arch table: it is MESSAGING.

The table added for #8529 (and extended for #8458) names AMD generations ROCm
PyTorch does not cover: gfx1010 / gfx1011 / gfx1012 (RDNA 1, Navi 10/14) and
gfx803 (Polaris 10/20/30). Every one of those must keep landing on CPU torch. If
any of them ever reaches an AMD per-arch wheel index, the fix stops being a
wording change: pip would install a repo.amd.com wheel built for another arch and
the user gets an import-time HIP failure instead of a slow-but-working CPU stack.

test_rdna1_unsupported_message_8529.py already asserts that property, but only
against the SHAPE of one Python dict (`_GFX_TO_AMD_INDEX_ARCH`). Mutation testing
showed three real routing changes that the whole suite still passes:

* an extra `gfx803|gfx1010|gfx1011|gfx1012) echo gfx103X-all ;;` arm in install.sh's
  `_amd_arch_index_family_for_gfx`, which sends UNSLOTH_ROCM_GFX_ARCH=gfx803 to
  https://repo.amd.com/rocm/whl/gfx103X-all/ instead of the CPU index;
* an extra `"gfx803" = "gfx110X-all"` entry in `$archFamilyMap`, in install.ps1 or
  studio/setup.ps1 (only ever caught transitively, and only when the two copies
  disagreed with each other);
* a behavioural bypass inside `_amd_arch_index_url` / `_windows_rocm_index_url`
  that hands back a family the dict does not contain, which a shape assertion on
  the dict cannot see at all.

So the tests below ask each source the routing question the installer asks it, in
that source's own language: the Python functions are CALLED, install.sh's case
table is EXECUTED under sh, and the PowerShell maps are read as maps (and, where
pwsh exists, evaluated). Every group carries a positive control, because "no AMD
index came back" is also what a renamed or unparsed table returns.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import textwrap
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from unsloth_pwsh_runner import run_pwsh


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_INSTALL_SH = PACKAGE_ROOT / "install.sh"
_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
_STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"


def _load_stack_module():
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_routing", _STACK_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()


# The four arches the messaging table owns. Nothing here may produce an index.
_UNSUPPORTED_ARCHES = ["gfx803", "gfx1010", "gfx1011", "gfx1012"]

# The same arches as the installers may actually receive them.
# UNSLOTH_ROCM_GFX_ARCH is user-typed (so any case), and a gcnArchName copied out of hipinfo/rocminfo carries the target
# features ("gfx1010:xnack-"), which is why the override reader lowercases and splits on ":".
# A routing bypass written against a prefix would take these too.
_UNSUPPORTED_ARCH_INPUTS = (
    _UNSUPPORTED_ARCHES
    + [a.upper() for a in _UNSUPPORTED_ARCHES]
    + [f"{a}:xnack-" for a in _UNSUPPORTED_ARCHES]
)

# Arches that MUST route, so that "no index" cannot pass for the right answer when the table has been renamed, emptied
# or parsed wrong. gfx1030 is RDNA 2 (RX 6800 XT) and gfx1100 is RDNA 3 (RX 7900 XTX); both ship AMD wheels today.
_ROUTABLE_ARCHES = [("gfx1030", "gfx103X-all"), ("gfx1100", "gfx110X-all")]


# ── The Python resolvers, called rather than inspected ───────────────────────


@pytest.fixture(autouse = True)
def _no_index_mirror():
    """Both resolvers honour a mirror override, and a host that has one set would
    answer with its URL instead of repo.amd.com. Clear them so the assertions below
    are about the arch tables and not about the environment running the suite."""
    with patch.dict(os.environ, {}, clear = False):
        for _v in ("UNSLOTH_AMD_ROCM_MIRROR", "UNSLOTH_ROCM_WINDOWS_MIRROR"):
            os.environ.pop(_v, None)
        yield


class TestPythonIndexResolversAreAskedDirectly:
    """`_GFX_TO_AMD_INDEX_ARCH` is where the answer SHOULD come from, but the callers
    go through `_amd_arch_index_url` / `_windows_rocm_index_url`, and only those two
    decide what pip is pointed at. Ask them."""

    @pytest.mark.parametrize("arch", _UNSUPPORTED_ARCH_INPUTS)
    @pytest.mark.parametrize("is_windows", [False, True], ids = ["linux", "windows"])
    def test_no_unsupported_arch_gets_an_index_url(self, arch, is_windows):
        # IS_WINDOWS is read inside _amd_arch_index_url, so both platform arms are reachable from this host; the Windows
        # arm is the one #8529 was filed from.
        with patch.object(stack_mod, "IS_WINDOWS", is_windows):
            url = stack_mod._amd_arch_index_url(arch)
        assert url is None, f"{arch} was routed to {url!r}; it must fall through to CPU torch"

    @pytest.mark.parametrize("arch", _UNSUPPORTED_ARCH_INPUTS)
    def test_no_unsupported_arch_gets_a_windows_index_url(self, arch):
        """The Windows resolver on its own: install_python_stack calls it directly at
        the Windows torch site, not only through _amd_arch_index_url."""
        url = stack_mod._windows_rocm_index_url(arch)
        assert url is None, f"{arch} was routed to {url!r}"

    @pytest.mark.parametrize("arch", _UNSUPPORTED_ARCH_INPUTS)
    @pytest.mark.parametrize("is_windows", [False, True], ids = ["linux", "windows"])
    def test_no_unsupported_arch_reaches_repo_amd_com(self, arch, is_windows):
        """The same claim stated as the consequence, so that a resolver which starts
        returning some other truthy non-index string still fails here."""
        with patch.object(stack_mod, "IS_WINDOWS", is_windows):
            url = stack_mod._amd_arch_index_url(arch) or ""
        assert "repo.amd.com" not in url, f"{arch} reaches an AMD wheel index: {url!r}"

    @pytest.mark.parametrize("arch,family", _ROUTABLE_ARCHES)
    @pytest.mark.parametrize("is_windows", [False, True], ids = ["linux", "windows"])
    def test_a_covered_arch_still_gets_its_index(self, arch, family, is_windows):
        """The positive control. Without it every assertion above passes on a build
        where the resolvers were gutted to `return None`."""
        with patch.object(stack_mod, "IS_WINDOWS", is_windows):
            url = stack_mod._amd_arch_index_url(arch)
        assert url is not None, f"{arch} lost its wheel index"
        assert url.endswith(f"/{family}/"), f"{arch} routed to {url!r}, expected the {family} index"

    @pytest.mark.parametrize("arch", _UNSUPPORTED_ARCHES)
    def test_no_unsupported_arch_is_a_key_of_the_family_map(self, arch):
        """The shape half, kept alongside the behavioural one: a table entry and a
        code bypass are different mistakes and each should name itself."""
        assert arch not in stack_mod._GFX_TO_AMD_INDEX_ARCH


# ── install.sh's case table, executed under sh ───────────────────────────────


def _sh_function_body(source: str, name: str) -> str:
    """Same extraction as test_rdna1_unsupported_message_8529.py: take the function
    verbatim from the installer so the shell evaluates the shipped text, not a
    Python re-implementation of it."""
    needle = f"{name}() {{"
    start = source.find(needle)
    assert start != -1, f"{name}() not found"
    depth = 0
    i = start + len(needle) - 1
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated {name}()")


def _run_sh_index_family(arch: str) -> "tuple[int, str]":
    body = _sh_function_body(
        _INSTALL_SH.read_text(encoding = "utf-8"), "_amd_arch_index_family_for_gfx"
    )
    script = f'{body}\n_amd_arch_index_family_for_gfx "$1"\n'
    out = subprocess.run(
        ["sh", "-c", script, "sh", arch],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 30,
    )
    return out.returncode, out.stdout.strip()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh on this host")
class TestInstallShIndexFamilyRuns:
    """install.sh picks the Linux AMD index from `_amd_arch_index_family_for_gfx`, and
    its three call sites treat a non-zero return as "no AMD wheels, use CPU". An arm
    added there is invisible to every Python-side assertion, so run the real case."""

    @pytest.mark.parametrize("arch", _UNSUPPORTED_ARCH_INPUTS)
    def test_no_unsupported_arch_yields_a_family(self, arch):
        rc, out = _run_sh_index_family(arch)
        assert rc != 0 and out == "", (
            f"install.sh maps {arch} to the {out!r} index family, so "
            f"UNSLOTH_ROCM_GFX_ARCH={arch} would install a repo.amd.com wheel built "
            f"for another arch instead of CPU PyTorch"
        )

    @pytest.mark.parametrize("arch,family", _ROUTABLE_ARCHES)
    def test_a_covered_arch_still_yields_its_family(self, arch, family):
        """The positive control: proves the function was found, sourced and reached."""
        rc, out = _run_sh_index_family(arch)
        assert (rc, out) == (0, family), f"{arch} resolved to {out!r} (rc {rc})"

    def test_the_index_selector_is_still_the_function_under_test(self):
        """The extraction above is only meaningful while install.sh actually asks this
        function; a renamed selector would leave the tests green against dead code."""
        src = _INSTALL_SH.read_text(encoding = "utf-8").replace("\r\n", "\n")
        calls = len(
            re.findall(r"^\s*[^#\n]*_amd_arch_index_family_for_gfx \"\$", src, re.MULTILINE)
        )
        assert calls >= 3, f"install.sh calls the index-family selector {calls} times, expected 3"


# ── install.sh's whole index selector, executed under sh ─────────────────────

# What get_torch_index_url calls that is not in the two functions extracted below.
# Each stub is the shape of a host with an AMD GPU, no NVIDIA GPU and no readable ROCm userspace, the arrangement that
# carries a user-pinned arch all the way to the routing decision. The probe stub reproduces the override arm of the
# real _probe_amd_gfx_arch so a case-shifted pin arrives at the selector as it would on a real host.
_INDEX_URL_STUBS = """
uname() { case "$1" in -m) echo x86_64 ;; *) echo Linux ;; esac; }
_has_usable_nvidia_gpu() { return 1; }
_has_amd_rocm_gpu() { return 0; }
_probe_amd_gfx_arch() { printf '%s\\n' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]'; }
_infer_linux_amd_gfx_arch() { printf '%s\\n' "${UNSLOTH_ROCM_GFX_ARCH:-}"; }
_infer_linux_unsupported_amd_gfx_arch() { return 1; }
_detect_rocm_version_tag() { [ -n "${_STUB_ROCM_TAG:-}" ] && printf '%s\\n' "$_STUB_ROCM_TAG"; }
"""


def _run_sh_get_torch_index_url(arch: str, rocm_tag: str = "") -> "tuple[str, str]":
    """Run install.sh's real get_torch_index_url with UNSLOTH_ROCM_GFX_ARCH=arch."""
    src = _INSTALL_SH.read_text(encoding = "utf-8")
    script = (
        _sh_function_body(src, "_amd_arch_index_family_for_gfx")
        + "\n"
        + _sh_function_body(src, "_amd_probe_arches")
        + "\n"
        + _sh_function_body(src, "_amd_agreed_index_family")
        + "\n"
        + _sh_function_body(src, "_amd_sole_index_arch")
        + "\n"
        + _sh_function_body(src, "get_torch_index_url")
        + "\n"
        + _INDEX_URL_STUBS
        + "\nget_torch_index_url\n"
    )
    env = dict(os.environ)
    # The two pins short-circuit the whole selector, and a mirror override would rewrite the base out from under the
    # assertions.
    for _v in (
        "UNSLOTH_TORCH_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_FAMILY",
        "UNSLOTH_PYTORCH_MIRROR",
        "UNSLOTH_AMD_ROCM_MIRROR",
    ):
        env.pop(_v, None)
    env["UNSLOTH_ROCM_GFX_ARCH"] = arch
    env["_STUB_ROCM_TAG"] = rocm_tag
    out = subprocess.run(
        ["sh", "-c", script],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        env = env,
        timeout = 60,
    )
    return out.stdout.strip(), out.stderr


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh on this host")
class TestInstallShIndexSelectorRuns:
    """The case table above is only one of get_torch_index_url's decisions, and the
    function is what install.sh actually calls. An arch test written straight into
    the selector -- a `case "$_amd_gfx_probe" in gfx1010) ...` shortcut ahead of the
    family lookup -- leaves the case table innocent and still routes the card. So ask
    the selector itself, with the arch pinned the way a user of an uncovered card is
    told to pin it."""

    @pytest.mark.parametrize("arch", _UNSUPPORTED_ARCH_INPUTS)
    def test_no_unsupported_arch_leaves_the_cpu_index(self, arch):
        url, err = _run_sh_get_torch_index_url(arch)
        assert url.endswith(
            "/cpu"
        ), f"UNSLOTH_ROCM_GFX_ARCH={arch} selected {url!r}, not the CPU index"
        assert "repo.amd.com" not in (
            url + err
        ), f"UNSLOTH_ROCM_GFX_ARCH={arch} reaches an AMD wheel index: {url!r}"

    def test_a_covered_arch_is_recognised_by_the_selector(self):
        """First positive control: the selector really does tell the two apart. A
        covered arch reaches the per-arch handoff (the reroute below get_torch_index_url
        turns that into the repo.amd.com URL; the selector's own job is to name it),
        and no unsupported arch does -- so "everything lands on /cpu" cannot be what
        makes the bans above pass."""
        covered, covered_err = _run_sh_get_torch_index_url("gfx1030")
        assert covered.endswith("/cpu"), f"gfx1030 selected {covered!r}"
        assert (
            "AMD per-arch wheels" in covered_err
        ), "gfx1030 no longer reaches the per-arch handoff, so the bans above prove nothing"
        for arch in _UNSUPPORTED_ARCHES:
            _url, err = _run_sh_get_torch_index_url(arch)
            assert (
                "AMD per-arch wheels" not in err
            ), f"{arch} reaches the per-arch handoff, which reroutes to repo.amd.com"

    def test_the_selector_still_reaches_its_amd_branch(self):
        """Second positive control: on the same stubbed host with a readable ROCm
        version the selector returns a ROCm index, which it can only do by running the
        AMD branch the test above depends on."""
        url, _err = _run_sh_get_torch_index_url("gfx1030", rocm_tag = "rocm6.4")
        assert url.endswith("/rocm6.4"), f"the AMD branch was not reached: {url!r}"


# setup.sh forwards `--rocm-gfx "$_setup_gfx"` to install_llama_prebuilt.py and the whisper
# installer, and keys the supported table on $_setup_mkt. The unsupported lookup exists to
# print a line, so its result must not flow into either: assigning it to _setup_gfx passes
# every arch assertion in this file and still ships --rocm-gfx gfx803.

_SETUP_SH_UNSUPPORTED_HELPERS = ("_setup_unsupported_gfx_any", "_setup_unsupported_gfx_from_name")


_ROUTED_SETUP_VARS = ("_setup_gfx", "_setup_mkt")

# Assignment targets anywhere in a line, not only at its start: the lookup is captured mid-condition
# (`elif _setup_unsup_gfx=$(...); then`), which is exactly where the routed variable would be substituted for the
# report-only one.
_SH_ASSIGN_TARGET = re.compile(r"(?:^|[;&|(]|\s)([A-Za-z_][A-Za-z0-9_]*)=")


def test_setup_sh_never_feeds_the_unsupported_lookup_into_a_routed_variable():
    src = _SETUP_SH.read_text(encoding = "utf-8").replace("\r\n", "\n")
    code = [line for line in src.splitlines() if not line.lstrip().startswith("#")]
    captures = [
        line.strip()
        for line in code
        if any(h in line for h in _SETUP_SH_UNSUPPORTED_HELPERS) and _SH_ASSIGN_TARGET.search(line)
    ]
    # Positive control: the report-only capture is still there to be checked.
    assert captures, "studio/setup.sh no longer captures the unsupported lookup at all"
    for line in captures:
        for target in _SH_ASSIGN_TARGET.findall(line):
            assert target not in _ROUTED_SETUP_VARS, (
                f"studio/setup.sh writes the unsupported lookup into ${target}: {line!r}. "
                f"That variable selects --rocm-gfx for the llama.cpp and whisper installers, "
                f"so an uncovered card would be built for as if it had wheels"
            )
    # The same ban one hop later: capturing the lookup in its own variable and then copying that into the routed one is
    # the same mutation written in two lines.
    relayed = [
        line.strip()
        for line in code
        if re.search(
            r"(?:^|[;&|(]|\s)(?:" + "|".join(_ROUTED_SETUP_VARS) + r")=[^\n]*_setup_unsup", line
        )
    ]
    assert (
        not relayed
    ), f"studio/setup.sh relays the unsupported lookup into a routed variable: {relayed}"
    forwards = [line for line in code if "--rocm-gfx" in line and "_setup_gfx" in line]
    assert forwards, "studio/setup.sh no longer forwards --rocm-gfx from $_setup_gfx (renamed?)"


@pytest.mark.parametrize("arch", _UNSUPPORTED_ARCHES)
def test_setup_sh_never_assigns_an_unsupported_arch_to_the_routed_variable(arch):
    """The same property from the other end: whatever names the unsupported arches,
    none of them may be written into the variable that becomes --rocm-gfx."""
    src = _SETUP_SH.read_text(encoding = "utf-8").replace("\r\n", "\n")
    hits = [
        line.strip()
        for line in src.splitlines()
        if not line.lstrip().startswith("#") and re.search(r"_setup_gfx=[^\n]*" + arch, line)
    ]
    assert not hits, f"studio/setup.sh routes {arch} into --rocm-gfx: {hits}"


# ── The PowerShell copies of the same map ────────────────────────────────────


def _ps_block(source: str, header: str, opener: str, closer: str) -> str:
    """The literal `@{...}` or `@(...)` that `header` opens, balanced."""
    start = source.find(header)
    assert start != -1, f"{header} not found"
    i = source.find(opener, start)
    depth = 0
    while i < len(source):
        if source[i] == opener:
            depth += 1
        elif source[i] == closer:
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated {header}")


def _ps_table_arches(path: Path, header: str, opener: str, closer: str) -> "list[str]":
    """Every gfx arch one PowerShell routing table routes, read out of its own file.
    CRLF-normalised first: install.ps1 and setup.ps1 both ship CRLF."""
    return _ps_arches(
        _ps_block(path.read_text(encoding = "utf-8").replace("\r\n", "\n"), header, opener, closer)
    )


def _ps_arches(block: str) -> "list[str]":
    """Every gfx arch the block routes: hashtable keys ("gfx1030" = ...) and plain
    array members alike. Comments are stripped first, since both blocks annotate
    their rows with generation names.

    Quoting is not part of the property: PowerShell takes 'gfx803', "gfx803" and a
    bare gfx803 as the same key or member, so all three have to be read, or a row
    added in the other style is silently reported as absent. The trailing guard
    stops a family VALUE from being misread as a routed arch: gfx103X-all shares a
    prefix with gfx103, and only the full token is a routing key."""
    stripped = "\n".join(line.split("#", 1)[0] for line in block.splitlines())
    keys = re.findall(r"""['"]?(gfx[0-9a-z]+)['"]?\s*=(?!=)""", stripped)
    if keys:
        return keys
    return re.findall(r"""['"]?(gfx[0-9a-z]+)(?![0-9a-zA-Z-])""", stripped)


# Each PowerShell routing table, checked in its OWN file.
# The parity test in test_rocm_arch_table_parity.py compares the copies against each other, which stays green when the
# same wrong arch is added to all of them.
_PS_TABLES = [
    (_INSTALL_PS1, "$archFamilyMap = @{", "{", "}"),
    (_SETUP_PS1, "$archFamilyMap = @{", "{", "}"),
    (_SETUP_PS1, "$_rocmWheelArches = @(", "(", ")"),
]
_PS_TABLE_IDS = [f"{p.name}:{h.split()[0][1:]}" for p, h, _o, _c in _PS_TABLES]

# Every way PowerShell can change $archFamilyMap: a keyed assignment, a rebind, a merge, or one of the IDictionary
# mutators. Reads (`.ContainsKey($a)`, `$map[$a]` as a value) deliberately do not match.
_PS_MAP_WRITE = re.compile(
    r"\$archFamilyMap\s*(?:\[[^\]]*\]\s*=(?!=)|=(?!=)|\+=|\.\s*(?:Add|Remove|Clear|set_Item)\s*\()",
    re.IGNORECASE,
)


class TestPowerShellRoutingTables:
    @pytest.mark.parametrize("path,header,opener,closer", _PS_TABLES, ids = _PS_TABLE_IDS)
    def test_the_table_parses_and_still_routes_a_covered_arch(self, path, header, opener, closer):
        """The positive control for the two tests below, which are both bans and would
        otherwise pass on a block that parsed to nothing."""
        arches = _ps_table_arches(path, header, opener, closer)
        assert arches, f"{path.name}: {header} parsed empty (moved or renamed?)"
        for _arch, _family in _ROUTABLE_ARCHES:
            assert _arch in arches, f"{path.name}: {header} no longer routes {_arch}"

    @pytest.mark.parametrize("arch", _UNSUPPORTED_ARCHES)
    @pytest.mark.parametrize("path,header,opener,closer", _PS_TABLES, ids = _PS_TABLE_IDS)
    def test_no_unsupported_arch_is_routed(self, path, header, opener, closer, arch):
        arches = _ps_table_arches(path, header, opener, closer)
        assert arch not in arches, (
            f"{path.name}: {header} routes {arch}, so a card the installer is about to "
            f"call uncovered would be sent to an AMD wheel index"
        )

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    def test_the_family_map_is_never_written_to_after_it_is_declared(self, path):
        """The tests above read the `@{...}` literal, which is the whole map only for
        as long as nothing edits it later. `$archFamilyMap["gfx803"] = "gfx103X-all"`
        (or `.Add("gfx803", ...)`) anywhere below the declaration routes gfx803 in the
        real installer while leaving the literal, and so every assertion on it, intact.
        The map is a constant table; require it to stay one."""
        src = path.read_text(encoding = "utf-8").replace("\r\n", "\n")
        block = _ps_block(src, "$archFamilyMap = @{", "{", "}")
        assert _PS_MAP_WRITE.search(block), (
            f"{path.name}: the write pattern no longer matches the declaration itself, "
            f"so it would not match an added row either"
        )
        rest = src.replace(block, "", 1)
        writes = [
            line.strip()
            for line in rest.splitlines()
            if _PS_MAP_WRITE.search(line) and not line.lstrip().startswith("#")
        ]
        assert not writes, (
            f"{path.name}: $archFamilyMap is modified after its declaration ({writes}), "
            f"so the routing this file checks is not the routing the installer performs"
        )
        # Positive control: the map is still consulted below the declaration, so the region searched above is the one a
        # late addition would have to live in.
        assert (
            "$archFamilyMap.ContainsKey" in rest
        ), f"{path.name}: nothing reads $archFamilyMap any more (renamed or removed?)"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh not available")
class TestPowerShellMapEvaluated:
    """The gate the installers actually run is `$archFamilyMap.ContainsKey($arch)`.
    Let PowerShell answer it, so a row this file's regex misreads (a differently
    quoted key, a splatted addition) is not silently treated as absent."""

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    def test_containskey_is_false_for_every_unsupported_arch(self, path):
        src = path.read_text(encoding = "utf-8").replace("\r\n", "\n")
        block = _ps_block(src, "$archFamilyMap = @{", "{", "}")
        probes = ", ".join(f'"{a}"' for a in _UNSUPPORTED_ARCHES)
        script = (
            f"{block}\n"
            f"foreach ($a in @({probes})) {{\n"
            f'    if ($archFamilyMap.ContainsKey($a)) {{ "ROUTED:$a" }}\n'
            f"}}\n"
            f"if ($archFamilyMap.ContainsKey('gfx1030')) {{ 'CONTROL_OK' }}\n"
        )
        # run_pwsh, not subprocess.run: the returncode assertion below reads a non-zero exit as $archFamilyMap failing
        # to evaluate, and a signal-killed interpreter would be filed as that same map being broken.
        # See tests/_shared/unsloth_pwsh_runner.py.
        out = run_pwsh(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 120,
        )
        assert out.returncode == 0, f"{path.name}: the map did not evaluate under pwsh"
        printed = out.stdout.split()
        assert "CONTROL_OK" in printed, f"{path.name}: gfx1030 lost its wheel index"
        routed = [line for line in printed if line.startswith("ROUTED:")]
        assert not routed, f"{path.name}: {routed} reach an AMD wheel index"

    def test_contains_is_false_for_every_unsupported_arch(self):
        """setup.ps1 gates the Windows AMD wheels on `$_rocmWheelArches -contains $arch`
        rather than on the map, so that list needs the same evaluated check: a member
        added in a quoting style this file's regex does not read would otherwise pass
        the textual ban above and still route the arch."""
        src = _SETUP_PS1.read_text(encoding = "utf-8").replace("\r\n", "\n")
        block = _ps_block(src, "$_rocmWheelArches = @(", "(", ")")
        probes = ", ".join(f'"{a}"' for a in _UNSUPPORTED_ARCHES)
        script = (
            f"{block}\n"
            f"foreach ($a in @({probes})) {{\n"
            f'    if ($_rocmWheelArches -contains $a) {{ "ROUTED:$a" }}\n'
            f"}}\n"
            f"if ($_rocmWheelArches -contains 'gfx1030') {{ 'CONTROL_OK' }}\n"
        )
        # run_pwsh, not subprocess.run: a crashed interpreter prints nothing, so the CONTROL_OK check below would see
        # the positive control missing and report setup.ps1's $_rocmWheelArches as wrong.
        # See tests/_shared/unsloth_pwsh_runner.py.
        out = run_pwsh(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 120,
        )
        assert out.returncode == 0, "setup.ps1: $_rocmWheelArches did not evaluate under pwsh"
        printed = out.stdout.split()
        assert "CONTROL_OK" in printed, "setup.ps1: gfx1030 lost its wheel index"
        routed = [line for line in printed if line.startswith("ROUTED:")]
        assert not routed, f"setup.ps1: {routed} reach an AMD wheel index"


# ── The second, independent arch to repo.amd.com gate ────────────────────────


# The Strix reroute picks a per-arch index without consulting either family map, so it is
# a routing site the assertions above cannot see. Reachability is narrow, but "narrow" is
# not the property this file defends, and the literal is one line per source.
_STRIX_SITES = [
    (_INSTALL_SH, r"^\s*(gfx[0-9a-z|]+)\)\s+_strix_gfx="),
    (_STACK_PY, r"^\s*_strix_gfx\s*=\s*\{([^}]*)\}"),
]
_STRIX_IDS = [p.name for p, _r in _STRIX_SITES]


@pytest.mark.parametrize("source_path,pattern", _STRIX_SITES, ids = _STRIX_IDS)
def test_the_strix_reroute_names_no_unsupported_arch(source_path, pattern):
    src = source_path.read_text(encoding = "utf-8")
    hits = re.findall(pattern, src, re.MULTILINE)
    assert hits, f"{source_path.name}: the Strix reroute arm was not found; was it renamed?"
    named = " ".join(hits)
    for arch in _UNSUPPORTED_ARCHES:
        assert arch not in named, (
            f"{source_path.name}: {arch} reached the Strix per-arch reroute, which "
            f"builds a repo.amd.com URL without going through the family map"
        )
    # Positive control: the arm really does name the arches it is supposed to route.
    assert "gfx1151" in named, f"{source_path.name}: extraction matched nothing useful"


# ── The CPU summary must blame the card the fallback is actually about ───────

_SUMMARY_GUARD_ANCHOR = "_covered_disp_gfx=$(_infer_linux_amd_gfx_arch"


def _summary_guard_snippet() -> str:
    """The peer guard as install.sh ships it. Extracted rather than retyped, so gutting
    it there fails here."""
    src = _INSTALL_SH.read_text(encoding = "utf-8").replace("\r\n", "\n")
    start = src.find(_SUMMARY_GUARD_ANCHOR)
    assert start != -1, "install.sh: the CPU summary's peer guard was not found"
    end = src.find('if [ -n "$_unsup_disp_gfx" ]; then', start)
    assert end != -1, "install.sh: the guard no longer feeds the unsupported-card arm"
    return src[start:end]


def _run_summary_guard(tmp_path, lspci_lines: "list[str]") -> str:
    """Which of the two CPU-summary arms wins on a host whose lspci says this."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    funcs = "\n".join(
        _sh_function_body(source, name)
        for name in (
            "_infer_amd_gfx_arch_from_gpu_name",
            "_infer_unsupported_amd_gfx_arch_from_gpu_name",
            "_infer_linux_unsupported_amd_gfx_arch",
            "_infer_linux_amd_gfx_arch",
            "_amd_arch_index_family_for_gfx",
            "_amd_probe_arches",
            "_amd_agreed_index_family",
            "_amd_sole_index_arch",
            "_amd_gpu_present_via_pci",
        )
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    fixture = tmp_path / "lspci.txt"
    fixture.write_text("\n".join(lspci_lines) + "\n", encoding = "utf-8")
    lspci = bin_dir / "lspci"
    lspci.write_text(f'#!/bin/sh\ncat "{fixture}"\n', encoding = "utf-8")
    lspci.chmod(0o755)
    script = (
        f"{funcs}\n{_summary_guard_snippet()}\n"
        'if [ -n "$_unsup_disp_gfx" ]; then echo "UNCOVERED $_unsup_disp_gfx"; '
        "else echo GENERIC; fi\n"
    )
    out = subprocess.run(
        ["sh", "-c", script],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 60,
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin"},
    )
    return out.stdout.strip()


_RX_5700 = (
    "01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. "
    "[AMD/ATI] Navi 10 [Radeon RX 5700/5700 XT] [1002:731f]"
)
_RX_580 = (
    "01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. "
    "[AMD/ATI] Ellesmere [Radeon RX 580] [1002:67df]"
)
_RX_7900 = (
    "02:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. "
    "[AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX] [1002:744c]"
)


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh on this host")
class TestInstallShCpuSummaryBlamesTheRightCard:
    """The end-of-run summary reaches its lspci lookup even when the arch read fine, so
    unlike the arm in get_torch_index_url it has no empty-probe gate. An RX 5700 beside
    an RX 7900 lands on CPU whenever the 7900's ROCm is too old, and naming the 5700
    there would replace the upgrade advice with advice false for the card at fault."""

    @pytest.mark.parametrize("lines", [[_RX_5700], [_RX_580]], ids = ["rx5700", "rx580"])
    def test_a_lone_uncovered_card_is_still_named(self, tmp_path, lines):
        """The positive control, and the case the whole PR exists for."""
        verdict = _run_summary_guard(tmp_path, lines)
        assert verdict.startswith(
            "UNCOVERED"
        ), f"install.sh's CPU summary no longer names the uncovered card: {verdict!r}"

    @pytest.mark.parametrize(
        "lines",
        [[_RX_5700, _RX_7900], [_RX_7900, _RX_5700], [_RX_580, _RX_7900]],
        ids = ["5700-first", "7900-first", "580-plus-7900"],
    )
    def test_a_covered_peer_keeps_the_summary_quiet(self, tmp_path, lines):
        verdict = _run_summary_guard(tmp_path, lines)
        assert verdict == "GENERIC", (
            "install.sh's CPU summary blamed an uncovered card on a host that also has "
            f"a covered one, so the real reason for the CPU fallback is hidden: {verdict!r}"
        )

    def test_a_covered_card_alone_says_nothing_about_uncovered_arches(self, tmp_path):
        assert _run_summary_guard(tmp_path, [_RX_7900]) == "GENERIC"


# ── setup.sh's KFD report must blame the right card too ──────────────────────

_SETUP_SH_FUNCS = (
    "_setup_supported_gfx_from_name",
    "_setup_unsupported_gfx_from_name",
    "_setup_unsupported_gfx_any",
)


def _run_setup_report(
    tmp_path,
    lspci_lines: "list[str]",
    mkt: str = "",
) -> str:
    """setup.sh's lookup on the KFD path, where no market name is available and the
    lookup falls back to lspci. Same fixtures as the install.sh summary above."""
    source = _SETUP_SH.read_text(encoding = "utf-8")
    funcs = "\n".join(textwrap.dedent(_sh_function_body(source, name)) for name in _SETUP_SH_FUNCS)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    fixture = tmp_path / "lspci.txt"
    fixture.write_text("\n".join(lspci_lines) + "\n", encoding = "utf-8")
    lspci = bin_dir / "lspci"
    lspci.write_text(f'#!/bin/sh\ncat "{fixture}"\n', encoding = "utf-8")
    lspci.chmod(0o755)
    script = (
        f'{funcs}\nif _g=$(_setup_unsupported_gfx_any "{mkt}"); then echo "UNCOVERED $_g"; '
        "else echo GENERIC; fi\n"
    )
    out = subprocess.run(
        ["sh", "-c", script],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 60,
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin"},
    )
    return out.stdout.strip()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh on this host")
class TestSetupShReportBlamesTheRightCard:
    """The KFD fallback detects the GPU with neither rocminfo nor amd-smi, so the report
    reads lspci and takes the first uncovered adapter. On a host whose RX 5700 enumerates
    before an RX 7900 that is the wrong card: selecting the 7900 and setting its arch
    does reach the supported path, so "no override can help" is false there."""

    @pytest.mark.parametrize("lines", [[_RX_5700], [_RX_580]], ids = ["rx5700", "rx580"])
    def test_a_lone_uncovered_card_is_still_named(self, tmp_path, lines):
        assert _run_setup_report(tmp_path, lines).startswith("UNCOVERED")

    @pytest.mark.parametrize(
        "lines",
        [[_RX_5700, _RX_7900], [_RX_7900, _RX_5700], [_RX_580, _RX_7900]],
        ids = ["5700-first", "7900-first", "580-plus-7900"],
    )
    def test_a_covered_peer_keeps_the_report_quiet(self, tmp_path, lines):
        assert (
            _run_setup_report(tmp_path, lines) == "GENERIC"
        ), "setup.sh named an uncovered card on a host that also carries a covered one"

    @pytest.mark.parametrize(
        "mkt",
        ["", "AMD Radeon Graphics", "Advanced Micro Devices, Inc. [AMD/ATI]"],
        ids = ["kfd-no-name", "generic-name", "vendor-only"],
    )
    def test_an_unmapped_market_name_still_reaches_the_lspci_scan(self, tmp_path, mkt):
        """rocminfo can hand back a nonempty name that maps to nothing. Returning that
        failure ended the lookup, so the lspci scan never ran and the report fell back to
        the plain "AMD ROCm" line this change exists to replace."""
        assert _run_setup_report(tmp_path, [_RX_5700], mkt).startswith(
            "UNCOVERED"
        ), f"an unmapped market name ({mkt!r}) hides an uncovered card lspci can see"

    def test_a_mapped_market_name_short_circuits(self):
        """The name still wins when it maps: no lspci call is needed or made."""
        source = _SETUP_SH.read_text(encoding = "utf-8")
        funcs = "\n".join(
            textwrap.dedent(_sh_function_body(source, name)) for name in _SETUP_SH_FUNCS
        )
        out = subprocess.run(
            ["sh", "-c", f'{funcs}\n_setup_unsupported_gfx_any "AMD Radeon RX 580"\n'],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 30,
            # sh itself still has to be found; the point is that lspci is not on PATH, so a fallback that ran would
            # come back empty and fail this.
            env = {"PATH": os.path.dirname(shutil.which("sh") or "/bin")},
        )
        assert out.stdout.strip() == "gfx803"

    def test_an_unmapped_name_beside_a_covered_peer_stays_quiet(self, tmp_path):
        verdict = _run_setup_report(tmp_path, [_RX_5700, _RX_7900], "AMD Radeon Graphics")
        assert verdict == "GENERIC"

    @pytest.mark.parametrize(
        "lines",
        [[_RX_5700, _RX_7900], [_RX_7900, _RX_5700]],
        ids = ["5700-first", "7900-first"],
    )
    def test_a_named_hit_is_guarded_by_the_peers_too(self, tmp_path, lines):
        """amd-smi reports ONE market name, the first device's, so on a mixed host the
        name itself is the uncovered card. Accepting it before the peer scan put the
        false verdict back through the other door."""
        verdict = _run_setup_report(tmp_path, lines, "AMD Radeon RX 5700 XT")
        assert verdict == "GENERIC", f"a named hit skipped the covered-peer guard: {verdict!r}"

    def test_a_named_hit_survives_a_host_with_no_lspci(self, tmp_path):
        """The guard must not become a silencer: with no adapter list there is no peer to
        find, and the single-card host this report exists for still has to be told."""
        source = _SETUP_SH.read_text(encoding = "utf-8")
        funcs = "\n".join(
            textwrap.dedent(_sh_function_body(source, name)) for name in _SETUP_SH_FUNCS
        )
        out = subprocess.run(
            [
                "sh",
                "-c",
                f'{funcs}\n_setup_unsupported_gfx_any "AMD Radeon RX 5700 XT"\n',
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 30,
            env = {"PATH": os.path.dirname(shutil.which("sh") or "/bin")},
        )
        assert out.stdout.strip() == "gfx1010"

    def test_the_supported_matcher_still_answers(self, tmp_path):
        """Positive control on the extracted table: without it the guard above would be
        vacuous, since a matcher that never matches also keeps the report quiet."""
        source = _SETUP_SH.read_text(encoding = "utf-8")
        script = (
            textwrap.dedent(_sh_function_body(source, "_setup_supported_gfx_from_name"))
            + '\n_setup_supported_gfx_from_name "AMD Radeon RX 7900 XTX"\n'
        )
        out = subprocess.run(["sh", "-c", script], stdout = subprocess.PIPE, text = True, timeout = 30)
        assert out.stdout.strip() == "gfx1100"


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh on this host")
@pytest.mark.parametrize(
    "url,family,pinned",
    [
        ("", "", False),
        ("   ", "", False),
        ("\t\n ", " ", False),
        ("https://example/gfx1010", "", True),
        ("", "rocm7.2", True),
    ],
    ids = ["unset", "spaces", "mixed-blank", "url", "family"],
)
def test_setup_sh_treats_a_blank_index_pin_as_unset(url, family, pinned):
    """get_torch_index_url trims both variables and treats a blank one as unset, so a
    blank value here must not suppress the CPU-only warning. The two lines that decide
    it are taken from setup.sh rather than retyped."""
    src = _SETUP_SH.read_text(encoding = "utf-8").replace("\r\n", "\n")
    start = src.index('_setup_unsup_pin="${UNSLOTH_TORCH_INDEX_URL')
    end = src.index('if [ -n "$_setup_unsup_pin" ]', start)
    snippet = textwrap.dedent(src[start:end])
    out = subprocess.run(
        [
            "sh",
            "-c",
            f'{snippet}\nif [ -n "$_setup_unsup_pin" ]; then echo PINNED; else echo UNSET; fi\n',
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 30,
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "UNSLOTH_TORCH_INDEX_URL": url,
            "UNSLOTH_TORCH_INDEX_FAMILY": family,
        },
    )
    assert out.stdout.strip() == ("PINNED" if pinned else "UNSET"), (
        f"setup.sh read UNSLOTH_TORCH_INDEX_URL={url!r} / _FAMILY={family!r} as "
        f"{out.stdout.strip()}"
    )


# ── The five unsupported tables must agree, not merely exist ─────────────────

_PARITY_NAMES = [
    # RDNA 1 (Navi 10 / 12 / 14), the #8529 cluster
    "AMD Radeon RX 5700 XT",
    "AMD Radeon RX 5700",
    "AMD Radeon RX 5600 XT",
    "AMD Radeon Pro 5600 XT",
    "AMD Radeon Pro 5700 XT",
    "AMD Radeon Pro W5700",
    "AMD Radeon Pro V520",
    "AMD Radeon Pro 5600M",
    "AMD Radeon RX 5500 XT",
    "AMD Radeon RX 5300",
    "AMD Radeon Pro W5500",
    "AMD Radeon Pro W5300",
    # Polaris 10/20/30, the #8458 cluster
    "AMD Radeon RX 580",
    "AMD Radeon RX 570",
    "AMD Radeon RX 590",
    "AMD Radeon RX 480",
    "AMD Radeon RX 470",
    "AMD Radeon Pro WX 7100",
    "AMD Radeon Pro WX 5100",
    # Must map to NOTHING in every copy
    "AMD Radeon RX 5800",
    "AMD Radeon RX 5900",
    "AMD Radeon RX 4800",
    "AMD Radeon RX 540",
    "AMD Radeon RX 550",
    "AMD Radeon RX 560",
    "AMD Radeon RX 460",
    "AMD Radeon RX 7900 XTX",
    "AMD Radeon RX 9070 XT",
    "AMD Radeon RX 6800 XT",
    "AMD Radeon PRO W7500",
    "AMD Radeon 890M Graphics",
    "AMD Radeon 780M Graphics",
    "AMD Instinct MI210",
    "AMD Radeon VII",
    "AMD Radeon Graphics",
    "Intel Arc A770",
    "NVIDIA GeForce RTX 4090",
    "",
]


def _py_unsupported(names):
    return [stack_mod._unsupported_gfx_arch_from_gpu_name(n) or "" for n in names]


def _sh_unsupported(path: Path, func: str, names):
    body = textwrap.dedent(_sh_function_body(path.read_text(encoding = "utf-8"), func))
    script = f'{body}\nfor n in "$@"; do {func} "$n" || echo ""; done\n'
    out = subprocess.run(
        ["sh", "-c", script, "sh", *names],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 60,
    )
    return [l.strip() for l in out.stdout.split("\n")][: len(names)]


def _ps_unsupported(path: Path, names):
    src = path.read_text(encoding = "utf-8").replace("\r\n", "\n")
    block = _ps_block(src, "$unsupportedNameArchTable = @(", "(", ")")
    probes = ", ".join("'" + n.replace("'", "''") + "'" for n in names)
    script = (
        f"{block}\n"
        f"foreach ($n in @({probes})) {{\n"
        f"  $hit = ''\n"
        f"  foreach ($row in $unsupportedNameArchTable) {{\n"
        f"    if ($n -match $row.P) {{ $hit = $row.A; break }}\n"
        f"  }}\n"
        f"  Write-Output $hit\n"
        f"}}\n"
    )
    # run_pwsh, not subprocess.run: this run's stdout is compared line by line against the Python table, so an
    # interpreter that died mid-list would show up as the PowerShell name-to-arch rows disagreeing.
    # See tests/_shared/unsloth_pwsh_runner.py.
    out = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 180,
    )
    assert out.returncode == 0, f"{path.name}: the unsupported table did not evaluate"
    return [l.strip() for l in out.stdout.split("\n")][: len(names)]


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh on this host")
def test_the_five_unsupported_tables_agree_on_every_name():
    """Five hand-kept copies of one table is the standing risk in this area, and the
    supported tables are already pinned to each other. The unsupported ones were not:
    a row added to a single file (RX 540 -> gfx803 in one of the five) passed the whole
    suite. Compare by EVALUATION, in each source's own language, so a copy that spells
    the same rule differently still counts as agreeing."""
    names = _PARITY_NAMES
    answers = {
        "install_python_stack.py": _py_unsupported(names),
        "install.sh": _sh_unsupported(
            _INSTALL_SH, "_infer_unsupported_amd_gfx_arch_from_gpu_name", names
        ),
        "setup.sh": _sh_unsupported(_SETUP_SH, "_setup_unsupported_gfx_from_name", names),
    }
    if shutil.which("pwsh"):
        answers["install.ps1"] = _ps_unsupported(_INSTALL_PS1, names)
        answers["setup.ps1"] = _ps_unsupported(_SETUP_PS1, names)

    for source, got in answers.items():
        assert len(got) == len(names), f"{source}: {len(got)} answers for {len(names)} names"

    disagreements = []
    for i, name in enumerate(names):
        seen = {src: got[i] for src, got in answers.items()}
        if len(set(seen.values())) > 1:
            disagreements.append((name, seen))
    assert not disagreements, "the unsupported-arch tables have drifted apart:\n" + "\n".join(
        f"  {n!r}: {s}" for n, s in disagreements
    )
    # Positive control: the corpus really does exercise the tables.
    ref = answers["install.sh"]
    assert ref[names.index("AMD Radeon RX 5700 XT")] == "gfx1010"
    assert ref[names.index("AMD Radeon RX 580")] == "gfx803"
    assert ref[names.index("AMD Radeon RX 7900 XTX")] == ""
