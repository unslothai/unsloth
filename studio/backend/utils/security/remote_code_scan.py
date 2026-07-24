# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static scan of a model's ``auto_map`` remote code, for the consent gate.

When a user opts into ``trust_remote_code``, the repo's ``auto_map`` Python
(``modeling_*.py`` etc.) is scanned BEFORE execution and suspicious patterns are
surfaced to inform consent. A warning aid, not a hard boundary: a determined
attacker can obfuscate past regexes, so the job is to raise the bar and inform the
hash-pinned consent. Containment (subprocess/venv) is separate; execution still
requires opt-in.

Single source of truth: ``scripts/scan_packages.py`` (the scanner CI runs via
``security-audit.yml``). We import its ``check_py_file`` so the gate inherits every
CI improvement with no drift; its heuristics are deliberately low-false-positive
(combinations flag, not bare ``subprocess``/``eval``). When ``scripts/`` is absent
(stripped install) we fall back to ``_FALLBACK_PATTERNS`` below; a test asserts the
canonical scanner loads in-repo so the fallback never silently takes over.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger
from utils.hf_cache_settings import active_hf_hub_cache

logger = get_logger(__name__)

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
_SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2}

# Bump on any ruleset change (patterns, severities). A persisted approval records the version
# it was scanned under; the consent cache ignores older-ruleset approvals so the same bytes
# are re-scanned and re-shown instead of silently auto-approved.
SCAN_RULES_VERSION = 1

# Configs that can carry an ``auto_map`` pointing at executable repo ``.py``.
# ``trust_remote_code`` runs code from ANY of these, so scanner and gate must read the
# same set (scanning only config.json/tokenizer would miss a custom-processor VLM).
REMOTE_CODE_CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


class RemoteCodeUnscannable(Exception):
    """The repo's executable code could not be fully fetched/listed to scan.

    Raised (not returned empty) so the gate distinguishes "code PRESENT but unreadable"
    (offline/gated/transient/404/listing failure) -> fail CLOSED, from "repo has NO .py"
    (empty result) -> trust_remote_code is a no-op -> allow. Conflating them would block
    a code-free repo or fail open on code we could not see.
    """


# Fallback patterns (used only if scripts/scan_packages.py is absent): (regex, check,
# severity). A flat subset of the canonical scanner so a stripped install still scans;
# the canonical scanner (imported below) supersedes it whenever the repo is present.
_FALLBACK_PATTERNS: tuple[tuple[re.Pattern, str, str], ...] = (
    (
        re.compile(
            r"\bexec\s*\(\s*(?:urllib|requests|httpx|urlopen)"
            r"|\bexec\s*\([^)]*\.(?:text|content|read)\s*\("
            r"|\beval\s*\([^)]*\.(?:text|content|read)\s*\("
            r"|\b__import__\s*\([^)]*\+",
            re.DOTALL,
        ),
        "loads-and-executes-remote-code",
        CRITICAL,
    ),
    (
        re.compile(
            r"\bsocket\b.*\bconnect\b.*\bsubprocess\b"
            r"|\bsocket\b.*\bconnect\b.*\b(?:sh|bash|cmd)\b"
            r"|\bpty\s*\.\s*spawn\b|\bos\s*\.\s*dup2\s*\(",
            re.DOTALL,
        ),
        "reverse/bind-shell",
        CRITICAL,
    ),
    (
        re.compile(
            r"169\.254\.169\.254|metadata\.google\.internal|/latest/meta-data"
            r"|/metadata/identity|169\.254\.170\.2"
        ),
        "cloud-metadata/IMDS-access",
        CRITICAL,
    ),
    (
        re.compile(
            r"(?:open|Path|read_text|read_bytes)\s*\([^)]*?"
            r"(?:\.ssh[/\\]|\.aws[/\\]|\.kube[/\\]|\.gnupg[/\\]|id_rsa|id_ed25519"
            r"|credentials\.json|\.git-credentials|\.npmrc|\.pypirc|/etc/shadow)"
            r"|(?:open|Path)\(\s*['\"]\.env['\"]\s*[,)]",
            re.DOTALL,
        ),
        "credential-file-access",
        CRITICAL,
    ),
    (
        re.compile(r"/tmp/\S+.*(?:subprocess|os\.system|os\.popen|Popen|chmod.*\+x)", re.DOTALL),
        "tmp-staged-dropper",
        CRITICAL,
    ),
    (
        re.compile(r"\bopenssl\s+(enc|rand|rsautl|pkeyutl|genrsa|dgst|s_client)\b"),
        "openssl-cli-exfil",
        HIGH,
    ),
    (
        re.compile(
            r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
            r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b"
        ),
        "subprocess/os-exec",
        HIGH,
    ),
    # Bare exec()/eval() only; the (?<![\w.]) excludes torch's ``module.eval()``.
    (re.compile(r"(?<![\w.])(?:exec|eval)\s*\("), "exec/eval", HIGH),
    (
        re.compile(
            r"\burllib\.request\b|\burlopen\s*\("
            r"|\brequests\s*\.\s*(get|post|put|patch|delete|head|Session)\b"
            r"|\bhttpx\s*\.\s*(get|post|put|patch|delete|Client|AsyncClient)\b"
            r"|\bsocket\s*\.\s*(socket|create_connection)\b|\bhttp\.client\b"
        ),
        "network-access",
        HIGH,
    ),
    (
        re.compile(
            r"\bmarshal\s*\.\s*(loads|load)\b"
            r"|\bcompile\s*\([^)]*['\"]exec['\"]\s*\)"
            r"|\b__import__\s*\(|\bgetattr\s*\(\s*__builtins__"
            r"|\b(?:b64decode|decodebytes)\s*\(.*(?:b64decode|decodebytes)\s*\(",
            re.DOTALL,
        ),
        "obfuscation",
        HIGH,
    ),
    (
        re.compile(
            r"-----BEGIN\s+(?:RSA\s+)?(?:PUBLIC|PRIVATE|ENCRYPTED|EC|DSA|OPENSSH)\s+KEY-----"
            r"|\bMII[A-Za-z0-9+/]{20,}",
            re.DOTALL,
        ),
        "embedded-key-material",
        HIGH,
    ),
    (
        re.compile(
            r"\bos\.environ\s*\.\s*copy\s*\(|\bdict\s*\(\s*os\.environ\s*\)"
            r"|\bjson\.dumps\s*\(\s*(?:dict\s*\(\s*)?os\.environ",
            re.IGNORECASE,
        ),
        "environment-harvest",
        HIGH,
    ),
    (
        re.compile(
            r"\bbase64\s*\.\s*(b64decode|decodebytes|b32decode|b16decode)\b"
            r"|\bcodecs\s*\.\s*decode\b"
        ),
        "base64/encoding-decode",
        MEDIUM,
    ),
    (re.compile(r"[A-Za-z0-9+/=]{200,}"), "large-base64-blob", MEDIUM),
)


@dataclass
class Finding:
    severity: str
    filename: str
    check: str
    evidence: str = ""
    # 1-based match line + surrounding code window (see _attach_location). For the UI;
    # None/[] when unlocatable.
    line: Optional[int] = None
    snippet: list = field(default_factory = list)


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory = list)
    fingerprint: str = ""

    @property
    def max_severity(self) -> Optional[str]:
        if not self.findings:
            return None
        return min((f.severity for f in self.findings), key = lambda s: _SEVERITY_ORDER[s])

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return "no suspicious patterns found"
        by = {}
        for f in self.findings:
            by.setdefault(f.severity, set()).add(f.check)
        parts = []
        for sev in (CRITICAL, HIGH, MEDIUM):
            if sev in by:
                parts.append(f"{sev}: {', '.join(sorted(by[sev]))}")
        return "; ".join(parts)

    def findings_payload(self) -> list[dict]:
        """Structured findings for the UI: one record per match, with line + snippet."""
        return [
            {
                "severity": f.severity,
                "file": f.filename,
                "check": f.check,
                "evidence": f.evidence,
                "line": f.line,
                "snippet": f.snippet,
            }
            for f in self.findings
        ]


# Canonical scanner: import scripts/scan_packages.py by file path (scripts/ is not an
# importable package from the backend root). It imports only stdlib at module level and
# guards its CLI under __main__, so importing it is side-effect-free.
_CANON_SENTINEL = object()
_canon_cache = _CANON_SENTINEL


def _load_canonical_scanner():
    """Return the ``scripts/scan_packages.py`` module, or None if unavailable."""
    global _canon_cache
    if _canon_cache is not _CANON_SENTINEL:
        return _canon_cache

    module = None
    # Walk up from this file to a repo root that contains scripts/scan_packages.py.
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "scripts" / "scan_packages.py"
        if candidate.is_file():
            try:
                spec = importlib.util.spec_from_file_location("unsloth_scan_packages", candidate)
                mod = importlib.util.module_from_spec(spec)
                sys.modules.setdefault("unsloth_scan_packages", mod)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                if hasattr(mod, "check_py_file"):
                    module = mod
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not load canonical scan_packages.py: %s", exc)
            break

    if module is None:
        logger.warning(
            "scripts/scan_packages.py not found; remote-code scan using the "
            "vendored fallback patterns."
        )
    _canon_cache = module
    return module


# Model-context-strict patterns. The canonical scanner only flags bare
# ``subprocess``/``eval`` in combinations (common in package build scripts), but a
# model's modeling_*.py never legitimately shells out, so the gate flags them alone
# (e.g. a bare ``subprocess.Popen`` in a config ``__init__``).
_MODEL_STRICT_PATTERNS: tuple[tuple[re.Pattern, str, str], ...] = (
    (
        re.compile(
            r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
            r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b"
        ),
        "subprocess/os-exec (model code)",
        HIGH,
    ),
    # Bare exec()/eval(); excludes attribute calls like torch ``module.eval()``.
    (re.compile(r"(?<![\w.])(?:exec|eval)\s*\("), "exec/eval (model code)", HIGH),
)


# Lines of context shown on each side of a flagged line in the dialog snippet.
_SNIPPET_CONTEXT = 3
# Cap a rendered line so a minified/one-line file can't bloat the payload.
_SNIPPET_MAX_LINE = 240
# Canonical evidence is formatted as "L<n>: <source line>" by _extract_evidence.
_EVIDENCE_LINE_RE = re.compile(r"^L(\d+):")


def _snippet_rows(
    content: str,
    line: int,
    col: Optional[int] = None,
    match_len: int = 0,
) -> list[dict]:
    """A `±_SNIPPET_CONTEXT`-line window around `line` (1-based). Rows are
    {number, text, is_match}; the matched row adds match_start/match_end for an inline
    highlight when a precise column span is known."""
    lines = content.splitlines()
    if not lines or line < 1:
        return []
    line = min(line, len(lines))
    lo = max(1, line - _SNIPPET_CONTEXT)
    hi = min(len(lines), line + _SNIPPET_CONTEXT)
    rows: list[dict] = []
    for n in range(lo, hi + 1):
        text = lines[n - 1]
        clipped = len(text) > _SNIPPET_MAX_LINE
        if clipped:
            text = text[:_SNIPPET_MAX_LINE] + " ..."
        row = {"number": n, "text": text, "is_match": n == line}
        if n == line and col is not None and match_len > 0 and not clipped:
            row["match_start"] = col
            row["match_end"] = min(col + match_len, len(text))
        rows.append(row)
    return rows


def _attach_location(
    content: str,
    finding: Finding,
    match: "Optional[re.Match]" = None,
) -> None:
    """Populate finding.line + finding.snippet. A regex match gives a precise
    line+column; canonical findings are located via the `L<n>:` prefix in their evidence."""
    if match is not None:
        before = content[: match.start()]
        line = before.count("\n") + 1
        col = match.start() - (before.rfind("\n") + 1)
        finding.line = line
        finding.snippet = _snippet_rows(content, line, col, match.end() - match.start())
        return
    first = finding.evidence.splitlines()[0] if finding.evidence else ""
    tag = _EVIDENCE_LINE_RE.match(first)
    if tag:
        line = int(tag.group(1))
        finding.line = line
        finding.snippet = _snippet_rows(content, line)


def _scan_content(content: str, filename: str) -> list[Finding]:
    findings: list[Finding] = []

    canon = _load_canonical_scanner()
    if canon is not None:
        # Canonical Finding is (severity, package, filename, check, evidence); adapt to
        # the gate's (severity, filename, check, evidence).
        for f in canon.check_py_file(content, filename, ""):
            finding = Finding(f.severity, f.filename, f.check, (f.evidence or "")[:120])
            _attach_location(content, finding)
            findings.append(finding)
    else:
        for pat, check, sev in _FALLBACK_PATTERNS:
            m = pat.search(content)
            if m:
                finding = Finding(sev, filename, check, m.group(0)[:120])
                _attach_location(content, finding, m)
                findings.append(finding)

    # Augment with the model-context-strict patterns the package scanner omits.
    have = {f.check for f in findings}
    for pat, check, sev in _MODEL_STRICT_PATTERNS:
        if check in have:
            continue
        m = pat.search(content)
        if m:
            finding = Finding(sev, filename, check, m.group(0)[:120])
            _attach_location(content, finding, m)
            findings.append(finding)
    return findings


def scan_remote_code_files(files: dict[str, str]) -> ScanResult:
    """Scan a mapping of {filename: content} and return aggregated findings."""
    result = ScanResult(fingerprint = remote_code_fingerprint(files))
    for name, content in files.items():
        if not name.endswith(".py"):
            continue
        result.findings.extend(_scan_content(content or "", name))
    return result


def remote_code_fingerprint(files: dict[str, str]) -> str:
    """Stable sha256 over the (sorted) file contents, for pinning consent."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update((files[name] or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def repo_remote_code_files(model_name: str, hf_token: Optional[str] = None) -> dict[str, str]:
    """Download a repo's executable ``.py`` (auto_map targets + modeling/config).

    Returns {filename: content}. An EMPTY dict means the repo ships no executable ``.py``
    (trust_remote_code is a no-op). Raises ``RemoteCodeUnscannable`` when code is present
    but cannot be fully fetched/listed (offline/gated/404/listing failure), so the caller
    fails closed rather than fingerprint a partial view of code transformers would run in
    full. The empty-vs-raise split lets the gate allow a code-free repo while still
    blocking unscannable code.
    """
    import json
    from pathlib import Path

    files: dict[str, str] = {}
    try:
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model_name):
            root = Path(normalize_path(model_name)).expanduser()
            # Walk ALL .py, not just the auto_map entry's static import closure. This is
            # DELIBERATE (see the remote-branch note): the entry can reach a sibling via
            # an absolute import, importlib, or exec, which a relative-import closure
            # misses, so closure-only scanning is a real bypass. Broad scan never
            # under-scans; the cost is a benign script can over-block, the safe direction
            # for an RCE gate (HIGH stays approvable; only CRITICAL hard-blocks).
            for p in root.rglob("*.py"):
                if p.is_file():
                    files[str(p.relative_to(root))] = p.read_text(errors = "replace")
            # A local config can still point auto_map at an EXTERNAL Hub repo
            # (owner/name--module.Class) that executes on load, so fetch it. Every config
            # that can declare auto_map is checked, so a custom processor's external code
            # is not missed.
            ext_refs = set()
            for name in REMOTE_CODE_CONFIG_FILES:
                p = root / name
                if p.is_file():
                    try:
                        ext_refs |= _auto_map_refs(json.loads(p.read_text()))
                    except Exception:
                        pass
            if not _add_external_refs(files, ext_refs, hf_token, model_name):
                raise RemoteCodeUnscannable(f"{model_name}: external auto_map code unreachable")
            return files

        from huggingface_hub import hf_hub_download, list_repo_files
        from huggingface_hub.utils import EntryNotFoundError

        # Collect auto_map refs from EVERY config that can declare one. A 404
        # (EntryNotFoundError) means the config is absent -> skip; any other failure is
        # transient/auth and could hide an auto_map, so fail closed (unscannable).
        refs = set()
        for cfg_name in REMOTE_CODE_CONFIG_FILES:
            try:
                cfg_path = hf_hub_download(
                    model_name,
                    cfg_name,
                    token = hf_token,
                    cache_dir = active_hf_hub_cache(),
                )
            except EntryNotFoundError:
                continue
            except Exception as exc:
                raise RemoteCodeUnscannable(
                    f"{model_name}: config {cfg_name} could not be fetched ({exc})"
                ) from exc
            try:
                refs |= _auto_map_refs(json.loads(Path(cfg_path).read_text()))
            except Exception:
                pass
        own_refs = {fn for repo, fn in refs if repo is None}
        # The full file list catches helper .py the auto_map code imports but does not
        # name. If we cannot list the repo, an imported module could be missed and the
        # fingerprint cover less than transformers runs, so fail closed (unscannable).
        try:
            repo_files = list_repo_files(model_name, token = hf_token)
        except Exception as exc:
            raise RemoteCodeUnscannable(f"{model_name}: could not list repo files ({exc})") from exc
        repo_file_set = set(repo_files)
        # Scan every present .py PLUS own-repo auto_map targets that ACTUALLY EXIST in
        # this revision. Scanning EVERY .py (not just the closure) is DELIBERATE: the
        # entry can reach a sibling via absolute import / importlib / exec, which a
        # relative-import closure misses, so closure-only scanning is a real bypass.
        # Broad scan never under-scans; the cost is a benign script can over-block, the
        # safe direction for an RCE gate (HIGH approvable; only CRITICAL hard-blocks). An
        # auto_map target absent from the listing is a STALE ref (an older config naming a
        # since-removed file, e.g. unsloth/PaddleOCR-VL names processing_ppocrvl.py but
        # ships processing_paddleocr_vl.py). transformers cannot execute an absent file,
        # so drop the stale ref rather than fail closed; present .py are still fully
        # scanned. This also absorbs a mis-derived dotted name (sub.mod.py vs sub/mod.py):
        # the bad name drops as stale while the real present file is scanned.
        present_py = {f for f in repo_files if f.endswith(".py")}
        stale_refs = own_refs - repo_file_set
        for fn in sorted(stale_refs):
            logger.info(
                "repo_remote_code_files(%s): ignoring stale own-repo auto_map target "
                "%s (absent from the repo listing; it cannot execute)",
                model_name,
                fn,
            )
        wanted = present_py | (own_refs & repo_file_set)
        for fn in sorted(wanted):
            try:
                fp = hf_hub_download(
                    model_name,
                    fn,
                    token = hf_token,
                    cache_dir = active_hf_hub_cache(),
                )
            except Exception as exc:
                # A .py CONFIRMED PRESENT could not be fetched. A partial set would
                # fingerprint "clean" while transformers later runs this file, so fail
                # closed. (Stale/absent refs were dropped above, so this only fires on a
                # present-file fetch failure.)
                raise RemoteCodeUnscannable(
                    f"{model_name}: present file {fn} could not be fetched ({exc})"
                ) from exc
            files[fn] = Path(fp).read_text(errors = "replace")
        # Code referenced from another repo executes too: scan it or fail closed.
        if not _add_external_refs(files, refs, hf_token, model_name):
            raise RemoteCodeUnscannable(f"{model_name}: external auto_map code unreachable")
    except RemoteCodeUnscannable:
        logger.warning("repo_remote_code_files(%s): unscannable; failing closed", model_name)
        raise
    except Exception as exc:
        # An unexpected error mid-scan means we could not complete it -> unscannable.
        raise RemoteCodeUnscannable(f"{model_name}: scan failed ({exc})") from exc
    # An empty dict here means the listing succeeded and the repo ships no executable .py
    # (nor fetchable external refs) -> trust_remote_code is a no-op for the caller.
    return files


def _iter_auto_map_strings(value):
    """Yield every string class-ref inside one ``auto_map`` value.

    A value is a bare string (``"modeling_x.Cls"``) or a list/tuple (transformers encodes
    a tokenizer as ``"AutoTokenizer": [slow, fast]``, possibly nested or with nulls).
    Flatten all forms so external tokenizer code in the list shape is scanned.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_auto_map_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_auto_map_strings(item)


def _auto_map_refs(cfg: dict) -> set:
    """``(repo, filename)`` pairs referenced by config auto_map.

    ``repo`` is ``None`` for own-repo code. An external ``owner/name--module.Class`` ref
    (transformers' cross-repo form) yields ``("owner/name", "module.py")`` so cross-repo
    code is scanned + fingerprinted too.
    """
    out = set()
    am = cfg.get("auto_map") or {}
    if isinstance(am, dict):
        for value in am.values():
            # A value may be a string OR a [slow, fast] list (tokenizers); cover both.
            for ref in _iter_auto_map_strings(value):
                # ref like "modeling_deepseekocr.Cls" or "owner/name--modeling.Cls"
                if "." not in ref:
                    continue
                module = ref.rsplit(".", 1)[0]  # drop trailing .ClassName
                if "--" in module:
                    repo, mod = module.split("--", 1)
                    out.add((repo or None, mod + ".py"))
                else:
                    out.add((None, module + ".py"))
    return out


def _auto_map_py(cfg: dict) -> set[str]:
    """Own-repo ``.py`` filenames referenced by auto_map (external refs excluded)."""
    return {fn for repo, fn in _auto_map_refs(cfg) if repo is None}


def external_auto_map_repos(model_name: str, hf_token: Optional[str] = None) -> set:
    """External Hub repos referenced by any of this model's auto_map configs.

    The ``owner/name`` repos ``_add_external_refs`` downloads. The scan route uses this so
    declining consent purges them too, not leaving untrusted external code cached.
    Best-effort, config/metadata-only: returns whatever can be read, never raises.
    """
    repos: set = set()
    try:
        import json
        from pathlib import Path

        from utils.paths import is_local_path, normalize_path

        if is_local_path(model_name):
            root = Path(normalize_path(model_name)).expanduser()
            for cfg_name in REMOTE_CODE_CONFIG_FILES:
                p = root / cfg_name
                if not p.is_file():
                    continue
                try:
                    refs = _auto_map_refs(json.loads(p.read_text()))
                except Exception:
                    continue
                repos.update(repo for repo, _fn in refs if repo)
            return repos

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        for cfg_name in REMOTE_CODE_CONFIG_FILES:
            try:
                cfg_path = hf_hub_download(
                    model_name,
                    cfg_name,
                    token = hf_token,
                    cache_dir = active_hf_hub_cache(),
                )
            except EntryNotFoundError:
                continue
            except Exception:
                continue
            try:
                refs = _auto_map_refs(json.loads(Path(cfg_path).read_text()))
            except Exception:
                continue
            repos.update(repo for repo, _fn in refs if repo)
    except Exception:
        return repos
    return repos


def _add_external_refs(files: dict, refs, hf_token, model_name: str) -> bool:
    """Download external-repo auto_map code into ``files`` (keyed ``repo--file``).

    transformers fetches the entry file AND its relative imports from the same external
    repo, so scanning only the entry would miss code in a ``helper.py`` it imports. Mirror
    the own-repo path: enumerate each external repo's ``.py`` and scan the whole set (plus
    the referenced entry files). Returns False if any external repo cannot be listed or a
    file cannot be fetched, so the caller fails closed.
    """
    # Group the explicit entry refs by external repo.
    entries: dict = {}
    for repo, fn in refs:
        if repo is None:
            continue
        entries.setdefault(repo, set()).add(fn)

    if not entries:
        return True

    from pathlib import Path

    from huggingface_hub import hf_hub_download, list_repo_files

    for repo, entry_files in entries.items():
        try:
            repo_files = list_repo_files(repo, token = hf_token)
        except Exception as exc:
            logger.warning(
                "repo_remote_code_files(%s): external repo %s unlistable (%s); failing closed",
                model_name,
                repo,
                exc,
            )
            return False
        # The loader's executable closure = every present .py plus any referenced entry
        # file. With a REAL (non-empty) listing, present_py covers the code, so an entry
        # ref absent from it is stale/mis-derived and is dropped rather than failing
        # closed (like the own-repo path). With an EMPTY listing we cannot prove the ref
        # stale, so keep fetching it and fail closed if unreachable; never under-scan. A
        # PRESENT file that cannot be fetched still fails closed below.
        repo_file_set = set(repo_files)
        present_py = {f for f in repo_files if f.endswith(".py")}
        if repo_file_set:
            for fn in sorted(set(entry_files) - repo_file_set):
                logger.info(
                    "repo_remote_code_files(%s): ignoring stale external auto_map target "
                    "%s:%s (absent from the repo listing; it cannot execute)",
                    model_name,
                    repo,
                    fn,
                )
            wanted = present_py | (set(entry_files) & repo_file_set)
        else:
            wanted = present_py | set(entry_files)
        for fn in sorted(wanted):
            try:
                fp = hf_hub_download(
                    repo,
                    fn,
                    token = hf_token,
                    cache_dir = active_hf_hub_cache(),
                )
            except Exception as exc:
                logger.warning(
                    "repo_remote_code_files(%s): external %s:%s unscannable (%s)",
                    model_name,
                    repo,
                    fn,
                    exc,
                )
                return False
            files[f"{repo}--{fn}"] = Path(fp).read_text(errors = "replace")
    return True
