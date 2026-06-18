# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static scan of a model's ``auto_map`` remote code, for the consent gate.

When a user opts into ``trust_remote_code`` for a model whose architecture is
defined by Python shipped in its HF repo (``config.json`` ``auto_map`` ->
``modeling_*.py`` / ``configuration_*.py``), we scan that code BEFORE executing
it and surface any suspicious patterns so the consent decision is informed.

This is a *warning aid*, not a hard boundary -- a determined attacker can
obfuscate past regexes. Its job is to raise the bar and inform the explicit,
hash-pinned user consent. Containment is provided separately (subprocess/venv
isolation); execution still only happens with the user's opt-in.

Single source of truth: the detection logic is the repo's canonical
supply-chain auditor, ``scripts/scan_packages.py`` -- the exact scanner CI runs
(``security-audit.yml``). We import its ``check_py_file`` so the load-time gate
inherits every improvement to the CI scanner with no drift. The combination
heuristics there are deliberately low-false-positive (bare ``subprocess`` /
``eval`` alone are not flagged; staged-payload / reverse-shell / IMDS
combinations are). When ``scripts/`` is not present on disk (a stripped
packaged install), we fall back to the vendored ``_FALLBACK_PATTERNS`` below; a
test asserts the canonical scanner loads in-repo so the fallback never silently
takes over in the canonical environment.
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

logger = get_logger(__name__)

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
_SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2}

# Single source of truth for the configs that can carry an ``auto_map`` pointing at
# executable repo ``.py``. ``trust_remote_code`` runs code referenced from ANY of
# these (model architecture, tokenizer, image/feature processor, processor, video
# processor), so both the scanner and the consent gate (which imports this) must
# read the same set -- scanning only config.json/tokenizer would miss a
# custom-processor VLM entirely. Keep this list and the gate in lockstep.
REMOTE_CODE_CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


class RemoteCodeUnscannable(Exception):
    """The repo's executable code could not be fully fetched/listed to scan.

    Raised (not returned empty) so the consent gate can tell two very different
    outcomes apart: code is PRESENT but we could not read all of it (offline /
    gated / transient / a present .py that 404s / a repo-listing failure) -> fail
    CLOSED; versus the repo simply has NO executable .py (an empty result) ->
    ``trust_remote_code`` is a no-op -> allow. Conflating them would either block a
    legitimate code-free repo (e.g. a GGUF repo whose config.json carries a
    vestigial auto_map) or fail open on code we could not see.
    """


# --- Fallback patterns (only used if scripts/scan_packages.py is absent) -----
# (regex, check-name, severity). A flat subset of the canonical scanner, kept so
# a stripped packaged install without scripts/ still scans. The canonical
# scanner (imported below) supersedes this whenever the repo is present.
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
    # Bare exec()/eval() only; exclude attribute calls like torch's
    # ``module.eval()`` (eval-mode), which are ubiquitous false positives.
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
    # 1-based line of the match, and a small surrounding code window (see
    # _attach_location). Populated for the UI; None/[] when unlocatable.
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
        """Structured findings for the UI: one record per match, including the
        line number and a small surrounding code window for the dialog."""
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


# --- Canonical scanner: import scripts/scan_packages.py (the CI scanner) ------
# Loaded by file path (scripts/ is not an importable package from the backend
# root). scan_packages.py imports only stdlib at module level (requests/etc. are
# lazy) and guards its CLI under ``if __name__ == "__main__"``, so importing it
# is side-effect-free and dependency-light.
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


# Model-context-strict patterns. The canonical scanner is tuned for *package*
# supply chain, where a bare ``subprocess`` (build scripts) or ``eval`` is
# common, so it only flags them in combinations. In a model's modeling_*.py
# none of these have a legitimate place -- a forward pass never shells out --
# so the load-time gate flags them on their own. This catches, for example, a
# bare ``subprocess.Popen`` sitting in a config ``__init__``.
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
    """A `±_SNIPPET_CONTEXT`-line window around `line` (1-based). Each row is
    {number, text, is_match}; the matched row also carries match_start/match_end
    for an inline highlight when a precise column span is known."""
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
    """Populate finding.line + finding.snippet. A regex match (model-strict /
    fallback) gives a precise line+column; canonical findings are located via the
    `L<n>:` prefix their evidence already carries."""
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
        # Canonical Finding is (severity, package, filename, check, evidence);
        # adapt to the gate's (severity, filename, check, evidence).
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

    Returns {filename: content}. An EMPTY dict means the repo genuinely ships no
    executable ``.py`` (so ``trust_remote_code`` is a no-op). Raises
    ``RemoteCodeUnscannable`` when code is present but cannot be fully fetched or
    listed (offline/gated/transient, a referenced ``.py`` that 404s, or a
    repo-listing failure that could hide an imported helper), so the caller can fail
    closed rather than fingerprint a partial view of code transformers would later
    execute in full. The empty-vs-raise split is what lets the gate allow a code-free
    repo while still blocking truly unscannable code.
    """
    import json
    from pathlib import Path

    files: dict[str, str] = {}
    try:
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model_name):
            root = Path(normalize_path(model_name)).expanduser()
            # Walk ALL .py recursively (relative-path keys): a modeling_*.py can
            # import nested helper modules that execute on a trust_remote_code
            # load, so they must be scanned + fingerprinted too.
            for p in root.rglob("*.py"):
                if p.is_file():
                    files[str(p.relative_to(root))] = p.read_text(errors = "replace")
            # A local config can still point auto_map at an EXTERNAL Hub repo
            # (owner/name--module.Class); that code executes on load, so fetch it.
            # Every config that can declare auto_map is checked, not just the model
            # + tokenizer, so a custom processor's external code is not missed.
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

        # Collect auto_map refs from EVERY config that can declare one (model,
        # tokenizer, image/feature processor, processor, video processor). A genuine
        # 404 (EntryNotFoundError) means the config is absent -> skip it; any other
        # failure is transient/auth and could hide an auto_map we must scan, so fail
        # closed (unscannable) rather than fingerprint an incomplete view.
        refs = set()
        for cfg_name in REMOTE_CODE_CONFIG_FILES:
            try:
                cfg_path = hf_hub_download(model_name, cfg_name, token = hf_token)
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
        # The full file list is needed to catch helper .py the auto_map code imports
        # but does not name. If we cannot list the repo, an imported module could be
        # missed and the fingerprint would cover less than transformers executes, so
        # fail closed (unscannable) rather than scan a partial set.
        try:
            repo_files = list_repo_files(model_name, token = hf_token)
        except Exception as exc:
            raise RemoteCodeUnscannable(f"{model_name}: could not list repo files ({exc})") from exc
        repo_file_set = set(repo_files)
        # Scan every present .py (the import closure the loader can execute) PLUS the
        # own-repo auto_map targets that ACTUALLY EXIST in this revision. An auto_map
        # target that is absent from the listing is a STALE ref -- an older config
        # pointing at a file the repo no longer ships (e.g. unsloth/PaddleOCR-VL's
        # tokenizer_config.json names processing_ppocrvl.py, but the repo ships
        # processing_paddleocr_vl.py). transformers cannot execute a file that is not
        # there, so ignore the stale ref rather than fail the whole repo closed; the
        # present .py are still fully scanned, which is the stronger coverage. This
        # also absorbs a mis-derived dotted-module filename (sub.mod.py vs sub/mod.py):
        # the bad name is dropped as stale while the real present file is scanned.
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
                fp = hf_hub_download(model_name, fn, token = hf_token)
            except Exception as exc:
                # A .py CONFIRMED PRESENT in the listing could not be fetched. Returning
                # the partial set would fingerprint "clean" code while transformers
                # later fetches and runs this file. Fail closed (unscannable) so the
                # caller blocks. (Stale/absent refs were dropped above and never reach
                # here, so this only fires on a present-file fetch failure.)
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
        # An UNEXPECTED error mid-scan means we could not complete it -> unscannable.
        raise RemoteCodeUnscannable(f"{model_name}: scan failed ({exc})") from exc
    # Reaching here with an empty dict means the listing succeeded and the repo
    # genuinely ships no executable .py (and no fetchable external refs) -> no remote
    # code -> the caller treats trust_remote_code as a no-op.
    return files


def _iter_auto_map_strings(value):
    """Yield every string class-ref inside one ``auto_map`` value.

    A value can be a bare string (``"modeling_x.Cls"``) OR a list/tuple --
    transformers encodes a tokenizer as ``"AutoTokenizer": [slow, fast]`` (and may
    nest), e.g. ``["owner/repo--tokenization_x.Slow", "owner/repo--tokenization_x.Fast"]``
    or ``[..., null]``. Flatten all forms so external tokenizer code in the list
    shape is scanned, not silently trusted.
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

    ``repo`` is ``None`` for code in the model's own repo. An external
    ``owner/name--module.Class`` ref (transformers' cross-repo form) yields
    ``("owner/name", "module.py")`` so the code transformers fetches and executes
    from another repo is scanned + fingerprinted too, not silently trusted.
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


def _add_external_refs(files: dict, refs, hf_token, model_name: str) -> bool:
    """Download external-repo auto_map code into ``files`` (keyed ``repo--file``).

    Returns False if any referenced external file cannot be fetched, so the
    caller fails closed rather than fingerprinting code it could not scan.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    for repo, fn in refs:
        if repo is None:
            continue
        try:
            fp = hf_hub_download(repo, fn, token = hf_token)
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
