#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Guard against breaking npm dependency removals in studio/frontend.

Diffs the current package.json against a git base, finds every package
that was removed, and confirms each is no longer referenced anywhere
in the repo. If a removed package is still imported and is not
transitively resolvable through the new lockfile, exits non-zero with
file:line citations.

Usage:
  python scripts/check_frontend_dep_removal.py
  python scripts/check_frontend_dep_removal.py --base origin/main
  python scripts/check_frontend_dep_removal.py --base HEAD~1
  python scripts/check_frontend_dep_removal.py --base-pkg PATH --head-lock PATH

Exit codes:
  0  every removed dep is safe (no source refs or still resolvable)
  1  at least one removed dep is referenced and not resolvable
  2  invocation error (bad args, missing file, git error)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_PKG = "studio/frontend/package.json"
FRONTEND_LOCK = "studio/frontend/package-lock.json"

DEP_FIELDS = (
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
)

# Files where seeing a package name does NOT count as usage.
EXPECTED_NOISE_FILES = {
    "studio/frontend/package.json",
    "studio/frontend/package-lock.json",
    "studio/backend/core/data_recipe/oxc-validator/package.json",
    "studio/backend/core/data_recipe/oxc-validator/package-lock.json",
}

# File types where a quoted string can be a module specifier.
JS_LIKE_EXT = re.compile(
    r"\.(ts|tsx|js|jsx|mjs|cjs|html|htm|css|scss|sass|json|jsonc)$"
)
# Files where JS import patterns could be a real module reference (.mdx is
# real ESM; .md code fences are not).
SCRIPT_LIKE_EXT = re.compile(r"\.(ts|tsx|js|jsx|mjs|cjs|mdx)$")
STYLE_EXT = re.compile(r"\.(css|scss|sass)$")
HTML_EXT = re.compile(r"\.(html|htm)$")
TS_LIKE_EXT = re.compile(r"\.(ts|tsx|mts|cts|mdx)$")
# Files where a removed package's CLI binary could be invoked.
COMMAND_LIKE_EXT = re.compile(r"(\.(ya?ml|sh|ps1|bat)$|(^|/)Dockerfile[^/]*$)")

GREP_INCLUDES = [
    "--include=*.ts",
    "--include=*.tsx",
    "--include=*.js",
    "--include=*.jsx",
    "--include=*.mjs",
    "--include=*.cjs",
    "--include=*.html",
    "--include=*.htm",
    "--include=*.css",
    "--include=*.scss",
    "--include=*.sass",
    "--include=*.json",
    "--include=*.jsonc",
    "--include=*.md",
    "--include=*.mdx",
    "--include=*.py",
    "--include=*.rs",
    "--include=*.toml",
    "--include=*.yml",
    "--include=*.yaml",
    "--include=*.sh",
    "--include=*.ps1",
    "--include=*.bat",
    "--include=Dockerfile*",
]
GREP_EXCLUDES = [
    "--exclude-dir=node_modules",
    "--exclude-dir=dist",
    "--exclude-dir=.git",
    "--exclude-dir=__pycache__",
    "--exclude-dir=target",
    "--exclude-dir=.next",
    "--exclude-dir=build",
    "--exclude-dir=.venv",
    "--exclude-dir=venv",
]

# A pip-installed playwright ref is the PyPI package, not npm.
PIP_PLAYWRIGHT = re.compile(
    r"(pip\s+install\s+['\"]?playwright"
    r"|python\s+-m\s+playwright"
    r"|from\s+playwright"
    r"|^\s*import\s+playwright)"
)


@dataclass
class Hit:
    file: str
    line: int
    kind: str
    snippet: str


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command, return stdout. On non-zero exit, return ''."""
    res = subprocess.run(
        cmd,
        cwd = cwd or REPO_ROOT,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    return res.stdout if res.returncode == 0 else ""


def read_pkg_at(base: str, path: str) -> dict:
    """Read JSON at `base:path` via git show. Empty dict if missing."""
    out = run(["git", "show", f"{base}:{path}"])
    if not out.strip():
        return {}
    return json.loads(out)


def read_pkg_file(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding = "utf-8"))


def all_decl_names(pkg: dict) -> set[str]:
    names: set[str] = set()
    for field in DEP_FIELDS:
        names.update((pkg.get(field) or {}).keys())
    return names


def _resolve_install_path(parent_path: str, name: str, pkgs: dict) -> str | None:
    """Walk up the nested node_modules chain from `parent_path` to find where
    `name` resolves, mirroring Node module resolution."""
    parts = parent_path.split("/node_modules/")
    for i in range(len(parts), 0, -1):
        prefix = "/node_modules/".join(parts[:i])
        trial = (prefix + "/node_modules/" if prefix else "node_modules/") + name
        if trial in pkgs:
            return trial
    if f"node_modules/{name}" in pkgs:
        return f"node_modules/{name}"
    return None


def _deps_of(meta: dict) -> dict:
    """Deps npm actually installs. Optional peers are skipped: they can't keep
    a removed top-level dep reachable on their own."""
    out = {}
    for field in ("dependencies", "optionalDependencies"):
        out.update(meta.get(field) or {})
    peer_meta = meta.get("peerDependenciesMeta") or {}
    for name, spec in (meta.get("peerDependencies") or {}).items():
        if (peer_meta.get(name) or {}).get("optional"):
            continue
        out[name] = spec
    return out


def reachable_from_head(head_pkg: dict, lock: dict) -> set[str]:
    """BFS the lockfile dep graph from `head_pkg`'s top-level deps. Returns the
    surviving install paths, excluding stale (orphaned) lockfile entries."""
    pkgs = lock.get("packages", {})
    if not pkgs:
        return set()
    roots = all_decl_names(head_pkg)
    seen: set[str] = set()
    frontier: list[str] = []
    for name in roots:
        p = _resolve_install_path("", name, pkgs)
        if p:
            frontier.append(p)
    while frontier:
        path = frontier.pop()
        if path in seen:
            continue
        seen.add(path)
        meta = pkgs.get(path, {})
        for dep_name in _deps_of(meta):
            p = _resolve_install_path(path, dep_name, pkgs)
            if p and p not in seen:
                frontier.append(p)
    return seen


def classify(pkg: str, file: str, content: str) -> str | None:
    """Return why `content` references `pkg`, or None.

    `content` may span multiple lines (multi-line imports/exports use re.DOTALL).
    Bare-spec regexes word-boundary the package name so `foobar` doesn't match
    `foo`. File-type gating restricts JS patterns to .ts/.tsx/.js/.jsx/.mjs/
    .cjs/.mdx, CSS to .css/.scss/.sass, HTML to .html/.htm, so a snippet inside
    a Python fixture or Markdown code block isn't mistaken for real npm usage.
    """
    if file in EXPECTED_NOISE_FILES:
        return None

    esc = re.escape(pkg)
    # Subpath gate: pkg must be followed by quote, `/`, or end-of-string.
    sub = r"(?:/[^'\"`]*)?"

    flags_dotall = re.DOTALL | re.MULTILINE

    is_script = bool(SCRIPT_LIKE_EXT.search(file))
    is_style = bool(STYLE_EXT.search(file))
    is_html = bool(HTML_EXT.search(file))
    is_ts = bool(TS_LIKE_EXT.search(file))

    # Gate out Python fixtures, Markdown code blocks, shell snippets, etc.
    is_json = file.endswith(".json") or file.endswith(".jsonc")
    if not (is_script or is_style or is_html or is_json):
        return None

    # CSS @import first so it doesn't collide with side-effect-import below.
    if is_style and re.search(rf"@import\s+['\"]{esc}{sub}['\"]", content):
        return "css_import"
    # Static imports, including multi-line `import { ... } from "pkg"`.
    if is_script and re.search(
        rf"(?<!@)\bimport\b[^;'\"]*?\bfrom\s+['\"]{esc}{sub}['\"]",
        content,
        flags_dotall,
    ):
        return "static_import"
    # Side-effect import `import "pkg"` (no `from`); lookbehind rules out @import.
    if is_script and re.search(rf"(?<!@)\bimport\s+['\"]{esc}{sub}['\"]", content):
        return "side_effect_import"
    # Dynamic import: `import("pkg")` and `await import("pkg")`.
    if is_script and re.search(rf"\bimport\(\s*['\"]{esc}{sub}['\"]\s*\)", content):
        return "dynamic_import"
    # require / require.resolve
    if is_script and re.search(
        rf"\brequire(?:\.resolve)?\(\s*['\"]{esc}{sub}['\"]\s*\)", content
    ):
        return "require"
    # Re-exports: `export * from`, `export { x } from`, `export type { Foo } from`.
    if is_script and re.search(
        rf"\bexport\b[^;'\"]*?\bfrom\s+['\"]{esc}{sub}['\"]",
        content,
        flags_dotall,
    ):
        return "re_export"
    # HTML script / link. Match pkg as a complete path segment so
    # `/node_modules/foo-extra/...` is not treated as usage of `foo`.
    html_pkg = rf"{esc}(?:/[^'\"#?]*)?(?=['\"#?])"
    if is_html and re.search(
        rf"<script[^>]*src\s*=\s*['\"][^'\"]*/{html_pkg}", content
    ):
        return "html_script"
    if is_html and re.search(rf"<link[^>]*href\s*=\s*['\"][^'\"]*/{html_pkg}", content):
        return "html_link"
    # TypeScript triple-slash
    if is_ts and re.search(
        rf"///\s*<reference\s+types\s*=\s*['\"]{esc}{sub}['\"]", content
    ):
        return "tsc_triple_slash"
    # new URL("pkg/...", import.meta.url)
    if is_script and re.search(rf"\bnew\s+URL\(\s*['\"]{esc}{sub}['\"]", content):
        return "new_url"
    # CSS url(...), quoted and unquoted, bounded so `pkg-extra` doesn't match.
    if is_style and re.search(
        rf"\burl\(\s*['\"]?(?:[^)'\"\s]+/)?{esc}(?:/[^)'\"`]*)?['\"]?\s*\)",
        content,
    ):
        return "css_url"
    # Template literal containing the package as the leading specifier
    if is_script and re.search(rf"`{esc}{sub}`", content):
        return "template_literal"
    # JSDoc / TS @import comment: `@import("pkg")`
    if is_script and re.search(rf"@import\(\s*['\"]{esc}{sub}['\"]\s*\)", content):
        return "jsdoc_import"
    # Bare quoted-string fallback (config plugin lists, vite aliases,
    # tsconfig paths, biome plugin arrays, shadcn registries).
    if not JS_LIKE_EXT.search(file):
        return None
    # pkg must be followed by `'`, `"`, or `/` so `foo` doesn't match `foobar`.
    if re.search(rf"['\"]{esc}(?:['\"]|/)", content):
        return "string_literal"
    return None


def lockfile_root_sync(head_pkg: dict, head_lock: dict) -> list[str]:
    """Warn if package-lock.json's <root> dep map disagrees with package.json
    (i.e. npm install was not re-run)."""
    warnings = []
    if not head_lock:
        return warnings
    root = head_lock.get("packages", {}).get("", {})
    lock_decl = {
        **(root.get("dependencies") or {}),
        **(root.get("devDependencies") or {}),
        **(root.get("peerDependencies") or {}),
        **(root.get("optionalDependencies") or {}),
    }
    pkg_decl = {}
    for f in DEP_FIELDS:
        pkg_decl.update(head_pkg.get(f) or {})
    only_in_lock = set(lock_decl) - set(pkg_decl)
    only_in_pkg = set(pkg_decl) - set(lock_decl)
    if only_in_lock:
        warnings.append(
            f"lockfile <root> lists deps not in package.json (lockfile stale): {sorted(only_in_lock)}"
        )
    if only_in_pkg:
        warnings.append(
            f"package.json declares deps not in lockfile <root> (run npm install): {sorted(only_in_pkg)}"
        )
    return warnings


def types_orphan_warnings(head_pkg: dict) -> list[str]:
    """Flag @types/<X> deps where <X> is no longer declared in package.json,
    which leaves dangling type packages."""
    decl = set()
    for f in DEP_FIELDS:
        decl.update((head_pkg.get(f) or {}).keys())
    warnings = []
    for name in decl:
        if not name.startswith("@types/"):
            continue
        # @types/scope__pkg provides types for @scope/pkg.
        target = name[len("@types/") :]
        if "__" in target:
            scope, sub = target.split("__", 1)
            target = f"@{scope}/{sub}"
        if target == "node":
            continue  # Node.js types are always implicit
        if target not in decl:
            warnings.append(
                f"@types/{target.replace('@', '').replace('/', '__')} present but '{target}' is not declared"
            )
    return warnings


_PKG_JSON_SKIP_KEYS = {
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
    "bundleDependencies",
    "bundledDependencies",
}

# Top-level fields whose contents are never package references.
_PKG_JSON_OPAQUE_KEYS = {
    "browserslist",  # browser queries
    "keywords",  # free-form strings
    "engines",  # node/npm version constraints
    "engineStrict",  # bool
    "packageManager",  # `pnpm@9.0.0` -- the package manager binary
    "volta",  # version pins for node/npm/yarn
    "files",  # paths included in publish
    "directories",  # paths
    "publishConfig",  # registry / access config
    "config",  # generic npm config values
    "main",
    "module",
    "browser",
    "types",
    "typings",
    "type",
    "exports",
    "imports",
    "bin",
    "man",  # author-side fields (not consumer refs)
    "scripts",  # handled separately via scripts_bin_refs()
    "repository",
    "bugs",
    "homepage",
    "funding",
    "author",
    "contributors",
    "maintainers",
    "license",
    "licenses",
    "name",
    "version",
    "description",
    "private",
    "sideEffects",
    "workspaces",  # paths/globs, NOT pkg names
}


def package_json_extra_refs(pkg: dict, target: str) -> list[str]:
    """Walk package.json (except dep declaration blocks) and return citations
    for string values or dict keys equal to `target` (or `target/subpath`).

    Catches refs that public dep-checkers commonly miss: overrides/resolutions/
    pnpm.overrides keys, pnpm.patchedDependencies, peerDependenciesMeta,
    prettier, eslintConfig.extends, stylelint, babel, jest, commitlint, etc.
    """
    target_sub = target + "/"
    cites: list[str] = []

    def matches(s: object) -> bool:
        return isinstance(s, str) and (s == target or s.startswith(target_sub))

    def walk(obj: object, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if path == "" and k in _PKG_JSON_SKIP_KEYS:
                    continue
                if path == "" and k in _PKG_JSON_OPAQUE_KEYS:
                    continue
                # Inside overrides/resolutions/etc., the KEY is a package ref.
                if matches(k):
                    cites.append(f"{path}.{k}" if path else k)
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            if matches(obj):
                cites.append(f"{path}: {obj}")

    walk(pkg, "")
    return cites


def build_bin_to_pkg(head_lock: dict) -> dict[str, str]:
    """Map a binary name (e.g. 'vite', 'eslint') to its providing package,
    from each lockfile entry's `bin` field."""
    out: dict[str, str] = {}
    if not head_lock:
        return out
    for path, meta in head_lock.get("packages", {}).items():
        if not path:
            continue
        name = path.split("node_modules/")[-1]
        bins = meta.get("bin")
        if isinstance(bins, dict):
            for binname in bins:
                out.setdefault(binname, name)
        elif isinstance(bins, str):
            out.setdefault(name.split("/")[-1], name)
    return out


_SCRIPT_TOKENIZE = re.compile(r"\s*(?:&&|\|\||;|\|(?!\|))\s*")

# Wrappers that delegate to a real CLI in the same shell word list; we skip
# past them and their flags to find the wrapped bin. Script-name wrappers
# (concurrently, npm-run-all, turbo, nx) are excluded: they reference script
# names, so the real bin lives in the target script's chunk we already tokenize.
_SCRIPT_WRAPPERS = {"cross-env", "dotenv", "dotenvx", "env-cmd"}
_ENV_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _next_real_bin(words: list[str], idx: int) -> str | None:
    """Walk `words` from `idx`, peeling env-prefix tokens, the package-manager
    runner (npx, pnpm exec, etc.), and known wrapper bins. Return the next
    real CLI binary, or None. Bounded by the chunk's word count."""
    seen_wrappers: set[str] = set()
    while idx < len(words):
        # 1. env-prefix run `FOO=bar BAZ="a b" cmd ...` (shlex pre-collapsed).
        while idx < len(words) and _ENV_PREFIX_RE.match(words[idx]):
            idx += 1
        if idx >= len(words):
            return None

        first = words[idx]
        # 2. Package-manager runner (npx/pnpm exec/yarn dlx/bunx): strip and
        # continue so the wrapped command re-enters the unwrap loop.
        if first in {"npx", "pnpx", "bunx"} and idx + 1 < len(words):
            idx += 1
            continue
        if (
            first in {"pnpm", "yarn"}
            and idx + 2 < len(words)
            and words[idx + 1] in {"exec", "dlx"}
        ):
            idx += 2
            continue

        # 3. Wrapper bin (cross-env, dotenv): skip its flags and env prefixes.
        bin_token = first.removeprefix("./node_modules/.bin/").removeprefix(
            "node_modules/.bin/"
        )
        if bin_token in _SCRIPT_WRAPPERS and bin_token not in seen_wrappers:
            seen_wrappers.add(bin_token)
            idx += 1
            # dotenv/dotenvx use `-e <file>` flags and an optional `--`.
            while idx < len(words):
                tok = words[idx]
                if tok.startswith("-") and tok != "--":
                    idx += 1
                    # `-e .env`: also skip the flag's argument.
                    if (
                        idx < len(words)
                        and not words[idx].startswith("-")
                        and not _ENV_PREFIX_RE.match(words[idx])
                    ):
                        idx += 1
                    continue
                if tok == "--":
                    idx += 1
                    break
                break
            continue
        return bin_token
    return None


def scripts_bin_refs(
    head_pkg: dict, bin_to_pkg: dict[str, str]
) -> dict[str, list[str]]:
    """Return `{package_name: ['scripts.X: cmd', ...]}` for every package
    referenced via its bin name in package.json scripts.

    Each script is split on shell separators; `_next_real_bin()` unwraps env
    prefixes, package-manager runners, and wrapper bins so `cross-env CI=1
    biome check` credits `biome`. Uses shlex.split so quoted env values survive.
    """
    import shlex

    scripts = head_pkg.get("scripts", {}) or {}
    refs: dict[str, list[str]] = {}
    for script_name, raw_cmd in scripts.items():
        if not isinstance(raw_cmd, str):
            continue
        for chunk in _SCRIPT_TOKENIZE.split(raw_cmd):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                words = shlex.split(chunk, posix = True)
            except ValueError:
                # Unbalanced quotes: fall back to plain split.
                words = chunk.split()
            if not words:
                continue
            bin_name = _next_real_bin(words, 0)
            if bin_name is None:
                continue
            pkg = bin_to_pkg.get(bin_name)
            if pkg:
                refs.setdefault(pkg, []).append(f"scripts.{script_name}: {raw_cmd}")
    return refs


def tsconfig_compiler_types_refs() -> set[str]:
    """Return package names in tsconfig*.json compilerOptions.types arrays.
    These are implicitly loaded by tsc and count as real uses."""
    out: set[str] = set()
    base = REPO_ROOT / "studio/frontend"
    for name in ("tsconfig.json", "tsconfig.app.json", "tsconfig.node.json"):
        path = base / name
        if not path.exists():
            continue
        try:
            text = path.read_text()
            # tsconfig allows comments; strip simple line comments.
            text = re.sub(r"//[^\n]*", "", text)
            data = json.loads(text)
        except (OSError, json.JSONDecodeError):
            continue
        types = (data.get("compilerOptions", {}) or {}).get("types", []) or []
        for t in types:
            if not isinstance(t, str):
                continue
            # `vite/client` resolves to the `vite` package.
            pkg = (
                t.split("/", 1)[0]
                if not t.startswith("@")
                else "/".join(t.split("/", 2)[:2])
            )
            out.add(pkg)
    return out


def enumerate_dep_usage(head_pkg: dict, head_lock: dict) -> dict[str, list]:
    """For every declared dep, classify usage into a dict of package-name lists:
    used, unused, type_pkg_kept (@types/X with X declared), type_pkg_orphan
    (@types/X with X gone). `unused` is a CANDIDATE list; verify before deletion.
    """
    decl = all_decl_names(head_pkg)
    bin_to_pkg = build_bin_to_pkg(head_lock) if head_lock else {}
    script_refs = scripts_bin_refs(head_pkg, bin_to_pkg)
    tsc_types = tsconfig_compiler_types_refs()

    results: dict[str, list] = {
        "used": [],
        "unused": [],
        "type_pkg_kept": [],
        "type_pkg_orphan": [],
    }
    for name in sorted(decl):
        if name.startswith("@types/"):
            target = name[len("@types/") :]
            if "__" in target:
                scope, sub = target.split("__", 1)
                target = f"@{scope}/{sub}"
            if target == "node":
                results["type_pkg_kept"].append(name)
            elif target in decl:
                results["type_pkg_kept"].append(name)
            else:
                results["type_pkg_orphan"].append(name)
            continue
        # Real-source-usage check
        hits = find_usage(name)
        used = bool(hits)
        # CLI usage in shell/workflow/Dockerfile. Skipped for @types/* (no CLI
        # binary; the bare-name bin candidate would false-match the runtime).
        if not used and not name.startswith("@types/") and find_command_usage(name):
            used = True
        # Bin scripts
        if not used and name in script_refs:
            used = True
        # package.json non-dep field references
        if not used and package_json_extra_refs(head_pkg, name):
            used = True
        # tsconfig compilerOptions.types implicit usage
        if not used and name in tsc_types:
            used = True
        if used:
            results["used"].append(name)
        else:
            results["unused"].append(name)
    return results


def find_imports_without_decl(head_pkg: dict) -> list[tuple[str, int, str]]:
    """Reverse check: find bare-specifier imports in studio/frontend/src with
    no matching package.json dep (import added but dep declaration forgotten).
    Covers import/require/dynamic-import shapes. Returns (file, line, spec).
    """
    decl = set()
    for f in DEP_FIELDS:
        decl.update((head_pkg.get(f) or {}).keys())
    # Exclude relative paths and the `@/` alias by requiring the specifier's
    # first char to be neither `.` nor `/`. Capture group is the specifier.
    pattern = (
        r"(?:\bfrom\s+|"
        r"\bimport\s+(?:\(\s*)?|"
        r"\brequire(?:\.resolve)?\(\s*)"
        r"['\"]([^'\"./][^'\"]*)['\"]"
    )
    args = [
        "grep",
        "-rnE",
        pattern,
        "--include=*.ts",
        "--include=*.tsx",
        "--include=*.js",
        "--include=*.jsx",
        "studio/frontend/src",
    ]
    out = run(args)
    missing = []
    for line in out.splitlines():
        m = re.match(r"^(?:\./)?([^:]+):(\d+):(.*)$", line)
        if not m:
            continue
        file, ln, content = m.group(1), int(m.group(2)), m.group(3)
        for spec_match in re.finditer(pattern, content):
            spec = spec_match.group(1)
            # Resolve to package name (strip subpath).
            if spec.startswith("@"):
                parts = spec.split("/", 2)
                pkg_name = "/".join(parts[:2]) if len(parts) >= 2 else spec
            else:
                pkg_name = spec.split("/", 1)[0]
            if pkg_name in decl:
                continue
            # Internal aliases like '@/foo' or builtin names.
            if pkg_name == "@":
                continue
            if pkg_name in {
                "node:fs",
                "node:path",
                "fs",
                "path",
                "url",
                "stream",
                "crypto",
                "buffer",
                "util",
                "events",
                "child_process",
            }:
                continue
            missing.append((file, ln, spec))
    return missing


def grep_repo(pat: str) -> list[tuple[str, int, str]]:
    args = ["grep", "-rnE", pat] + GREP_INCLUDES + GREP_EXCLUDES + ["."]
    out = run(args)
    rows = []
    for line in out.splitlines():
        m = re.match(r"^(\./)?([^:]+):(\d+):(.*)$", line)
        if m:
            rows.append((m.group(2), int(m.group(3)), m.group(4)))
    return rows


_file_lines_cache: dict[str, list[str]] = {}


def _read_file(path: str) -> list[str]:
    if path not in _file_lines_cache:
        try:
            _file_lines_cache[path] = (
                Path(path).read_text(errors = "replace").splitlines()
            )
        except (OSError, UnicodeDecodeError):
            _file_lines_cache[path] = []
    return _file_lines_cache[path]


def find_usage(pkg: str) -> list[Hit]:
    """Return real usages of `pkg` (pip-playwright filtered separately).

    For each grep hit, also feed a multi-line window into classify() so
    multi-line imports get picked up.
    """
    rows = grep_repo(re.escape(pkg))
    hits = []
    seen_keys: set[tuple[str, str]] = set()
    for file, lineno, content in rows:
        if pkg == "playwright" and PIP_PLAYWRIGHT.search(content):
            continue
        # Try the single-line classify first.
        kind = classify(pkg, file, content)
        if not kind:
            # Multi-line window (25 lines each side) so Prettier's
            # one-import-per-line formatting still pairs `import` with `from`.
            lines = _read_file(file)
            lo = max(0, lineno - 26)
            hi = min(len(lines), lineno + 25)
            window = "\n".join(lines[lo:hi])
            kind = classify(pkg, file, window)
        if kind:
            key = (file, kind)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append(Hit(file, lineno, kind, content[:160]))
    return hits


def _candidate_bin_names(pkg: str) -> set[str]:
    """Bin names a removed package's CLI could be invoked under. Most npm CLIs
    use the package name; scoped ones expose an unscoped bin (@biomejs/biome ->
    biome)."""
    return {pkg, pkg.rsplit("/", 1)[-1]}


def find_command_usage(pkg: str) -> list[Hit]:
    """Find package CLI invocations in shell/workflow/Dockerfile surfaces (npx,
    bunx, pnpm exec, yarn dlx, or bare `pkg --flag`). Bounded to
    COMMAND_LIKE_EXT so `npx foo` in a TS fixture isn't mistaken for real use.
    """
    bins = sorted(_candidate_bin_names(pkg), key = len, reverse = True)
    esc_bins = "|".join(re.escape(b) for b in bins)
    # grep ERE pattern. Built without f-strings to avoid clashing with the
    # POSIX `[[:space:]]` literals.
    grep_pat = (
        r"(^|[[:space:]:;&|(\[])"
        r"(npx[[:space:]]+|pnpm[[:space:]]+exec[[:space:]]+"
        r"|yarn[[:space:]]+(dlx[[:space:]]+)?|bunx[[:space:]]+)?"
        r"(" + esc_bins + r")"
        r"([[:space:])};|\]]|$)"
    )
    py_pat = re.compile(
        r"(^|[\s:;&|(\[])"
        r"(?:npx\s+|pnpm\s+exec\s+|yarn\s+(?:dlx\s+)?|bunx\s+)?"
        r"(" + esc_bins + r")"
        r"([\s)};|\]]|$)"
    )
    hits: list[Hit] = []
    seen: set[tuple[str, int]] = set()
    for file, lineno, content in grep_repo(grep_pat):
        if not COMMAND_LIKE_EXT.search(file):
            continue
        if pkg == "playwright" and PIP_PLAYWRIGHT.search(content):
            continue
        if not py_pat.search(content):
            continue
        key = (file, lineno)
        if key in seen:
            continue
        seen.add(key)
        hits.append(Hit(file, lineno, "command_bin", content[:160]))
    return hits


def types_target_name(pkg: str) -> str | None:
    """Strip `@types/` and decode scope-encoding to the runtime package name
    (`@types/foo__bar` -> `@foo/bar`). None for non-@types packages."""
    if not pkg.startswith("@types/"):
        return None
    target = pkg[len("@types/") :]
    if "__" in target:
        scope, sub = target.split("__", 1)
        return f"@{scope}/{sub}"
    return target


def find_types_runtime_usage(pkg: str, tsc_types: set[str]) -> list[Hit]:
    """For a removed `@types/X`, find usages of `X` itself (triple-slash
    reference, tsconfig types, runtime import). If any exist, @types/X stays."""
    target = types_target_name(pkg)
    if target is None:
        return []
    hits = find_usage(target)
    if target in tsc_types:
        hits.append(
            Hit(
                "studio/frontend/tsconfig*.json",
                0,
                "tsconfig_types",
                f'compilerOptions.types includes "{target}"',
            )
        )
    return hits


def main() -> int:
    p = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--base",
        default = "origin/main",
        help = "git ref to diff against (default: origin/main). "
        "Examples: HEAD~1, main, a-tag, a-sha.",
    )
    p.add_argument(
        "--base-pkg", help = "optional override: read base package.json from this path"
    )
    p.add_argument(
        "--base-lock",
        help = "optional override: read base package-lock.json from this path. "
        "Used to recover the bin -> package mapping for removed packages so "
        "scripts.foo still flags as a usage even after the PR drops node_modules/foo.",
    )
    p.add_argument(
        "--head-pkg",
        default = str(REPO_ROOT / FRONTEND_PKG),
        help = "head package.json path (default: working tree)",
    )
    p.add_argument(
        "--head-lock",
        default = str(REPO_ROOT / FRONTEND_LOCK),
        help = "head lockfile path (default: working tree). "
        "Reachability analysis runs against this lockfile.",
    )
    p.add_argument("--verbose", action = "store_true")
    p.add_argument(
        "--strict",
        action = "store_true",
        help = "Also fail on hygiene warnings (lockfile sync, "
        "@types orphans, imports without declared dep, unused deps).",
    )
    p.add_argument(
        "--enumerate-dead",
        action = "store_true",
        help = "Print every declared dep that appears unused anywhere "
        "in the repo. Informational; does not fail unless --strict.",
    )
    args = p.parse_args()

    if args.base_pkg:
        base_pkg = read_pkg_file(Path(args.base_pkg))
    else:
        base_pkg = read_pkg_at(args.base, FRONTEND_PKG)
    head_pkg = read_pkg_file(Path(args.head_pkg))
    if not base_pkg:
        print(
            f"ERROR: could not read base package.json at {args.base}:{FRONTEND_PKG}",
            file = sys.stderr,
        )
        return 2
    if not head_pkg:
        print(
            f"ERROR: could not read head package.json at {args.head_pkg}",
            file = sys.stderr,
        )
        return 2

    head_lock_path = Path(args.head_lock)
    if not head_lock_path.exists():
        print(
            f"ERROR: head lockfile not found at {head_lock_path}",
            file = sys.stderr,
        )
        return 2
    head_lock = read_pkg_file(head_lock_path)

    # Base lockfile is best-effort: only used to recover the bin -> package
    # mapping for packages the PR removes, so a scripts.biome cite still fires
    # when @biomejs/biome is dropped from the head lockfile.
    if args.base_lock:
        base_lock_path = Path(args.base_lock)
        base_lock = read_pkg_file(base_lock_path) if base_lock_path.exists() else {}
    else:
        base_lock = read_pkg_at(args.base, FRONTEND_LOCK)

    base_names = all_decl_names(base_pkg)
    head_names = all_decl_names(head_pkg)
    removed = sorted(base_names - head_names)

    # Hygiene checks compute up front so they run on both the removal-present
    # and removal-empty paths (so --strict fails on hygiene-only issues).
    sync_warns = lockfile_root_sync(head_pkg, head_lock)
    types_warns = types_orphan_warnings(head_pkg)
    missing_imports = find_imports_without_decl(head_pkg)
    enum = enumerate_dep_usage(head_pkg, head_lock) if args.enumerate_dead else None

    def _print_hygiene() -> None:
        if sync_warns:
            print("Lockfile sync warnings:")
            for w in sync_warns:
                print(f"  - {w}")
            print()
        if types_warns:
            print("@types orphan warnings:")
            for w in types_warns:
                print(f"  - {w}")
            print()
        if missing_imports:
            print(
                f"Imports without a matching package.json dep ({len(missing_imports)}):"
            )
            for file, ln, spec in missing_imports[:20]:
                print(f"  - {file}:{ln}  imports '{spec}'")
            print()
        if enum is not None:
            print("Dead-dep enumeration:")
            if enum["unused"]:
                print(f"  unused ({len(enum['unused'])}):")
                for n in enum["unused"]:
                    print(f"    - {n}")
            else:
                print("  unused: none")
            if enum["type_pkg_orphan"]:
                print(f"  type_pkg_orphan ({len(enum['type_pkg_orphan'])}):")
                for n in enum["type_pkg_orphan"]:
                    print(f"    - {n}")
            if args.verbose:
                print(f"  used: {len(enum['used'])}")
                print(f"  type_pkg_kept: {len(enum['type_pkg_kept'])}")
            print()

    hygiene_strict_fail = args.strict and (
        sync_warns
        or types_warns
        or missing_imports
        or (enum is not None and (enum["unused"] or enum["type_pkg_orphan"]))
    )

    if not removed:
        print("[OK] no dependencies removed from studio/frontend/package.json")
        if args.enumerate_dead or sync_warns or types_warns or missing_imports:
            print()
            _print_hygiene()
        if hygiene_strict_fail:
            print("FAIL (--strict): one or more hygiene warnings present")
            return 1
        return 0

    print(
        f"Checking {len(removed)} removed package(s) from studio/frontend/package.json"
    )
    print(f"Base: {args.base}    Head: working tree")
    print()

    reachable_paths = reachable_from_head(head_pkg, head_lock) if head_lock else set()
    # bin -> package map from the head lockfile, layering base-lockfile entries
    # for removed packages so scripts.biome still flags when @biomejs/biome is
    # dropped (head lockfile no longer maps it).
    bin_to_pkg = build_bin_to_pkg(head_lock) if head_lock else {}
    base_bin_to_pkg = build_bin_to_pkg(base_lock) if base_lock else {}
    removed_set = set(removed)
    for bin_name, pkg_name in base_bin_to_pkg.items():
        if pkg_name in removed_set:
            bin_to_pkg.setdefault(bin_name, pkg_name)
    script_refs = scripts_bin_refs(head_pkg, bin_to_pkg)
    tsc_types = tsconfig_compiler_types_refs()

    def reachable_install_paths(name: str) -> tuple[str | None, list[str]]:
        """Return (top_level_path, nested_paths). top_level is what bare
        `import "name"` resolves to; nested copies are only visible inside
        their parent package."""
        top = f"node_modules/{name}"
        top_path = top if top in reachable_paths else None
        nested = sorted(
            p
            for p in reachable_paths
            if p != top and p.endswith(f"/node_modules/{name}")
        )
        return top_path, nested

    failures: list[tuple[str, list[Hit]]] = []
    for name in removed:
        hits = find_usage(name)
        # CLI invocations in shell scripts / workflows / Dockerfiles.
        hits.extend(find_command_usage(name))
        # @types/X is "used" if X is referenced as a type or runtime import.
        hits.extend(find_types_runtime_usage(name, tsc_types))
        for cite in script_refs.get(name, []):
            hits.append(Hit("studio/frontend/package.json", 0, "script_bin", cite))
        for cite in package_json_extra_refs(head_pkg, name):
            hits.append(Hit("studio/frontend/package.json", 0, "pkg_json_field", cite))
        top, nested = reachable_install_paths(name)
        importable_top_level = top is not None
        # Bare specifier `name` resolves ONLY to top-level node_modules/<name>;
        # nested copies are invisible to src/ files.
        if hits and not importable_top_level:
            status = "FAIL"
        elif hits and importable_top_level:
            status = "OK-via-transitive"
        else:
            status = "OK"
        print(f"  [{status}] {name}")
        if top:
            print(f"    reachable (top-level): {top}")
        if nested:
            print(
                f"    reachable (nested, NOT importable from src/): {nested[0]}"
                + (f" (+{len(nested)-1} more)" if len(nested) > 1 else "")
            )
        if hits:
            for h in hits[:5]:
                print(f"    [{h.kind}] {h.file}:{h.line}  {h.snippet}")
        if status == "FAIL":
            failures.append((name, hits))
        if args.verbose and not hits and not (top or nested):
            print("    no references, not reachable -- clean removal")

    print()

    _print_hygiene()

    if failures:
        print(
            f"FAIL: {len(failures)} removed package(s) still referenced and not resolvable"
        )
        for name, _ in failures:
            print(f"  - {name}")
        return 1
    if hygiene_strict_fail:
        print("FAIL (--strict): one or more hygiene warnings present")
        return 1

    print("PASS: all removed packages are safe to drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
