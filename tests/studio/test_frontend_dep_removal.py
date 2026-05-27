#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Edge-case suite for scripts/check_frontend_dep_removal.py.

Each case patches a copy of studio/frontend/package.json to remove (or
move) a specific dependency, invokes the checker against the real
working tree's lockfile, and asserts the verdict matches expectations.

Run:
  python tests/studio/test_frontend_dep_removal.py

Exits 0 iff every case behaves as expected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HEAD_PKG = REPO / "studio/frontend/package.json"
HEAD_LOCK = REPO / "studio/frontend/package-lock.json"
SCRIPT = REPO / "scripts/check_frontend_dep_removal.py"


@dataclass
class Case:
    id: str
    desc: str
    remove: list[str]
    expected_status: str  # "PASS" | "FAIL"
    expected_failures: list[str]
    move_to_dev: list[str] | None = None  # rare: deps moved, not removed


CASES: list[Case] = [
    Case(
        "C1",
        "removing next-themes breaks 2 src imports",
        ["next-themes"],
        "FAIL",
        ["next-themes"],
    ),
    Case(
        "C2",
        "removing @xyflow/react breaks recipe-studio src imports "
        "(no other declared dep pulls @xyflow/react)",
        ["@xyflow/react"],
        "FAIL",
        ["@xyflow/react"],
    ),
    Case(
        "C3",
        "removing katex is safe: streamdown/math, mermaid, "
        "rehype-katex all keep it at top level",
        ["katex"],
        "PASS",
        [],
    ),
    Case("C4", "removing clsx is safe: streamdown keeps it", ["clsx"], "PASS", []),
    Case(
        "C5",
        "removing react is safe: peer of countless packages",
        ["react"],
        "PASS",
        [],
    ),
    Case(
        "C6",
        "removing @radix-ui/react-slot is safe: pulled by "
        "radix-ui umbrella + @assistant-ui/react",
        ["@radix-ui/react-slot"],
        "PASS",
        [],
    ),
    Case(
        "C7",
        "removing zustand is safe: @assistant-ui/react keeps "
        "top-level zustand@5.x (nested xyflow 4.x is irrelevant "
        "to src imports)",
        ["zustand"],
        "PASS",
        [],
    ),
    Case(
        "C8",
        "multi-remove with mixed safety: next-themes + "
        "@huggingface/hub + dexie all unsafe",
        ["next-themes", "@huggingface/hub", "dexie"],
        "FAIL",
        ["next-themes", "@huggingface/hub", "dexie"],
    ),
    Case(
        "C9",
        "removing @huggingface/hub breaks 5+ src imports",
        ["@huggingface/hub"],
        "FAIL",
        ["@huggingface/hub"],
    ),
    Case(
        "C10",
        "removing tailwind-merge is safe: streamdown keeps it",
        ["tailwind-merge"],
        "PASS",
        [],
    ),
    Case(
        "C11",
        "removing a non-existent name is a no-op",
        ["__never_existed_in_pkg__"],
        "PASS",
        [],
    ),
    Case(
        "C12",
        "moving @hugeicons/react from deps to devDeps is NOT a "
        "removal (still declared)",
        [],
        "PASS",
        [],
        move_to_dev = ["@hugeicons/react"],
    ),
    Case(
        "C13",
        "removing @huggingface/hub AND @xyflow/react together: both "
        "are root-only deps with no other parents, so both should FAIL",
        ["@huggingface/hub", "@xyflow/react"],
        "FAIL",
        ["@huggingface/hub", "@xyflow/react"],
    ),
    Case(
        "C14",
        "removing dexie breaks src imports (no other declared " "dep needs it)",
        ["dexie"],
        "FAIL",
        ["dexie"],
    ),
    Case(
        "C15",
        "removing motion (used in 20+ src imports including "
        "framer-motion-style animations); no transitive parent",
        ["motion"],
        "FAIL",
        ["motion"],
    ),
    Case(
        "C16",
        "removing canvas-confetti (imported in confetti.tsx); " "no transitive parent",
        ["canvas-confetti"],
        "FAIL",
        ["canvas-confetti"],
    ),
    Case(
        "C17",
        "removing recharts (imported in chart.tsx); no transitive " "parent",
        ["recharts"],
        "FAIL",
        ["recharts"],
    ),
    Case(
        "C18",
        "removing js-yaml is safe: @eslint/eslintrc keeps it "
        "(triggers @types/js-yaml orphan warning, non-fatal)",
        ["js-yaml"],
        "PASS",
        [],
    ),
    Case(
        "C19",
        "removing node-forge (imported in providers-api.ts); " "no transitive parent",
        ["node-forge"],
        "FAIL",
        ["node-forge"],
    ),
    Case(
        "C20",
        "removing @tauri-apps/api is safe: all 5 @tauri-apps "
        "plugins declare it as a direct dep",
        ["@tauri-apps/api"],
        "PASS",
        [],
    ),
    Case(
        "C21",
        "removing mammoth (imported in runtime-provider.tsx); " "no transitive parent",
        ["mammoth"],
        "FAIL",
        ["mammoth"],
    ),
    Case(
        "C22",
        "removing unpdf (imported in runtime-provider.tsx); " "no transitive parent",
        ["unpdf"],
        "FAIL",
        ["unpdf"],
    ),
    Case(
        "C23",
        "removing remark-gfm is safe: streamdown declares it " "as a direct dep",
        ["remark-gfm"],
        "PASS",
        [],
    ),
    Case(
        "C24",
        "removing date-fns is safe: react-day-picker and "
        "@base-ui/react both declare it as a direct dep",
        ["date-fns"],
        "PASS",
        [],
    ),
    Case(
        "C25",
        "removing vite is safe: @vitejs/plugin-react and @tailwindcss/vite "
        "keep it via peer (bin still resolves)",
        ["vite"],
        "PASS",
        [],
    ),
    Case(
        "C26",
        "removing typescript is safe: 11 transitive @typescript-eslint/* "
        "parents keep tsc bin alive",
        ["typescript"],
        "PASS",
        [],
    ),
    Case(
        "C27",
        "removing eslint is safe: typescript-eslint and eslint-plugin-* "
        "peers keep eslint bin alive",
        ["eslint"],
        "PASS",
        [],
    ),
    Case(
        "C28",
        "removing @biomejs/biome breaks scripts.biome:check / biome:fix "
        "(no transitive parents, biome bin orphans)",
        ["@biomejs/biome"],
        "FAIL",
        ["@biomejs/biome"],
    ),
    Case(
        "C29",
        "removing both @biomejs/biome AND @vitejs/plugin-react together: "
        "biome dies outright; vite loses one of its two retained peers "
        "but @tailwindcss/vite still keeps it",
        ["@biomejs/biome", "@vitejs/plugin-react"],
        "FAIL",
        ["@biomejs/biome", "@vitejs/plugin-react"],
    ),
]


def synth_head(head_pkg: dict, case: Case) -> dict:
    out = json.loads(json.dumps(head_pkg))
    for name in case.remove:
        for field in (
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
        ):
            (out.get(field) or {}).pop(name, None)
    if case.move_to_dev:
        for name in case.move_to_dev:
            v = (out.get("dependencies") or {}).pop(name, None)
            if v is not None:
                out.setdefault("devDependencies", {})[name] = v
    return out


def run_case(case: Case, head_pkg: dict) -> tuple[bool, str]:
    synth = synth_head(head_pkg, case)
    with tempfile.NamedTemporaryFile("w", suffix = ".json", delete = False) as f:
        json.dump(synth, f, indent = 2)
        synth_path = f.name
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base-pkg",
                str(HEAD_PKG),
                "--head-pkg",
                synth_path,
                "--head-lock",
                str(HEAD_LOCK),
            ],
            capture_output = True,
            text = True,
        )
    finally:
        os.unlink(synth_path)

    actual_status = {0: "PASS", 1: "FAIL"}.get(proc.returncode, f"RC{proc.returncode}")
    failure_pkgs: list[str] = []
    in_summary = False
    for line in proc.stdout.splitlines():
        if "FAIL:" in line and "removed package" in line:
            in_summary = True
            continue
        if in_summary and line.strip().startswith("- "):
            failure_pkgs.append(line.strip()[2:])

    ok = actual_status == case.expected_status and set(failure_pkgs) == set(
        case.expected_failures
    )
    return ok, (
        f"expected: status={case.expected_status} fails={sorted(case.expected_failures)}\n"
        f"actual:   status={actual_status} fails={sorted(failure_pkgs)}\n"
        f"--- stdout (first 30 lines) ---\n" + "\n".join(proc.stdout.splitlines()[:30])
    )


# ---------------------------------------------------------------------------
# Classifier unit tests: feed hand-crafted snippets directly into classify()
# and assert the returned kind. Covers sneaky import shapes that an
# adversarial / careless dev might use to obscure a real usage.
# ---------------------------------------------------------------------------

# Import the script's classify() by file path so this test does not need
# the package to be installed.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_dep_check", str(SCRIPT))
_dep_check = _ilu.module_from_spec(_spec)
sys.modules["_dep_check"] = _dep_check  # required so @dataclass can resolve annotations
_spec.loader.exec_module(_dep_check)
classify = _dep_check.classify
_next_real_bin = _dep_check._next_real_bin
scripts_bin_refs = _dep_check.scripts_bin_refs


@dataclass
class ClassifyCase:
    id: str
    desc: str
    pkg: str
    file: str
    content: str
    expected_kind: str | None  # None means "no detection"


CLASSIFY_CASES: list[ClassifyCase] = [
    # Bog-standard shapes
    ClassifyCase(
        "U01",
        "single-line static import",
        "next-themes",
        "src/x.tsx",
        'import { ThemeProvider } from "next-themes";',
        "static_import",
    ),
    ClassifyCase(
        "U02",
        "side-effect import",
        "katex",
        "src/x.tsx",
        'import "katex/dist/katex.min.css";',
        "side_effect_import",
    ),
    ClassifyCase(
        "U03",
        "dynamic import",
        "@tauri-apps/api",
        "src/x.tsx",
        'const { x } = await import("@tauri-apps/api/window");',
        "dynamic_import",
    ),
    ClassifyCase(
        "U04",
        "require()",
        "lodash",
        "src/x.js",
        'const _ = require("lodash");',
        "require",
    ),
    ClassifyCase(
        "U05",
        "CSS @import",
        "tailwindcss",
        "src/x.css",
        '@import "tailwindcss";',
        "css_import",
    ),
    # Sneaky shapes
    ClassifyCase(
        "U06",
        "multi-line static import",
        "next-themes",
        "src/x.tsx",
        'import {\n  ThemeProvider,\n  useTheme,\n} from "next-themes";',
        "static_import",
    ),
    ClassifyCase(
        "U07",
        "import type",
        "@huggingface/hub",
        "src/x.ts",
        'import type { PipelineType } from "@huggingface/hub";',
        "static_import",
    ),
    ClassifyCase(
        "U08",
        "export * from re-export",
        "@some-org/secrets",
        "src/x.ts",
        'export * from "@some-org/secrets";',
        "re_export",
    ),
    ClassifyCase(
        "U09",
        "export { x } from re-export",
        "lodash-es",
        "src/x.ts",
        'export { foo, bar } from "lodash-es";',
        "re_export",
    ),
    ClassifyCase(
        "U10",
        "export type ... from re-export",
        "@huggingface/hub",
        "src/x.ts",
        'export type { Foo } from "@huggingface/hub";',
        "re_export",
    ),
    ClassifyCase(
        "U11",
        "multi-line export from re-export",
        "@some/pkg",
        "src/x.ts",
        'export {\n  thing,\n  other,\n} from "@some/pkg";',
        "re_export",
    ),
    ClassifyCase(
        "U12",
        "JSDoc @import",
        "react",
        "src/x.ts",
        '/** @type {import("react").FC} */\nconst Foo = () => null;',
        "dynamic_import",
    ),
    ClassifyCase(
        "U13",
        "template literal package path",
        "@assistant-ui/react",
        "src/x.tsx",
        "const url = `@assistant-ui/react`;",
        "template_literal",
    ),
    ClassifyCase(
        "U14",
        "new URL import-meta",
        "monaco-editor",
        "src/x.ts",
        'new URL("monaco-editor/esm/vs/editor/editor.worker", import.meta.url);',
        "new_url",
    ),
    ClassifyCase(
        "U15",
        "tsc triple-slash type ref",
        "@types/some-pkg",
        "src/x.ts",
        '/// <reference types="@types/some-pkg" />',
        "tsc_triple_slash",
    ),
    ClassifyCase(
        "U16",
        "HTML script src",
        "alpinejs",
        "index.html",
        '<script src="/node_modules/alpinejs/dist/cdn.min.js"></script>',
        "html_script",
    ),
    ClassifyCase(
        "U17",
        "HTML link href",
        "alpinejs",
        "index.html",
        '<link rel="stylesheet" href="/node_modules/alpinejs/dist/style.css">',
        "html_link",
    ),
    ClassifyCase(
        "U18",
        "bare quoted string in tsconfig paths",
        "@huggingface/hub",
        "tsconfig.json",
        '"paths": { "hf": ["@huggingface/hub/*"] }',
        "string_literal",
    ),
    ClassifyCase(
        "U19",
        "vite alias key",
        "@dagrejs/dagre",
        "vite.config.ts",
        '"@dagrejs/dagre": path.resolve(__dirname, "./..."),',
        "string_literal",
    ),
    # False-positive guards (these should NOT detect)
    ClassifyCase(
        "U20",
        "different package with shared prefix",
        "foo",
        "src/x.ts",
        'import { x } from "foobar";',
        None,
    ),
    ClassifyCase(
        "U21",
        "package mentioned in plain comment text",
        "react",
        "src/x.ts",
        "// We migrated from react-router to tanstack-router",
        None,
    ),
    ClassifyCase(
        "U22",
        "package name as a URL path tail is NOT detected "
        "(boundary rule: pkg must be followed by quote or `/`)",
        "react",
        "src/x.ts",
        'const docs = "https://example.com/react";',
        None,
    ),
    ClassifyCase(
        "U23",
        "package name in Python file (ignored, "
        "Python can never import npm packages)",
        "playwright",
        "tests/x.py",
        'label: str = "playwright"',
        None,
    ),
    ClassifyCase(
        "U24",
        "exact-prefix collision: pkg 'lodash' and 'lodash-es'",
        "lodash",
        "src/x.ts",
        'import _ from "lodash-es";',
        None,
    ),
    ClassifyCase(
        "U25",
        "scoped pkg substring collision",
        "@radix-ui/react-label",
        "src/x.ts",
        'import x from "@radix-ui/react-label-extra";',
        None,
    ),
    ClassifyCase(
        "U26",
        "package only mentioned in a markdown link",
        "react",
        "README.md",
        "See [react](https://react.dev).",
        None,
    ),
    ClassifyCase(
        "U27",
        "side-effect import with subpath",
        "katex",
        "src/x.css",
        '@import "katex/dist/katex.min.css";',
        "css_import",
    ),
    ClassifyCase(
        "U28",
        "require.resolve",
        "lodash",
        "build/x.cjs",
        'const path = require.resolve("lodash/fp");',
        "require",
    ),
    ClassifyCase(
        "U29",
        "TypeScript ambient `declare module`",
        "@tanstack/react-router",
        "src/app/router.tsx",
        'declare module "@tanstack/react-router" {\n  interface X {}\n}',
        "string_literal",
    ),
    ClassifyCase(
        "U30",
        "namespace import `import * as X from pkg`",
        "@radix-ui/react-slot",
        "src/x.tsx",
        'import * as Slot from "@radix-ui/react-slot";',
        "static_import",
    ),
    ClassifyCase(
        "U31",
        "combined default + named import",
        "react",
        "src/x.tsx",
        'import React, { useState } from "react";',
        "static_import",
    ),
    ClassifyCase(
        "U32",
        "default-as-named import alias",
        "react",
        "src/x.tsx",
        'import { default as R } from "react";',
        "static_import",
    ),
    ClassifyCase(
        "U33",
        "re-export default",
        "lodash",
        "src/x.ts",
        'export { default } from "lodash";',
        "re_export",
    ),
    ClassifyCase(
        "U34",
        "re-export default as alias",
        "lodash",
        "src/x.ts",
        'export { default as _ } from "lodash";',
        "re_export",
    ),
    ClassifyCase(
        "U35",
        ".then() dynamic import (no await)",
        "@tauri-apps/api",
        "src/x.ts",
        'import("@tauri-apps/api/window").then(m => m.x());',
        "dynamic_import",
    ),
    ClassifyCase(
        "U36",
        "TypeScript import() in type position",
        "react",
        "src/x.ts",
        'type C = import("react").ComponentType;',
        "dynamic_import",
    ),
    # File-type gating (codex P1: JS classifiers must not fire on
    # non-script files). Python fixtures and Markdown code blocks often
    # contain literal JS-shaped strings for documentation or test data,
    # so a bare `import x from "pkg"` inside a .py / .md / .sh / .yml is
    # not a real npm usage.
    ClassifyCase(
        "U37",
        "JS import snippet inside a Python fixture string is NOT a usage",
        "next-themes",
        "tests/studio/something.py",
        "snippet = 'import x from \"next-themes\";'",
        None,
    ),
    ClassifyCase(
        "U38",
        "JS import snippet inside a Markdown code fence is NOT a usage",
        "next-themes",
        "docs/example.md",
        '```ts\nimport x from "next-themes";\n```',
        None,
    ),
    ClassifyCase(
        "U39",
        "JS import inside a shell script is NOT classified as a JS usage",
        "next-themes",
        "scripts/build.sh",
        'echo "import x from \\"next-themes\\";"',
        None,
    ),
    ClassifyCase(
        "U40",
        "JS import inside a YAML workflow is NOT classified as a JS usage",
        "next-themes",
        ".github/workflows/x.yml",
        "run: echo 'import x from \"next-themes\";'",
        None,
    ),
    # HTML script/link must respect package-name boundaries: a
    # `/node_modules/foo-extra/...` reference does NOT use `foo`.
    ClassifyCase(
        "U41",
        "HTML <script src=...> with similar-prefix package is NOT a match",
        "foo",
        "index.html",
        '<script src="/node_modules/foo-extra/dist/index.js"></script>',
        None,
    ),
    ClassifyCase(
        "U42",
        "HTML <link href=...> with similar-prefix package is NOT a match",
        "foo",
        "index.html",
        '<link rel="stylesheet" href="/node_modules/foo-extra/dist/style.css">',
        None,
    ),
    ClassifyCase(
        "U43",
        "HTML <script src=...> with exact package match IS a match",
        "foo",
        "index.html",
        '<script src="/node_modules/foo/dist/index.js"></script>',
        "html_script",
    ),
    # CSS url() unquoted variant -- valid CSS, must classify the same
    # as the quoted variant.
    ClassifyCase(
        "U44",
        "CSS url() unquoted bare package path",
        "katex",
        "src/x.css",
        "src: url(katex/dist/fonts/font.woff2);",
        "css_url",
    ),
    ClassifyCase(
        "U45",
        "CSS url() quoted bare package path still works",
        "katex",
        "src/x.css",
        'src: url("katex/dist/fonts/font.woff2");',
        "css_url",
    ),
]


def run_classify_unit_tests() -> int:
    passed = 0
    for c in CLASSIFY_CASES:
        actual = classify(c.pkg, c.file, c.content)
        ok = actual == c.expected_kind
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {c.id}: {c.desc}")
        if not ok:
            print(f"      pkg={c.pkg!r} file={c.file!r}")
            print(f"      content={c.content!r}")
            print(f"      expected={c.expected_kind!r}, actual={actual!r}")
        if ok:
            passed += 1
    print()
    print(f"{passed}/{len(CLASSIFY_CASES)} classify-unit cases pass")
    return 0 if passed == len(CLASSIFY_CASES) else 1


# ---------------------------------------------------------------------------
# Adversarial end-to-end cases: drop a sneaky synthetic file into src/,
# run the checker, then clean up. Catches the case where pattern detection
# regresses for a real grep+classify pipeline (not just classify in isolation).
# ---------------------------------------------------------------------------

ADVERSARIAL_TMP_DIR = REPO / "studio/frontend/src/__dep_check_adversarial__"


@dataclass
class AdvCase:
    id: str
    desc: str
    filename: str
    content: str
    target_pkg: str
    expected_status: str
    expected_failures: list[str]


ADV_CASES: list[AdvCase] = [
    AdvCase(
        "A01",
        "multi-line import of removed pkg should FAIL",
        "adv01.ts",
        'import {\n  foo,\n  bar,\n} from "__adv_only_pkg_a__";\n',
        "__adv_only_pkg_a__",
        "FAIL",
        ["__adv_only_pkg_a__"],
    ),
    AdvCase(
        "A02",
        "export * from removed pkg should FAIL",
        "adv02.ts",
        'export * from "__adv_only_pkg_b__";\n',
        "__adv_only_pkg_b__",
        "FAIL",
        ["__adv_only_pkg_b__"],
    ),
    AdvCase(
        "A03",
        "export { x } from removed pkg should FAIL",
        "adv03.ts",
        'export { foo, bar } from "__adv_only_pkg_c__";\n',
        "__adv_only_pkg_c__",
        "FAIL",
        ["__adv_only_pkg_c__"],
    ),
    AdvCase(
        "A04",
        "export type ... from removed pkg should FAIL",
        "adv04.ts",
        'export type { Foo } from "__adv_only_pkg_d__";\n',
        "__adv_only_pkg_d__",
        "FAIL",
        ["__adv_only_pkg_d__"],
    ),
    AdvCase(
        "A05",
        "package with similar prefix should NOT trigger FAIL",
        "adv05.ts",
        # The file imports __adv_only_pkg_e_extra__, but we will try
        # to "remove" the shorter __adv_only_pkg_e__ name. The shorter
        # name has zero real usage, so removal must be safe.
        'import x from "__adv_only_pkg_e_extra__";\n',
        "__adv_only_pkg_e__",
        "PASS",
        [],
    ),
    AdvCase(
        "A06",
        "dynamic import of removed pkg should FAIL",
        "adv06.ts",
        'const m = await import("__adv_only_pkg_f__");\n',
        "__adv_only_pkg_f__",
        "FAIL",
        ["__adv_only_pkg_f__"],
    ),
    AdvCase(
        "A07",
        "new URL of removed pkg should FAIL",
        "adv07.ts",
        'const w = new URL("__adv_only_pkg_g__/worker.js", import.meta.url);\n',
        "__adv_only_pkg_g__",
        "FAIL",
        ["__adv_only_pkg_g__"],
    ),
    AdvCase(
        "A08",
        "string-concat dynamic import is unanalyzable (PASS)",
        "adv08.ts",
        'const m = await import("__adv_only_" + "pkg_h__");\n',
        "__adv_only_pkg_h__",
        "PASS",
        [],
    ),
    AdvCase(
        "A09",
        "package referenced only inside a JS comment "
        "is conservatively flagged via the string_literal fallback "
        "(this is acceptable -- err on the side of caution)",
        "adv09.ts",
        '// TODO: import x from "__adv_only_pkg_i__"\n',
        "__adv_only_pkg_i__",
        "FAIL",
        ["__adv_only_pkg_i__"],
    ),
    AdvCase(
        "A10",
        "package referenced only in a Python file should " "NOT trigger a JS FAIL",
        "adv10.py",
        'label = "__adv_only_pkg_j__"\n',
        "__adv_only_pkg_j__",
        "PASS",
        [],
    ),
    AdvCase(
        "A11",
        "package mentioned in a markdown doc file is "
        "ignored by JS-like-only string_literal",
        "adv11.md",
        "See [docs](https://example.com/__adv_only_pkg_k__).\n",
        "__adv_only_pkg_k__",
        "PASS",
        [],
    ),
    AdvCase(
        "A12",
        "JSDoc @import of removed pkg should FAIL",
        "adv12.ts",
        '/** @type {import("__adv_only_pkg_l__").Foo} */\n' "const x = null;\n",
        "__adv_only_pkg_l__",
        "FAIL",
        ["__adv_only_pkg_l__"],
    ),
    # Prettier formats a long named-import list one identifier per line.
    # 22 imports + braces puts the `import` keyword ~22 lines away from
    # the `from "pkg"` clause. Before the window widening, the classify
    # multi-line fallback used ±4 lines, which silently missed every
    # such block. This case fails with the old window and passes once
    # the window is wide enough (currently ±25).
    AdvCase(
        "A13",
        "Prettier-style 22-identifier multi-line import should FAIL "
        "(exercises the widened multi-line classify window)",
        "adv13.ts",
        "import {\n"
        + "".join(f"  ident_{i:02d},\n" for i in range(22))
        + '} from "__adv_only_pkg_m__";\n',
        "__adv_only_pkg_m__",
        "FAIL",
        ["__adv_only_pkg_m__"],
    ),
]


# ---------------------------------------------------------------------------
# package.json field-reference cases: simulate `prettier: "@x/config"`,
# `eslintConfig.extends`, `overrides`, `peerDependenciesMeta`, etc.
# These test the package_json_extra_refs() coverage. Cross-checked against
# the patterns used by Tailwind, Stylelint, Prettier, Next.js, Astro,
# TypeScript, ESLint, SvelteKit, Storybook, Vite, and TanStack/Query
# manifests.
# ---------------------------------------------------------------------------


@dataclass
class PkgFieldCase:
    id: str
    desc: str
    field_patch: dict  # extra fields to merge into synth_head package.json
    target_pkg: str
    expected_status: str
    expected_failures: list[str]


PKG_FIELD_CASES: list[PkgFieldCase] = [
    PkgFieldCase(
        "P01",
        "removing pkg referenced only in `prettier` string field",
        {"prettier": "__pkg_prettier_config__"},
        "__pkg_prettier_config__",
        "FAIL",
        ["__pkg_prettier_config__"],
    ),
    PkgFieldCase(
        "P02",
        "removing pkg referenced in `eslintConfig.extends` array",
        {"eslintConfig": {"extends": ["__pkg_eslint_cfg__"]}},
        "__pkg_eslint_cfg__",
        "FAIL",
        ["__pkg_eslint_cfg__"],
    ),
    PkgFieldCase(
        "P03",
        "removing pkg referenced in `stylelint.plugins`",
        {"stylelint": {"plugins": ["__pkg_stylelint_plugin__"]}},
        "__pkg_stylelint_plugin__",
        "FAIL",
        ["__pkg_stylelint_plugin__"],
    ),
    PkgFieldCase(
        "P04",
        "removing pkg referenced in `babel.presets`",
        {"babel": {"presets": [["__pkg_babel_preset__", {"opt": 1}]]}},
        "__pkg_babel_preset__",
        "FAIL",
        ["__pkg_babel_preset__"],
    ),
    PkgFieldCase(
        "P05",
        "removing pkg used as a key in `overrides`",
        {"overrides": {"__pkg_overridden__": "^1.0.0"}},
        "__pkg_overridden__",
        "FAIL",
        ["__pkg_overridden__"],
    ),
    PkgFieldCase(
        "P06",
        "removing pkg used as a key in `pnpm.overrides`",
        {"pnpm": {"overrides": {"__pkg_pnpm_override__": "^1.0.0"}}},
        "__pkg_pnpm_override__",
        "FAIL",
        ["__pkg_pnpm_override__"],
    ),
    PkgFieldCase(
        "P07",
        "removing pkg used as a key in `pnpm.patchedDependencies`",
        {"pnpm": {"patchedDependencies": {"__pkg_patched__": "patches/x.patch"}}},
        "__pkg_patched__",
        "FAIL",
        ["__pkg_patched__"],
    ),
    PkgFieldCase(
        "P08",
        "removing pkg used as a key in `peerDependenciesMeta`",
        {"peerDependenciesMeta": {"__pkg_peer_meta__": {"optional": True}}},
        "__pkg_peer_meta__",
        "FAIL",
        ["__pkg_peer_meta__"],
    ),
    PkgFieldCase(
        "P09",
        "removing pkg referenced in `jest.preset` string",
        {"jest": {"preset": "__pkg_jest_preset__"}},
        "__pkg_jest_preset__",
        "FAIL",
        ["__pkg_jest_preset__"],
    ),
    PkgFieldCase(
        "P10",
        "removing pkg referenced in `commitlint.extends`",
        {"commitlint": {"extends": ["__pkg_commitlint__"]}},
        "__pkg_commitlint__",
        "FAIL",
        ["__pkg_commitlint__"],
    ),
    PkgFieldCase(
        "P11",
        "removing pkg referenced in `renovate.extends`",
        {"renovate": {"extends": ["__pkg_renovate__"]}},
        "__pkg_renovate__",
        "FAIL",
        ["__pkg_renovate__"],
    ),
    PkgFieldCase(
        "P12",
        "removing pkg referenced in `remarkConfig.plugins`",
        {"remarkConfig": {"plugins": ["__pkg_remark__"]}},
        "__pkg_remark__",
        "FAIL",
        ["__pkg_remark__"],
    ),
    PkgFieldCase(
        "P13",
        "removing pkg with subpath ref in tool config (`pkg/config`)",
        {"prettier": "__pkg_prettier_sub__/config"},
        "__pkg_prettier_sub__",
        "FAIL",
        ["__pkg_prettier_sub__"],
    ),
    PkgFieldCase(
        "P14",
        "false-positive guard: similar-prefix package in tool config",
        {"prettier": "__pkg_short_extra__/config"},
        "__pkg_short__",
        "PASS",
        [],
    ),
    PkgFieldCase(
        "P15",
        "false-positive guard: package-named string in `browserslist` "
        "must NOT trigger (browserslist values are browser queries, "
        "never package names)",
        {"browserslist": ["last 2 versions", "__pkg_browserslist__"]},
        "__pkg_browserslist__",
        "PASS",
        [],
    ),
    PkgFieldCase(
        "P16",
        "false-positive guard: matching string in `keywords` field",
        {"keywords": ["__pkg_keyword__", "foo"]},
        "__pkg_keyword__",
        "PASS",
        [],
    ),
    PkgFieldCase(
        "P17",
        "false-positive guard: matching string in `workspaces` (paths)",
        {"workspaces": ["packages/__pkg_workspace_path__"]},
        "__pkg_workspace_path__",
        "PASS",
        [],
    ),
    PkgFieldCase(
        "P18",
        "false-positive guard: matching value in `files` field",
        {"files": ["dist/__pkg_in_files__"]},
        "__pkg_in_files__",
        "PASS",
        [],
    ),
    PkgFieldCase(
        "P19",
        "false-positive guard: matching `packageManager` string",
        {"packageManager": "__pkg_in_pm__@1.0.0"},
        "__pkg_in_pm__",
        "PASS",
        [],
    ),
]


def run_pkg_field_cases() -> int:
    head_pkg = json.loads(HEAD_PKG.read_text())
    passed = 0
    for pc in PKG_FIELD_CASES:
        synth_head = json.loads(json.dumps(head_pkg))
        # Apply the field patch (deep-merge isn't needed; we control the keys).
        for k, v in pc.field_patch.items():
            synth_head[k] = v
        # Base has the target in dependencies; head does not. The extra field
        # in synth_head references the target pkg even though it's no longer
        # in deps.
        synth_base = json.loads(json.dumps(head_pkg))
        synth_base.setdefault("dependencies", {})[pc.target_pkg] = "^1.0.0"
        with tempfile.NamedTemporaryFile("w", suffix = ".json", delete = False) as f:
            json.dump(synth_base, f, indent = 2)
            base_path = f.name
        with tempfile.NamedTemporaryFile("w", suffix = ".json", delete = False) as f:
            json.dump(synth_head, f, indent = 2)
            head_path = f.name
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-pkg",
                    base_path,
                    "--head-pkg",
                    head_path,
                    "--head-lock",
                    str(HEAD_LOCK),
                ],
                capture_output = True,
                text = True,
                cwd = str(REPO),
            )
        finally:
            os.unlink(base_path)
            os.unlink(head_path)
        actual_status = {0: "PASS", 1: "FAIL"}.get(
            proc.returncode, f"RC{proc.returncode}"
        )
        fails: list[str] = []
        in_summary = False
        for line in proc.stdout.splitlines():
            if "FAIL:" in line and "removed package" in line:
                in_summary = True
                continue
            if in_summary and line.strip().startswith("- "):
                fails.append(line.strip()[2:])
        # The expected_failures includes the tolerated-FP case (P15); we
        # accept BOTH expected_status and expected_failures matches.
        ok = actual_status == pc.expected_status and set(fails) == set(
            pc.expected_failures
        )
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {pc.id}: {pc.desc}")
        if not ok:
            print(
                f"      expected: status={pc.expected_status} fails={pc.expected_failures}"
            )
            print(f"      actual:   status={actual_status} fails={fails}")
            for ln in proc.stdout.splitlines()[:25]:
                print(f"      {ln}")
        if ok:
            passed += 1
    print()
    print(f"{passed}/{len(PKG_FIELD_CASES)} package.json-field cases pass")
    return 0 if passed == len(PKG_FIELD_CASES) else 1


def run_adversarial_cases() -> int:
    ADVERSARIAL_TMP_DIR.mkdir(parents = True, exist_ok = True)
    head_pkg = json.loads(HEAD_PKG.read_text())
    passed = 0
    for ac in ADV_CASES:
        # Drop the synthetic file.
        fpath = ADVERSARIAL_TMP_DIR / ac.filename
        try:
            fpath.write_text(ac.content)
            # Build a synthetic base that has the target pkg added; head
            # is the real head (without it). The script sees the pkg as
            # removed and scans the repo, which now includes our file.
            synth_base = json.loads(json.dumps(head_pkg))
            synth_base.setdefault("dependencies", {})[ac.target_pkg] = "^1.0.0"
            with tempfile.NamedTemporaryFile("w", suffix = ".json", delete = False) as f:
                json.dump(synth_base, f, indent = 2)
                base_path = f.name
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--base-pkg",
                        base_path,
                        "--head-pkg",
                        str(HEAD_PKG),
                        "--head-lock",
                        str(HEAD_LOCK),
                    ],
                    capture_output = True,
                    text = True,
                    cwd = str(REPO),
                )
            finally:
                os.unlink(base_path)
            actual_status = {0: "PASS", 1: "FAIL"}.get(
                proc.returncode, f"RC{proc.returncode}"
            )
            fails = []
            in_summary = False
            for line in proc.stdout.splitlines():
                if "FAIL:" in line and "removed package" in line:
                    in_summary = True
                    continue
                if in_summary and line.strip().startswith("- "):
                    fails.append(line.strip()[2:])
            ok = actual_status == ac.expected_status and set(fails) == set(
                ac.expected_failures
            )
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {ac.id}: {ac.desc}")
            if not ok:
                print(
                    f"      expected: status={ac.expected_status} fails={ac.expected_failures}"
                )
                print(f"      actual:   status={actual_status} fails={fails}")
                for ln in proc.stdout.splitlines()[:20]:
                    print(f"      {ln}")
            if ok:
                passed += 1
        finally:
            try:
                fpath.unlink()
            except FileNotFoundError:
                pass
    # Clean up the directory.
    try:
        ADVERSARIAL_TMP_DIR.rmdir()
    except OSError:
        pass
    print()
    print(f"{passed}/{len(ADV_CASES)} adversarial cases pass")
    return 0 if passed == len(ADV_CASES) else 1


# ---------------------------------------------------------------------------
# Dead-dep enumeration cases.
# ---------------------------------------------------------------------------


@dataclass
class EnumCase:
    id: str
    desc: str
    add_deps: dict[str, str]
    add_dev_deps: dict[str, str]
    field_patch: dict
    extra_file: tuple[str, str] | None  # (relative_path, content) or None
    expected_unused: set[str]
    expected_used: set[str]
    expected_orphan_types: set[str]


ENUM_CASES: list[EnumCase] = [
    EnumCase(
        "E01",
        "fake dep with no usage anywhere is flagged unused",
        {"__enum_fake_unused_pkg__": "^1.0.0"},
        {},
        {},
        None,
        {"__enum_fake_unused_pkg__"},
        set(),
        set(),
    ),
    EnumCase(
        "E02",
        "fake dep referenced via vite.config-style import is flagged used "
        "(uses a real adversarial file as the import site)",
        {"__enum_used_via_src__": "^1.0.0"},
        {},
        {},
        (
            "src/__dep_check_adversarial__/enum_e02.ts",
            'import x from "__enum_used_via_src__";\n',
        ),
        set(),
        {"__enum_used_via_src__"},
        set(),
    ),
    EnumCase(
        "E03",
        "fake dep referenced only in package.json `overrides` is flagged used",
        {"__enum_used_via_overrides__": "^1.0.0"},
        {},
        {"overrides": {"__enum_used_via_overrides__": "^1.0.0"}},
        None,
        set(),
        {"__enum_used_via_overrides__"},
        set(),
    ),
    EnumCase(
        "E04",
        "@types/X where X is declared -> kept (NOT orphan)",
        {"__enum_real_pkg__": "^1.0.0"},
        {"@types/__enum_real_pkg__": "^1.0.0"},
        {},
        (
            "src/__dep_check_adversarial__/enum_e04.ts",
            'import x from "__enum_real_pkg__";\n',
        ),
        set(),
        {"__enum_real_pkg__"},
        set(),
    ),
    EnumCase(
        "E05",
        "@types/X where X is NOT declared anywhere -> orphan",
        {},
        {"@types/__enum_orphan_pkg__": "^1.0.0"},
        {},
        None,
        set(),
        set(),
        {"@types/__enum_orphan_pkg__"},
    ),
]


def run_enum_cases() -> int:
    head_pkg = json.loads(HEAD_PKG.read_text())
    passed = 0
    ADVERSARIAL_TMP_DIR.mkdir(parents = True, exist_ok = True)
    for ec in ENUM_CASES:
        synth_head = json.loads(json.dumps(head_pkg))
        synth_head.setdefault("dependencies", {}).update(ec.add_deps)
        synth_head.setdefault("devDependencies", {}).update(ec.add_dev_deps)
        for k, v in ec.field_patch.items():
            synth_head[k] = v
        # Drop any temp source file if needed.
        fpath = None
        if ec.extra_file:
            rel, content = ec.extra_file
            fpath = REPO / rel
            fpath.parent.mkdir(parents = True, exist_ok = True)
            fpath.write_text(content)
        with tempfile.NamedTemporaryFile("w", suffix = ".json", delete = False) as f:
            json.dump(synth_head, f, indent = 2)
            head_path = f.name
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-pkg",
                    str(HEAD_PKG),
                    "--head-pkg",
                    head_path,
                    "--head-lock",
                    str(HEAD_LOCK),
                    "--enumerate-dead",
                ],
                capture_output = True,
                text = True,
                cwd = str(REPO),
            )
        finally:
            os.unlink(head_path)
            if fpath:
                try:
                    fpath.unlink()
                except FileNotFoundError:
                    pass
        # Parse the dead-dep enumeration output.
        unused: set[str] = set()
        orphans: set[str] = set()
        in_unused = False
        in_orphan = False
        for line in proc.stdout.splitlines():
            s = line.strip()
            if s.startswith("unused ("):
                in_unused = True
                in_orphan = False
                continue
            if s.startswith("type_pkg_orphan ("):
                in_unused = False
                in_orphan = True
                continue
            if s.startswith("used:") or s.startswith("type_pkg_kept:"):
                in_unused = in_orphan = False
                continue
            if s.startswith("- "):
                if in_unused:
                    unused.add(s[2:])
                elif in_orphan:
                    orphans.add(s[2:])
        unused_ok = ec.expected_unused.issubset(unused) and (
            not ec.expected_used or not (ec.expected_used & unused)
        )
        orphan_ok = ec.expected_orphan_types.issubset(orphans)
        ok = unused_ok and orphan_ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {ec.id}: {ec.desc}")
        if not ok:
            print(f"      expected unused superset: {sorted(ec.expected_unused)}")
            print(f"      expected used NOT in unused: {sorted(ec.expected_used)}")
            print(
                f"      expected orphans superset: {sorted(ec.expected_orphan_types)}"
            )
            print(f"      actual unused: {sorted(unused)}")
            print(f"      actual orphans: {sorted(orphans)}")
            for ln in proc.stdout.splitlines()[:30]:
                print(f"      {ln}")
        if ok:
            passed += 1
    # Cleanup tmp dir if empty.
    try:
        ADVERSARIAL_TMP_DIR.rmdir()
    except OSError:
        pass
    print()
    print(f"{passed}/{len(ENUM_CASES)} enumeration cases pass")
    return 0 if passed == len(ENUM_CASES) else 1


# ---------------------------------------------------------------------------
# Script-wrapper cases: exercise scripts_bin_refs / _next_real_bin so a
# package.json script like `cross-env CI=1 biome check` correctly credits
# `@biomejs/biome` rather than the wrapper itself. The 10x reviewer flagged
# the original "first non-env token" heuristic as too narrow: any project
# using cross-env / dotenv / dotenvx / env-cmd / a quoted env value would
# bypass the bin-name check.
# ---------------------------------------------------------------------------


@dataclass
class WrapperCase:
    id: str
    desc: str
    raw_cmd: str
    expected_bin: str | None  # None means "no real bin (e.g. unwrappable)"


WRAPPER_CASES: list[WrapperCase] = [
    WrapperCase(
        "W01",
        "cross-env wraps the real bin",
        "cross-env CI=1 biome check .",
        "biome",
    ),
    WrapperCase(
        "W02",
        "cross-env with multiple env tokens after the wrapper",
        "cross-env A=1 B=2 NODE_ENV=prod biome check",
        "biome",
    ),
    WrapperCase(
        "W03",
        "bare env-prefix run (no wrapper) still peels the env tokens",
        "FOO=bar biome check",
        "biome",
    ),
    WrapperCase(
        "W04",
        "quoted env value with spaces (shlex preserves it as one word)",
        'FOO="a b" biome check',
        "biome",
    ),
    WrapperCase(
        "W05",
        "npx + cross-env: runner peels, wrapper peels, real bin wins",
        "npx cross-env CI=1 biome check",
        "biome",
    ),
    WrapperCase(
        "W06",
        "pnpm exec + cross-env",
        "pnpm exec cross-env CI=1 biome check",
        "biome",
    ),
    WrapperCase(
        "W07",
        "dotenv with the `--` separator before the wrapped command",
        "dotenv -- biome check",
        "biome",
    ),
    WrapperCase(
        "W08",
        "dotenv with a flag-arg pair and `--` separator",
        "dotenv -e .env -- biome check",
        "biome",
    ),
    WrapperCase(
        "W09",
        "leading `./node_modules/.bin/` prefix is stripped",
        "./node_modules/.bin/biome check",
        "biome",
    ),
    WrapperCase(
        "W10",
        "concurrently is NOT a script wrapper -- it dispatches by "
        "script *name*, not bin, so the real bin is `concurrently` "
        "itself (the wrapped script names are credited by their own "
        "scripts entries, which scripts_bin_refs iterates separately)",
        'concurrently "npm:dev" "npm:typecheck"',
        "concurrently",
    ),
]


def run_wrapper_cases() -> int:
    import shlex

    passed = 0
    for wc in WRAPPER_CASES:
        try:
            words = shlex.split(wc.raw_cmd, posix = True)
        except ValueError:
            words = wc.raw_cmd.split()
        actual = _next_real_bin(words, 0)
        ok = actual == wc.expected_bin
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {wc.id}: {wc.desc}")
        if not ok:
            print(f"      raw_cmd={wc.raw_cmd!r}")
            print(f"      expected={wc.expected_bin!r}, actual={actual!r}")
        if ok:
            passed += 1

    # End-to-end integration: feed scripts_bin_refs a synthetic head_pkg
    # whose scripts use a wrapper, and confirm the package owning the
    # wrapped bin is credited (rather than the wrapper). This is the
    # actual call path used by find_command_usage().
    int_total = 0
    int_passed = 0
    int_cases = [
        (
            "I01",
            "cross-env wrapping `biome` credits @biomejs/biome",
            {"lint": "cross-env CI=1 biome check"},
            {"biome": "@biomejs/biome"},
            "@biomejs/biome",
        ),
        (
            "I02",
            "dotenv -- biome credits @biomejs/biome",
            {"lint": "dotenv -- biome check"},
            {"biome": "@biomejs/biome"},
            "@biomejs/biome",
        ),
        (
            "I03",
            "quoted env value before bin still credits the bin's owner",
            {"lint": 'FOO="a b" biome check .'},
            {"biome": "@biomejs/biome"},
            "@biomejs/biome",
        ),
        (
            "I04",
            "&& chain: both halves credit their owning packages",
            {"build": "tsc -b && cross-env CI=1 biome check"},
            {"tsc": "typescript", "biome": "@biomejs/biome"},
            None,  # checked via owning_pkgs below
        ),
    ]
    for case_id, desc, scripts, bin_to_pkg, expect_owner in int_cases:
        int_total += 1
        refs = scripts_bin_refs({"scripts": scripts}, bin_to_pkg)
        if case_id == "I04":
            owners = set(refs.keys())
            ok = owners == {"typescript", "@biomejs/biome"}
        else:
            ok = expect_owner in refs
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {case_id}: {desc}")
        if not ok:
            print(f"      scripts={scripts!r} bin_to_pkg={bin_to_pkg!r}")
            print(f"      refs={refs!r}")
        if ok:
            int_passed += 1

    total = len(WRAPPER_CASES) + int_total
    print()
    print(f"{passed + int_passed}/{total} wrapper-script cases pass")
    return 0 if (passed == len(WRAPPER_CASES) and int_passed == int_total) else 1


def main() -> int:
    head_pkg = json.loads(HEAD_PKG.read_text())
    print(f"Running {len(CASES)} edge cases against {SCRIPT.relative_to(REPO)}")
    print()
    results: list[tuple[Case, bool, str]] = []
    for c in CASES:
        ok, detail = run_case(c, head_pkg)
        results.append((c, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {c.id}: {c.desc}")
        if not ok:
            for line in detail.splitlines():
                print(f"      {line}")
    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"{passed}/{total} edge cases pass")

    print()
    print(f"Running {len(CLASSIFY_CASES)} classify() unit cases")
    print()
    cls_rc = run_classify_unit_tests()

    print()
    print(f"Running {len(ADV_CASES)} adversarial end-to-end cases")
    print()
    adv_rc = run_adversarial_cases()

    print()
    print(f"Running {len(PKG_FIELD_CASES)} package.json-field cases")
    print()
    pkg_rc = run_pkg_field_cases()

    print()
    print(f"Running {len(ENUM_CASES)} dead-dep enumeration cases")
    print()
    enum_rc = run_enum_cases()

    print()
    print(
        f"Running {len(WRAPPER_CASES)} script-wrapper cases "
        "(_next_real_bin + scripts_bin_refs end-to-end)"
    )
    print()
    wrap_rc = run_wrapper_cases()

    if (
        passed == total
        and cls_rc == 0
        and adv_rc == 0
        and pkg_rc == 0
        and enum_rc == 0
        and wrap_rc == 0
    ):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
