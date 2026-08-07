# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for #7732: regeneration branches must survive load/import."""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
CHAT_TREE = (
    WORKDIR / "studio" / "frontend" / "src" / "features" / "chat" / "utils" / "chat-message-tree.ts"
)
FRONTEND_DIR = WORKDIR / "studio" / "frontend"
TEMP = WORKDIR / "temp" / "chat_message_tree_7732"


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not CHAT_TREE.exists():
        pytest.skip("chat-message-tree.ts not present")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if result.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _ensure_harness():
    TEMP.mkdir(parents = True, exist_ok = True)
    (FRONTEND_DIR / "register.mjs").write_text(
        "import { register } from 'node:module';\nregister('./loader.mjs', import.meta.url);\n",
        encoding = "utf-8",
    )
    (FRONTEND_DIR / "loader.mjs").write_text(
        "export function resolve(specifier, context, next) {\n"
        "  if (specifier.endsWith('/types')) return next(specifier + '.ts', context);\n"
        "  return next(specifier, context);\n"
        "}\n",
        encoding = "utf-8",
    )


def _run(script: str) -> dict:
    _require_node()
    _ensure_harness()
    script_path = FRONTEND_DIR / "temp_chat_message_tree_run.mts"
    script_path.write_text(script, encoding = "utf-8")
    env = dict(os.environ, NODE_NO_WARNINGS = "1")
    result = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--import=./register.mjs",
            "--no-warnings",
            "temp_chat_message_tree_run.mts",
        ],
        cwd = str(FRONTEND_DIR),
        capture_output = True,
        text = True,
        timeout = 30,
        env = env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    last = [line for line in result.stdout.strip().splitlines() if line.strip()][-1]
    return json.loads(last)


def _tree_path():
    rel = os.path.relpath(CHAT_TREE, FRONTEND_DIR).replace("\\", "/")
    return f"./{rel}"


def test_repair_assistant_parent_ids_restores_sibling_branches():
    rel = _tree_path()
    out = _run(
        textwrap.dedent(
            f"""
            import {{ repairAssistantParentIds, resolveHeadMessageId }} from '{rel}';
            import {{ MessageRepository }} from '@assistant-ui/core/internal';

            const userId = 'user-1';
            let t = 1000;
            const stored = [
              {{ id: userId, role: 'user', parentId: null, createdAt: t }},
              {{ id: 'asst-1', role: 'assistant', parentId: userId, createdAt: t += 100 }},
            ];
            for (let i = 2; i <= 6; i++) {{
              t += 100;
              stored.push({{ id: `asst-${{i}}`, role: 'assistant', parentId: null, createdAt: t }});
            }}

            const repaired = repairAssistantParentIds(stored);
            const headId = resolveHeadMessageId(repaired);
            const repo = new MessageRepository();
            repo.import({{
              headId,
              messages: repaired.map((m) => ({{
                parentId: m.parentId ?? null,
                message: {{
                  id: m.id,
                  role: m.role,
                  content: [{{ type: 'text', text: m.id }}],
                  status: {{ type: 'complete', reason: 'unknown' }},
                  metadata: {{ custom: {{}} }},
                  createdAt: new Date(m.createdAt),
                  attachments: [],
                }},
              }})),
            }});

            console.log(JSON.stringify({{
              branchCount: repo.getBranches('asst-6').length,
              exported: repo.export().messages.length,
            }}));
            """
        )
    )
    assert out["branchCount"] == 6
    assert out["exported"] == 7


def test_partial_db_loss_still_reports_two_when_only_ends_remain():
    """When only the first and latest regen rows survive storage, picker shows 1/2."""
    rel = _tree_path()
    out = _run(
        textwrap.dedent(
            f"""
            import {{ MessageRepository }} from '@assistant-ui/core/internal';

            const userId = 'user-1';
            const stored = [
              {{ id: userId, role: 'user', parentId: null, createdAt: 1000 }},
              {{ id: 'asst-1', role: 'assistant', parentId: userId, createdAt: 1100 }},
              {{ id: 'asst-6', role: 'assistant', parentId: userId, createdAt: 1600 }},
            ];
            const repo = new MessageRepository();
            for (const m of stored) {{
              repo.addOrUpdateMessage(m.parentId, {{
                id: m.id,
                role: m.role,
                content: [{{ type: 'text', text: m.id }}],
                status: {{ type: 'complete', reason: 'unknown' }},
                metadata: {{ custom: {{}} }},
                createdAt: new Date(m.createdAt),
                attachments: [],
              }});
            }}
            repo.resetHead('asst-6');
            console.log(JSON.stringify({{ branchCount: repo.getBranches('asst-6').length }}));
            """
        )
    )
    assert out["branchCount"] == 2


def test_resolve_head_prefers_continued_branch_over_newer_regen_sibling():
    rel = _tree_path()
    out = _run(
        textwrap.dedent(
            f"""
            import {{ prepareBranchedMessagesForImport, resolveHeadMessageId }} from '{rel}';
            import {{ MessageRepository }} from '@assistant-ui/core/internal';

            const userId = 'user-1';
            const stored = [
              {{ id: userId, role: 'user', parentId: null, createdAt: 1000 }},
              {{ id: 'asst-1', role: 'assistant', parentId: userId, createdAt: 1100 }},
              {{ id: 'asst-2', role: 'assistant', parentId: userId, createdAt: 1500 }},
              {{ id: 'user-2', role: 'user', parentId: 'asst-1', createdAt: 1200 }},
              {{ id: 'asst-3', role: 'assistant', parentId: 'user-2', createdAt: 1300 }},
            ];
            const prepared = prepareBranchedMessagesForImport(stored);
            const headId = resolveHeadMessageId(prepared);
            const repo = new MessageRepository();
            repo.import({{
              headId,
              messages: prepared.map((m) => ({{
                parentId: m.parentId ?? null,
                message: {{
                  id: m.id,
                  role: m.role,
                  content: [{{ type: 'text', text: m.id }}],
                  status: {{ type: 'complete', reason: 'unknown' }},
                  metadata: {{ custom: {{}} }},
                  createdAt: new Date(m.createdAt),
                  attachments: [],
                }},
              }})),
            }});
            console.log(JSON.stringify({{ headId, active: repo.headId }}));
            """
        )
    )
    assert out["headId"] == "asst-3"
    assert out["active"] == "asst-3"
