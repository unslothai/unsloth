# Comment cleanup across the tool-call parser PRs (#5620, #5624, #5704)

## Context

The three Studio tool-call PRs are correct and merge-pending, but their comments
are verbose: many own-line block comments restate the code, and several
docstrings run 200+ chars. The user wants every comment in the PR-introduced code
made succinct (fewer lines, removed where the code is self-evident, aggressively
trimmed on small internal helpers) **without changing behavior or destroying
intent** -- correctness "why" notes and upstream provenance must survive in
compressed form. Style reference: `unslothai/scripts#128`, which (now on `main`)
also ships the verification tool we will gate on: `scripts/comment_tools.py`
(`collect` = enumerate comments; `check` = prove an edit touched only
comments/whitespace, not code).

This is a **comment-only** change: no symbol renames, no logic edits, no test
expectation changes, no import churn. All three PRs touch only Python (no TS/JS).

## Scope (what to touch)

Touch **only comments/docstrings introduced or modified by the PRs**, never
pre-existing comments in the large host files.

Comment surface (measured with `comment_tools.py collect`):
- `core/inference/tool_call_parser.py` -- net-new (1424 of 1581 lines are PR
  code): **203 line comments + 23 docstrings (14 over 200 chars)**. This is ~90%
  of the work.
- `core/inference/safetensors_agentic.py` -- 33 PR-added comment lines (verbose
  re-prompt / dedup / bare-JSON-gating "why" blocks).
- `routes/inference.py` -- 14 PR-added comment lines.
- `core/inference/llama_cpp.py` -- 10 PR-added comment lines.
- `core/tool_healing.py` -- 9 PR-added comment lines.
- `core/inference/inference.py` -- 5 PR-added comment lines.
- Test files (`tests/test_*tool*`, `test_gemma_*`, `test_pr5624_regressions.py`,
  etc.) -- conservative pass: drop comments that restate the test name; keep one
  compact note explaining why a fixture is intentionally malformed.

## Classification rules (applied per comment)

1. **CUT** -- delete entirely when the next line / function / variable name makes
   it obvious (e.g. `# loop over matches` above a `for`, `out: list = []  # result`,
   one-line docstrings on tiny private helpers whose name says it all).
2. **SHORTEN** -- compress multi-sentence comments and long docstrings to one or
   two lines, keeping only the non-obvious essence.
3. **PRESERVE (compressed)** -- keep correctness/provenance notes, shortened to
   the shortest unambiguous form:
   - upstream port refs ("ports llama.cpp `common_chat_parse_glm_4_5` @ 51fa458a")
   - SGLang/vLLM parity notes
   - strict-mode / `allow_incomplete` rationale
   - non-obvious malformed-emission, Unicode/full-width-pipe marker, and
     stripping-order rationale (these guard maintainers from "simplifying" odd-but-
     intentional behavior).

Be most aggressive on leading-underscore internal helpers; keep a concise
docstring on the few exported names (`parse_tool_calls_from_text`,
`has_tool_signal`, `strip_tool_markup`, `TOOL_XML_SIGNALS`, `TOOL_ERROR_NUDGE`).

### tool_call_parser.py specifics (from the classification pass)

**Cut aggressively (tiny private helpers — name says it all):** docstrings on
`_balanced_bracket_end` (263), `_balanced_brace_end` (1014),
`_gemma_balanced_brace_end` (1045), `_gemma_parse_value` (1070),
`_gemma_parse_mapping_body` (1207), `_skip_mistral_call_id` (294); plus obvious
restatement comments (e.g. 331 "skip whitespace", 356 array-shape label). Keep at
most the one non-obvious clause ("ignores braces inside JSON strings").

**NEVER touch (load-bearing — keep verbatim or compress only):** the four
`51fa458a92d6` provenance lines (1256-1258, 1380, 1451); O(N^2)/DoS notes
(220-222, 1314-1316); every strict-mode / `allow_incomplete` rationale block;
whitespace-preservation notes (166-169, 512-517, 1402-1406); full-width-pipe /
Unicode notes (200-202, 636-639); the shared-source "can never be left
un-stripped" invariant (63-66); and the "scanning forward could skip an
intervening call" correctness notes (1359-1362, 1577-1579).

**Keep a concise docstring** on the public surface (`parse_tool_calls_from_text`,
`strip_tool_markup`, the module format-map docstring); `has_tool_signal` needs
none. Compress the ~14 docstrings/blocks over 200 chars (entry-fn ordering
rationale at 425/434-438/447-452 -> 1-2 lines each, but the "run only after
tool_healing finds nothing, so a strict-rejected call is never re-healed" point
must survive).

### Representative before/after (from `safetensors_agentic.py`)
```
# Without a grammar constraint a small model can loop, emitting the same tool
# call dozens of times in one turn (llama-server's lazy grammar prevents this
# on the GGUF side). Collapse exact-duplicate calls within a turn and cap the
# number kept so one runaway turn cannot fan out into many tool executions.
```
->
```
# No grammar constraint -> a small model can repeat one call many times (GGUF's
# lazy grammar prevents this). Collapse exact dups and cap the count per turn.
```

## Critical files
- `studio/backend/core/inference/tool_call_parser.py` (primary)
- `studio/backend/core/inference/safetensors_agentic.py`
- `studio/backend/routes/inference.py`
- `studio/backend/core/inference/llama_cpp.py`
- `studio/backend/core/tool_healing.py`
- `studio/backend/core/inference/inference.py`
- `studio/backend/tests/*` (conservative)
- Tool (read-only, already cloned): `workspace_38/scripts_repo/scripts/comment_tools.py`

## Execution order (respect the #5620 -> #5624 stack)

For each branch: record `BASE=$(git rev-parse HEAD)` before editing, edit
comments, then run the **three gates** below, then commit + push.

1. **#5620 `studio-tools-multi-format-v2`** (base): clean the shared files
   (tool_call_parser.py's Qwen/Llama/Mistral/Gemma sections, tool_healing.py,
   safetensors_agentic.py, llama_cpp.py, routes/inference.py, its tests).
2. **Merge #5620 -> #5624** (comment-only; merges cleanly). Then on **#5624
   `studio-tools-deepseek-glm-kimi`** clean only the #5624-only additions: the
   DeepSeek/GLM/Kimi parser sections of tool_call_parser.py + inference.py + its
   extra tests.
3. **#5704 `daniel/studio-mistral-tool-call-rescue`** (independent): clean its own
   changed comments (its distinct tool_call_parser.py / tool_healing.py /
   routes/inference.py / llama_cpp.py + tests) from a clean checkout.

Push in dependency order: #5620, then updated #5624, then #5704. If pre-commit.ci
has advanced a remote head, `git fetch` + `git rebase` (or reset to remote) before
pushing, as in prior rounds.

## Verification gates (per branch, all must pass before commit)

1. **Comment-only proof (AST):**
   `python <repo>/../scripts_repo/scripts/comment_tools.py check --changed --base $BASE --repo <unsloth_repo> --strip-docstrings`
   -> must exit 0 (proves no code signature changed; docstring edits allowed).
2. **Byte-compile:** `python -m py_compile` over every edited `.py` (catches edit slips).
3. **Parser suite stays green (same pass count, currently 329):**
   `python -m pytest -q tests/test_tool_call_parser_strict.py tests/test_safetensors_tool_loop.py tests/test_tool_xml_strip.py tests/test_gemma_tool_parse_edge_cases.py tests/test_llama_cpp_tool_loop.py tests/test_gguf_tool_non_streaming.py tests/test_safetensors_capability_advertise.py tests/test_tool_loop_controller.py`
   (run from `studio/backend`). Pre-existing env failures (sqlite-vec/tilelang/
   fastmcp) are unrelated.

Also eyeball `git diff` to confirm only comment lines moved and that line count
dropped. Report per branch: comment-only? (gate 1), py_compile, test pass count,
commit hash, push status.
