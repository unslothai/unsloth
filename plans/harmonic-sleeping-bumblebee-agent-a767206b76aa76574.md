# Audit: DeepSeek / GLM / Kimi tool-call parser vs upstream

Target: `unsloth/studio/backend/core/inference/tool_call_parser.py`
Branch: `studio-tools-deepseek-glm-kimi`
Scope: DeepSeek R1/V3/V3.1, GLM 4.5/4.6/4.7, Kimi K2. Read-only audit; no files edited except this report.

## Marker codepoints (ground truth = HF chat templates) — ALL CORRECT
- DeepSeek pipe = U+FF5C `｜`, block = U+2581 `▁`; verified byte-for-byte against
  `hf_templates/deepseek_r1/.../tokenizer_config.json` and `deepseek_v31/...`.
  Our `_DEEPSEEK_END/_CALL_BEGIN/_SEP/_CALL_END` match exactly.
- R1 emission (template): `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>` + type + `<｜tool▁sep｜>` + name + `\n```json\n` + args + `\n``` ` + `<｜tool▁call▁end｜>`.
- V3.1 emission: `...<｜tool▁call▁begin｜>` + name + `<｜tool▁sep｜>` + args(bare json) + `<｜tool▁call▁end｜>`.
- GLM (template glm46): `\n<tool_call>` + name + `\n` + per-arg `<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n` + `</tool_call>`; non-string values via `tojson(ensure_ascii=False)`, strings raw. Matches our `_GLM_*`.
- Kimi: ASCII pipe U+007C; id rendered from `tool_call['id']` (`functions.NAME:IDX`); args `tojson` if not string. Matches our `_KIMI_*`.

## FINDINGS (by severity)

### REAL-BUG 1 — Short DeepSeek opener `<｜tool▁calls｜>` leaks raw markup (strip gap)
- our file:line: strip patterns at tool_call_parser.py:75 and :88 use `<｜tool[▁_]calls[▁_]begin｜>` (require the `begin` suffix); opener regex `_DEEPSEEK_BEGIN_RE` (:190-192) and `TOOL_XML_SIGNALS` (:54) DO accept the short `<｜tool▁calls｜>`.
- Effect: input with the short opener is PARSED into a call but never STRIPPED, so the raw
  `<｜tool▁calls｜>function<｜tool▁sep｜>...` text leaks to the user in both streaming and final output.
- Triggering input (also an existing test, tests/test_safetensors_tool_loop.py:761 `test_r1_short_form_outer_marker`):
  `<｜tool▁calls｜>function<｜tool▁sep｜>get_time\n```json\n{"city":"Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
  - parse: 1 call (correct). strip_tool_markup(final=True): returns the input UNCHANGED (leak). Canonical `<｜tool▁calls▁begin｜>` strips to `''`.
- Upstream: llama.cpp chat-diff-analyzer fixes `section_start="<｜tool▁calls▁begin｜>"`; the short form is our extra tolerance, so we own consistency. If we accept it for parse+signal, we must strip it.
- Fix: in `_TOOL_CLOSED_PATS` (:75) and `_TOOL_ALL_PATS` (:88) replace `tool[▁_]calls[▁_]begin` with an optional-`begin` form, e.g. `<｜tool(?:[▁_ ]calls[▁_ ]begin|[▁_]calls|\\_calls\\_begin)｜>` mirroring `_DEEPSEEK_BEGIN_RE`. Simplest: reuse `_DEEPSEEK_BEGIN_RE.pattern` to build both strip regexes so opener tolerance and strip tolerance can never drift again.

### REAL-BUG 2 — Kimi ignores `allow_incomplete` (strict mode heals truncated calls)
- our file:line: `_parse_kimi_tool_calls` (:1385) / `_parse_kimi_section_body` (:1422) never read `allow_incomplete`; signature accepts it but it is unused.
- Effect: with Auto-Heal OFF (`allow_incomplete=False`), a section truncated mid-stream (complete JSON body but missing `<|tool_call_end|>` and `<|tool_calls_section_end|>`) is still emitted as a call. Violates the contract documented at `parse_tool_calls_from_text` (:399-414): strict mode "only accepts a well-formed, closed call."
- Triggering input: `<|tool_calls_section_begin|><|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{"x":1}` (no end markers).
  - ours strict: returns the call. Upstream vLLM `tool_call_regex.findall` returns `[]` (regex requires `<|tool_call_end|>`).
- Fix: in `_parse_kimi_tool_calls`, when `not allow_incomplete`, require `section_end >= 0` before parsing the body (mirror DeepSeek :1225); in `_parse_kimi_section_body` require the per-call `<|tool_call_end|>` to be present (find it and bound the JSON to before it) before appending, else skip in strict mode.

### REAL-BUG 3 — GLM ignores `allow_incomplete` (strict mode heals calls missing `</tool_call>`)
- our file:line: `_parse_glm_tool_calls` (:1326) never reads `allow_incomplete`.
- Effect: with Auto-Heal OFF, a call missing `</tool_call>` (`close == -1`, body taken out to EOF) is still emitted.
- Triggering input: `<tool_call>a\n<arg_key>x</arg_key>\n<arg_value>1</arg_value>` (no `</tool_call>`).
  - ours strict: returns the call. Upstream SGLang/vLLM `func_call_regex = <tool_call>.*?</tool_call>` returns `[]` (requires close tag).
- Fix: in `_parse_glm_tool_calls`, when `not allow_incomplete and close < 0`, skip the call (continue/break) instead of using `body_end = len(content)`.

### EDGE-CASE 4 — GLM literal backslash-n separators not handled
- our file:line: `_GLM_TC_OPEN_RE` (:218) stops the name only at a real `\n`, `<arg_key>`, or `</tool_call>`; `_GLM_ARG_PAIR_RE` (:222) only sees real newlines/`<arg_key>`.
- Effect: if newlines are escaped to literal `\n` (two chars) in transport, the name is captured as `get_weather\n` and args may be dropped.
- Triggering input: `<tool_call>get_weather\\n<arg_key>city</arg_key>\\n<arg_value>Beijing</arg_value>\\n</tool_call>` (backslash-n literals) -> ours: name `get_weather\n`, args `{}`.
- Upstream: SGLang Glm4MoeDetector/Glm47MoeDetector explicitly handle both via `(?:\\n|\n)` in `func_detail_regex` and `func_arg_regex`.
- Severity edge-case: the HF glm46 template renders REAL newlines (jinja `'\n'` -> U+000A), confirmed from chat_template.jinja. Only matters if a transport layer escapes them. Low priority; mirror SGLang's `(?:\\n|\n)` tolerance if we want parity.

### COSMETIC 5 — Kimi `:IDX` suffix optional (more lenient than all 3 upstreams)
- our file:line: `_KIMI_ID_RE` (:238) `^(?:functions\.)?([\w\.\-]+)(?::(\d+))?$` makes `:IDX` optional.
- Effect: `functions.get_weather` (no `:0`) parses for us; vLLM (`[^<]+:\d+`) and SGLang (`:\d+` required) would reject it. Deliberate robustness, not a bug; canonical template always emits `:IDX`. Note only if strict upstream-parity is desired.

## Multi-call handling (advance = `pos = brace_end + 1`) — CORRECT, matches upstream intent
- DeepSeek V3.1: two calls where the 2nd lacks `<｜tool▁call▁end｜>` -> both parsed. The `pos = brace_end + 1` advance (:1319) correctly re-locates the next `<｜tool▁sep｜>` and does NOT skip the in-between call (the old "search forward for end marker" approach would have). Matches vLLM/SGLang `findall` of each `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` over a shared envelope.
- Kimi: 2nd call missing `<|tool_call_end|>` -> both parsed; `pos = brace_end + 1` (:1494) is correct for the same reason.
- GLM: first call missing `</tool_call>` -> our parser merges both arg sets under the first name. VERIFIED this is exactly what SGLang does too (`<tool_call>.*?</tool_call>` swallows the nested `<tool_call>` as text, then `func_arg_regex.findall` picks up both pairs). CONSISTENT with upstream, not a bug.

## Name extraction — CORRECT
- DeepSeek V3.1 name-before-sep, V3/R1 `function<sep>NAME\n```json`, optional `<｜tool▁call▁begin｜>` absent: all parse. Dotted/hyphenated (`mcp.srv.list-issues`) preserved. Walk-back stops at `\n<>` (the `>` closes `<｜tool▁call▁begin｜>`; pipe is U+FF5C not `>`), correct.
- Kimi `functions.NAME:IDX` -> strips `functions.` and `:IDX`; dotted `functions.module.fn:0` -> `fn` (last segment), matches vLLM `split(":")[0].split(".")[-1]`. Hyphen names OK. Bare-counter id (`3`) dropped (matches vLLM; we have no schema to infer).
- GLM name-then-args, zero-arg (`<tool_call>get_current_date</tool_call>` -> `{}`), hyphen/underscore names OK.

## Strict-mode (truncated envelope) — DeepSeek CORRECT, Kimi/GLM BROKEN
- DeepSeek: strict + no `<｜tool▁calls▁end｜>` -> `[]` (correct, :1225). Healed -> emits. Matches XML/Mistral strict paths.
- Kimi/GLM: see REAL-BUG 2 & 3.

## CONFIRMED CORRECT
1. DeepSeek marker codepoints (U+FF5C / U+2581) byte-for-byte vs both HF templates.
2. `_DEEPSEEK_BEGIN_RE` accepts all 5 opener variants AND prefers the full `begin` form over the short form (alternation order; `.end()` lands after the full marker).
3. DeepSeek R1 fence path, V3/V3.1 bare-json path, multi-call shared envelope, 2nd-call-missing-end recovery, dotted/hyphen names, optional `<｜tool▁call▁begin｜>`.
4. DeepSeek strict-mode truncation rejection.
5. Kimi marker set + ASCII pipe; full-id round-trip preserved on `id`; name = last dotted segment; bare-counter dropped; multi-section outer loop; bare `<|tool_call_begin|>` (no section) fallback; 2nd-call-missing-end recovery.
6. GLM 4.5/4.6 (newline) + 4.7 (no newline) openers, zero-arg call, tojson object values, numeric coercion, raw string-value whitespace preservation (matches vLLM glm4_moe never-strip), back-to-back multi-call, GLM-vs-Qwen first-char `[^\n<{]` disambiguation.
7. GLM multi-call merge-on-missing-close matches SGLang behavior exactly.

## Suggested fix priority
1. REAL-BUG 1 (short-opener strip leak) — user-visible markup leak; reuse `_DEEPSEEK_BEGIN_RE.pattern` in strip regexes.
2. REAL-BUG 2 + 3 (Kimi/GLM strict-mode) — honor `allow_incomplete` to match the documented contract and upstream non-streaming parsers.
3. EDGE-CASE 4 (GLM literal `\n`) — optional `(?:\\n|\n)` parity with SGLang.
