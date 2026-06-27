########## gemini-code-assist[bot] :: studio/backend/core/tool_healing.py:48 ##########
![high](https://www.gstatic.com/codereviewagent/high-priority.svg)

During streaming generation (in the `_S_STREAMING` state), `_strip_tool_markup_streaming` is called on every new token. Because the unclosed regexes for `[TOOL_CALLS]` and `[ARGS]` require the opening brace `{` (e.g., `[TOOL_CALLS]\w+\s*\{.*$`), any partial tool call generated before the `{` (such as `[TOOL_CALLS]web_search` or `web_search[ARGS]`) will not match the regex and will be streamed/leaked to the UI. Once the `{` is generated, the regex matches and strips it from the cumulative text, but the already-streamed prefix cannot be erased from the UI, leaving a permanent visual leak.

Removing the requirement for `{` and matching the prefix immediately (similar to `<tool_call>.*$`) prevents this leak.

```suggestion
    re.compile(r"\[TOOL_CALLS\].*$", re.DOTALL),
    re.compile(r"\b\w+\[ARGS\].*$", re.DOTALL),
```

########## gemini-code-assist[bot] :: studio/backend/routes/inference.py:1589 ##########
![high](https://www.gstatic.com/codereviewagent/high-priority.svg)

Similar to the streaming leak in `_TOOL_ALL_PATS`, the new patterns added to `_TOOL_XML_RE` require the opening brace `{` and a balanced JSON block. If the speculative buffer contains an incomplete or unclosed tool call at the end of the stream (e.g., `[TOOL_CALLS]web_search{"query":"weather"`), it will not be stripped and will leak to the UI.

We should update these patterns to support matching unclosed blocks up to the end of the string (`\Z`), similar to how the other XML/Gemma shapes are handled.

```suggestion
    r"|\[TOOL_CALLS\]\w+\s*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|.*?(?:\Z))"
    r"|\b\w+\[ARGS\]\s*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|.*?(?:\Z))",
```

########## gemini-code-assist[bot] :: studio/backend/core/tool_healing.py:568 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

The parsing loops and dictionary construction for Pattern 3 (`[TOOL_CALLS]`) and Pattern 4 (`[ARGS]`) are identical. We can consolidate them into a single loop over the two regexes to reduce code duplication and improve maintainability.

```python
    # Pattern 3 & 4: Mistral bracket-tag [TOOL_CALLS]name{json} and Rehearsal name[ARGS]{json}.
    if not tool_calls:
        for regex in (_MISTRAL_BRACKET_RE, _REHEARSAL_RE):
            for m in regex.finditer(content):
                tool_name = m.group(1)
                args_start = m.end()
                args_end = _balanced_json_span(content, args_start)
                if args_end is None:
                    continue
                try:
                    args = json.loads(content[args_start : args_end + 1])
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(args, dict):
                    continue
                tool_calls.append(
                    {
                        "id": f"call_{id_offset + len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args),
                        },
                    }
                )
            if tool_calls:
                break
```

<details>
<summary>References</summary>

1. To improve efficiency, avoid redundant data iterations. Combine checks and transformations into a single loop and return computed values for callers to reuse.
</details>

########## chatgpt-codex-connector[bot] :: studio/backend/core/tool_healing.py:81 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Accept hyphenated bracket tool names**

When enabled MCP/custom tools contain `-` in the function name (allowed by `_OPENAI_FN_NAME_RE` in `studio/backend/core/inference/tools.py:836-839` and already supported by the XML/Gemma parsers), these new regexes do not capture the full name. For example `[TOOL_CALLS]mcp__srv__list-issues{"q":"x"}` is dropped entirely, while `mcp__srv__list-issues[ARGS]{"q":"x"}` matches only `issues`, so the tool loop rejects the real MCP call as disabled or can misdirect it if a shorter tool name exists.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/tool_healing.py:48 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Strip partial bracket tags while buffering**

In the GGUF stream path, `llama_cpp.py:8230-8237` enters DRAINING as soon as `[TOOL_CALLS]` or `[ARGS]` appears, but these new in-progress strip patterns do not match until the following tool name and `{` have also arrived. If the stream splits after the tag (for example `[TOOL_CALLS]` in one delta and `web_search{...}` in the next, or `python[ARGS]` before `{`), `_strip_tool_markup_streaming` emits the raw bracket marker to the UI before the call is parsed; the existing XML patterns strip bare open markers and these need equivalent partial-tag coverage.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/tool_healing.py:386 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve think tags inside tool arguments**

When a parsed tool call's argument string contains literal `<think>...</think>` markup (for example python or render_html code that the user asked to output), this unconditional pre-strip runs before any tool-call spans are identified, so it removes the literal from the argument or can make the JSON tool call invalid; `<function=python><parameter=code>print("<think>x</think>")</parameter></function>` now executes with `{"code":"print(\"\")"}`. Restrict think-block removal to text outside tool-call payloads so real tool arguments are not corrupted.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/tool_healing.py:525 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Parse the Mistral tool-call shapes already used in repo**

The in-repo Mistral/Ollama templates emit `[TOOL_CALLS] [` / `[TOOL_CALLS][` JSON-array calls (`unsloth/ollama_template_mappers.py:99-101` and `129-131`) and Mistral-small-3.2 emits `[TOOL_CALLS]name[CALL_ID]i[ARGS]{...}` (`unsloth/ollama_template_mappers.py:192-194`), but this new path only handles `[TOOL_CALLS]name{...}`. Those outputs either produce no parsed calls or, for the `[CALL_ID]` form, fall through to the rehearsal parser and execute/reject tool `0` instead of the real tool name, so the advertised Mistral fallback still fails for supported templates.

Useful? React with 👍 / 👎.

