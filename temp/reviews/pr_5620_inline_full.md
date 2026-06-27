########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:503 ##########
![high](https://www.gstatic.com/codereviewagent/high-priority.svg)

The `allow_incomplete` parameter is defined in the signature of `_parse_llama3_python_tag` but is ignored when parsing the `NAME.call(...)` built-in form. If `allow_incomplete` is `False` (strict mode) and the closing parenthesis is missing (`depth > 0` after the loop), the incomplete call should be rejected rather than parsed.

```python
        while i < len(content) and depth > 0:
            ch = content[i]
            if in_string:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            i += 1
        if not allow_incomplete and depth > 0:
            continue
        body = content[m.end() : i]
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:696 ##########
![high](https://www.gstatic.com/codereviewagent/high-priority.svg)

The `_parse_mistral_array` helper does not accept or respect the `allow_incomplete` parameter. When `allow_incomplete` is `False`, the parser should not fall back to the healing path for unclosed arrays if `json.loads` fails.

```suggestion
    if content[k] == "[":
        return _parse_mistral_array(content, k, id_offset, allow_incomplete = allow_incomplete)
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:755 ##########
![high](https://www.gstatic.com/codereviewagent/high-priority.svg)

Update `_parse_mistral_array` to accept the `allow_incomplete` parameter so it can respect strict mode.

```suggestion
def _parse_mistral_array(content: str, start: int, id_offset: int, allow_incomplete: bool = True) -> list[dict]:
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:791 ##########
![high](https://www.gstatic.com/codereviewagent/high-priority.svg)

If `allow_incomplete` is `False`, do not fall back to the healing path for unclosed arrays when `json.loads` fails.

```suggestion
    except (json.JSONDecodeError, ValueError):
        if not allow_incomplete:
            return out
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:839 ##########
![high](https://www.gstatic.com/codereviewagent/high-priority.svg)

The `_parse_gemma_tool_calls` function ignores `allow_incomplete` when the closing `<|tool_call|>` tag is missing (`end_marker < 0`). Under strict mode (`allow_incomplete=False`), unclosed Gemma 4 tool calls should be rejected.

```python
    for m in _GEMMA_TC_RE.finditer(content):
        name = m.group(1)
        body_start = m.end() - 1
        end_marker = content.find(_GEMMA_TC_END, body_start)
        if not allow_incomplete and end_marker < 0:
            continue
        scan_end = end_marker if end_marker >= 0 else len(content)
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/llama_cpp.py:50 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

The import of `TOOL_XML_SIGNALS` is redundant because it is also imported as `_SHARED_TOOL_XML_SIGNALS` on the next line, and the un-aliased name is no longer used in this file.

```suggestion
    TOOL_XML_SIGNALS as _SHARED_TOOL_XML_SIGNALS,
```

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/safetensors_agentic.py:473 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Buffer bare JSON tool calls before streaming**

For Llama-3.2 bare-JSON emissions like `{"name":"web_search","parameters":...}`, this STREAMING-path safety parse only runs after the leading `{` has already been treated as normal content because it is not in `TOOL_XML_SIGNALS`. The call is executed, but the raw JSON has already been sent to the UI and `content_text` remains the same JSON because `strip_tool_markup` has no bare-JSON marker to remove. Buffer this form before emitting content, or suppress/clear the safety-parsed payload when it becomes a tool call.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/tool_call_parser.py:505 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Reject truncated Llama calls in strict mode**

When Auto-Heal is disabled, `allow_incomplete=False` should prevent partial calls from executing, but the Llama-3 `.call(...)` parser still builds arguments after the scan reaches EOF with `depth > 0`. A truncated emission such as `<|python_tag|>get_weather.call(location="Tokyo"` is therefore executed in strict mode; check the unmatched depth and skip unless incomplete healing is allowed.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/routes/inference.py:1283 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Advertise attribute-style function templates**

This allow-list omits the `<function name="...">` marker even though the new parser accepts that attribute form. Safetensors/MLX templates for that format can be classified as tool-capable by the generic `tools` checks, then get suppressed here because none of these markers match, so users never see tool support despite the loop being able to parse the calls.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/llama_cpp.py:7959 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Strip new GGUF tool markers before yielding**

The GGUF loop now watches for shared markers such as `[TOOL_CALLS]` and `<|python_tag|>`, but its streaming cleanup still uses the legacy `_TOOL_ALL_PATS` from `core.tool_healing`, which does not remove those formats. If a server returns one of these text-format calls after the buffer has entered STREAMING, the raw marker can be emitted to clients before the end-of-turn safety parser executes the tool; update the streaming strip path to use the same multi-format patterns.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/tool_call_parser.py:696 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Reject truncated Mistral arrays in strict mode**

This branch ignores `allow_incomplete`, so pre-v11 Mistral arrays still go through the healing path when Auto-Heal is disabled. For example, `[TOOL_CALLS] [{"name":"web_search","arguments":{"q":"x"}}` is missing the closing `]` but `_parse_mistral_array` will still recover and execute it; pass strictness through and skip unclosed arrays when `allow_incomplete=False`.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/safetensors_agentic.py:68 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Use enabled tool names in safetensors re-prompt**

When a safetensors request enables only other tools, such as `terminal`, `render_html`, `search_knowledge_base`, or MCP tools, this re-prompt still tells the model to call `web_search` or `python`. In those contexts the nudge steers the model toward disabled tools, leading to controller no-ops or a failed final answer instead of the enabled action; build the instruction from `active_tools` like the GGUF path does.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/routes/inference.py:1604 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Import the parser through the existing backend path**

This lazy import uses the package path `studio.backend...`, but `run.py` is explicitly self-contained and only inserts `studio/backend` so the rest of this module imports `core`, `routes`, and `state` as top-level modules. In that supported launch mode, any Anthropic/server-tool path that calls `_strip_tool_xml` raises `ModuleNotFoundError`; import from `core.inference.tool_call_parser` instead.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/tool_call_parser.py:133 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Parse every semicolon-separated Llama dot-call**

The parser docs say Llama-3 dot-call emissions can contain multiple calls separated by semicolons, but this regex requires every call to be preceded by a fresh `<|python_tag|>`. For an emission like `<|python_tag|>a.call(x=1); b.call(y=2)`, only `a` is returned and the second tool call is silently dropped, so parallel/compound Llama tool requests lose work.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/tool_call_parser.py:451 ##########
**<sub><sub>![P3 Badge](https://img.shields.io/badge/P3-lightgrey?style=flat)</sub></sub>  Allow strict zero-argument attribute calls**

In strict mode this rejects every attribute-style function call that has no `<param>` children, even if it is a complete `<function name="..."></function>` call. Zero-argument tools are valid for MCP and custom schemas, so with Auto-Heal disabled those calls are skipped solely because `param_starts` is empty; accept closed zero-arg calls with `{}`.

Useful? React with 👍 / 👎.

