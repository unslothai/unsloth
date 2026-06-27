########## gemini-code-assist[bot] :: studio/backend/core/inference/inference.py:255 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

Defensively check if `tokenizer` is `None` before attempting to access or modify its attributes. If both `tokenizer` and `processor` are missing from `model_info`, `tokenizer` will be `None`, leading to an `AttributeError` when setting `tokenizer.chat_template`.

```python
        tokenizer = model_info.get("tokenizer") or model_info.get("processor")
        if tokenizer is None:
            return None
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/inference.py:1082 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

Avoid redundant template rendering on every turn when tools are enabled. Since whether a template ignores tools is a static property of the template itself, we can cache this boolean on `model_info` after the first check. This avoids a redundant Jinja rendering pass on all subsequent turns, significantly improving performance.

```python
            if tools:
                ignores_tools = model_info.get("template_ignores_tools")
                if ignores_tools is None:
                    probe_no_tools = self._apply_chat_template_for_generation(
                        tokenizer,
                        template_messages,
                        tools = None,
                        enable_thinking = enable_thinking,
                        reasoning_effort = reasoning_effort,
                        preserve_thinking = preserve_thinking,
                    )
                    ignores_tools = (formatted_prompt == probe_no_tools)
                    model_info["template_ignores_tools"] = ignores_tools

                if ignores_tools:
                    native_prompt = self._render_with_native_template(
                        model_info,
                        template_messages,
                        tools,
                        enable_thinking,
                        reasoning_effort,
                        preserve_thinking,
                    )
                    if native_prompt:
                        logger.info(
                            "Override template for '%s' dropped tool schemas; "
                            "using the model's native template for this "
                            "tool-calling turn.",
                            self.active_model_name,
                        )
                        formatted_prompt = native_prompt
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:1281 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

Avoid potential skipping of subsequent tool calls if an earlier call is missing its closing tag. If the current tool call is missing its closing tag (e.g., due to truncation or model error) but a subsequent tool call is complete, `body.find(_DEEPSEEK_CALL_END, brace_end + 1)` will find the *next* tool call's closing tag, causing `pos` to skip past the next tool call entirely. Since the loop already correctly finds the next `_DEEPSEEK_SEP` from `pos`, we can simply set `pos = brace_end + 1` without searching for or skipping the closing tag, making the parser much more robust against missing/malformed closing tags.

```suggestion
        # Skip past optional ``<｜tool▁call▁end｜>``.
        pos = brace_end + 1
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:1415 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

Avoid potential skipping of subsequent tool calls if an earlier call is missing its closing tag. Setting `pos = brace_end + 1` is simpler and much more robust against missing/malformed closing tags.

```suggestion
                pos = brace_end + 1
```

########## gemini-code-assist[bot] :: studio/backend/core/inference/tool_call_parser.py:1455 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

Avoid potential skipping of subsequent tool calls if an earlier call is missing its closing tag. Setting `pos = brace_end + 1` is simpler and much more robust against missing/malformed closing tags.

```suggestion
        pos = brace_end + 1
```

########## gemini-code-assist[bot] :: studio/backend/routes/inference.py:1601 ##########
![medium](https://www.gstatic.com/codereviewagent/medium-priority.svg)

Ensure truncated DeepSeek and Kimi tool-call section blocks are robustly stripped. If the block is truncated mid-stream (which is very common during streaming or when max tokens are reached), these patterns will fail to match, leaving raw markup leaked to the user interface. Adding `|\Z` to the non-greedy match (similar to the other patterns in the regex) ensures that truncated blocks are robustly stripped.

```suggestion
    r"|<｜tool[▁_]calls[▁_]begin｜>.*?(?:<｜tool▁calls▁end｜>|\Z)"
    r"|<\|tool_calls_section_begin\|>.*?(?:<\|tool_calls_section_end\|>|\Z)"
```

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/tool_call_parser.py:1192 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Honor Auto-Heal off for DeepSeek truncation**

When `auto_heal_tool_calls=False`, the safetensors loop passes `allow_incomplete=False`, but this DeepSeek parser ignores that flag and treats a missing section end as if the body simply runs to EOF. A DeepSeek V3/R1 emission with balanced JSON but no `<｜tool▁call▁end｜>`/`<｜tool▁calls▁end｜>` will therefore still execute as a real tool call instead of being preserved as malformed text, unlike the existing strict XML behavior.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/tool_call_parser.py:54 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Handle the short DeepSeek opener end-to-end**

This accepts `<｜tool▁calls｜>` as a DeepSeek tool signal, but the safetensors capability whitelist and cleanup regexes only cover the `...calls...begin`/`call_begin` variants. For a DeepSeek R1 template/output using this llama.cpp-supported short opener, the route can suppress `supports_tools`, and if tools are forced the parsed turn leaves raw `<｜tool▁calls｜>...` markup in assistant content because stripping does not match it. Please keep advertisement and stripping patterns in sync with every accepted signal.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/tool_call_parser.py:252 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Buffer wrapper-less Gemma tool calls**

For Gemma-4 streams where `skip_special_tokens` removes the `<|tool_call>` wrapper, this new `call:NAME{...}` parser only fires after the turn finishes, but `call:` is not a streaming signal. In the safetensors SSE path the initial `call:web_search{...` text is therefore emitted as normal assistant content and cannot be retracted when the parser later executes the tool, so users see raw tool-call markup before the final answer.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/inference.py:242 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Reuse the loaded template for gated models**

When a template override drops tool schemas, this fallback reloads the tokenizer by `active_model_name` without the `hf_token` (or the already-loaded `chat_template_info`) that was used to load the model. For private/gated or local-cache-only models, `AutoTokenizer.from_pretrained` can fail here even though the native template is already available on the loaded tokenizer, so the tool-calling turn falls back to the override prompt with no tools rendered.

Useful? React with 👍 / 👎.

########## chatgpt-codex-connector[bot] :: studio/backend/core/inference/safetensors_agentic.py:587 ##########
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Respect the requested tool-call budget**

This hard cap silently drops every distinct tool call after the eighth before execution, regardless of the caller's `max_tool_calls_per_message` / `max_tool_iterations` budget (the route defaults that budget to 25). If a model emits nine independent calls in one tool turn, only the first eight are represented in the assistant message and executed, so the omitted calls receive no result or budget-exhausted nudge.

Useful? React with 👍 / 👎.

