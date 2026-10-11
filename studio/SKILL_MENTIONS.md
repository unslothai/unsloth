# Explicit Agent Skill mentions

With an enabled skill in the effective model's Unsloth Studio tool catalog, Unsloth Studio reads
the complete `SKILL.md` for an explicit `@skill-name` **before the first
generation**. With Code off, the composer offers `read_skill` alone, and only for
a thread whose user messages mention an enabled skill, so a plain chat stays out
of the tool loop. It does not depend on a model calling `read_skill`. This loads
instructions, not referenced resources, scripts, or skill creation. Existing
account discovery, capability, tool selection and permission policy still apply.
Ask mode waits for the ordinary scoped read approval; Auto permits this read-only
operation. A denied/failed load is shown as unavailable, not successful, and the
model is told the named skill was not loaded; a denied skill is never read.

## Intent contract

The composer currently serializes a picker selection as plain `@name`, without
selected-intent metadata. Thus selected mentions and exact typed tokens have the
same contract: a lowercase, whitespace-delimited skill name in the **latest user
message**, optionally followed by sentence punctuation. Prefix matching is never
used. Inline/fenced/indented Markdown code, blockquotes, balanced quotation spans,
email addresses, URL/path suffixes, and assistant/history mentions do not load
instructions. Unquoted pasted prose is indistinguishable from typed prose and
uses the same contract. Use quotation marks or code formatting when discussing a
mention literally. Code and blockquotes follow CommonMark block structure, with one
exception: user text is shown unrendered, so a line typed directly under a quote
without its own `>` is a reply, not quoted text. A backslash-escaped quote mark does
not end a quotation span, and an apostrophe inside a word is not a quote mark. This
is deliberately not a Markdown plugin/runtime.

## Context and evidence

The secure resource reader provides one full manifest snapshot (no paginated
splice). Total newly injected manifests are capped at 32,000 UTF-8 bytes, or twice
the known local context-token window in bytes, whichever is smaller. An oversized
manifest is refused whole with a context-budget explanation; it is never marked
as completely loaded after truncation. Native prompt fitting remains authoritative
and retains the injected system instructions; a provider may still refuse its own
context window. External model windows are not inferred from the resident local
model. Loading does not promise successful inference.

Backend `skill_load` events report loading, actual successful reads (character
count and SHA256), or failure. The UI persists a distinct `studio_load_skill`
activity card for replay/approval, **not an assistant/model function call**. It is
excluded from outbound model tool history. Provider-authored control frames are
sanitized so a provider cannot forge this evidence.

Automatic loads are request-local. The next explicit mention securely re-reads
the current resource, including retries and same-chat follow-ups; a saved UI card
is never taken as proof that instructions survived compaction. If the complete
current manifest is already in actual system/tool context, it is not injected a
second time. Continue/partial-assistant resumes do not activate old mentions.

The Studio chat-completions path covers local GGUF, safetensors, managed engines,
and external Studio-owned tool loops (including the shared Codex transport). It
does not change client-tool passthrough or introduce mention execution in unrelated
Anthropic/OpenAI compatibility endpoints. Tool-less native models and Code-off
requests retain their existing behavior.
