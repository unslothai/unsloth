# Enable manual compaction for external Studio-tool providers

## Summary

Allow the existing `/compact` action to work for external OpenAI-compatible
providers when the selected model supports Unsloth Studio tools.

## Problem

The chat menu and request adapter previously treated manual compaction as a
local GGUF/MLX-only feature. A remote Unsloth/llama.cpp model could support
tools and saved-chat recall, but the client refused to queue `/compact` and did
not send `compact_context`.

## Changes

- Recognize external models with Studio-tool support as valid compaction targets.
- Preserve the existing one-shot compaction request queue.
- Send `context_overflow: "truncate_oldest"` and `compact_context: true` for an
  explicit external `/compact` request, even when automatic compaction is off.
- Keep automatic compaction behavior unchanged for API providers and local
  GGUF/MLX/managed engines.
- Keep backend capability and archive checks authoritative.
- Add regression coverage for the request fields and menu behavior.

## Scope and compatibility

This change does not expose ChatGPT subscription/OAuth providers through the
public API. It only enables the existing saved-chat/tool-based compaction path
for external providers that already advertise Studio-tool support.

## Tests

Targeted tests:

- `tests/manual-compaction.test.ts`: 17 passed
- `tests/auto-compaction-settings.test.ts`: 17 passed

Full frontend suite on the development machine:

- 14,919 passed
- 4 unrelated failures: three legacy attachment charset cases and one
  `GMT`/`UTC` timezone formatting case.

`git diff --check` passes.

## Reviewer notes

The implementation is source-level and does not include generated bundles,
rebuilt desktop binaries, or local signing artifacts.
