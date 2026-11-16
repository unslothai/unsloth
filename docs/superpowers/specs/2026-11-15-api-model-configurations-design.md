# API-selectable GGUF configurations (#17820)

Expose several owner-defined API aliases for one downloaded GGUF with distinct launch settings. A harness chooses the alias through the ordinary `model` field and discovers it through `/v1/models`; weights are downloaded once. The user requested autonomous design, planning, implementation and GOAT review at each task and on the final branch. Design choices below are the implementer's calls under that authorization.

## Approaches

1. Extend existing aliases with `load_config` (selected): reuses discovery, owner-only settings, model authorization and loading. Small public extension; lifecycle comparisons must retain configuration identity.
2. Introduce separate profiles/endpoints: duplicates persistence and discovery and requires more UI/API surface.
3. Bind browser presets to API identities: couples account-specific sampling/prompt presets to install-wide launch settings and overlaps ongoing preset work.

## Contract

`PUT /api/settings/model-aliases` accepts the existing `aliases` list. Each entry may include `load_config`, an object with all three required fields: `max_seq_length` (strict integer 1..1048576), `cache_type_kv` (one of the existing supported KV cache types), and `speculative_type` (`auto`, `off`, or `ngram`). Unknown keys, null fields, booleans as integers and malformed configurations are rejected. Absent/null load_config retains legacy alias behavior. Empty objects are invalid. Existing alias validation and catalog filtering remain intact. No database migration or new dependency.

Configurations apply to GGUF text generation only, with Keep multiple models loaded Off. Unsupported targets/modes must fail clearly before any model teardown. These deliberate first-version limits avoid silently serving a kept slot with the wrong configuration. No automatic downloads are initiated for configured aliases whose target cannot resolve locally. Existing auto-switch, loaded-only, keyless, access and lifecycle policies are not widened. With auto-switch off, an already matching runtime may serve; a mismatch fails instead of silently using the wrong settings.

Only middleware resolving a stored alias can attach configuration metadata, in the ASGI scope bound to the rewritten target. JSON extension fields cannot manufacture it. Configuration is request-local, never persisted into the model's ordinary settings. Request rewriting still runs before access checks and does not disclose extra model paths. Alias records continue to be listed only when their target is visible.

A matching resident identity AND matching requested context/cache/speculative intent can be reused. A different configuration must reach the existing locked load path. Recheck after acquiring lifecycle gates. Compare original requested intent, since Auto may choose a different effective speculative algorithm. Merge configured fields over ordinary saved settings before converting them to load arguments, including stripping conflicting extra flags. Failed loads never mark a configuration successful. Ordinary aliases/bare models preserve existing behavior (including adopting the currently loaded configuration).

## UI and documentation

Configuration is initially set through the owner settings API. Existing alias UI must round-trip load_config when renaming/deleting other entries so it cannot erase API-created settings. No new UI controls or preset redesign. Document exact setup, two examples, auto-switch requirement, reload latency, hardware limits and sequential/multi-model restriction.

## Verification

Use real validation/middleware and existing mocked-load integration tests without downloading large models. Cover persistence/schema round trips, malformed inputs, middleware provenance and nested JSON, catalog visibility, A→B→B→A switching, failed loads, flag conflicts, disabled switching, missing/non-GGUF targets, multi-model and loaded-only restrictions, and ordinary alias compatibility. Run relevant existing backend suites, default offline pytest, frontend parser tests/typecheck where feasible, and formatting checks. Hardware generation is desirable but must be reported separately if unavailable.

## Anti-goals and review risks

Do not silently accept a profile and answer with another runtime. Do not mutate global model overrides per request, bypass access checks, change sampling defaults, introduce concurrent model copies, or claim hardware validation from mocks. GOAT review is read-only; fixes happen afterward under the implementation authorization. Each task review uses its pinned commit range; the final review uses the complete branch.
