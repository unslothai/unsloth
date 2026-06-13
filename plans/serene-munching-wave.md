# Shrink PR #5351 diff without changing functionality

## Context

PR #5351 (Studio chat document extraction) is +12,875/-661 over 56 files vs main. The bulk is genuinely new feature code, but ~1,200-1,500 lines of the diff are noise, dead code, duplication, and test copy-paste that can be removed with zero behavior change. Branch: `pr-5351-head` (= `etherll/document-extractor-refactor`), head `46787857c`, merge-base `27d43a31f` (current main). All findings below were verified against the live tree (zero-caller claims grep-confirmed; CRLF claim confirmed with `cat -A` and `--ignore-all-space`).

Guardrails: OCR presets, model swap/restore, cross-tab lock, NDJSON streaming, preview UI (stack, tabs, TOC, search, virtualization), retry, and the concurrency gate ALL stay. No API surface changes (keep GET+POST route variants). Do not merge the backend dataclass/pydantic pairs (intentional import isolation, pinned by tests). Excluded as functionality-changing: planner's suggestions to simplify the document stack/preview UI, drop the localStorage lock, or move MIME classification between layers.

## Execution: 3 parallel forks (file-disjoint), then synthesis

### Fork A — frontend (~430-520 lines)

**A1. tabs.tsx phantom rewrite (~165 lines of diff).** `origin/main:studio/frontend/src/components/ui/tabs.tsx` is CRLF; the PR converted to LF, so git shows +181/-134 when the real change is +63/-18. Restore main's exact bytes (CRLF), then re-apply ONLY the minimal functional change: the controlled-state guard in `Tabs` (`if (value === undefined) setInternal(v)` before `onValueChange?.(v)`). Revert the `TabsList` ResizeObserver/measured-pill rewrite to main's `layoutId` animation and revert the cosmetic `data-active:` → `data-[state=active]:` class flips. Verify `document-preview-panel.tsx` (the only new consumer, controlled usage) renders/animates with main's implementation. No `.gitattributes`/pre-commit rule covers `.tsx` line endings, so CRLF will stick.

**A2. Dead code removal (grep-confirmed zero callers, ~200 lines):**
- `stores/chat-runtime-store.ts`: `loadInferenceParams`, `saveInferenceParams`, `loadPresetSource`, `loadInt`, `saveInt`, `hasShownInferencePersistenceWarning` (~93)
- `components/.../attachment-chip-primitives.tsx`: `AttachmentChipIcon`, `AttachmentChipMeta`, `AttachmentChipStatusBadge`, the `tone` prop + `toneClass()` + `rootReady/rootVisual/rootWarning/rootDanger` token keys, `titleRow`/`detail` token keys (~65)
- `utils/extraction-queue.ts`: `getExtractionQueueDepth`, `getExtractionActiveCount` (~8)
- `utils/ocr-model-orchestrator.ts`: `resetOcrModelQueueForTests` (no frontend test framework exists) (~8)
- `utils/document-extraction.ts`: `firstDocumentImageDataUrl` (~9)
- `utils/ocr-model-presets.ts`: `hasSelectedOcrModel` (~3); `types.ts`: `DocumentCompleteAttachment` (~3); `types/api.ts`: `OpenAIChatContentPart` alias (~3)
- `utils/ocr-model-lock.ts`: `startedAt` field plumbing (lease checks use only `ownerId`/`expiresAt`) (~6)
- `components/document-stack.tsx`: constant `scale: 1` threaded through `getStackCardLayout` → animate props (~6)

**A3. Dedup/reuse (~110-150):**
- `formatTokens` triplicated in `attachment.tsx`, `doc-attachment-chip.tsx` (`formatDocumentTokens`), `document-preview-panel.tsx` → one export in `utils/document-extraction.ts` (~10)
- `resolveCurrentDocumentVisualPolicy` byte-identical in `runtime-provider.tsx:130` and `shared-composer.tsx:172` → move to `utils/document-extraction.ts` (~6)
- `ocr-model-orchestrator.ts`: replace private `mergeRecommendedInference` (line 290) + `toFiniteNumber` with existing `mergeBackendRecommendedInference` from `presets/preset-policy.ts:254` (~30-40); extract the Qwen reasoning-default block shared with `hooks/use-chat-model-runtime.ts:710` into one helper (~10)
- `ocr-model-orchestrator.ts`: consolidate `restoreUnloadedSnapshot`/`restoreSnapshotOrReconcile`/`retryRestoreSnapshot` shared scaffolding (divergence check → "Skipped/Could not restore" toast → `reconcileStoreFromStatus` → reload) into one `restoreOrReconcile(snapshot, identity, opts)` + `buildRestoreToast(snapshot)` (~60-90)
- `chat-settings-sheet.tsx`: dedupe `runProbe` vs the mount-probe `useEffect` (~12); fold the near-identical Default/None popover buttons into a 2-entry map (~20)

Do NOT touch (risk > reward): `normalizeSpeculativeType` (the orchestrator copy maps to different values than the canonical), full `applyStatusToStore` consolidation, the pass-through queue machinery (concurrency-sensitive), `buildDocSubtitle` vs `documentAttachmentSummary` (subtitle wording differs slightly).

### Fork B — backend src (~150-200 lines)

**B1. `routes/inference.py` extract endpoint:** the non-streaming and streaming branches duplicate the identical 7-entry exception→(status, detail) table (501/504/503/499/422/415-or-400/500). Add `_doc_exc_to_status_detail(exc) -> (int, str)` + `_ndjson_error(status, detail)`; both branches call it (~70-90). Add `_start_extraction(...)` for the twice-verbatim 13-kwarg `_extract_document` task call and a shared race/teardown helper (`asyncio.wait` FIRST_COMPLETED + cancel/shield + finally block) (~30-45). Add `_page_limit_detail(n)` for the 3x-repeated 413 message (~12-18).

**B2. trust_remote_code threading:** `bool(model_cfg.get("trust_remote_code", False) or inference_cfg.get(...))` is written 5x — `routes/inference.py:1246,1669`, `routes/models.py:185`, `utils/models/model_config.py:2297`. Add `requires_trust_remote_code(defaults: dict) -> bool` in `model_config.py`, import everywhere (~25-40).

**B3 (optional, do last):** `document_extractor.py` `_encoded_figure` helper for the two `ExtractedFigure(...)` encoded-image constructions (~10-15). Skip `_build_extract_options` inlining.

Match repo style: kwarg spacing `key = value` in calls (ruff hook `run_ruff_format.py` enforces it); run `pre-commit run --files <changed>` locally if available to avoid pre-commit.ci churn.

### Fork C — tests (~400-550 lines, coverage identical)

**C1. `studio/backend/tests/test_chat_document_routes.py` (1066):** `_make_app` (line 772) already patches the 5 standard seams but 10 tests rebuild the app inline — route them through `_make_app` (extend with `extra_patches`/`fake_extract` kwargs). Add `make_extract_result(**overrides)` factory for the 7x-verbatim `SimpleNamespace(...)` result. Parametrize the ~8 "raise X → expect status Y" mapping tests into one `@pytest.mark.parametrize` table (~220-300).

**C2. `studio/backend/tests/test_chat_document_extraction.py` (919):** `install_fake_extract(monkeypatch, returns=...)` helper for the 10x `DOCUMENT_EXTRACTION_AVAILABLE`+`_run_extract_sync` setattr pairs; `make_figures(n, *, encoded_until, captioned)` for the 5x figure list comprehension; `vlm_cap(source=...)` for the 9x `VlmCapability(...)` construction. Put shared factories in `studio/backend/tests/conftest.py` (exists) (~150-220).

**C3. `tests/studio/conftest.py` (new, ~8 lines):** absorb the 5x-repeated `sys.path` backend bootstrap from the five regression test files; keep `test_html_independent_of_inference.py`'s local `_BACKEND` used inside its subprocess source string (~20-30 net).

**C4.** `test_stream_cancel_registration_timing.py`: one `_find_chat_impl_fn(tree)` helper for the 3-4x AST-walk block (~15-25). `test_models_get_model_config_case_resolution.py`: shared YAML-TRC-vision defaults helper + parametrize the two hf-token-rejection tests (~25-40).

## Synthesis and verification

1. Forks work in worktrees; merge their diffs onto `pr-5351-head` (file-disjoint, no conflicts expected).
2. Validate: `npx tsc -b --force --pretty false` (studio/frontend); full backend suite `python3 -m pytest studio/backend/tests/test_chat_document_extraction.py test_chat_document_routes.py test_openai_tool_passthrough.py test_anthropic_messages.py test_vision_cache.py test_inference_worker.py tests/studio/ -q` (expect 1229 passed, 1 skipped — C-phase consolidation may change collected count via parametrize; assert no failures and that every original test scenario still exists); confirm tabs behavior by checking `document-preview-panel.tsx` against main's tabs API.
3. Confirm diffstat: `git diff origin/main...HEAD --shortstat` — target roughly +11.4K/-550 (from +12,875/-661).
4. Run `review_workflow` (reviewer.py + 3-fork consensus) over the refactor commit; fix findings.
5. Commit (user voice, no AI mentions/emojis/em dashes) and push to `etherll/document-extractor-refactor`. Re-confirm PR is MERGEABLE.

## Expected outcome

~1,200-1,500 fewer diff lines (≈10-12%): dead code and CRLF noise (~400), backend dedup (~180), frontend dedup (~250), test consolidation (~450). Anything beyond this requires dropping or splitting features (preview UI ~1.9K, OCR orchestration ~2.6K), which is out of scope per "without destroying functionality".
