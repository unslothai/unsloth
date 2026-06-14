# PR #5351 diff-shrink analysis (backend + tests scope)

Read-only analysis. No functional change intended; preserve test coverage.
Findings and estimated diff-line savings are in the final assistant message.

## Ranked savings (est. PR diff-line reduction)
1. routes/inference.py: collapse duplicated streaming vs non-streaming doc
   exception->(status,detail) mapping + extract/disconnect orchestration into
   shared helpers (`_DOC_EXC_MAP`, `_doc_exc_status_detail`,
   `_run_extraction_with_disconnect`, `_PAGE_LIMIT_DETAIL`). ~120-160 lines.
2. tests/test_chat_document_routes.py: route every test through existing
   `_make_app`; add `make_extract_result` + ndjson reader to conftest;
   parametrize status-code mapping tests. ~220-300 lines.
3. tests/test_chat_document_extraction.py: shared `patch_extract` fixture +
   `make_figures` factory; collapse fake_extract closures. ~150-220 lines.
4. tests/studio/conftest.py: absorb the 4-line `_BACKEND` sys.path bootstrap
   from 5 files; ~25-35 lines.
5. trust_remote_code: shared `_requires_trust_remote_code(defaults)` helper in
   model_config; reused by load_model/validate_model/models.py/from_identifier.
   ~25-40 lines.
6. document_extractor.py: figure-builder + warnings helpers; modest. ~30-60.
7. test_stream_cancel_registration_timing.py: `_find_impl_fn` helper for the
   repeated dual-name AST walk. ~15-25 lines.

Total realistic: ~600-850 lines off a +12875 diff for this scope.
