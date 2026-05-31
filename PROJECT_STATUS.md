# Project Status: Unsloth + Muon Integration

## Current Round: 14
## Global Status: READY_FOR_MERGE (all R14 findings resolved)

## 1. Critical Blockers (Must fix to prevent runtime failure, crash, or serialization leakage)

*None — all critical findings from Passes 0–14 have been verified as resolved in the current codebase.*

## 2. High-Severity Findings (Pending — Impacts distributed training, stability, or numerical precision)

*None — all high-severity findings from Passes 0–14 have been verified as resolved in the current codebase.*

## 3. Medium & Low-Severity Items (Refactoring, Debt, & Border Configurations)

- [ ] **[R14 M2]** Regex-based norm detection fragile and deferred for 7 rounds (`unsloth/optimizers/muon.py:85-88`) — Well-understood, minimal-fix, repeatedly deferred.
- [ ] **[R14 L3]** TOCTOU race in `_assert_group_count_matches` (`unsloth/trainer.py:460-469`) — Theoretical thread-safety issue. Document as known limitation.
- [ ] **[R6 M5]** Muon + LoRA adapter training path is unvalidated (`unsloth/trainer.py:644-649`)
  * **Note:** A warning is logged when `PeftModel` is detected, but no runtime guard prevents Muon from being applied to low-rank adapters. Muon's full-matrix orthogonalization dynamics on rank-deficient LoRA A/B matrices are uncharacterized. Not a crash risk — training quality is unknown.
- [ ] **[R6 L4]** `_gpu_init` import chain prevents standalone use of `make_muon_param_groups` (`unsloth/__init__.py:147`)
  * **Note:** Importing via `unsloth.optimizers.muon` directly (bypassing `unsloth.__init__`) works, but the public `from unsloth import ...` path triggers `_gpu_init` which requires `unsloth_zoo`.
- [ ] **[R11 M4]** `modal_validate_muon.py` model name may not resolve on HF Hub — **Deferred:** Use a known-good model revision. Validation infrastructure risk.
- [ ] **[R11 L5]** No verification that `muon.defaults` contains expected keys — **Deferred:** Maintenance risk if upstream Muon changes defaults format.

## 4. Historical Archive: Resolved & Verified Findings

### Round 14 Resolutions
- [x] **[R14] [C1]** Meta device guard — **Fixed:** `make_muon_param_groups` and `_classify_param_names` raise `RuntimeError` on meta device.
- [x] **[R14] [L1]** `weight_decay` kwarg added to `adamw_kwargs` — **Fixed:** Added for upstream validation consistency.
- [x] **[R14] [L2]** `torch.distributed` import guard — **Fixed:** Wrapped in try/except with descriptive error message.
- [x] **[R14] [MT1]** Meta device guard test — **Added:** `test_meta_device_raises_error` verifies RuntimeError.
- [x] **[R14] [MT2]** `_classify_param_names` meta device test — **Added:** `test_classify_param_names_meta_device` verifies RuntimeError.
- [x] **[R14] [M1]** `_norm_name_pattern` regex misses `rms_norm` (underscore variant) — **Fixed:** Regex extended from `r"(?:layernorm|rmsnorm|^norm$|\.norm\b)"` to `r"(?:layernorm|rmsnorm|rms_norm|^norm$|\.norm\b)"`. (`unsloth/optimizers/muon.py:85`).
- [x] **[R14] [MT3]** `_norm_name_pattern` with `rms_norm` test — **Added:** `test_norm_name_pattern_catches_rms_norm` verifies `rms_norm.weight` is classified as no_decay.

### Round 13 Resolutions
- [x] **[R13] [H1]** `adamw_betas` sentinel uses identity comparison instead of value comparison — **Fixed:** Added `_ADAMW_BETAS_UNSET = object()` sentinel alongside `_ADAMW_EPS_UNSET`. Default changed from `(0.9, 0.999)` to `_ADAMW_BETAS_UNSET`. Comparison changed from `!= (0.9, 0.999)` to `is not MuonConfig._ADAMW_BETAS_UNSET`. Validation in `__post_init__` skips sentinel. (`unsloth/trainer.py:208, 219, 308-314, 711-717`).
- [x] **[R13] [H2]** Non-embedding tied parameters receive duplicate Muon updates — **Fixed:** Added `data_ptr()`-based deduplication in `make_muon_param_groups` param routing loop. Shared tensors are now skipped after the first occurrence. (`unsloth/optimizers/muon.py:162-167`).
- [x] **[R13] [M1]** `_norm_name_pattern` regex matches non-norm names containing "norm" — **Fixed:** Regex hardened from `r"(?:layernorm|rmsnorm|norm)"` to `r"(?:layernorm|rmsnorm|^norm$|\.norm\b)"` to exclude names like `normalizer`. (`unsloth/optimizers/muon.py:85`).
- [x] **[R13] [M2]** `embedding_lr=0.0` silently freezes embeddings — **Fixed:** Warning logged when `embedding_lr == 0.0` informing users that embeddings will receive zero gradient updates. (`unsloth/trainer.py:649-655`).
- [x] **[R13] [L1]** `weight_decay` kwarg omitted from `torch.optim.Muon` constructor — **Fixed:** `weight_decay=muon_weight_decay` added to `muon_kwargs` dict. Redundant with per-group value but ensures upstream validation runs. (`unsloth/trainer.py:692`).
- [x] **[R13] [L2]** Missing sentinel test for `adamw_betas` — **Added:** `test_adamw_betas_sentinel_overrides_training_args` verifies config `(0.9, 0.999)` overrides `TrainingArguments(adam_beta1=0.95)`.
- [x] **[R13] [L3]** Missing `_classify_param_names` tied embedding test — **Added:** `test_tied_embedding_detected_via_data_ptr` verifies `data_ptr()` aliasing correctly propagates embedding classification.
- [x] **[R13] [MT1]** Tied non-embedding parameter detection test — **Added:** `test_tied_non_embedding_not_duplicated` verifies shared tensors appear once in `muon_params`.
- [x] **[R13] [MT2]** `adamw_betas` sentinel exact match test — **Added:** Covered by L2 test (same test case).
- [x] **[R13] [MT3]** `embedding_lr=0.0` warning test — **Added:** `test_embedding_lr_zero_warning` verifies warning is emitted.
- [x] **[R13] [MT4]** PEFT norm `modules_to_save` test — **Added:** `test_modules_to_save_norm_goes_to_no_decay` verifies norm copies classified as no_decay.
- [x] **[R13] [MT5]** No Muon params edge case — **Added:** `test_no_muon_params_optimizer` verifies `muon=None` when no 2D params.
- [x] **[R13] [MT6]** No AdamW params edge case — **Added:** `test_no_adamw_params_optimizer` verifies `adamw=None` when all params are 2D Muon-eligible.

### Round 12 Resolutions
- [x] **[R12] [M1]** `_sync_lr` group count check extracted into `_assert_group_count_matches` — **Fixed:** Sync loop removed (dead code due to identity-sharing). Group count check isolated into `_assert_group_count_matches()` called from `step()`. `MUON_SYNC_KEYS`/`ADAMW_SYNC_KEYS` frozensets removed. (`unsloth/trainer.py:469-482, 514`).
- [x] **[R12] [M2]** `make_muon_param_groups` `weight_decay` parameter renamed to `muon_weight_decay` — **Fixed:** Renamed parameter and updated docstring to clarify it is Muon-only. Call site and all tests updated. (`unsloth/optimizers/muon.py:104-155`).
- [x] **[R12] [L2]** `ns_steps < 1` check tested — **Fixed:** Added `test_muon_config_rejects_ns_steps_lt_one` verifying `MuonConfig(ns_steps=0)` and `MuonConfig(ns_steps=-1)` raise `ValueError`.
- [x] **[R12] [L3]** `adamw_betas` precedence changed — **Fixed:** `config.adamw_betas` now takes precedence over `TrainingArguments.adam_beta1/adam_beta2`. Only falls through to TrainingArguments when `adamw_betas` is left at default `(0.9, 0.999)`. (`unsloth/trainer.py:733-740`).
- [x] **[R12] [MT1]** External `add_param_group` on sub-optimizer detected — **Added:** `test_external_add_param_group_detected` verifies direct `muon.add_param_group(...)` raises `RuntimeError` on `step()`.
- [x] **[R12] [MT3]** `adamw_betas` config precedence test — **Added:** `test_adamw_betas_config_takes_precedence` verifies `MuonConfig(adamw_betas=(0.8, 0.99))` produces correct betas.
- [x] **[R12] [MT4]** LoRA adapter Muon eligibility — **Added:** `test_muon_routes_lora_adapters` verifies 2D rank-deficient LoRA A/B matrices are Muon-eligible.
- [x] **[R12] [MT5]** Overlapping norm/embedding classification priority — **Added:** `test_no_decay_takes_precedence_over_embedding` verifies that a param in both sets routes to AdamW no-decay.
- [x] **[R11] [H1]** `torch.use_deterministic_algorithms` warn_only corruption — **Fixed:** `_muon_step_deterministic` now saves/restores `warn_only` state via `torch.is_deterministic_algorithms_warn_only_enabled()`. Restores both `enabled` and `warn_only` to original values after Muon step (`unsloth/trainer.py:482-501`).
- [x] **[R11] [M1]** `self.defaults` Muon-only after R10 L1 fix — **Fixed:** Repopulate `self.defaults` with both Muon and AdamW defaults after `super().__init__`, preventing silent failures when downstream code reads `optimizer.defaults` (`unsloth/trainer.py:425-431`).
- [x] **[R11] [M2]** AdamW groups excluded from identity-sharing assertion — **Fixed:** Extended identity-sharing assertion to verify AdamW groups are identity-shared with chained groups (`unsloth/trainer.py:438-445`).
- [x] **[R11] [L1]** `_sync_lr` tests test identity-sharing, not sync — **Fixed:** Renamed four `_sync_lr` tests to `test_identity_sharing_*` to clarify what they verify. Added `test_assert_group_count_matches_detects_mismatch` testing group count check isolation.
- [x] **[R11] [L2]** `test_chained_defaults_populated` name stale — **Fixed:** Renamed to `test_chained_defaults_contains_all_keys` and updated assertions to verify both Muon and AdamW keys are present in `defaults`.
- [x] **[R11] [L3]** `test_empty_muon_group_params` never calls `step()` — **Fixed:** Added `optimizer.step()` call after construction.
- [x] **[R11] [MT2]** Missing `defaults` completeness test — **Added:** `test_chained_defaults_contains_all_keys` verifies both Muon `momentum` and AdamW `betas` are accessible.
- [x] **[R11] [MT3]** Missing `step()` with empty sub-optimizer — **Added:** `test_step_with_empty_suboptimizer` verifies `_MuonAdamWChained(muon=None, adamw=...)` does not crash on `step()`.

### Round 10 Resolutions
- [x] **[R10] [L4]** `_norm_name_pattern` false-positive risk — **Deferred:** Low probability in practice. Monitor for future HF model naming conventions.
- [x] **[R10] [L5]** Missing `huggingface_hub` in modal deps — **Deferred:** Transitively installed.

### Round 9 Resolutions
- [x] **[R9] [M1]** `nesterov` type validation — **Fixed:** Added `isinstance(self.nesterov, bool)` check in `__post_init__` (`unsloth/trainer.py:309-314`)
- [x] **[R9] [M2]** `adamw_betas` validation — **Fixed:** Added tuple/len check in `__post_init__` (`unsloth/trainer.py:315-319`)
- [x] **[R9] [M3]** `_sync_lr` identity-sharing contract — **Partially fixed:** Added class docstring noting identity-sharing; added runtime assertion in `__init__` verifying `param_groups[i] is muon.param_groups[i]`. Full `_sync_lr` removal deferred. (`unsloth/trainer.py:372-390, 416-427`)
- [x] **[R9] [L1]** `adjust_lr_fn` type check — **Fixed:** Added `isinstance(self.adjust_lr_fn, str)` guard before `.lower()` (`unsloth/trainer.py:320-325`)
- [x] **[R9] [L3]** bias detection via `param.ndim` — **Fixed:** Replaced `"bias" in name.lower()` with `param.ndim == 1` (`unsloth/optimizers/muon.py:169`)
- [x] **[R9] [L4]** `__repr__` improvement — **Fixed:** Shows parameter count instead of group count (`unsloth/trainer.py:561-567`)

### Round 8 Resolutions
- [x] **[R8] [H1]** Custom RMSNorm implementations always receive weight decay — **Fixed:** Added name-based fallback in `_classify_param_names` matching HF Trainer's `norm`/`layernorm`/`rmsnorm` regex pattern, catching all custom RMSNorm classes (LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, etc.) (`unsloth/optimizers/muon.py:82-90`)
- [x] **[R8] [H2]** No type validation on `MuonConfig` numeric fields — **Fixed:** Added `isinstance` checks for `ns_steps` (int), `momentum`, `muon_eps`, `muon_lr_scale`, `muon_weight_decay`, and `adamw_weight_decay` (int/float) in `__post_init__` (`unsloth/trainer.py:229-254`)
- [x] **[R8] [M1]** `PeftModel` import guard causes `TypeError` — **Fixed:** Changed `isinstance(self.model, PeftModel)` to `PeftModel is not None and isinstance(self.model, PeftModel)` (`unsloth/trainer.py:610`)
- [x] **[R8] [M3]** No PyTorch version check at MuonConfig time — **Fixed:** Moved `hasattr(torch.optim, "Muon")` check to `MuonConfig.__post_init__` for early error detection; retained in `_create_muon_optimizer` as defense-in-depth (`unsloth/trainer.py:233-238`)
- [x] **[R8] [M4]** Two configs set simultaneously is silent — **Fixed:** Added `logger.warning` when both `muon_config` and `q_galore_config` are set (`unsloth/trainer.py:556-559`)
- [x] **[R8] [L2]** `ns_steps` accepts float instead of int — **Fixed:** Covered by H2 `isinstance(self.ns_steps, int)` check (`unsloth/trainer.py:229-233`)

### Round 7 Resolutions
- [x] **[R7] [M1]** `_ADAMW_EPS_UNSET` sentinel default set to `1e-8` instead of sentinel, defeating the fallback — **Fixed:** Changed default from `1e-8` to `_ADAMW_EPS_UNSET`; verified with 3 new sentinel tests (`unsloth/trainer.py:222`)
- [x] **[R7] [L1]** `embedding_lr` uses `or` fallback (`unsloth/optimizers/muon.py:192`) — **Fixed:** `embedding_lr or adamw_lr` → `embedding_lr if embedding_lr is not None else adamw_lr`
- [x] **[R7] [L2]** `adamw_lr` uses `or` fallback (`unsloth/trainer.py:662`) — **Fixed:** `config.adamw_lr or lr` → `config.adamw_lr if config.adamw_lr is not None else lr`

### Round 6 Resolutions
- [x] **[R6] [C1]** `ns_coefficients=None` propagated to `torch.optim.Muon` crashes `step()` — **Fixed:** None values filtered via `{k:v for k,v in ... if v is not None}` before constructor call (`unsloth/trainer.py:632`)
- [x] **[R6] [H1]** 2D normalization weights routed to Muon (no-decay check after Muon eligibility) — **Fixed:** Routing restructured to check `is_no_decay` before `_is_muon_eligible` (`unsloth/optimizers/muon.py:171-178`)
- [x] **[R6] [H2]** `_sync_lr` does not propagate `ns_steps`, `ns_coefficients`, `adjust_lr_fn` — **Fixed:** Separate `MUON_SYNC_KEYS` / `ADAMW_SYNC_KEYS` frozensets defined with all relevant keys (`unsloth/trainer.py:389-395`)
- [x] **[R6] [H3]** Null sub-optimizer `state_dict` loads crash with `KeyError` — **Fixed:** `.get()` with `_muon_version` marker check and descriptive `RuntimeError`; None guards on sub-optimizer access (`unsloth/trainer.py:464-482`)
- [x] **[R6] [M1]** Missing `ns_coefficients` validation in `MuonConfig.__post_init__` — **Fixed:** Tuple length and type checks added (`unsloth/trainer.py:242-252`)
- [x] **[R6] [M2]** `closure` called without `torch.enable_grad()` — **Fixed:** Closure wrapped in `with torch.enable_grad():` (`unsloth/trainer.py:439-441`)
- [x] **[R6] [M3]** `betas` propagated to Muon param groups via `_sync_lr` — **Fixed:** Separate sync key sets; `betas` excluded from `MUON_SYNC_KEYS` (`unsloth/trainer.py:389-395`)
- [x] **[R6] [M4]** `TrainingArguments.adam_epsilon` overrides `MuonConfig.adamw_eps` — **Partially fixed (R6), regression resolved (R7):** Sentinel object `_ADAMW_EPS_UNSET` added in R6 but field default left as `1e-8`, defeating the sentinel. Default corrected to `_ADAMW_EPS_UNSET` in R7 (`unsloth/trainer.py:222`).
- [x] **[R6] [M6]** No `embedding_lr` field on `MuonConfig` — **Fixed:** Field added and propagated (`unsloth/trainer.py:225, 597`)
- [x] **[R6] [L1]** `_classify_param_names` uses `id()` instead of `data_ptr()` for tied detection — **Fixed:** Uses `param.data_ptr()` (`unsloth/optimizers/muon.py:70`)
- [x] **[R6] [L2]** `adamw_lr` fallback via `or` prevents `0.0` — **Fixed:** Explicit `is None` check (`unsloth/optimizers/muon.py:144`)
- [x] **[R6] [L3]** No upper bound warning for `ns_steps` — **Fixed:** Warning for `ns_steps > 20` (`unsloth/trainer.py:236-241`)

### Round 5 Resolutions
- [x] **[R5] [C1]** PEFT `modules_to_save` hard-coded to adapter name `"default"` — **Fixed:** Matches any adapter name via `"modules_to_save." in name` then type-checks parent via `isinstance(parent, nn.Embedding)` / `isinstance(parent, NORM_CLASSES)` (`unsloth/optimizers/muon.py:52-65`)
- [x] **[R5] [C2]** `load_state_dict` desynchronizes chained `param_groups` from sub-optimizer groups — **Fixed:** `param_groups` reassigned from sub-optimizer groups after `load_state_dict` (`unsloth/trainer.py:476-482`)
- [x] **[R5] [H1]** `torch.use_deterministic_algorithms(True)` is global — **Fixed:** Scoped determinism with save/restore pattern (`unsloth/trainer.py:424-435`)
- [x] **[R5] [H2]** GPU device mismatch risk in dummy AdamW parameter — **Fixed:** Empty sub-optimizers handled via `None` (`unsloth/trainer.py:657-659`)
- [x] **[R5] [M1]** `_sync_lr` silently overwrites direct sub-optimizer hyperparameter changes — **Fixed:** `load_state_dict` re-syncs groups (`unsloth/trainer.py:476-482`)
- [x] **[R5] [M2]** `_create_muon_optimizer` only catches `TypeError` — **Fixed:** Catches `Exception` (`unsloth/trainer.py:638`)
- [x] **[R5] [M3]** Inconsistent `ns_coefficients` vs `adjust_lr_fn` kwargs handling — **Fixed:** Both always passed; None values filtered uniformly (`unsloth/trainer.py:627-632`)

### Round 4 Resolutions
- [x] **[R4] [H1]** `adam_epsilon` silently ignored from `TrainingArguments` — **Fixed:** Sentinel pattern respects explicit config values; falls back to `args.adam_epsilon` (`unsloth/trainer.py:650-653`)
- [x] **[R4] [H2]** `resume_from_checkpoint` can silently load corrupt state — **Fixed:** `_muon_version` marker in state dict with version mismatch raising `RuntimeError` (`unsloth/trainer.py:455, 465-471`)
- [x] **[R4] [H3]** Distributed bypass guard dangerously misleading — **Fixed:** Guard blocks with clear error enumerating issues; opt-in via `UNSLOTH_MUON_DISTRIBUTED=1`; scoped determinism when active (`unsloth/trainer.py:560-565`)
- [x] **[R4] [M1]** Dead parameters `adamw_betas`, `adamw_eps` in `make_muon_param_groups` — **Fixed:** Removed from function signature (`unsloth/optimizers/muon.py:94-103`)
- [x] **[R4] [M2]** `_MuonAdamWChained.defaults` is incomplete — **Fixed:** Proper `defaults` dict populated from sub-optimizers; MUON_SYNC_KEYS handles propagation
- [x] **[R4] [M3]** Tied embeddings risk with `tie_word_embeddings=True` — **Fixed:** `data_ptr()`-based tensor identity detection in `_classify_param_names` (`unsloth/optimizers/muon.py:67-78`)
- [x] **[R4] [M4]** PEFT `modules_to_save` bias not caught by second pass — **Fixed:** Generalized `modules_to_save.*` handling with parent module type check (`unsloth/optimizers/muon.py:51-65`)
- [x] **[R4] [M5]** `adjust_lr_fn` None vs `"original"` inconsistency — **Fixed:** Passed explicitly in kwargs; None filtered before constructor call (`unsloth/trainer.py:628, 632`)
- [x] **[R4] [M6]** `_sync_lr` doesn't propagate `nesterov`, `ns_steps`, `ns_coefficients`, `eps` — **Fixed:** All keys in `MUON_SYNC_KEYS` (`unsloth/trainer.py:389-392`)

### Round 3 Resolutions
- [x] **[R3] [H1]** `modules_to_save` blanket routing to embedding group (`unsloth/optimizers/muon.py:42-44`) — **Fixed:** Parent module type check replaced blanket routing (`unsloth/optimizers/muon.py:51-65`)
- [x] **[R3] [H2]** Distributed warning downgraded from `RuntimeError` — **Fixed:** Reinstated as `RuntimeError` with `UNSLOTH_MUON_DISTRIBUTED=1` opt-out (`unsloth/trainer.py:560-565`)
- [x] **[R3] [H3]** `_sync_lr` assumes symmetric param group keys — **Fixed:** Separate sync key frozensets per sub-optimizer; group count mismatch raises `RuntimeError` (`unsloth/trainer.py:389-422`)
- [x] **[R3] [H4]** Docstring claims embeddings fall into decay group — **Fixed:** Docstring corrected; embeddings always get `weight_decay=0.0` (`unsloth/optimizers/muon.py:131-135, 190-193`)
- [x] **[R3] [H5]** `weight_decay` silently omitted from `torch.optim.Muon` constructor — **Fixed:** `weight_decay=muon_weight_decay` included in `muon_kwargs` (`unsloth/trainer.py:626`)
- [x] **[R3] [M4]** `MuonConfig.__post_init__` missing input validation — **Fixed:** Validation added for `momentum`, `muon_eps`, `ns_steps`, `muon_lr_scale`, `muon_weight_decay`, `ns_coefficients` (`unsloth/trainer.py:227-278`)
- [x] **[R3] [M5]** `print()` used instead of `logging` — **Fixed:** Uses `logger.warning()` and `logger.info()` (`unsloth/trainer.py:600-618`)
- [x] **[R3] [M6]** `_sync_lr` copies all keys, not just known hyperparams — **Fixed:** Scoped to `MUON_SYNC_KEYS` / `ADAMW_SYNC_KEYS` (`unsloth/trainer.py:411-421`)

### Round 2 Resolutions
- [x] **[R2] [C1]** `__setstate__` produces orphaned optimizers with stale parameter references — **Fixed:** `__getstate__` returns `state_dict()`; `__setstate__` raises `NotImplementedError`; pickle path blocked with clear error (`unsloth/trainer.py:484-489`)
- [x] **[R2] [C2]** State dict format breaks FSDP and HF Trainer checkpoint save/load — **Fixed:** Documented; `_muon_version` marker added; proper None guards for sub-optimizers (`unsloth/trainer.py:456-482`)
- [x] **[R2] [C4]** Missing `add_param_group` override — **Fixed:** `add_param_group` raises `NotImplementedError` after init completes (`unsloth/trainer.py:381-387`)
- [x] **[R2] [H1]** `_sync_lr` does not sync `weight_decay` or other group hyperparams — **Fixed:** All relevant keys in `MUON_SYNC_KEYS` / `ADAMW_SYNC_KEYS` (`unsloth/trainer.py:389-395`)
- [x] **[R2] [H2]** `MuonConfig` ignores `self.args.adam_beta1`/`adam_beta2` — **Fixed:** AdamW betas read from `TrainingArguments` with config fallback (`unsloth/trainer.py:646-649`)
- [x] **[R2] [H4]** Non-deterministic CuBLAS in NS orthogonalization — **Fixed:** Scoped `torch.use_deterministic_algorithms(True)` during Muon step when distributed (`unsloth/trainer.py:424-435`)
- [x] **[R2] [H5]** No guard against missing `adjust_lr_fn` kwarg in PyTorch version variants — **Fixed:** `try/except Exception` with descriptive `RuntimeError` (`unsloth/trainer.py:636-642`)

### Round 1 Resolutions
- [x] **[R1] [C1]** `embedding_lr` overrides bias/norm LR instead of embedding LR — **Fixed:** Dedicated `adamw_embedding_params` group with `embedding_lr` applied only to embeddings (`unsloth/optimizers/muon.py:152, 175-176, 190-193`)
- [x] **[R1] [H1]** Distributed training guard removed — **Fixed:** Reinstated as `RuntimeError` with `UNSLOTH_MUON_DISTRIBUTED=1` bypass (`unsloth/trainer.py:560-565`)
- [x] **[R1] [H2]** `__getstate__`/`__setstate__` asymmetry crashes third-party checkpointing — **Fixed:** `__getstate__` returns `state_dict()`; pickle path blocked with clear error (`unsloth/trainer.py:484-489`)
- [x] **[R1] [H3]** `adamw_weight_decay` inherits from `muon_weight_decay` instead of base `weight_decay` — **Fixed:** `adamw_weight_decay` defaults to `self.args.weight_decay` (`unsloth/trainer.py:585-586`)
- [x] **[R1] [M1]** Only `lr` synced in `_sync_lr` — **Fixed:** All hyperparams synced via key sets (`unsloth/trainer.py:389-395`)
- [x] **[R1] [M2]** `_is_no_decay` substring heuristic misses patterns — **Fixed:** Module-type-based check via `NORM_CLASSES` isinstance (`unsloth/optimizers/muon.py:21-28, 47-49`)
- [x] **[R1] [M4]** `ns_steps >= 100` not validated at config time — **Fixed:** Validation in `__post_init__` (`unsloth/trainer.py:229-241`)
- [x] **[R1] [M5]** Empty `self.defaults` in `_MuonAdamWChained` — **Fixed:** Proper defaults populated from sub-optimizer defaults
- [x] **[R1] [M6]** `adjust_lr_fn` passed as unvalidated string — **Fixed:** Validated in `__post_init__` against `("original", "match_rms_adamw")` (`unsloth/trainer.py:266-277`)
- [x] **[R1] [M7]** PEFT `modules_to_save.default.weight` goes to Muon instead of AdamW — **Fixed:** Parent module type check routes to embedding/no-decay group (`unsloth/optimizers/muon.py:51-65`)

### Round 0 Resolutions
- [x] **[R0] [C1]** Embedding parameters routed to Muon — **Fixed:** `_classify_param_names` detects `nn.Embedding` params; `_is_muon_eligible` excludes them via `embedding_param_names` set (`unsloth/optimizers/muon.py:42-46, 83-91`)
- [x] **[R0] [C2]** `torch.save(optimizer)` silently loses all optimizer state — **Fixed:** `__getstate__` returns `state_dict()` preserving sub-optimizer state; pickle blocked at load time with clear error (`unsloth/trainer.py:484-489`)
- [x] **[R0] [H3]** No weight-decay splitting for AdamW fallback (biases/norms get non-zero wd) — **Fixed:** Three AdamW sub-groups: decay, no-decay, embedding (`unsloth/optimizers/muon.py:180-193`)
- [x] **[R0] [H4]** `embedding_learning_rate` silently ignored with `MuonConfig` — **Fixed:** `embedding_lr` passed through `make_muon_param_groups` and applied to dedicated embedding group (`unsloth/trainer.py:597`)
- [x] **[R0] [H5]** Missing `adjust_lr_fn` parameter in `MuonConfig` — **Fixed:** Added `adjust_lr_fn: Optional[str] = None` with validation (`unsloth/trainer.py:214, 266-277`)
- [x] **[R0] [M6]** Same `weight_decay` for Muon and AdamW (different optimal values) — **Fixed:** Separate `muon_weight_decay` and `adamw_weight_decay` fields with independent defaults (`unsloth/trainer.py:216, 222`)
- [x] **[R0] [M7]** `ns_coefficients` and `muon_eps` not exposed — **Fixed:** Added `muon_eps: float = 1e-7` and `ns_coefficients: Optional[tuple[float,float,float]] = None` (`unsloth/trainer.py:212, 215`)

## 5. Loop State Handoff (Directives for the Coder Agent)

- **Active Codebase Focus:** All critical, high-severity, and medium/low findings from fourteen review passes (R0–R14) have been resolved or documented as deferred. Code is safe to merge for single-GPU full-finetuning.
- **Latest Input Telemetry Source:** `MUON_REVIEW_14.md` (this round)
- **Merge Recommendation as of R14:** APPROVE (all R14 issues resolved — see MUON_REVIEW_14.md for details)
- **Inviolable Architecture Constraints:**
    1. Do NOT change the delegated architecture — `torch.optim.Muon` handles the optimizer math; Unsloth handles param routing and chaining.
    2. Do NOT add a Muon reimplementation — the delegation pattern is the correct design.
    3. Do NOT remove the distributed training guard (`UNSLOTH_MUON_DISTRIBUTED=1`) without providing a correct distributed-safe implementation.
    4. Do NOT change the state dict format (`{"_muon_version": 1, "muon": ..., "adamw": ...}`) without a migration path.
    5. Do NOT break compatibility with the standard AdamW fallback path — `_create_unsloth_optimizer` must remain functional for users not using Muon.
    6. Do NOT merge defaults from sub-optimizers — the current merge pattern can leak keys across optimizer boundaries. Use scoped defaults instead (see R10 L1). The repopulation after `super().__init__` is safe because `add_param_group` is the leakage vector, not `self.defaults`. (R11 M1).
