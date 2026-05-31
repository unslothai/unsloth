# Project Status: Unsloth + Muon Integration

## Current Round: 9
## Global Status: READY_FOR_MERGE (with notes)

## 1. Critical Blockers (Must fix to prevent runtime failure, crash, or serialization leakage)

*None — all critical findings from Passes 0–6 have been verified as resolved in the current codebase.*

## 2. High-Severity Findings (Pending — Impacts distributed training, stability, or numerical precision)

*None — all high-severity findings from Passes 0–6 have been verified as resolved in the current codebase.*

## 3. Medium & Low-Severity Items (Refactoring, Debt, & Border Configurations)

- [ ] **[R6 M5]** Muon + LoRA adapter training path is unvalidated (`unsloth/trainer.py:644-649`)
  * **Note:** A warning is logged when `PeftModel` is detected, but no runtime guard prevents Muon from being applied to low-rank adapters. Muon's full-matrix orthogonalization dynamics on rank-deficient LoRA A/B matrices are uncharacterized. Not a crash risk — training quality is unknown.
- [ ] **[R6 L4]** `_gpu_init` import chain prevents standalone use of `make_muon_param_groups` (`unsloth/__init__.py:147`)
  * **Note:** Importing via `unsloth.optimizers.muon` directly (bypassing `unsloth.__init__`) works, but the public `from unsloth import ...` path triggers `_gpu_init` which requires `unsloth_zoo`.
- [ ] **[R9 M3]** `_sync_lr` identity-sharing contract undocumented (`unsloth/trainer.py:434-459`)
  * **Note:** `_sync_lr` is a no-op due to identity sharing. A future refactor that deep-copies groups silently breaks LR propagation. Deferred cleanup (_sync_lr removal deferred; assertion and docstring added in R9).

## 4. Historical Archive: Resolved & Verified Findings

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

- **Active Codebase Focus:** All critical, high-severity, and medium/low findings from nine review passes (R0–R9) have been resolved (with the exception of deferred cleanup for `_sync_lr` removal and `_gpu_init` reorder, and open benchmarking for LoRA+Muon quality). Code is safe to merge for single-GPU full-finetuning.
- **Latest Input Telemetry Source:** `MUON_REVIEW_9.md` (this round)
- **Merge Recommendation as of R9:** APPROVE (with notes — see MUON_REVIEW_9.md for details)
- **Inviolable Architecture Constraints:**
  1. Do NOT change the delegated architecture — `torch.optim.Muon` handles the optimizer math; Unsloth handles param routing and chaining.
  2. Do NOT add a Muon reimplementation — the delegation pattern is the correct design.
  3. Do NOT remove the distributed training guard (`UNSLOTH_MUON_DISTRIBUTED=1`) without providing a correct distributed-safe implementation.
  4. Do NOT change the state dict format (`{"_muon_version": 1, "muon": ..., "adamw": ...}`) without a migration path.
  5. Do NOT break compatibility with the standard AdamW fallback path — `_create_unsloth_optimizer` must remain functional for users not using Muon.
