# Cherry-Pick Policy

This document defines the process for backporting changes from upstream dependencies (e.g., `llama.cpp`, `transformers`, `trl`, `peft`) into the Unsloth repository. It is adapted from the panthro.cpp backport policy and is intended to keep upstream syncs safe, reviewable, and measurable.

## When to use this policy

Use this policy when you are cherry-picking commits from upstream into Unsloth, including:

- Syncing the bundled `llama.cpp` / `llama-server` source or prebuilt binaries.
- Backporting a bug fix from a dependency that has not landed in a released Unsloth package.
- Pulling a performance or feature patch from upstream into a Unsloth fork or feature branch.

Do **not** use this policy for original Unsloth features, routine dependency version bumps, or vendored file updates that are already covered by an automated sync script (e.g., `scripts/sync-*.sh`).

---

## Goal and scope

**Goal:** Bring upstream changes into Unsloth only when the benefit clearly outweighs the risk, and only after the change has been validated against Unsloth's entry points and tests.

**Scope:** Any cherry-pick that touches one of the following surfaces must follow the full policy:

- `studio/backend/` (llama.cpp / llama-server inference, model loading, quantization, speculative decoding).
- `unsloth/` (training, fine-tuning, patching, export, chat templates, Ollama templates).
- `unsloth_cli/` (CLI commands, `studio` subcommand, server launch flags).
- `pyproject.toml` (dependency pins, optional extras, build requirements).
- `tests/` (test framework, fixtures, or shared test utilities).

Cherry-picks that only touch documentation, CI, or asset files may be reviewed with a lighter process but still require a clean test run and an explicit decision record.

---

## Architecture constraint

Upstream changes must be layered on top of Unsloth's existing paths without replacing or bypassing them. In particular:

- The llama-server inference path must remain compatible with the `UNSLOTH_LLAMA_SERVER_PORT` pinning mechanism and the `--llama-server-port` CLI flag.
- Training and patching paths must remain compatible with the existing `unsloth` fast-path kernels and patching logic.
- Ollama/chat template overrides in `unsloth/ollama_template_mappers.py` and `unsloth/chat_templates.py` must not be silently overwritten by upstream template changes; they must be merged intentionally.

If a cherry-pick cannot satisfy this constraint without invasive refactoring, it should be **deferred** rather than partially landed.

---

## Cherry-pick workflow

### 1. Identify the candidate

Before cherry-picking, gather the following information:

- Upstream commit SHA or PR number.
- Files changed and lines of code touched.
- Upstream issue/PR discussion that explains the motivation and known risks.
- Whether the change has been released in a stable upstream version or is still experimental.

### 2. Classify the risk

| Risk level | Indicators | Required validation |
|------------|------------|---------------------|
| **Low** | Documentation, isolated bug fix, test-only change, no public API change. | Targeted tests + lint/typecheck. |
| **Medium** | New optional flag, internal refactor, performance change with clear surface. | Targeted tests + relevant test suite slice + build check. |
| **High** | Core inference change, new dependency, public API change, cross-cutting feature. | Full relevant test suite + build/package check + manual smoke + benchmark gate if applicable. |

### 3. Create a dedicated branch

Branch naming convention:

```
cherry-pick/<upstream>-<pr-or-sha>-<short-description>
```

Examples:

```
cherry-pick/llama-cpp-19493-speculative-checkpointing
cherry-pick/transformers-31987-qwen3-loader-fix
```

### 4. Apply the cherry-pick cleanly

- Prefer a single, clean cherry-pick with the original commit metadata preserved.
- If the original commit does not apply cleanly, resolve conflicts and **amend the commit message** to note the conflict resolution and why it was safe.
- If the change requires multiple upstream commits, cherry-pick them in the same order as upstream and keep each commit as a separate step unless squash is explicitly justified.

### 5. Validate the change

Run the cheapest meaningful validation first. Use the following commands as a starting point; broaden the set only when a shared contract or high-risk path changed.

For Python-only changes:

```bash
python -m pytest tests/ -x -q
```

For CLI changes:

```bash
python -m pytest tests/test_cli/ -x -q
python unsloth-cli.py --help
```

For Studio backend / llama.cpp changes:

```bash
python -m pytest tests/studio/ -x -q
python -m pytest tests/test_llama.py -x -q
```

For Ollama / chat template changes:

```bash
python -m pytest tests/test_ollama.py -x -q
python -c "from unsloth.ollama_template_mappers import OLLAMA_TEMPLATES; print(list(OLLAMA_TEMPLATES.keys()))"
```

For packaging changes:

```bash
python -m build --sdist --wheel
pip install dist/*.whl --force-reinstall --no-deps
```

If a relevant test does not exist, add one before the cherry-pick is considered complete.

### 6. Run the safety gates

After validation, pass the following gates before merging or continuing to the next cherry-pick:

1. **Correctness gate:** All tests in the relevant slice pass.
2. **Build/package gate:** The package still builds and the installable artifact is importable.
3. **API compatibility gate:** Public Python APIs and CLI flags remain backward-compatible unless an explicit break is approved.
4. **Performance gate (if applicable):** A benchmark or smoke test shows no regression, or the improvement is documented and worth the added complexity.
5. **Handoff gate:** The handoff or policy file is updated with the decision, commit, validation results, and any deferred follow-ups.

### 7. Decide whether to keep or defer

After validation, make one of three decisions:

- **Keep:** The cherry-pick is safe, tested, and the value is clear. Continue with the normal PR process.
- **Defer:** The change is too invasive, not yet stable upstream, or the validation results are inconclusive. Record the reason and close or pause the branch.
- **Reject:** The change conflicts with Unsloth architecture or is superseded by an existing Unsloth implementation. Record the reason.

Do not partially land a high-risk cherry-pick just to keep the branch alive.

---

## Branch cleanup and handoff

When a cherry-pick branch is complete or deferred, update one of the following:

- `handoffs/` note if the branch is part of a longer-running sync effort.
- This `CHERRY-PICK-POLICY.md` file if the policy itself needs refinement.
- A dedicated branch handoff file if the work spans multiple sessions or agents.

A handoff must include:

- Upstream commit/PR reference.
- Files changed in Unsloth.
- Validation commands run and their results.
- Decision (keep / defer / reject) and rationale.
- Any known follow-ups or unresolved risks.

---

## Examples

### Low-risk: upstream bug fix for a chat template

1. Identify the upstream commit that fixes a template tokenization issue.
2. Cherry-pick it onto `cherry-pick/transformers-31987-chatml-fix`.
3. Run the Ollama/chat template tests.
4. If the fix improves a template already in `unsloth/chat_templates.py`, merge the changes manually and keep the Unsloth overrides intact.
5. Record the result in the PR description.

### Medium-risk: new llama.cpp speculative decoding option

1. Identify the upstream PR and read the discussion for known limitations.
2. Cherry-pick onto `cherry-pick/llama-cpp-19164-ngram-mod`.
3. Wire the new option through the Studio backend without bypassing existing port-pinning and server scheduling.
4. Add a regression test in `tests/studio/` or `tests/test_llama.py`.
5. Run the Studio backend tests and a short smoke test against a local model.
6. If results are positive, keep; if not, defer with a benchmark note.

### High-risk: cross-cutting upstream API refactor

1. Identify the upstream change and list every Unsloth file that would need to change.
2. If the change cannot be isolated behind an existing abstraction, mark it as **deferred** in the handoff rather than partially landing it.
3. Only reconsider if the upstream refactor is required to fix a critical bug or unblock a release.

---

## Exceptions

Maintainers may approve an exception to this policy for:

- Emergency security fixes from upstream.
- Critical bug fixes that block a release.
- Automated syncs that are already validated by a separate CI pipeline.

Any exception must be documented in the commit message and, if significant, in a handoff note.
