# Pinned UEmbed reference fixture

- Upstream: `https://github.com/Alibaba-NLP/UEmbed`
- Commit: `b0d9483faeb152879c847c3ddad4e3534f7e1dd8`
- Source path at that commit: `src/models/qwen35_embedding.py`
- Normalized snapshot SHA-256: `689e1968d526fe8750882b2a50045aa980d1328e7b3c65068e52954178d35b85`
- Reference class: `Qwen35Embedder`

`vendor/qwen35_embedding.py` is the upstream snapshot with trailing whitespace removed. It sits
under a `vendor/` directory because it is third-party source carrying its own licence, and both
the repo-wide license-header lint and the ruff/pre-commit formatters skip that path. Reformatting
it would change the bytes whose SHA-256 is pinned above.

The upstream class opens
`sparse_info.json` and `sparse_weights.pt` with `os.path.join(model_name_or_path, ...)`, so
a Hub ID silently disables sparse loading. `reference_module.py` is a test-only adapter:
it resolves a known Hub ID to its immutable cached revision and delegates to the unchanged
upstream class. It also registers the dynamically loaded snapshot in `sys.modules` before
execution, matching normal Python import semantics required by Transformers 5.4 class
registration.

Because this pinned source exposes `process()` rather than `encode()`, the adapter's
`encode()` only converts strings to upstream's `{"text": ...}` input dictionaries and
delegates to `process()` without changing its math. The checkpoint declares float32 model
weights but ships bf16 sparse heads, so the adapter aligns the reference model to its loaded
sidecar dtype before sparse `F.linear`; detached fp16/bf16 results are promoted to float32
only at the parity test's NumPy boundary. All compatibility glue stays outside the pooling
equations; the pinned upstream snapshot code remains semantically unchanged.

The parity test uses this tracked adapter by default, so a clean checkout only needs the
model selection:

```bash
export UNSLOTH_UEMBED_PARITY_MODEL=Alibaba-NLP/UEmbed-2B
pytest -q tests/python/test_uembed_parity.py
```

To override the adapter or checkpoint revision explicitly:

```bash
export UNSLOTH_UEMBED_REFERENCE_MODULE="$PWD/tests/python/fixtures/uembed_reference/reference_module.py"
export UNSLOTH_UEMBED_REFERENCE_REVISION=e7501a4d1be34ac4c7f8d1565cbeaa5b3f5b41b3
```

Set `UNSLOTH_UEMBED_REFERENCE_MODULE` to an empty string to use the parity test's independent
stock-Transformers reference implementation instead. These fixtures contain no model weights
or datasets.
