# Cloudflare Clef joint schema model (Apache-2.0)

- Upstream: https://huggingface.co/Cloudflare/clef (`joint_schema_model.py`, identical in
  https://huggingface.co/Cloudflare/clef-flash), revision and sha256 in `clef_manifest.json`.
- Licence: `LICENSE` beside this file, copied unmodified from the upstream repo.

`joint_schema_model.py` is byte-identical to upstream and is the reference implementation:
`unsloth/models/clef.py` imports its record encoding and layers, and subclasses its head with
Unsloth's batched, memory-bounded forward. Ruff and the kwargs formatter skip this directory
(`[tool.ruff] extend-exclude` and the `ruff-format-with-kwargs` hook in `.pre-commit-config.yaml`)
so the bytes keep matching the manifest; `tests/test_decision_clef_head.py` checks the digests.
