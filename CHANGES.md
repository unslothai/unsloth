# Changes

## Fix: PyTorch 2.6.0 Compatibility - recompile_limit removal (#4728)

### Problem
`torch._dynamo.config.recompile_limit` was unconditionally set in
`studio/backend/core/training/trainer.py`. This attribute was renamed to
`cache_size_limit` in PyTorch 2.1 and completely removed in PyTorch 2.6.0,
causing an `AttributeError` on import for Python 3.12 users with fresh installs.

### Files Modified
- `studio/backend/core/training/trainer.py` (line 45)

### Fix
Replaced the unconditional assignment with version-aware feature detection:

```python
# Handle PyTorch version differences for dynamo recompile limit:
# - PyTorch 2.1+: renamed to cache_size_limit
# - PyTorch 2.6+: recompile_limit was completely removed
if hasattr(torch._dynamo.config, 'cache_size_limit'):
    torch._dynamo.config.cache_size_limit = 64
elif hasattr(torch._dynamo.config, 'recompile_limit'):
    torch._dynamo.config.recompile_limit = 64
```

### How to Test
- Import the trainer module with PyTorch 2.6.0+ — should no longer raise `AttributeError`
- Verify training still works on older PyTorch versions (2.0.x) that have `recompile_limit`

### Risks / Concerns
None. The `hasattr` guard is safe across all PyTorch versions and silently skips
setting the limit if neither attribute exists in some future version.
