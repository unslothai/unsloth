# Confined file edits

This first secure-tools split changes only sandboxed `edit_file` and its native
filesystem boundary. Linux command supervision, process fences, shutdown cleanup,
and approval UI are reviewed in the separate command follow-up.

POSIX reads and mutations traverse opened directory descriptors without following
symlinks. Windows opens each component without following reparse points and checks
identity, DACLs and basic metadata. Both paths reject hard links, changed content,
and unsupported file types before publication. File creation and replacement are
atomic and bounded. Cancellation releases only this edit's acquired slot.

Existing text matching, multiple edits, newline/BOM preservation, modes, and result
strings retain their public contract. Sandboxed symlink paths remain refused;
explicit Full access retains the existing unconfined edit behavior. Project roots
come from persisted records and retain the deletion lease. Ordinary chats continue
to use their conversation workspace, including IDs with a project prefix.

The native CI matrix runs the edit contract on Linux, macOS and Windows. It does
not install bubblewrap or execute command supervision. Run from `studio/backend`:

```sh
python -m pytest tests/test_edit_file_tool.py tests/test_secure_cross_platform_tools.py -q
```

Native Windows checks and the hosted matrix must pass on each new head; a local
macOS pass does not certify the Windows filesystem implementation.
