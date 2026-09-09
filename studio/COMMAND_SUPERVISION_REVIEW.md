# Secure project edits and command execution

This is the second source split from #9673, independently based on upstream
`main@0dcea7574cdd3ff86b626ec1c6b9d314639e1134`. It uses Studio's existing
managed workspaces. The small project authority module is shared with #10576;
repository instructions, folder selection, verification, Git operations,
delegation, scheduling, and chat continuity are outside this patch.

## Behavior

- Sandboxed `edit_file` uses descriptor-relative POSIX operations or verified
  Windows handles, rejects links and nonregular targets, checks expected content
  and file identity, preserves POSIX mode bits, and publishes replacements
  atomically. Creates do not overwrite a nonempty existing file. Windows edits
  preserve the DACL, attributes, and creation time; read-only targets,
  unsupported streams, and replacement metadata are rejected.
- Linux managed-project Python and terminal calls run through bubblewrap with a
  restricted filesystem, no network, scrubbed environment, bounded output, and
  an identity-verified PID namespace. The project lease and mutation lock remain
  held until descendant cleanup is proven. Failed cleanup stays quarantined and
  is retried during shutdown.
- macOS and Windows managed-project commands fail before launching user code
  because the supervisor cannot provide equivalent containment there. Ordinary
  conversation commands and explicit Full access retain their existing paths.
  Windows file editing is a separate capability from Windows command execution.
- Cancellation interrupts edits waiting for a workspace mutation lock. Command
  preflight rejects oversized workspaces instead of scanning indefinitely.
- A failed project lookup cannot fall back to ordinary command execution. A
  stored chat whose id begins with `project-` retains its own conversation root.
- The edit card shows the requested file, replacement count, arguments awaiting
  approval, and operation errors. Remembered per-tool approval does not approve
  later local edit, Python, or terminal calls in modes that request confirmation.

Content checks detect changes observed before publication; this is not an
atomic compare-and-swap against arbitrary external filesystem writers. The
mutation lock coordinates Studio's guarded operations. Full access and unrelated
host processes are outside that coordination.

## Validation scope

Local validation uses the current source, rather than historical aggregate
receipts. The focused native macOS run covers edits, supervisor behavior, and
shutdown. The full frontend suite has 7,120 passing tests. The adjacent backend
selection has 1,407 passes and four platform skips. Frontend typecheck, build,
and bundle budget pass; changed frontend files add no ESLint findings relative
to the base. The workflow guard selection has 530 passes. Storage deletion fixtures
must use a disposable directory outside macOS system temp paths because Studio
intentionally refuses deletion under `/private/var` and `/private/tmp`.

The Linux/macOS/Windows workflow executes model-free native filesystem tests.
Linux must exercise the real bubblewrap boundary; Windows exercises native file
mutation and junction rejection plus command refusal. Portable mocks alone do
not certify native Windows behavior. Hosted results must be read from the exact
published head before making a platform claim.

Packaged desktop applications, physical UI acceptance, real models/providers,
and release certification remain separate gates.

## Native API references

- [Microsoft NtSetInformationFile](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntsetinformationfile)
  and [FILE_RENAME_INFORMATION](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information):
  native handle-relative publication, traversal access on the parent, and
  filename lengths measured in UTF-16 bytes.
- [bubblewrap source](https://github.com/containers/bubblewrap/blob/main/bubblewrap.c):
  namespace status, blocked startup, and PID-namespace lifecycle.

The next split is verification workflows and hooks.

This follow-up is dependent on #10577, which owns the native edit boundary. This diff adds command supervision, process fences, shutdown recovery, and approval UI.
