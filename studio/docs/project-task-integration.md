# Project task execution and review

This layer connects the durable engine in #10655 to model execution, owned Git
worktrees, authenticated HTTP routes and a project task panel. It is a separate
source split. Install these prerequisite layers before exposing task execution:

| Prerequisite | Exact revision used in CI |
| --- | --- |
| #10655 task engine | `191ac8ba86943eb0bd54a9f1532c5d2ed384c6c5` |
| #10633 project lifecycle | `41e6153ddfce1ed41a6b4e32e9e1ded8c8042e2b` |
| #10577 secure edits | `c6118ec5a653e94158bd637bc59bc18656ca6257` |
| #10594 Git and worktrees | `0e174c1ac4b808b14c37389a987f7c3a23bdfd1d` |

The task endpoints return 503 when prerequisites are missing or their protocol
versions are incompatible. Shutdown also tolerates an unavailable task service. The integration
requires the task retirement and worktree guard protocol markers, so an older
Git/lifecycle implementation cannot silently admit work without its fences.
CI checks the source split first, then merges these exact revisions and tests the
composed behavior. Windows exercises refusal and portable runtime/API contracts;
owned worktree execution remains POSIX-only, following the Git prerequisite.

## User workflow

Open a project, expand **Project tasks**, select a model in Chat, and describe a
change or review. The coordinator can read its checkout and delegate up to two
children. Reviewer children can read; implementers can read and edit. Each
attempt starts from the commit captured when the root was submitted, including
explicit retries. Primary uncommitted changes are not copied into these worktrees.
Children do not inherit another child's dirty changes and cannot delegate again.

The panel shows the model, role, attempt, parent and retry relationships, budgets,
status, output and owned worktree ID. Cancel signals live workers and the durable
record. Retry creates an explicit successor with a fresh worktree; it is never
automatic. The backend remains authoritative when a stale UI offers a retry.
Polling returns bounded summaries for the newest 100 attempts, fetching their
worktree bindings in one database query; expanding a result
fetches its full output separately. Network failures never automatically resend
task mutations, including through Tauri's network retry wrapper.

**Review worktree changes** shows tracked differences from the starting commit
and bounded new-file previews. Symlinks, unreadable, binary or oversized new files
are labelled unavailable. Incomplete previews say so. Bound task worktrees are
preserved after completion, cancellation or failure. A checkout whose setup fails
before binding is rolled back only when Git can prove ownership and that it is
clean; otherwise its Git record remains available for recovery. No commit, merge,
publication or command execution occurs automatically unless command execution
was explicitly enabled for the task. The existing Git panel
can show the owned checkout location for further inspection and explicit Git work.

## Execution and boundaries

- Supported runtimes: saved tool-capable providers that accept an output token
  limit, and an already loaded matching GGUF runtime with idle slot clearing.
  The subscription endpoint rejects output caps and is refused here. Safetensors
  integration is also deferred. No runtime is loaded or substituted by a task.
- A server snapshot binds provider routing, model membership and the encrypted
  credential row's version. Credentials are resolved only in the server. Ciphertext
  fingerprints, native root identities and project instructions are omitted from
  API responses. Model-process replacement and provider/key changes reject work.
- The existing Studio tool loop receives a private callback and an explicit
  task-only catalogue. Unknown tools never fall back to the general executor.
  Every task tool result is capped before model replay using the shared
  `UNSLOTH_TOOL_RESULT_MAX_CHARS` setting, including file reads, listings and child
  results. Truncated reads cannot supply the exact whole-file contents required
  for replacement. Task tools cannot access network tools, Git metadata,
  arbitrary roots or interactive-chat permission overrides. An optional command
  layer adds bounded test/build execution for opted-in implementer children.
- Each model turn reserves at most 1,024 output tokens before dispatch, also
  respecting the saved provider output-token cap, with no refunds for absent
  usage metadata. Changing that cap invalidates queued runtime snapshots. The sum of requested caps is bounded by the
  attempt allocation. The shared local model admission queue accounts for prompt,
  tool schema and output tokens using the loaded aggregate KV budget. Its slot
  and token lease end before any tool invocation or child wait.
- Worktree paths are resolved from owned Git records and marker proofs. Reads and
  edits use the secure mutation boundary with captured device/inode identity,
  serialized file access, exact expected contents, no-follow descriptors and
  bounded UTF-8 payloads. Only implementer children receive mutation tools.
- Every executor retains a shared native project fence, its session lease and
  an exclusive worktree fence until all physical tool calls return. An expired
  database lease cannot make an active worktree mergeable or removable.
- Archive/delete fences task admission, cancels and drains tasks, then acquires
  the exclusive native task-project fence before entering Git retirement. This
  includes workers in another process whose durable leases have expired. New
  executors remain fenced through the archive/delete transaction even if its
  database lease renewal fails. A failed drain leaves the project intact.
- Shutdown cancels the runner and reports an incomplete drain. Uncooperative
  native work retains its fences; the API does not claim that it stopped.

## Validation scope

Tests run the actual shared tool loop, SQLite task engine, one-slot model
admission and real Git worktrees with deterministic simulated model responses.
Two parent workers delegate, wait, and finish while children edit distinct
checkouts. Separate tests cover API authentication/scope, forbidden renderer
authority, runtime/key drift, request-cap accounting, stalled-stream cancellation,
role restrictions, path/symlink escape, native retirement/worktree fences, and
bounded review/output projection. Regression cases also inject setup failures
before worktree binding, retain dirty or bound checkouts, exercise incompatible
prerequisite shutdown, check one binding query per full polling page, and enforce
provider and model-visible result caps. Adjacent engine, loop, lifecycle, worktree and
secure-edit suites are included in native CI, plus frontend API, type and build
checks. This does not claim live-model quality, performance or GPU qualification.

The lazy task entry adds 195 compressed startup bytes with the same local
toolchain: main `191b69c12` measured 1,620,018 bytes and the integration measured
1,620,213 bytes, both with 82 eager chunks. The existing 1,620,000-byte ceiling was
already exceeded by 18 bytes on that baseline. The budget increases by 1 KiB;
the task panel, API code and review controls remain in a separate lazy chunk.


## Optional command layer

The command capability is off by default and remains unavailable without the
separate task-command implementation and a supported Linux host. The API captures
`allowCommands` in the server snapshot; only implementers receive its tool, and
retries recheck support. The panel reads bounded command evidence separately from
model-generated task results, including after cancellation. Command count, output,
timeout, confinement and provenance are enforced by that optional source layer.
Existing Git and retirement process fences protect quarantined commands; no extra
nested project flock is introduced by these hooks.
