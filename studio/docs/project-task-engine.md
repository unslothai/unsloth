# Durable project task engine

`core.agent_workspace.task_state` and `task_runner` provide the internal execution
foundation for project background agents. This layer has no HTTP endpoints,
renderer controls, model loaders, or tool dispatcher. Importing it starts no work.
An application constructs `ProjectTaskRunner` with a trusted server executor.

## Ownership and recovery

Each attempt has an immutable instruction and JSON snapshot, a project ID, a
private worker token, and a 30-second lease. Tokens never appear in public task
records or events. SQLite write transactions serialize creation, claiming,
cancellation, child reservations, and retry accounting across connections.

Queued attempts also expire. A submitted task waiting for a worker retains its
lease through the runner's heartbeat. Expired queued or running attempts become
`interrupted`; opening the runner never automatically executes saved tasks.
Recovery commits even when the request that discovered expiry is refused, such
as a late heartbeat from the old owner. Late workers cannot publish results.

Cancellation moves queued tasks directly to `cancelled` and running tasks to
`cancelling`. Running tasks retain ownership while their executor unwinds.
Cancellation takes precedence over a concurrent successful result. Parent
cancellation also cancels its children. Archiving a project requests cancellation
on the next store operation; deleting its database row cascades task history.

Retries create new attempts and preserve the previous result and events. They
require an explicit request, have a single successor per attempt, and stop after
three attempts. A parent cannot retry while its children are active. The runner
also checks for local workers that have not returned, even if their database
leases expired.

## Bounded delegation

The default runner has two root workers and two separate child workers. Two roots
can therefore wait for their children without occupying the child execution lane.
The configured bounds are one to four root workers and one to eight child workers.
The durable queue admits at most 128 active attempts across projects/runners.

A root explicitly reserves a child-count limit (at most eight) and a cumulative
delegated output-token budget (at most 131,072). Each child reserves at most
32,768 output tokens. Reservations are not refunded on failure or cancellation;
child retries consume another reservation. The count is the number of logical
children; the attempt cap bounds retries. Children cannot delegate recursively.
Root retries start a new bounded root attempt.

Root attempts default to a 900-second deadline, with a maximum of one hour.
Children inherit the parent's deadline. The runner signals cancellation when it
expires. Executors must enforce `maxOutputTokens` across their model turns; the
engine cannot measure token usage inside an arbitrary callback. Results are
limited to one MiB of JSON and snapshots to 128 KiB.

## Executor integration contract

The executor receives `TaskContext`, returns a JSON object, and uses
`context.delegate`, `wait_child`, and `retry_child` for bounded delegation.
`context.task` is a copy, so changing it cannot change durable authority.

Before accepting a renderer request, the service must resolve the saved model
selection, permission policy, and project/worktree identity on the server. The
snapshot must exclude credentials and record enough identity to reject a changed
runtime or workspace. The engine does not turn arbitrary snapshot fields or a
role name into authority to access files, run tools, or contact providers.

For each model/tool operation, the adapter must:

1. Call `context.check()` and validate the captured runtime/workspace identity.
2. Enforce the captured role and permission policy through the existing confined
   file and supervised command APIs, including their in-flight operation fences.
3. Pass `context.cancel_event` and the remaining deadline to active operations.
4. Release model admission while waiting for a child. Separate Python workers do
   not solve contention for a model slot held by a waiting parent.
5. Preserve the selected model/provider and output-token cap. An unavailable
   selection must fail explicitly instead of silently loading or routing elsewhere.

A callback must cooperate with cancellation. Python threads cannot be forcibly
stopped safely. `shutdown(timeout=...)` stops admission, signals cancellation, and
returns `False` if any worker or lease watcher has not stopped within the bound.
Workers are daemon threads so an uncooperative callback cannot hold process exit
open. Cancellation does not undo operations already performed.

## Project retirement

The lifecycle adapter must call `begin_project_retirement` before draining tasks
or acquiring the Git/workspace deletion fence. The retirement owner renews its
lease while the operation is in progress and releases it with
`finish_project_retirement`; a different owner cannot release it.

`project_has_active_tasks` reports durable state. `runner.project_is_idle` also
checks live workers in that runner. Check it only while holding the retirement
admission fence; otherwise a new submission may race the observation. Neither
check replaces the existing cross-process filesystem/process operation fence.
Drain every owning runner before removing or merging a workspace.

If a parent returns while children remain active, the engine cancels and drains
them. A child that exceeds the drain timeout leaves the parent `interrupted` and
the project busy until that child's worker returns. This is not successful cleanup.

## Validation scope

The focused tests exercise real SQLite transactions and worker threads, including
concurrent claims and child-budget reservations, both root workers waiting on
children, late results, retry caps, project retirement, and shutdown races.
The CI matrix runs the engine and adjacent project-storage tests on Linux,
Windows, and macOS with Python 3.11 and 3.12.

Model/tool adapters, owned child worktrees, renderer controls, and real-model
qualification are subsequent integration layers. These tests do not certify
model admission, tool confinement, or an end-to-end background coding workflow.

The engine is now included with the task execution/UI consumer in #10658. Its
state and runner tests run in that integration matrix; no standalone engine
workflow is needed in the combined branch.
