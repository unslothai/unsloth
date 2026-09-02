# Sloth Graphs

Sloth Graphs are durable, project-scoped workflows for composing existing Sloth Loop style agent work with model, tool, condition, approval, and output nodes.

The graph coordinator is not a second agent runtime. Loop and model nodes submit work through `core.agent_workspace.background.BackgroundTaskManager`, which remains the single provider-neutral execution boundary. Tool nodes use the existing MCP client and an explicit graph permission allow-list.

## Contract

Each graph has an immutable, numbered revision. A revision contains:

- typed nodes and directed edges;
- bounded input and output JSON schemas using `type`, object `properties`, `required`, `additionalProperties`, and array `items`;
- tool-server permissions;
- bounded node count, run time, output size, execution attempts, and reserved output tokens;
- project ownership and creation timestamps.

The validator rejects duplicate IDs, dangling edges, self edges, cycles, unreachable nodes, unsupported joins, invalid node configuration, and graphs that do not start at one input node and terminate at an output node. Condition nodes may select a true or false edge.

Graph runs pin the selected revision at creation. Loop and model nodes must name a durable runtime, and run creation captures its credential-free provider routing and account binding. Tool nodes capture a digest of the enabled MCP endpoint, saved headers, OAuth mode, and OAuth account. Execution fails closed if any bound resource changes before dispatch, including after an approval. Durable state includes the run, every node execution attempt, loop task checkpoints, append-only events, tool-effect records, and approval decisions. Run actions are project-scoped and support queued execution, inspection, pause, resume, cancellation, retry lineage, idempotency keys, and restart recovery. A restart interrupts ordinary active work but completes a persisted cancellation as cancelled.

Every node revision carries an immutable retry policy with bounded attempts, backoff, and retry categories. Approval nodes never retry automatically. Loop and model nodes may retry automatically only with `permissionMode: off`. Side-effecting tool nodes require a rendered idempotency key and never retry automatically. The tool-effect ledger returns a prior completed result for the same project, server, tool, key, and argument digest. Receipts survive graph and stopped-run history deletion for the lifetime of the project. Receipt schema migration is transactional and recovers a legacy table left by an interrupted older migration. An in-flight or ambiguous effect fails closed and requires external verification before an operator chooses a new key. MCP configuration is revalidated immediately before dispatch.

Loop and model executions persist their background task ID and terminal status as a node checkpoint. Pause is cooperative for native Loop work: the active task reaches a durable boundary, then resume reuses its completed result. Hard cancellation reasons cannot be downgraded by a later pause. A stopped checkpoint can start a replacement task only when its revision explicitly uses `permissionMode: off`; side-effect-capable checkpoints fail closed for project inspection. The run token budget is enforced conservatively by reserving each new model task's maximum output tokens in the same transaction as its checkpoint marker. The iteration budget counts every node attempt, including retries. The run-time budget uses the original persisted start time across pause, resume, and process restart.

Sequential recovery reconstructs the completed path from the pinned DAG and each condition result instead of relying on timestamp order. Run creation, start, immediate pause or cancellation, restart recovery, approval changes, node admission, node completion, and terminal run state commit with their matching events. Spending an iteration, creating its node execution, and appending `node.started` is one transaction. Node output, its completion event, and the next cursor also commit atomically, and a lost commit acknowledgement reuses the durable completion instead of downgrading it. Cancellation always dominates a concurrent worker failure or pause finish. A pending approval cannot be decided after its run is cancelled, cancelling, completed, or failed. Reusing a run idempotency key with different input or a different revision is rejected.

Templates may reference `input`, `previous`, and `nodes.<id>` in loop instructions, model prompts, and tool arguments. Approval nodes remain pending until an authenticated project user decides them. Tool server IDs must appear in `permissions.allowedToolServerIds`.

## API and Studio

The authenticated routes are under `/api/agent-workspace/projects/{project_id}`:

- `POST /graphs/validate`, `POST /graphs`, `GET /graphs`, `GET /graphs/{graph_id}`, `GET /graphs/{graph_id}/revisions`, `PUT /graphs/{graph_id}`, and `DELETE /graphs/{graph_id}`. Deletion is refused while a run is active and removes stopped run history with the graph;
- `POST /graphs/{graph_id}/runs` and `GET /graphs/{graph_id}/runs`;
- `GET /graph-runs/{run_id}`, `GET /graph-runs/{run_id}/events`;
- `POST /graph-runs/{run_id}/start`, `pause`, `resume`, `cancel`, and `retry`;
- `POST /graph-runs/{run_id}/approvals/{approval_id}`.

The Agent Workspace panel includes a node palette, React Flow canvas, edge editor, node configuration and mapping panel, server-side validation, immutable revision history, advanced contract JSON, test runs, lifecycle controls, node execution inspection, event history, and approval actions. Pointer and keyboard users can create and reconnect edges. Client-side topology checks refuse self edges, joins, cycles, duplicate edges, invalid branches, terminal output sources, input targets, and unsupported fan-out before the server performs authoritative validation.

Test runs submit the exact revision open in the editor. Dirty graph contracts block test runs, while edited run input remains runnable and participates in discard guards until a run accepts it. Graph, run, and mutation responses are fenced by graph ID, run ID, and request generation so a late poll cannot reselect an older run. Dirty drafts are protected across project tabs, project switching, project deletion, new-chat submission, deferred composer sends, project-list navigation, workspace capability refreshes, and window unload. Compare mode keeps the mounted project editor and does not trigger a discard prompt. Dynamic editor errors, draft state, and polled run status use alert or live status semantics. Authenticated authored strings round-trip without output redaction; execution output and errors retain their existing redaction boundary. The backend remains authoritative for validation, revision pinning, project scope, permissions, retry safety, budgets, and state transitions.

## Verification boundary

Focused validation covers graph persistence, schema migration, revision and resource pinning, required runtimes, DAG rejection, schema enforcement, sequential execution through the existing adapter boundary, bounded retries, durable run-time, token and iteration exhaustion, atomic node admission, checkpoint reuse, pause and resume, cancellation dominance, atomic lifecycle and approval events, late approval rejection, restart recovery, project deletion fencing, project scoping, strict idempotent starts and retries, duplicate side-effect prevention across graph deletion, ambiguous-effect fencing, malformed input, MCP endpoint and OAuth account changes after approval, and revision APIs. Frontend validation executes graph state, topology, route, run identity, and one-shot navigation authorization helpers. It also server-renders the draft, live status, and alert primitives used by the editor. A real local stdio MCP subprocess graph smoke is included. Real model or provider execution and packaged desktop runs remain separate release gates.
