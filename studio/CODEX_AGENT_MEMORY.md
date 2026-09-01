# Agent memory and dreaming

Unsloth Studio projects now have a portable memory layer under:

```text
.unsloth/memory/
  organization/
  project/
  agent/
  session/
```

Entries are Markdown files. `.metadata.json` records the current SHA-256 hash,
version, update actor, and transcript provenance. Previous contents are kept in
`.versions/`. Writes use an atomic replacement and an exclusive lock. Updates
must provide the hash that was read, so a concurrent edit is rejected and can be
redrafted against the new contents.

Agents receive bounded organization and project memory in project-session
context. The prompt labels memory as data, not instructions. Agent tools are
available only in persisted project sessions:

- `memory_search`
- `memory_read`
- `memory_write`
- `memory_update`

Agents can write project, agent, and session namespaces. Organization memory is
read-only to agents. The authenticated workspace API can review and edit every
namespace, subject to the same path and hash checks.

## Dreaming

Dreaming is an asynchronous, review-only background task. The user selects one
or more project transcripts and may provide steering instructions. Each
transcript is analyzed independently with bounded message and byte limits,
then the orchestrator aggregates repeated explicit preferences into proposals.
The input transcripts and current memory are never modified by the analysis.

Dream output remains in the durable background-task result until a person
accepts or rejects each proposal. Accepting a proposal writes a versioned
Markdown entry with the source transcript IDs and dream ID. Rejecting it leaves
the memory store unchanged. Active dreaming tasks use the same cancellation,
retry, shutdown interruption, and project-deletion fencing as other Studio
background tasks.

Relevant endpoints are:

```text
GET    /api/agent-workspace/projects/{project_id}/memory
GET    /api/agent-workspace/projects/{project_id}/memory/entry?path=...
PUT    /api/agent-workspace/projects/{project_id}/memory/entry
DELETE /api/agent-workspace/projects/{project_id}/memory/entry
GET    /api/agent-workspace/projects/{project_id}/memory/transcripts
POST   /api/agent-workspace/projects/{project_id}/memory/dreams
GET    /api/agent-workspace/projects/{project_id}/memory/dreams
POST   /api/agent-workspace/projects/{project_id}/memory/dreams/{dream_id}/cancel
POST   /api/agent-workspace/projects/{project_id}/memory/dreams/{dream_id}/proposals/{proposal_id}
```

The current analyzer is intentionally deterministic and local. It provides the
production lifecycle, evidence, permissions, and review boundary without
silently training or changing model weights. A future model-backed analyzer can
replace the per-transcript analyzer while keeping the same proposal contract.
