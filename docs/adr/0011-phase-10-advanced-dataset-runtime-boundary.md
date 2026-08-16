# ADR 0011: Phase 10 advanced dataset runtime boundary

- Status: Accepted
- Date: 2026-08-16
- Scope: Phase 10 only

## Context

The normative backend repository is `/Users/baran/Desktop/rag-backend`, while the
running `rag-platform-backend` image identifies itself as `v0.26.4`
(`cb93883…`). The local backend HEAD (`a0e091e…` at the start of Phase 10)
contains additional compilation, artifact, navigation and dataset-skill routes
that are not registered by the deployed image. Treating local HEAD routes as
runtime-capable would expose controls that always fail; ignoring them would
violate the mandatory route coverage rule.

The active v0.26.4 surface already provides metadata configuration and bulk
metadata operations, tags, knowledge/artifact graphs, artifact page CRUD,
dataset skill reads, dataset index lifecycle, embedding checks, ingestion logs,
and registrations for tenant-owned global skill spaces/search/index
configuration. Authenticated smoke later established that those registrations
are not functionally usable in the deployed environment: `skill_spaces` is
missing (MySQL 1146) and Elasticsearch refuses connections.

## Decision

1. The frontend implements every functionally usable, user-meaningful Phase 10
   route through typed functions in `advanced-dataset-api.ts`; React components
   do not call the network directly. Registered routes whose deployed
   prerequisites fail remain typed/contract-tested but are classified
   `runtime-disabled` and expose no mutation controls.
2. The Documents dataset workspace exposes a lazy-loaded **Gelişmiş** area. Its
   Metadata, Etiketler, Grafik, Artifact, İndeks & ingestion and Beceriler tabs
   are separate lazy chunks. Experimental graph/artifact/skill surfaces are
   visibly labelled.
3. Dataset-owned compiled skills are read-only and visually separated from
   tenant-owned global skill spaces. The global scope first probes its runtime;
   the current deployment renders the missing-table/search-service reason and
   hides CRUD, config, search, index and reindex controls.
4. Graph payloads use explicit detail limits and incremental artifact-node
   reads. Long-running index work polls every three seconds only while active
   and visible; cleanup aborts timers/requests and generation checks reject stale
   responses. Cancel (`wipe=false`) and destructive cleanup (`wipe=true`) are
   distinct actions.
5. The legacy `knowledge_graph`, `run_graphrag`, `run_raptor`,
   `trace_graphrag`, and `trace_raptor` routes remain API-only compatibility
   aliases. The product uses the canonical graph/index state machine, avoiding
   a duplicate UI state machine; backend auth decorators and service contracts
   remain recorded in the matrix.
6. The route inventory scans the pinned runtime source and performs a scoped
   forward scan of local backend Phase 10 declarations. The 34 forward-only
   method/path records are classified `runtime-disabled`, never hidden behind a
   feature flag and never called by product UI.

## Runtime evidence

`scripts/rag-platform/phase-10-runtime-smoke.mjs` authenticates with an in-memory
throwaway account and verifies representative active metadata, tags, graph,
artifact, ingestion and skill handlers. It then probes nginx, Python `9380` and
Go `9384` directly. Compilation status and the new artifact topic/structure/
alteration routes return 404 on all three surfaces. Navigation is also absent:
Python's legacy fallback wraps method-not-allowed as HTTP 200/code 100, while Go
returns 404. The smoke treats that documented wrapper as an absent route, not a
successful capability.

The same smoke reaches the global skill handlers but receives code 103:
`rag_platform.skill_spaces` does not exist, and Elasticsearch at the configured
endpoint refuses connections. All twelve global skill routes are therefore
functional `runtime-disabled` records until both prerequisites are healthy.

## Security and data handling

All Phase 10 endpoints remain authenticated. The frontend stores no provider
keys, passwords or tokens in feature state or persistent storage and logs no
request payloads. Destructive artifact, index and global-space operations
require explicit confirmation. Backend ownership checks remain authoritative;
permission failures render a dedicated state rather than an empty result.

## Consequences

Phase 10 can be complete against the deployed contract without guessing future
request or response shapes. Deploying a backend image that contains the 34
forward-only routes requires regenerating the inventory, rerunning authenticated
smoke tests, and implementing their now-reachable UI actions in the phase that
authorizes that runtime transition. This ADR does not authorize Phase 11 work or
any backend source modification.
