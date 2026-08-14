#!/usr/bin/env node
/**
 * Rag Platform endpoint coverage matrix generator + validator.
 *
 * Reads docs/rag-platform/route-inventory.json (the authority for which routes
 * exist) and assigns every record exactly one product class, one target phase,
 * an owner, an auth role, a consumer, an implementation status, test evidence
 * and a justification. Outputs are generated, never hand-edited, so the matrix
 * can never disagree with the inventory it is derived from.
 *
 * Classes (plan §5 rule 13):
 *   frontend-screen    a dedicated Rag Platform view renders this response
 *   frontend-action    the UI calls it from within a screen (mutation or sub-read)
 *   api-only           live contract with no UI of its own (protocol, compat shim)
 *   external-callback  inbound request from a third party, not from our UI
 *   internal           backend/runtime plumbing, never a product capability
 *   unsupported        the deployment cannot serve it, so nothing is built on it
 *
 * How `class` is decided for a record the deployment cannot serve: the product
 * decision is `unsupported` — we build nothing against a closed route — and the
 * justification names ADR 0005 plus, where one exists, the reachable route that
 * covers the same capability. The record is never dropped (plan §5 rule 19).
 *
 * Statuses:
 *   contract-verified  a scrubbed fixture in docs/rag-platform/fixtures records
 *                      the live request/response pair
 *   planned            classified, implementation belongs to its target phase
 *   in-progress        implementation started, phase not closed
 *   implemented        UI path + typed service + automated test in place
 *   runtime-disabled   nginx/service topology cannot serve it (see ADR 0005)
 *   not-proxied        reachable only on its own port, opt-in at startup
 *
 * Usage:
 *   node scripts/rag-platform/coverage-matrix.mjs [--check]
 *
 *   --check   Do not write. Exit 1 on drift against the committed outputs, on
 *             any unclassified record, on a duplicate, or on a record present in
 *             one artifact but not the other (CI gate, plan §4 + §15).
 */

import {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, "..", "..");
const OUT_DIR = join(FRONTEND_ROOT, "docs", "rag-platform");
const FIXTURE_DIR = join(OUT_DIR, "fixtures");
const INVENTORY_PATH = join(OUT_DIR, "route-inventory.json");

const checkOnly = process.argv.slice(2).includes("--check");

if (!existsSync(INVENTORY_PATH)) {
  console.error(
    `route inventory missing: ${relative(FRONTEND_ROOT, INVENTORY_PATH)} — run: node scripts/rag-platform/route-inventory.mjs`,
  );
  process.exit(2);
}

const inventory = JSON.parse(readFileSync(INVENTORY_PATH, "utf8"));

// ---------------------------------------------------------------------------
// Path canonicalisation. Fixtures carry real ids, the inventory carries
// parameter names, and the Go router uses a third syntax. All three normalise
// to `{p}` so a fixture can be matched back to the route it exercised.
// ---------------------------------------------------------------------------

function canonicalPath(path) {
  return path
    .split("?")[0]
    .replace(/<[^>]+>:<[^>]+>/g, "{p}")
    .replace(/<[^>]+>/g, "{p}")
    .replace(/:[^/]+/g, "{p}")
    .replace(/\*[^/]+/g, "{p}")
    .replace(/\/[0-9a-f]{32}(?=\/|$)/g, "/{p}")
    .replace(
      /\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=\/|$)/g,
      "/{p}",
    );
}

function familyOf(route) {
  // The MCP server mounts its transport at bare paths (/mcp, /sse, /messages/),
  // so the first path segment would scatter one protocol across three families
  // — /messages/ would land in the memory-messages family. Group by service.
  if (route.service === "mcp") return "mcp";
  const segments = route.path.split("/").filter(Boolean);
  const versionIndex = segments.indexOf("v1");
  return (
    (versionIndex >= 0 ? segments[versionIndex + 1] : segments[0]) || "(root)"
  );
}

// ---------------------------------------------------------------------------
// Target phase. Base mapping is the plan §4 scope matrix, family by family.
// PHASE_OVERRIDES handles the families the plan splits across several phases
// (datasets in particular) and is evaluated first.
// ---------------------------------------------------------------------------

const FAMILY_PHASE = {
  "(root)": 1,
  health: 1,
  healthz: 1,
  live: 1,
  language: 1,
  system: 1,
  auth: 2,
  users: 2,
  user: 2,
  providers: 3,
  models: 3,
  "all-models": 14,
  pipelines: 3,
  embeddings: 3,
  rerank: 3,
  audio: 3,
  components: 11,
  datasets: 4,
  documents: 5,
  document: 5,
  thumbnails: 5,
  tasks: 5,
  chunk: 6,
  retrieval: 6,
  chats: 7,
  sessions: 7,
  chat: 8,
  langfuse: 9,
  skills: 10,
  agents: 11,
  mcp: 11,
  sse: 11,
  plugin: 11,
  connectors: 12,
  connector: 12,
  files: 12,
  file: 12,
  folders: 12,
  workspace: 12,
  workspaces: 12,
  "workspace-files": 12,
  memories: 13,
  messages: 13,
  searches: 13,
  admin: 14,
  tenants: 14,
  tenant: 14,
  settings: 14,
  "chat-channels": 14,
  chatbots: 14,
  agentbots: 14,
  searchbots: 14,
  chats_openai: 14,
  agents_openai: 14,
  openai: 14,
  dify: 14,
  "compilation-templates": 14,
  "compilation-template-groups": 14,
  compilation_templates: 14,
  compilation_template_groups: 14,
  llm: 14,
};

/** Ordered; first match wins. Tested against the canonical path. */
const PHASE_OVERRIDES = [
  // Provider/model first-run flows explicitly owned by phase 3.
  [/^\/api\/v1\/users\/me\/models$/, 3],
  [/^\/api\/v1\/chat\/to_model$/, 3],
  [/^\/api\/v1\/file\/(ocr|parse)$/, 3],
  // system: tokens and stats are the phase 9 observability/API-key surfaces,
  // while the public auth-capability config belongs to phase 2. The raw plural
  // configs surface is backend/admin plumbing and is reviewed in phase 14.
  [/^\/api\/v1\/system\/tokens/, 9],
  [/^\/api\/v1\/system\/(stats|keys)/, 9],
  [/^\/api\/v1\/system\/status$/, 9],
  [/^\/api\/v1\/system\/config$/, 2],
  [/^\/(api\/)?v1\/system\/configs$/, 14],
  [/^\/api\/v1\/system\/(variables|environments|oceanbase)/, 14],
  [/^\/api\/v1\/system\/config\/log/, 14],
  // datasets is split across five phases by the plan §4 rows.
  [/^\/api\/v1\/datasets\/\{p\}\/documents\/\{p\}\/(chunks|structure\/graph)/, 6],
  [/^\/api\/v1\/datasets\/\{p\}\/documents\/\{p\}\/metadata\/config$/, 10],
  [/^\/api\/v1\/datasets\/\{p\}\/documents\/(metadatas|batch-update-status)$/, 10],
  [/^\/api\/v1\/datasets\/\{p\}\/documents/, 5],
  [/^\/api\/v1\/datasets\/\{p\}\/(chunks|search)/, 6],
  [/^\/api\/v1\/datasets\/search$/, 10],
  [/^\/api\/v1\/datasets\/\{p\}\/(any_artifact|any_skill|embedding)$/, 10],
  [/^\/api\/v1\/datasets\/\{p\}\/\{p\}$/, 10],
  [/^\/api\/v1\/datasets\/\{p\}\/embedding\/check$/, 10],
  [/^\/api\/v1\/datasets\/(\{p\}\/)?(metadata|tags)/, 10],
  [
    /^\/api\/v1\/datasets\/\{p\}\/(artifacts|graph|knowledge_graph|navigation|skills|index|ingestion|run_graphrag|run_raptor|trace_graphrag|trace_raptor|compilation)/,
    10,
  ],
  [/^\/api\/v1\/datasets\/\{p\}\/(commits|changes)/, 12],
  [/^\/api\/v1\/datasets\/ingestion\/tasks$/, 5],
  // documents: preview/media/artifact are the phase 5 viewer surfaces.
  [/^\/api\/v1\/documents\/(artifact|images)\//, 5],
  // chats: completions and feedback belong to the phase 8 chat runtime.
  [/^\/api\/v1\/chats\/\{p\}\/completions$/, 8],
  [/\/feedback$/, 8],
  // agents: the whole family is phase 11 including the webhook pair.
  [/^\/api\/v1\/agents\//, 11],
];

function phaseOf(route, canonical) {
  for (const [pattern, phase] of PHASE_OVERRIDES) {
    if (pattern.test(canonical)) return phase;
  }
  const phase = FAMILY_PHASE[familyOf(route)];
  return phase === undefined ? null : phase;
}

// ---------------------------------------------------------------------------
// Owner: the frontend feature module that owns the surface.
// ---------------------------------------------------------------------------

const FAMILY_OWNER = {
  "(root)": "platform",
  health: "platform",
  healthz: "platform",
  live: "platform",
  language: "platform",
  system: "platform",
  settings: "platform",
  langfuse: "platform",
  "compilation-templates": "platform",
  "compilation-template-groups": "platform",
  compilation_templates: "platform",
  compilation_template_groups: "platform",
  auth: "auth",
  users: "auth",
  user: "auth",
  tenants: "admin",
  tenant: "admin",
  admin: "admin",
  providers: "models",
  models: "models",
  "all-models": "models",
  pipelines: "models",
  embeddings: "models",
  rerank: "models",
  audio: "models",
  components: "models",
  llm: "models",
  datasets: "knowledge",
  skills: "knowledge",
  chunk: "knowledge",
  retrieval: "knowledge",
  documents: "documents",
  document: "documents",
  thumbnails: "documents",
  tasks: "documents",
  chats: "chat",
  chat: "chat",
  sessions: "chat",
  "chat-channels": "chat",
  chatbots: "chat",
  chats_openai: "chat",
  openai: "chat",
  dify: "chat",
  agents: "agents",
  agentbots: "agents",
  agents_openai: "agents",
  mcp: "agents",
  sse: "agents",
  plugin: "agents",
  connectors: "files",
  connector: "files",
  files: "files",
  file: "files",
  folders: "files",
  workspace: "files",
  workspaces: "files",
  "workspace-files": "files",
  memories: "memory",
  messages: "memory",
  searches: "search",
  searchbots: "search",
};

// ---------------------------------------------------------------------------
// Auth role: the inventory's decorator-level auth value mapped to the role a
// caller must hold. Kept as an explicit table so an unseen decorator fails loud.
// ---------------------------------------------------------------------------

const AUTH_ROLE = {
  public: "anonymous",
  session: "tenant-user",
  login_required: "tenant-user",
  "login_required(AUTH_JWT,AUTH_API,AUTH_BETA)":
    "tenant-user / api-key / embed-token",
  "beta-token": "embed-token",
  "admin-session": "platform-admin",
  "mcp-api-key": "mcp-api-key",
};

// ---------------------------------------------------------------------------
// Class rules. Ordered, first match wins. Every rule carries the justification
// that goes into the matrix, so a class is never asserted without a reason.
//
// `when` fields: service (regex), method (regex), path (regex over the canonical
// path), notes (regex over the inventory note).
// ---------------------------------------------------------------------------

const SCREEN = "frontend-screen";
const ACTION = "frontend-action";
const API_ONLY = "api-only";
const CALLBACK = "external-callback";
const INTERNAL = "internal";
const UNSUPPORTED = "unsupported";

const FRONTEND = "rag-platform-frontend";

const CLASS_RULES = [
  // -- 0. Phase 1 system routes whose active Go implementation must not be
  // swallowed by the broader historical go-unreachable rule below. ----------
  {
    id: "raw-system-config-internal",
    when: { path: /^\/(api\/)?v1\/system\/configs$/, runtimeEnabled: true },
    class: INTERNAL,
    consumer: "backend operator only",
    justification:
      "Returns the process runtime configuration, including credential-bearing backend fields. It is never exposed by the frontend typed system API or persisted in browser state; source/contract and negative-export security tests only.",
  },
  {
    id: "root-liveness-internal",
    when: { path: /^\/health$/, service: /^go-api$/ },
    class: INTERNAL,
    consumer: "deployment health probe",
    justification:
      "Static Go service liveness used by deployment probes. The user-facing connection store uses the canonical /api/v1/system endpoints; no UI calls this root path.",
  },
  {
    id: "active-system-health-probe",
    when: {
      path: /^\/api\/v1\/system\/(ping|healthz|version)$/,
      runtimeEnabled: true,
    },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Feeds the phase 1 connection store and the small Settings > Connections readiness card. Existing auth, RAG and chat calls remain on their prior clients.",
  },
  {
    id: "auth-capability-config",
    when: { path: /^\/api\/v1\/system\/config$/, runtimeEnabled: true },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Safe registration/password-login capability response used by the phase 2 auth UI; it is not part of the phase 1 connection store.",
  },
  {
    id: "system-status-action",
    when: { path: /^\/api\/v1\/system\/status$/, runtimeEnabled: true },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Authenticated dependency and worker status used by the phase 9 operations UI, not by the anonymous phase 1 connection check.",
  },

  // -- 1. Records the deployment cannot serve -------------------------------
  {
    id: "source-only-runtime-gap",
    when: {
      notes: /backend worktree-only implemented pipeline catalog/,
      runtimeEnabled: false,
    },
    class: UNSUPPORTED,
    consumer: "none (implemented source is newer than the deployed runtime)",
    justification:
      "The normative backend worktree implements the public pipeline catalog, but the pinned v0.26.4 runtime source and active hybrid proxy do not register it. Source, generated proxy target and live HTTP 404 evidence are recorded in runtime-disabled.md; the UI renders an explicit disabled reason and never invents catalog entries.",
  },
  {
    id: "source-only-stub",
    when: {
      notes: /backend worktree-only.*CodeNotImplemented/,
      runtimeEnabled: false,
    },
    class: UNSUPPORTED,
    consumer: "none (route absent from deployed source and handler is a stub)",
    justification:
      "Declared only in the newer backend worktree, absent from the pinned v0.26.4 runtime image, and its current handler returns CodeNotImplemented. Source, proxy destination and live 404 evidence are recorded in runtime-disabled.md and the Faz 2 auth contract; no false UI control is rendered.",
  },
  {
    id: "runtime-unreachable",
    when: { runtimeEnabled: false },
    class: UNSUPPORTED,
    consumer: "none (route closed at runtime)",
    justification:
      "The active hybrid nginx location does not forward this implementation; recorded with source, generated-proxy and live port/proxy evidence in runtime-disabled.md. The owned image runs all four services, so a same-path alternate may remain reachable on the selected target. Decision: ADR 0005.",
  },

  // Active v0.26.4 Go compatibility endpoints remain externally callable,
  // but the product uses the canonical /api/v1 Python contracts. Keeping them
  // api-only avoids a second frontend state machine while preserving contract
  // and auth/security coverage.
  {
    id: "legacy-user-compat",
    when: { path: /^\/v1\/user\//, service: /^go-api$/, runtimeEnabled: true },
    class: API_ONLY,
    consumer: "legacy API client",
    justification:
      "Active v0.26.4 compatibility surface. Rag Platform UI uses the canonical /api/v1 auth/users routes; source-equivalence and auth contract tests cover this duplicate without exposing a second UI path.",
  },

  // -- 2. Deprecated compatibility shims -----------------------------------
  {
    id: "deprecated-shim",
    when: { notes: /backward-compat|deprecated/i },
    class: API_ONLY,
    consumer: "legacy API client",
    justification:
      "Deprecated shim kept by upstream for older clients; its own docstring names the replacement route, which carries the product class. No UI is built on it; contract + auth tests only.",
  },

  // -- 3. Protocol transports ----------------------------------------------
  {
    id: "mcp-transport",
    when: { service: /^mcp$/ },
    class: API_ONLY,
    consumer: "external MCP client",
    justification:
      "MCP wire transport on its own port (9382), opt-in via --enable-mcpserver and not proxied by nginx. Consumed by MCP clients, never by our UI; contract + api-key auth tests only.",
  },
  {
    id: "openai-compat",
    when: { path: /^\/api\/v1\/(openai|dify)\// },
    class: API_ONLY,
    consumer: "external API client (OpenAI/Dify compatible)",
    justification:
      "Third-party protocol compatibility surface, deliberately kept out of the core chat UI (plan §4, phase 14). Contract, auth and quota tests only.",
  },
  // The two `*_openai/<id>/chat/completions` paths are deliberately absent
  // here: both are backward_compat shims (`backward_compat.py:108,126`) and so
  // are already classified by `deprecated-shim` above.

  // -- 4. Inbound third-party callbacks ------------------------------------
  {
    id: "oauth-callback",
    when: { path: /\/callback$/, method: /^GET$/ },
    class: CALLBACK,
    consumer: "OAuth identity/storage provider",
    justification:
      "Unauthenticated inbound redirect target for the provider, not called by our UI. Requires state/nonce validation, redirect allow-list and secret-redaction tests.",
  },
  {
    id: "agent-webhook",
    when: { path: /^\/api\/v1\/agents\/\{p\}\/webhook$/ },
    class: CALLBACK,
    consumer: "external webhook sender",
    justification:
      "Public inbound agent trigger: the handler carries no login decorator, unlike its /webhook/test sibling. Auth is per-agent token only, so abuse, rate-limit and payload-validation tests are mandatory.",
  },

  // -- 5. Backend runtime plumbing -----------------------------------------
  {
    id: "language-probe",
    when: { path: /^\/api\/v1\/language$/ },
    class: INTERNAL,
    consumer: "backend runtime detection",
    justification:
      "Returns the backend implementation language so an upstream client can pick a Go or Python code path. Not a product capability; the Rag Platform frontend targets one contract. Contract test only.",
  },

  // -- 6. Connection / readiness indicator (phase 1) -----------------------
  {
    id: "health-probe",
    when: { path: /^\/api\/v1\/system\/(ping|healthz|version)$/ },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Feeds the connection and readiness indicator (plan §4, phase 1). No dedicated page; polled from the app shell.",
  },
  {
    id: "admin-health-probe",
    when: { path: /^\/api\/v1\/admin\/(ping|version)$/ },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Readiness and build probe for the admin service, polled by the operations screen header rather than rendering a page of its own.",
  },

  // -- 7. Auth -------------------------------------------------------------
  {
    id: "auth-actions",
    when: { path: /^\/api\/v1\/auth\// },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Login, logout, login-channel discovery and password recovery are actions invoked from the auth screens (plan §4, phase 2).",
  },
  {
    id: "tenant-model-selection-read",
    when: { method: /^GET$/, path: /^\/api\/v1\/users\/me\/models$/ },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Settings > Connections reads tenant role and model selections for permission state and first-run readiness. The response is normalized by the auth domain adapter and never exposes provider credentials.",
  },
  {
    id: "tenant-model-selection-bulk-alias",
    when: { method: /^PATCH$/, path: /^\/api\/v1\/users\/me\/models$/ },
    class: API_ONLY,
    consumer: "legacy bulk model-selection client",
    justification:
      "Bulk tenant-field compatibility contract. Rag Platform UI updates one capability at a time through PATCH /models/default so server confirmation and the embedding destructive warning remain atomic; contract tests retain this bulk alias without a second UI state machine.",
  },
  {
    id: "phase3-default-model-bulk-alias",
    when: {
      service: /^go-api$/,
      path: /^\/api\/v1\/models$/,
      method: /^PATCH$/,
    },
    class: API_ONLY,
    consumer: "legacy/batch provider-model client",
    justification:
      "Bulk default-model compatibility form. The UI uses PATCH /models/default per capability so server confirmation and the embedding destructive warning remain atomic; contract tests retain this alias without a second UI state machine.",
  },
  {
    id: "phase3-python-provider-model-aliases",
    when: {
      service: /^python-api$/,
      path: /^\/api\/v1\/providers\/\{p\}\/instances\/\{p\}\/models(\/\{p\})?$/,
      method: /^(PUT|POST)$/,
    },
    class: API_ONLY,
    consumer: "legacy/batch provider-model client",
    justification:
      "Compatibility or batch form of a capability exposed atomically by the canonical Phase 3 model/default actions. Contract tests verify the exact body while the UI uses POST/PATCH instance-model actions and PATCH /models/default to avoid duplicate state machines.",
  },
  {
    id: "phase3-provider-telemetry-api-only",
    when: {
      path: /^\/api\/v1\/providers\/\{p\}\/instances\/\{p\}\/(balance|tasks)(\/\{p\})?$/,
      method: /^GET$/,
    },
    class: API_ONLY,
    consumer: "provider operations client",
    justification:
      "Balance and asynchronous task telemetry are operational provider contracts, not connection or model configuration steps. The focused Connections UI omits them; exact typed contracts and response adapters remain covered without adding secondary dashboard controls to the setup flow.",
  },
  {
    id: "user-self",
    when: { path: /^\/api\/v1\/users(\/[^/]+)*$/ },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Registration, session bootstrap and profile/default-model mutation, invoked from the auth and settings screens rather than rendering a page of their own.",
  },

  // -- Phase 5 document protocol/compatibility surfaces -------------------
  {
    id: "phase5-generic-create-api-only",
    when: { method: /^POST$/, path: /^\/api\/v1\/documents$/ },
    class: API_ONLY,
    consumer: "trusted document-service client",
    justification:
      "The low-level create contract requires an explicit created_by principal and does not accept file bytes. The product upload path is dataset-scoped POST /datasets/{id}/documents; exposing caller-supplied ownership in the UI would violate the Phase 5 ownership boundary. Exact request/response and auth tests retain this service contract.",
  },
  {
    id: "phase5-ingestion-task-protocol",
    when: { path: /^\/api\/v1\/datasets\/ingestion\/tasks$/ },
    class: API_ONLY,
    consumer: "ingestion worker / trusted API client",
    justification:
      "The v0.26.4 GET handler requires a JSON body, which browser Fetch forbids, while dataset document stop is safely available through POST /datasets/{id}/documents/stop. Task list/stop/remove remain typed protocol contracts and are not duplicated as unsafe browser controls.",
  },
  {
    id: "phase5-task-cancel-protocol",
    when: { path: /^\/api\/v1\/tasks\/\{p\}(\/cancel)?$/ },
    class: API_ONLY,
    consumer: "task-aware API client",
    justification:
      "Generic ingest returns only a boolean and no task id, so the document UI cannot safely invent a task identity. The exact PATCH stop and POST cancel contracts are typed and tested for clients that already possess an authorized task id.",
  },
  {
    id: "phase5-dataset-document-download",
    when: {
      method: /^GET$/,
      path: /^\/api\/v1\/datasets\/\{p\}\/documents\/\{p\}$/,
    },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Binary attachment action in Documents → Dataset documents. Metadata comes from the collection response; the authenticated Blob is never placed in persistent storage and its object URL is revoked after use.",
  },

  // -- Phase 6 chunk/retrieval compatibility and internal surfaces ---------
  {
    id: "phase6-chunk-legacy-list",
    when: { method: /^POST$/, path: /^\/api\/v1\/chunk\/list$/ },
    class: API_ONLY,
    consumer: "legacy chunk API client",
    justification:
      "Active POST-with-body compatibility list. Documents → Chunks uses the canonical dataset/document-scoped GET route; the exact legacy body remains typed and contract-tested without a duplicate UI state machine.",
  },
  {
    id: "phase6-chunk-internal-update",
    when: { method: /^POST$/, path: /^\/api\/v1\/chunk\/update$/ },
    class: INTERNAL,
    consumer: "Go internal client only",
    justification:
      "The route registration explicitly marks this endpoint internal-only. Its handler reads dataset_id and document_id from path parameters that the flat route does not define, so the browser UI never calls it; source, auth and negative-export evidence retain the boundary.",
  },
  {
    id: "phase6-legacy-parse-stop-alias",
    when: {
      method: /^(POST|DELETE)$/,
      path: /^\/api\/v1\/datasets\/\{p\}\/chunks$/,
      runtimeEnabled: true,
    },
    class: API_ONLY,
    consumer: "legacy ingestion client",
    justification:
      "Legacy parse/stop aliases for dataset documents. Documents already exposes the canonical /documents/parse and /documents/stop actions from Phase 5; the exact aliases remain typed and contract-tested without adding indistinguishable controls.",
  },
  {
    id: "phase6-dataset-search-alias",
    when: {
      method: /^POST$/,
      path: /^\/api\/v1\/datasets\/\{p\}\/search$/,
      runtimeEnabled: true,
    },
    class: API_ONLY,
    consumer: "dataset-scoped retrieval API client",
    justification:
      "Dataset-scoped search is a compatibility form of the same retrieval capability. Documents → Retrieval Playground uses canonical POST /retrieval because it supports explicit dataset_ids and document_ids; this alias is typed and contract-tested only.",
  },

  // -- 8. Primary entity list + detail reads = dedicated screens -----------
  {
    id: "entity-screen",
    when: {
      method: /^(GET|HEAD)$/,
      path: new RegExp(
        `^/api/v1/(${[
          "datasets",
          "chats",
          "agents",
          "connectors",
          "files",
          "memories",
          "searches",
          "providers",
          "tenants",
          "chat-channels",
          "compilation-template-groups",
          "compilation-templates",
          "mcp/servers",
          "models",
          "messages",
          "skills",
          "folders",
        ].join("|")})(/\\{p\\})?$`,
      ),
    },
    class: SCREEN,
    consumer: FRONTEND,
    justification:
      "Primary entity list or detail read; a dedicated Rag Platform route renders it.",
  },
  {
    id: "nested-screen",
    when: {
      method: /^(GET|HEAD)$/,
      path: new RegExp(
        `^/api/v1/(${[
          "datasets/\\{p\\}/documents",
          "datasets/\\{p\\}/documents/\\{p\\}",
          "datasets/\\{p\\}/documents/\\{p\\}/chunks",
          "datasets/\\{p\\}/documents/\\{p\\}/chunks/\\{p\\}",
          "datasets/\\{p\\}/chunks",
          "chats/\\{p\\}/sessions",
          "chats/\\{p\\}/sessions/\\{p\\}",
          "agents/\\{p\\}/sessions",
          "agents/\\{p\\}/sessions/\\{p\\}",
          "agents/\\{p\\}/versions",
          "agents/\\{p\\}/versions/\\{p\\}",
          "agents/\\{p\\}/webhook/logs",
          "agents/templates",
          "connectors/\\{p\\}/logs",
          "providers/\\{p\\}/instances",
          "providers/\\{p\\}/instances/\\{p\\}",
          "providers/\\{p\\}/models",
          "documents/\\{p\\}",
          "documents/\\{p\\}/preview",
          "tenants/\\{p\\}/users",
          "system/stats",
          "system/tokens",
          "langfuse/api-key",
          "searchbots/detail",
          "plugin/tools",
        ].join("|")})$`,
      ),
    },
    class: SCREEN,
    consumer: FRONTEND,
    justification:
      "Nested collection or detail read that backs its own tab or panel inside the owning screen.",
  },
  {
    id: "admin-screen",
    when: {
      method: /^GET$/,
      path: new RegExp(
        `^/api/v1/admin/(${[
          "auth",
          "users",
          "users/\\{p\\}",
          "users/\\{p\\}/[^/]+",
          "roles",
          "roles/\\{p\\}/permission",
          "services",
          "services/\\{p\\}",
          "sandbox/[^/]+",
          "configs?",
          "environments",
          "log_levels",
          "variables",
        ].join("|")})$`,
      ),
    },
    class: SCREEN,
    consumer: FRONTEND,
    justification:
      "Platform-admin read rendered by a dedicated operations screen (plan §4, phase 14).",
  },

  // -- 9. Everything else the UI calls -------------------------------------
  {
    id: "ui-action",
    when: { service: /^python-(api|admin)$/ },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Mutation or sub-read invoked from within the owning screen; no dedicated route of its own.",
  },
  {
    id: "go-ui-action",
    when: { service: /^go-(api|admin)$/, runtimeEnabled: true },
    class: ACTION,
    consumer: FRONTEND,
    justification:
      "Reachable Go mutation or sub-read invoked from within the owning screen; no dedicated route of its own. Phase implementation must still provide a typed service, UI path and test evidence.",
  },
];

function matches(rule, route, canonical) {
  const { when } = rule;
  if (when.service && !when.service.test(route.service)) return false;
  if (when.method && !when.method.test(route.method)) return false;
  if (when.path && !when.path.test(canonical)) return false;
  if (when.notes && !when.notes.test(route.notes || "")) return false;
  if (
    when.runtimeEnabled !== undefined &&
    route.runtime_enabled !== when.runtimeEnabled
  )
    return false;
  return true;
}

function classifyRecord(route, canonical) {
  for (const rule of CLASS_RULES) {
    if (matches(rule, route, canonical)) {
      return {
        class: rule.class,
        rule: rule.id,
        consumer: rule.consumer,
        justification: rule.justification,
      };
    }
  }
  return {
    class: "unclassified",
    rule: null,
    consumer: "unknown",
    justification:
      "No classification rule matched — the matrix generator must be extended.",
  };
}

// ---------------------------------------------------------------------------
// Hand-verified replacements for closed routes.
//
// The automatic equivalence test compares method plus path (modulo mount
// prefix), so it cannot see a capability that survived under a *different*
// name: the Go API's `GET /v1/user/info` is served by Python as
// `GET /api/v1/users/me`. Guessing those from path similarity would be exactly
// the "do not guess endpoints" failure, so each entry below was read out of
// backend source and its replacement confirmed present in the inventory. The
// validator re-checks every replacement on each run, so a rename that later
// disappears fails the build instead of silently understating the gap.
//
// Keys are `METHOD canonical_path` of the CLOSED route.
// ---------------------------------------------------------------------------

const VERIFIED_REPLACEMENTS = {
  "GET /v1/user/info": {
    replacement: "GET /api/v1/users/me",
    evidence:
      "Go `internal/router/router.go:236` -> Python `api/apps/restful_apis/user_api.py:381` (`user_profile`)",
  },
  "POST /v1/user/setting": {
    replacement: "PATCH /api/v1/users/me",
    evidence:
      "Go `internal/router/router.go:242` -> Python `api/apps/restful_apis/user_api.py:302` (`setting_user`); the verb changed from POST to PATCH in the REST rewrite",
  },
  "GET /v1/tenant/list": {
    replacement: "GET /api/v1/tenants",
    evidence:
      "Go `internal/router/router.go:240` -> Python `api/apps/restful_apis/tenant_api.py:155` (`tenant_list`)",
  },
  "GET /api/v1/tenant/list": {
    replacement: "GET /api/v1/tenants",
    evidence:
      "Go `internal/router/router.go:291` -> Python `api/apps/restful_apis/tenant_api.py:155` (`tenant_list`)",
  },
  "GET /api/v1/all-models": {
    replacement: "GET /api/v1/models",
    evidence:
      "Go `internal/router/router.go:590` (`ListTenantAddedModels`) -> Python `api/apps/restful_apis/models_api.py:31` (`get_added_models`)",
  },
  "PATCH /api/v1/models": {
    replacement: "PATCH /api/v1/models/default",
    evidence:
      "Go `internal/router/router.go:582` (`SetModels`) -> Python `api/apps/restful_apis/models_api.py:162` (`set_default_models`)",
  },
  "POST /api/v1/audio/speech": {
    replacement: "POST /api/v1/chat/audio/speech",
    evidence:
      "Go `internal/router/router.go:572` -> Python `api/apps/restful_apis/chat_api.py:1082`",
  },
  "POST /api/v1/audio/transcriptions": {
    replacement: "POST /api/v1/chat/audio/transcription",
    evidence:
      "Go `internal/router/router.go:571` -> Python `api/apps/restful_apis/chat_api.py:1110`",
  },
  "POST /api/v1/chunk/list": {
    replacement: "GET /api/v1/datasets/{p}/documents/{p}/chunks",
    evidence:
      "Go `internal/router/router.go:705` -> Python `api/apps/restful_apis/chunk_api.py:461`; the POST-with-body listing became a GET with path scope",
  },
  "GET /v1/user/tenant_info": {
    replacement: "GET /api/v1/users/me/models",
    evidence:
      "Go `internal/router/router.go:238` (`tenantHandler.TenantInfo`) -> Python `api/apps/restful_apis/user_api.py:567`, whose handler is still named `tenant_info`; the REST rewrite renamed the path to `/users/me/models` because the payload is the caller's tenant model selection",
  },
  "POST /v1/user/set_tenant_info": {
    replacement: "PATCH /api/v1/users/me/models",
    evidence:
      "Go `internal/router/router.go:246` (`userHandler.SetTenantInfo`) -> Python `api/apps/restful_apis/user_api.py:605`, handler `set_tenant_info`",
  },
  // Document and file routes the REST rewrite re-scoped under their dataset.
  // Go's flat `/documents` and `/document/*` groups became dataset-scoped
  // collections, which is why a path comparison cannot see the survival.
  "POST /api/v1/document/list": {
    replacement: "GET /api/v1/datasets/{p}/documents",
    evidence:
      "Go `internal/router/router.go:696` and `:304` both bind `documentHandler.ListDocuments` -> Python `api/apps/restful_apis/document_api.py:729`; the listing became dataset-scoped and moved from POST-with-body to GET",
  },
  "GET /api/v1/documents": {
    replacement: "GET /api/v1/datasets/{p}/documents",
    evidence:
      "Go `internal/router/router.go:304` (`documentHandler.ListDocuments`) -> Python `api/apps/restful_apis/document_api.py:729`",
  },
  "PUT /api/v1/documents/{p}": {
    replacement: "PATCH /api/v1/datasets/{p}/documents/{p}",
    evidence:
      "Go `internal/router/router.go:307` (`documentHandler.UpdateDocument`) -> Python `api/apps/restful_apis/document_api.py:191`; the verb became PATCH and the document is addressed within its dataset",
  },
  "DELETE /api/v1/documents/{p}": {
    replacement: "DELETE /api/v1/datasets/{p}/documents",
    evidence:
      "Go `internal/router/router.go:308` (`documentHandler.DeleteDocument`) -> Python `api/apps/restful_apis/document_api.py:1137`, which deletes the ids named in the body within a dataset scope",
  },
  "POST /api/v1/document/metadata/summary": {
    replacement: "GET /api/v1/datasets/{p}/metadata/summary",
    evidence:
      "Go `internal/router/router.go:697` (`documentHandler.MetadataSummary`) -> Python `api/apps/restful_apis/document_api.py:324` (`metadata_summary`)",
  },
  "POST /api/v1/document/set_meta": {
    replacement: "PATCH /api/v1/datasets/{p}/documents/metadatas",
    evidence:
      "Go `internal/router/router.go:698` (`documentHandler.SetMeta`) -> Python `api/apps/restful_apis/document_api.py:1345`, the dataset-scoped bulk metadata update",
  },
  "GET /api/v1/files/{p}/versions": {
    replacement: "GET /api/v1/workspace-files/{p}/versions",
    evidence:
      "Go `internal/router/router.go:450` (`fileCommitHandler.GetFileVersionHistory`) -> Python `api/apps/restful_apis/file_commit_api.py:370` (`get_file_version_history`), described in source as `shared across all entity types`",
  },

  // Admin and system routes the REST rewrite renamed rather than dropped. Each
  // Python target was read to confirm it does the same job, not merely that a
  // similar path exists.
  "GET /api/v1/admin/config/log": {
    replacement: "GET /api/v1/admin/log_levels",
    evidence:
      "Go `internal/admin/router.go:91` (`handler.GetLogLevel`) -> Python `admin/server/routes.py:660` (`get_logger_levels`)",
  },
  "PUT /api/v1/admin/config/log": {
    replacement: "PUT /api/v1/admin/log_levels",
    evidence:
      "Go `internal/admin/router.go:92` (`handler.SetLogLevel`) -> Python `admin/server/routes.py:672` (`set_logger_level`)",
  },
  "GET /api/v1/admin/variables/{p}": {
    replacement: "GET /api/v1/admin/variables",
    evidence:
      "Go `internal/admin/router.go:86` (`handler.ShowVariable`) -> Python `admin/server/routes.py:442` `get_variable`, which lists when the request has no body and returns a single variable when the body carries `var_name` (:446-457); the name moved from the path into the payload",
  },
  "GET /api/v1/system/variables": {
    replacement: "GET /api/v1/admin/variables",
    evidence:
      "Go `internal/router/router.go:665` (`systemHandler.ListVariables`) -> Python `admin/server/routes.py:442`; runtime variables moved from the API service to the admin service in the rewrite",
  },
  "PUT /api/v1/system/variables": {
    replacement: "PUT /api/v1/admin/variables",
    evidence:
      "Go `internal/router/router.go:666` (`systemHandler.SetVariable`) -> Python `admin/server/routes.py:420` (`set_variable`)",
  },
  "GET /api/v1/system/variables/{p}": {
    replacement: "GET /api/v1/admin/variables",
    evidence:
      "Go `internal/router/router.go:667` (`systemHandler.ShowVariable`) -> Python `admin/server/routes.py:442`, which takes `var_name` in the body rather than the path",
  },
  "GET /api/v1/system/environments": {
    replacement: "GET /api/v1/admin/environments",
    evidence:
      "Go `internal/router/router.go:670` (`systemHandler.ListEnvironments`) -> Python `admin/server/routes.py:478`",
  },
  // `system/keys` became `system/tokens`: same resource (the caller's own API
  // keys), same three operations, same service.
  "GET /api/v1/system/keys": {
    replacement: "GET /api/v1/system/tokens",
    evidence:
      "Go `internal/router/router.go:685` (`systemHandler.ListAPIKeys`) -> Python `api/apps/restful_apis/system_api.py:242` (`token_list`)",
  },
  "POST /api/v1/system/keys": {
    replacement: "POST /api/v1/system/tokens",
    evidence:
      "Go `internal/router/router.go:687` (`systemHandler.CreateKey`) -> Python `api/apps/restful_apis/system_api.py:290` (`new_token`)",
  },
  "DELETE /api/v1/system/keys/{p}": {
    replacement: "DELETE /api/v1/system/tokens/{p}",
    evidence:
      "Go `internal/router/router.go:689` (`systemHandler.DeleteKey`) -> Python `api/apps/restful_apis/system_api.py:340` (`rm`)",
  },

  // The memory-message trio differs only in how the composite key is spelled in
  // the route pattern, not on the wire. Go declares one segment,
  // `:memory_message`, and splits it itself: `parseMemoryMessagePath` requires
  // exactly `memory_id:message_id` (`internal/handler/memory.go:640-652`).
  // Python declares the same segment as two patterns joined by a literal colon,
  // `<memory_id>:<message_id>`. A caller sends the identical URL either way, so
  // the canonicalizer's `{p}` vs `{p}:{p}` difference is a notation artifact.
  // The canonical forms differ (`{p}` against `{p}{p}`) because canonicalPath
  // erases `<...>` before the colon rule runs, so Python's two-pattern segment
  // collapses to two adjacent placeholders. That is why the automatic test misses
  // this pair even though the URLs are byte-identical.
  "DELETE /api/v1/messages/{p}": {
    replacement: "DELETE /api/v1/messages/{p}{p}",
    evidence:
      "Go `internal/router/router.go:517` (`memoryHandler.ForgetMessage`) -> Python `api/apps/restful_apis/memory_api.py:234`; both take `memory_id:message_id` in one path segment",
  },
  "PUT /api/v1/messages/{p}": {
    replacement: "PUT /api/v1/messages/{p}{p}",
    evidence:
      "Go `internal/router/router.go:518` (`memoryHandler.UpdateMessage`) -> Python `api/apps/restful_apis/memory_api.py:248`",
  },
  "GET /api/v1/messages/{p}/content": {
    replacement: "GET /api/v1/messages/{p}{p}/content",
    evidence:
      "Go `internal/router/router.go:519` (`memoryHandler.GetMessageContent`) -> Python `api/apps/restful_apis/memory_api.py:320`",
  },

  // Infrastructure probes. Recorded as replacements rather than losses because
  // both handler bodies were read and do the same job; the response *shape*
  // differs, which matters to whoever writes the probe config and is called out
  // in the evidence, but no capability disappears.
  "GET /health": {
    replacement: "GET /api/v1/system/ping",
    evidence:
      'Go `internal/router/router.go:154` -> `internal/handler/system.go:52`, a static `{"status":"ok"}` with no dependency check; Python\'s unauthenticated static liveness probe is `api/apps/restful_apis/system_api.py:36`, which returns `"pong"`. Same capability, different body — probe configs must not assert on the payload',
  },
  "POST /v1/user/setting/password": {
    replacement: "PATCH /api/v1/users/me",
    evidence:
      "Go `internal/router/router.go:244` (`userHandler.ChangePassword`) -> Python `api/apps/restful_apis/user_api.py:302` `setting_user`, which changes the password when the body carries `password` + `new_password` (:335-347); self-service password change folded into the profile PATCH rather than keeping its own route",
  },
};

// The file-commit family is the one place where a *whole* prefix was renamed.
// Go mounts the same eight handlers three times — `/folders/:folder_id/…`
// ("takes folder_id directly", `internal/router/router.go:460`), the
// `/workspace/:folder_id/…` alias ("workspace_id == folder_id", `:474`), and
// `/datasets/…`. Python HEAD mounts two: `/datasets/<entity_id>` with
// `resolver_type="datasets"`, and `/workspaces/<entity_id>` with no resolver
// (`file_commit_api.py:364-365`).
//
// The equality that matters is the parameter's meaning, not the prefix spelling.
// With `resolver_type=None` the handler returns `entity_id  # already a
// folder_id` (`file_commit_api.py:100-101`), so Python's `/workspaces/<id>` and
// both closed Go prefixes take the identical value. Sixteen closed routes
// therefore collapse onto the same eight live ones — a rename, not a gap.
//
// Recorded as a loop rather than sixteen literals because the suffixes are one
// verified list: hand-copying it twice invites the transcription error that a
// single list cannot make. Each replacement is still re-checked for reachability
// by the validator below.
for (const [suffix, method, pyLine, goFolders, goWorkspace] of [
  ["/commits", "POST", 108, 464, 477],
  ["/commits", "GET", 141, 465, 478],
  ["/commits/diff", "GET", 281, 466, 479],
  ["/commits/{p}", "GET", 200, 467, 480],
  ["/commits/{p}/files", "GET", 250, 468, 481],
  ["/commits/{p}/tree", "GET", 313, 469, 482],
  ["/commits/{p}/files/{p}/content", "GET", 329, 470, 483],
  ["/changes", "GET", 302, 471, 484],
]) {
  const replacement = `${method} /api/v1/workspaces/{p}${suffix}`;
  for (const [prefix, goLine, kind] of [
    ["folders", goFolders, "takes folder_id directly"],
    [
      "workspace",
      goWorkspace,
      "alias for /folders/, workspace_id == folder_id",
    ],
  ]) {
    VERIFIED_REPLACEMENTS[`${method} /api/v1/${prefix}/{p}${suffix}`] = {
      replacement,
      evidence:
        `Go \`internal/router/router.go:${goLine}\` (${kind}) -> Python ` +
        `\`api/apps/restful_apis/file_commit_api.py:${pyLine}\`, mounted at ` +
        "`/workspaces/<entity_id>` with no resolver, so `entity_id` is the folder_id",
    };
  }
}

// Routes upstream itself marks as internal to the Go process. Their closure
// removes no user-facing capability, so they are recorded as internal rather
// than as a gap. The marker is a source comment, not our inference.
const GO_INTERNAL_ONLY = new Set([
  "POST /api/v1/tenant/chunk_store",
  "DELETE /api/v1/tenant/chunk_store",
  "POST /api/v1/tenant/metadata_store",
  "DELETE /api/v1/tenant/metadata_store",
  "POST /api/v1/document/delete_meta",
  "POST /api/v1/chunk/update",
]);

// v0.26.4 has no registered Go route whose handler is a not-implemented stub.
// Keep the version-scoped map so the validator continues to fail loudly if a
// future pinned ref adds one and the inventory rules need explicit evidence.
const GO_NOT_IMPLEMENTED = new Map();

// ---------------------------------------------------------------------------
// Test evidence. Fixture interactions are matched back to routes by canonical
// method+path, so evidence is discovered rather than asserted.
// ---------------------------------------------------------------------------

const SMOKE_EVIDENCE = {
  "GET /api/v1/system/ping":
    "smoke: 127.0.0.1:9380 -> 200 (`runtime-disabled.md`)",
  "GET /api/v1/admin/ping":
    "smoke: 127.0.0.1:9381 -> 200 (`runtime-disabled.md`)",
};

const PHASE_IMPLEMENTATION_EVIDENCE = {
  "python-api|GET /api/v1/system/config": {
    status: "implemented",
    uiPath: "Login → runtime registration/password-login capability probe",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#getPlatformAuthCapabilities`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/platform-auth-form.test.tsx`",
    ],
  },
  "python-api|POST /api/v1/auth/login": {
    status: "implemented",
    uiPath: "Login → email and password → Giriş yap",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#loginPlatformUser`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/auth-crypto.test.ts`",
      "`src/integrations/platform-backend/__tests__/platform-auth-form.test.tsx`",
    ],
  },
  "python-api|POST /api/v1/auth/logout": {
    status: "implemented",
    uiPath: "Account menu → Logout",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#logoutPlatformUser`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
    ],
  },
  "python-api|POST /api/v1/auth/password/forgot/captcha": {
    status: "implemented",
    uiPath: "Login → Parolamı unuttum → Güvenlik kodunu getir",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#requestForgotPasswordCaptcha`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
    ],
  },
  "python-api|POST /api/v1/auth/password/forgot/otp": {
    status: "implemented",
    uiPath: "Login → Parolamı unuttum → Doğrulama kodu gönder",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#sendForgotPasswordOtp`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
    ],
  },
  "python-api|POST /api/v1/auth/password/forgot/otp/verify": {
    status: "implemented",
    uiPath: "Login → Parolamı unuttum → Kodu doğrula",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#verifyForgotPasswordOtp`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
    ],
  },
  "python-api|POST /api/v1/auth/password/reset": {
    status: "implemented",
    uiPath: "Login → Parolamı unuttum → Parolayı yenile",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#resetForgottenPlatformPassword`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/auth-crypto.test.ts`",
    ],
  },
  "go-api|GET /api/v1/auth/login/channels": {
    status: "implemented",
    uiPath: "Login → runtime-probed Kurumsal giriş buttons",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#getPlatformAuthCapabilities`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/platform-auth-form.test.tsx`",
    ],
  },
  "go-api|GET /api/v1/auth/login/{p}": {
    status: "implemented",
    uiPath: "Login → returned OAuth channel → provider redirect",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#getPlatformOAuthLoginUrl`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/oauth.test.ts`",
      "`src/integrations/platform-backend/__tests__/platform-auth-form.test.tsx`",
    ],
  },
  "go-api|GET /api/v1/auth/oauth/{p}/callback": {
    status: "implemented",
    uiPath:
      "OAuth provider → backend callback → fixed app root → /chat or /login",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#consumePlatformOAuthRedirect`",
    evidence: ["`src/integrations/platform-backend/__tests__/oauth.test.ts`"],
  },
  "python-api|POST /api/v1/users": {
    status: "implemented",
    uiPath: "Login → Hesap oluştur",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#registerPlatformUser`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/platform-auth-errors.test.ts`",
      "`src/integrations/platform-backend/__tests__/platform-auth-form.test.tsx`",
    ],
  },
  "python-api|GET /api/v1/users/me": {
    status: "implemented",
    uiPath:
      "Protected route hydration and Settings → Profile → profile identity",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#getCurrentPlatformUser`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/auth-guards.test.ts`",
      "`src/features/settings/tabs/profile-tab.test.tsx`",
      "`src/features/profile/hooks/use-platform-profile-sync.test.tsx`",
      "`src/features/profile/components/profile-personalization-panel.test.tsx`",
    ],
  },
  "python-api|PATCH /api/v1/users/me": {
    status: "implemented",
    uiPath:
      "Settings → Profile → profile identity / Settings → General → Change password",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#updatePlatformProfile,changePlatformPassword`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/auth-crypto.test.ts`",
      "`src/features/settings/tabs/profile-tab.test.tsx`",
      "`src/features/profile/components/profile-personalization-panel.test.tsx`",
    ],
  },
  "python-api|GET /api/v1/users/me/models": {
    status: "contract-verified",
    uiPath: "—",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#getCurrentPlatformTenantModels`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/features/profile/components/profile-personalization-panel.test.tsx` (Profile exclusion)",
    ],
  },
  "python-api|PATCH /api/v1/users/me/models": {
    status: "contract-verified",
    uiPath: "—",
    typedService:
      "`src/integrations/platform-backend/auth-api.ts#updateCurrentPlatformTenantModels`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts`",
      "`src/features/profile/components/profile-personalization-panel.test.tsx` (Profile exclusion)",
    ],
  },
  "go-api|GET /v1/user/info": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts` (canonical equivalent)",
    ],
  },
  "go-api|GET /v1/user/logout": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`docs/rag-platform/phase-2-auth-contract.md` (legacy compatibility classification)",
    ],
  },
  "go-api|POST /v1/user/setting": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts` (canonical equivalent)",
    ],
  },
  "go-api|POST /v1/user/setting/password": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts` (canonical equivalent)",
    ],
  },
  "go-api|GET /v1/user/tenant_info": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts` (canonical equivalent)",
    ],
  },
  "go-api|POST /v1/user/set_tenant_info": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/auth-api.test.ts` (canonical equivalent)",
    ],
  },
  "python-api|GET /api/v1/system/ping": {
    status: "contract-verified",
    uiPath: "—",
    typedService:
      "`src/integrations/platform-backend/system-api.ts#getSystemPing`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/system-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/connection-store.test.ts`",
    ],
  },
  "python-api|GET /api/v1/system/version": {
    status: "contract-verified",
    uiPath: "—",
    typedService:
      "`src/integrations/platform-backend/system-api.ts#getSystemVersion`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/system-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/connection-store.test.ts`",
    ],
  },
  "python-api|GET /api/v1/system/healthz": {
    status: "contract-verified",
    uiPath: "—",
    typedService:
      "`src/integrations/platform-backend/system-api.ts#getSystemHealth`",
    evidence: [
      "`src/integrations/platform-backend/__tests__/system-api.test.ts`",
      "`src/integrations/platform-backend/__tests__/connection-store.test.ts`",
    ],
  },
  "go-api|GET /health": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "direct smoke: 127.0.0.1:9384/health -> 200",
      "`src/integrations/platform-backend/__tests__/system-api.test.ts` (canonical health service only)",
    ],
  },
  "python-api|GET /v1/system/healthz": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/system-api.test.ts` (compatibility contract)",
    ],
  },
  "go-api|GET /v1/system/configs": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/system-api.test.ts` (negative export/security contract)",
    ],
  },
  "go-api|GET /api/v1/system/configs": {
    status: "contract-verified",
    uiPath: "—",
    typedService: null,
    evidence: [
      "`src/integrations/platform-backend/__tests__/system-api.test.ts` (negative export/security contract)",
    ],
  },
};

const PHASE3_TEST_EVIDENCE = [
  "`src/integrations/platform-backend/__tests__/model-api.test.ts`",
  "`src/integrations/platform-backend/__tests__/model-readiness.test.ts`",
  "`src/features/chat/chat-providers-dialog.test.tsx`",
  "`src/features/settings/tabs/connections-tab.test.tsx`",
  "`src/features/settings/components/platform-model-tools.test.tsx`",
  "`src/features/chat/components/platform-chat-model-selector.test.tsx`",
];

const PHASE3_IMPLEMENTATION = {
  "GET /api/v1/users/me/models": [
    "implemented",
    "Settings → Connections → permission/readiness; Chat → Select model → current chat default",
    "auth-api.ts#getCurrentPlatformTenantModels",
  ],
  "PATCH /api/v1/users/me/models": [
    "contract-verified",
    "—",
    "auth-api.ts#updateCurrentPlatformTenantModels",
  ],
  "GET /api/v1/providers": [
    "implemented",
    "Settings → Connections → connection list / Add connection provider selector",
    "model-api.ts#listAvailableProviders,listConfiguredProviders",
  ],
  "PUT /api/v1/providers": [
    "implemented",
    "Settings → Connections → Add connection → provider instance ekle",
    "model-api.ts#addProvider",
  ],
  "GET /api/v1/providers/{p}": [
    "implemented",
    "Settings → Connections → configured provider → detail",
    "model-api.ts#getProvider",
  ],
  "DELETE /api/v1/providers/{p}": [
    "implemented",
    "Settings → Connections → configured provider → delete",
    "model-api.ts#deleteProvider",
  ],
  "GET /api/v1/providers/{p}/models": [
    "implemented",
    "Settings → Connections → configured provider → Models → provider catalog",
    "model-api.ts#listProviderModels",
  ],
  "GET /api/v1/providers/{p}/models/{p}": [
    "implemented",
    "Settings → Connections → configured provider → Models → catalog model detail",
    "model-api.ts#getProviderModel",
  ],
  "POST /api/v1/providers/{p}/connection": [
    "implemented",
    "Settings → Connections → Add connection → test before save",
    "model-api.ts#testProviderConnection",
  ],
  "GET /api/v1/providers/{p}/instances": [
    "implemented",
    "Settings → Connections → configured provider rows",
    "model-api.ts#listProviderInstances",
  ],
  "POST /api/v1/providers/{p}/instances": [
    "implemented",
    "Settings → Connections → Add connection → provider instance ekle",
    "model-api.ts#createProviderInstance",
  ],
  "DELETE /api/v1/providers/{p}/instances": [
    "implemented",
    "Settings → Connections → configured provider → delete",
    "model-api.ts#deleteProviderInstances",
  ],
  "GET /api/v1/providers/{p}/instances/{p}": [
    "implemented",
    "Settings → Connections → configured provider → detail",
    "model-api.ts#getProviderInstance",
  ],
  "PUT /api/v1/providers/{p}/instances/{p}": [
    "implemented",
    "Settings → Connections → configured provider → edit",
    "model-api.ts#updateProviderInstance",
  ],
  "GET /api/v1/providers/{p}/instances/{p}/connection": [
    "implemented",
    "Settings → Connections → configured provider → test connection",
    "model-api.ts#testProviderInstanceConnection",
  ],
  "GET /api/v1/providers/{p}/instances/{p}/balance": [
    "contract-verified",
    "—",
    "model-api.ts#getProviderInstanceBalance",
  ],
  "GET /api/v1/providers/{p}/instances/{p}/tasks": [
    "contract-verified",
    "—",
    "model-api.ts#listProviderTasks",
  ],
  "GET /api/v1/providers/{p}/instances/{p}/tasks/{p}": [
    "contract-verified",
    "—",
    "model-api.ts#getProviderTask",
  ],
  "GET /api/v1/providers/{p}/instances/{p}/models": [
    "implemented",
    "Settings → Connections → configured provider → Models → saved + live supported catalog",
    "model-api.ts#listInstanceModels,listSupportedInstanceModels",
  ],
  "POST /api/v1/providers/{p}/instances/{p}/models": [
    "implemented",
    "Settings → Connections → configured provider → Models → add",
    "model-api.ts#addInstanceModel",
  ],
  "PATCH /api/v1/providers/{p}/instances/{p}/models/{p}": [
    "implemented",
    "Settings → Connections → configured provider → Models → enable/disable",
    "model-api.ts#updateInstanceModel",
  ],
  "DELETE /api/v1/providers/{p}/instances/{p}/models": [
    "implemented",
    "Settings → Connections → configured provider → Models → delete",
    "model-api.ts#deleteInstanceModels",
  ],
  "PUT /api/v1/providers/{p}/instances/{p}/models": [
    "contract-verified",
    "—",
    "model-api.ts#updateInstanceModel (atomic canonical equivalent)",
  ],
  "POST /api/v1/providers/{p}/instances/{p}/models/{p}": [
    "contract-verified",
    "—",
    "model-api.ts#chatToModel (canonical utility equivalent)",
  ],
  "GET /api/v1/models": [
    "implemented",
    "Settings → Connections → configured provider → Models; Chat → Select model → active chat models",
    "model-api.ts#listTenantModels",
  ],
  "PATCH /api/v1/models": [
    "contract-verified",
    "—",
    "model-api.ts#setDefaultModel (atomic canonical equivalent)",
  ],
  "GET /api/v1/models/default": [
    "implemented",
    "Settings → Connections → configured provider → Model defaults; Chat → Select model → current chat default",
    "model-api.ts#getDefaultModels",
  ],
  "PATCH /api/v1/models/default": [
    "implemented",
    "Settings → Connections → capability default; Chat → Select model → choose chat default",
    "model-api.ts#setDefaultModel",
  ],
  "POST /api/v1/chat/to_model": [
    "implemented",
    "Settings → Connections → configured provider → Yetkili model araçları → Chat to model",
    "model-api.ts#chatToModel",
  ],
  "POST /api/v1/embeddings": [
    "implemented",
    "Settings → Connections → configured provider → Yetkili model araçları → Embedding",
    "model-api.ts#createEmbeddings",
  ],
  "POST /api/v1/rerank": [
    "implemented",
    "Settings → Connections → configured provider → Yetkili model araçları → Rerank",
    "model-api.ts#rerankDocuments",
  ],
  "POST /api/v1/audio/transcriptions": [
    "implemented",
    "Settings → Connections → configured provider → Yetkili model araçları → Audio transcription",
    "model-api.ts#transcribeAudio",
  ],
  "POST /api/v1/audio/speech": [
    "implemented",
    "Settings → Connections → configured provider → Yetkili model araçları → Audio speech",
    "model-api.ts#synthesizeSpeech",
  ],
  "POST /api/v1/file/ocr": [
    "implemented",
    "Settings → Connections → configured provider → Yetkili model araçları → OCR",
    "model-api.ts#ocrFile",
  ],
  "POST /api/v1/file/parse": [
    "implemented",
    "Settings → Connections → configured provider → Yetkili model araçları → File parse",
    "model-api.ts#parseFile",
  ],
};

for (const [key, [status, uiPath, service]] of Object.entries(
  PHASE3_IMPLEMENTATION,
)) {
  for (const backendService of ["python-api", "go-api"]) {
    const inventoryRoute = inventory.routes
      .flatMap((route) => [route, ...(route.alternates ?? [])])
      .find(
        (route) =>
          route.service === backendService &&
          `${route.method} ${canonicalPath(route.path)}` === key &&
          route.runtime_enabled === true,
      );
    if (!inventoryRoute) continue;
    PHASE_IMPLEMENTATION_EVIDENCE[`${backendService}|${key}`] = {
      status,
      uiPath,
      typedService: `\`src/integrations/platform-backend/${service}\``,
      evidence: PHASE3_TEST_EVIDENCE,
    };
  }
}

const PHASE4_TEST_EVIDENCE = [
  "src/integrations/platform-backend/__tests__/dataset-api.test.ts",
  "src/features/rag/api/platform-dataset-adapter.test.ts",
  "src/features/rag/components/knowledge-base-dialog.test.tsx",
];

Object.assign(PHASE_IMPLEMENTATION_EVIDENCE, {
  "python-api|GET /api/v1/datasets": {
    status: "implemented",
    uiPath:
      "Chat composer → RAG → Manage knowledge bases → paginated/searchable list",
    typedService:
      "src/integrations/platform-backend/dataset-api.ts#listPlatformDatasets",
    evidence: PHASE4_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/datasets": {
    status: "implemented",
    uiPath:
      "Chat composer → RAG → Manage knowledge bases → Yeni → Oluştur",
    typedService:
      "src/integrations/platform-backend/dataset-api.ts#createPlatformDataset",
    evidence: PHASE4_TEST_EVIDENCE,
  },
  "python-api|DELETE /api/v1/datasets": {
    status: "implemented",
    uiPath:
      "Chat composer → RAG → Manage knowledge bases → Delete → confirmed Sil",
    typedService:
      "src/integrations/platform-backend/dataset-api.ts#deletePlatformDatasets",
    evidence: PHASE4_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/datasets/{p}": {
    status: "implemented",
    uiPath:
      "Chat composer → RAG → Manage knowledge bases → dataset edit action",
    typedService:
      "src/integrations/platform-backend/dataset-api.ts#getPlatformDataset",
    evidence: PHASE4_TEST_EVIDENCE,
  },
  "go-api|PUT /api/v1/datasets/{p}": {
    status: "implemented",
    uiPath:
      "Chat composer → RAG → Manage knowledge bases → dataset edit → Değişiklikleri kaydet",
    typedService:
      "src/integrations/platform-backend/dataset-api.ts#updatePlatformDataset",
    evidence: PHASE4_TEST_EVIDENCE,
  },
});

const PHASE5_TEST_EVIDENCE = [
  "src/integrations/platform-backend/__tests__/document-api.test.ts",
  "src/features/documents/use-document-library.test.tsx",
];

Object.assign(PHASE_IMPLEMENTATION_EVIDENCE, {
  "go-api|GET /api/v1/datasets/{p}/documents": {
    status: "implemented",
    uiPath: "Sidebar → Documents → Dataset documents → document table",
    typedService:
      "src/integrations/platform-backend/document-api.ts#listDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/datasets/{p}/documents": {
    status: "implemented",
    uiPath: "Sidebar → Documents → dropzone / Dosya seç",
    typedService:
      "src/integrations/platform-backend/document-api.ts#uploadDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/datasets/{p}/documents": {
    status: "implemented",
    uiPath: "Sidebar → Documents → dropzone / Dosya seç",
    typedService:
      "src/integrations/platform-backend/document-api.ts#uploadDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/datasets/{p}/documents/{p}": {
    status: "implemented",
    uiPath: "Sidebar → Documents → document row → İndir",
    typedService:
      "src/integrations/platform-backend/document-api.ts#downloadDatasetDocument",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|PATCH /api/v1/datasets/{p}/documents/{p}": {
    status: "implemented",
    uiPath: "Sidebar → Documents → document row → Yeniden adlandır",
    typedService:
      "src/integrations/platform-backend/document-api.ts#updateDatasetDocument",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/datasets/{p}/documents/parse": {
    status: "implemented",
    uiPath: "Sidebar → Documents → İşle / Yeniden işle",
    typedService:
      "src/integrations/platform-backend/document-api.ts#parseDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/datasets/{p}/documents/parse": {
    status: "implemented",
    uiPath: "Sidebar → Documents → İşle / Yeniden işle",
    typedService:
      "src/integrations/platform-backend/document-api.ts#parseDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/datasets/{p}/documents/stop": {
    status: "implemented",
    uiPath: "Sidebar → Documents → running document → Durdur",
    typedService:
      "src/integrations/platform-backend/document-api.ts#stopDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|DELETE /api/v1/datasets/{p}/documents": {
    status: "implemented",
    uiPath: "Sidebar → Documents → document selection → Sil → confirm",
    typedService:
      "src/integrations/platform-backend/document-api.ts#deleteDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|GET /api/v1/documents/{p}/preview": {
    status: "implemented",
    uiPath: "Sidebar → Documents → document name / Önizle",
    typedService:
      "src/integrations/platform-backend/document-api.ts#fetchDocumentPreview",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/documents/images/{p}": {
    status: "implemented",
    uiPath: "Sidebar → Documents → Medya → authorized thumbnail image",
    typedService:
      "src/integrations/platform-backend/document-api.ts#fetchDocumentImage",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|GET /api/v1/thumbnails": {
    status: "implemented",
    uiPath: "Sidebar → Documents → document row → Medya",
    typedService:
      "src/integrations/platform-backend/document-api.ts#listDocumentThumbnails",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/documents/artifact/{p}": {
    status: "implemented",
    uiPath: "Sidebar → Documents → Medya → artifact filename → Artifact aç",
    typedService:
      "src/integrations/platform-backend/document-api.ts#fetchDocumentArtifact",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/documents/upload": {
    status: "implemented",
    uiPath: "Sidebar → Documents → Genel belgeler → Bağımsız dosya inceleme",
    typedService:
      "src/integrations/platform-backend/document-api.ts#inspectDocumentUploads",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/documents/{p}": {
    status: "runtime-disabled",
    uiPath: "Sidebar → Documents → Genel belgeler → explicit security-disabled notice",
    typedService:
      "src/integrations/platform-backend/document-api.ts#getGenericDocument",
    evidence: [
      "docs/rag-platform/runtime-disabled.md (Phase 5 functional security gap)",
      "src/features/documents/document-library-page.tsx (disabled-state UI)",
      ...PHASE5_TEST_EVIDENCE,
    ],
  },
  "go-api|PUT /api/v1/documents/{p}": {
    status: "implemented",
    uiPath: "Sidebar → Documents → Genel belgeler → Belge kimliğiyle yönetim → Kaydet",
    typedService:
      "src/integrations/platform-backend/document-api.ts#updateGenericDocument",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|DELETE /api/v1/documents/{p}": {
    status: "implemented",
    uiPath: "Sidebar → Documents → Genel belgeler → Belge kimliğiyle yönetim → Sil",
    typedService:
      "src/integrations/platform-backend/document-api.ts#deleteGenericDocument",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/documents/ingest": {
    status: "implemented",
    uiPath: "Sidebar → Documents → Genel belgeler → Belge kimliğiyle yönetim → Ingest",
    typedService:
      "src/integrations/platform-backend/document-api.ts#ingestGenericDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/documents": {
    status: "runtime-disabled",
    uiPath: "Sidebar → Documents → Genel belgeler → explicit runtime-disabled notice",
    typedService:
      "src/integrations/platform-backend/document-api.ts#listGenericDocuments",
    evidence: [
      "docs/rag-platform/runtime-disabled.md (Phase 5 functional runtime gap)",
      "src/features/documents/document-library-page.tsx (disabled-state UI)",
    ],
  },
  "go-api|POST /api/v1/documents": {
    status: "contract-verified",
    uiPath: "— (trusted service contract; caller-supplied created_by is not exposed)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#createGenericDocument",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/datasets/ingestion/tasks": {
    status: "runtime-disabled",
    uiPath: "— (browser GET-body contract is unusable; dataset stop route is used)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#listDatasetIngestionTasks",
    evidence: [
      "docs/rag-platform/runtime-disabled.md (Phase 5 functional runtime gap)",
      ...PHASE5_TEST_EVIDENCE,
    ],
  },
  "go-api|PUT /api/v1/datasets/ingestion/tasks": {
    status: "contract-verified",
    uiPath: "— (trusted task-aware client)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#stopDatasetIngestionTasks",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|DELETE /api/v1/datasets/ingestion/tasks": {
    status: "contract-verified",
    uiPath: "— (trusted task-aware client)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#removeDatasetIngestionTasks",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/tasks/{p}/cancel": {
    status: "contract-verified",
    uiPath: "— (generic ingest returns no task id)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#cancelPlatformTask",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|PATCH /api/v1/tasks/{p}": {
    status: "contract-verified",
    uiPath: "— (generic ingest returns no task id)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#stopPlatformTask",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|PUT /api/v1/datasets/{p}/documents/{p}": {
    status: "contract-verified",
    uiPath: "— (legacy update alias; UI uses canonical PATCH)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#updateDatasetDocument",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/document/delete_meta": {
    status: "contract-verified",
    uiPath: "— (Go-internal metadata compatibility contract)",
    typedService: null,
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/document/list": {
    status: "contract-verified",
    uiPath: "— (legacy flat-list alias; UI uses dataset-scoped list)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#listDatasetDocuments",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/document/metadata/summary": {
    status: "contract-verified",
    uiPath: "— (metadata belongs to Phase 10)",
    typedService: null,
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/document/set_meta": {
    status: "contract-verified",
    uiPath: "— (metadata belongs to Phase 10)",
    typedService: null,
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|GET /api/v1/document/download/{p}": {
    status: "contract-verified",
    uiPath: "— (deprecated attachment alias; UI uses dataset download)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#downloadDatasetDocument",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|GET /api/v1/document/get/{p}": {
    status: "contract-verified",
    uiPath: "— (deprecated preview alias; UI uses /documents/{id}/preview)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#fetchDocumentPreview",
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|GET /v1/document/download/{p}": {
    status: "contract-verified",
    uiPath: "— (deprecated attachment alias)",
    typedService: null,
    evidence: PHASE5_TEST_EVIDENCE,
  },
  "python-api|POST /v1/document/upload_info": {
    status: "contract-verified",
    uiPath: "— (legacy alias; UI uses /api/v1/documents/upload)",
    typedService:
      "src/integrations/platform-backend/document-api.ts#inspectDocumentUploads",
    evidence: PHASE5_TEST_EVIDENCE,
  },
});

const PHASE6_TEST_EVIDENCE = [
  "src/integrations/platform-backend/__tests__/chunk-api.test.ts",
  "src/features/documents/dataset-quality-workspace.test.tsx",
  "src/features/documents/document-library-page.test.tsx",
  "src/features/documents/document-asset-dialog.test.tsx",
];
const PHASE6_CHUNK_UI =
  "Sidebar → Documents → Dataset belgeleri → Chunks";
const PHASE6_RETRIEVAL_UI =
  "Sidebar → Documents → Dataset belgeleri → Retrieval";

Object.assign(PHASE_IMPLEMENTATION_EVIDENCE, {
  "go-api|GET /api/v1/datasets/{p}/documents/{p}/chunks": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → paginated/virtualized list`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#listDocumentChunks",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/datasets/{p}/documents/{p}/chunks": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → Yeni chunk → Kaydet`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#createDocumentChunk",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|PATCH /api/v1/datasets/{p}/documents/{p}/chunks": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → selection → Etkinleştir/Kapat`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#setDocumentChunksEnabled",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|DELETE /api/v1/datasets/{p}/documents/{p}/chunks": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → selection → Sil → confirmation`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#deleteDocumentChunks",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|GET /api/v1/datasets/{p}/documents/{p}/chunks/{p}": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → chunk → Düzenle`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#getDocumentChunk",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|PATCH /api/v1/datasets/{p}/documents/{p}/chunks/{p}": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → chunk → Düzenle → Kaydet`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#updateDocumentChunk",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "python-api|POST /api/v1/retrieval": {
    status: "implemented",
    uiPath: `${PHASE6_RETRIEVAL_UI} → Retrieval çalıştır`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#retrievePlatformChunks",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "python-api|GET /api/v1/datasets/{p}/documents/{p}/structure/graph": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → Yapı grafiği`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#getDocumentStructureGraph",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "python-api|DELETE /api/v1/datasets/{p}/documents/{p}/structure/graph": {
    status: "implemented",
    uiPath: `${PHASE6_CHUNK_UI} → Yapı grafiği → Grafiği sil → confirmation`,
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#deleteDocumentStructureGraph",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/chunk/list": {
    status: "contract-verified",
    uiPath: "— (legacy list alias; UI uses canonical scoped GET)",
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#listDocumentChunksCompatibility",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/chunk/update": {
    status: "contract-verified",
    uiPath: "— (Go internal-only route; no browser export)",
    typedService: null,
    evidence: [
      "docs/rag-platform/route-inventory.md (upstream internal-only note)",
      ...PHASE6_TEST_EVIDENCE,
    ],
  },
  "go-api|POST /api/v1/datasets/{p}/chunks": {
    status: "contract-verified",
    uiPath: "— (legacy parse alias; UI uses /documents/parse)",
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#parseDatasetChunksCompatibility",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "python-api|DELETE /api/v1/datasets/{p}/chunks": {
    status: "contract-verified",
    uiPath: "— (legacy stop alias; UI uses /documents/stop)",
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#stopDatasetChunksCompatibility",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "go-api|POST /api/v1/datasets/{p}/search": {
    status: "contract-verified",
    uiPath: "— (dataset-scoped alias; UI uses canonical /retrieval)",
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#searchDatasetChunksCompatibility",
    evidence: PHASE6_TEST_EVIDENCE,
  },
  "python-api|PUT /api/v1/datasets/{p}/documents/{p}/chunks/{p}": {
    status: "contract-verified",
    uiPath: "— (backward-compat alias; UI uses canonical PATCH)",
    typedService:
      "src/integrations/platform-backend/chunk-api.ts#updateDocumentChunkCompatibility",
    evidence: PHASE6_TEST_EVIDENCE,
  },
});

/**
 * Per-route findings verified against the running backend that a reader of the
 * row needs in order to trust it. Keyed by canonical `METHOD path`.
 */
const ROUTE_FINDINGS = {
  // Recorded as a finding rather than a replacement because the protocol is
  // still served while the *route* is not: the MCP service listens on 9382 and
  // nginx never proxies it under any scheme, so the endpoint is unreachable
  // through the published surface even though the capability exists in the
  // deployment. Calling that a rename would hide the reachability gap; calling
  // it a plain loss would overstate it.
  "POST /api/v1/mcp":
    "The MCP protocol itself is not lost: the standalone MCP service serves " +
    "`POST /mcp` (`mcp/server/server.py:783-784`) and `GET /sse` " +
    "(`:748`) on port 9382. That service is `not-proxied` — nginx exposes no " +
    "location for 9382 under any `API_PROXY_SCHEME` — so an MCP client must " +
    "address the container port directly rather than the API surface. What the " +
    "closed Go route removes is the proxied `/api/v1/mcp` entry point and its " +
    "`BetaAuthMiddleware` header-based user resolution, not MCP support.",
  "DELETE /api/v1/datasets/{p}/knowledge_graph":
    "Upstream defect (verified live, HTTP 200 code=100): the shim calls " +
    "`dataset_api.delete_knowledge_graph`, which the route module does not define " +
    "(the implementation lives at `api/apps/services/dataset_api_service.py:537`), " +
    "so every call raises AttributeError before any ownership check. Use the " +
      "forward route `DELETE /api/v1/datasets/{dataset_id}/index?type=graph`.",
  "GET /api/v1/documents":
    "Phase 5 functional runtime gap: the active v0.26.4 Go route reuses ListDocuments, " +
    "which reads c.Param(\"dataset_id\") on the flat /documents group and therefore " +
    "runs dataset ownership against an empty id. Hybrid routes the request to 9384; " +
    "the UI renders an explicit runtime-disabled notice instead of an empty list.",
  "GET /api/v1/documents/{p}":
    "Phase 5 functional security gap: the active v0.26.4 Go handler authenticates " +
    "the session but discards the principal and returns GetDocumentByID directly, " +
    "without dataset ownership/Accessible verification. The typed contract is kept " +
    "under tests, but the frontend does not expose the unsafe read action and renders " +
    "an explicit runtime-disabled explanation. PUT and DELETE independently enforce " +
    "dataset ownership before mutation.",
  "GET /api/v1/datasets/ingestion/tasks":
    "Phase 5 functional runtime gap: the active v0.26.4 handler calls ShouldBindJSON " +
    "for dataset_id on a GET request. Browser Fetch rejects GET bodies, and the handler " +
    "does not read the query parameter. The UI polls dataset document state and uses the " +
    "reachable Python POST /datasets/{dataset_id}/documents/stop contract instead.",
  "GET /api/v1/documents/images/{p}":
    "The active image handler authenticates but does not independently repeat document " +
    "ownership lookup. The UI only derives image ids from the authenticated thumbnail " +
    "response and never accepts an arbitrary image id; this residual backend limitation " +
    "is recorded in the Phase 5 result report.",
  "POST /api/v1/datasets/{p}/documents":
    "The active Go v0.26.4 upload is runtime-incompatible with the deployed SQL " +
    "schema because it inserts document.meta_fields. The generated hybrid proxy " +
    "uses the ownership-checked Python equivalent on 9380; live PDF/TXT/DOCX " +
    "multi-upload passed through that canonical target. The Go implementation is " +
    "classified as a runtime-disabled alternate.",
  "POST /api/v1/datasets/{p}/documents/parse":
    "The generated runtime override selects Python 9380 so canonical document_ids " +
    "jobs are consumed by the deployed Python task executor. The Go alternate is " +
    "runtime-disabled for this deployment; live PDF/TXT/DOCX tasks reached 100%.",
  "POST /api/v1/chunk/update":
    "The active v0.26.4 router labels this route `Internal API only for GO`. " +
    "Its flat registration supplies no dataset_id/document_id path parameters, " +
    "while the handler requires both before mutation. Phase 6 intentionally has " +
    "no browser service export; the canonical scoped PATCH route is the product action.",
};

function collectFixtureEvidence() {
  const evidence = new Map();
  if (!existsSync(FIXTURE_DIR)) return evidence;
  for (const file of readdirSync(FIXTURE_DIR).sort()) {
    if (!file.endsWith(".json")) continue;
    const fixture = JSON.parse(readFileSync(join(FIXTURE_DIR, file), "utf8"));
    for (const interaction of fixture.interactions ?? []) {
      const key = `${interaction.request.method} ${canonicalPath(interaction.request.path)}`;
      if (!evidence.has(key)) evidence.set(key, []);
      evidence.get(key).push(`\`fixtures/${file}\` #${interaction.name}`);
    }
  }
  return evidence;
}

const fixtureEvidence = collectFixtureEvidence();

// ---------------------------------------------------------------------------
// Build one matrix record per inventory record, alternates included: a second
// implementation of the same method+path is a separate contract surface and
// gets its own row rather than being folded away.
// ---------------------------------------------------------------------------

// Two services mount the same handlers under different prefixes: the Go API
// exposes `/v1/file/root_folder` while the Python API serves the identical
// capability at `/api/v1/file/root_folder`. Comparing raw paths would report
// the closed Go route as a lost capability when the feature is in fact live,
// so equivalence is tested on the mount-agnostic remainder of the path.
function mountAgnosticKey(method, canonical) {
  return `${method} ${canonical.replace(/^\/(api\/v1|v1|api)(?=\/)/, "")}`;
}

const reachableByMountAgnosticKey = new Map();
for (const route of inventory.routes) {
  for (const candidate of [route, ...(route.alternates ?? [])]) {
    if (candidate.runtime_enabled !== true) continue;
    const key = mountAgnosticKey(
      candidate.method,
      canonicalPath(candidate.path),
    );
    if (!reachableByMountAgnosticKey.has(key)) {
      reachableByMountAgnosticKey.set(key, candidate);
    }
  }
}

function buildRecord(route, parent) {
  const canonical = canonicalPath(route.path);
  const family = familyOf(route);
  const classification = classifyRecord(route, canonical);
  const phase = phaseOf(route, canonical);
  const key = `${route.method} ${canonical}`;
  const implementation =
    PHASE_IMPLEMENTATION_EVIDENCE[`${route.service}|${key}`];

  const evidence = [];
  if (route.runtime_enabled === false) {
    evidence.push("`runtime-disabled.md` (source + proxy + port probe)");
  }
  if (route.runtime_enabled === null) {
    evidence.push("`route-inventory.md` (not proxied; own-port startup flag)");
  }
  if (route.runtime_enabled === true) {
    if (fixtureEvidence.has(key)) evidence.push(...fixtureEvidence.get(key));
    if (SMOKE_EVIDENCE[key]) evidence.push(SMOKE_EVIDENCE[key]);
  }

  let status;
  if (route.runtime_enabled === false) status = "runtime-disabled";
  else if (route.runtime_enabled === null) status = "not-proxied";
  else if (evidence.length > 0) status = "contract-verified";
  else status = "planned";

  if (route.runtime_enabled === true && implementation) {
    status = implementation.status;
    evidence.push(...implementation.evidence);
  }

  if (evidence.length === 0) {
    evidence.push(phase === null ? "none" : `pending (Faz ${phase})`);
  }

  const isFrontend =
    classification.class === SCREEN || classification.class === ACTION;
  const uiPath =
    route.runtime_enabled === false
      ? "—"
      : (implementation?.uiPath ??
        (isFrontend ? `pending (Faz ${phase})` : "—"));
  const typedService = implementation?.typedService ?? null;

  // A closed route is only a lost capability if nothing reachable serves the
  // same method and path. For a nested alternate that equivalent is its own
  // parent record, which the inventory does not repeat on the child; failing
  // both, a live route under a different mount prefix still serves it.
  const mountEquivalent =
    route.runtime_enabled === false
      ? reachableByMountAgnosticKey.get(
          mountAgnosticKey(route.method, canonical),
        )
      : undefined;
  const renamed =
    route.runtime_enabled === false ? VERIFIED_REPLACEMENTS[key] : undefined;
  const goInternalOnly =
    route.runtime_enabled === false && GO_INTERNAL_ONLY.has(key);
  const notImplemented =
    route.runtime_enabled === false ? GO_NOT_IMPLEMENTED.get(key) : undefined;
  const equivalent =
    route.equivalent_reachable_route ??
    (parent && parent.runtime_enabled === true
      ? { service: parent.service, path: parent.path }
      : mountEquivalent
        ? { service: mountEquivalent.service, path: mountEquivalent.path }
        : renamed
          ? { renamed_to: renamed.replacement }
          : null);

  let justification = classification.justification;
  if (route.runtime_enabled === false) {
    if (goInternalOnly) {
      justification +=
        " Upstream marks this route `// Internal API only for GO` at its registration, so it was never a user-facing capability and its closure removes nothing from the product.";
    } else if (notImplemented) {
      justification +=
        ` Its Go handler ${notImplemented} is a stub whose whole body returns` +
        " `CodeNotImplemented`, so the route answers “not implemented” even when reachable." +
        " Nothing was built on it and nothing is lost by its closure; it is an unbuilt upstream" +
        " feature, not a capability this deployment gave up.";
    } else if (renamed) {
      justification += ` Capability preserved under a different name: \`${renamed.replacement}\` (${renamed.evidence}).`;
    } else if (!equivalent) {
      justification +=
        " No reachable route serves this method and path, so the capability is lost in this deployment.";
    } else if (equivalent.path === route.path) {
      justification += ` Capability preserved: ${equivalent.service} serves the same method and path.`;
    } else {
      justification += ` Capability preserved: ${equivalent.service} serves the same method at \`${equivalent.path}\`.`;
    }
  }
  if (route.notes) {
    justification += ` Upstream note: ${route.notes}.`;
  }
  // Findings attach whatever the runtime state: a closed route's finding is
  // often the very thing that qualifies its loss.
  if (ROUTE_FINDINGS[key]) {
    justification += ` ${ROUTE_FINDINGS[key]}`;
  }
  if (parent) {
    justification += ` Second implementation of \`${parent.method} ${parent.path}\` (${parent.service}).`;
  }

  return {
    method: route.method,
    path: route.path,
    canonical_path: canonical,
    family,
    service: route.service,
    service_port: route.service_port,
    proxy_mode: route.proxy_mode,
    proxy_destination: route.proxy_destination,
    runtime:
      route.runtime_enabled === true
        ? "enabled"
        : route.runtime_enabled === false
          ? "disabled"
          : "not-proxied",
    runtime_disabled_reason: route.runtime_disabled_reason,
    source: route.source,
    class: classification.class,
    class_rule: classification.rule,
    target_phase: phase,
    owner: FAMILY_OWNER[family] ?? "unassigned",
    auth: route.auth,
    auth_role: AUTH_ROLE[route.auth] ?? "unmapped",
    consumer: classification.consumer,
    status,
    ui_path: uiPath,
    typed_service: typedService,
    test_evidence: evidence,
    justification,
    is_alternate: Boolean(parent),
    replaced_by: renamed?.replacement ?? null,
    go_internal_only: goInternalOnly,
    go_not_implemented: Boolean(notImplemented),
    capability_lost:
      route.runtime_enabled === false &&
      !equivalent &&
      !goInternalOnly &&
      !notImplemented,
  };
}

const records = [];
for (const route of inventory.routes) {
  records.push(buildRecord(route, null));
  for (const alternate of route.alternates ?? []) {
    records.push(buildRecord(alternate, route));
  }
}

// ---------------------------------------------------------------------------
// Validation. Runs on every invocation, not only under --check: a generator
// that can emit an unclassified or duplicated record is not a gate.
// ---------------------------------------------------------------------------

const problems = [];

for (const record of records) {
  if (record.class === "unclassified") {
    problems.push(
      `unclassified: ${record.method} ${record.path} (${record.service})`,
    );
  }
  if (record.target_phase === null) {
    problems.push(
      `no target phase: ${record.method} ${record.path} (family ${record.family})`,
    );
  }
  if (record.auth_role === "unmapped") {
    problems.push(
      `unmapped auth role "${record.auth}": ${record.method} ${record.path}`,
    );
  }
  if (record.owner === "unassigned") {
    problems.push(
      `no owner for family "${record.family}": ${record.method} ${record.path}`,
    );
  }
  if (
    record.status === "implemented" &&
    (record.class === SCREEN || record.class === ACTION) &&
    (!record.typed_service ||
      record.ui_path === "—" ||
      record.ui_path.startsWith("pending"))
  ) {
    problems.push(
      `implemented frontend route lacks typed service/UI path: ${record.method} ${record.path} (${record.service})`,
    );
  }
}

const seen = new Map();
for (const record of records) {
  const key = `${record.service}|${record.method}|${record.path}|${record.source}`;
  if (seen.has(key)) problems.push(`duplicate record: ${key}`);
  seen.set(key, record);
}

// Every hand-asserted replacement must still resolve while its old route is
// closed. The owned full-parity proxy can reopen the old Go route; in that case
// the replacement remains useful migration history but no longer suppresses a
// runtime gap and is not an error.
{
  const reachableKeys = new Set();
  const closedKeys = new Set();
  const allKeys = new Set();
  for (const route of inventory.routes) {
    for (const candidate of [route, ...(route.alternates ?? [])]) {
      const key = `${candidate.method} ${canonicalPath(candidate.path)}`;
      allKeys.add(key);
      if (candidate.runtime_enabled === true) reachableKeys.add(key);
      if (candidate.runtime_enabled === false) closedKeys.add(key);
    }
  }
  for (const [closed, { replacement }] of Object.entries(
    VERIFIED_REPLACEMENTS,
  )) {
    if (!allKeys.has(closed)) {
      problems.push(`verified replacement source route disappeared: ${closed}`);
    } else if (
      closedKeys.has(closed) &&
      !reachableKeys.has(closed) &&
      !reachableKeys.has(replacement)
    ) {
      problems.push(
        `verified replacement target is not reachable: ${closed} -> ${replacement}`,
      );
    }
  }
  for (const internal of GO_INTERNAL_ONLY) {
    if (!allKeys.has(internal)) {
      problems.push(`go-internal-only marker route disappeared: ${internal}`);
    }
  }
  for (const stub of GO_NOT_IMPLEMENTED.keys()) {
    if (!allKeys.has(stub)) {
      problems.push(`not-implemented marker route disappeared: ${stub}`);
    }
  }
}

// Inventory <-> matrix parity, both directions.
const inventoryKeys = new Set();
for (const route of inventory.routes) {
  inventoryKeys.add(
    `${route.service}|${route.method}|${route.path}|${route.source}`,
  );
  for (const alternate of route.alternates ?? []) {
    inventoryKeys.add(
      `${alternate.service}|${alternate.method}|${alternate.path}|${alternate.source}`,
    );
  }
}
for (const key of inventoryKeys) {
  if (!seen.has(key)) problems.push(`missing from matrix: ${key}`);
}
for (const key of seen.keys()) {
  if (!inventoryKeys.has(key))
    problems.push(`matrix record not in inventory: ${key}`);
}

// ---------------------------------------------------------------------------
// Totals
// ---------------------------------------------------------------------------

function tally(list, pick) {
  const counts = new Map();
  for (const item of list) {
    const key = pick(item);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

const CLASS_ORDER = [
  SCREEN,
  ACTION,
  API_ONLY,
  CALLBACK,
  INTERNAL,
  UNSUPPORTED,
  "unclassified",
];
const STATUS_ORDER = [
  "implemented",
  "contract-verified",
  "in-progress",
  "planned",
  "runtime-disabled",
  "not-proxied",
];

const byClass = tally(records, (r) => r.class);
const byStatus = tally(records, (r) => r.status);
const byPhase = tally(records, (r) => r.target_phase);
const reachable = records.filter((r) => r.runtime === "enabled");
const byClassReachable = tally(reachable, (r) => r.class);

const totals = {
  records: records.length,
  top_level_routes: inventory.routes.length,
  alternate_implementations: records.filter((r) => r.is_alternate).length,
  reachable: reachable.length,
  unclassified: byClass.get("unclassified") ?? 0,
  capability_lost: records.filter((r) => r.capability_lost).length,
  by_class: Object.fromEntries(
    CLASS_ORDER.filter((c) => byClass.has(c)).map((c) => [c, byClass.get(c)]),
  ),
  by_class_reachable: Object.fromEntries(
    CLASS_ORDER.filter((c) => byClassReachable.has(c)).map((c) => [
      c,
      byClassReachable.get(c),
    ]),
  ),
  by_status: Object.fromEntries(
    STATUS_ORDER.filter((s) => byStatus.has(s)).map((s) => [
      s,
      byStatus.get(s),
    ]),
  ),
  by_target_phase: Object.fromEntries(
    [...byPhase.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([phase, count]) => [String(phase), count]),
  ),
};

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function renderMarkdown() {
  const lines = [];
  lines.push("# Rag Platform — endpoint coverage matrix");
  lines.push("");
  lines.push("<!-- GENERATED FILE. Do not edit by hand.");
  lines.push("     Regenerate: node scripts/rag-platform/coverage-matrix.mjs");
  lines.push(
    "     CI gate:    node scripts/rag-platform/coverage-matrix.mjs --check -->",
  );
  lines.push("");
  lines.push(
    "Every backend record in `route-inventory.json` — top-level routes and the",
    "second implementations recorded as alternates — carries exactly one class,",
    "one target phase, an owner, an auth role, a consumer, an implementation",
    "status, test evidence and a justification. Nothing is dropped for being",
    "closed at runtime.",
  );
  lines.push("");
  lines.push(
    `- Route inventory: \`docs/rag-platform/route-inventory.json\` (${inventory.totals.routes} routes)`,
  );
  lines.push(
    `- Active proxy scheme: \`${inventory.proxy.scheme}\` (from ${inventory.proxy.scheme_source})`,
  );
  lines.push(`- Backend API version: \`${inventory.backend.api_version}\``);
  lines.push(
    "- Runtime evidence: `docs/rag-platform/runtime-disabled.md`; decision record `docs/adr/0005-backend-proxy-scheme.md`",
  );
  lines.push("- Contract fixtures: `docs/rag-platform/fixtures/`");
  lines.push("");

  lines.push("## Classes");
  lines.push("");
  lines.push("| Class | Meaning | Evidence required |");
  lines.push("| --- | --- | --- |");
  lines.push(
    "| `frontend-screen` | A dedicated Rag Platform view renders this response | UI route/component path, typed service, automated test |",
  );
  lines.push(
    "| `frontend-action` | Called from inside a screen (mutation or sub-read) | UI access path, typed service, automated test |",
  );
  lines.push(
    "| `api-only` | Live contract with no UI of its own (protocol, deprecated shim) | Justification + contract/security test |",
  );
  lines.push(
    "| `external-callback` | Inbound request from a third party, not from our UI | Justification + contract/security test |",
  );
  lines.push(
    "| `internal` | Backend/runtime plumbing, never a product capability | Justification + contract test |",
  );
  lines.push(
    "| `unsupported` | The deployment cannot serve it, so nothing is built on it | Justification + runtime-disabled evidence |",
  );
  lines.push("");
  lines.push(
    "A record the deployment cannot serve is classified `unsupported`: the product",
    "decision is that nothing is built against a closed route. Its justification",
    "names the runtime evidence and, where one exists, the reachable route that",
    "covers the same capability — so a lost capability is visible by evidence,",
    "not by silence. The upstream nature of such a record (internal-only",
    "annotation, OAuth callback, duplicate implementation) is preserved in the",
    "same field.",
  );
  lines.push("");

  lines.push("## Statuses");
  lines.push("");
  lines.push("| Status | Meaning |");
  lines.push("| --- | --- |");
  lines.push(
    "| `implemented` | UI path + typed service + automated test in place |",
  );
  lines.push(
    "| `contract-verified` | A scrubbed fixture records the live request/response pair |",
  );
  lines.push("| `in-progress` | Implementation started, phase not closed |");
  lines.push(
    "| `planned` | Classified; implementation belongs to its target phase |",
  );
  lines.push("| `runtime-disabled` | The deployed topology cannot serve it |");
  lines.push(
    "| `not-proxied` | Reachable only on its own port, opt-in at startup |",
  );
  lines.push("");
  lines.push(
    "Faz 0 ships no product UI, so no record is `implemented` yet. The release",
    "gate requires `planned = 0` and `in-progress = 0` across user-meaningful",
    "public endpoints; at Faz 0 those counts are expected to be non-zero.",
  );
  lines.push("");

  lines.push("## Totals");
  lines.push("");
  lines.push("| Metric | Count |");
  lines.push("| --- | --- |");
  lines.push(
    `| records (routes + alternate implementations) | ${totals.records} |`,
  );
  lines.push(`| — top-level routes | ${totals.top_level_routes} |`);
  lines.push(
    `| — alternate implementations | ${totals.alternate_implementations} |`,
  );
  lines.push(`| reachable in the active scheme | ${totals.reachable} |`);
  lines.push(
    `| closed with no reachable equivalent (capability lost) | ${totals.capability_lost} |`,
  );
  lines.push(`| **unclassified** | **${totals.unclassified}** |`);
  lines.push("");
  lines.push("### By class");
  lines.push("");
  lines.push("| Class | All records | Reachable only |");
  lines.push("| --- | --- | --- |");
  for (const cls of CLASS_ORDER) {
    if (!byClass.has(cls)) continue;
    lines.push(
      `| \`${cls}\` | ${byClass.get(cls)} | ${byClassReachable.get(cls) ?? 0} |`,
    );
  }
  lines.push("");
  lines.push("### By status");
  lines.push("");
  lines.push("| Status | Count |");
  lines.push("| --- | --- |");
  for (const status of STATUS_ORDER) {
    if (!byStatus.has(status)) continue;
    lines.push(`| \`${status}\` | ${byStatus.get(status)} |`);
  }
  lines.push("");
  lines.push("### By target phase");
  lines.push("");
  lines.push("| Phase | Records | Reachable | Owners |");
  lines.push("| --- | --- | --- | --- |");
  for (const phase of [...byPhase.keys()].sort((a, b) => a - b)) {
    const group = records.filter((r) => r.target_phase === phase);
    const owners = [...new Set(group.map((r) => r.owner))].sort().join(", ");
    lines.push(
      `| ${phase} | ${group.length} | ${group.filter((r) => r.runtime === "enabled").length} | ${owners} |`,
    );
  }
  lines.push("");

  lines.push("## Lost capabilities");
  lines.push("");
  lines.push(
    "Routes that are closed in the active proxy scheme **and** have no reachable",
    "route serving the same method and path. A closed route with a live",
    "equivalent is not listed here: the capability survives, only that",
    "implementation of it is unreachable. Every row below is a capability the",
    "deployed topology cannot serve at all, so it is a gap to decide on rather",
    "than a duplicate to ignore.",
  );
  lines.push("");

  const lost = records.filter((r) => r.capability_lost);
  if (lost.length === 0) {
    lines.push("None: every closed route has a reachable equivalent.");
    lines.push("");
  } else {
    const reachableFamilies = new Set(
      records.filter((r) => r.runtime === "enabled").map((r) => r.family),
    );
    const lostByFamily = new Map();
    for (const record of lost) {
      if (!lostByFamily.has(record.family)) lostByFamily.set(record.family, []);
      lostByFamily.get(record.family).push(record);
    }
    const wholeFamilies = [...lostByFamily.keys()]
      .filter((family) => !reachableFamilies.has(family))
      .sort();

    lines.push(
      `${lost.length} routes across ${lostByFamily.size} families. ` +
        `${wholeFamilies.length} of those families have no reachable route at ` +
        "all, so the whole feature area is missing rather than thinned: " +
        `${wholeFamilies.map((f) => `\`${f}\``).join(", ")}.`,
    );
    lines.push("");
    lines.push(
      "| Family | Lost | Family has reachable routes | Owner | Phase |",
    );
    lines.push("| --- | --- | --- | --- | --- |");
    for (const family of [...lostByFamily.keys()].sort()) {
      const group = lostByFamily.get(family);
      const owners = [...new Set(group.map((r) => r.owner))].sort().join(", ");
      const phases = [...new Set(group.map((r) => r.target_phase))]
        .sort((a, b) => a - b)
        .join(", ");
      lines.push(
        `| \`${family}\` | ${group.length} | ${reachableFamilies.has(family) ? "yes (partial gap)" : "**no**"} | ${owners} | ${phases} |`,
      );
    }
    lines.push("");
    lines.push("<details>");
    lines.push(`<summary>All ${lost.length} lost routes</summary>`);
    lines.push("");
    lines.push("| Method | Path | Service | Source | Owner | Phase |");
    lines.push("| --- | --- | --- | --- | --- | --- |");
    for (const family of [...lostByFamily.keys()].sort()) {
      for (const record of lostByFamily
        .get(family)
        .sort(
          (a, b) =>
            a.path.localeCompare(b.path) || a.method.localeCompare(b.method),
        )) {
        lines.push(
          `| ${record.method} | \`${record.path}\` | ${record.service} | \`${record.source}\` | ${record.owner} | ${record.target_phase} |`,
        );
      }
    }
    lines.push("");
    lines.push("</details>");
    lines.push("");
  }

  lines.push("## Records");
  lines.push("");
  lines.push(
    "Grouped by target phase, then by backend family. `Runtime` is the state in",
    "the active proxy scheme; `Evidence` links the fixture, smoke probe or",
    "runtime record that backs the row.",
  );
  lines.push("");

  for (const phase of [...byPhase.keys()].sort((a, b) => a - b)) {
    const group = records.filter((r) => r.target_phase === phase);
    lines.push(`### Faz ${phase} (${group.length} records)`);
    lines.push("");
    const familyCount = new Map();
    for (const record of group) {
      familyCount.set(record.family, (familyCount.get(record.family) ?? 0) + 1);
    }
    const families = [...familyCount.keys()].sort(
      (a, b) => familyCount.get(b) - familyCount.get(a) || a.localeCompare(b),
    );
    for (const family of families) {
      const familyGroup = group
        .filter((r) => r.family === family)
        .sort(
          (a, b) =>
            a.path.localeCompare(b.path) || a.method.localeCompare(b.method),
        );
      lines.push(`#### \`${family}\` (${familyGroup.length})`);
      lines.push("");
      lines.push(
        "| Method | Path | Service | Class | Owner | Auth role | Consumer | Status | Runtime | Typed service | UI path | Evidence | Justification | Source |",
      );
      lines.push(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
      );
      for (const record of familyGroup) {
        lines.push(
          `| ${record.method} | \`${record.path}\` | ${record.service}@${record.service_port} | \`${record.class}\` | ` +
            `${record.owner} | ${record.auth_role} | ${record.consumer} | \`${record.status}\` | ` +
            `${record.runtime === "disabled" ? "**disabled**" : record.runtime} | ${record.typed_service ?? "—"} | ${record.ui_path} | ` +
            `${record.test_evidence.join("<br>")} | ${record.justification} | \`${record.source}\` |`,
        );
      }
      lines.push("");
    }
  }

  lines.push("## Validation");
  lines.push("");
  lines.push(
    "`node scripts/rag-platform/coverage-matrix.mjs --check` fails on any of:",
    "an unclassified record, a record without a target phase, an unmapped auth",
    "role or owner, a duplicate, a record present in one artifact but not the",
    "other, or drift between these outputs and a fresh generation.",
  );
  lines.push("");
  return `${lines.join("\n")}\n`;
}

const jsonPath = join(OUT_DIR, "endpoint-coverage-matrix.json");
const mdPath = join(OUT_DIR, "endpoint-coverage-matrix.md");
const jsonText = `${JSON.stringify(
  {
    generated_by: "scripts/rag-platform/coverage-matrix.mjs",
    derived_from: "docs/rag-platform/route-inventory.json",
    backend: inventory.backend,
    proxy: {
      scheme: inventory.proxy.scheme,
      scheme_source: inventory.proxy.scheme_source,
    },
    totals,
    records,
  },
  null,
  2,
)}\n`;
const mdText = renderMarkdown();

if (problems.length > 0) {
  for (const problem of problems.slice(0, 40)) console.error(problem);
  if (problems.length > 40) console.error(`… and ${problems.length - 40} more`);
  console.error(
    `coverage matrix validation failed: ${problems.length} problem(s)`,
  );
  process.exit(1);
}

if (checkOnly) {
  let drift = false;
  for (const [path, expected] of [
    [jsonPath, jsonText],
    [mdPath, mdText],
  ]) {
    if (!existsSync(path)) {
      console.error(`missing generated file: ${relative(FRONTEND_ROOT, path)}`);
      drift = true;
      continue;
    }
    if (readFileSync(path, "utf8") !== expected) {
      console.error(`stale generated file: ${relative(FRONTEND_ROOT, path)}`);
      drift = true;
    }
  }
  if (drift) {
    console.error(
      "coverage matrix is out of date — run: node scripts/rag-platform/coverage-matrix.mjs",
    );
    process.exit(1);
  }
  console.log(
    `coverage matrix up to date (${totals.records} records, unclassified=${totals.unclassified})`,
  );
  process.exit(0);
}

if (!existsSync(OUT_DIR) || !statSync(OUT_DIR).isDirectory()) {
  console.error(`output dir missing: ${OUT_DIR}`);
  process.exit(2);
}

writeFileSync(jsonPath, jsonText);
writeFileSync(mdPath, mdText);
console.log(
  `wrote ${relative(FRONTEND_ROOT, mdPath)} and ${relative(FRONTEND_ROOT, jsonPath)}: ` +
    `${totals.records} records, unclassified=${totals.unclassified}, reachable=${totals.reachable}`,
);
