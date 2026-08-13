#!/usr/bin/env node
/**
 * Rag Platform contract matrix generator.
 *
 * Plan line 419 asks for a document that lists, per row, the existing frontend
 * function, the endpoint it calls today, the Rag Platform backend endpoint that
 * will serve it, the transform needed, and the phase that does the work.
 *
 * Two of those five columns are facts and three are decisions, so this script
 * splits them:
 *
 *   scanned    The frontend function and the path it calls are read out of
 *              studio/frontend/src on every run. Nobody hand-maintains ~270
 *              call sites, and a stale left-hand column is worse than none.
 *   declared   The backend target, the transform and the phase are hand-written
 *              in MAPPINGS below, because no static rule can decide that
 *              a Studio-local auth route becomes its `/api/v1/users/me` equivalent.
 *
 * Every declared backend target is then re-checked against the generated
 * endpoint coverage matrix: if a target is not a route that exists and is
 * reachable under the active proxy scheme, this script fails. That is what stops
 * the mapping from quietly rotting as the backend moves — a rename upstream
 * breaks the build instead of leaving a wrong instruction in a document a
 * future phase will follow.
 *
 * Usage:
 *   node scripts/rag-platform/contract-matrix.mjs [--check]
 *
 *   --check   Do not write. Exit 1 if the committed output differs from a fresh
 *             scan (CI drift guard), or if any declared target is unreachable.
 */

import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, "..", "..");
const SRC_ROOT = join(FRONTEND_ROOT, "studio", "frontend", "src");
const OUT_DIR = join(FRONTEND_ROOT, "docs", "rag-platform");
const OUT_MD = join(OUT_DIR, "contract-matrix.md");
const COVERAGE_JSON = join(OUT_DIR, "endpoint-coverage-matrix.json");

const checkOnly = process.argv.slice(2).includes("--check");

// ---------------------------------------------------------------------------
// Frontend scan
// ---------------------------------------------------------------------------

/**
 * Base-path constants that a call site may prefix onto its path. Read from
 * source rather than assumed: `lib/api-base.ts` prefixes every call through
 * `apiUrl`, and a few features add a further constant of their own.
 */
const BASE_CONSTANTS = {
  RAG_BASE: "/api/rag",
  DATA_DESIGNER_API_BASE: "/api/data-recipe",
  DEFAULT_BASE: "/api/data-recipe",
  OVERRIDES_URL: "/api/settings/openai-auto-switch/overrides",
};

/**
 * Functions whose first argument is the path. `apiUrl` is included because it is
 * the single prefixer every call goes through: a bare `fetch(apiUrl("/x"))`
 * carries its literal there rather than at the fetch, and `authFetch` calls it
 * internally. Sites are deduplicated by file+line, so a call that matches both
 * patterns is recorded once.
 */
const CLIENT_CALLS = ["authFetch", "fetch", "fetchWithTimeout", "fetchWithRetry", "apiUrl"];

/**
 * Paths whose literal is not at the fetch site because a thin wrapper prefixes
 * it. The wrapper's own prefix is recorded here so a scanned `"/knowledge-bases"`
 * resolves to the full path; the sub-paths are still scanned from the callers.
 */
const WRAPPER_PREFIXES = [
  { file: "features/rag/api/rag-api.ts", fns: ["ragRequest", "ragUpload"], prefix: "/api/rag" },
  {
    file: "features/recipe-studio/api/index.ts",
    fns: ["postJson", "getJson"],
    prefix: "/api/data-recipe",
  },
  { file: "features/chat/api/mcp-servers-api.ts", fns: ["mcpRequest"], prefix: "/api/mcp/servers" },
];

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry.startsWith(".")) continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (/\.(ts|tsx)$/.test(entry) && !/\.d\.ts$/.test(entry)) out.push(full);
  }
  return out;
}

/**
 * Read the string or template literal that starts at `start` and collapse every
 * interpolation to `{p}`.
 *
 * This is a scanner rather than a regex because the interpolations nest: a
 * conditional query suffix is written `` `/api/chat/threads${qs ? `?${qs}` : ""}` ``,
 * where both the braces and the backticks are nested two deep. A regex either
 * stops at the inner backtick or swallows the wrong closing brace, and both
 * failures produce a path that looks real but is not.
 *
 * Returns null when the argument is not a literal.
 */
function readPathLiteral(source, start) {
  let i = start;
  while (i < source.length && /\s/.test(source[i])) i += 1;
  const quote = source[i];
  if (quote === '"' || quote === "'") {
    const end = source.indexOf(quote, i + 1);
    return end < 0 ? null : source.slice(i + 1, end);
  }
  if (quote !== "`") return null;

  let out = "";
  i += 1;
  while (i < source.length) {
    const char = source[i];
    if (char === "\\") {
      out += source[i + 1] ?? "";
      i += 2;
      continue;
    }
    if (char === "`") return out;
    if (char === "$" && source[i + 1] === "{") {
      // Skip the interpolation, matching braces and honouring nested literals.
      const exprStart = i + 2;
      let depth = 1;
      i += 2;
      while (i < source.length && depth > 0) {
        const inner = source[i];
        if (inner === "{") depth += 1;
        else if (inner === "}") depth -= 1;
        else if (inner === "`" || inner === '"' || inner === "'") {
          // Step over the nested literal so its braces do not count.
          i += 1;
          while (i < source.length && source[i] !== inner) {
            if (source[i] === "\\") i += 1;
            else if (inner === "`" && source[i] === "$" && source[i + 1] === "{") {
              let d = 1;
              i += 2;
              while (i < source.length && d > 0) {
                if (source[i] === "{") d += 1;
                else if (source[i] === "}") d -= 1;
                i += 1;
              }
              continue;
            }
            i += 1;
          }
        }
        i += 1;
      }
      // A bare base constant is a known literal prefix, not a parameter.
      const expr = source.slice(exprStart, i - 1).trim();
      out += BASE_CONSTANTS[expr] ?? "{p}";
      continue;
    }
    out += char;
    i += 1;
  }
  return null;
}

/**
 * Reduce a collapsed literal to the route it addresses: drop the query string,
 * and drop a trailing `{p}` that is not its own path segment — that shape is an
 * appended query (`/threads${qs}`), not a path parameter.
 */
function templatePath(raw) {
  let path = raw;
  const q = path.indexOf("?");
  if (q >= 0) path = path.slice(0, q);
  while (path.endsWith("{p}") && !path.endsWith("/{p}")) path = path.slice(0, -3);
  return path;
}

/**
 * The enclosing function name for a byte offset, found by walking backwards to
 * the nearest declaration. Accurate enough for a reference column and needs no
 * parser.
 */
function enclosingFunction(source, index) {
  const before = source.slice(0, index);
  const matches = [
    ...before.matchAll(
      /(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)|(?:export\s+)?const\s+([A-Za-z0-9_]+)\s*(?::[^=]*)?=\s*(?:async\s*)?\(/g,
    ),
  ];
  const last = matches.at(-1);
  return last ? (last[1] ?? last[2]) : "(module scope)";
}

/**
 * The HTTP method for a call, read from the nearest `method:` in the options
 * object. The window stops at a blank line or a closing brace at column zero so
 * a GET cannot inherit the verb of the next function down the file.
 */
function methodAt(source, index, fallback = "GET") {
  let tail = source.slice(index, index + 400);
  const stop = Math.min(
    ...[tail.indexOf("\n\n"), tail.indexOf("\n}\n")].filter((i) => i > 0).concat(tail.length),
  );
  tail = tail.slice(0, stop);
  const match = tail.match(/method:\s*"([A-Z]+)"/);
  return match ? match[1] : fallback;
}

function scanFrontend() {
  const calls = new Map(); // "METHOD path" -> {method, path, sites:[]}
  const seen = new Set(); // file:line:path — `apiUrl` and its caller are one call

  const record = (calls, method, path, file, line, fn) => {
    const site = `${file}:${line}:${path}`;
    if (seen.has(site)) return;
    seen.add(site);
    const key = `${method} ${path}`;
    if (!calls.has(key)) calls.set(key, { method, path, sites: [] });
    calls.get(key).sites.push({ file, line, fn });
  };

  for (const file of walk(SRC_ROOT)) {
    const source = readFileSync(file, "utf8");
    const rel = relative(SRC_ROOT, file);
    const wrapper = WRAPPER_PREFIXES.find((w) => w.file === rel);

    for (const client of CLIENT_CALLS) {
      const re = new RegExp(`\\b${client}(?:<[^>]*>)?\\(`, "g");
      for (const match of source.matchAll(re)) {
        const raw = readPathLiteral(source, match.index + match[0].length);
        if (!raw || !raw.includes("/")) continue;
        if (raw.startsWith("http")) continue; // external host, not our contract

        // A wrapper's own fetch carries only the variable, not a path.
        if (raw.startsWith("{p}")) continue;
        const path = templatePath(raw);
        if (!path.startsWith("/")) continue;
        // `authFetch(`${RAG_BASE}${path}`)` collapses to the bare prefix, which is
        // not a route. The real paths come from scanning the wrapper's callers.
        if (wrapper && path === wrapper.prefix) continue;

        const line = source.slice(0, match.index).split("\n").length;
        record(
          calls,
          methodAt(source, match.index),
          path,
          rel,
          line,
          enclosingFunction(source, match.index),
        );
      }
    }

    // A wrapper's callers name only the sub-path; prefix it here.
    if (wrapper) {
      for (const fn of wrapper.fns) {
        const re = new RegExp(`\\b${fn}(?:<[^>]*>)?\\(`, "g");
        for (const match of source.matchAll(re)) {
          const raw = readPathLiteral(source, match.index + match[0].length);
          if (raw === null || !raw.startsWith("/")) continue;
          const path = templatePath(wrapper.prefix + raw);
          const line = source.slice(0, match.index).split("\n").length;
          // An upload wrapper takes no options object: the verb is the wrapper's.
          const fallback = /upload/i.test(fn) ? "POST" : "GET";
          record(
            calls,
            methodAt(source, match.index, fallback),
            path,
            rel,
            line,
            enclosingFunction(source, match.index),
          );
        }
      }
    }
  }
  return calls;
}

// ---------------------------------------------------------------------------
// Declared mappings
// ---------------------------------------------------------------------------

const NOT_BACKED = "—";

/**
 * Frontend call -> Rag Platform backend endpoint.
 *
 * `to`        The backend path exactly as the coverage matrix spells it, so the
 *             reference can be machine-checked. `null` means the platform
 *             backend has no equivalent: those rows are decisions, not gaps, and
 *             carry the reason in `transform`.
 * `transform` What has to change between the two shapes. This is the column a
 *             later phase implements against, so it names the concrete
 *             mismatch, not "map fields".
 * `phase`     The plan phase that owns the work.
 *
 * Only rows whose frontend path the scan actually finds are emitted, so a
 * mapping for a call site that has since been deleted surfaces as an unused
 * declaration rather than a phantom row.
 */
const MAPPINGS = {
  // --- auth (Faz 2) ------------------------------------------------------
  "POST /api/auth/login": {
    to: "POST /api/v1/auth/login",
    transform:
      "Studio posts `{username,password}` and reads a session cookie; the platform takes `{email,password}` and answers with the token in the `Authorization` response header. Adapter must lift the header into the token store.",
    phase: 2,
  },
  "GET /api/auth/status": {
    to: "GET /api/v1/users/me",
    transform:
      "No status endpoint exists; identity is the probe. 200 means authenticated, 401 means not. Adapter converts the user payload into the `{authenticated, user}` shape the guards expect.",
    phase: 2,
  },
  "POST /api/auth/logout": {
    to: "POST /api/v1/auth/logout",
    transform: "Same intent; drop the local token after the call rather than relying on a cookie.",
    phase: 2,
  },
  "POST /api/auth/refresh": {
    to: null,
    transform:
      "The platform issues no refresh token — `auth/login` returns one bearer token with no paired refresh route. The 401-refresh-and-retry path in `features/auth/api.ts` must become re-authentication. Recorded in ADR 0002.",
    phase: 2,
  },
  "POST /api/auth/change-password": {
    to: "PATCH /api/v1/users/me",
    transform:
      "Self-service password change folds into the profile PATCH (`{password, new_password}`) instead of a dedicated route.",
    phase: 2,
  },
  "GET /api/auth/api-keys": {
    to: "GET /api/v1/system/tokens",
    transform: "Rename only; response is the platform token list envelope.",
    phase: 9,
  },
  "POST /api/auth/api-keys": {
    to: "POST /api/v1/system/tokens",
    transform: "Rename only.",
    phase: 9,
  },
  "DELETE /api/auth/api-keys/{p}": {
    to: "DELETE /api/v1/system/tokens/<token>",
    transform:
      "The path parameter is the token value, not an opaque key id, so the list response must carry whatever the delete call addresses.",
    phase: 9,
  },

  // --- documents (Faz 5) ------------------------------------------------
  "GET /api/rag/documents": {
    to: null,
    transform:
      "No cross-dataset document listing exists — every platform document route is dataset-scoped. The all-documents view must iterate datasets or be dropped; decided in Faz 5.",
    phase: 5,
  },
  "DELETE /api/rag/documents/{p}": {
    to: "DELETE /api/v1/datasets/<dataset_id>/documents",
    transform:
      "Requires a `dataset_id` the current caller does not hold, and takes ids in the body. Document identity in the UI must start carrying its dataset.",
    phase: 5,
  },
  "GET /api/rag/documents/{p}/preview-target": {
    to: "GET /api/v1/documents/<doc_id>/preview",
    transform:
      "Studio resolves a chunk to a page and offset; the platform preview returns the rendered document only. Chunk-to-page resolution has no platform equivalent and must be recomputed client-side from chunk positions.",
    phase: 5,
  },
  "GET /api/rag/documents/{p}/file-url": {
    to: "GET /api/v1/document/get/<doc_id>",
    transform:
      "Studio expects a URL to hand the viewer; the platform streams the bytes. The adapter must build an object URL from the response instead of passing a URL through.",
    phase: 5,
  },
  "GET /api/rag/threads/{p}/documents": {
    to: null,
    transform:
      "Thread-scoped documents do not exist: platform attachment scope is the dataset. Decision record required by plan line 424; the likely resolution is a per-session dataset.",
    phase: 5,
  },
  "POST /api/rag/threads/{p}/documents": {
    to: null,
    transform: "Same as the thread document listing — no thread-level document scope exists.",
    phase: 5,
  },
  "GET /api/rag/projects/{p}/documents": {
    to: "GET /api/v1/datasets/<dataset_id>/documents",
    transform:
      "Under the Project→Chat mapping (ADR 0003) a project's sources become a dataset bound to the assistant, so this is an ordinary dataset document list.",
    phase: 5,
  },
  "POST /api/rag/projects/{p}/documents": {
    to: "POST /api/v1/datasets/<dataset_id>/documents",
    transform: "As the project document listing, plus the upload/parse chaining.",
    phase: 5,
  },

  // --- parse lifecycle: jobs have no platform resource (Faz 5) ----------
  "GET /api/rag/jobs/{p}": {
    to: "GET /api/v1/datasets/<dataset_id>/documents/<document_id>",
    transform:
      "The platform has no job resource. Parse progress lives on the document, so a job id must become a document reference and the job read becomes a document poll.",
    phase: 5,
  },
  "GET /api/rag/jobs/{p}/events": {
    to: null,
    transform:
      "No SSE parse-event stream: `documents/parse` returns immediately and progress is polled. The event-stream consumer becomes a poller — a behaviour change to record in Faz 5.",
    phase: 5,
  },

  // Retrieval (`POST /api/v1/retrieval`) has no row: Studio never calls a
  // standalone search endpoint. Retrieval reaches the backend only inside a
  // completion, so it enters through the completions row in Faz 8 and is
  // exercised by the retrieval contract fixture rather than by a UI call site.

  // --- threads -> sessions, projects -> chats (Faz 7) -------------------
  "GET /api/chat/threads": {
    to: "GET /api/v1/chats/<chat_id>/sessions",
    transform:
      "Threads are sessions under an assistant (ADR 0003), so a flat thread list becomes per-chat session lists and the caller needs a chat id in scope.",
    phase: 7,
  },
  "GET /api/chat/threads/{p}": {
    to: "GET /api/v1/chats/<chat_id>/sessions/<session_id>",
    transform: "Adds the owning chat id to the address.",
    phase: 7,
  },
  "POST /api/chat/threads": {
    to: "POST /api/v1/chats/<chat_id>/sessions",
    transform:
      "Session creation needs an existing assistant, so Faz 7 must create or reuse one before the first thread.",
    phase: 7,
  },
  "PATCH /api/chat/threads/{p}": {
    to: "PATCH /api/v1/chats/<chat_id>/sessions/<session_id>",
    transform: "Rename plus chat scope.",
    phase: 7,
  },
  "DELETE /api/chat/threads": {
    to: "DELETE /api/v1/chats/<chat_id>/sessions",
    transform:
      "Bulk delete by id list is scoped to one chat, so a cross-chat bulk delete becomes one call per chat.",
    phase: 7,
  },
  "DELETE /api/chat": {
    to: "DELETE /api/v1/chats/<chat_id>/sessions",
    transform:
      "'Clear all chats' has no single platform call: it becomes enumerate chats, then delete each chat's sessions. The `delete_files` sandbox cleanup is Studio-local and has no platform half.",
    phase: 7,
  },
  "GET /api/chat/count": {
    to: "GET /api/v1/chats/<chat_id>/sessions",
    transform:
      "No count endpoint; the session list envelope carries the total. Counting across chats means summing per chat.",
    phase: 7,
  },
  "POST /api/chat/threads/{p}/fork": {
    to: null,
    transform:
      "No fork primitive. Forking must be composed client-side (create session, replay messages) or dropped; decision record required by plan line 424.",
    phase: 7,
  },
  "GET /api/chat/threads/{p}/messages/{p}/forks": {
    to: null,
    transform: "Follows the fork decision — with no fork primitive there is no sibling listing.",
    phase: 7,
  },
  "GET /api/chat/threads/{p}/messages": {
    to: "GET /api/v1/chats/<chat_id>/sessions/<session_id>",
    transform:
      "Messages are not a separate collection: they arrive inside the session object, so the list call folds into the session read.",
    phase: 7,
  },
  "GET /api/chat/threads/{p}/messages/{p}": {
    to: "GET /api/v1/chats/<chat_id>/sessions/<session_id>",
    transform: "Single-message read becomes a client-side lookup within the session payload.",
    phase: 7,
  },
  "PUT /api/chat/threads/{p}/messages/{p}": {
    to: null,
    transform:
      "No per-message write. The platform appends through completions and offers only delete and feedback on an existing message, so client-side message editing has no backend path.",
    phase: 7,
  },
  "PUT /api/chat/threads/{p}/messages": {
    to: null,
    transform: "Same as the single-message write — no message-collection replace exists.",
    phase: 7,
  },
  "POST /api/chat/messages:batch": {
    to: null,
    transform:
      "Batch message read is a Studio pagination optimisation; the platform ships whole sessions, so the batch call disappears rather than being repointed.",
    phase: 7,
  },
  "GET /api/chat/projects": {
    to: "GET /api/v1/chats",
    transform: "Projects become chat assistants (ADR 0003).",
    phase: 7,
  },
  "GET /api/chat/projects/{p}": {
    to: "GET /api/v1/chats/<chat_id>",
    transform: "As the project listing.",
    phase: 7,
  },
  "POST /api/chat/projects": {
    to: "POST /api/v1/chats",
    transform:
      "Assistant creation requires model and prompt configuration a Studio project never carried; defaults come from Faz 3 readiness.",
    phase: 7,
  },
  "PATCH /api/chat/projects/{p}": {
    to: "PATCH /api/v1/chats/<chat_id>",
    transform: "As the project listing.",
    phase: 7,
  },
  "DELETE /api/chat/projects/{p}": {
    to: "DELETE /api/v1/chats",
    transform:
      "Body-list delete rather than per-id; the `delete_files` query flag is Studio-local sandbox cleanup with no platform half.",
    phase: 7,
  },

  // --- completions and cancellation (Faz 8) ----------------------------
  "POST /v1/chat/completions": {
    to: "POST /api/v1/chats/<chat_id>/completions",
    transform:
      "Studio speaks OpenAI chat-completions. The platform's own completion route is assistant-scoped and streams its own SSE envelope carrying `reference` blocks; the OpenAI-compatible surface (`chats_openai/<chat_id>/chat/completions`) exists but drops those references, so Faz 8 uses the native route and adapts the stream.",
    phase: 8,
  },
  "POST /api/inference/cancel": {
    to: null,
    transform:
      "No server-side cancellation route. Aborting the fetch is the only stop, which leaves the backend generation running. Decision record required by plan line 424.",
    phase: 8,
  },

  // --- system / health (Faz 1) -----------------------------------------
  "GET /api/health": {
    to: "GET /api/v1/system/ping",
    transform:
      "Studio's health payload carries device type and version; the platform ping returns `\"pong\"`. Version comes from `system/version` and device type has no equivalent, so the composite must be assembled or the callers narrowed.",
    phase: 1,
  },
  "GET /api/system": {
    to: "GET /api/v1/system/status",
    transform:
      "Platform status reports service component health, not host hardware. The hardware half of Studio's system panel has no backend source and stays Studio-local.",
    phase: 1,
  },
  "GET /api/system/hardware": {
    to: null,
    transform:
      "Host GPU/RAM detection drives Studio's training-method policy. It is a desktop capability with no platform counterpart and stays on the Studio backend.",
    phase: 1,
  },
  "GET /openapi.json": {
    to: null,
    transform:
      "Used only by `repair-legacy-chat-titles` to sniff whether a route exists. The probe should be deleted rather than repointed.",
    phase: 1,
  },

  // --- providers and models (Faz 3) ------------------------------------
  "GET /api/providers/registry": {
    to: "GET /api/v1/providers",
    transform:
      "Studio's registry is a static catalogue of provider types; the platform lists configured providers. The catalogue half must come from the same call's metadata.",
    phase: 3,
  },
  "GET /api/providers/": {
    to: "GET /api/v1/providers",
    transform: "Same intent, platform envelope; trailing slash dropped.",
    phase: 3,
  },
  "POST /api/providers/": {
    to: "PUT /api/v1/providers",
    transform:
      "Creation is a PUT upsert on the collection, not a POST; instances and models are then configured through the nested instance routes.",
    phase: 3,
  },
  "PUT /api/providers/{p}": {
    to: "PUT /api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>",
    transform:
      "Studio's flat provider config maps onto a provider + instance pair, so one update becomes two levels of address.",
    phase: 3,
  },
  "DELETE /api/providers/{p}": {
    to: "DELETE /api/v1/providers/<provider_id_or_name>",
    transform: "Rename only.",
    phase: 3,
  },
  "POST /api/providers/test": {
    to: "POST /api/v1/providers/<provider_id_or_name>/connection",
    transform:
      "Connection test moves from a body-identified provider to a path-identified one, so the provider must exist before it can be tested — Studio tests before saving.",
    phase: 3,
  },
  "POST /api/providers/models": {
    to: "GET /api/v1/providers/<provider_id_or_name>/models",
    transform: "POST-with-body becomes a GET on the provider; verb and identification both change.",
    phase: 3,
  },
  "GET /api/providers/public-key": {
    to: null,
    transform:
      "Studio fetches an RSA public key to encrypt provider API keys in transit. The platform accepts provider credentials over TLS with no client-side envelope encryption, so this call disappears; the security implication belongs in the Faz 3 ADR.",
    phase: 3,
  },
  "GET /api/models/list": {
    to: "GET /api/v1/models",
    transform:
      "Studio lists local model files; the platform lists configured LLM entries. Different notion of 'model' — Faz 3 decides which the picker shows.",
    phase: 3,
  },
  "GET /v1/models": {
    to: "GET /api/v1/models",
    transform: "OpenAI-compatible list replaced by the platform model list.",
    phase: 3,
  },

  // --- MCP (Faz 11) ----------------------------------------------------
  "GET /api/mcp/servers/": {
    to: "GET /api/v1/mcp/servers",
    transform: "Trailing slash dropped; platform envelope.",
    phase: 11,
  },
  "POST /api/mcp/servers/": {
    to: "POST /api/v1/mcp/servers",
    transform: "Trailing slash dropped.",
    phase: 11,
  },
  "PUT /api/mcp/servers/{p}": {
    to: "PUT /api/v1/mcp/servers/<mcp_id>",
    transform: "Rename only.",
    phase: 11,
  },
  "DELETE /api/mcp/servers/{p}": {
    to: "DELETE /api/v1/mcp/servers/<mcp_id>",
    transform: "Rename only.",
    phase: 11,
  },
  "POST /api/mcp/servers/test": {
    to: "POST /api/v1/mcp/servers/<mcp_id>/test",
    transform:
      "Test needs a saved server id rather than an inline config, so the dialog must save before it can test.",
    phase: 11,
  },
  "POST /api/mcp/servers/import": {
    to: "POST /api/v1/mcp/servers/import",
    transform: "Direct match.",
    phase: 11,
  },
  "POST /api/mcp/servers/{p}/refresh": {
    to: "POST /api/v1/mcp/servers/<mcp_id>/test",
    transform:
      "No separate tool-refresh route; the test call returns the server's current tool list, which is what refresh consumes.",
    phase: 11,
  },

  // --- Studio-only chat surfaces with no platform counterpart ----------
  "GET /api/chat/settings": {
    to: null,
    transform:
      "Persisted chat UI preferences (model pick, sampling, panel state) are client settings stored by the Studio backend. The platform stores assistant configuration, not per-user UI state, so this stays Studio-local.",
    phase: 7,
  },
  "PUT /api/chat/settings": {
    to: null,
    transform: "As the settings read — Studio-local UI state.",
    phase: 7,
  },
  "GET /api/chat/attachments": {
    to: null,
    transform:
      "Studio stores message attachments (pasted images, extracted text) in its own store. Platform attachments are dataset documents, so the gallery has no platform equivalent and stays Studio-local.",
    phase: 7,
  },
  "GET /api/chat/attachments/{p}/{p}/file": {
    to: null,
    transform: "As the attachment listing — Studio-local blob store.",
    phase: 7,
  },
  "DELETE /api/chat/attachments/{p}/{p}": {
    to: null,
    transform: "As the attachment listing — Studio-local blob store.",
    phase: 7,
  },
  "GET /api/chat/export": {
    to: null,
    transform:
      "Whole-history export is a Studio data-portability feature. The platform exposes no bulk export, so it must be composed from session reads or stay Studio-local; decided in Faz 7.",
    phase: 7,
  },
  "GET /api/chat/import-ledger": {
    to: null,
    transform:
      "The import ledger is Studio's own migration bookkeeping and has no meaning to the platform.",
    phase: 7,
  },
  "POST /api/chat/import-ledger": {
    to: null,
    transform: "As the ledger read — Studio-local migration bookkeeping.",
    phase: 7,
  },

  // --- deep research: unsupported (ADR 0004) ---------------------------
  "POST /api/chat/research-runs": {
    to: null,
    transform:
      "Deep-research orchestration is Studio-only with no platform counterpart; classified unsupported in ADR 0004.",
    phase: 4,
  },
  "GET /api/chat/research-runs/{p}": {
    to: null,
    transform: "Part of the unsupported deep-research surface (ADR 0004).",
    phase: 4,
  },
  "GET /api/chat/research-runs/active": {
    to: null,
    transform: "Part of the unsupported deep-research surface (ADR 0004).",
    phase: 4,
  },
  "POST /api/chat/research-runs/{p}/{p}": {
    to: null,
    transform: "Part of the unsupported deep-research surface (ADR 0004).",
    phase: 4,
  },
  "POST /api/chat/research-runs/{p}/events": {
    to: null,
    transform: "Part of the unsupported deep-research surface (ADR 0004).",
    phase: 4,
  },
  "PUT /api/chat/research-runs/{p}/plan": {
    to: null,
    transform: "Part of the unsupported deep-research surface (ADR 0004).",
    phase: 4,
  },
};

/**
 * Frontend surfaces that stay on the Studio backend rather than moving to the
 * platform. Recorded so the matrix accounts for every scanned pair instead of
 * listing only the mapped ones: a reader must be able to tell "not mapped yet"
 * from "deliberately not moving".
 */
const STUDIO_ONLY_PREFIXES = [
  ["/api/train", "Fine-tuning is Studio's own product surface; the platform has no training API."],
  ["/api/export", "Model export/merge/GGUF is Studio-local."],
  ["/api/hub", "Hugging Face hub cache and download management is Studio-local."],
  ["/api/models", "Local model file inventory, GGUF variants and folder scanning are Studio-local."],
  [
    "/api/inference",
    "Local inference server control (load/unload, audio, images, video, sandbox) is Studio-local.",
  ],
  ["/api/data-recipe", "Synthetic data recipe studio is Studio-local."],
  ["/api/settings", "Desktop application settings are Studio-local."],
  ["/api/prompts", "Prompt library is Studio-local storage."],
  ["/api/studio", "Studio updater and install-source metadata."],
  ["/api/llama", "llama.cpp updater is Studio-local."],
  ["/api/picker", "Chat-template validation is Studio-local."],
  ["/api/profile", "Studio usage statistics."],
  ["/api/security", "Remote-code scanning of downloaded models is Studio-local."],
  ["/api/shutdown", "Desktop process control."],
  ["/api/video", "Local video generation is Studio-local."],
  ["/api/audio", "Local audio generation is Studio-local."],
  ["/api/images", "Local image generation is Studio-local."],
];

function studioOnlyReason(path) {
  for (const [prefix, reason] of STUDIO_ONLY_PREFIXES) {
    if (path === prefix || path.startsWith(prefix + "/")) return reason;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Validation against the generated coverage matrix
// ---------------------------------------------------------------------------

function canonicalContractKey(method, path) {
  const canonical = path
    .split("?")[0]
    .replace(/<[^>]+>:<[^>]+>/g, "{p}")
    .replace(/<[^>]+>/g, "{p}")
    .replace(/:[A-Za-z_][A-Za-z0-9_]*/g, "{p}")
    .replace(/\*[A-Za-z_][A-Za-z0-9_]*/g, "{p}")
    .replace(/\{p\}/g, "{p}");
  return `${method} ${canonical}`;
}

function canonicalDeclaredTarget(target) {
  const separator = target.indexOf(" ");
  return canonicalContractKey(target.slice(0, separator), target.slice(separator + 1));
}

function loadReachable() {
  if (!existsSync(COVERAGE_JSON)) {
    console.error(
      `missing ${relative(FRONTEND_ROOT, COVERAGE_JSON)} — run coverage-matrix.mjs first`,
    );
    process.exit(2);
  }
  const parsed = JSON.parse(readFileSync(COVERAGE_JSON, "utf8"));
  const records = parsed.records ?? parsed;
  const reachable = new Set();
  for (const rec of records) {
    const runtime = rec.runtime ?? rec.runtime_state;
    const enabled = runtime === "enabled" || rec.runtime_enabled === true;
    if (enabled) reachable.add(canonicalContractKey(rec.method, rec.path));
  }
  return reachable;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function render(calls, reachable) {
  const rows = [];
  const problems = [];
  const unmapped = [];
  const studioOnly = new Map();

  for (const [key, call] of [...calls.entries()].sort()) {
    const mapping = MAPPINGS[key];
    if (mapping) {
      if (mapping.to && !reachable.has(canonicalDeclaredTarget(mapping.to))) {
        problems.push(`declared target is not reachable: ${key} -> ${mapping.to}`);
      }
      const site = call.sites[0];
      rows.push({
        fn: site.fn,
        site: `${site.file}:${site.line}`,
        from: key,
        to: mapping.to ?? NOT_BACKED,
        transform: mapping.transform,
        phase: mapping.phase,
        extraSites: call.sites.length - 1,
      });
      continue;
    }
    const reason = studioOnlyReason(call.path);
    if (reason) {
      if (!studioOnly.has(reason)) studioOnly.set(reason, []);
      studioOnly.get(reason).push(key);
      continue;
    }
    unmapped.push({ key, site: `${call.sites[0].file}:${call.sites[0].line}` });
  }

  for (const key of Object.keys(MAPPINGS)) {
    if (!calls.has(key)) problems.push(`mapping for a call site the scan does not find: ${key}`);
  }

  const studioCount = [...studioOnly.values()].reduce((n, v) => n + v.length, 0);
  const lines = [];
  lines.push("# Rag Platform contract matrix");
  lines.push("");
  lines.push(
    "Generated by `scripts/rag-platform/contract-matrix.mjs`. Do not edit by hand — rerun the script.",
  );
  lines.push("");
  lines.push(
    "The frontend function, call site and current endpoint are scanned from `studio/frontend/src`. " +
      "The backend endpoint, transform and phase are declared in the script and re-verified on every " +
      "run against `endpoint-coverage-matrix.json`: a declared target that is not reachable under the " +
      "active proxy scheme fails the run rather than sitting here as a wrong instruction.",
  );
  lines.push("");
  lines.push(`- scanned frontend method+path pairs: **${calls.size}**`);
  lines.push(
    `- mapped to a Rag Platform endpoint: **${rows.filter((r) => r.to !== NOT_BACKED).length}**`,
  );
  lines.push(
    `- mapped but with no platform equivalent: **${rows.filter((r) => r.to === NOT_BACKED).length}**`,
  );
  lines.push(`- Studio-local, not migrating: **${studioCount}**`);
  lines.push(`- not yet mapped: **${unmapped.length}**`);
  lines.push("");

  lines.push("## Migration rows");
  lines.push("");
  lines.push(
    "| frontend function | call site | current endpoint | Rag Platform endpoint | transform | faz |",
  );
  lines.push("| --- | --- | --- | --- | --- | --- |");
  for (const row of rows.sort((a, b) => a.phase - b.phase || a.from.localeCompare(b.from))) {
    const site = row.extraSites > 0 ? `${row.site} (+${row.extraSites})` : row.site;
    const to = row.to === NOT_BACKED ? NOT_BACKED : `\`${row.to}\``;
    lines.push(
      `| \`${row.fn}\` | \`${site}\` | \`${row.from}\` | ${to} | ${row.transform} | ${row.phase} |`,
    );
  }
  lines.push("");

  lines.push("## Studio-local surfaces (not migrating)");
  lines.push("");
  lines.push(
    "These calls stay on the Studio backend. They are listed so the matrix accounts for every " +
      "scanned pair: a reader can tell a deliberate non-migration from an unmapped gap.",
  );
  lines.push("");
  for (const [reason, keys] of [...studioOnly.entries()].sort((a, b) => b[1].length - a[1].length)) {
    lines.push(`- **${keys.length} endpoints** — ${reason}`);
  }
  lines.push("");

  if (unmapped.length > 0) {
    lines.push("## Not yet mapped");
    lines.push("");
    lines.push(
      "Scanned calls with neither a declared mapping nor a Studio-local classification. Each must " +
        "gain one before the phase that touches it.",
    );
    lines.push("");
    lines.push("| endpoint | call site |");
    lines.push("| --- | --- |");
    for (const item of unmapped.sort((a, b) => a.key.localeCompare(b.key))) {
      lines.push(`| \`${item.key}\` | \`${item.site}\` |`);
    }
    lines.push("");
  }

  lines.push("## Scan limits");
  lines.push("");
  lines.push(
    "- Query strings are dropped, so `?since=` variants collapse onto one route. Query parameters " +
      "are part of the contract and are named in the transform column where they matter.",
  );
  lines.push(
    "- Call sites that take a caller-supplied absolute URL, and those that consume server-minted " +
      "signed URLs, carry no literal path and are therefore absent from the scan.",
  );
  lines.push(
    "- External hosts (Hugging Face) are excluded: they are not part of the backend contract.",
  );
  lines.push(
    "- Tauri IPC callers (`features/settings/api/launch-at-login.ts`, `features/native-intents/api.ts`) " +
      "make no HTTP call and so have no row.",
  );
  lines.push("");

  return { markdown: lines.join("\n") + "\n", problems };
}

// ---------------------------------------------------------------------------

const calls = scanFrontend();
const reachable = loadReachable();
const { markdown, problems } = render(calls, reachable);

if (problems.length > 0) {
  console.error("contract matrix validation failed:");
  for (const problem of problems) console.error(`  - ${problem}`);
  process.exit(1);
}

if (checkOnly) {
  const current = existsSync(OUT_MD) ? readFileSync(OUT_MD, "utf8") : "";
  if (current !== markdown) {
    console.error(`${relative(FRONTEND_ROOT, OUT_MD)} is stale — rerun contract-matrix.mjs`);
    process.exit(1);
  }
  console.log(`contract matrix up to date (${calls.size} scanned pairs)`);
} else {
  writeFileSync(OUT_MD, markdown);
  console.log(`wrote ${relative(FRONTEND_ROOT, OUT_MD)} (${calls.size} scanned pairs)`);
}
