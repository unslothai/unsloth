#!/usr/bin/env node
/**
 * Rag Platform P0 contract fixture capture.
 *
 * Drives the running backend through the flows the next phase's client has to
 * implement, and records the real request/response pair for each step. Nothing
 * in the fixtures is hand-written: if a field is in there, the backend sent it.
 *
 * Flows captured, in dependency order:
 *   auth       register -> login -> whoami -> logout
 *   dataset    create -> list -> get -> update
 *   document   upload -> list -> parse -> status
 *   chunk      list -> get -> update
 *   retrieval  POST /retrieval
 *   chat       create assistant -> list -> get
 *   session    create -> list -> get
 *   stream     completion with stream=true (SSE frame shapes)
 *
 * SECRET HANDLING. Fixtures are committed, so nothing sensitive may reach them.
 * Every value is passed through scrub() before it is written:
 *   * request bodies:  password / encrypted password never leave this process
 *   * response bodies: access_token, api_key, Authorization, provider keys and
 *     the throwaway account's e-mail are replaced with stable placeholders
 *   * headers:         only a small allowlist is recorded at all
 * The session token lives in a local variable and is never logged or written.
 *
 * The account is created fresh per run with a random local part and is only
 * ever used against the local stack.
 *
 * Usage:
 *   node scripts/rag-platform/capture-fixtures.mjs [--base http://127.0.0.1]
 *                                                  [--keep] [--dry-run]
 *
 *   --keep      Do not delete the artifacts the run creates (default: clean up).
 *   --dry-run   Probe reachability and print the plan without writing fixtures.
 */

import { execFileSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, "..", "..");
const OUT_DIR = join(FRONTEND_ROOT, "docs", "rag-platform", "fixtures");
const CONTAINER = "rag-platform-backend";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
};
const BASE = (flag("--base", "http://127.0.0.1") || "").replace(/\/+$/, "");
const KEEP = args.includes("--keep");
const DRY_RUN = args.includes("--dry-run");

// ---------------------------------------------------------------------------
// Throwaway credentials. Never printed, never written to a fixture.
// ---------------------------------------------------------------------------

const RUN_ID = Math.random().toString(36).slice(2, 10);
const ACCOUNT = {
  nickname: "Fixture Capture",
  email: `fixture-${RUN_ID}@rag-platform.invalid`,
  password: `Fx-${RUN_ID}-${Math.random().toString(36).slice(2, 10)}`,
};

/**
 * The backend decrypts the password with conf/private.pem, so the client has to
 * encrypt with the matching public key: RSAES-PKCS1-V1_5 over the base64 of the
 * plaintext (api/utils/crypt.py crypt/decrypt). Reading the key out of the
 * running container keeps this honest — a stale copied key would silently break.
 *
 * node:crypto rather than node-forge: this script runs from the repo root where
 * node-forge does not resolve (it is a studio/frontend dependency), and the
 * primitive is the same. The frontend service will use node-forge, which is
 * already a dependency there.
 */
function encryptPassword(plain) {
  const pem = execFileSync("docker", ["exec", CONTAINER, "cat", "conf/public.pem"], {
    encoding: "utf8",
  });
  const base64Plain = Buffer.from(plain, "utf8").toString("base64");
  const encrypted = publicEncrypt(
    { key: pem, padding: constants.RSA_PKCS1_PADDING },
    Buffer.from(base64Plain, "utf8"),
  );
  return encrypted.toString("base64");
}

// ---------------------------------------------------------------------------
// Scrubbing
// ---------------------------------------------------------------------------

const SECRET_KEYS = new Set([
  "access_token",
  "api_key",
  "apikey",
  "authorization",
  "beta",
  "password",
  "secret",
  "secret_key",
  "token",
]);
const SECRET_KEY_HINTS = [/api[_-]?key/i, /secret/i, /token/i, /password/i, /credential/i];

const scrubbedPaths = new Set();

function isExactSecretKey(key) {
  return SECRET_KEYS.has(key.toLowerCase());
}

function isHintedSecretKey(key) {
  return SECRET_KEY_HINTS.some((pattern) => pattern.test(key));
}

/**
 * Recursive redaction. Records which JSON paths were touched so the fixture can
 * declare its own redactions instead of the reader having to trust a claim.
 *
 * Two tiers, because a single name-based rule is wrong in both directions:
 *   * exact match against SECRET_KEYS — always redacted, whatever the value is.
 *   * name *hint* (…token…, …secret…, …key…) — redacted only when the value is
 *     a string. Every credential this backend emits is a string; a number never
 *     is. Without that guard the /token/i hint eats the parser tuning knobs
 *     `chunk_token_num`, `batch_chunk_token_size` and `max_token`, which are
 *     integers the next phase's client has to send back verbatim — redacting
 *     them destroys the contract the fixture exists to record.
 */
function scrub(value, path = "$") {
  if (Array.isArray(value)) return value.map((item, index) => scrub(item, `${path}[${index}]`));
  if (value && typeof value === "object") {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      const childPath = `${path}.${key}`;
      const secret =
        isExactSecretKey(key) || (isHintedSecretKey(key) && typeof item === "string" && item !== "");
      if (secret) {
        scrubbedPaths.add(childPath);
        out[key] = `<redacted:${key}>`;
        continue;
      }
      out[key] = scrub(item, childPath);
    }
    return out;
  }
  if (typeof value === "string") {
    let text = value;
    if (text.includes(ACCOUNT.email)) {
      scrubbedPaths.add(path);
      text = text.replaceAll(ACCOUNT.email, "<redacted:email>");
    }
    if (text.includes(ACCOUNT.password)) {
      scrubbedPaths.add(path);
      text = text.replaceAll(ACCOUNT.password, "<redacted:password>");
    }
    return text;
  }
  return value;
}

const RECORDED_HEADERS = ["content-type", "cache-control", "connection", "transfer-encoding"];

function recordHeaders(headers) {
  const out = {};
  for (const name of RECORDED_HEADERS) {
    const value = headers.get(name);
    if (value) out[name] = value;
  }
  // Presence matters for the auth contract; the value must not be recorded.
  if (headers.get("authorization")) out.authorization = "<redacted:authorization>";
  return out;
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

let sessionToken = null;
const captured = [];

async function call(name, method, path, { body, form, auth = true, raw = false } = {}) {
  const headers = {};
  if (auth && sessionToken) headers.Authorization = sessionToken;
  let payload;
  if (form) {
    payload = form;
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }

  const started = Date.now();
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: payload,
    signal: AbortSignal.timeout(120_000),
  });
  const elapsedMs = Date.now() - started;

  const authHeader = response.headers.get("authorization");
  if (authHeader) sessionToken = authHeader;

  const text = raw ? null : await response.text();
  let parsed = null;
  if (text !== null) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { "<non-json body>": text.slice(0, 400) };
    }
  }

  const record = {
    name,
    request: {
      method,
      path,
      headers: auth && sessionToken ? { Authorization: "<redacted:authorization>" } : {},
      body: form
        ? "<multipart/form-data>"
        : body === undefined
          ? null
          : scrub(body, `${name}.request`),
    },
    response: {
      http_status: response.status,
      headers: recordHeaders(response.headers),
      body: parsed === null ? null : scrub(parsed, `${name}.response`),
    },
    elapsed_ms_note: elapsedMs > 5_000 ? "slow (amd64 emulation)" : undefined,
  };
  captured.push(record);
  console.log(
    `  ${method.padEnd(6)} ${path.padEnd(52)} ${response.status} ` +
      `${parsed && typeof parsed.code === "number" ? `code=${parsed.code}` : ""}`,
  );
  return { response, parsed, record };
}

function requireOk(step, parsed, response) {
  // The envelope reports business errors as HTTP 200 with code != 0, so HTTP
  // status alone is not a success signal.
  if (response.status >= 400) throw new Error(`${step}: HTTP ${response.status}`);
  if (parsed && typeof parsed.code === "number" && parsed.code !== 0) {
    throw new Error(`${step}: envelope code ${parsed.code} — ${parsed.message}`);
  }
  return parsed?.data;
}

// ---------------------------------------------------------------------------
// Flows
// ---------------------------------------------------------------------------

const created = { datasetIds: [], chatIds: [] };

async function flowAuth() {
  console.log("auth");
  const encrypted = encryptPassword(ACCOUNT.password);
  const register = await call("auth.register", "POST", "/api/v1/users", {
    auth: false,
    body: { nickname: ACCOUNT.nickname, email: ACCOUNT.email, password: encrypted },
  });
  requireOk("register", register.parsed, register.response);

  const login = await call("auth.login", "POST", "/api/v1/auth/login", {
    auth: false,
    body: { email: ACCOUNT.email, password: encrypted },
  });
  requireOk("login", login.parsed, login.response);
  if (!sessionToken) throw new Error("login returned no Authorization header");

  const me = await call("auth.whoami", "GET", "/api/v1/users/me");
  requireOk("whoami", me.parsed, me.response);

  await call("auth.login_channels", "GET", "/api/v1/auth/login/channels", { auth: false });
  await call("auth.unauthorized", "GET", "/api/v1/users/me", { auth: false });
}

async function flowDataset() {
  console.log("dataset");
  const create = await call("dataset.create", "POST", "/api/v1/datasets", {
    body: { name: `fixture-dataset-${RUN_ID}` },
  });
  const dataset = requireOk("dataset.create", create.parsed, create.response);
  const datasetId = dataset?.id;
  if (!datasetId) throw new Error("dataset.create returned no id");
  created.datasetIds.push(datasetId);

  await call("dataset.list", "GET", "/api/v1/datasets?page=1&page_size=10");
  await call("dataset.get", "GET", `/api/v1/datasets/${datasetId}`);
  await call("dataset.update", "PUT", `/api/v1/datasets/${datasetId}`, {
    body: { description: "captured by scripts/rag-platform/capture-fixtures.mjs" },
  });
  await call("dataset.not_found", "GET", "/api/v1/datasets/00000000000000000000000000000000");
  return datasetId;
}

async function flowDocument(datasetId) {
  console.log("document");
  const content =
    "Rag Platform fixture document.\n\n" +
    "The retrieval fixture needs a chunk to return, so this paragraph exists to be " +
    "indexed and retrieved. Rag Platform stores datasets, documents and chunks.\n";
  const form = new FormData();
  form.append("file", new Blob([content], { type: "text/plain" }), `fixture-${RUN_ID}.txt`);

  const upload = await call(
    "document.upload",
    "POST",
    `/api/v1/datasets/${datasetId}/documents`,
    { form },
  );
  const uploaded = requireOk("document.upload", upload.parsed, upload.response);
  const documentId = Array.isArray(uploaded) ? uploaded[0]?.id : uploaded?.id;
  if (!documentId) throw new Error("document.upload returned no id");

  await call("document.list", "GET", `/api/v1/datasets/${datasetId}/documents?page=1&page_size=10`);
  await call("document.parse", "POST", `/api/v1/datasets/${datasetId}/chunks`, {
    body: { document_ids: [documentId] },
  });
  await call(
    "document.status",
    "GET",
    `/api/v1/datasets/${datasetId}/documents?page=1&page_size=10`,
  );
  return documentId;
}

/**
 * Reads the parse state of one document out of a list response. `run` is the
 * TaskStatus enum written by DocumentService (`api/db/__init__.py`: 0 UNSTART,
 * 1 RUNNING, 2 CANCEL, 3 DONE, 4 FAIL) and `progress` goes negative when the
 * task executor gives up, with the reason in `progress_msg`.
 */
function readParseState(listResult, documentId) {
  const data = listResult.parsed?.data;
  const docs = data?.docs ?? data?.documents ?? (Array.isArray(data) ? data : []);
  return docs.find?.((doc) => doc.id === documentId) ?? null;
}

async function flowChunk(datasetId, documentId) {
  console.log("chunk");
  // Parsing is asynchronous; poll until a chunk exists, the task executor
  // reports failure, or the budget runs out. The failure check matters: with no
  // embedding model configured the parse task fails within seconds, and polling
  // for chunks that can never arrive would only waste two minutes and hide the
  // reason. The failing status response is itself the smoke-test evidence.
  let chunkId = null;
  let parseFailure = null;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const list = await call(
      `chunk.list${attempt === 0 ? "" : `.retry${attempt}`}`,
      "GET",
      `/api/v1/datasets/${datasetId}/documents/${documentId}/chunks?page=1&page_size=10`,
    );
    const data = list.parsed?.data;
    const chunks = data?.chunks ?? data?.items ?? (Array.isArray(data) ? data : []);
    if (chunks?.length) {
      chunkId = chunks[0].id ?? chunks[0].chunk_id;
      break;
    }

    const status = await call(
      `document.parse_status${attempt === 0 ? "" : `.retry${attempt}`}`,
      "GET",
      `/api/v1/datasets/${datasetId}/documents?page=1&page_size=10`,
    );
    const doc = readParseState(status, documentId);
    if (doc && (String(doc.run) === "4" || Number(doc.progress) < 0)) {
      parseFailure = {
        run: doc.run,
        progress: doc.progress,
        progress_msg: doc.progress_msg,
        chunk_count: doc.chunk_count ?? doc.chunk_num,
      };
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 6_000));
  }
  if (parseFailure) {
    console.log(`  parse failed: ${String(parseFailure.progress_msg).slice(0, 160)}`);
    return null;
  }
  if (!chunkId) {
    console.log("  (no chunk produced within budget — recorded as such)");
    return null;
  }
  await call(
    "chunk.get",
    "GET",
    `/api/v1/datasets/${datasetId}/documents/${documentId}/chunks/${chunkId}`,
  );
  // PATCH is the route in api/apps/restful_apis/chunk_api.py:1116; the PUT alias
  // lives in api/apps/backward_compat.py and is not the forward contract.
  await call(
    "chunk.update",
    "PATCH",
    `/api/v1/datasets/${datasetId}/documents/${documentId}/chunks/${chunkId}`,
    { body: { available: true } },
  );
  return chunkId;
}

async function flowRetrieval(datasetId, documentId) {
  console.log("retrieval");
  await call("retrieval.search", "POST", "/api/v1/retrieval", {
    body: {
      question: "What does Rag Platform store?",
      dataset_ids: [datasetId],
      document_ids: documentId ? [documentId] : undefined,
      page: 1,
      page_size: 5,
    },
  });
}

/**
 * `dataset_ids` is optional on chat creation, and passing it costs the capture
 * its whole chat/session/stream branch when the dataset has no chunks:
 * _validate_dataset_ids (chat_api.py:384) rejects any dataset with
 * `chunk_num == 0` before the chat row is written. A dataset-bound attempt is
 * still recorded first — the rejection is itself part of the contract the next
 * phase has to surface — then the capture proceeds with a dataset-free chat so
 * the session and streaming shapes get captured either way.
 */
async function flowChat(datasetId) {
  console.log("chat + session");
  const bound = await call("chat.create_with_dataset", "POST", "/api/v1/chats", {
    body: { name: `fixture-chat-bound-${RUN_ID}`, dataset_ids: [datasetId] },
  });
  const boundId = bound.parsed?.code === 0 ? bound.parsed?.data?.id : null;
  if (boundId) created.chatIds.push(boundId);

  const create = boundId
    ? bound
    : await call("chat.create", "POST", "/api/v1/chats", {
        body: { name: `fixture-chat-${RUN_ID}` },
      });
  const chat = requireOk("chat.create", create.parsed, create.response);
  const chatId = chat?.id;
  if (!chatId) throw new Error("chat.create returned no id");
  if (!boundId) created.chatIds.push(chatId);

  await call("chat.list", "GET", "/api/v1/chats?page=1&page_size=10");
  await call("chat.get", "GET", `/api/v1/chats/${chatId}`);

  const session = await call("session.create", "POST", `/api/v1/chats/${chatId}/sessions`, {
    body: { name: `fixture-session-${RUN_ID}` },
  });
  const sessionData = requireOk("session.create", session.parsed, session.response);
  const sessionId = sessionData?.id;

  await call("session.list", "GET", `/api/v1/chats/${chatId}/sessions?page=1&page_size=10`);
  if (sessionId) {
    await call("session.get", "GET", `/api/v1/chats/${chatId}/sessions/${sessionId}`);
  }
  return { chatId, sessionId };
}

/**
 * Streaming is captured differently: the value of the fixture is the SSE frame
 * shape, so frames are recorded individually rather than as one body. Frame
 * count is capped — the contract is in the first and last frames.
 */
async function flowStream(chatId, sessionId) {
  console.log("stream");
  // The forward route is POST /api/v1/chat/completions (chat_api.py:1228) with
  // `chat_id` in the body. `/api/v1/chats/<chat_id>/completions` still answers
  // but is the deprecated alias (backward_compat.py:91), which logs a
  // deprecation warning and forwards to the same handler — capturing it would
  // record a contract the next phase must not build on.
  const path = "/api/v1/chat/completions";
  const body = {
    // _normalize_completion_messages (chat_api.py:240) accepts either
    // `messages` or the legacy `question`; `messages` is the forward shape.
    messages: [{ role: "user", content: "What does Rag Platform store?" }],
    chat_id: chatId,
    stream: true,
    session_id: sessionId ?? undefined,
  };
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: sessionToken },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });

  // A streaming request does not always get a stream back. When the request is
  // rejected before generation starts, Quart answers the same route with a
  // single JSON envelope (`content-type: application/json`), which has no
  // `\n\n` and so yields zero SSE frames. Recording it as "0 frames" would
  // throw away the only thing the response contains — the rejection reason.
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("text/event-stream")) {
    const raw = await response.text();
    let envelope;
    try {
      envelope = scrub(JSON.parse(raw), "stream.body");
    } catch {
      envelope = { "<non-json body>": raw.slice(0, 600) };
    }
    captured.push({
      name: "stream.completion",
      request: {
        method: "POST",
        path,
        headers: { Authorization: "<redacted:authorization>" },
        body: scrub(body, "stream.request"),
      },
      response: {
        http_status: response.status,
        headers: recordHeaders(response.headers),
        note: "Response was not text/event-stream; recorded as a single envelope.",
        body: envelope,
      },
    });
    console.log(
      `  POST   ${path.padEnd(52)} ${response.status} non-stream code=${envelope?.code ?? "?"}`,
    );
    return;
  }

  const frames = [];
  if (response.body) {
    const decoder = new TextDecoder();
    let buffer = "";
    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        if (frame) frames.push(frame);
        boundary = buffer.indexOf("\n\n");
      }
      if (frames.length >= 40) break;
    }
  }

  const parseFrame = (frame) => {
    const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
    if (!dataLine) return { "<raw frame>": frame.slice(0, 300) };
    const payload = dataLine.slice(5).trim();
    if (payload === "[DONE]") return "[DONE]";
    try {
      return scrub(JSON.parse(payload), "stream.frame");
    } catch {
      return { "<non-json data>": payload.slice(0, 300) };
    }
  };

  captured.push({
    name: "stream.completion",
    request: {
      method: "POST",
      path,
      headers: { Authorization: "<redacted:authorization>" },
      body: scrub(body, "stream.request"),
    },
    response: {
      http_status: response.status,
      headers: recordHeaders(response.headers),
      sse: {
        frame_count_captured: frames.length,
        truncated: frames.length >= 40,
        first_frames: frames.slice(0, 3).map(parseFrame),
        last_frames: frames.slice(-3).map(parseFrame),
      },
    },
  });
  console.log(`  POST   ${path.padEnd(52)} ${response.status} frames=${frames.length}`);
}

async function cleanup() {
  if (KEEP) {
    console.log("cleanup skipped (--keep)");
    return;
  }
  console.log("cleanup");
  // Both deletes are collection-level with an `ids` body: dataset_api.py:164 and
  // chat_api.py:775. The per-id `DELETE /api/v1/chats/<id>` (chat_api.py:761)
  // exists too, but `DELETE /api/v1/datasets/<id>` does not — calling it returns
  // 200 with `code:100, <MethodNotAllowed '405: Method Not Allowed'>`. Using the
  // collection form for both keeps cleanup uniform and matches the inventory.
  if (created.chatIds.length) {
    await call("cleanup.chat", "DELETE", "/api/v1/chats", {
      body: { ids: created.chatIds },
    }).catch(() => {});
  }
  if (created.datasetIds.length) {
    await call("cleanup.dataset", "DELETE", "/api/v1/datasets", {
      body: { ids: created.datasetIds },
    }).catch(() => {});
  }
  await call("auth.logout", "POST", "/api/v1/auth/logout").catch(() => {});
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

async function main() {
  const ping = await fetch(`${BASE}/api/v1/system/ping`, {
    signal: AbortSignal.timeout(15_000),
  }).catch((error) => {
    throw new Error(`backend not reachable at ${BASE}: ${error.message}`);
  });
  if (!ping.ok) throw new Error(`backend ping returned HTTP ${ping.status}`);

  if (DRY_RUN) {
    console.log(`backend reachable at ${BASE}; dry run, nothing captured`);
    return;
  }

  let failure = null;
  try {
    await flowAuth();
    const datasetId = await flowDataset();
    const documentId = await flowDocument(datasetId);
    const chunkId = await flowChunk(datasetId, documentId);
    await flowRetrieval(datasetId, documentId);
    const { chatId, sessionId } = await flowChat(datasetId);
    await flowStream(chatId, sessionId);
    if (!chunkId) console.log("note: chunk fixtures recorded without a chunk id");
  } catch (error) {
    failure = error;
    console.error(`capture stopped: ${error.message}`);
  } finally {
    await cleanup().catch((error) => console.error(`cleanup failed: ${error.message}`));
  }

  if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });

  const groups = new Map();
  for (const record of captured) {
    const group = record.name.split(".")[0];
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(record);
  }

  for (const [group, records] of groups) {
    const file = join(OUT_DIR, `${group}.json`);
    const payload = {
      generated_by: "scripts/rag-platform/capture-fixtures.mjs",
      captured_against: {
        base_url: BASE,
        source_image: "infiniflow/ragflow:v0.26.4",
        proxy_scheme: "python",
        api_version: "v1",
      },
      redaction: {
        policy:
          "Secrets never reach this file. Keys matching api_key/secret/token/password/" +
          "credential and the Authorization header are replaced with placeholders; the " +
          "throwaway account's e-mail and password are replaced wherever they appear in " +
          "any string value.",
        placeholders: ["<redacted:authorization>", "<redacted:email>", "<redacted:password>"],
      },
      interactions: records,
    };
    writeFileSync(file, `${JSON.stringify(payload, null, 2)}\n`);
    console.log(`wrote ${relative(FRONTEND_ROOT, file)} (${records.length} interactions)`);
  }

  if (failure) process.exitCode = 1;
}

await main();
