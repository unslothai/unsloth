#!/usr/bin/env node

/** Authenticated, secret-safe Phase 12 hybrid runtime smoke. */
import { execFileSync, spawnSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";

const base = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");
const runId = Math.random().toString(36).slice(2, 10);
const logMarker = `p12-marker-${runId}`;
const account = {
  nickname: "Phase 12 Smoke",
  email: `phase12-${runId}@rag-platform.invalid`,
  password: `P12-${runId}-${Math.random().toString(36).slice(2, 10)}`,
};
let authorization = "";
let connectorId = "";
let folderId = "";
let fileId = "";

function encryptedPassword() {
  const pem = execFileSync(
    "docker",
    ["exec", "rag-platform-backend", "cat", "conf/public.pem"],
    { encoding: "utf8" },
  );
  return publicEncrypt(
    { key: pem, padding: constants.RSA_PKCS1_PADDING },
    Buffer.from(Buffer.from(account.password).toString("base64")),
  ).toString("base64");
}

function envelopeCode(payload) {
  if (!payload || typeof payload !== "object") return null;
  return typeof payload.code === "number" || typeof payload.code === "string"
    ? Number(payload.code)
    : null;
}

async function request(
  method,
  path,
  body,
  { authenticated = true, displayPath = path, timeoutMs = 45_000 } = {},
) {
  const headers = new Headers();
  if (authenticated && authorization) headers.set("Authorization", authorization);
  const isForm = body instanceof FormData;
  if (body !== undefined && !isForm) headers.set("Content-Type", "application/json");
  const response = await fetch(`${base}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  authorization = response.headers.get("authorization") || authorization;
  const contentType = response.headers.get("content-type")?.toLowerCase() || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : null;
  if (!contentType.includes("application/json")) await response.body?.cancel().catch(() => undefined);
  const code = envelopeCode(payload);
  console.log(`${method.padEnd(6)} ${displayPath.padEnd(66)} ${response.status} code=${code ?? "—"}`);
  if ([404, 405, 502, 503, 504].includes(response.status)) {
    throw new Error(`${method} ${displayPath} did not reach an active handler`);
  }
  return { response, payload, code };
}

async function success(method, path, body, options) {
  const result = await request(method, path, body, options);
  if (!result.response.ok || result.code !== 0) {
    throw new Error(`${method} ${options?.displayPath ?? path} failed: HTTP ${result.response.status}, code ${result.code ?? "—"}`);
  }
  return result.payload?.data;
}

async function requireAuthBoundary(method, path) {
  const result = await request(method, path, undefined, { authenticated: false });
  if (result.response.status !== 401 && result.code !== 401) {
    throw new Error(`${method} ${path} did not enforce authentication`);
  }
}

const encrypted = encryptedPassword();

try {
  await success("POST", "/api/v1/users", { nickname: account.nickname, email: account.email, password: encrypted }, { authenticated: false });
  await success("POST", "/api/v1/auth/login", { email: account.email, password: encrypted }, { authenticated: false });
  if (!authorization) throw new Error("Login returned no Authorization header");

  await requireAuthBoundary("GET", "/api/v1/connectors");
  await requireAuthBoundary("GET", "/api/v1/files");

  const connector = await success("POST", "/api/v1/connectors", {
    name: `Phase 12 ${runId}`,
    source: "rest_api",
    config: {
      url: "https://jsonplaceholder.typicode.com/posts",
      method: "GET",
      auth_type: "none",
      content_fields: ["title", "body"],
      max_pages: 1,
      request_delay: 0,
    },
    refresh_freq: 60,
    prune_freq: 120,
    timeout_secs: 3,
  });
  connectorId = connector?.id || "";
  if (!connectorId) throw new Error("Connector create returned no id");
  const connectorPath = `/api/v1/connectors/${encodeURIComponent(connectorId)}`;
  await success("GET", "/api/v1/connectors");
  await success("GET", connectorPath);
  await success("PATCH", connectorPath, { refresh_freq: 30, prune_freq: 90, timeout_secs: 2 });
  await success("POST", `${connectorPath}/test`);
  await success("GET", `${connectorPath}/logs?page=1&page_size=10`);

  const folder = await success("POST", "/api/v1/files", {
    name: `phase12-${runId}`,
    type: "folder",
  });
  folderId = folder?.id || "";
  if (!folderId) throw new Error("Folder create returned no id");
  const upload = new FormData();
  upload.set("parent_id", folderId);
  upload.append("file", new Blob(["phase 12 smoke\n"], { type: "text/plain" }), `phase12-${runId}.txt`);
  const uploaded = await success("POST", "/api/v1/files", upload, { timeoutMs: 120_000 });
  fileId = Array.isArray(uploaded) ? uploaded[0]?.id || "" : uploaded?.id || "";
  if (!fileId) throw new Error("File upload returned no id");
  await success("GET", `/api/v1/files?parent_id=${encodeURIComponent(folderId)}&page=1&page_size=20`);
  await success("GET", `/api/v1/files/${encodeURIComponent(fileId)}/parent`);
  await success("GET", `/api/v1/files/${encodeURIComponent(fileId)}/ancestors`);
  await request("GET", `/api/v1/files/${encodeURIComponent(fileId)}`, undefined, { displayPath: "/api/v1/files/:id" });
  await success("POST", "/api/v1/files/move", { src_file_ids: [fileId], dest_file_id: folderId, new_name: `renamed-${runId}.txt` });
  await success("GET", `/api/v1/files/${encodeURIComponent(fileId)}/versions`);

  const scope = `/api/v1/folders/${encodeURIComponent(folderId)}`;
  await success("GET", `${scope}/changes`);
  const commit = await success("POST", `${scope}/commits`, {
    message: `Phase 12 ${runId}`,
    files: [{ file_id: fileId, file_name: `renamed-${runId}.txt`, operation: "add", content: "phase 12 smoke\n" }],
  });
  const commitId = commit?.id || "";
  if (!commitId) throw new Error("Commit create returned no id");
  await success("GET", `${scope}/commits?page=1&page_size=20`);
  await success("GET", `${scope}/commits/${encodeURIComponent(commitId)}`);
  await success("GET", `${scope}/commits/${encodeURIComponent(commitId)}/files`);
  await success("GET", `${scope}/commits/${encodeURIComponent(commitId)}/tree`);
  await success("GET", `${scope}/commits/${encodeURIComponent(commitId)}/files/${encodeURIComponent(fileId)}/content`);
  await success("GET", `${scope}/commits/diff?from=${encodeURIComponent(commitId)}&to=${encodeURIComponent(commitId)}`);

  const oauthCredentials = {
    web: {
      client_id: `${logMarker}.apps.invalid`,
      client_secret: `not-a-secret-${runId}`,
      project_id: "rag-platform-smoke",
      auth_uri: "https://accounts.google.com/o/oauth2/auth",
      token_uri: "https://oauth2.googleapis.com/token",
      auth_provider_x509_cert_url: "https://www.googleapis.com/oauth2/v1/certs",
      redirect_uris: [`${base}/connector-oauth/google-drive/callback`],
    },
  };
  await success("POST", "/api/v1/connectors/google/oauth/web/start?type=google-drive", {
    credentials: oauthCredentials,
    redirect_uri: `${base}/connector-oauth/google-drive/callback`,
  }, { displayPath: "/api/v1/connectors/google/oauth/web/start?type=google-drive" });
  await request("GET", "/connector-oauth/google-drive/callback?error=access_denied", undefined, { authenticated: false, displayPath: "/connector-oauth/google-drive/callback (frontend)" });
  await request("GET", "/api/v1/connectors/google-drive/oauth/web/callback?state=invalid-phase12&error=access_denied", undefined, { authenticated: false, displayPath: "/api/v1/connectors/google-drive/oauth/web/callback" });

  const logResult = spawnSync("docker", ["logs", "rag-platform-backend"], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (logResult.status !== 0) throw new Error("Backend logs could not be inspected");
  const logs = `${logResult.stdout ?? ""}${logResult.stderr ?? ""}`;
  if (logs.includes(logMarker)) throw new Error("OAuth credential marker appeared in backend logs");
  console.log("OAuth credential log-redaction probe: PASS");
  console.log("Phase 12 authenticated hybrid route smoke: PASS");
} finally {
  if (authorization && connectorId) await success("DELETE", `/api/v1/connectors/${encodeURIComponent(connectorId)}`).catch(() => undefined);
  if (authorization && folderId) await success("DELETE", "/api/v1/files", { ids: [folderId] }).catch(() => undefined);
  if (authorization) await success("POST", "/api/v1/auth/logout").catch(() => undefined);
}
