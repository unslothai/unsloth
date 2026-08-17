#!/usr/bin/env node

/** Authenticated, secret-safe Phase 13 hybrid route smoke. */
import { execFileSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";

const base = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");
const runId = Math.random().toString(36).slice(2, 10);
const account = {
  nickname: "Phase 13 Smoke",
  email: `phase13-${runId}@rag-platform.invalid`,
  password: `P13-${runId}-${Math.random().toString(36).slice(2, 10)}`,
};
let authorization = "";
let memoryId = "";
let searchId = "";

function encryptedPassword() {
  const pem = execFileSync("docker", ["exec", "rag-platform-backend", "cat", "conf/public.pem"], { encoding: "utf8" });
  return publicEncrypt(
    { key: pem, padding: constants.RSA_PKCS1_PADDING },
    Buffer.from(Buffer.from(account.password).toString("base64")),
  ).toString("base64");
}

function envelopeCode(payload) {
  if (!payload || typeof payload !== "object") return null;
  return typeof payload.code === "number" || typeof payload.code === "string" ? Number(payload.code) : null;
}

async function request(method, path, body, { authenticated = true, timeoutMs = 45_000 } = {}) {
  const headers = new Headers();
  if (authenticated && authorization) headers.set("Authorization", authorization);
  if (body !== undefined) headers.set("Content-Type", "application/json");
  const response = await fetch(`${base}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  authorization = response.headers.get("authorization") || authorization;
  const contentType = response.headers.get("content-type")?.toLowerCase() || "";
  const payload = contentType.includes("application/json") ? await response.json().catch(() => null) : null;
  if (!contentType.includes("application/json")) await response.body?.cancel().catch(() => undefined);
  const code = envelopeCode(payload);
  console.log(`${method.padEnd(6)} ${path.padEnd(72)} ${response.status} code=${code ?? "—"}`);
  if ([404, 405, 502, 503, 504].includes(response.status)) throw new Error(`${method} ${path} did not reach an active handler`);
  return { response, payload, code, contentType };
}

async function success(method, path, body, options) {
  const result = await request(method, path, body, options);
  if (!result.response.ok || result.code !== 0) throw new Error(`${method} ${path} failed: HTTP ${result.response.status}, code ${result.code ?? "—"}`);
  return result.payload?.data;
}

async function requireAuthBoundary(method, path) {
  const result = await request(method, path, undefined, { authenticated: false });
  if (result.response.status !== 401 && result.code !== 401) throw new Error(`${method} ${path} did not enforce authentication`);
}

const encrypted = encryptedPassword();
try {
  await success("POST", "/api/v1/users", { nickname: account.nickname, email: account.email, password: encrypted }, { authenticated: false });
  await success("POST", "/api/v1/auth/login", { email: account.email, password: encrypted }, { authenticated: false });
  if (!authorization) throw new Error("Login returned no Authorization header");

  await requireAuthBoundary("GET", "/api/v1/memories");
  await requireAuthBoundary("GET", "/api/v1/messages?memory_id=probe");
  await requireAuthBoundary("GET", "/api/v1/searches");

  const createdMemory = await success("POST", "/api/v1/memories", {
    name: `Phase 13 ${runId}`,
    memory_type: ["raw"],
    embd_id: `phase13-embed-${runId}`,
    llm_id: `phase13-chat-${runId}`,
  });
  memoryId = createdMemory?.id || "";
  if (!memoryId) throw new Error("Memory create returned no id");
  const memoryPath = `/api/v1/memories/${encodeURIComponent(memoryId)}`;
  await success("GET", "/api/v1/memories?page=1&page_size=20");
  await success("GET", `${memoryPath}/config`);
  await success("GET", `${memoryPath}?page=1&page_size=20`);
  await success("PUT", memoryPath, { permissions: "me", memory_size: 1048576, forgetting_policy: "FIFO" });
  await request("GET", `/api/v1/messages?memory_id=${encodeURIComponent(memoryId)}&limit=10`);
  await request("GET", `/api/v1/messages/search?memory_id=${encodeURIComponent(memoryId)}&query=phase13&top_n=5`);

  const createdSearch = await success("POST", "/api/v1/searches", { name: `Phase 13 ${runId}`, description: "runtime smoke" });
  searchId = createdSearch?.search_id || createdSearch?.id || createdSearch || "";
  if (!searchId || typeof searchId !== "string") throw new Error("Search create returned no id");
  const searchPath = `/api/v1/searches/${encodeURIComponent(searchId)}`;
  await success("GET", "/api/v1/searches?page=1&page_size=20");
  const detail = await success("GET", searchPath);
  await success("PUT", searchPath, { name: detail?.name || `Phase 13 ${runId}`, description: "runtime smoke updated", search_config: detail?.search_config || {} });
  // Missing dataset is an expected business validation response after the active handler.
  await request("POST", `${searchPath}/completion`, { question: "phase 13 route probe" });
  await request("POST", `${searchPath}/completions`, { question: "phase 13 route probe" });

  console.log("Phase 13 authenticated hybrid route smoke: PASS");
} finally {
  if (authorization && memoryId) await success("DELETE", `/api/v1/memories/${encodeURIComponent(memoryId)}`).catch(() => undefined);
  if (authorization && searchId) await success("DELETE", `/api/v1/searches/${encodeURIComponent(searchId)}`).catch(() => undefined);
  if (authorization) await success("POST", "/api/v1/auth/logout").catch(() => undefined);
}
