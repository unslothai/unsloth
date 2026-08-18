#!/usr/bin/env node

/**
 * Secret-safe Phase 10 hybrid runtime smoke.
 *
 * Proves representative active metadata/tag/graph/artifact/ingestion/skill
 * handlers are registered (an authenticated business error is acceptable for a
 * deliberately nonexistent dataset), and proves the owned current Go runtime
 * serves Phase 10 routes while the Python-only navigation-search declaration
 * remains explicitly absent. Credentials are generated in memory and never
 * printed or persisted.
 */
import { execFileSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";

const proxyBase = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");
const directBases = ["http://127.0.0.1:9380", "http://127.0.0.1:9384"];
const runId = Math.random().toString(36).slice(2, 10);
const password = `P10-${runId}-${Math.random().toString(36).slice(2, 10)}`;
const email = `phase10-${runId}@rag-platform.invalid`;
let authorization = "";

const encrypted = publicEncrypt(
  {
    key: execFileSync(
      "docker",
      ["exec", "rag-platform-backend", "cat", "conf/public.pem"],
      { encoding: "utf8" },
    ),
    padding: constants.RSA_PKCS1_PADDING,
  },
  Buffer.from(Buffer.from(password).toString("base64")),
).toString("base64");

function code(payload) {
  return payload && typeof payload === "object" && "code" in payload
    ? Number(payload.code)
    : null;
}

async function call(base, method, path, body, authenticated = true) {
  const headers = new Headers();
  if (authenticated && authorization) headers.set("Authorization", authorization);
  if (body !== undefined) headers.set("Content-Type", "application/json");
  const response = await fetch(`${base}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(30_000),
  });
  authorization = response.headers.get("authorization") || authorization;
  const payload = method === "HEAD" ? null : await response.json().catch(() => null);
  return { response, payload };
}

async function authSuccess(method, path, body, authenticated = true) {
  const result = await call(proxyBase, method, path, body, authenticated);
  if (!result.response.ok || code(result.payload) !== 0)
    throw new Error(`${method} ${path} failed: HTTP ${result.response.status}`);
  return result.payload?.data;
}

async function registered(method, path, body) {
  const result = await call(proxyBase, method, path, body);
  const message =
    result.payload && typeof result.payload === "object" && "message" in result.payload
      ? String(result.payload.message).replace(/[\r\n]+/g, " ").slice(0, 120)
      : "";
  console.log(
    `${method.padEnd(6)} ${path.padEnd(64)} ${result.response.status} code=${code(result.payload) ?? "—"}${message ? ` message=${message}` : ""}`,
  );
  if ([404, 405, 502, 503, 504].includes(result.response.status))
    throw new Error(`${method} ${path} did not reach the active runtime handler`);
}

async function absentEverywhere(method, path) {
  for (const base of [proxyBase, ...directBases]) {
    const result = await call(base, method, path);
    const message =
      result.payload && typeof result.payload === "object" && "message" in result.payload
        ? String(result.payload.message)
        : "";
    const wrappedMissing =
      code(result.payload) === 100 &&
      (message.includes("MethodNotAllowed") || message.includes("NotFound"));
    console.log(
      `${method.padEnd(6)} ${path.padEnd(52)} ${new URL(base).port || "80"}=${result.response.status}/${code(result.payload) ?? "—"}`,
    );
    if (
      result.response.status !== 404 &&
      result.response.status !== 405 &&
      !wrappedMissing
    )
      throw new Error(`${method} ${path} unexpectedly exists at ${base}`);
  }
}

try {
  await authSuccess("POST", "/api/v1/users", { nickname: "Phase 10 Smoke", email, password: encrypted }, false);
  await authSuccess("POST", "/api/v1/auth/login", { email, password: encrypted }, false);
  if (!authorization) throw new Error("Login returned no Authorization header");

  const missing = "00000000000000000000000000000000";
  await registered("GET", `/api/v1/datasets/${missing}/metadata/config`);
  await registered("GET", `/api/v1/datasets/${missing}/tags`);
  await registered("GET", `/api/v1/datasets/${missing}/graph`);
  await registered("GET", `/api/v1/datasets/${missing}/artifacts`);
  await registered("GET", `/api/v1/datasets/${missing}/ingestions/summary`);
  await registered("GET", `/api/v1/datasets/${missing}/skills`);
  await registered("GET", "/api/v1/skills/spaces");
  await registered("POST", "/api/v1/skills/search", { space_id: "default", query: "", page: 1, page_size: 1 });

  await registered("GET", `/api/v1/datasets/${missing}/compilation/status`);
  await registered("GET", `/api/v1/datasets/${missing}/artifacts/topics`);
  await registered("GET", `/api/v1/datasets/${missing}/artifacts/structure`);
  await registered("GET", `/api/v1/datasets/${missing}/artifacts/alteration`);
  await registered("GET", `/api/v1/datasets/${missing}/navigation`);
  await absentEverywhere("GET", `/api/v1/datasets/${missing}/navigation/search?q=rag`);

  console.log("Phase 10 authenticated hybrid/owned-Go runtime smoke: PASS");
} finally {
  if (authorization)
    await authSuccess("POST", "/api/v1/auth/logout").catch(() => undefined);
}
