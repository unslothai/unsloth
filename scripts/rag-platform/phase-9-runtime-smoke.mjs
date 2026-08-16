#!/usr/bin/env node

/**
 * Authenticated, secret-safe Phase 9 hybrid route smoke.
 *
 * Creates a throwaway tenant user and two API tokens, verifies the selected
 * Python/Go contracts through nginx, revokes both tokens, and never prints or
 * persists credentials, bearer headers, provider secrets, or token values.
 * Langfuse mutations are checked at the auth boundary because a successful
 * save requires real third-party credentials and would alter tenant config.
 */
import { execFileSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";

const base = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");
const runId = Math.random().toString(36).slice(2, 10);
const account = {
  nickname: "Phase 9 Smoke",
  email: `phase9-${runId}@rag-platform.invalid`,
  password: `P9-${runId}-${Math.random().toString(36).slice(2, 10)}`,
};
let authorization = "";
let canonicalToken = "";
let aliasToken = "";

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
  { authenticated = true, displayPath = path } = {},
) {
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
  const payload = await response.json().catch(() => null);
  const code = envelopeCode(payload);
  console.log(
    `${method.padEnd(6)} ${displayPath.padEnd(52)} ${response.status} code=${code ?? "—"}`,
  );
  if ([404, 405, 502, 503, 504].includes(response.status)) {
    throw new Error(
      `${method} ${displayPath} did not reach its active handler: HTTP ${response.status}`,
    );
  }
  return { response, payload, code };
}

async function requiredSuccess(method, path, body, options) {
  const result = await request(method, path, body, options);
  if (!result.response.ok || result.code !== 0) {
    throw new Error(
      `${method} ${options?.displayPath ?? path} failed: HTTP ${result.response.status}, code ${result.code ?? "—"}`,
    );
  }
  return result.payload?.data;
}

async function requireAuthBoundary(method, path) {
  const result = await request(method, path, undefined, {
    authenticated: false,
  });
  if (result.response.status !== 401 && result.code !== 401) {
    throw new Error(`${method} ${path} did not enforce the unauthenticated boundary`);
  }
}

const encrypted = encryptedPassword();

try {
  for (const method of ["GET", "POST", "PUT", "DELETE"]) {
    await requireAuthBoundary(method, "/api/v1/langfuse/api-key");
  }

  await requiredSuccess(
    "POST",
    "/api/v1/users",
    { nickname: account.nickname, email: account.email, password: encrypted },
    { authenticated: false },
  );
  await requiredSuccess(
    "POST",
    "/api/v1/auth/login",
    { email: account.email, password: encrypted },
    { authenticated: false },
  );
  if (!authorization) throw new Error("Login returned no Authorization header");

  const status = await requiredSuccess("GET", "/api/v1/system/status");
  if (!status || typeof status !== "object") {
    throw new Error("System status returned no object");
  }
  const stats = await requiredSuccess("GET", "/api/v1/system/stats");
  if (!stats || typeof stats !== "object") {
    throw new Error("System stats returned no object");
  }
  await requiredSuccess("GET", "/api/v1/langfuse/api-key");

  const canonicalCreated = await requiredSuccess(
    "POST",
    "/api/v1/system/tokens",
  );
  canonicalToken = canonicalCreated?.token || "";
  if (!canonicalToken) throw new Error("Canonical token create returned no token");
  const canonicalList = await requiredSuccess("GET", "/api/v1/system/tokens");
  if (!Array.isArray(canonicalList) || !canonicalList.some((item) => item?.token === canonicalToken)) {
    throw new Error("Canonical token was not returned by the list contract");
  }

  const aliasCreated = await requiredSuccess("POST", "/api/v1/system/keys", {});
  aliasToken = aliasCreated?.token || "";
  if (!aliasToken) throw new Error("System key alias create returned no token");
  const aliasList = await requiredSuccess("GET", "/api/v1/system/keys");
  if (!Array.isArray(aliasList) || !aliasList.some((item) => item?.token === aliasToken)) {
    throw new Error("System key alias was not returned by the list contract");
  }

  await requiredSuccess(
    "DELETE",
    `/api/v1/system/tokens/${encodeURIComponent(canonicalToken)}`,
    undefined,
    { displayPath: "/api/v1/system/tokens/:key" },
  );
  canonicalToken = "";
  await requiredSuccess(
    "DELETE",
    `/api/v1/system/keys/${encodeURIComponent(aliasToken)}`,
    undefined,
    { displayPath: "/api/v1/system/keys/:key" },
  );
  aliasToken = "";

  console.log("Phase 9 authenticated hybrid route smoke: PASS");
} finally {
  if (authorization && canonicalToken) {
    await requiredSuccess(
      "DELETE",
      `/api/v1/system/tokens/${encodeURIComponent(canonicalToken)}`,
      undefined,
      { displayPath: "/api/v1/system/tokens/:key" },
    ).catch(() => undefined);
  }
  if (authorization && aliasToken) {
    await requiredSuccess(
      "DELETE",
      `/api/v1/system/keys/${encodeURIComponent(aliasToken)}`,
      undefined,
      { displayPath: "/api/v1/system/keys/:key" },
    ).catch(() => undefined);
  }
  if (authorization) {
    await requiredSuccess("POST", "/api/v1/auth/logout").catch(() => undefined);
  }
}
