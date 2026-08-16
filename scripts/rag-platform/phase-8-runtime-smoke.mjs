#!/usr/bin/env node

/**
 * Authenticated, secret-safe Phase 8 hybrid route smoke.
 *
 * The local runtime intentionally has no default chat/audio provider. This
 * proves active proxy reachability, auth and handler contract selection without
 * treating provider-configuration errors as route failures. Provider-success
 * SSE remains covered by the source-verified deterministic fixture.
 */
import { execFileSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";

const base = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");
const runId = Math.random().toString(36).slice(2, 10);
const account = {
  nickname: "Phase 8 Smoke",
  email: `phase8-${runId}@rag-platform.invalid`,
  password: `P8-${runId}-${Math.random().toString(36).slice(2, 10)}`,
};
let authorization = "";
let chatId = "";
let sessionId = "";

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

function safeCode(payload) {
  return payload &&
    typeof payload === "object" &&
    (typeof payload.code === "number" || typeof payload.code === "string")
    ? payload.code
    : null;
}

async function rawRequest(method, path, body, authenticated = true) {
  const headers = new Headers();
  if (authenticated && authorization) headers.set("Authorization", authorization);
  let requestBody;
  if (body instanceof FormData) requestBody = body;
  else if (body !== undefined) {
    headers.set("Content-Type", "application/json");
    requestBody = JSON.stringify(body);
  }
  const response = await fetch(`${base}${path}`, {
    method,
    headers,
    body: requestBody,
    signal: AbortSignal.timeout(30_000),
  });
  authorization = response.headers.get("authorization") || authorization;
  if ([404, 405, 502, 503, 504].includes(response.status)) {
    throw new Error(`${method} ${path} did not reach its active handler: HTTP ${response.status}`);
  }
  return response;
}

async function envelopeRequest(method, path, body, authenticated = true) {
  const response = await rawRequest(method, path, body, authenticated);
  const payload = await response.json().catch(() => null);
  const code = safeCode(payload);
  console.log(
    `${method.padEnd(6)} ${path.padEnd(82)} ${response.status} code=${code ?? "—"}`,
  );
  return { response, payload, code };
}

async function requiredSuccess(method, path, body, authenticated = true) {
  const result = await envelopeRequest(method, path, body, authenticated);
  if (!result.response.ok || result.code !== 0) {
    throw new Error(`${method} ${path} failed: HTTP ${result.response.status}, code ${result.code ?? "—"}`);
  }
  return result.payload?.data;
}

async function probeCompletion() {
  const path = "/api/v1/chat/completions";
  const response = await rawRequest("POST", path, {
    chat_id: chatId,
    session_id: sessionId,
    question: "Phase 8 route smoke",
    stream: true,
    legacy: false,
  });
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();
  let code = null;
  let terminal = false;
  if (contentType.includes("text/event-stream")) {
    for (const block of text.split(/\r?\n\r?\n/)) {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      const frame = JSON.parse(data);
      code = safeCode(frame);
      if (frame?.data === true) terminal = true;
    }
    if (code === 0 && !terminal) {
      throw new Error("Completion SSE ended without data:true");
    }
  } else {
    code = safeCode(JSON.parse(text || "null"));
  }
  console.log(
    `${"POST".padEnd(6)} ${path.padEnd(82)} ${response.status} code=${code ?? "—"} content-type=${contentType.split(";", 1)[0]}`,
  );
}

const encrypted = encryptedPassword();

try {
  await requiredSuccess(
    "POST",
    "/api/v1/users",
    { nickname: account.nickname, email: account.email, password: encrypted },
    false,
  );
  await requiredSuccess(
    "POST",
    "/api/v1/auth/login",
    { email: account.email, password: encrypted },
    false,
  );
  if (!authorization) throw new Error("Login returned no Authorization header");

  const chat = await requiredSuccess("POST", "/api/v1/chats", {
    name: `phase-8-chat-${runId}`,
    dataset_ids: [],
  });
  chatId = chat?.id || "";
  if (!chatId) throw new Error("Chat create returned no id");
  const session = await requiredSuccess(
    "POST",
    `/api/v1/chats/${chatId}/sessions`,
    { name: "Phase 8 Smoke" },
  );
  sessionId = session?.id || "";
  if (!sessionId) throw new Error("Session create returned no id");

  await probeCompletion();
  await envelopeRequest(
    "PUT",
    `/api/v1/chats/${chatId}/sessions/${sessionId}/messages/missing-turn/feedback`,
    { thumbup: false, feedback: "route smoke" },
  );
  await envelopeRequest("POST", "/api/v1/chat/mindmap", {
    question: "Phase 8 route smoke",
    kb_ids: [],
  });
  await envelopeRequest("POST", "/api/v1/chat/recommendation", {
    question: "Phase 8 route smoke",
  });
  await envelopeRequest("POST", "/api/v1/chat/audio/speech", {
    text: "Phase 8 route smoke",
  });
  const audio = new FormData();
  audio.append("file", new Blob(["RIFF"], { type: "audio/wav" }), "smoke.wav");
  audio.append("stream", "false");
  await envelopeRequest("POST", "/api/v1/chat/audio/transcription", audio);

  console.log(
    "Phase 8 authenticated hybrid route smoke: PASS (provider-success unavailable in local tenant)",
  );
} finally {
  if (authorization && sessionId && chatId) {
    await requiredSuccess("DELETE", `/api/v1/chats/${chatId}/sessions`, {
      ids: [sessionId],
    }).catch(() => undefined);
  }
  if (authorization && chatId) {
    await requiredSuccess("DELETE", `/api/v1/chats/${chatId}`).catch(
      () => undefined,
    );
  }
  if (authorization) {
    await requiredSuccess("POST", "/api/v1/auth/logout").catch(() => undefined);
  }
}
