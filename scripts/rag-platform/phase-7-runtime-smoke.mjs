#!/usr/bin/env node

/**
 * Authenticated, secret-safe Phase 7 Chat/Session runtime smoke.
 * Creates a throwaway user and artifacts, exercises the active hybrid paths,
 * then removes every Chat/Session it created. Credentials and bearer tokens
 * remain process-local and are never printed or written.
 */
import { execFileSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";

const base = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");
const runId = Math.random().toString(36).slice(2, 10);
const account = {
  nickname: "Phase 7 Smoke",
  email: `phase7-${runId}@rag-platform.invalid`,
  password: `P7-${runId}-${Math.random().toString(36).slice(2, 10)}`,
};
let authorization = "";

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

async function request(method, path, body, authenticated = true) {
  const headers = new Headers();
  if (authenticated && authorization) {
    headers.set("Authorization", authorization);
  }
  if (body !== undefined) headers.set("Content-Type", "application/json");
  const response = await fetch(`${base}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(30_000),
  });
  authorization = response.headers.get("authorization") || authorization;
  const payload = await response.json().catch(() => null);
  const code = payload && typeof payload.code === "number" ? payload.code : null;
  console.log(`${method.padEnd(6)} ${path.padEnd(72)} ${response.status} code=${code ?? "—"}`);
  if (!response.ok || (code !== null && code !== 0)) {
    throw new Error(
      `${method} ${path} failed: HTTP ${response.status}, code ${code ?? "—"}`,
    );
  }
  return payload?.data;
}

const encrypted = encryptedPassword();
let chatId = "";
let sessionId = "";

try {
  await request(
    "POST",
    "/api/v1/users",
    {
      nickname: account.nickname,
      email: account.email,
      password: encrypted,
    },
    false,
  );
  await request(
    "POST",
    "/api/v1/auth/login",
    { email: account.email, password: encrypted },
    false,
  );
  if (!authorization) throw new Error("Login returned no Authorization header");

  const createdChat = await request("POST", "/api/v1/chats", {
    name: `phase-7-chat-${runId}`,
    dataset_ids: [],
  });
  chatId = createdChat?.id || "";
  if (!chatId) throw new Error("Chat create returned no id");
  await request("GET", "/api/v1/chats?page=1&page_size=100");
  await request("GET", `/api/v1/chats/${chatId}`);
  await request("PATCH", `/api/v1/chats/${chatId}`, {
    name: `phase-7-chat-renamed-${runId}`,
    dataset_ids: [],
  });
  await request("PUT", `/api/v1/chats/${chatId}`, {
    name: `phase-7-chat-replaced-${runId}`,
    dataset_ids: [],
  });

  const createdSession = await request(
    "POST",
    `/api/v1/chats/${chatId}/sessions`,
    { name: "New Chat" },
  );
  sessionId = createdSession?.id || "";
  if (!sessionId) throw new Error("Session create returned no id");
  await request(
    "GET",
    `/api/v1/chats/${chatId}/sessions?page=1&page_size=100`,
  );
  await request("GET", `/api/v1/chats/${chatId}/sessions/${sessionId}`);
  await request("PATCH", `/api/v1/chats/${chatId}/sessions/${sessionId}`, {
    name: "Renamed Chat",
  });
  await request("PUT", `/api/v1/chats/${chatId}/sessions/${sessionId}`, {
    name: "Compatibility Rename",
  });
  await request("DELETE", `/api/v1/chats/${chatId}/sessions`, {
    ids: [sessionId],
  });
  sessionId = "";
  await request("DELETE", `/api/v1/chats/${chatId}`);
  chatId = "";

  console.log("Phase 7 authenticated runtime smoke: PASS");
} finally {
  if (authorization && sessionId && chatId) {
    await request("DELETE", `/api/v1/chats/${chatId}/sessions`, {
      ids: [sessionId],
    }).catch(() => undefined);
  }
  if (authorization && chatId) {
    await request("DELETE", `/api/v1/chats/${chatId}`).catch(() => undefined);
  }
  if (authorization) {
    await request("POST", "/api/v1/auth/logout").catch(() => undefined);
  }
}
