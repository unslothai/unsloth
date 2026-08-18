#!/usr/bin/env node

/**
 * Authenticated, secret-safe Phase 11 hybrid runtime smoke.
 *
 * Creates an ephemeral user, Agent and MCP server, exercises the active
 * Python/Go route split, then removes created records. No password, token,
 * credential-bearing body or response payload is logged or persisted.
 */
import { execFileSync } from "node:child_process";
import { constants, publicEncrypt } from "node:crypto";

const base = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");
const runId = Math.random().toString(36).slice(2, 10);
const account = {
  nickname: "Phase 11 Smoke",
  email: `phase11-${runId}@rag-platform.invalid`,
  password: `P11-${runId}-${Math.random().toString(36).slice(2, 10)}`,
};
let authorization = "";
let agentId = "";
let mcpId = "";
let sessionId = "";
let versionId = "";

const dsl = {
  components: {
    begin: {
      obj: { component_name: "Begin", params: {} },
      downstream: ["message"],
      upstream: [],
    },
    message: {
      obj: {
        component_name: "Message",
        params: { content: ["{sys.query}"] },
      },
      downstream: [],
      upstream: ["begin"],
    },
  },
  history: [],
  retrieval: [],
  path: [],
  globals: {
    "sys.query": "",
    "sys.user_id": "",
    "sys.conversation_turns": 0,
    "sys.files": [],
  },
  variables: {},
};

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
  { authenticated = true, displayPath = path, timeoutMs = 30_000 } = {},
) {
  const headers = new Headers();
  if (authenticated && authorization)
    headers.set("Authorization", authorization);
  if (body !== undefined) headers.set("Content-Type", "application/json");
  const response = await fetch(`${base}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  authorization = response.headers.get("authorization") || authorization;
  const contentType = response.headers.get("content-type")?.toLowerCase() || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : null;
  if (contentType.includes("text/event-stream")) {
    await response.body?.cancel().catch(() => undefined);
  }
  const code = envelopeCode(payload);
  console.log(
    `${method.padEnd(6)} ${displayPath.padEnd(68)} ${response.status} code=${code ?? "—"}`,
  );
  if ([404, 405, 502, 503, 504].includes(response.status)) {
    throw new Error(`${method} ${displayPath} did not reach an active handler`);
  }
  return { response, payload, code, contentType };
}

async function success(method, path, body, options) {
  const result = await request(method, path, body, options);
  if (!result.response.ok || result.code !== 0) {
    throw new Error(
      `${method} ${options?.displayPath ?? path} failed: HTTP ${result.response.status}, code ${result.code ?? "—"}`,
    );
  }
  return result.payload?.data;
}

async function reachesHandler(method, path, body, options) {
  return request(method, path, body, options);
}

async function requireAuthBoundary(method, path) {
  const result = await request(method, path, undefined, {
    authenticated: false,
  });
  if (result.response.status !== 401 && result.code !== 401) {
    throw new Error(`${method} ${path} did not enforce authentication`);
  }
}

const encrypted = encryptedPassword();

try {
  await success(
    "POST",
    "/api/v1/users",
    { nickname: account.nickname, email: account.email, password: encrypted },
    { authenticated: false },
  );
  await success(
    "POST",
    "/api/v1/auth/login",
    { email: account.email, password: encrypted },
    { authenticated: false },
  );
  if (!authorization) throw new Error("Login returned no Authorization header");

  const created = await success("POST", "/api/v1/agents", {
    title: `Phase 11 ${runId}`,
    description: "ephemeral runtime smoke",
    dsl,
  });
  agentId = created?.id || "";
  if (!agentId) throw new Error("Agent create returned no id");
  const agentPath = `/api/v1/agents/${encodeURIComponent(agentId)}`;

  await requireAuthBoundary(
    "GET",
    "/api/v1/agents/attachments/missing-attachment/preview",
  );

  await success("GET", "/api/v1/agents");
  await success("GET", agentPath);
  await success("PUT", agentPath, { description: "updated", dsl });
  await success("PUT", `${agentPath}/tags`, { tags: ["phase-11"] });
  await success("GET", "/api/v1/agents/templates");
  await success("GET", "/api/v1/agents/prompts");
  await success("GET", "/api/v1/agents/tags");
  await success("GET", "/api/v1/components");
  await reachesHandler("GET", `${agentPath}/components/begin/input-form`);
  await reachesHandler("POST", `${agentPath}/components/begin/debug`, {
    params: { "sys.query": { value: "smoke" } },
  });

  const session = await success("POST", `${agentPath}/sessions`, {
    name: `Session ${runId}`,
  });
  sessionId = session?.id || "";
  if (!sessionId) throw new Error("Session create returned no id");
  await success("GET", `${agentPath}/sessions`);
  await success(
    "GET",
    `${agentPath}/sessions/${encodeURIComponent(sessionId)}`,
  );

  const version = await success("POST", `${agentPath}/publish`, {
    title: `Phase 11 ${runId}`,
    dsl,
  });
  versionId = version?.id || "";
  await success("GET", `${agentPath}/versions`);
  if (versionId) {
    await success(
      "GET",
      `${agentPath}/versions/${encodeURIComponent(versionId)}`,
    );
  }

  await reachesHandler("POST", `${agentPath}/run`, { user_input: "smoke" });
  await reachesHandler("POST", "/api/v1/agents/chat/completions", {
    agent_id: agentId,
    query: "smoke",
    session_id: sessionId,
    stream: false,
  });
  await reachesHandler(
    "POST",
    `/api/v1/tasks/${encodeURIComponent(sessionId)}/cancel`,
  );
  await reachesHandler("POST", "/api/v1/agents/rerun", {
    id: "missing-document",
    component_id: "begin",
    dsl,
  });
  await reachesHandler("POST", "/api/v1/agents/test_db_connection", {
    db_type: "mysql",
    database: "missing",
    username: "missing",
    host: "127.0.0.1",
    port: 1,
    password: "ephemeral-not-logged",
  });
  await reachesHandler("POST", `${agentPath}/upload`);
  await reachesHandler("GET", "/api/v1/agents/download?id=missing-file");
  await reachesHandler(
    "GET",
    "/api/v1/agents/attachments/missing-attachment/preview?ext=pdf&mime_type=application%2Fpdf&filename=smoke.pdf",
    undefined,
    { displayPath: "/api/v1/agents/attachments/:id/preview" },
  );
  await reachesHandler(
    "GET",
    "/api/v1/agents/attachments/missing-attachment/download?ext=pdf&mime_type=application%2Fpdf&filename=smoke.pdf",
    undefined,
    { displayPath: "/api/v1/agents/attachments/:id/download" },
  );
  await reachesHandler("GET", `${agentPath}/webhook/logs`);
  for (const method of ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]) {
    await reachesHandler(
      method,
      `${agentPath}/webhook/test`,
      ["GET", "HEAD"].includes(method) ? undefined : { input: "smoke" },
      { displayPath: `/api/v1/agents/:id/webhook/test [${method}]` },
    );
    await reachesHandler(
      method,
      `${agentPath}/webhook`,
      ["GET", "HEAD"].includes(method) ? undefined : { input: "smoke" },
      {
        authenticated: false,
        displayPath: `/api/v1/agents/:id/webhook [${method}]`,
      },
    );
  }

  const mcp = await success("POST", "/api/v1/mcp/servers", {
    name: `Phase 11 ${runId}`,
    url: "https://example.com/mcp",
    server_type: "sse",
    variables: {},
    headers: {},
  });
  mcpId = mcp?.id || "";
  if (!mcpId) throw new Error("MCP create returned no id");
  const mcpPath = `/api/v1/mcp/servers/${encodeURIComponent(mcpId)}`;
  await success("GET", "/api/v1/mcp/servers");
  await success("GET", mcpPath);
  await success("PUT", mcpPath, { description: "updated" });
  await reachesHandler("POST", `${mcpPath}/test`, {
    name: `Phase 11 ${runId}`,
    url: "https://example.com/mcp",
    server_type: "sse",
    variables: {},
    headers: {},
    timeout: 2,
  });
  await reachesHandler("POST", "/api/v1/mcp/servers/import", {
    mcpServers: {},
    timeout: 2,
  });
  await success("GET", "/api/v1/plugin/tools");

  await success("POST", `${agentPath}/reset`);
  console.log("Phase 11 authenticated hybrid route smoke: PASS");
} finally {
  if (authorization && agentId && versionId) {
    await success(
      "DELETE",
      `/api/v1/agents/${encodeURIComponent(agentId)}/versions/${encodeURIComponent(versionId)}`,
    ).catch(() => undefined);
  }
  if (authorization && agentId && sessionId) {
    await success(
      "DELETE",
      `/api/v1/agents/${encodeURIComponent(agentId)}/sessions/${encodeURIComponent(sessionId)}`,
    ).catch(() => undefined);
  }
  if (authorization && mcpId) {
    await success(
      "DELETE",
      `/api/v1/mcp/servers/${encodeURIComponent(mcpId)}`,
    ).catch(() => undefined);
  }
  if (authorization && agentId) {
    await success(
      "DELETE",
      `/api/v1/agents/${encodeURIComponent(agentId)}`,
    ).catch(() => undefined);
  }
  if (authorization) {
    await success("POST", "/api/v1/auth/logout").catch(() => undefined);
  }
}
