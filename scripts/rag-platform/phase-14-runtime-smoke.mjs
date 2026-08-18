#!/usr/bin/env node

/** Secret-safe Phase 14 hybrid/direct route and authorization boundary smoke. */
const publicBase = (process.argv[2] || "http://127.0.0.1").replace(/\/+$/, "");

function envelopeCode(payload) {
  if (!payload || typeof payload !== "object") return null;
  return typeof payload.code === "number" || typeof payload.code === "string"
    ? Number(payload.code)
    : null;
}

async function probe(base, method, path, expectedStatuses, label, expectedCodes = []) {
  const response = await fetch(`${base}${path}`, {
    method,
    signal: AbortSignal.timeout(15_000),
    redirect: "manual",
  });
  const contentType = response.headers.get("content-type")?.toLowerCase() || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : null;
  if (!contentType.includes("application/json")) {
    await response.body?.cancel().catch(() => undefined);
  }
  const code = envelopeCode(payload);
  console.log(`${label.padEnd(29)} ${method.padEnd(6)} ${path.padEnd(54)} ${response.status} code=${code ?? "—"}`);
  if (!expectedStatuses.includes(response.status) && !expectedCodes.includes(code)) {
    throw new Error(`${label} ${method} ${path}: expected ${expectedStatuses.join("/")}, received ${response.status}`);
  }
  return { code, response };
}

await probe(publicBase, "GET", "/api/v1/admin/ping", [200], "hybrid go-admin");
await probe("http://127.0.0.1:9383", "GET", "/api/v1/admin/ping", [200], "direct go-admin");
await probe("http://127.0.0.1:9381", "GET", "/api/v1/admin/ping", [200, 404], "direct python-admin");

for (const path of [
  "/api/v1/admin/auth",
  "/api/v1/admin/users",
  "/api/v1/admin/sandbox/providers",
  "/api/v1/tenants",
  "/api/v1/chat-channels",
  "/api/v1/compilation_template_groups",
  "/api/v1/thumbnails",
  "/api/v1/documents/phase14-probe/preview",
]) {
  await probe(publicBase, "GET", path, [401, 403], "anonymous denied", [401, 403]);
}

await probe(publicBase, "GET", "/api/v1/dify/retrieval/health", [200], "compatibility health");
await probe(publicBase, "POST", "/api/v1/mcp", [200, 401, 403], "beta MCP auth", [102, 401, 403]);

for (const path of [
  "/api/v1/llm/aimlapi/authorize/start",
  "/api/v1/llm/aimlapi/authorize/poll",
]) {
  await probe(publicBase, "POST", path, [401, 403], "AIMLAPI auth boundary", [401, 403]);
}

// The owned Phase 14 image deploys the normative Go route spelling.
await probe(publicBase, "GET", "/api/v1/compilation-template-groups", [401, 403], "runtime overlay route", [401, 403]);

for (const [method, path] of [
  ["GET", "/api/v1/tenants/phase14-probe"],
  ["PUT", "/api/v1/tenants/phase14-probe"],
  ["PUT", "/api/v1/tenants/phase14-probe/users/phase14-probe/role"],
]) {
  await probe(publicBase, method, path, [401, 403], "tenant auth boundary", [401, 403]);
}

for (const path of [
  "/api/v1/tenant/insert_chunks_from_file",
  "/api/v1/tenant/insert_metadata_from_file",
]) {
  await probe(publicBase, "POST", path, [404], "removed legacy internal route", [404]);
  await probe("http://127.0.0.1:9384", "POST", path, [404], "direct removed internal route", [404]);
}

for (const path of [
  "/api/v1/tenant/dev_insert_chunks_from_file",
  "/api/v1/tenant/dev_insert_metadata_from_file",
]) {
  await probe(publicBase, "POST", path, [401, 403], "tenant internal auth", [401, 403]);
  await probe("http://127.0.0.1:9384", "POST", path, [401, 403], "direct internal auth", [401, 403]);
}

console.log("Phase 14 hybrid/direct authorization and runtime-overlay smoke: PASS");
