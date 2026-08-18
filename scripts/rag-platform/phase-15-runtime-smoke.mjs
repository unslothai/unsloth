#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const base = (process.argv.find((arg) => /^https?:\/\//.test(arg)) || "http://127.0.0.1").replace(/\/+$/, "");
const full = process.argv.includes("--full");
const release = process.argv.includes("--release");

if (release && !base.startsWith("https://")) {
  throw new Error("release smoke requires an https:// same-origin URL");
}

async function expectStatus(url, statuses) {
  const response = await fetch(url, { signal: AbortSignal.timeout(15_000) });
  if (!statuses.includes(response.status)) {
    throw new Error(`${url} returned ${response.status}; expected ${statuses.join("/")}`);
  }
  return response;
}

await expectStatus("http://127.0.0.1:9380/api/v1/system/ping", [200]);
await expectStatus("http://127.0.0.1:9381/api/v1/admin/ping", [200]);
await expectStatus("http://127.0.0.1:9383/api/v1/admin/ping", [200]);
await expectStatus("http://127.0.0.1:9384/health", [200]);
await expectStatus(`${base}/api/v1/system/ping`, [200]);
await expectStatus(`${base}/healthz`, [200]);
await expectStatus(`${base}/live`, [200]);
await expectStatus(`${base}/api/v1/language`, [200]);
const document = await expectStatus(`${base}/`, [200]);
const csp = document.headers.get("content-security-policy") ?? "";
if (!csp.includes("default-src 'self'") || !csp.includes("object-src 'none'")) {
  throw new Error("live same-origin proxy is missing the Phase 15 CSP");
}
if (document.headers.get("x-content-type-options") !== "nosniff") {
  throw new Error("live same-origin proxy is missing nosniff");
}
const ping = await fetch(`${base}/api/v1/system/ping`, {
  signal: AbortSignal.timeout(15_000),
});
if (ping.headers.get("access-control-allow-origin") === "*") {
  throw new Error("live production proxy exposes wildcard CORS");
}

if (full) {
  const suites = [7, 8, 9, 10, 11, 12, 13, 14].map(
    (phase) => `scripts/rag-platform/phase-${phase}-runtime-smoke.mjs`,
  );
  for (const suite of suites) {
    const result = spawnSync(process.execPath, [resolve(ROOT, suite), base], {
      cwd: ROOT,
      env: process.env,
      stdio: "inherit",
    });
    if (result.status !== 0) throw new Error(`${suite} failed`);
  }
}

console.log(`Phase 15 runtime smoke PASS (four services, same-origin hardening${full ? ", Phase 7-14 suites" : ""})`);
