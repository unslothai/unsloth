#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const BACKEND = process.env.RAG_PLATFORM_BACKEND_DIR || resolve(ROOT, "../rag-backend");
const failures = [];
const read = (path) => readFileSync(resolve(ROOT, path), "utf8");
const nginx = read("infra/rag-platform/rag-platform.hybrid.conf");
const nginxProxy = read("infra/rag-platform/proxy.conf");
const securityHeaders = read("infra/rag-platform/security-headers.conf");
const compose = read("infra/rag-platform/docker-compose.rag-platform.yml");
const ownedEnv = read("infra/rag-platform/.env.rag-platform");
const dockerfile = read("infra/rag-platform/Dockerfile.backend-with-go");
const baseline = JSON.parse(read("docs/rag-platform/license-baseline.json"));

function requireMatch(value, pattern, label) {
  if (!pattern.test(value)) failures.push(label);
}
requireMatch(ownedEnv, /^API_PROXY_SCHEME=hybrid$/m, "hybrid proxy is not pinned");
requireMatch(
  compose,
  /image:\s*\$\{RAG_PLATFORM_BACKEND_IMAGE:-rag-platform-backend:0\.26\.4\}/,
  "backend image default is not pinned",
);
if (/image:\s*[^\n]*:latest/.test(compose)) failures.push("latest image tag is forbidden");
requireMatch(compose, /rag-platform-readiness/, "four-service healthcheck is missing");
requireMatch(dockerfile, /rag-platform-frontend-dist/, "owned production frontend is not in the image");
requireMatch(securityHeaders, /Content-Security-Policy/, "CSP header is missing");
requireMatch(securityHeaders, /Strict-Transport-Security/, "HSTS header is missing");
requireMatch(nginxProxy, /proxy_buffering off;/, "SSE buffering is not disabled");
requireMatch(nginxProxy, /proxy_read_timeout 3600s;/, "stream timeout is not explicit");
requireMatch(nginx, /client_max_body_size 1g;/, "upload limit is not explicit");
if (/Access-Control-Allow-Origin\s+["']?\*/.test(nginx)) {
  failures.push("wildcard production CORS is forbidden");
}

for (const [relativePath, expected] of Object.entries(baseline.sha256)) {
  const absolute = relativePath.startsWith("backend/")
    ? resolve(BACKEND, relativePath.slice("backend/".length))
    : resolve(ROOT, relativePath);
  const actual = createHash("sha256").update(readFileSync(absolute)).digest("hex");
  if (actual !== expected) failures.push(`${relativePath} license checksum changed`);
}

const publicUrl = process.env.RAG_PLATFORM_PUBLIC_URL?.trim();
if (process.argv.includes("--runtime")) {
  if (!publicUrl || !publicUrl.startsWith("https://")) {
    failures.push("RAG_PLATFORM_PUBLIC_URL must be an https:// URL for release");
  }
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`security gate: ${failure}`);
  process.exit(1);
}
console.log("release security gate PASS");
