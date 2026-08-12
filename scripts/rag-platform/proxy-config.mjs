#!/usr/bin/env node

/** Generate the owned full-parity hybrid nginx map from the route inventory. */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const INVENTORY = join(ROOT, "docs/rag-platform/route-inventory.json");
const OUTPUT = join(ROOT, "infra/rag-platform/rag-platform.hybrid.conf");
const checkOnly = process.argv.includes("--check");

function routeRegex(path) {
  let out = "";
  for (let index = 0; index < path.length; ) {
    const rest = path.slice(index);
    const quart = rest.match(/^<[^>]+>/);
    const gin = rest.match(/^:[A-Za-z_][A-Za-z0-9_]*/);
    const wildcard = rest.match(/^\*[A-Za-z_][A-Za-z0-9_]*/);
    if (quart || gin) {
      out += "[^/]+";
      index += (quart ?? gin)[0].length;
    } else if (wildcard) {
      out += ".*";
      index += wildcard[0].length;
    } else {
      out += /[\\^$.*+?()[\]{}|]/.test(path[index]) ? `\\${path[index]}` : path[index];
      index += 1;
    }
  }
  return `^${out}/?$`;
}

function canonicalPath(path) {
  return path
    .replace(/<[^>]+>/g, "{p}")
    .replace(/:[A-Za-z_][A-Za-z0-9_]*/g, "{p}")
    .replace(/\*[A-Za-z_][A-Za-z0-9_]*/g, "{p}")
    .replace(/\/+$/, "");
}

function probePath(path) {
  return path
    .replace(/<[^>]+>/g, "x")
    .replace(/:[A-Za-z_][A-Za-z0-9_]*/g, "x")
    .replace(/\*[A-Za-z_][A-Za-z0-9_]*/g, "x");
}

const inventory = JSON.parse(readFileSync(INVENTORY, "utf8"));
const goRoutes = new Map();
for (const route of inventory.routes) {
  if (route.service !== "go-api" && route.service !== "go-admin") continue;
  const regex = routeRegex(route.path);
  goRoutes.set(`${route.method}:${regex}`, {
    method: route.method,
    regex,
    port: route.service_port,
    path: route.path,
  });
}

const routes = [...goRoutes.values()].sort(
  (a, b) =>
    b.path.replace(/<[^>]+>|:[^/]+|\*[^/]+/g, "").length -
      a.path.replace(/<[^>]+>|:[^/]+|\*[^/]+/g, "").length ||
    a.method.localeCompare(b.method) ||
    a.path.localeCompare(b.path),
);

// A parameterized Go route can otherwise steal a more specific Python route.
// Example: Go `DELETE /datasets/:id/:index_type` also regex-matches Python's
// `DELETE /datasets/<id>/artifacts`, although the Go handler only accepts the
// index types graph/raptor/mindmap. Emit stable Python specificity overrides
// whenever the concrete canonical contracts differ but the Go regex overlaps.
const pythonOverrides = new Map();
for (const pythonRoute of inventory.routes) {
  if (pythonRoute.service !== "python-api" && pythonRoute.service !== "python-admin") continue;
  const pythonCanonical = canonicalPath(pythonRoute.path);
  const pythonProbe = probePath(pythonRoute.path);
  for (const goRoute of routes) {
    if (goRoute.method !== pythonRoute.method) continue;
    if (canonicalPath(goRoute.path) === pythonCanonical) continue;
    if (!/[<:*]/.test(goRoute.path)) continue;
    if (!new RegExp(goRoute.regex).test(pythonProbe)) continue;
    const regex = routeRegex(pythonRoute.path);
    pythonOverrides.set(`${pythonRoute.method}:${regex}`, {
      method: pythonRoute.method,
      regex,
      port: pythonRoute.service_port,
      path: pythonRoute.path,
    });
  }
}

const specificityOverrides = [...pythonOverrides.values()].sort(
  (a, b) =>
    b.path.replace(/<[^>]+>|:[^/]+|\*[^/]+/g, "").length -
      a.path.replace(/<[^>]+>|:[^/]+|\*[^/]+/g, "").length ||
    a.method.localeCompare(b.method) ||
    a.path.localeCompare(b.path),
);

const lines = [
  "####################################################################",
  "# GENERATED FILE. Do not edit by hand.",
  "# Regenerate: node scripts/rag-platform/proxy-config.mjs",
  "#",
  "# Full-parity hybrid routing: Python is the fallback; every registered Go",
  "# method+path is routed to its Go service. Method-aware dispatch prevents a",
  "# Go-only mutation from stealing a Python-only read at the same path.",
  "####################################################################",
  "",
  "map $uri $rag_platform_python_port {",
  "    default 9380;",
  "    ~^/api/v1/admin(?:/|$) 9381;",
  "}",
  "",
  'map "$request_method$uri" $rag_platform_service_port {',
  "    default $rag_platform_python_port;",
];

for (const route of specificityOverrides) {
  lines.push(`    # rag-platform-route ${route.method} ${route.regex} ${route.port}`);
  lines.push(`    ~^${route.method}${route.regex.slice(1)} ${route.port};`);
}

for (const route of routes) {
  lines.push(`    # rag-platform-route ${route.method} ${route.regex} ${route.port}`);
  if (!route.regex.startsWith("^")) {
    throw new Error(`expected anchored route regex, got: ${route.regex}`);
  }
  lines.push(`    ~^${route.method}${route.regex.slice(1)} ${route.port};`);
}

lines.push(
  "}",
  "",
  "server {",
  "    listen 80;",
  "    server_name _;",
  "    root /ragflow/web/dist;",
  "",
  "    gzip on;",
  "    gzip_min_length 1k;",
  "    gzip_comp_level 9;",
  "    gzip_types text/plain application/javascript application/x-javascript text/css application/xml text/javascript application/x-httpd-php image/jpeg image/gif image/png;",
  "    gzip_vary on;",
  '    gzip_disable "MSIE [1-6]\\.";',
  "",
  "    # rag-platform-default ^/api/v1/admin(?:/|$) 9381",
  "    # rag-platform-default ^/(?:v1|api)(?:/|$) 9380",
  "    location ~ ^/(v1|api) {",
  "        proxy_pass http://127.0.0.1:$rag_platform_service_port;",
  "        include proxy.conf;",
  "    }",
  "",
  "    location / {",
  "        index index.html;",
  "        try_files $uri $uri/ /index.html;",
  "    }",
  "",
  "    location ~ ^/static/(css|js|media)/ {",
  "        expires 10y;",
  "        access_log off;",
  "    }",
  "}",
  "",
);

const expected = lines.join("\n");
if (checkOnly) {
  if (!existsSync(OUTPUT) || readFileSync(OUTPUT, "utf8") !== expected) {
    console.error(`${relative(ROOT, OUTPUT)} is stale — rerun proxy-config.mjs`);
    process.exit(1);
  }
  console.log(
    `proxy config up to date (${routes.length} Go routes, ` +
      `${specificityOverrides.length} Python specificity overrides)`,
  );
} else {
  writeFileSync(OUTPUT, expected);
  console.log(
    `wrote ${relative(ROOT, OUTPUT)} (${routes.length} Go routes, ` +
      `${specificityOverrides.length} Python specificity overrides)`,
  );
}
