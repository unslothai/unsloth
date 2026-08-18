#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const INVENTORY = resolve(ROOT, "docs/rag-platform/route-inventory.json");
const BASELINE = resolve(ROOT, "docs/rag-platform/route-inventory-phase14-baseline.json");
const REPORT = resolve(ROOT, "docs/rag-platform/route-inventory-phase15-diff.json");
const accept = process.argv.includes("--accept");
const check = process.argv.includes("--check");
const reviewed = process.argv.includes("--reviewed");

function normalizedRoutes(inventory) {
  return inventory.routes
    .map((route) => ({
      method: route.method,
      path: route.path,
      service: route.service,
      auth: route.auth,
      source: route.source,
      runtime_enabled: route.runtime_enabled,
      proxy_destination: route.proxy_destination,
    }))
    .sort((a, b) =>
      `${a.method}\0${a.path}\0${a.service}`.localeCompare(
        `${b.method}\0${b.path}\0${b.service}`,
      ),
    );
}

function digest(routes) {
  return createHash("sha256").update(JSON.stringify(routes)).digest("hex");
}

function key(route) {
  return `${route.method} ${route.path} [${route.service}]`;
}

const inventory = JSON.parse(readFileSync(INVENTORY, "utf8"));
const currentRoutes = normalizedRoutes(inventory);
const current = {
  schema: 1,
  captured_from_phase: 14,
  backend: inventory.backend,
  route_count: currentRoutes.length,
  sha256: digest(currentRoutes),
  routes: currentRoutes,
};

if (accept) {
  writeFileSync(BASELINE, `${JSON.stringify(current, null, 2)}\n`);
}
if (!existsSync(BASELINE)) {
  throw new Error("Phase 14 route baseline is missing; run with --accept after review.");
}

const baseline = JSON.parse(readFileSync(BASELINE, "utf8"));
const baselineByKey = new Map(baseline.routes.map((route) => [key(route), route]));
const currentByKey = new Map(currentRoutes.map((route) => [key(route), route]));
const added = currentRoutes.filter((route) => !baselineByKey.has(key(route)));
const removed = baseline.routes.filter((route) => !currentByKey.has(key(route)));
const changed = currentRoutes.flatMap((route) => {
  const before = baselineByKey.get(key(route));
  return before && JSON.stringify(before) !== JSON.stringify(route)
    ? [{ key: key(route), before, after: route }]
    : [];
});
const report = {
  schema: 1,
  baseline_phase: 14,
  baseline_backend: baseline.backend,
  current_backend: inventory.backend,
  totals: { added: added.length, removed: removed.length, changed: changed.length },
  added,
  removed,
  changed,
};
const expected = `${JSON.stringify(report, null, 2)}\n`;

if (check) {
  if (!existsSync(REPORT) || readFileSync(REPORT, "utf8") !== expected) {
    console.error("route inventory diff report is stale");
    process.exit(1);
  }
} else {
  writeFileSync(REPORT, expected);
}

if ((added.length || removed.length || changed.length) && !reviewed) {
  console.error(
    `route inventory drift: added=${added.length}, removed=${removed.length}, changed=${changed.length}`,
  );
  process.exit(1);
}
if (added.length || removed.length || changed.length) {
  console.log(
    `reviewed route inventory drift PASS (added=${added.length}, removed=${removed.length}, changed=${changed.length})`,
  );
} else {
  console.log(`route inventory matches Phase 14 baseline (${currentRoutes.length} routes)`);
}
