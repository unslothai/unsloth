#!/usr/bin/env node

import { gzipSync } from "node:zlib";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const DIST = resolve(ROOT, "studio/frontend/dist");
const manifestPath = resolve(DIST, ".vite/manifest.json");
if (!existsSync(manifestPath)) {
  throw new Error("production manifest missing; run npm run build first");
}
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const jsFiles = [...new Set(Object.values(manifest).flatMap((entry) => [entry.file, ...(entry.dynamicImports ?? [])]).filter((file) => file?.endsWith(".js")))];
let totalGzip = 0;
const failures = [];
for (const file of jsFiles) {
  const absolute = resolve(DIST, file);
  if (!existsSync(absolute)) continue;
  const raw = statSync(absolute).size;
  const gzip = gzipSync(readFileSync(absolute)).byteLength;
  totalGzip += gzip;
  if (raw > 6_000_000) failures.push(`${file} raw size ${raw} exceeds 6 MB`);
  if (gzip > 1_500_000) failures.push(`${file} gzip size ${gzip} exceeds 1.5 MB`);
}
if (totalGzip > 8_000_000) failures.push(`total JS gzip size ${totalGzip} exceeds 8 MB`);
if (!Object.values(manifest).some((entry) => entry.isDynamicEntry)) {
  failures.push("no lazy/dynamic production chunk is present");
}
if (failures.length) {
  for (const failure of failures) console.error(`performance gate: ${failure}`);
  process.exit(1);
}
console.log(`performance gate PASS (${jsFiles.length} JS chunks, ${totalGzip} gzip bytes)`);
