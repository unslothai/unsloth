#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const roots = [
  resolve(import.meta.dirname, "../.."),
  process.env.RAG_PLATFORM_BACKEND_DIR || resolve(import.meta.dirname, "../../../rag-backend"),
];
const patterns = [
  { name: "private-key", regex: /-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----/ },
  { name: "github-token", regex: /\bgh[opusr]_[A-Za-z0-9]{30,}\b/ },
  { name: "openai-key", regex: /\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b/ },
  { name: "aws-access-key", regex: /\bAKIA[0-9A-Z]{16}\b/ },
];
const allow = [
  /(?:^|\/)fixtures\//,
  /(?:^|\/)__tests__\//,
  /(?:^|\/)test(?:s)?\//,
  /\.example$/,
  /package-lock\.json$/,
  /secret-scan\.mjs$/,
  // Scripts must recognize key headers to generate/verify runtime-only keys;
  // neither file contains key material.
  /^infra\/rag-platform\/backend-entrypoint\.sh$/,
  /^scripts\/rag-platform\/auth-key-contract\.sh$/,
  // Localized security guidance contains a documented fake access-key shape.
  /^studio\/frontend\/src\/i18n\/locales\/en\.ts$/,
  // Backend unit fixtures intentionally embed synthetic key blocks.
  /_test\.go$/,
];
const findings = [];

for (const root of roots) {
  const files = execFileSync(
    "git",
    ["-C", root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    {
    encoding: "utf8",
    },
  ).split("\0").filter(Boolean);
  for (const relative of files) {
    if (allow.some((entry) => entry.test(relative))) continue;
    const absolute = resolve(root, relative);
    if (!existsSync(absolute)) continue;
    let content;
    try {
      content = readFileSync(absolute, "utf8");
    } catch {
      continue;
    }
    for (const pattern of patterns) {
      if (pattern.regex.test(content)) findings.push(`${root}:${relative}:${pattern.name}`);
    }
  }
}

if (findings.length > 0) {
  for (const finding of findings) console.error(`secret scan finding: ${finding}`);
  process.exit(1);
}
console.log(`secret scan PASS (${roots.length} repositories; contents redacted)`);
