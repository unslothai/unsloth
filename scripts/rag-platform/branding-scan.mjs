#!/usr/bin/env node

/**
 * Faz 0 white-label release gate.
 *
 * This deliberately parses TypeScript instead of grepping the tree. The legacy
 * name is valid in protocol identifiers, persistence keys, upstream URLs and
 * licence attribution, while it is forbidden in text a user can read. String
 * literals and JSX text are therefore inspected and every exception is an
 * explicit, reasoned rule below.
 *
 * Usage:
 *   node scripts/rag-platform/branding-scan.mjs
 *   node scripts/rag-platform/branding-scan.mjs --build
 */

import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "../../studio/frontend/node_modules/typescript/lib/typescript.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const SOURCE_ROOT = join(ROOT, "studio/frontend/src");
const INDEX_HTML = join(ROOT, "studio/frontend/index.html");
const DIST_ROOT = join(ROOT, "studio/frontend/dist");
const checkBuild = process.argv.includes("--build");
const vendorPattern = /\b(?:unsloth|ragflow|infiniflow)\b/i;

const ALLOWLIST = [
  {
    reason: "upstream URL or e-mail address",
    test: (value) => /(?:https?:\/\/|mailto:)[^\s]*?(?:unsloth|ragflow|infiniflow)/i.test(value),
  },
  {
    reason: "literal upstream CLI/package command",
    test: (value) =>
      /(?:^|[\s`'"/])(?:unsloth|unsloth-cli|unsloth-zoo)(?:\s|$|[=:./-])/i.test(value),
  },
  {
    reason: "protocol, environment, persistence, CSS or telemetry identifier",
    test: (value) =>
      /(?:UNSLOTH_[A-Z0-9_]+|unsloth[_:-][a-z0-9_-]+|[a-z0-9_-]+[_:-]unsloth|\.unsloth-|--unsloth-|@unsloth\b)/i.test(
        value,
      ),
  },
  {
    reason: "third-party model or repository identifier",
    test: (value) => /(?:^|[\s'"`([])unsloth(?:ai)?\/[A-Za-z0-9_.-]+/i.test(value),
  },
  {
    reason: "licence/attribution label in the About screen",
    test: (value, file) =>
      (file.endsWith("/features/settings/tabs/about-tab.tsx") ||
        file.endsWith("/i18n/locales/en.ts")) &&
      /^(?:Unsloth|Unsloth Studio|Unsloth AI(?: Inc\.)?|.*Unsloth.*(?:AGPL|Apache|licen[cs]e|copyright|upstream).*)$/i.test(
        value.trim(),
      ),
  },
  {
    reason: "fixed backend/login protocol value documented by ADR 0000",
    test: (value, file) =>
      (file.endsWith("/features/auth/components/auth-form.tsx") ||
        file.endsWith("/i18n/locales/en.ts")) &&
      /^unsloth$/i.test(value.trim()),
  },
  {
    reason: "fixed OAuth handoff cookie identifier documented by ADR 0002",
    test: (value, file) =>
      (file.endsWith("/integrations/platform-backend/auth-api.ts") ||
        file.endsWith("/integrations/platform-backend/__tests__/oauth.test.ts")) &&
      /^ragflow_auth(?:=; Path=\/; Max-Age=0|=opaque-cookie-token; Path=\/; SameSite=Lax)?$/i.test(
        value.trim(),
      ),
  },
];

function walk(dir) {
  const files = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    const stat = statSync(path);
    if (stat.isDirectory()) files.push(...walk(path));
    else if ([".ts", ".tsx"].includes(extname(path))) files.push(path);
  }
  return files;
}

function allowed(value, file) {
  return ALLOWLIST.some((rule) => rule.test(value, file));
}

function inspectSourceFile(path, findings) {
  const sourceText = readFileSync(path, "utf8");
  const source = ts.createSourceFile(
    path,
    sourceText,
    ts.ScriptTarget.Latest,
    true,
    path.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const file = relative(ROOT, path).replaceAll("\\", "/");

  function report(node, value) {
    if (!vendorPattern.test(value) || allowed(value, `/${file}`)) return;
    const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
    findings.push({ file, line: line + 1, value: value.replace(/\s+/g, " ").trim() });
  }

  function visit(node) {
    if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isTemplateHead(node) ||
      ts.isTemplateMiddle(node) ||
      ts.isTemplateTail(node) ||
      ts.isJsxText(node)
    ) {
      report(node, node.text);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

function inspectHtml(path, label, findings) {
  const html = readFileSync(path, "utf8");
  for (const match of html.matchAll(/<(title|meta\s+name=["'](?:description|application-name)["'][^>]*)>([^<]*)<\/title>|<meta\s+name=["'](?:description|application-name)["'][^>]*content=["']([^"']*)/gi)) {
    const value = match[2] ?? match[3] ?? "";
    if (vendorPattern.test(value)) findings.push({ file: label, line: 1, value });
  }
  if (!/<title>Rag Platform<\/title>/i.test(html)) {
    findings.push({ file: label, line: 1, value: "document title is not Rag Platform" });
  }
}

const findings = [];
for (const file of walk(SOURCE_ROOT)) inspectSourceFile(file, findings);
inspectHtml(INDEX_HTML, "studio/frontend/index.html", findings);

if (checkBuild) {
  const builtIndex = join(DIST_ROOT, "index.html");
  if (!existsSync(builtIndex)) {
    console.error("branding build audit failed: studio/frontend/dist/index.html is missing");
    process.exit(1);
  }
  inspectHtml(builtIndex, "studio/frontend/dist/index.html", findings);
}

if (findings.length) {
  console.error(`branding audit failed: ${findings.length} user-visible vendor occurrence(s)`);
  for (const finding of findings) {
    console.error(`  ${finding.file}:${finding.line}: ${JSON.stringify(finding.value)}`);
  }
  process.exit(1);
}

console.log(
  `branding ${checkBuild ? "source/build" : "source"} audit passed (` +
    `${walk(SOURCE_ROOT).length} TypeScript files; ${ALLOWLIST.length} reasoned allowlist rules)`,
);
