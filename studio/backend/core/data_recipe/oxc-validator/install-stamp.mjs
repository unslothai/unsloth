// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Lets setup skip `npm ci` here when node_modules is still exactly what the last successful one left.
//   node install-stamp.mjs write   after a successful `npm ci`: record the inputs and the tree.
//   node install-stamp.mjs check   exit 0 only when the inputs and every file match that record.
// Any doubt (no stamp, unreadable file, changed byte) exits 1, and setup runs `npm ci` as before.
import { createHash } from "node:crypto";
import {
  lstatSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const MODULES = join(ROOT, "node_modules");
const STAMP_NAME = ".unsloth-install-stamp.json";
const STAMP = join(MODULES, STAMP_NAME);
// Setup deletes these from node_modules after every install.
const PRUNED = new Set(["AGENTS.md", "CLAUDE.md"]);
// npm settings that only pick where packages come from or how npm talks; they never change the tree.
const NEUTRAL_CONFIG = /(registry|^prefix|^cache|^loglevel|^fund|^audit|^progress|^update_notifier|^userconfig|^globalconfig)$/;

const sha256 = (data) => createHash("sha256").update(data).digest("hex");
const fileHash = (path) => sha256(readFileSync(path));
const optionalHash = (path) => {
  try {
    return fileHash(path);
  } catch {
    return null;
  }
};

function libc() {
  if (process.platform !== "linux") return "";
  try {
    process.report.excludeNetwork = true;
    return process.report.getReport().header.glibcVersionRuntime ? "glibc" : "musl";
  } catch {
    return "unknown";
  }
}

// What `npm ci` reads besides the lockfile: Node, the platform its optional binaries are picked
// for, and any npm setting that could add or drop packages.
function inputs() {
  const config = Object.keys(process.env)
    .map((k) => [k.toLowerCase(), process.env[k]])
    .filter(([k]) => k.startsWith("npm_config_"))
    .map(([k, v]) => [k.slice("npm_config_".length).replace(/-/g, "_"), v])
    .filter(([k]) => !NEUTRAL_CONFIG.test(k))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  const env = (name) => process.env[`npm_config_${name}`] || process.env[`NPM_CONFIG_${name.toUpperCase()}`];
  const userrc = env("userconfig") || join(homedir(), ".npmrc");
  const nodeBin = dirname(process.execPath);
  const prefix = env("prefix") || (process.platform === "win32" ? nodeBin : dirname(nodeBin));
  const globalrc = env("globalconfig") || join(prefix, "etc", "npmrc");
  return {
    version: 1,
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    libc: libc(),
    package: fileHash(join(ROOT, "package.json")),
    lock: fileHash(join(ROOT, "package-lock.json")),
    config,
    npmrc: optionalHash(join(ROOT, ".npmrc")),
    userNpmrc: optionalHash(userrc),
    globalNpmrc: optionalHash(globalrc),
  };
}

// Every path under node_modules with its type and content, in a fixed order.
function treeHash() {
  const digest = createHash("sha256");
  const walk = (dir, rel) => {
    for (const name of readdirSync(dir).sort()) {
      const path = join(dir, name);
      const relPath = rel ? `${rel}/${name}` : name;
      if (!rel && name.startsWith(STAMP_NAME)) continue;
      const st = lstatSync(path);
      if (st.isSymbolicLink()) {
        digest.update(`L\0${relPath}\0${readlinkSync(path)}\n`);
      } else if (st.isDirectory()) {
        digest.update(`D\0${relPath}\n`);
        walk(path, relPath);
      } else if (!PRUNED.has(name)) {
        digest.update(`F\0${relPath}\0${st.size}\0${fileHash(path)}\n`);
      }
    }
  };
  walk(MODULES, "");
  return digest.digest("hex");
}

// The packages npm recorded as installed are on disk at the locked versions: oxlint, oxc-parser,
// oxc-transform and the platform binaries they load.
function packagesPresent() {
  const lock = JSON.parse(readFileSync(join(ROOT, "package-lock.json"), "utf8")).packages || {};
  const hidden = JSON.parse(readFileSync(join(MODULES, ".package-lock.json"), "utf8")).packages || {};
  const declared = Object.keys(JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")).dependencies || {});
  if (!declared.length || !declared.includes("oxlint") || !declared.includes("oxc-parser")) return false;
  for (const name of declared) {
    if (!hidden[`node_modules/${name}`]) return false;
  }
  for (const [key, entry] of Object.entries(hidden)) {
    if (!key.startsWith("node_modules/") || !lock[key] || lock[key].version !== entry.version) return false;
    const manifest = JSON.parse(readFileSync(join(ROOT, key, "package.json"), "utf8"));
    if (manifest.version !== entry.version) return false;
  }
  return true;
}

function current() {
  return {
    ...inputs(),
    hiddenLock: fileHash(join(MODULES, ".package-lock.json")),
    tree: treeHash(),
  };
}

function check() {
  const recorded = JSON.parse(readFileSync(STAMP, "utf8"));
  return JSON.stringify(recorded) === JSON.stringify(current()) && packagesPresent();
}

function write() {
  if (!packagesPresent()) return false;
  const tmp = `${STAMP}.${process.pid}.tmp`;
  try {
    writeFileSync(tmp, JSON.stringify(current()));
    renameSync(tmp, STAMP);
  } catch (error) {
    rmSync(tmp, { force: true });
    throw error;
  }
  return true;
}

const mode = process.argv[2];
let ok = false;
try {
  if (mode === "check") ok = check();
  else if (mode === "write") ok = write();
} catch {
  ok = false;
}
process.exitCode = ok ? 0 : 1;
