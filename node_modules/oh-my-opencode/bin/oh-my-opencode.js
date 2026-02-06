#!/usr/bin/env node
// bin/oh-my-opencode.js
// Wrapper script that detects platform and spawns the correct binary

import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { getPlatformPackage, getBinaryPath } from "./platform.js";

const require = createRequire(import.meta.url);

/**
 * Detect libc family on Linux
 * @returns {string | null} 'glibc', 'musl', or null if detection fails
 */
function getLibcFamily() {
  if (process.platform !== "linux") {
    return undefined; // Not needed on non-Linux
  }
  
  try {
    const detectLibc = require("detect-libc");
    return detectLibc.familySync();
  } catch {
    // detect-libc not available
    return null;
  }
}

function main() {
  const { platform, arch } = process;
  const libcFamily = getLibcFamily();
  
  // Get platform package name
  let pkg;
  try {
    pkg = getPlatformPackage({ platform, arch, libcFamily });
  } catch (error) {
    console.error(`\noh-my-opencode: ${error.message}\n`);
    process.exit(1);
  }
  
  // Resolve binary path
  const binRelPath = getBinaryPath(pkg, platform);
  
  let binPath;
  try {
    binPath = require.resolve(binRelPath);
  } catch {
    console.error(`\noh-my-opencode: Platform binary not installed.`);
    console.error(`\nYour platform: ${platform}-${arch}${libcFamily === "musl" ? "-musl" : ""}`);
    console.error(`Expected package: ${pkg}`);
    console.error(`\nTo fix, run:`);
    console.error(`  npm install ${pkg}\n`);
    process.exit(1);
  }
  
  // Spawn the binary
  const result = spawnSync(binPath, process.argv.slice(2), {
    stdio: "inherit",
  });
  
  // Handle spawn errors
  if (result.error) {
    console.error(`\noh-my-opencode: Failed to execute binary.`);
    console.error(`Error: ${result.error.message}\n`);
    process.exit(2);
  }
  
  // Handle signals
  if (result.signal) {
    const signalNum = result.signal === "SIGTERM" ? 15 : 
                      result.signal === "SIGKILL" ? 9 :
                      result.signal === "SIGINT" ? 2 : 1;
    process.exit(128 + signalNum);
  }

  process.exit(result.status ?? 1);
}

main();
