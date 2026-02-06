// postinstall.mjs
// Runs after npm install to verify platform binary is available

import { createRequire } from "node:module";
import { getPlatformPackage, getBinaryPath } from "./bin/platform.js";

const require = createRequire(import.meta.url);

/**
 * Detect libc family on Linux
 */
function getLibcFamily() {
  if (process.platform !== "linux") {
    return undefined;
  }
  
  try {
    const detectLibc = require("detect-libc");
    return detectLibc.familySync();
  } catch {
    return null;
  }
}

function main() {
  const { platform, arch } = process;
  const libcFamily = getLibcFamily();
  
  try {
    const pkg = getPlatformPackage({ platform, arch, libcFamily });
    const binPath = getBinaryPath(pkg, platform);
    
    // Try to resolve the binary
    require.resolve(binPath);
    console.log(`✓ oh-my-opencode binary installed for ${platform}-${arch}`);
  } catch (error) {
    console.warn(`⚠ oh-my-opencode: ${error.message}`);
    console.warn(`  The CLI may not work on this platform.`);
    // Don't fail installation - let user try anyway
  }
}

main();
