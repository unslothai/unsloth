// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure presentation helpers for OS isolation, importable without React or path aliases so
 *  the node tests can check them against the backend sources. */

/** One plain sentence per limitation identifier a backend can emit. A code without text would
 *  render as its raw identifier (see permission-mode-select.tsx), which tests forbid. */
export const TOOL_ISOLATION_LIMITATION_TEXT: Readonly<Record<string, string>> = {
  deprecated_undocumented_sbpl:
    "Apple deprecates sandbox-exec and does not document SBPL for third-party products.",
  detached_descendant_cleanup_unverified:
    "Cleanup of descendants that create a new session or double-fork is unverified.",
  pytorch_posix_shm_namespace_shared:
    "PyTorch tensor sharing uses macOS's host POSIX shared-memory namespace. Access is limited to PyTorch's randomized names, but the namespace is not private.",
  nested_userns_blocked_by_seccomp:
    "This Bubblewrap version has no --disable-userns option, so nested user namespaces are blocked with a seccomp filter instead.",
  ipv6_unavailable_on_host:
    "IPv6 loopback is unavailable on this host, so the IPv6 leg of the isolation probe was skipped.",
  all_application_packages_ambient_read:
    "This Windows AppContainer can read files shared with all application packages, such as Program Files and Windows. The user profile, the network and other processes stay out of reach.",
  null_device_and_named_pipes_denied:
    "Inside the Windows sandbox, Python cannot open NUL or create named pipes, so multiprocessing and imports that need them (such as torch) fail; use Limited or Full access for that work.",
};

/** Human label for the sandbox backend. The Windows backend has two profiles: the
 *  less-privileged AppContainer (LPAC) and the plain zero-capability AppContainer it falls back
 *  to when LPAC cannot start the interpreter, and the two must not share a name. */
export function backendLabel(
  backend: string | null,
  environment: string,
  profileId: string | null = null,
): string {
  if (backend === "windows-lpac") {
    return profileId?.startsWith("windows-appcontainer")
      ? "AppContainer (Windows)"
      : "LPAC (Windows)";
  }
  if (backend === "macos-seatbelt") {
    return "Seatbelt (lifecycle unverified)";
  }
  if (!backend?.toLowerCase().includes("bubblewrap")) {
    return backend || "OS sandbox";
  }
  const normalized = environment.toLowerCase();
  if (normalized.includes("wsl")) {
    return "Bubblewrap (WSL2)";
  }
  if (normalized.includes("colab")) {
    return "Bubblewrap (Colab)";
  }
  if (normalized.includes("container")) {
    return "Bubblewrap (Container)";
  }
  return "Bubblewrap";
}
