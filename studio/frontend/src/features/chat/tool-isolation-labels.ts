// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure presentation helpers for OS isolation, importable without React or path aliases so
 *  the node tests can check them against the backend sources. */

/** One plain sentence per limitation identifier a backend can emit. A code without text would
 *  render as its raw identifier (see permission-mode-select.tsx), which tests forbid. */
export const TOOL_ISOLATION_LIMITATION_TEXT: Readonly<Record<string, string>> = {
  unrestricted_network:
    "Tools can reach any network host that Studio can access.",
  host_files_readable:
    "Tools can read files Studio can access, including private documents and credentials.",
  srt_windows_system_dns_unfenced:
    "Windows system DNS requests are not restricted.",
  srt_windows_shared_account_grants:
    "Windows runs share an account. Isolation between simultaneous runs is unverified.",
  srt_macos_system_dns_unfenced:
    "macOS system DNS requests are not restricted.",
  srt_private_unix_ipc_unqualified:
    "Private communication between Python workers and tensor sharing is unverified.",
  srt_runtime_unavailable:
    "Sandbox setup is incomplete. Required mode blocks tools until it is fixed.",
  srt_platform_unqualified:
    "This platform has not passed isolation checks. Required mode blocks tools.",
  srt_platform_qualification_incomplete:
    "SRT is in preview. Security and compatibility checks are incomplete, including CUDA.",
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
  concurrent_launches_share_the_container:
    "Tool calls running at the same time on Windows share one sandbox container, so while both run they can see each other's temporary files and named objects, and each other's chat working folders. Everything outside those stays out of reach.",
  null_device_and_named_pipes_denied:
    "Inside the Windows sandbox, Python cannot open NUL or create named pipes, so multiprocessing and imports that need them (such as torch) fail; use Limited or Full access for that work.",
  user_profile_readable:
    "Limited can read your documents, credentials, other runs’ temporary files, and other processes’ memory. Only writes are restricted.",
  user_profile_unreadable:
    "Limited mode on Windows could not read your user profile on this machine, so tools cannot see documents or credentials stored there. Writes stay confined to the chat working folder and a private temp folder, as they are in either case.",
  named_pipes_denied:
    "Limited mode on Windows cannot create named pipes on this machine, so multiprocessing queues and pools, and a DataLoader with num_workers above zero, will fail. Importing torch, single-process training and ordinary subprocesses are unaffected; use Full access for the rest.",
  network_unrestricted:
    "Limited can reach any network host.",
  everyone_writable_objects_writable:
    "Limited can still write to folders with Everyone write access, including some public and temporary folders.",
  network_allowlist_invalid:
    "UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST could not be parsed or names no host, so network access for sandboxed tools stays off until it is fixed.",
  restricted_token_unavailable:
    "Windows write restrictions are unavailable. Limited uses software checks only and can write anywhere Studio can.",
  proxy_allowlist_only_https_connect:
    "Only HTTPS to listed hosts is allowed through the proxy. HTTP, other ports, and other hosts are blocked.",
  network_allowlist_unsupported_on_windows:
    "The network allowlist is not offered on Windows; OS-isolated launches there have no network.",
  terminal_runtime_unselectable:
    "Terminal sandbox setup failed. Required mode blocks Terminal commands.",
};

/** Human label for the way Limited mode is implemented when it is more than the software
 *  safeguards. Null means plain Limited (no OS-level confinement at all). */
export function limitedBackendLabel(limitedBackend: string | null): string | null {
  if (!limitedBackend || limitedBackend === "process-guard") {
    // The process guard is the software-safeguards Limited every platform has; it adds
    // no OS-level confinement, so it gets no qualifier.
    return null;
  }
  if (limitedBackend === "windows-restricted-token") {
    return "restricted token (Windows)";
  }
  return limitedBackend;
}

/** One line naming what an allowlist launch may reach, from the backend's host list. */
export function networkAllowlistSummary(hosts: readonly string[]): string {
  if (hosts.length === 0) {
    return "No hosts are allowlisted on this backend.";
  }
  const families: string[] = [];
  if (hosts.some((host) => host.endsWith("pypi.org") || host.endsWith("pythonhosted.org"))) {
    families.push("PyPI");
  }
  if (hosts.some((host) => host.endsWith("huggingface.co") || host.endsWith("hf.co"))) {
    families.push("Hugging Face");
  }
  if (hosts.some((host) => host.endsWith("github.com") || host.endsWith("githubusercontent.com"))) {
    families.push("GitHub");
  }
  const count = `${hosts.length} ${hosts.length === 1 ? "host" : "hosts"}`;
  const named = families.length > 0 ? `${families.join(", ")} (${count})` : count;
  return `HTTPS only, to ${named}. Tools can send data to these hosts; all others are blocked.`;
}

/** Human label for the sandbox backend. The Windows backend has two profiles: the
 *  less-privileged AppContainer (LPAC) and the plain zero-capability AppContainer it falls back
 *  to when LPAC cannot start the interpreter, and the two must not share a name. */
export function backendLabel(
  backend: string | null,
  environment: string,
  profileId: string | null = null,
): string {
  if (backend === "srt") {
    return "Sandbox Runtime (SRT)";
  }
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
