export function isolationLimitation(value: string): string {
  const labels: Record<string, string> = {
    srt_windows_system_dns_unfenced: "System DNS remains available.",
    srt_windows_shared_account_grants: "SRT sessions share a Windows account and runtime read permissions.",
    no_os_isolation: "No OS isolation.",
    host_files_readable: "Files available to Studio may be readable.",
    unrestricted_network: "Network access is unrestricted.",
    detached_descendant_cleanup_unverified: "Cleanup of detached child processes is not verified.",
  };
  return labels[value] ?? value.replaceAll("_", " ");
}
