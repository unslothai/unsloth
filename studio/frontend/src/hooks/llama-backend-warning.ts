export function resolveLlamaBackendForWarning({
  backend,
  envBackend,
}: {
  backend: string | null;
  envBackend: string | null;
}): string | null {
  return envBackend && envBackend !== "auto" ? envBackend : backend;
}
