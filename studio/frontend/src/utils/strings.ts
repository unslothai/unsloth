export function normalizeNonEmptyName(
  value: string,
  fallback = "Unnamed",
): string {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : fallback;
}

