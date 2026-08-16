export function modelRefreshErrorMessage(
  error: unknown,
  surfaceErrors = true,
): string | null {
  if (!surfaceErrors) {
    return null;
  }
  return error instanceof Error ? error.message : "Failed to load models";
}
