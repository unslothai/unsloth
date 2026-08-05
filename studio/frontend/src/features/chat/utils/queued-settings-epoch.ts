export function shouldAdvanceQueuedSettingsEpoch(
  currentValues: Readonly<object>,
  nextValues: Readonly<object>,
  trackQueuedSettings = true,
): boolean {
  if (!trackQueuedSettings) {
    return false;
  }
  const keys = new Set([
    ...Object.keys(currentValues),
    ...Object.keys(nextValues),
  ]);
  const currentRecord = currentValues as Record<string, unknown>;
  const nextRecord = nextValues as Record<string, unknown>;
  return Array.from(keys).some(
    (key) => !Object.is(currentRecord[key], nextRecord[key]),
  );
}
