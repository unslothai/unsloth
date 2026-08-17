const KEY = "rag-platform-memory-consent-v1";

function read(): Record<string, boolean> {
  try {
    const value = JSON.parse(window.localStorage.getItem(KEY) ?? "{}") as unknown;
    return value && typeof value === "object" ? (value as Record<string, boolean>) : {};
  } catch {
    return {};
  }
}

export function hasPlatformMemoryConsent(memoryId: string): boolean {
  return read()[memoryId] === true;
}

export function setPlatformMemoryConsent(memoryId: string, consent: boolean) {
  const values = read();
  if (consent) values[memoryId] = true;
  else delete values[memoryId];
  try {
    window.localStorage.setItem(KEY, JSON.stringify(values));
  } catch {
    // The consent still applies to the current rendered state if storage is unavailable.
  }
}
