


/** Lowercase and strip whitespace and separators so "Llama-3.1" matches "llama 3 1". */
export function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s_.-]/g, "");
}

export function tokenizeQuery(query: string): string[] {
  return query
    .split(/\s+/)
    .map((token) => normalizeForSearch(token))
    .filter((token) => token.length > 0);
}

export function matchTokens(haystack: string, tokens: readonly string[]): boolean {
  if (tokens.length === 0) return true;
  const normalized = normalizeForSearch(haystack);
  for (const token of tokens) {
    if (!normalized.includes(token)) return false;
  }
  return true;
}
