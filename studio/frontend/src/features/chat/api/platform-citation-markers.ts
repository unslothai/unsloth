const PLATFORM_CITATION_MARKER_RE =
  /[ \t]*(?:\[|\(|【)\s*ID\s*[:： ]*\s*\d+\s*(?:\]|\)|】)/gi;

/**
 * Rag Platform uses `[ID:n]` as an internal pointer into the separately
 * returned reference payload. The UI renders that payload as document source
 * buttons, so exposing the raw pointer in the answer is redundant and
 * confusing. Keep the persisted/backend answer intact and normalize only the
 * frontend domain boundary.
 */
export function stripPlatformCitationMarkers(text: string): string {
  return text.replace(PLATFORM_CITATION_MARKER_RE, "");
}
