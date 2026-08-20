


export { looksLikeLocalPath } from "../../../lib/local-path.ts";

const WINDOWS_PATH_SEPARATOR_RE = /\\/g;
const TRAILING_PATH_SEPARATOR_RE = /\/+$/;

export function localPathCacheKey(path: string | null | undefined): string {
  return (
    path
      ?.replace(WINDOWS_PATH_SEPARATOR_RE, "/")
      .replace(TRAILING_PATH_SEPARATOR_RE, "") ?? ""
  );
}

/** Whether a selected row can be routed to the Images / Video pages, which resolve a routed
 * `model` as a Hub id. Only a FILESYSTEM row cannot: an HF-cache row is a complete Hub
 * snapshot carrying the repo id, and inventory dedup can leave it as the only row for that
 * repo, so excluding it by kind alone dropped it into a chat the backend then refuses. */
export function routableToMediaPage(
  kind: "discover" | "cache" | "local",
  localSource: string | null | undefined,
): boolean {
  return kind !== "local" || localSource === "hf_cache";
}
