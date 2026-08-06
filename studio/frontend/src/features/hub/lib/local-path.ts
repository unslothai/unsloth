


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
