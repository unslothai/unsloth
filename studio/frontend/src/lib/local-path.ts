


const LOCAL_PATH_PREFIX_RE =
  /^(?:[\\/]|\.{1,2}(?:$|[\\/])|~(?:$|[\\/])|~[^\\/]+[\\/]|[A-Za-z]:)/;

export function looksLikeLocalPath(input: string): boolean {
  const value = input.trim();
  return value.length > 0 && LOCAL_PATH_PREFIX_RE.test(value);
}
