


const PATH_SEPARATOR_RE = /[\\/]/;
const TRAILING_PATH_SEPARATOR_RE = /[\\/]+$/;

export function pathDisplayName(value: string): string {
  const withoutTrailingSeparators = value.replace(
    TRAILING_PATH_SEPARATOR_RE,
    "",
  );
  const candidate = withoutTrailingSeparators || value;
  return candidate.split(PATH_SEPARATOR_RE).pop() || candidate;
}
