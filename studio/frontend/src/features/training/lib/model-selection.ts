


import { looksLikeLocalPath } from "@/lib/local-path";

export function isLocalTrainingModelSelection({
  model,
  knownCached,
  localPath,
}: {
  model: string | null;
  knownCached: boolean;
  localPath: string | null;
}): boolean {
  return Boolean(
    model &&
      (looksLikeLocalPath(model) ||
        (!knownCached && localPath && localPath.trim())),
  );
}
