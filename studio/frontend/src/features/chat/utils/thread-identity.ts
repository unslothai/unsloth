type ThreadIdentity = {
  targetThreadId: string | null | undefined;
  mainThreadId: string | null | undefined;
  remoteThreadId: string | null | undefined;
};

/**
 * assistant-ui keeps a local thread id after persistence and exposes the
 * backend session id as remoteId. Either value identifies the mounted thread.
 */
export function isThreadAttached({
  targetThreadId,
  mainThreadId,
  remoteThreadId,
}: ThreadIdentity): boolean {
  return (
    !targetThreadId ||
    targetThreadId === mainThreadId ||
    targetThreadId === remoteThreadId
  );
}
