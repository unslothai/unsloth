


const ASSISTANT_LOCAL_THREAD_ID_PREFIX = "__LOCALID_";

export function isAssistantLocalThreadId(
	threadId: string | null | undefined,
): boolean {
	return Boolean(threadId?.startsWith(ASSISTANT_LOCAL_THREAD_ID_PREFIX));
}
