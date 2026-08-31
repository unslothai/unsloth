// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ASSISTANT_LOCAL_THREAD_ID_PREFIX = "__LOCALID_";

export function isAssistantLocalThreadId(
	threadId: string | null | undefined,
): boolean {
	return Boolean(threadId?.startsWith(ASSISTANT_LOCAL_THREAD_ID_PREFIX));
}
