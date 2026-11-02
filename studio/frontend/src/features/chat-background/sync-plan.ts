// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatBackgroundRecord } from "@/features/settings/api/device-sync";

export type PullPlan = "push" | "keep" | "clear" | "adjust" | "download";

export function planBackgroundPull(
	remote: Pick<ChatBackgroundRecord, "saved" | "updatedAt" | "image">,
	sync: { updatedAt: number | null; sha256: string | null; dirty: boolean },
	hasLocal: boolean,
): PullPlan {
	if (sync.dirty || (!remote.saved && hasLocal)) return "push";
	if (!remote.saved || remote.updatedAt === sync.updatedAt) return "keep";
	if (!remote.image) return hasLocal ? "clear" : "keep";
	return hasLocal && remote.image.sha256 === sync.sha256 ? "adjust" : "download";
}
