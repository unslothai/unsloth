// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
	clearChatBackground,
	loadChatBackground,
	loadChatBackgroundImage,
	saveChatBackgroundAdjustments,
	saveChatBackgroundImage,
} from "@/features/settings/api/device-sync";
import { sanitizeAdjustments } from "./model";
import { planBackgroundPull } from "./sync-plan";
import {
	applyServerBackground,
	hydrateBackground,
	readBackgroundSync,
	setBackgroundChangeListener,
	useBackgroundStore,
	writeBackgroundSync,
} from "./storage";

const PUSH_DEBOUNCE_MS = 800;

let queue: Promise<void> = Promise.resolve();
let timer: ReturnType<typeof setTimeout> | undefined;
let changes = 0;

function enqueue(task: () => Promise<void>) {
	queue = queue.then(task).catch(() => undefined);
	return queue;
}

async function push() {
	const started = changes;
	const value = useBackgroundStore.getState().value;
	if (!value) {
		const record = await clearChatBackground();
		if (changes === started)
			await writeBackgroundSync({ dirty: false, updatedAt: record.updatedAt, sha256: null });
		return;
	}
	let { sha256 } = await readBackgroundSync();
	if (!sha256) {
		sha256 = (await saveChatBackgroundImage(value.blob)).image?.sha256 ?? null;
		// The image is the account's now even if adjustments moved meanwhile.
		if (useBackgroundStore.getState().value?.blob === value.blob)
			await writeBackgroundSync({ sha256 });
	}
	const { blob: _blob, ...adjustments } = value;
	const record = await saveChatBackgroundAdjustments(adjustments).catch(
		async (error: unknown) => {
			// Another browser removed the image in between: upload it again next time.
			await writeBackgroundSync({ sha256: null });
			throw error;
		},
	);
	if (changes === started)
		await writeBackgroundSync({ dirty: false, updatedAt: record.updatedAt, sha256 });
}

async function pull() {
	await hydrateBackground();
	const [remote, sync] = await Promise.all([
		loadChatBackground(),
		readBackgroundSync(),
	]);
	const local = useBackgroundStore.getState().value;
	const plan = planBackgroundPull(remote, sync, Boolean(local));
	if (plan === "push") return push();
	if (plan === "keep") return;
	const stamp = { updatedAt: remote.updatedAt, sha256: remote.image?.sha256 ?? null };
	if (plan === "clear" || !remote.image) {
		await applyServerBackground(null, stamp);
		return;
	}
	const blob =
		plan === "adjust" && local ? local.blob : await loadChatBackgroundImage();
	await applyServerBackground(
		{ ...sanitizeAdjustments(remote.adjustments ?? {}), blob },
		stamp,
	);
}

/** Keeps the background with the account, so another browser or origin shows the same one. */
export function startChatBackgroundSync(): () => void {
	setBackgroundChangeListener(() => {
		changes++;
		clearTimeout(timer);
		timer = setTimeout(() => void enqueue(push), PUSH_DEBOUNCE_MS);
	});
	void enqueue(pull);
	return () => {
		setBackgroundChangeListener(null);
		clearTimeout(timer);
	};
}
