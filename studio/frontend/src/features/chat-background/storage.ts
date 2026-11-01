// SPDX-License-Identifier: AGPL-3.0-only
import Dexie from "dexie";
import { useEffect } from "react";
import { create } from "zustand";
import { accountDatabaseName } from "@/lib/account-transition";
import {
	sanitizeAdjustments,
	type ChatBackgroundValue,
	type BackgroundAdjustments,
	type BackgroundError,
} from "./model";

export const useBackgroundStore = create<{
	value: ChatBackgroundValue | null;
	error: BackgroundError | null;
	ready: boolean;
	busy: boolean;
}>(() => ({ value: null, error: null, ready: false, busy: false }));
let database: Dexie | undefined;
let account = "";
let revision = 0;
let committed: ChatBackgroundValue | null = null;
let writes: Promise<unknown> = Promise.resolve();
let hydration: Promise<void> | undefined;
/** What the server last confirmed for this browser's copy; `dirty` = a local edit it has not seen. */
export type BackgroundSyncState = {
	updatedAt: number | null;
	sha256: string | null;
	dirty: boolean;
};
const CLEAN_SYNC: BackgroundSyncState = {
	updatedAt: null,
	sha256: null,
	dirty: false,
};
let onLocalChange: (() => void) | null = null;
export function setBackgroundChangeListener(listener: (() => void) | null) {
	onLocalChange = listener;
}
export async function readBackgroundSync(): Promise<BackgroundSyncState> {
	const row = await currentDatabase().table("backgrounds").get("sync");
	return { ...CLEAN_SYNC, ...(row?.state ?? {}) };
}
export async function writeBackgroundSync(
	patch: Partial<BackgroundSyncState>,
): Promise<void> {
	const db = currentDatabase();
	const state = { ...(await readBackgroundSync()), ...patch };
	await db.table("backgrounds").put({ id: "sync", state });
}
function currentDatabase() {
	const name = accountDatabaseName("unsloth-chat-background");
	if (!database || account !== name) {
		database?.close();
		account = name;
		revision++;
		database = new Dexie(name);
		database.version(1).stores({ backgrounds: "id" });
		hydration = undefined;
		committed = null;
		writes = Promise.resolve();
		useBackgroundStore.setState({
			value: null,
			error: null,
			ready: false,
			busy: false,
		});
	}
	return database;
}
export function hydrateBackground(): Promise<void> {
	const db = currentDatabase();
	if (!hydration) {
		const ticket = revision;
		hydration = db
			.table("backgrounds")
			.get("current")
			.then((row) => {
				if (ticket !== revision) return;
				committed =
					row?.value?.blob instanceof Blob
						? { ...sanitizeAdjustments(row.value), blob: row.value.blob }
						: null;
				useBackgroundStore.setState({ value: committed, ready: true });
			})
			.catch(() => {
				if (ticket === revision)
					useBackgroundStore.setState({ error: "storage", ready: true });
			});
	}
	return hydration;
}
export function useChatBackground() {
	useEffect(() => {
		void hydrateBackground();
	}, []);
	return useBackgroundStore();
}
export function setBackgroundError(error: BackgroundError) {
	useBackgroundStore.setState({ error });
}
async function persist(
	value: ChatBackgroundValue | null,
	optimistic: boolean,
	fromServer?: Partial<BackgroundSyncState>,
): Promise<boolean> {
	const db = currentDatabase();
	const ticket = ++revision;
	if (optimistic) useBackgroundStore.setState({ value, error: null });
	const write = writes.then(async () => {
		const sameImage = Boolean(value && committed && value.blob === committed.blob);
		if (value) await db.table("backgrounds").put({ id: "current", value });
		else await db.table("backgrounds").delete("current");
		if (db === database) committed = value;
		await writeBackgroundSync(
			fromServer
				? { ...fromServer, dirty: false }
				: sameImage
					? { dirty: true }
					: { dirty: true, sha256: null },
		);
	});
	writes = write.catch(() => undefined);
	try {
		await write;
		if (ticket === revision)
			useBackgroundStore.setState({ value, error: null, ready: true });
		if (!fromServer) onLocalChange?.();
		return true;
	} catch {
		if (ticket === revision)
			useBackgroundStore.setState({
				value: committed,
				error: "storage",
				ready: true,
			});
		return false;
	}
}
/** Adopts the account's copy without marking it as a local edit to send back. */
export function applyServerBackground(
	value: ChatBackgroundValue | null,
	sync: Pick<BackgroundSyncState, "updatedAt" | "sha256">,
): Promise<boolean> {
	return persist(value, false, sync);
}
export function patchBackground(patch: Partial<BackgroundAdjustments>) {
	const value = useBackgroundStore.getState().value;
	if (value)
		void persist(
			{ ...value, ...sanitizeAdjustments({ ...value, ...patch }) },
			true,
		);
}
let uploadTicket = 0;
export function beginBackgroundUpload() {
	useBackgroundStore.setState({ busy: true, error: null });
	return ++uploadTicket;
}
export async function finishBackgroundUpload(
	ticket: number,
	value: ChatBackgroundValue,
) {
	if (ticket !== uploadTicket) return;
	await persist(value, false);
	if (ticket === uploadTicket) useBackgroundStore.setState({ busy: false });
}
export function failBackgroundUpload(ticket: number, error: BackgroundError) {
	if (ticket === uploadTicket)
		useBackgroundStore.setState({ busy: false, error });
}
export async function resetChatBackground(): Promise<boolean> {
	uploadTicket++;
	useBackgroundStore.setState({ busy: false });
	return persist(null, false);
}
