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
let patchRevision = 0;
let committed: ChatBackgroundValue | null = null;
// Kept with the image it describes: another tab can replace the stored one while this one still shows its own.
let committedSha: string | null = null;
let writes: Promise<unknown> = Promise.resolve();
let hydration: Promise<void> | undefined;
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
export function backgroundRevision(): number {
	return revision;
}
export async function readBackgroundSync(): Promise<BackgroundSyncState> {
	const table = currentDatabase().table("backgrounds");
	const row = await table.get("sync");
	return {
		...CLEAN_SYNC,
		...(row?.state ?? {}),
		sha256: committedSha,
	};
}
export async function writeBackgroundSync(
	patch: Partial<Omit<BackgroundSyncState, "sha256">>,
): Promise<void> {
	const table = currentDatabase().table("backgrounds");
	const row = await table.get("sync");
	const { sha256: _sha, ...state } = { ...CLEAN_SYNC, ...(row?.state ?? {}), ...patch };
	await table.put({ id: "sync", state });
}
export async function recordUploadedImage(
	blob: Blob,
	sha256: string | null,
): Promise<void> {
	const db = currentDatabase();
	const write = writes.then(async () => {
		if (committed?.blob !== blob) return;
		await db.table("backgrounds").put({ id: "current", value: committed, sha256 });
		if (db === database) committedSha = sha256;
	});
	writes = write.catch(() => undefined);
	await write;
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
		committedSha = null;
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
				committedSha =
					committed && typeof row?.sha256 === "string" ? row.sha256 : null;
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
	if (optimistic) {
		patchRevision = ticket;
		useBackgroundStore.setState({ value, error: null });
	}
	const write = writes.then(async () => {
		// A patch is skipped only for a newer patch: both keep the image and its sha256.
		if (ticket !== revision && (fromServer || (optimistic && patchRevision === revision && db === database))) return;
		const sameImage = Boolean(value && committed && value.blob === committed.blob);
		const sha256 = fromServer ? (fromServer.sha256 ?? null) : sameImage ? committedSha : null;
		if (value) await db.table("backgrounds").put({ id: "current", value, sha256 });
		else await db.table("backgrounds").delete("current");
		if (db === database) {
			committed = value;
			committedSha = value ? sha256 : null;
		}
		await writeBackgroundSync(
			fromServer
				? { updatedAt: fromServer.updatedAt ?? null, dirty: false }
				: { dirty: true },
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
export function applyServerBackground(
	value: ChatBackgroundValue | null,
	sync: Pick<BackgroundSyncState, "updatedAt" | "sha256">,
	expectedRevision: number,
): Promise<boolean> {
	if (expectedRevision !== revision) return Promise.resolve(false);
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
/** Resolves false when a newer upload took over or the image could not be stored. */
export async function finishBackgroundUpload(
	ticket: number,
	value: ChatBackgroundValue,
): Promise<boolean> {
	if (ticket !== uploadTicket) return false;
	const stored = await persist(value, false);
	if (ticket === uploadTicket) useBackgroundStore.setState({ busy: false });
	return stored;
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
