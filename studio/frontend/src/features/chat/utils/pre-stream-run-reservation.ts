const DEFAULT_THREAD_ID = "__default";

type PreStreamRunReservation = {
  threadIds: Set<string>;
  usesLocalModel: boolean;
  cancel?: (threadIds: string[]) => void;
  cancelled: boolean;
  claimed: boolean;
};

export interface LocalPreStreamRunReservation {
  token: symbol;
  threadIds: string[];
}

export interface PreStreamRunReservationOptions {
  usesLocalModel: boolean;
  cancel?: (threadIds: string[]) => void;
}

const reservations = new Map<symbol, PreStreamRunReservation>();
const reservationByThreadId = new Map<string, symbol>();

export function preStreamRunThreadIdsForAdapter(
  unstableThreadId: string | null | undefined,
  activeThreadId: string | null | undefined,
): string[] {
  return preStreamRunThreadIdsForRuntime([unstableThreadId], activeThreadId);
}

export function preStreamRunThreadIdsForRuntime(
  runtimeThreadIds: Iterable<string | null | undefined>,
  activeThreadId: string | null | undefined,
): string[] {
  const identifiedThreadIds = [
    ...new Set(Array.from(runtimeThreadIds).filter(Boolean)),
  ] as string[];
  if (identifiedThreadIds.length > 0) {
    return identifiedThreadIds;
  }
  return activeThreadId ? [activeThreadId] : [];
}

function normalizedThreadIds(
  threadIds: Iterable<string | null | undefined>,
): string[] {
  const ids = [...new Set(Array.from(threadIds).filter(Boolean))] as string[];
  return ids.length > 0 ? ids : [DEFAULT_THREAD_ID];
}

export function findPreStreamRunReservation(
  threadIds: Iterable<string | null | undefined>,
): symbol | null {
  for (const threadId of normalizedThreadIds(threadIds)) {
    const token = reservationByThreadId.get(threadId);
    if (token) {
      return token;
    }
  }
  return null;
}

export function hasPreStreamRunReservation(
  threadIds: Iterable<string | null | undefined>,
): boolean {
  return findPreStreamRunReservation(threadIds) !== null;
}

export function reservePreStreamRun(
  threadIds: Iterable<string | null | undefined>,
  options: PreStreamRunReservationOptions = { usesLocalModel: true },
): symbol | null {
  const ids = normalizedThreadIds(threadIds);
  if (ids.some((threadId) => reservationByThreadId.has(threadId))) {
    return null;
  }
  const token = Symbol("pre-stream-run");
  reservations.set(token, {
    threadIds: new Set(ids),
    usesLocalModel: options.usesLocalModel,
    cancel: options.cancel,
    cancelled: false,
    claimed: false,
  });
  for (const threadId of ids) {
    reservationByThreadId.set(threadId, token);
  }
  return token;
}

export function listLocalPreStreamRunReservations(): LocalPreStreamRunReservation[] {
  return Array.from(reservations.entries())
    .filter(
      ([, reservation]) => reservation.usesLocalModel && !reservation.cancelled,
    )
    .map(([token, reservation]) => ({
      token,
      threadIds: [...reservation.threadIds].filter(
        (threadId) => threadId !== DEFAULT_THREAD_ID,
      ),
    }));
}

function cancelPreStreamReservation(token: symbol): boolean {
  const reservation = reservations.get(token);
  if (!reservation || reservation.cancelled) return false;
  reservation.cancelled = true;
  try {
    reservation.cancel?.(
      [...reservation.threadIds].filter((id) => id !== DEFAULT_THREAD_ID),
    );
  } catch {
    // A response may have ended between the user's gesture and cancellation.
  }
  reservation.cancel = undefined;
  for (const id of reservation.threadIds) {
    if (reservationByThreadId.get(id) === token)
      reservationByThreadId.delete(id);
  }
  if (!reservation.claimed) reservations.delete(token);
  return true;
}

export function cancelPreStreamRunReservations(
  tokens: Iterable<symbol>,
): number {
  let cancelled = 0;
  for (const token of new Set(tokens)) {
    // Model unload and app-wide stop actions remain local-model-only.
    if (
      reservations.get(token)?.usesLocalModel &&
      cancelPreStreamReservation(token)
    )
      cancelled += 1;
  }
  return cancelled;
}

/** A composer steering gesture may cancel its own local or external response. */
export function cancelPreStreamRunForThreadIds(
  threadIds: readonly string[],
): boolean {
  // An unresolved composer must never target the shared default reservation.
  if (threadIds.length === 0) return false;
  const token = findPreStreamRunReservation(threadIds);
  return token ? cancelPreStreamReservation(token) : false;
}

export function claimPreStreamRunReservation(token: symbol): boolean {
  const reservation = reservations.get(token);
  if (!reservation || reservation.cancelled) {
    return false;
  }
  reservation.claimed = true;
  return true;
}

export function isPreStreamRunReservationCancelled(token: symbol): boolean {
  return reservations.get(token)?.cancelled ?? false;
}

export function adoptPreStreamRunReservation(
  token: symbol,
  threadIds: Iterable<string | null | undefined>,
): boolean {
  const reservation = reservations.get(token);
  if (!reservation || reservation.cancelled) {
    return false;
  }
  const ids = normalizedThreadIds(threadIds);
  if (
    ids.some((threadId) => {
      const existing = reservationByThreadId.get(threadId);
      return existing !== undefined && existing !== token;
    })
  ) {
    return false;
  }
  for (const threadId of ids) {
    reservation.threadIds.add(threadId);
    reservationByThreadId.set(threadId, token);
  }
  return true;
}

export function releasePreStreamRunReservation(token: symbol): boolean {
  const reservation = reservations.get(token);
  if (!reservation) {
    return false;
  }
  reservations.delete(token);
  for (const threadId of reservation.threadIds) {
    if (reservationByThreadId.get(threadId) === token) {
      reservationByThreadId.delete(threadId);
    }
  }
  return true;
}

export function releasePreStreamRunForThreadIds(
  threadIds: Iterable<string | null | undefined>,
): boolean {
  const token = findPreStreamRunReservation(threadIds);
  return token ? releasePreStreamRunReservation(token) : false;
}
