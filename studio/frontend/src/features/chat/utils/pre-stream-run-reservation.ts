const DEFAULT_THREAD_ID = "__default";

type PreStreamRunReservation = {
  threadIds: Set<string>;
};

const reservations = new Map<symbol, PreStreamRunReservation>();
const reservationByThreadId = new Map<string, symbol>();

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
): symbol | null {
  const ids = normalizedThreadIds(threadIds);
  if (ids.some((threadId) => reservationByThreadId.has(threadId))) {
    return null;
  }
  const token = Symbol("pre-stream-run");
  reservations.set(token, { threadIds: new Set(ids) });
  for (const threadId of ids) {
    reservationByThreadId.set(threadId, token);
  }
  return token;
}

export function adoptPreStreamRunReservation(
  token: symbol,
  threadIds: Iterable<string | null | undefined>,
): boolean {
  const reservation = reservations.get(token);
  if (!reservation) {
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
