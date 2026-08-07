


/** Run `task` over `items`, at most `limit` at a time, in order. */
export async function runWithConcurrency<T>(
  items: readonly T[],
  limit: number,
  task: (item: T) => Promise<void>,
): Promise<void> {
  const lanes = Math.max(1, Math.min(limit, items.length));
  let next = 0;
  await Promise.all(
    Array.from({ length: lanes }, async () => {
      while (next < items.length) {
        const item = items[next];
        next += 1;
        await task(item as T);
      }
    }),
  );
}
