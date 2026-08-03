export class ChatHistoryClearBoundary {
  private generation = 0;
  private pendingPersistence = new Set<Promise<unknown>>();

  capture(): number {
    return this.generation;
  }

  advance(): number {
    this.generation += 1;
    return this.generation;
  }

  trackPending<T>(work: Promise<T>): Promise<T> {
    this.pendingPersistence.add(work);
    work.then(
      () => this.pendingPersistence.delete(work),
      () => this.pendingPersistence.delete(work),
    );
    return work;
  }

  async waitForPending(): Promise<void> {
    while (this.pendingPersistence.size > 0) {
      await Promise.allSettled([...this.pendingPersistence]);
    }
  }
}

export const chatHistoryClearBoundary = new ChatHistoryClearBoundary();
