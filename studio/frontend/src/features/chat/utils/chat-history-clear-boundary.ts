export class ChatHistoryClearBoundary {
  private generation = 0;

  capture(): number {
    return this.generation;
  }

  advance(): number {
    this.generation += 1;
    return this.generation;
  }
}

export const chatHistoryClearBoundary = new ChatHistoryClearBoundary();
