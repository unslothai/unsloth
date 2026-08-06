


/**
 * The live download pollers, one per dictation model. Each STT engine owns its
 * own download state, so a Qwen transfer and a Whisper one really do run at the
 * same time, and starting one must not drop the other's panel row.
 */
export class SttDownloadTrackers {
  private readonly running = new Map<string, () => void>();

  has(model: string): boolean {
    return this.running.has(model);
  }

  /** Stops this model's previous poller, if any, and registers the new one. */
  start(model: string, stop: () => void): void {
    this.stop(model);
    this.running.set(model, stop);
  }

  stop(model: string): void {
    const stop = this.running.get(model);
    this.running.delete(model);
    stop?.();
  }
}
