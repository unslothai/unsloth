import type { PlatformSseEvent } from "./types";

function isTerminalData(data: string): boolean {
  const trimmed = data.trim();
  if (trimmed === "[DONE]") return true;
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return (
      typeof parsed === "object" &&
      parsed !== null &&
      "data" in parsed &&
      parsed.data === true
    );
  } catch {
    return false;
  }
}

export class PlatformSseParser {
  private readonly decoder = new TextDecoder();
  private buffer = "";
  private dataLines: string[] = [];
  private eventName: string | undefined;
  private eventId: string | undefined;
  private retry: number | undefined;

  feed(chunk: string | Uint8Array): PlatformSseEvent[] {
    this.buffer +=
      typeof chunk === "string"
        ? chunk
        : this.decoder.decode(chunk, { stream: true });
    return this.consumeCompleteLines(false);
  }

  end(): PlatformSseEvent[] {
    this.buffer += this.decoder.decode();
    const events = this.consumeCompleteLines(true);
    const finalEvent = this.dispatch();
    if (finalEvent) events.push(finalEvent);
    return events;
  }

  private consumeCompleteLines(flush: boolean): PlatformSseEvent[] {
    const events: PlatformSseEvent[] = [];
    let text = this.buffer;
    let trailingCarriageReturn = false;

    if (!flush && text.endsWith("\r")) {
      trailingCarriageReturn = true;
      text = text.slice(0, -1);
    }

    text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const lines = text.split("\n");
    this.buffer = flush ? "" : (lines.pop() ?? "");
    if (trailingCarriageReturn) this.buffer += "\r";

    for (const line of lines) {
      const event = this.consumeLine(line);
      if (event) events.push(event);
    }

    if (flush && this.buffer) {
      const event = this.consumeLine(this.buffer.replace(/\r$/, ""));
      if (event) events.push(event);
      this.buffer = "";
    }

    return events;
  }

  private consumeLine(line: string): PlatformSseEvent | undefined {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return undefined;

    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    switch (field) {
      case "data":
        this.dataLines.push(value);
        break;
      case "event":
        this.eventName = value;
        break;
      case "id":
        if (!value.includes("\0")) this.eventId = value;
        break;
      case "retry": {
        const retry = Number(value);
        if (Number.isInteger(retry) && retry >= 0) this.retry = retry;
        break;
      }
    }
    return undefined;
  }

  private dispatch(): PlatformSseEvent | undefined {
    if (this.dataLines.length === 0) {
      this.eventName = undefined;
      this.retry = undefined;
      return undefined;
    }

    const data = this.dataLines.join("\n");
    const event: PlatformSseEvent = {
      data,
      event: this.eventName,
      id: this.eventId,
      retry: this.retry,
      terminal: isTerminalData(data),
    };
    this.dataLines = [];
    this.eventName = undefined;
    this.retry = undefined;
    return event;
  }
}

export async function* parsePlatformSseStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<PlatformSseEvent> {
  const parser = new PlatformSseParser();
  const reader = stream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const event of parser.feed(value)) yield event;
    }
    for (const event of parser.end()) yield event;
  } finally {
    reader.releaseLock();
  }
}
