// src/tools/pty/buffer.ts
import type { SearchMatch } from "./types";

const parsed = parseInt(process.env.PTY_MAX_BUFFER_LINES || "50000", 10);
const DEFAULT_MAX_LINES = isNaN(parsed) ? 50000 : parsed;

export class RingBuffer {
  private lines: string[] = [];
  private maxLines: number;

  constructor(maxLines: number = DEFAULT_MAX_LINES) {
    this.maxLines = maxLines;
  }

  append(data: string): void {
    const newLines = data.split("\n");
    for (const line of newLines) {
      this.lines.push(line);
      if (this.lines.length > this.maxLines) {
        this.lines.shift();
      }
    }
  }

  read(offset: number = 0, limit?: number): string[] {
    const start = Math.max(0, offset);
    const end = limit !== undefined ? start + limit : this.lines.length;
    return this.lines.slice(start, end);
  }

  search(pattern: RegExp): SearchMatch[] {
    const matches: SearchMatch[] = [];
    for (let i = 0; i < this.lines.length; i++) {
      const line = this.lines[i];
      if (line !== undefined && pattern.test(line)) {
        matches.push({ lineNumber: i + 1, text: line });
      }
    }
    return matches;
  }

  get length(): number {
    return this.lines.length;
  }

  clear(): void {
    this.lines = [];
  }
}
