// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import type {
  ReasoningHighlightReply,
  ReasoningHighlightRequest,
} from "./reasoning-highlight";

export type ReasoningLineTokens = Map<
  number,
  ReasoningHighlightReply["lines"][number]["tokens"]
>;
let worker: Worker | null = null;
let nextClient = 0;
let idle: ReturnType<typeof setTimeout> | undefined;
const listeners = new Map<number, (reply: ReasoningHighlightReply) => void>();

function getWorker(): Worker | null {
  if (idle) clearTimeout(idle);
  if (worker) return worker;
  if (typeof Worker === "undefined") return null;
  try {
    worker = new Worker(
      new URL("./reasoning-highlight.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.onmessage = ({ data }: MessageEvent<ReasoningHighlightReply>) =>
      listeners.get(data.client)?.(data);
    worker.onerror = () => {
      worker?.terminate();
      worker = null;
    };
    return worker;
  } catch {
    return null;
  }
}

/** Text is synchronous; expensive grammar work must never hold up chat or scrolling. */
export function useReasoningHighlight(
  source: string,
  language: string | null,
  lines: number[],
): ReasoningLineTokens {
  const [tokens, setTokens] = useState<ReasoningLineTokens>(() => new Map());
  const client = useRef<number | null>(null);
  const revision = useRef(0);
  const sent = useRef<{ worker: Worker; source: string } | null>(null);
  const lineKey = lines.join(",");
  useEffect(() => {
    const instance = getWorker();
    if (!instance) return;
    const id = (client.current ??= ++nextClient);
    const version = ++revision.current;
    listeners.set(id, (reply) => {
      if (reply.revision !== version) return;
      setTokens(new Map(reply.lines.map(({ line, tokens }) => [line, tokens])));
    });
    const request: ReasoningHighlightRequest = {
      client: id,
      revision: version,
      source:
        sent.current?.worker === instance &&
        source.startsWith(sent.current.source)
          ? {
              from: sent.current.source.length,
              text: source.slice(sent.current.source.length),
            }
          : source,
      language,
      lines: lineKey.split(",").filter(Boolean).map(Number),
    };
    instance.postMessage(request);
    sent.current = { worker: instance, source };
  }, [source, language, lineKey]);
  useEffect(
    () => () => {
      if (client.current !== null) {
        sent.current = null;
        listeners.delete(client.current);
        worker?.postMessage({ cancel: client.current });
      }
      if (listeners.size === 0)
        idle = setTimeout(() => {
          worker?.terminate();
          worker = null;
        }, 10_000);
    },
    [],
  );
  return tokens;
}
