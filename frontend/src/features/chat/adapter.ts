import type { ChatModelAdapter, ChatModelRunResult } from "@assistant-ui/react";

const API = import.meta.env.VITE_INFERENCE_URL || "/api/chat/generate";
type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];
type RunMessages = Parameters<ChatModelAdapter["run"]>[0]["messages"];

function makeBody(messages: RunMessages): string {
  return JSON.stringify({
    messages: messages.map((m) => ({
      role: m.role,
      content: m.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join(""),
    })),
  });
}

export function parseThinkTags(raw: string): ChatModelRunResult["content"] {
  const parts: ContentPart[] = [];
  const thinkStart = raw.indexOf("<think>");
  if (thinkStart === -1) {
    if (raw) {
      parts.push({ type: "text", text: raw });
    }
    return parts;
  }
  const before = raw.slice(0, thinkStart);
  if (before.trim()) {
    parts.push({ type: "text", text: before });
  }

  const thinkEnd = raw.indexOf("</think>");
  if (thinkEnd === -1) {
    const reasoning = raw.slice(thinkStart + 7);
    if (reasoning) {
      parts.push({ type: "reasoning", text: reasoning });
    }
    return parts;
  }
  const reasoning = raw.slice(thinkStart + 7, thinkEnd);
  if (reasoning) {
    parts.push({ type: "reasoning", text: reasoning });
  }

  const after = raw.slice(thinkEnd + 8);
  if (after) {
    parts.push({ type: "text", text: after });
  }
  return parts;
}

export function createStreamAdapter(apiUrl: string = API): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: makeBody(messages),
        signal: abortSignal,
      });
      const reader = res.body?.getReader();
      if (!reader) {
        throw new Error("Response body is empty");
      }
      const decoder = new TextDecoder();
      let text = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        text += decoder.decode(value, { stream: true });
        const parts = parseThinkTags(text) ?? [];
        if (parts.length > 0) {
          yield { content: parts };
        }
      }
    },
  };
}
