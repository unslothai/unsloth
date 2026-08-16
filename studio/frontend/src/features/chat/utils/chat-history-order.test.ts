import { describe, expect, it } from "vitest";

import type { MessageRecord } from "../types";
import { prepareChatHistoryMessages } from "./chat-history-order";

function message(
  id: string,
  parentId: string | null,
  role: MessageRecord["role"],
  createdAt: number,
): MessageRecord {
  return {
    id,
    threadId: "session-1",
    parentId,
    role,
    content: [{ type: "text", text: id }],
    createdAt,
  };
}

describe("prepareChatHistoryMessages", () => {
  it("preserves a parented backend chain when timestamp units differ", () => {
    const history = [
      message("prologue", null, "assistant", 1_786_770_118_407),
      message("user-1", "prologue", "user", 1_786_770_118_408),
      message("assistant-1", "user-1", "assistant", 1_786_770_122),
      message("user-2", "assistant-1", "user", 1_786_770_118_410),
      message("assistant-2", "user-2", "assistant", 1_786_770_386),
    ];

    const prepared = prepareChatHistoryMessages(history);

    expect(prepared.hasParentIds).toBe(true);
    expect(prepared.messages.map(({ id }) => id)).toEqual([
      "prologue",
      "user-1",
      "assistant-1",
      "user-2",
      "assistant-2",
    ]);
    const seen = new Set<string>();
    for (const item of prepared.messages) {
      expect(
        typeof item.parentId !== "string" || seen.has(item.parentId),
      ).toBe(true);
      seen.add(item.id);
    }
  });

  it("moves a child behind its known parent without reordering siblings", () => {
    const prepared = prepareChatHistoryMessages([
      message("child", "parent", "assistant", 1),
      message("parent", null, "user", 2),
      message("sibling", "parent", "assistant", 3),
    ]);

    expect(prepared.messages.map(({ id }) => id)).toEqual([
      "parent",
      "child",
      "sibling",
    ]);
  });

  it("keeps chronological ordering for legacy histories without parents", () => {
    const prepared = prepareChatHistoryMessages([
      message("assistant", null, "assistant", 20),
      message("user", null, "user", 10),
    ]);

    expect(prepared.hasParentIds).toBe(false);
    expect(prepared.messages.map(({ id }) => id)).toEqual([
      "user",
      "assistant",
    ]);
  });
});
