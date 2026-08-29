// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  streamedToolCallArguments,
  toolCallReplayArguments,
} from "../src/features/chat/tool-call-arguments.ts";
import {
  type StreamedToolCallPart,
  bindStreamedToolCallBackendIds,
  findDelayedStableToolCallPartIndex,
  findOldestUnownedStreamedToolCallPartIndex,
  findStreamedToolCallPartIndex,
  fragmentStartsNewToolCall,
  isRepeatedJsonSnapshot,
  mergeStreamedToolCallName,
  mintStreamedToolCallId,
  sameJsonDocument,
  splitTopLevelJsonDocuments,
} from "../src/features/chat/tool-call-id.ts";

interface DeltaFragment {
  id?: string;
  index?: number;
  name?: string;
  arguments: string;
}

type AccumulatedPart = StreamedToolCallPart & {
  argsText: string;
  toolName?: string;
  splitTail?: boolean;
};

/** Accumulate `delta.tool_calls[]` fragments the way the chat adapter does. */
function accumulate(
  fragments: DeltaFragment[],
): AccumulatedPart[] {
  const parts: AccumulatedPart[] = [];
  const usedIds = new Set<string>();
  const providerIds = new Map<string, string>();
  const append = (
    fragment: DeltaFragment,
    argsText: string,
    id = fragment.id,
    splitTail = false,
  ) => {
    parts.push({
      toolCallId:
        id ?? mintStreamedToolCallId(parts, fragment.index, usedIds),
      toolName: fragment.name,
      argsText,
      ...(id ? { _has_stable_id: true } : {}),
      ...(fragment.index !== undefined ? { _delta_index: fragment.index } : {}),
      ...(splitTail ? { splitTail: true } : {}),
    });
    usedIds.add(parts[parts.length - 1].toolCallId);
  };
  for (const rawFragment of fragments) {
    let fragment = rawFragment;
    let partId = fragment.id ? providerIds.get(fragment.id) : undefined;
    if (fragment.id && !partId) {
      const providerId = fragment.id;
      const delayed = findDelayedStableToolCallPartIndex(
        parts,
        fragment.index,
        fragment.name ?? "",
        fragment.arguments,
      );
      if (delayed !== -1) {
        const provisionalId = parts[delayed].toolCallId;
        partId = providerId;
        const displaced = parts.findIndex(
          (part, index) => index !== delayed && part.toolCallId === partId,
        );
        if (displaced !== -1 && provisionalId !== partId) {
          parts[displaced].toolCallId = provisionalId;
        } else {
          usedIds.delete(provisionalId);
        }
        parts[delayed].toolCallId = partId;
        parts[delayed]._has_stable_id = true;
        if (sameJsonDocument(fragment.arguments, parts[delayed].argsText)) {
          fragment = { ...fragment, arguments: "" };
        }
      } else {
        partId = usedIds.has(providerId)
          ? mintStreamedToolCallId(parts, fragment.index, usedIds)
          : providerId;
      }
      providerIds.set(providerId, partId);
      usedIds.add(partId);
    }
    const exact = partId
      ? parts.findIndex((part) => part.toolCallId === partId)
      : -1;
    const matched =
      exact !== -1
        ? exact
        : partId && !fragment.name && !fragment.arguments.trim()
          ? findOldestUnownedStreamedToolCallPartIndex(parts, fragment.index)
          : findStreamedToolCallPartIndex(parts, partId, fragment.index);
    const matchedPart = matched === -1 ? undefined : parts[matched];
    const exactId = Boolean(
      partId && matchedPart?.toolCallId === partId,
    );
    if (
      exactId &&
      isRepeatedJsonSnapshot(matchedPart?.argsText ?? "", fragment.arguments)
    ) {
      fragment = { ...fragment, arguments: "" };
    }
    const settled = matchedPart
      ? splitTopLevelJsonDocuments(matchedPart.argsText)
      : null;
    const target =
      matched !== -1 &&
      !exactId &&
      settled?.complete.length === 1 &&
      !settled.tail &&
      Boolean(matchedPart?.toolName) &&
      Boolean(fragment.name) &&
      !fragment.arguments.trim()
        ? -1
        : matched;
    if (!exactId) {
      const split = splitTopLevelJsonDocuments(
        (matchedPart?.argsText ?? "") + fragment.arguments,
      );
      const segments = [...split.complete, ...(split.tail ? [split.tail] : [])];
      if (segments.length > 1) {
        const firstNewSegment = matchedPart ? 1 : 0;
        if (matchedPart)
          parts[matched] = { ...matchedPart, argsText: segments[0] };
        for (const [segmentIndex, segment] of segments
          .slice(firstNewSegment)
          .entries()) {
          append(
            fragment,
            segment,
            segmentIndex === 0
              ? partId
              : mintStreamedToolCallId(parts, fragment.index, usedIds),
            Boolean(split.tail && segment === split.tail),
          );
        }
        continue;
      }
    }
    if (target === -1) {
      append(fragment, fragment.arguments, partId);
      continue;
    }
    const mergedArgsText = parts[target].argsText + fragment.arguments;
    const mergedDocuments = splitTopLevelJsonDocuments(mergedArgsText);
    parts[target] = {
      ...parts[target],
      ...(partId ? { toolCallId: partId, _has_stable_id: true } : {}),
      toolName: mergeStreamedToolCallName(
        parts[target].toolName ?? "",
        fragment.name ?? "",
      ),
      argsText: mergedArgsText,
      splitTail:
        parts[target].splitTail &&
        !(
          mergedDocuments.complete.length === 1 &&
          !mergedDocuments.tail
        ),
    };
    if (partId) usedIds.add(partId);
  }
  return parts.filter((part) => !part.splitTail);
}

test("tool rounds that both reuse index 0 stay separate calls", () => {
  const parts = accumulate([
    { id: "call-A", index: 0, arguments: '{"query":' },
    { index: 0, arguments: '"first"}' },
    { id: "call-B", index: 0, arguments: '{"query":' },
    { index: 0, arguments: '"second"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-A", '{"query":"first"}'],
      ["call-B", '{"query":"second"}'],
    ],
  );
});

test("a fragment with a fresh id never merges into a slot another id owns", () => {
  assert.equal(
    findStreamedToolCallPartIndex(
      [{ toolCallId: "call-A", _delta_index: 0, _has_stable_id: true }],
      "call-B",
      0,
    ),
    -1,
  );
});

test("an id stamped on a later fragment adopts its own opening slot", () => {
  const parts = accumulate([
    { index: 0, arguments: "" },
    { id: "call-A", index: 0, arguments: '{"query":' },
    { index: 0, arguments: '"first"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [["call-A", '{"query":"first"}']],
  );
});

test("a late id still keeps the next round's call separate", () => {
  const parts = accumulate([
    { index: 0, arguments: "" },
    { id: "call-A", index: 0, arguments: '{"query":"first"}' },
    { id: "call-B", index: 0, arguments: '{"query":"second"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-A", '{"query":"first"}'],
      ["call-B", '{"query":"second"}'],
    ],
  );
});

test("parallel calls in one round accumulate per index", () => {
  const parts = accumulate([
    { id: "call-A", index: 0, arguments: '{"a":' },
    { id: "call-B", index: 1, arguments: '{"b":' },
    { index: 0, arguments: "1}" },
    { index: 1, arguments: "2}" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-A", '{"a":1}'],
      ["call-B", '{"b":2}'],
    ],
  );
});

test("an id-less stream still accumulates by index alone", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"q":' },
    { index: 0, arguments: '"x"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [["tool_call_0", '{"q":"x"}']],
  );
});

test("a fragment carrying neither id nor index starts its own call", () => {
  assert.equal(
    findStreamedToolCallPartIndex(
      [{ toolCallId: "call-A", _delta_index: 0 }],
      undefined,
      undefined,
    ),
    -1,
  );
});

test("replayed arguments keep parsable text and fall back otherwise", () => {
  assert.equal(
    toolCallReplayArguments('{"query":"first"}', { query: "first" }),
    '{"query":"first"}',
  );
  assert.equal(
    toolCallReplayArguments('{"query":"first"}{"query":"second"}', {
      _raw: '{"query":"first"}{"query":"second"}',
    }),
    "{}",
  );
  assert.equal(
    toolCallReplayArguments("", { query: "first" }),
    '{"query":"first"}',
  );
  assert.equal(toolCallReplayArguments(undefined, undefined), "{}");
  assert.equal(
    toolCallReplayArguments(undefined, { _raw: "user supplied" }),
    '{"_raw":"user supplied"}',
  );
});

test("id-less parallel calls renumbered to index 0 stay separate calls", () => {
  // LiteLLM's proxy can strip ids and renumber every parallel call's index to
  // 0. Each opening fragment then "continues" the newest slot-0 part and the
  // argument JSONs concatenate into one malformed string that poisons the
  // thread on replay (#9807).
  const parts = accumulate([
    { index: 0, arguments: '{"url":"https://example.com/1"}' },
    { index: 0, arguments: '{"url":"https://example.com/2"}' },
    { index: 0, arguments: '{"query":"example search"}' },
    { index: 0, arguments: '{"url":"https://example.com/3"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    [
      '{"url":"https://example.com/1"}',
      '{"url":"https://example.com/2"}',
      '{"query":"example search"}',
      '{"url":"https://example.com/3"}',
    ],
  );
  assert.equal(new Set(parts.map((part) => part.toolCallId)).size, 4);
});

test("an id-less chunked continuation still merges into its own call", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"url":' },
    { index: 0, arguments: '"https://example.com/1"}' },
    { index: 0, arguments: '{"url":"https://example.com/2"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    ['{"url":"https://example.com/1"}', '{"url":"https://example.com/2"}'],
  );
});

test("JSON document boundaries survive strings, nesting, whitespace, and Unicode", () => {
  const documents = [
    JSON.stringify({ nested: { items: [1, { text: '} ] { " \\\\ 雪' }] } }),
    "[]",
    "{}",
    '[{"emoji":"🦥"}]',
  ];
  const split = splitTopLevelJsonDocuments(` \n${documents.join(" \t\r\n")}  `);

  assert.deepEqual(split.complete, documents);
  assert.equal(split.tail, "");
});

test("JSON document boundaries keep an incomplete final document as the tail", () => {
  assert.deepEqual(splitTopLevelJsonDocuments('{"a":1} \n ["open"'), {
    complete: ['{"a":1}'],
    tail: '["open"',
  });
});

test("JSON document boundaries reject mismatched or non-document text", () => {
  for (const text of [
    '{"a":1]',
    '{"a":1}suffix',
    '"{\\"a\\":1}"',
    'true{"a":1}',
  ]) {
    assert.deepEqual(splitTopLevelJsonDocuments(text), {
      complete: [],
      tail: text,
    });
  }
});

test("new-call detection works when the SSE boundary lands inside either document", () => {
  const first = '{"path":"C:\\\\tmp","nested":[{"text":"{x}"}]}';
  const second = '{"query":"雪 🦥"}';
  for (let firstCut = 1; firstCut <= first.length; firstCut += 1) {
    const existing = first.slice(0, firstCut);
    const remainder = first.slice(firstCut) + second;
    assert.equal(
      fragmentStartsNewToolCall(existing, remainder),
      true,
      `missed boundary after first character ${firstCut}`,
    );
  }
  for (let secondCut = 1; secondCut <= second.length; secondCut += 1) {
    assert.equal(
      fragmentStartsNewToolCall(first, second.slice(0, secondCut)),
      true,
      `missed partial second document at character ${secondCut}`,
    );
  }
});

test("a document-looking continuation is not split while the first document is open", () => {
  assert.equal(
    fragmentStartsNewToolCall('{"text":"prefix', '{\\"nested\\":true}"}'),
    false,
  );
});

test("a name-only opening does not rename the completed call before it", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"a":1}' },
    { index: 0, name: "beta", arguments: "" },
    { index: 0, arguments: '{"b":2}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["alpha", '{"a":1}'],
      ["beta", '{"b":2}'],
    ],
  );
});

test("whitespace arguments still leave a name-only opening separate", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: "{}" },
    { index: 0, name: "beta", arguments: " " },
    { index: 0, arguments: '{"b":2}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["alpha", "{}"],
      ["beta", ' {"b":2}'],
    ],
  );
});

test("one fragment can close a call and open the next call", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"a":' },
    { index: 0, name: "beta", arguments: '1}{"b":2}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["alpha", '{"a":1}'],
      ["beta", '{"b":2}'],
    ],
  );
});

test("one fresh fragment can contain several complete calls", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"a":1}{"b":2}[]' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    ['{"a":1}', '{"b":2}', "[]"],
  );
  assert.equal(new Set(parts.map((part) => part.toolCallId)).size, 3);
});

test("a fresh multi-document fragment keeps its stable id on the first call", () => {
  const parts = accumulate([
    {
      id: "call-a",
      index: 0,
      name: "alpha",
      arguments: '{"a":1}{"b":2}',
    },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-a", '{"a":1}'],
      ["tool_call_0", '{"b":2}'],
    ],
  );
});

test("delayed ids claim same-index calls in announcement order", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"a":1}' },
    { index: 0, name: "beta", arguments: '{"b":2}' },
    { id: "call-a", index: 0, arguments: "" },
    { id: "call-b", index: 0, arguments: "" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.toolName, part.argsText]),
    [
      ["call-a", "alpha", '{"a":1}'],
      ["call-b", "beta", '{"b":2}'],
    ],
  );
});

test("minted ids never reuse an adopted or stable id", () => {
  const reserved = new Set(["tool_call_0", "tool_call_2"]);
  const parts = [{ toolCallId: "stable" }, { toolCallId: "tool_call_1" }];

  assert.equal(mintStreamedToolCallId(parts, 0, reserved), "tool_call_3");
});

test("a delayed id carrying an exact snapshot does not duplicate the call", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"a":1}' },
    { id: "call-a", index: 0, name: "alpha", arguments: '{"a":1}' },
    { index: 0, name: "beta", arguments: '{"b":2}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-a", '{"a":1}'],
      ["tool_call_0", '{"b":2}'],
    ],
  );
});

test("an existing stable id ignores a repeated cumulative snapshot", () => {
  const parts = accumulate([
    { id: "call-a", index: 0, name: "alpha", arguments: '{"a":1}' },
    { id: "call-a", index: 0, name: "alpha", arguments: '{"a":1}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [["call-a", '{"a":1}']],
  );
  assert.equal(isRepeatedJsonSnapshot('{"a":', '{"a":'), false);
});

test("a delayed id adopts a semantically equal JSON snapshot", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{ "a": 1, "nested": {"x": 2} }' },
    {
      id: "call-a",
      index: 0,
      name: "alpha",
      arguments: '{"nested":{"x":2},"a":1}',
    },
    { index: 0, name: "beta", arguments: '{"b":2}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-a", '{ "a": 1, "nested": {"x": 2} }'],
      ["tool_call_0", '{"b":2}'],
    ],
  );
});

test("JSON snapshot matching does not round unsafe integers", () => {
  assert.equal(
    sameJsonDocument(
      '{"id":9007199254740992}',
      '{"id":9007199254740993}',
    ),
    false,
  );
  assert.equal(
    sameJsonDocument(
      ' {"id":9007199254740993} ',
      '{"id":9007199254740993}',
    ),
    true,
  );
  assert.equal(
    sameJsonDocument(
      '{"id":"9007199254740993","ok":true}',
      '{"ok":true,"id":"9007199254740993"}',
    ),
    true,
  );
});

test("a stable id collision mints a distinct stream id", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"a":1}' },
    { id: "tool_call_0", index: 1, name: "beta", arguments: '{"b":2}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.toolCallId),
    ["tool_call_0", "tool_call_1"],
  );

  const backendIds = new Map([["tool_call_0", "tool_call_0"]]);
  bindStreamedToolCallBackendIds(
    backendIds,
    "tool_call_0",
    "tool_call_1",
  );
  assert.deepEqual([...backendIds], [
    ["tool_call_0", "tool_call_0"],
    ["tool_call_1", "tool_call_1"],
  ]);
});

test("decoded object arguments preserve their JSON payload", () => {
  assert.equal(
    streamedToolCallArguments({ query: "雪", nested: [1, { ok: true }] }),
    '{"query":"雪","nested":[1,{"ok":true}]}',
  );
});

test("an incomplete split tail is not persisted as a call", () => {
  const parts = accumulate([
    { id: "call-a", index: 0, name: "alpha", arguments: '{"a":1}{"b":' },
  ]);

  assert.deepEqual(parts.map((part) => part.argsText), ['{"a":1}']);
});

test("a delayed id does not make an incomplete split tail persist", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"a":1}{"b":' },
    { id: "call-b", index: 0, name: "alpha", arguments: "" },
  ]);

  assert.deepEqual(parts.map((part) => part.argsText), ['{"a":1}']);
});

test("fragmented names still match a delayed stable id", () => {
  const parts = accumulate([
    { index: 0, name: "web", arguments: '{"query":"' },
    { index: 0, name: "_search", arguments: 'value"}' },
    {
      index: 0,
      id: "call-search",
      name: "web_search",
      arguments: '{"query":"value"}',
    },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.toolName, part.argsText]),
    [["call-search", "web_search", '{"query":"value"}']],
  );
});

test("delayed adoption retargets a displaced provisional id", () => {
  const parts = accumulate([
    { index: 0, name: "alpha", arguments: '{"query":"first"}' },
    { index: 0, name: "beta", arguments: '{"query":"second"}' },
    {
      index: 0,
      id: "tool_call_1",
      name: "alpha",
      arguments: '{"query":"first"}',
    },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["tool_call_1", '{"query":"first"}'],
      ["tool_call_0", '{"query":"second"}'],
    ],
  );
});

test("a late name attaches to its argument-only call", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"query":"late-name"}' },
    { index: 0, name: "web_search", arguments: "" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [["web_search", '{"query":"late-name"}']],
  );
});

test("delayed ids ignore completed calls from earlier rounds", () => {
  const parts: StreamedToolCallPart[] = [
    {
      toolCallId: "tool_call_0",
      toolName: "alpha",
      argsText: '{"query":"old"}',
      result: "done",
      _delta_index: 0,
    },
    {
      toolCallId: "tool_call_1",
      toolName: "alpha",
      argsText: '{"query":"new"}',
      _delta_index: 0,
    },
  ];

  assert.equal(findDelayedStableToolCallPartIndex(parts, 0, "", ""), 1);
});
