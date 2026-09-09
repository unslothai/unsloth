// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";
registerBundlerResolver();
const { shouldOfferMinPRecovery } = await import(
  "../src/features/chat/lib/min-p-recovery.ts"
);
const { formatApiErrorBody } = await import(
  "../src/lib/format-fastapi-error.ts"
);
const { maxTokensIsTheLimit } = await import(
  "../src/features/chat/api/generation-length.ts"
);

// Execute the production parser without loading unrelated model-management imports.
const source = readSrc("features/chat/api/chat-api.ts");
const tree = ts.createSourceFile(
  "chat-api.ts",
  source,
  ts.ScriptTarget.Latest,
  true,
);
const names = new Set([
  "streamChatCompletions",
  "parseSseEvent",
  "parseErrorText",
  "StreamInterruptedError",
  "GenerationLengthError",
  "hasNonWhitespaceText",
  "classifyStructuredDeltaContent",
]);
const declarations = tree.statements.filter(
  (node) =>
    (ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node)) &&
    node.name &&
    names.has(node.name.text),
);
assert.equal(declarations.length, names.size);
const executable = ts.transpileModule(
  declarations.map((node) => node.getText(tree)).join("\n"),
  {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
    },
  },
).outputText;

async function parse(response: Response) {
  const exports: Record<
    string,
    (payload: object, signal: AbortSignal) => AsyncGenerator<unknown>
  > = {};
  new Function(
    "exports",
    "authFetch",
    "formatApiErrorBody",
    "maxTokensIsTheLimit",
    executable,
  )(exports, async () => response, formatApiErrorBody, maxTokensIsTheLimit);
  const chunks: unknown[] = [];
  for await (const chunk of exports.streamChatCompletions(
    {},
    new AbortController().signal,
  ))
    chunks.push(chunk);
  return chunks;
}

const message =
  "The min_p and logit_bias sampling parameters are not yet supported with speculative decoding";

test("production SSE parser preserves the sampling rejection before DONE", async () => {
  const response = new Response(
    `data: ${JSON.stringify({ error: { message } })}\n\ndata: [DONE]\n\n`,
  );
  await assert.rejects(parse(response), (error: Error) => {
    assert.equal(error.message, message);
    assert.equal(
      shouldOfferMinPRecovery(error.message, "vllm", {
        minP: 0.01,
        minPMode: "server-default",
      }),
      true,
    );
    return true;
  });
});

test("non-200 upstream error also reaches the recovery predicate", async () => {
  await assert.rejects(
    parse(Response.json({ error: { message } }, { status: 400 })),
    (error: Error) => {
      assert.ok(error.message.includes(message));
      assert.equal(
        shouldOfferMinPRecovery(error.message, "vllm", { minP: 0.01 }),
        true,
      );
      return true;
    },
  );
});

test("ordinary upstream errors are preserved without sampling recovery", async () => {
  await assert.rejects(
    parse(
      new Response(
        'data: {"error":{"message":"model is unavailable"}}\n\ndata: [DONE]\n\n',
      ),
    ),
    (error: Error) => {
      assert.equal(error.message, "model is unavailable");
      assert.equal(
        shouldOfferMinPRecovery(error.message, "vllm", { minP: 0.01 }),
        false,
      );
      return true;
    },
  );
});

test("a valid generated answer still completes", async () => {
  const chunk = {
    choices: [{ delta: { content: "Hello" }, finish_reason: "stop" }],
  };
  assert.deepEqual(
    await parse(
      new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`),
    ),
    [chunk],
  );
});
