// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { toolArgText } from "../src/components/assistant-ui/tool-arg-text.ts";

const CARDS_DIR = "../src/components/assistant-ui/";

const sourceFile = (relative: string): ts.SourceFile => {
  const path = fileURLToPath(new URL(relative, import.meta.url));
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
};

test("toolArgText renders whatever the model sent as text", () => {
  assert.equal(toolArgText(42), "42");
  assert.equal(toolArgText(0), "0");
  assert.equal(toolArgText(true), "true");
  assert.equal(toolArgText(["ls", "-la"]), '["ls","-la"]');
  assert.equal(toolArgText({ cmd: "ls" }), '{"cmd":"ls"}');
  assert.equal(toolArgText("print(1)"), "print(1)");
});

// `{"toString":null}` shadows Object.prototype's toString, so coercing it throws.
test("toolArgText survives an object that cannot be coerced", () => {
  const hostile = JSON.parse('{"code":{"toString":null}}').code;
  assert.equal(toolArgText(hostile), '{"toString":null}');
  assert.equal(
    toolArgText(JSON.parse('[{"toString":null}]')),
    '[{"toString":null}]',
  );
  assert.equal(
    toolArgText(JSON.parse('{"valueOf":null,"toString":null}')),
    '{"valueOf":null,"toString":null}',
  );
});

// Only JSON is capped; capping a string would truncate legitimate `code`.
test("toolArgText caps a serialised object but never a string", () => {
  const long = "x".repeat(500_000);
  assert.equal(toolArgText(long), long);

  // Wide rather than deep: engine recursion limits differ.
  const wide = JSON.parse(`[${new Array(200_000).fill(0).join(",")}]`);
  const out = toolArgText(wide);
  assert.ok(out.length <= 100_001, `serialised object was ${out.length} chars`);
  assert.ok(out.endsWith("…"), "a truncated value says so");
  assert.ok(out.startsWith("[0,0,0,"), "the head of the value still shows");

  assert.equal(toolArgText({ cmd: "ls" }), '{"cmd":"ls"}');
});

// Cards branch on the empty string to show their writing state.
test("toolArgText maps a missing argument to the empty string", () => {
  assert.equal(toolArgText(undefined), "");
  assert.equal(toolArgText(null), "");
});

test("a coerced argument survives the calls the cards make on it", () => {
  assert.equal(toolArgText(42).split("\n")[0], "42");
  assert.equal(toolArgText(42).slice(0, 60), "42");
  assert.equal(toolArgText(42).trim(), "42");
});

test("toolArgText never throws", () => {
  const shapes: unknown[] = [
    undefined,
    null,
    0,
    -1.5,
    true,
    "",
    "text",
    [],
    {},
    [1, [2, [3]]],
    { nested: { deep: [1, 2] } },
    JSON.parse('{"toString":null}'),
    JSON.parse('[{"toString":null}]'),
    JSON.parse('{"valueOf":null,"toString":null}'),
    Object.create(null),
  ];
  for (const shape of shapes) {
    assert.doesNotThrow(() => toolArgText(shape));
    assert.equal(typeof toolArgText(shape), "string");
  }
});

/** Scoped to the component: tool-ui-web-search.tsx declares a module-level `url` too. */
function readConsts(
  relative: string,
  component: string,
  name: string,
): string[] {
  const source = sourceFile(relative);
  let body: ts.Node | undefined;
  const findComponent = (node: ts.Node): void => {
    if (ts.isVariableDeclaration(node) && node.name.getText() === component) {
      body = node.initializer;
    }
    if (ts.isFunctionDeclaration(node) && node.name?.getText() === component) {
      body = node.body;
    }
    node.forEachChild(findComponent);
  };
  source.forEachChild(findComponent);
  assert.ok(body, `${relative} does not declare ${component}`);

  const found: string[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText() === name &&
      node.initializer
    ) {
      found.push(node.initializer.getText());
    }
    node.forEachChild(visit);
  };
  body.forEachChild(visit);
  return found;
}

const COERCED: ReadonlyArray<
  readonly [file: string, component: string, props: readonly string[]]
> = [
  ["tool-ui-python.tsx", "PythonToolUIImpl", ["code"]],
  ["tool-ui-terminal.tsx", "TerminalToolUIImpl", ["command"]],
  ["tool-ui-knowledge-base.tsx", "KnowledgeBaseToolUIImpl", ["query"]],
  ["tool-ui-read-skill.tsx", "ReadSkillToolUIImpl", ["name", "resource"]],
  ["tool-ui-web-search.tsx", "WebSearchToolUIImpl", ["query", "url"]],
  [
    "tool-ui-code-execution.tsx",
    "CodeExecutionToolUIImpl",
    ["command", "path"],
  ],
  [
    "tool-ui-image-generation.tsx",
    "ImageGenerationToolUIImpl",
    // These come off the RESULT, so they are provider JSON and can be non-strings too.
    ["prompt", "resultPrompt", "size", "quality", "mime"],
  ],
  ["tool-fallback.tsx", "ToolFallbackTrigger", ["name"]],
];

const COERCION_CALL = /^toolArgText\(/;

test("every card reads its text arguments through toolArgText", () => {
  for (const [file, component, props] of COERCED) {
    for (const prop of props) {
      const [initializer, ...rest] = readConsts(
        CARDS_DIR + file,
        component,
        prop,
      );
      assert.ok(initializer, `${file} does not declare ${prop}`);
      assert.equal(rest.length, 0, `${file} declares ${prop} more than once`);
      assert.match(
        initializer,
        COERCION_CALL,
        `${file} reads ${prop} without coercing it; a model that sends a number there takes all of Unsloth down`,
      );
    }
  }
});

const TYPEOF_GUARDED: ReadonlyArray<readonly [string, string]> = [
  ["tool-ui-render-html.tsx", "RenderHtmlToolUIImpl"],
];

// Adding a new card must fail here until its argument handling is decided.
test("no tool card escapes the coercion policy", () => {
  const present = readdirSync(
    fileURLToPath(new URL(CARDS_DIR, import.meta.url)),
  )
    .filter((f) => f.startsWith("tool-ui-") && f.endsWith(".tsx"))
    .sort();
  const accounted = [
    ...COERCED.map(([file]) => file),
    ...TYPEOF_GUARDED.map(([file]) => file),
  ]
    // COERCED also covers tool-fallback.tsx, the card for every unknown tool.
    .filter((file) => file.startsWith("tool-ui-"))
    .sort();
  assert.deepEqual(
    present,
    accounted,
    "a tool card is not listed in COERCED or TYPEOF_GUARDED",
  );
});

test("the image card survives a result whose fields are not strings", () => {
  const file = `${CARDS_DIR}tool-ui-image-generation.tsx`;
  const raw = readFileSync(
    fileURLToPath(new URL(file, import.meta.url)),
    "utf8",
  );
  const parser = raw.match(/const parseImageSize = \([\s\S]*?\n\};/);
  assert.ok(parser, "parseImageSize moved");

  const one = (name: string): string => {
    const [initializer, ...rest] = readConsts(
      file,
      "ImageGenerationToolUIImpl",
      name,
    );
    assert.ok(initializer, `${file} does not declare ${name}`);
    assert.equal(rest.length, 0, `${file} declares ${name} more than once`);
    return `const ${name} = ${initializer};`;
  };
  const body = [
    parser[0],
    "const imageResult = __result;",
    one("size"),
    one("quality"),
    one("mime"),
    one("imageDimensions"),
    one("imageSrc"),
    one("imageMetadata"),
    "return { imageDimensions, imageSrc, imageMetadata };",
  ].join("\n");
  const render = new Function(
    "__result",
    "toolArgText",
    ts.transpileModule(body, {
      compilerOptions: { target: ts.ScriptTarget.ES2022 },
    }).outputText,
  ) as (
    result: Record<string, unknown>,
    coerce: typeof toolArgText,
  ) => Record<string, unknown>;
  const run = (result: Record<string, unknown>) =>
    render({ image_b64: "AAA", ...result }, toolArgText);

  assert.deepEqual(run({ size: 1024, quality: "hd" }), {
    imageDimensions: null,
    imageSrc: "data:image/png;base64,AAA",
    imageMetadata: "1024 · hd · image/png",
  });
  const hostile = JSON.parse('{"toString":null}');
  assert.doesNotThrow(() => run({ size: hostile }));
  assert.doesNotThrow(() => run({ quality: hostile }));
  assert.doesNotThrow(() => run({ image_mime: hostile }));
  assert.doesNotThrow(() => run({ image_mime: 42 }));
  assert.deepEqual(
    run({ size: "1024x768", quality: "hd", image_mime: "image/webp" }),
    {
      imageDimensions: { width: 1024, height: 768 },
      imageSrc: "data:image/webp;base64,AAA",
      imageMetadata: "1024x768 · hd · image/webp",
    },
  );
});

// A non-string tool name matches nothing in thread.tsx's by_name map, so it lands here.
test("the fallback card survives a tool name that is not a string", () => {
  const source = readFileSync(
    fileURLToPath(new URL(`${CARDS_DIR}tool-fallback.tsx`, import.meta.url)),
    "utf8",
  );
  assert.match(
    source,
    /const name = toolArgText\(toolName\);/,
    "tool-fallback.tsx reads toolName without coercing it",
  );
  assert.match(
    source,
    /formatMcpToolName\(name, mcpServer, mcpTool\) \?\? name/,
    "tool-fallback.tsx passes the raw toolName to formatMcpToolName",
  );
  assert.equal(toolArgText(123).startsWith("mcp__"), false);
});

const TYPEOF_GUARD = /typeof parsedArgs\.\w+ === "string"/;

test("the typeof-guarded card still guards", () => {
  for (const [file, component] of TYPEOF_GUARDED) {
    for (const prop of ["code", "title"]) {
      const [initializer] = readConsts(CARDS_DIR + file, component, prop);
      assert.ok(initializer, `${file} does not declare ${prop}`);
      assert.match(
        initializer,
        TYPEOF_GUARD,
        `${file} reads ${prop} without a type guard`,
      );
    }
  }
});
