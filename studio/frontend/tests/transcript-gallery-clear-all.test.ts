// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";

type Element = {
  type: unknown;
  props: Record<string, unknown>;
  children: unknown[];
};

function galleryRenderer(scope: Record<string, unknown>) {
  const source = readSrc("features/audio/transcript-gallery.tsx");
  const tree = ts.createSourceFile(
    "transcript-gallery.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const component = tree.statements.find(
    (node): node is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(node) && node.name?.text === "TranscriptGallery",
  );
  assert.ok(component?.body);
  const params = component.parameters
    .map((param) => param.getText(tree))
    .join(", ");
  const { outputText } = ts.transpileModule(
    `(function (${params}) ${component.body.getText(tree)})`,
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        jsx: ts.JsxEmit.React,
        jsxFactory: "h",
      },
    },
  );
  const stubs = new Map<string, unknown>();
  const proxy = new Proxy(scope, {
    has: (target, key) =>
      typeof key === "string" && (key in target || !(key in globalThis)),
    get: (target, key) => {
      if (typeof key !== "string") return undefined;
      if (key in target) return target[key];
      if (!stubs.has(key))
        stubs.set(
          key,
          Object.defineProperty(() => null, "name", { value: key }),
        );
      return stubs.get(key);
    },
  });
  const gallery = new Function(
    "scope",
    `with (scope) return ${outputText.replace('"use strict";', "").trim().replace(/;$/, "")};`,
  )(proxy);
  return () => gallery(scope.props) as Element;
}

function find(node: unknown, match: (element: Element) => boolean): Element[] {
  if (Array.isArray(node)) return node.flatMap((child) => find(child, match));
  if (!node || typeof node !== "object" || !("children" in node)) return [];
  const element = node as Element;
  return [
    ...(match(element) ? [element] : []),
    ...find(element.children, match),
  ];
}

const text = (element: Element): string =>
  element.children
    .map((child) =>
      typeof child === "string"
        ? child
        : child && typeof child === "object" && "children" in child
          ? text(child as Element)
          : "",
    )
    .join("");

test("Clear all deletes the history only after the in-app confirmation", async () => {
  const state: unknown[] = [];
  const refs: { current: unknown }[] = [];
  let stateIndex = 0;
  let refIndex = 0;
  const deletes: unknown[][] = [];
  let nativeConfirms = 0;
  const previousWindow = (globalThis as { window?: unknown }).window;
  (globalThis as { window?: unknown }).window = {
    confirm: async () => {
      nativeConfirms += 1;
      return false;
    },
  };
  try {
    const render = galleryRenderer({
      props: {
        active: true,
        currentId: null,
        latest: null,
        autoSelect: false,
        onSelect: () => {},
        onDelete: () => {},
        canSelect: () => true,
        sendHandlersFor: () => ({}),
      },
      h: (
        type: unknown,
        props: Record<string, unknown> | null,
        ...children: unknown[]
      ) => ({
        type,
        props: props ?? {},
        children: children.flat(),
      }),
      useState: (initial: unknown) => {
        const index = stateIndex++;
        if (!(index in state)) state[index] = initial;
        return [
          state[index],
          (value: unknown) => {
            state[index] =
              typeof value === "function"
                ? (value as (v: unknown) => unknown)(state[index])
                : value;
          },
        ];
      },
      useRef: (initial: unknown) => (refs[refIndex++] ??= { current: initial }),
      useCallback: (callback: unknown) => callback,
      useLayoutEffect: (effect: () => void) => effect(),
      useEffect: () => {},
      deleteTranscript: async (...args: unknown[]) => {
        deletes.push(args);
      },
      listTranscripts: async () => ({ transcripts: [], next_cursor: null }),
      toast: { error: (message: string) => assert.fail(message) },
    });
    const view = () => {
      stateIndex = 0;
      refIndex = 0;
      return render();
    };
    view();
    state[0] = [
      { id: "a", title: "First", created_at: "2026-10-27T00:00:00Z" },
    ];

    const clearAll = find(
      view(),
      (element) =>
        (element.type as { name?: string }).name === "Button" &&
        text(element) === "Clear all",
    );
    assert.equal(clearAll.length, 1);
    (clearAll[0].props.onClick as () => void)();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(
      deletes,
      [],
      "Clear all must not delete before the user confirms",
    );

    const action = find(
      view(),
      (element) =>
        (element.type as { name?: string }).name === "AlertDialogAction",
    );
    assert.equal(action.length, 1);
    assert.equal(
      find(
        view(),
        (element) => (element.type as { name?: string }).name === "AlertDialog",
      )[0].props.open,
      true,
    );
    (action[0].props.onClick as () => void)();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(deletes, [[]]);
    assert.equal(nativeConfirms, 0);
  } finally {
    (globalThis as { window?: unknown }).window = previousWindow;
  }
});
