import assert from "node:assert/strict";
import test from "node:test";
import type { SharedRunConfigControls as Controls } from "../src/features/share-run-configs/config-controls.tsx";
import type { SharedRunConfigReview as Review } from "../src/features/share-run-configs/config-review.tsx";
import * as events from "../src/features/share-run-configs/editor-events.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const fields = await import("../src/features/share-run-configs/fields.ts");
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { modelConfigDraftKey } = await import(
  "../src/features/model-picker/model-config/model-config-draft.ts"
);
const { createRunConfigInbox } = await import(
  "../src/features/share-run-configs/inbox.ts"
);

function elements(node: unknown): StubElement[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node || typeof node !== "object" || !("props" in node)) return [];
  const element = node as StubElement;
  return [element, ...elements(element.props.children)];
}

function text(node: unknown): string {
  if (Array.isArray(node)) return node.map(text).join("");
  if (node && typeof node === "object" && "props" in node)
    return text((node as StubElement).props.children);
  return typeof node === "string" || typeof node === "number"
    ? String(node)
    : "";
}

test("only an open Share dialog protects the popover; pending imports allow focus dismissal", () => {
  const inbox = createRunConfigInbox();
  const target = {
    id: "owner/Model-GGUF",
    displayName: "Model",
    ggufVariant: "Q4_K_M",
    isGguf: true,
    apiLoadable: true,
    meta: { source: "hub" as const, isLora: false },
  };
  const key = modelConfigDraftKey(target.id, target.ggufVariant);
  inbox.submit({
    id: "pending",
    draftKey: key,
    value: { config: { nParallel: 3 } },
  });
  let sharing = false;
  const dialog = Symbol("share dialog");
  const button = Symbol("button");
  const { SharedRunConfigControls } = loadWithStubs<{
    SharedRunConfigControls: typeof Controls;
  }>(
    new URL(
      "../src/features/share-run-configs/config-controls.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState: () => [
          sharing,
          (value: boolean) => {
            sharing = value;
          },
        ],
        useSyncExternalStore: (_subscribe: unknown, get: () => unknown) =>
          get(),
        useEffect: () => undefined,
        useLayoutEffect: () => undefined,
      },
      "@/components/ui/button": { Button: button },
      "../model-picker/model-config/model-config-draft": {
        modelConfigDraftKey,
      },
      "./inbox": { runConfigInbox: inbox },
      "./editor-events": events,
      "./import-config": { scheduleRunConfigImport: () => undefined },
      "./share-dialog": { ShareRunConfigDialog: dialog },
    },
  );
  const props = {
    target,
    config: DEFAULT_PER_MODEL_CONFIG,
    ready: true,
    hydrated: true,
    canImport: true,
    disabled: false,
    onImport: () => undefined,
  };
  const render = () => elements(SharedRunConfigControls(props));
  const guarded = (tree: StubElement[]) =>
    events.keepSharedRunConfigOpen({
      querySelector: (selector: string) => {
        assert.equal(selector, `[${events.SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE}]`);
        return (
          tree.find(
            (element) =>
              element.props[events.SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE] !==
              undefined,
          ) ?? null
        );
      },
    } as unknown as ParentNode);
  const initial = render();
  assert.equal(guarded(initial), false);
  assert.equal(
    initial.some((element) => element.type === dialog),
    false,
  );
  const share = initial.find((element) => element.type === button);
  assert.ok(share);
  (share.props.onClick as () => void)();
  const opened = render();
  assert.equal(guarded(opened), true);
  const shownDialog = opened.find((element) => element.type === dialog);
  assert.ok(shownDialog);
  (shownDialog.props.onClose as () => void)();
  assert.equal(guarded(render()), false);
  assert.equal(events.keepSharedRunConfigOpen(null), false);
});

test("edit cancellation includes contained controls and excludes portaled dialog controls", (t) => {
  class NodeFake extends EventTarget {
    children = new Set<NodeFake>();
    contains(node: NodeFake) {
      return node === this || this.children.has(node);
    }
  }
  const original = Object.getOwnPropertyDescriptor(globalThis, "Node");
  Object.defineProperty(globalThis, "Node", {
    value: NodeFake,
    configurable: true,
  });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, "Node", original);
    else Reflect.deleteProperty(globalThis, "Node");
  });
  const editor = new NodeFake();
  const numericInput = new NodeFake();
  const portaledTextarea = new NodeFake();
  editor.children.add(numericInput);
  const currentTarget = editor as unknown as Node;
  assert.equal(
    events.isRunConfigEditorChange({ currentTarget, target: numericInput }),
    true,
  );
  assert.equal(
    events.isRunConfigEditorChange({ currentTarget, target: portaledTextarea }),
    false,
  );
  assert.equal(
    events.isRunConfigEditorChange({
      currentTarget,
      target: new EventTarget(),
    }),
    false,
  );
});

test("review renders field labels and full prompt/argument values as text without HTML injection", () => {
  const { SharedRunConfigReview } = loadWithStubs<{
    SharedRunConfigReview: typeof Review;
  }>(
    new URL(
      "../src/features/share-run-configs/config-review.tsx",
      import.meta.url,
    ),
    { "react/jsx-runtime": stubJsxRuntime(), "./fields": fields },
  );
  assert.equal(SharedRunConfigReview({ config: null }), null);
  const prompt = "<img src=x onerror=alert(1)>\nContinue after thinking";
  const args = ["--rope-scaling", "yarn"];
  const tree = SharedRunConfigReview({
    config: {
      reasoningBudgetMessage: prompt,
      llamaExtraArgs: args,
      maxSeqLength: null,
    },
  });
  assert.ok(text(tree).includes("Settings changed by link (3)"));
  assert.ok(text(tree).includes("Reasoning budget message"));
  assert.ok(text(tree).includes(prompt));
  assert.ok(text(tree).includes(JSON.stringify(args)));
  assert.ok(text(tree).includes("Default"));
  assert.ok(
    elements(tree).every(
      (element) => !("dangerouslySetInnerHTML" in element.props),
    ),
  );
  assert.ok(
    text(SharedRunConfigReview({ config: {} })).includes("already match"),
  );
});

test("startup intake remains mounted while navigation UI loads only for a pending link", () => {
  const inbox = createRunConfigInbox();
  const effects: (() => void | (() => void))[] = [];
  const received: string[] = [];
  const editor = Symbol("lazy link editor");
  let authChanged: (() => void) | undefined;
  let disposed = false;
  let revisions = 0;
  Object.assign(window.location, {
    href: "http://localhost/chat",
    origin: "http://localhost",
  });
  const { SharedRunConfigLinkHandler } = loadWithStubs<{
    SharedRunConfigLinkHandler: () => unknown;
  }>(
    new URL(
      "../src/features/share-run-configs/link-handler.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        lazy: () => editor,
        Suspense: Symbol("suspense"),
        useState: () => [
          0,
          () => {
            revisions += 1;
          },
        ],
        useRef: (current: unknown) => ({ current }),
        useSyncExternalStore: (_subscribe: unknown, get: () => unknown) =>
          get(),
        useEffect: (effect: () => void | (() => void)) => effects.push(effect),
      },
      "@tanstack/react-router": { useRouterState: () => ({ href: "/chat" }) },
      "./inbox": { runConfigInbox: inbox },
      "./receive-link": {
        receiveStartupRunConfigUrl: (url: string) => received.push(url),
        receiveRunConfigUrl: (url: string) => received.push(url),
        subscribeRunConfigSession: (onChange: () => void) => {
          authChanged = onChange;
          return () => {
            disposed = true;
          };
        },
      },
    },
  );
  assert.equal(SharedRunConfigLinkHandler(), null);
  const cleanup = effects[0]();
  effects[1]();
  assert.deepEqual(received, ["http://localhost/chat"]);
  inbox.submit({ id: "link", value: { config: { nParallel: 3 } } });
  const shown = elements(SharedRunConfigLinkHandler()).find(
    (element) => element.type === editor,
  );
  assert.equal(shown?.props.pending, inbox.getSnapshot());
  authChanged?.();
  assert.equal(revisions, 1);
  inbox.clear("link");
  assert.equal(SharedRunConfigLinkHandler(), null);
  cleanup?.();
  assert.equal(disposed, true);
});
