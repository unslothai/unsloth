// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { type Plugin, createServer } from "vite";

import type {
  ProjectVerificationCheck,
  ProjectVerificationRun,
  ProjectVerificationRunSummary,
} from "../src/features/chat/api/project-verification-api.ts";

const FRONTEND_ROOT = fileURLToPath(new URL("..", import.meta.url));
const VIRTUAL_AUTH = "\0verification-test-auth";
const VIRTUAL_FORMAT = "\0verification-test-format";
const VIRTUAL_UI = "\0verification-test-ui";
const STUBBED_UI_MODULES = new Set([
  "@/components/ui/button",
  "@/components/ui/checkbox",
  "@/components/ui/input",
  "@/components/ui/textarea",
]);

function verificationSsrStubs(): Plugin {
  return {
    name: "verification-ssr-stubs",
    enforce: "pre",
    resolveId(source) {
      if (source === "@/features/auth") {
        return VIRTUAL_AUTH;
      }
      if (source === "@/lib/format-fastapi-error") {
        return VIRTUAL_FORMAT;
      }
      if (STUBBED_UI_MODULES.has(source)) {
        return VIRTUAL_UI;
      }
      return null;
    },
    load(id) {
      if (id === VIRTUAL_AUTH) {
        return `
          export async function authFetch(input, init) {
            const handler = globalThis.__verificationTestAuthFetch;
            if (typeof handler !== "function") {
              throw new Error("Unexpected verification request");
            }
            return handler(input, init);
          }
        `;
      }
      if (id === VIRTUAL_FORMAT) {
        return "export function formatApiErrorBody() { return null; }";
      }
      if (id === VIRTUAL_UI) {
        return `
          import { createElement } from "react";
          const component = (tag) => (props) => createElement(tag, props);
          export const Button = component("button");
          export const Checkbox = component("input");
          export const Input = component("input");
          export const Textarea = component("textarea");
        `;
      }
      return null;
    },
  };
}

type VerificationAuthFetch = (
  input: string,
  init?: RequestInit,
) => Promise<Response> | Response;

class VerificationFakeNode extends EventTarget {
  readonly nodeType: number;
  readonly nodeName: string;
  ownerDocument: VerificationFakeDocument;
  parentNode: VerificationFakeNode | null = null;
  childNodes: VerificationFakeNode[] = [];
  nodeValue = "";
  private directText = "";

  constructor(
    nodeType: number,
    nodeName: string,
    ownerDocument: VerificationFakeDocument,
  ) {
    super();
    this.nodeType = nodeType;
    this.nodeName = nodeName;
    this.ownerDocument = ownerDocument;
  }

  appendChild<TNode extends VerificationFakeNode>(node: TNode): TNode {
    node.parentNode?.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }

  insertBefore<TNode extends VerificationFakeNode>(
    node: TNode,
    before: VerificationFakeNode,
  ): TNode {
    const index = this.childNodes.indexOf(before);
    if (index < 0) {
      throw new Error("Insertion target is not a child.");
    }
    node.parentNode?.removeChild(node);
    node.parentNode = this;
    this.childNodes.splice(index, 0, node);
    return node;
  }

  removeChild<TNode extends VerificationFakeNode>(node: TNode): TNode {
    const index = this.childNodes.indexOf(node);
    if (index < 0) {
      throw new Error("Removal target is not a child.");
    }
    this.childNodes.splice(index, 1);
    node.parentNode = null;
    return node;
  }

  contains(node: VerificationFakeNode | null): boolean {
    for (let current = node; current !== null; current = current.parentNode) {
      if (current === this) {
        return true;
      }
    }
    return false;
  }

  get firstChild(): VerificationFakeNode | null {
    return this.childNodes[0] ?? null;
  }

  get lastChild(): VerificationFakeNode | null {
    return this.childNodes.at(-1) ?? null;
  }

  get nextSibling(): VerificationFakeNode | null {
    if (!this.parentNode) {
      return null;
    }
    const siblings = this.parentNode.childNodes;
    return siblings[siblings.indexOf(this) + 1] ?? null;
  }

  get textContent(): string {
    if (this.nodeType === 3 || this.nodeType === 8) {
      return this.nodeValue;
    }
    return (
      this.directText +
      this.childNodes.map((child) => child.textContent).join("")
    );
  }

  set textContent(value: string) {
    this.childNodes = [];
    this.directText = String(value);
  }
}

class VerificationFakeElement extends VerificationFakeNode {
  readonly tagName: string;
  readonly localName: string;
  namespaceURI = "http://www.w3.org/1999/xhtml";
  value = "";
  checked = false;
  disabled = false;
  open = false;
  readonly style: Record<string, unknown> & {
    setProperty: (name: string, value: string) => void;
    removeProperty: (name: string) => void;
  };
  private readonly attributes = new Map<string, string>();

  constructor(tagName: string, ownerDocument: VerificationFakeDocument) {
    super(1, tagName.toUpperCase(), ownerDocument);
    this.tagName = tagName.toUpperCase();
    this.localName = tagName.toLowerCase();
    const style: Record<string, unknown> = {};
    this.style = Object.assign(style, {
      setProperty(name: string, value: string) {
        style[name] = value;
      },
      removeProperty(name: string) {
        delete style[name];
      },
    });
  }

  setAttribute(name: string, value: unknown): void {
    this.attributes.set(name, String(value));
  }

  setAttributeNS(_namespace: string, name: string, value: unknown): void {
    this.setAttribute(name, value);
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  hasAttribute(name: string): boolean {
    return this.attributes.has(name);
  }

  removeAttribute(name: string): void {
    this.attributes.delete(name);
  }

  removeAttributeNS(_namespace: string, name: string): void {
    this.removeAttribute(name);
  }

  closest(selector: string): VerificationFakeElement | null {
    if (this.localName === selector.toLowerCase()) {
      return this;
    }
    return this.parentNode instanceof VerificationFakeElement
      ? this.parentNode.closest(selector)
      : null;
  }

  get innerHTML(): string {
    return this.textContent;
  }

  set innerHTML(value: string) {
    this.textContent = value;
  }

  focus(): void {
    this.ownerDocument.activeElement = this;
  }

  blur(): void {
    if (this.ownerDocument.activeElement === this) {
      this.ownerDocument.activeElement = null;
    }
  }
}

class VerificationFakeDocument extends VerificationFakeNode {
  readonly documentElement: VerificationFakeElement;
  readonly body: VerificationFakeElement;
  defaultView: VerificationFakeWindow | null = null;
  activeElement: VerificationFakeElement | null = null;

  constructor() {
    super(9, "#document", null as unknown as VerificationFakeDocument);
    this.ownerDocument = this;
    this.documentElement = this.createElement("html");
    this.body = this.createElement("body");
    this.documentElement.appendChild(this.body);
  }

  createElement(tagName: string): VerificationFakeElement {
    return new VerificationFakeElement(tagName, this);
  }

  createElementNS(namespace: string, tagName: string): VerificationFakeElement {
    const element = this.createElement(tagName);
    element.namespaceURI = namespace;
    return element;
  }

  createTextNode(value: string): VerificationFakeNode {
    const node = new VerificationFakeNode(3, "#text", this);
    node.nodeValue = String(value);
    return node;
  }

  createComment(value: string): VerificationFakeNode {
    const node = new VerificationFakeNode(8, "#comment", this);
    node.nodeValue = String(value);
    return node;
  }
}

class VerificationFakeWindow extends EventTarget {
  readonly document: VerificationFakeDocument;
  readonly Node = VerificationFakeNode;
  readonly Element = VerificationFakeElement;
  readonly HTMLElement = VerificationFakeElement;
  readonly HTMLIFrameElement = class {};
  readonly location = { protocol: "http:" };
  readonly navigator = { userAgent: "verification-panel-test" };
  readonly setTimeout = globalThis.setTimeout.bind(globalThis);
  readonly clearTimeout = globalThis.clearTimeout.bind(globalThis);

  constructor(document: VerificationFakeDocument) {
    super();
    this.document = document;
  }

  getComputedStyle(): Record<string, string> {
    return {};
  }

  getSelection(): null {
    return null;
  }
}

function installVerificationDom(): {
  document: VerificationFakeDocument;
  window: VerificationFakeWindow;
  restore: () => void;
} {
  const document = new VerificationFakeDocument();
  const window = new VerificationFakeWindow(document);
  document.defaultView = window;
  const storage = new Map<string, string>();
  const replacements: Record<string, unknown> = {
    window,
    document,
    navigator: window.navigator,
    getComputedStyle: window.getComputedStyle.bind(window),
    localStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, String(value)),
      removeItem: (key: string) => storage.delete(key),
      clear: () => storage.clear(),
    },
    fetch: (input: RequestInfo | URL, init?: RequestInit) => {
      const handler = (
        globalThis as typeof globalThis & {
          __verificationTestAuthFetch?: VerificationAuthFetch;
        }
      ).__verificationTestAuthFetch;
      if (!handler) {
        throw new Error("Unexpected verification request");
      }
      return handler(
        typeof input === "string" ? input : input.toString(),
        init,
      );
    },
    Node: VerificationFakeNode,
    Element: VerificationFakeElement,
    HTMLElement: VerificationFakeElement,
    HTMLIFrameElement: window.HTMLIFrameElement,
    IS_REACT_ACT_ENVIRONMENT: true,
  };
  const previous = new Map<string, PropertyDescriptor | undefined>();
  for (const [name, value] of Object.entries(replacements)) {
    previous.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, {
      configurable: true,
      writable: true,
      value,
    });
  }
  return {
    document,
    window,
    restore() {
      for (const [name, descriptor] of previous) {
        if (descriptor) {
          Object.defineProperty(globalThis, name, descriptor);
        } else {
          Reflect.deleteProperty(globalThis, name);
        }
      }
      Reflect.deleteProperty(globalThis, "__verificationTestAuthFetch");
    },
  };
}

function verificationElements(
  root: VerificationFakeNode,
  tagName?: string,
): VerificationFakeElement[] {
  const matches: VerificationFakeElement[] = [];
  const visit = (node: VerificationFakeNode) => {
    if (
      node instanceof VerificationFakeElement &&
      (tagName === undefined || node.localName === tagName)
    ) {
      matches.push(node);
    }
    for (const child of node.childNodes) {
      visit(child);
    }
  };
  visit(root);
  return matches;
}

function verificationReactProps(
  element: VerificationFakeElement,
): Record<string, unknown> {
  const key = Object.keys(element).find((candidate) =>
    candidate.startsWith("__reactProps$"),
  );
  assert.ok(key, `React props were not attached to <${element.localName}>.`);
  return (element as unknown as Record<string, unknown>)[key] as Record<
    string,
    unknown
  >;
}

function verificationResponse(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers:
      status === 204 ? undefined : { "Content-Type": "application/json" },
  });
}

const panel = await readFile(
  new URL(
    "../src/features/chat/components/project-verification-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);
const controls = await readFile(
  new URL(
    "../src/features/chat/components/project-checks-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("project checks mount verification and hook review", () => {
  assert.match(
    controls,
    /import \{ ProjectVerificationPanel \} from "\.\/project-verification-panel";/,
  );
  assert.match(controls, /<ProjectVerificationPanel[\s\S]*project=\{project\}/);

});

test("profile edits use revision and workspace CAS without gating editing on execution", () => {
  assert.match(
    panel,
    /saveProjectVerificationConfig\(project\.id, \{[\s\S]*checks: payload,[\s\S]*expectedRevision: config\.revision,[\s\S]*workspaceRevision: config\.workspaceRevision/,
  );
  assert.match(
    panel,
    /const maySave = Boolean\([\s\S]*workspaceAvailable[\s\S]*!runActive[\s\S]*validationError === null[\s\S]*dirty \|\| !config\.active/,
  );
  const saveGate = panel.slice(
    panel.indexOf("const maySave"),
    panel.indexOf("const mayRun"),
  );
  assert.doesNotMatch(saveGate, /execution\.available/);
  assert.match(
    panel,
    /status === 422[\s\S]*Verification profile rejected: \$\{nextError\.message\}/,
  );
  assert.match(
    panel,
    /verificationConflictRequiresRefresh\(nextError\)[\s\S]*await refresh\(\)[\s\S]*local draft was preserved/,
  );
  assert.match(panel, /verificationConflictRequiresRefresh\(nextError\)/);
  assert.match(
    panel,
    /const commitConfig = useCallback\([\s\S]*projectVerificationConfigCanReplace\(configRef\.current, next\)[\s\S]*configRef\.current = next;[\s\S]*setConfig\(next\)/,
  );
  const saveSuccess = panel.slice(
    panel.indexOf("const next = await saveProjectVerificationConfig"),
    panel.indexOf("} catch (nextError)", panel.indexOf("async function save")),
  );
  assert.match(
    saveSuccess,
    /if \(!commitConfig\(next\)\) \{[\s\S]*return;[\s\S]*\}[\s\S]*const nextDraft = draftChecks\(next\.checks\)/,
  );
  assert.doesNotMatch(
    saveSuccess,
    /configRef\.current = next|setConfig\(next\)/,
  );
  assert.doesNotMatch(
    panel,
    /Math\.trunc\(check\.(?:timeoutSeconds|logLimitBytes)\)/,
  );
  const publicChecks = panel.slice(
    panel.indexOf("export function projectVerificationPublicChecks"),
    panel.indexOf("function checksMatch"),
  );
  assert.match(publicChecks, /name: check\.name/);
  assert.match(publicChecks, /kind: check\.kind/);
  assert.match(publicChecks, /command: check\.command/);
  assert.doesNotMatch(publicChecks, /\.trim\(\)/);
  assert.match(panel, /const payload = publicDraft/);
});

test("run start fails closed on profile, workspace, and execution capability", () => {
  const runGate = panel.slice(
    panel.indexOf("const mayRun"),
    panel.indexOf("function updateCheck"),
  );
  assert.match(runGate, /config\?\.active/);
  assert.match(runGate, /profileFresh/);
  assert.match(runGate, /config\.execution\.available/);
  assert.match(runGate, /workspaceAvailable/);
  assert.match(runGate, /!dirty/);
  assert.match(runGate, /config\.checks\.length > 0/);
  assert.match(runGate, /!runActive/);
  assert.match(
    panel,
    /config\.execution\.reason \?\?[\s\S]*No verification command will run/,
  );
  assert.match(
    panel,
    /verificationConflictRequiresRefresh\(nextError\)\) \{[\s\S]*setProfileFresh\(false\);[\s\S]*await refresh\(\)/,
  );
});

test("an active run locks the saved profile until its pinned checks finish", () => {
  assert.match(
    panel,
    /const runActive =[\s\S]*activeRunCurrent && projectVerificationRunIsActive\(activeRun\)/,
  );
  assert.match(
    panel,
    /The saved profile is locked until this verification run finishes\./,
  );
  assert.match(
    panel,
    /disabled=\{busy \|\| runActive \|\| !workspaceAvailable\}/,
  );
});

test("project switches fail closed before the next scoped snapshot arrives", () => {
  assert.match(
    panel,
    /const configCurrent = config\?\.projectId === project\.id/,
  );
  assert.match(
    panel,
    /const activeRunCurrent = activeRun\?\.projectId === project\.id/,
  );
  assert.match(
    panel,
    /incoming\.filter\(\(run\) => run\.projectId === project\.id\)/,
  );
  assert.match(panel, /configCurrent &&[\s\S]*config\?\.active/);
  assert.match(
    panel,
    /Refresh the verification profile before starting a run\.[\s\S]*Editing stays[\s\S]*available\./,
  );
  assert.match(
    panel,
    /<ProjectVerificationPanelForProject key=\{project\.id\} project=\{project\} \/>/,
  );
  assert.match(panel, /pollError\.projectId !== projectId/);
});

test("refresh derives active state only from the accepted config revision", () => {
  const refreshBlock = panel.slice(
    panel.indexOf("const refresh = useCallback"),
    panel.indexOf(
      "useEffect(() =>",
      panel.indexOf("const refresh = useCallback"),
    ),
  );
  assert.match(
    refreshBlock,
    /currentConfig =\s*configRef\.current\?\.projectId === project\.id[\s\S]*projectVerificationConfigFromRefresh\([\s\S]*currentConfig,[\s\S]*configResult\.value,[\s\S]*refreshOwnsRunState[\s\S]*currentConfig =\s*configRef\.current\?\.projectId === project\.id/,
  );
  assert.match(refreshBlock, /currentActiveRun\(currentConfig, mergedRuns\)/);
  assert.doesNotMatch(
    refreshBlock,
    /currentActiveRun\(configResult\.value, mergedRuns\)/,
  );
  assert.match(
    refreshBlock,
    /projectVerificationRunsFromRefresh\([\s\S]*runsResult\.value,[\s\S]*refreshOwnsRunState/,
  );
  assert.match(
    refreshBlock,
    /if \(refreshOwnsRunState\) \{[\s\S]*commitActiveRun/,
  );
  assert.match(
    panel,
    /configStateGenerationRef\.current \+= 1;[\s\S]*setConfig\(next\)/,
  );
  assert.match(
    refreshBlock,
    /const configStateGeneration = configStateGenerationRef\.current[\s\S]*const refreshOwnsConfigState =[\s\S]*configStateGeneration === configStateGenerationRef\.current/,
  );
  assert.match(
    refreshBlock,
    /else if \(refreshOwnsConfigState\) \{\s*setProfileFresh\(false\)/,
  );
  assert.match(
    refreshBlock,
    /projectVerificationRefreshFailureMessage\(\s*configResult,\s*refreshOwnsConfigState/,
  );
  assert.match(
    refreshBlock,
    /projectVerificationRefreshFailureMessage\(\s*runsResult,\s*refreshOwnsRunState/,
  );
});

test("active runs recover, poll, cancel, and ignore stale component work", () => {
  assert.match(panel, /currentActiveRun\(currentConfig, mergedRuns\)/);
  assert.match(panel, /config\.activeRun/);
  assert.match(panel, /projectVerificationRunCanReplace\(previous, run\)/);
  assert.match(
    panel,
    /next\.evidenceRevision !== previous\.evidenceRevision[\s\S]*next\.evidenceRevision > previous\.evidenceRevision/,
  );
  assert.match(panel, /runStateGenerationRef\.current \+= 1/);
  assert.match(
    panel,
    /const runStateGeneration = runStateGenerationRef\.current[\s\S]*projectVerificationHistoryCanReplace\([\s\S]*runStateGeneration,[\s\S]*runStateGenerationRef\.current/,
  );
  assert.match(
    panel,
    /else if \(refreshOwnsRunState\) \{[\s\S]*setPollError\(null\)/,
  );
  assert.match(panel, /activeRun\.cancelRequested === true/);
  assert.match(panel, /const backoff = new ProjectVerificationPollBackoff\(\)/);
  assert.match(panel, /schedule\(backoff\.successDelay\(\)\)/);
  assert.match(
    panel,
    /afterEvidenceRevision = projectVerificationAcceptedPollMarker\([\s\S]*afterEvidenceRevision,[\s\S]*accepted\.run/,
  );
  assert.match(
    panel,
    /afterEvidenceRevision = projectVerificationPollingEvidenceMarker\([\s\S]*afterEvidenceRevision,[\s\S]*nextError,[\s\S]*\);[\s\S]*schedule\(delay\)/,
  );
  assert.match(panel, /const delay = backoff\.failureDelay\(nextError\)/);
  assert.match(
    panel,
    /if \(delay === null\) \{[\s\S]*removeRun\(activeRunId\)[\s\S]*commitActiveRun\(null\)[\s\S]*clearConfigActiveRun\(activeRunId\)[\s\S]*setProfileFresh\(false\)[\s\S]*return;/,
  );
  assert.match(panel, /}\s*schedule\(delay\);/);
  assert.match(panel, /window\.setTimeout\(\(\) => void poll\(\), delay\)/);
  assert.match(
    panel,
    /getProjectVerificationRun\([\s\S]*project\.id,[\s\S]*activeRunId,[\s\S]*afterEvidenceRevision/,
  );
  assert.match(
    panel,
    /cancelProjectVerificationRun\([\s\S]*project\.id,[\s\S]*activeRun\.id/,
  );
  assert.match(panel, /requests\.retire\(\)/);
  assert.match(panel, /polls\.retire\(\)/);
  assert.match(panel, /mutations\.retire\(\)/);
  assert.match(panel, /mutationGuard\.current\.accepts\(revision\)/);
  assert.match(
    panel,
    /setPollError\(\{[\s\S]*projectId: project\.id,[\s\S]*runId: activeRunId,[\s\S]*message: errorMessage\(nextError\)/,
  );
  assert.match(
    panel,
    /visibleProjectVerificationPollError\([\s\S]*project\.id,[\s\S]*activeRunId/,
  );
  const pollCatch = panel.slice(
    panel.indexOf("} catch (nextError)", panel.indexOf("const poll = async")),
    panel.indexOf("schedule(delay);", panel.indexOf("const poll = async")),
  );
  assert.match(
    pollCatch,
    /projectVerificationPollingEvidenceRegressed\([\s\S]*removeRun\(activeRunId\)[\s\S]*clearConfigActiveRun\(activeRunId\)/,
  );
  assert.doesNotMatch(pollCatch, /requestGuard\.current\.begin\(\)/);

  const unchangedStart = panel.indexOf("if (next === null)");
  const changedStart = panel.indexOf(
    "const mergedRuns = applyRuns([next])",
    unchangedStart,
  );
  assert.ok(unchangedStart >= 0 && changedStart > unchangedStart);
  const unchangedBranch = panel.slice(unchangedStart, changedStart);
  assert.match(unchangedBranch, /schedule\(backoff\.successDelay\(\)\)/);
  assert.match(unchangedBranch, /return;/);
  assert.doesNotMatch(
    unchangedBranch,
    /applyRuns|setActiveRun|commitActiveRun|setConfig/,
  );
});

test("bounded output is plain visible text and freshness stays explicitly unverified", () => {
  assert.match(panel, /MAX_RENDERED_OUTPUT_CHARACTERS = 128_000/);
  assert.match(panel, /rawOutput\.slice\(0, MAX_RENDERED_OUTPUT_CHARACTERS\)/);
  assert.match(
    panel,
    /visibleOutput\.slice\(0, MAX_RENDERED_OUTPUT_CHARACTERS\)/,
  );
  assert.match(panel, /<pre[\s\S]*dir="ltr"/);
  assert.match(panel, /style=\{\{ unicodeBidi: "isolate" \}\}/);
  assert.doesNotMatch(panel, /dangerouslySetInnerHTML/);
  assert.match(
    panel,
    /Source freshness is unverified\.[\s\S]*do not yet certify that the project source stayed[\s\S]*unchanged\./,
  );
  assert.match(panel, /Source freshness: unverified\./);
  assert.match(
    panel,
    /id=\{`\$\{check\.draftId\}-command`\}[\s\S]*dir="ltr"[\s\S]*style=\{\{ unicodeBidi: "isolate" \}\}/,
  );
  assert.match(
    panel,
    /const showExactCommandPreview = projectVerificationCommandNeedsExactPreview\(\s*check\.command,\s*\);[\s\S]*showExactCommandPreview \?[\s\S]*Exact command preview\.[\s\S]*visibleProjectVerificationCommand\(check\.command\)/,
  );
  assert.match(panel, /visibleProjectVerificationCommand\(result\.command\)/);
});

test("panel helpers preserve exact drafts, suppress stale failures, and keep collapsed history body-free", async (t) => {
  const server = await createServer({
    root: FRONTEND_ROOT,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    resolve: { alias: { "@": path.resolve(FRONTEND_ROOT, "src") } },
    plugins: [verificationSsrStubs()],
  });
  t.after(async () => server.close());
  const loaded = await server.ssrLoadModule(
    "/src/features/chat/components/project-verification-panel.tsx",
  );
  const loadedApi = await server.ssrLoadModule(
    "/src/features/chat/api/project-verification-api.ts",
  );
  const VerificationRunCard = loaded.VerificationRunCard as (props: {
    run: ProjectVerificationRun | ProjectVerificationRunSummary;
    expanded?: boolean;
    fetchDetailsOnOpen?: boolean;
  }) => ReturnType<typeof createElement>;
  const mergeRuns = loaded.mergeRuns as (
    current: ProjectVerificationRunSummary[],
    incoming: ProjectVerificationRunSummary[],
    replaceTerminalHistory?: boolean,
  ) => ProjectVerificationRunSummary[];
  const historyCanReplace = loaded.projectVerificationHistoryCanReplace as (
    refreshGeneration: number,
    currentGeneration: number,
  ) => boolean;
  const configFromRefresh = loaded.projectVerificationConfigFromRefresh as (
    current: {
      projectId: string;
      activeRun: ProjectVerificationRunSummary | null;
    },
    incoming: {
      projectId: string;
      activeRun: ProjectVerificationRunSummary | null;
    },
    refreshOwnsRunState: boolean,
  ) => { projectId: string; activeRun: ProjectVerificationRunSummary | null };
  const runsFromRefresh = loaded.projectVerificationRunsFromRefresh as (
    incoming: ProjectVerificationRunSummary[],
    refreshOwnsRunState: boolean,
  ) => ProjectVerificationRunSummary[];
  const acceptedRun = loaded.projectVerificationAcceptedRun as (
    merged: ProjectVerificationRunSummary[],
    incoming: ProjectVerificationRunSummary,
  ) => {
    run: ProjectVerificationRunSummary;
    incomingAccepted: boolean;
  } | null;
  const acceptedPollMarker = loaded.projectVerificationAcceptedPollMarker as (
    current: number | undefined,
    accepted: ProjectVerificationRunSummary,
  ) => number;
  const activeRunAfterMerge = loaded.projectVerificationActiveRunAfterMerge as (
    current: ProjectVerificationRun | ProjectVerificationRunSummary | null,
    mergedActive: ProjectVerificationRunSummary | null,
    incoming: ProjectVerificationRun | ProjectVerificationRunSummary,
    incomingAccepted: boolean,
  ) => ProjectVerificationRun | ProjectVerificationRunSummary | null;
  const runCardKey = loaded.projectVerificationRunCardKey as (
    run: ProjectVerificationRun | ProjectVerificationRunSummary,
  ) => string;
  const detailMatches = loaded.projectVerificationDetailMatches as (
    detail: ProjectVerificationRun,
    run: ProjectVerificationRun | ProjectVerificationRunSummary,
  ) => boolean;
  const visiblePollError = loaded.visibleProjectVerificationPollError as (
    pollError: {
      projectId: string;
      runId: string;
      message: string;
      terminal: boolean;
    } | null,
    projectId: string,
    activeRunId: string | null,
  ) => string | null;
  const withoutRun = loaded.withoutProjectVerificationRun as (
    runs: ProjectVerificationRunSummary[],
    runId: string,
  ) => ProjectVerificationRunSummary[];
  const publicChecks = loaded.projectVerificationPublicChecks as (
    checks: ReadonlyArray<ProjectVerificationCheck & { draftId?: string }>,
  ) => ProjectVerificationCheck[];
  const refreshFailureMessage =
    loaded.projectVerificationRefreshFailureMessage as (
      result: PromiseSettledResult<unknown>,
      refreshOwnsState: boolean,
    ) => string | null;
  const checksError = loadedApi.projectVerificationChecksError as (
    checks: ProjectVerificationCheck[],
  ) => string | null;

  const exactDraft = {
    draftId: "exact-draft",
    name: " Tests ",
    kind: " test ",
    command: "\nprintf ok\n",
    required: true,
    timeoutSeconds: 300,
    logLimitBytes: 262_144,
  };
  assert.deepEqual(publicChecks([exactDraft]), [
    {
      name: exactDraft.name,
      kind: exactDraft.kind,
      command: exactDraft.command,
      required: exactDraft.required,
      timeoutSeconds: exactDraft.timeoutSeconds,
      logLimitBytes: exactDraft.logLimitBytes,
    },
  ]);
  assert.equal(
    checksError(publicChecks([{ ...exactDraft, command: "printf ok\u00A0" }])),
    "Check commands cannot contain non-ASCII Unicode whitespace.",
  );
  assert.equal(
    checksError(publicChecks([{ ...exactDraft, command: "\uFEFFprintf ok" }])),
    "Check names, kinds, and commands cannot contain Unicode format controls or default-ignorable code points.",
  );
  assert.equal(
    checksError(publicChecks([{ ...exactDraft, name: "\u0085" }])),
    "Every check needs a name, kind, and command.",
  );
  assert.equal(
    checksError(publicChecks([{ ...exactDraft, kind: "safe\u001B" }])),
    "Check names and kinds cannot contain C0 control characters or DEL.",
  );
  for (const field of ["name", "kind", "command"] as const) {
    assert.equal(
      checksError(publicChecks([{ ...exactDraft, [field]: "safe\uD800" }])),
      "Check names, kinds, and commands cannot contain lone UTF-16 surrogates.",
    );
  }
  const rejectedRefresh = {
    status: "rejected" as const,
    reason: new Error("late config request failed"),
  };
  assert.equal(refreshFailureMessage(rejectedRefresh, false), null);
  assert.equal(
    refreshFailureMessage(rejectedRefresh, true),
    "late config request failed",
  );
  assert.equal(
    refreshFailureMessage({ status: "fulfilled", value: null }, true),
    null,
  );

  const histories = Array.from({ length: 5 }, (_, index) => {
    const historical: ProjectVerificationRunSummary = {
      id: `history-${index}`,
      projectId: "project",
      status: "passed",
      configRevision: index + 1,
      workspaceRevision: 12,
      evidenceRevision: index + 1,
      historySequence: index + 1,
      cancelRequested: false,
      error: null,
      startedAt: 1_000 + index,
      updatedAt: 2_000 + index,
      completedAt: 3_000 + index,
      sourceFreshness: "unverified",
      evidenceStatus: "not_loaded",
    };
    return historical;
  });

  const rollbackHistory = Array.from({ length: 20 }, (_, index) => ({
    ...histories[index % histories.length],
    id: `rollback-${index}`,
    evidenceRevision: index + 1,
    historySequence: index + 1,
    startedAt: 10_000 + index,
    updatedAt: 20_000 + index,
    completedAt: 30_000 + index,
  }));
  const newestAfterRollback = {
    ...rollbackHistory[0],
    id: "newest-after-rollback",
    evidenceRevision: 21,
    historySequence: 21,
    startedAt: 1,
    updatedAt: 2,
    completedAt: 3,
  };
  const mergedRollback = mergeRuns(rollbackHistory, [newestAfterRollback]);
  assert.equal(mergedRollback.length, 20);
  assert.equal(mergedRollback[0].id, newestAfterRollback.id);
  assert.equal(
    mergedRollback.some((run) => run.id === rollbackHistory[0].id),
    false,
  );

  const trusted = {
    ...rollbackHistory[10],
    evidenceRevision: 50,
    updatedAt: 100,
  };
  const staleHigherClock = {
    ...trusted,
    evidenceRevision: 49,
    updatedAt: 99_999,
  };
  assert.equal(
    mergeRuns([trusted], [staleHigherClock])[0].evidenceRevision,
    50,
  );

  const active = {
    ...histories[0],
    id: "active",
    status: "running",
    historySequence: null,
    completedAt: null,
  };
  const authoritative = mergeRuns([active, histories[0]], [histories[1]], true);
  assert.deepEqual(
    new Set(authoritative.map((run) => run.id)),
    new Set([active.id, histories[1].id]),
  );
  assert.equal(historyCanReplace(7, 7), true);
  assert.equal(historyCanReplace(7, 8), false);
  const cachedAfterTerminalEvent = mergeRuns(rollbackHistory, [
    newestAfterRollback,
  ]);
  const afterDelayedList = mergeRuns(
    cachedAfterTerminalEvent,
    rollbackHistory,
    historyCanReplace(0, 1),
  );
  assert.equal(afterDelayedList[0].id, newestAfterRollback.id);
  assert.equal(
    visiblePollError(
      {
        projectId: "project",
        runId: "active",
        message: "retry",
        terminal: false,
      },
      "project",
      "active",
    ),
    "retry",
  );
  assert.equal(
    visiblePollError(
      {
        projectId: "project",
        runId: "active",
        message: "retry",
        terminal: false,
      },
      "project",
      null,
    ),
    null,
  );
  const afterMissingRun = withoutRun([active, histories[0]], active.id);
  const afterMissingRefresh = mergeRuns(
    afterMissingRun,
    runsFromRefresh([active, ...histories], false),
    true,
  );
  assert.equal(
    afterMissingRefresh.some((run) => run.id === active.id),
    false,
  );
  assert.equal(
    configFromRefresh(
      { projectId: "project", activeRun: null },
      { projectId: "project", activeRun: active },
      false,
    ).activeRun,
    null,
  );
  const priorEpoch = {
    ...active,
    id: "epoch-reset",
    evidenceRevision: 50,
  };
  const lowerEpoch = {
    ...priorEpoch,
    evidenceRevision: 2,
    updatedAt: priorEpoch.updatedAt - 1,
  };
  const afterEpochReset = mergeRuns(withoutRun([priorEpoch], priorEpoch.id), [
    lowerEpoch,
  ]);
  assert.equal(afterEpochReset[0].evidenceRevision, 2);
  assert.equal(
    acceptedRun(afterEpochReset, lowerEpoch)?.incomingAccepted,
    true,
  );
  assert.equal(acceptedPollMarker(undefined, afterEpochReset[0]), 2);

  const newerEvent = {
    ...active,
    id: "event-poll-race",
    evidenceRevision: 8,
    updatedAt: 50_000,
    cancelRequested: true,
  };
  const delayedPoll = {
    ...newerEvent,
    evidenceRevision: 7,
    cancelRequested: false,
  };
  const afterDelayedPoll = mergeRuns([newerEvent], [delayedPoll]);
  const delayedAcceptance = acceptedRun(afterDelayedPoll, delayedPoll);
  assert.ok(delayedAcceptance);
  assert.equal(delayedAcceptance.incomingAccepted, false);
  assert.equal(delayedAcceptance.run.evidenceRevision, 8);
  assert.equal(acceptedPollMarker(7, delayedAcceptance.run), 8);
  const exactNewerEvent: ProjectVerificationRun = {
    ...newerEvent,
    checks: [],
    results: [],
  };
  const activeAfterDelayedPoll = activeRunAfterMerge(
    exactNewerEvent,
    delayedAcceptance.run,
    delayedPoll,
    false,
  );
  assert.ok(activeAfterDelayedPoll);
  assert.strictEqual(activeAfterDelayedPoll, exactNewerEvent);
  assert.equal("results" in activeAfterDelayedPoll, true);

  const detailAtPriorEpoch: ProjectVerificationRun = {
    ...priorEpoch,
    status: "passed",
    historySequence: 50,
    completedAt: priorEpoch.updatedAt,
    checks: [],
    results: [],
  };
  assert.notEqual(runCardKey(priorEpoch), runCardKey(lowerEpoch));
  assert.notEqual(
    runCardKey(priorEpoch),
    runCardKey({ ...priorEpoch, updatedAt: priorEpoch.updatedAt - 1 }),
  );
  assert.equal(
    detailMatches(detailAtPriorEpoch, lowerEpoch),
    true,
    "the keyed child must isolate detail state when a marker regresses",
  );
  assert.equal(
    runCardKey(lowerEpoch),
    runCardKey({ ...lowerEpoch }),
    "equal markers keep the same child while marker changes force a remount",
  );
  assert.equal(
    visiblePollError(
      {
        projectId: "project",
        runId: "active",
        message: "not authorized",
        terminal: true,
      },
      "project",
      null,
    ),
    "not authorized",
  );
  assert.equal(
    visiblePollError(
      {
        projectId: "project",
        runId: "active",
        message: "not authorized",
        terminal: true,
      },
      "project",
      "next-run",
    ),
    null,
  );
  assert.equal(
    visiblePollError(
      {
        projectId: "previous-project",
        runId: "active",
        message: "stale authorization error",
        terminal: true,
      },
      "project",
      null,
    ),
    null,
  );

  const collapsed = renderToStaticMarkup(
    createElement(
      "div",
      null,
      histories.map((run) =>
        createElement(VerificationRunCard, { key: run.id, run }),
      ),
    ),
  );
  for (const run of histories) {
    assert.match(collapsed, new RegExp(`Run ${run.id}`));
    assert.equal("checks" in run, false);
    assert.equal("results" in run, false);
  }
  assert.equal(
    collapsed.match(/Details not loaded/g)?.length,
    histories.length,
  );
  assert.doesNotMatch(collapsed, /passed/);

  const expandedSummary = renderToStaticMarkup(
    createElement(VerificationRunCard, {
      run: histories[0],
      expanded: true,
    }),
  );
  assert.match(expandedSummary, /Details not loaded/);
  assert.match(expandedSummary, /Loading run details/);
  assert.doesNotMatch(expandedSummary, /passed/);

  const exactRun: ProjectVerificationRun = {
    ...histories[0],
    checks: [],
    results: [
      {
        name: "Tests",
        kind: "test",
        command: "python -m pytest",
        required: true,
        timeoutSeconds: 300,
        logLimitBytes: 262_144,
        status: "passed",
        exitCode: 0,
        output: "EXACT_EVIDENCE_BODY",
        outputBytes: 19,
        outputTruncated: false,
        startedAt: 1_100,
        completedAt: 1_200,
        durationMs: 100,
      },
    ],
  };
  const expandedExact = renderToStaticMarkup(
    createElement(VerificationRunCard, { run: exactRun, expanded: true }),
  );
  assert.match(expandedExact, /passed/);
  assert.match(expandedExact, /EXACT_EVIDENCE_BODY/);
});

test("history details fetch only after expansion and remain request guarded", () => {
  const cardStart = panel.indexOf("export function VerificationRunCard");
  const detailsStart = panel.indexOf(
    "function VerificationRunDetails",
    cardStart,
  );
  assert.ok(cardStart >= 0 && detailsStart > cardStart);
  const card = panel.slice(cardStart, detailsStart);
  assert.match(
    card,
    /<VerificationRunCardForMarker[\s\S]*key=\{projectVerificationRunCardKey\(run\)\}/,
  );
  assert.match(card, /if \(\s*!open\) \{[\s\S]*return;/);
  assert.match(card, /getProjectVerificationRun\(runProjectId, runId\)/);
  assert.match(card, /const revision = guard\.begin\(\)/);
  assert.match(card, /if \(!guard\.accepts\(revision\)\)/);
  assert.match(card, /next\.evidenceRevision < runEvidenceRevision/);
  assert.match(card, /const runProjectId = run\.projectId/);
  assert.match(card, /const runEvidenceRevision = run\.evidenceRevision/);
  assert.match(
    card,
    /\}, \[[\s\S]*runEvidenceRevision,[\s\S]*runHasDetails,[\s\S]*runId,[\s\S]*runProjectId,[\s\S]*runUpdatedAt,[\s\S]*\]\);/,
  );
  assert.doesNotMatch(card, /open, run\]\);/);
});

test("mounted panel preserves a local draft when a newer config event lands in the same turn", async (t) => {
  const dom = installVerificationDom();
  const server = await createServer({
    root: FRONTEND_ROOT,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    resolve: { alias: { "@": path.resolve(FRONTEND_ROOT, "src") } },
    plugins: [verificationSsrStubs()],
  });
  const loaded = await server.ssrLoadModule(
    "/src/features/chat/components/project-verification-panel.tsx",
  );
  const loadedApi = await server.ssrLoadModule(
    "/src/features/chat/api/project-verification-api.ts",
  );
  const ProjectVerificationPanel = loaded.ProjectVerificationPanel as (
    props: Record<string, unknown>,
  ) => ReturnType<typeof createElement>;
  const baseCheck: ProjectVerificationCheck = {
    name: "Tests",
    kind: "test",
    command: "python -m pytest",
    required: true,
    timeoutSeconds: 300,
    logLimitBytes: 262_144,
  };
  const config = {
    projectId: "project",
    workspaceAvailable: true,
    workspaceRevision: 3,
    active: true,
    activeRun: null,
    checks: [baseCheck],
    revision: 1,
    updatedAt: 100,
    sourceFreshness: "unverified" as const,
    execution: { available: true, backend: "bubblewrap", reason: null },
  };
  const handler: VerificationAuthFetch = (input) => {
    if (input.includes("?limit=")) {
      return verificationResponse({ runs: [] });
    }
    if (input.endsWith("/verification")) {
      return verificationResponse(config);
    }
    return verificationResponse({ detail: "Unexpected request" }, 500);
  };
  (
    globalThis as typeof globalThis & {
      __verificationTestAuthFetch: VerificationAuthFetch;
    }
  ).__verificationTestAuthFetch = handler;

  const container = dom.document.createElement("div");
  dom.document.body.appendChild(container);
  const root = createRoot(container as unknown as Element);
  t.after(async () => {
    await act(async () => root.unmount());
    await server.close();
    dom.restore();
  });
  await act(() => {
    root.render(
      createElement(ProjectVerificationPanel, {
        project: {
          id: "project",
          name: "Project",
          workspaceKind: "folder",
          workspaceAvailable: true,
          workspaceRevision: 3,
          archived: false,
          createdAt: 1,
          updatedAt: 1,
        },
      }),
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
  });

  const nameInput = verificationElements(container, "input").find((element) =>
    element.getAttribute("id")?.endsWith("-name"),
  );
  assert.ok(nameInput, `Mounted panel text: ${container.textContent}`);
  const localName = "Local draft must survive";
  const onChange = verificationReactProps(nameInput).onChange as (event: {
    target: { value: string };
  }) => void;
  const remoteConfig = {
    ...config,
    revision: 2,
    updatedAt: 200,
    checks: [{ ...baseCheck, name: "Remote saved name" }],
  };
  await act(async () => {
    onChange({ target: { value: localName } });
    dom.window.dispatchEvent(
      new CustomEvent(loadedApi.PROJECT_VERIFICATION_UPDATED_EVENT as string, {
        detail: { projectId: "project", config: remoteConfig },
      }),
    );
  });

  assert.equal(verificationReactProps(nameInput).value, localName);
  assert.match(
    container.textContent,
    /Save this draft before running verification/,
  );
});

test("mounted recovery keeps config evidence visible when the first detail poll fails", async (t) => {
  const dom = installVerificationDom();
  const server = await createServer({
    root: FRONTEND_ROOT,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
    resolve: { alias: { "@": path.resolve(FRONTEND_ROOT, "src") } },
    plugins: [verificationSsrStubs()],
  });
  const loaded = await server.ssrLoadModule(
    "/src/features/chat/components/project-verification-panel.tsx",
  );
  const ProjectVerificationPanel = loaded.ProjectVerificationPanel as (
    props: Record<string, unknown>,
  ) => ReturnType<typeof createElement>;
  const check: ProjectVerificationCheck = {
    name: "Tests",
    kind: "test",
    command: "python -m pytest",
    required: true,
    timeoutSeconds: 300,
    logLimitBytes: 262_144,
  };
  const activeRun: ProjectVerificationRun = {
    id: "recovered-run",
    projectId: "project",
    status: "running",
    configRevision: 4,
    workspaceRevision: 3,
    evidenceRevision: 7,
    cancelRequested: false,
    error: null,
    startedAt: 100,
    updatedAt: 200,
    completedAt: null,
    historySequence: null,
    sourceFreshness: "unverified",
    checks: [check],
    results: [
      {
        ...check,
        status: "running",
        exitCode: null,
        output: "EXACT_RECOVERED_OUTPUT",
        outputBytes: 22,
        outputTruncated: false,
        startedAt: 100,
        completedAt: null,
        durationMs: null,
      },
    ],
  };
  const config = {
    projectId: "project",
    workspaceAvailable: true,
    workspaceRevision: 3,
    active: true,
    activeRun,
    checks: [check],
    revision: 4,
    updatedAt: 99,
    sourceFreshness: "unverified" as const,
    execution: { available: true, backend: "bubblewrap", reason: null },
  };
  let detailPolls = 0;
  const handler: VerificationAuthFetch = (input) => {
    if (input.endsWith("/verification")) {
      return verificationResponse(config);
    }
    if (input.includes("?limit=")) {
      return verificationResponse({
        runs: [
          {
            id: activeRun.id,
            projectId: activeRun.projectId,
            status: activeRun.status,
            configRevision: activeRun.configRevision,
            workspaceRevision: activeRun.workspaceRevision,
            evidenceRevision: activeRun.evidenceRevision,
            cancelRequested: activeRun.cancelRequested,
            error: activeRun.error,
            startedAt: activeRun.startedAt,
            updatedAt: activeRun.updatedAt,
            completedAt: activeRun.completedAt,
            historySequence: activeRun.historySequence,
            sourceFreshness: activeRun.sourceFreshness,
          },
        ],
      });
    }
    if (input.includes(`/verifications/${activeRun.id}`)) {
      detailPolls += 1;
      return verificationResponse(
        { detail: "Transient detail poll failure" },
        503,
      );
    }
    return verificationResponse({ detail: "Unexpected request" }, 500);
  };
  (
    globalThis as typeof globalThis & {
      __verificationTestAuthFetch: VerificationAuthFetch;
    }
  ).__verificationTestAuthFetch = handler;

  const container = dom.document.createElement("div");
  dom.document.body.appendChild(container);
  const root = createRoot(container as unknown as Element);
  t.after(async () => {
    await act(async () => root.unmount());
    await server.close();
    dom.restore();
  });
  await act(() => {
    root.render(
      createElement(ProjectVerificationPanel, {
        project: {
          id: "project",
          name: "Project",
          workspaceKind: "folder",
          workspaceAvailable: true,
          workspaceRevision: 3,
          archived: false,
          createdAt: 1,
          updatedAt: 1,
        },
      }),
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 40));
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
  });

  assert.equal(detailPolls, 1);
  assert.match(container.textContent, /EXACT_RECOVERED_OUTPUT/);
  assert.match(container.textContent, /Transient detail poll failure/);
  assert.doesNotMatch(container.textContent, /Loading run details/);
});
