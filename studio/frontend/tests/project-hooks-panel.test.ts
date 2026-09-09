// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";

import { projectHookHandlerPresentation } from "../src/features/chat/components/project-hook-handler-presentation.ts";

const panel = await readFile(
  new URL(
    "../src/features/chat/components/project-hooks-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);
const controls = await readFile(
  new URL(
    "../src/features/chat/components/project-hooks-landing-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);
const handlerPresentation = await readFile(
  new URL(
    "../src/features/chat/components/project-hook-handler-presentation.ts",
    import.meta.url,
  ),
  "utf8",
);

test("the hooks landing panel binds review to the saved project", () => {
  assert.match(
    controls,
    /import \{ ProjectHooksPanel \} from "\.\/project-hooks-panel";/,
  );
  assert.match(controls, /<ProjectHooksPanel[\s\S]*project=\{project\}/);
});

test("trust confirmation captures and revalidates the complete identity", () => {
  const refreshStart = panel.indexOf("const refresh = useCallback");
  const effectStart = panel.indexOf("useEffect(() =>", refreshStart);
  const refreshSource = panel.slice(refreshStart, effectStart);
  assert.ok(refreshStart >= 0 && effectStart > refreshStart);
  assert.match(
    refreshSource,
    /Promise\.allSettled\(\[[\s\S]*getProjectHooks\(project\.id\),[\s\S]*getProjectHooksTrustState\(project\.id\)/,
  );
  assert.doesNotMatch(refreshSource, /trustProjectHooks\(/);

  assert.match(panel, /useState<ProjectHooksTrustIdentity \| null>\(null\)/);
  assert.match(
    panel,
    /function trustIdentityFor[\s\S]*return \{[\s\S]*projectId,[\s\S]*workspaceRevision,[\s\S]*contentHash: hooks\.contentHash,[\s\S]*revision: hooks\.revision/,
  );
  assert.match(panel, /const workspaceRevision = hooks\?\.workspaceRevision/);
  assert.match(
    panel,
    /const identity = trustIdentityFor\(project\.id, hooks\)/,
  );
  assert.match(panel, /setTrustConfirmation\(identity\)/);
  assert.match(
    panel,
    /projectHooksTrustIdentityMatches\(identity, \{[\s\S]*projectId: project\.id,[\s\S]*hooks/,
  );
  assert.match(
    panel,
    /trustProjectHooks\(identity\.projectId, \{[\s\S]*workspaceRevision: identity\.workspaceRevision,[\s\S]*contentHash: identity\.contentHash,[\s\S]*revision: identity\.revision/,
  );
  assert.match(
    panel,
    /Project hooks changed\. Refresh and review the current file\./,
  );
  assert.match(
    panel,
    /<AlertDialogTitle>Trust project hooks\?<\/AlertDialogTitle>/,
  );
  assert.match(panel, /value=\{trustConfirmation\.contentHash\}/);
  assert.match(
    panel,
    /Workspace revision \{trustConfirmation\.workspaceRevision\}/,
  );
});

test("panel shows the full hash and isolates visibly escaped project text", () => {
  assert.doesNotMatch(panel, /shortProjectHooksHash/);
  assert.match(panel, /value=\{hooks\.contentHash\}/);
  assert.match(panel, /value=\{storedTrust\.contentHash\}/);
  assert.match(panel, /visibleProjectHookText\(value \?\? null\)/);
  assert.match(panel, /<pre[\s\S]*dir="ltr"/);
  assert.match(panel, /style=\{\{ unicodeBidi: "isolate" \}\}/);
  for (const expression of [
    "hooks.sourcePath",
    "hooks.description",
    "group.matcher",
    "handler.command",
    "handler.commandWindows",
    "handler.statusMessage",
    "handler.id",
  ]) {
    assert.match(
      panel,
      new RegExp(`value=\\{${expression.replace(".", "\\.")}\\}`),
    );
  }
});

test("refresh and mutation clear stale snapshots and pending confirmation", () => {
  const refreshStart = panel.indexOf("const refresh = useCallback");
  const effectStart = panel.indexOf("useEffect(() =>", refreshStart);
  const refreshSource = panel.slice(refreshStart, effectStart);
  assert.match(
    refreshSource,
    /setTrustConfirmation\(null\);[\s\S]*setMutation\(null\);[\s\S]*setHooks\(null\);[\s\S]*setLoading\(true\)/,
  );

  const mutationStart = panel.indexOf("async function applyMutation");
  const trustStart = panel.indexOf("function requestTrust", mutationStart);
  const mutationSource = panel.slice(mutationStart, trustStart);
  assert.match(
    mutationSource,
    /setTrustConfirmation\(null\);[\s\S]*setHooks\(null\);[\s\S]*setMutation\(key\);[\s\S]*setLoading\(true\)/,
  );
  assert.match(mutationSource, /setLoading\(false\)/);
});

test("refresh rejects a hooks snapshot raced by a newer trust revision", () => {
  const refreshStart = panel.indexOf("const refresh = useCallback");
  const effectStart = panel.indexOf("useEffect(() =>", refreshStart);
  const refreshSource = panel.slice(refreshStart, effectStart);
  assert.match(
    refreshSource,
    /trustStateResult\.status === "fulfilled"[\s\S]*trustStateResult\.value\.revision !== hooksResult\.value\.revision/,
  );
  assert.match(
    refreshSource,
    /setHooks\(null\);[\s\S]*setTrustState\(trustStateResult\.value\);[\s\S]*Project hook trust changed during refresh\. Refresh again\./,
  );
  assert.match(
    refreshSource,
    /else \{[\s\S]*setHooks\(hooksResult\.value\);[\s\S]*setTrustState\(trustStateFromHooks\(hooksResult\.value\)\)/,
  );
});

test("loading disables refresh, trust, revoke, and handler actions", () => {
  assert.match(panel, /disabled=\{loading \|\| busy\}/g);
  assert.match(
    panel,
    /disabled=\{loading \|\| busy \|\| !hooks\.contentHash\}/,
  );
  assert.match(
    panel,
    /disabled=\{loading \|\| busy \|\| trustConfirmation === null\}/,
  );
  assert.match(panel, /disabled=\{loading \|\| busy\}/);
  assert.match(panel, /disabled=\{!trusted \|\| disabled\}/);
  assert.match(panel, /disabled=\{loading \|\| busy\}/);
});

test("unavailable workspace fallback surfaces stored trust and can revoke it", () => {
  assert.match(
    panel,
    /return hooks === null[\s\S]*projectAvailable !== false[\s\S]*hooks\.workspaceAvailable !== false/,
  );
  assert.match(
    panel,
    /currentWorkspaceAvailable\([\s\S]*project\.workspaceAvailable,[\s\S]*hooks/,
  );
  assert.match(panel, /storedTrust=\{storedTrust\}/);
  assert.match(
    panel,
    /Stored trust is inactive while the workspace is unavailable\./,
  );
  assert.match(panel, /Stored SHA-256/);
  assert.match(
    panel,
    /const revision = hooks\?\.trusted \? hooks\.revision : storedTrust\?\.revision/,
  );
  assert.match(panel, /revokeProjectHooks\(project\.id, \{ revision \}\)/);
  assert.match(
    panel,
    /Reconnect the project folder to inspect the current hook file/,
  );
});

test("stored trust remains revocable across absent, changed, error, and unavailable states", () => {
  assert.match(panel, /getProjectHooksTrustState/);
  assert.match(
    panel,
    /useState<ProjectHooksTrustState \| null>\([\s\S]*null,[\s\S]*\)/,
  );
  assert.match(
    panel,
    /The current hook file could not be inspected\. Stored trust is inactive until the exact file and workspace can be verified\./,
  );
  assert.match(
    panel,
    /Stored trust is inactive while the project hook file is absent\./,
  );
  assert.match(
    panel,
    /Stored trust does not match the current hook file and workspace, so it is inactive\./,
  );
  assert.match(
    panel,
    /Stored trust is inactive while the workspace is unavailable\./,
  );
  assert.match(
    panel,
    /function StoredTrustNotice[\s\S]*onRevokeTrust\(\)\.catch/,
  );
});

test("handler UI separates saved preference from effective active state", () => {
  assert.match(panel, /checked=\{handler\.enabled\}/);
  assert.match(
    panel,
    /projectHookHandlerPresentation\(\{[\s\S]*trusted,[\s\S]*enabled: handler\.enabled,[\s\S]*active: handler\.active,[\s\S]*background: handler\.async/,
  );
  assert.match(
    panel,
    /setProjectHookHandlerEnabled\(projectId, handler\.id, \{[\s\S]*workspaceRevision,[\s\S]*contentHash,[\s\S]*revision,[\s\S]*enabled/,
  );
  assert.match(handlerPresentation, /background, inactive in this version/);
  assert.match(
    handlerPresentation,
    /foreground, may control supported pre-events/,
  );
  assert.match(handlerPresentation, /trusted && enabled && active/);
});

test("copy limits execution to reviewed synchronous tool hooks", () => {
  assert.match(panel, /Synchronous PreToolUse and PostToolUse hooks run around project/);
  assert.match(panel, /Other lifecycle and background hooks are inactive/);
  assert.match(panel, /Hooks cannot grant permissions or rewrite tool arguments/);
  assert.match(panel, /hooks\?\.execution/);
});

test("every async state update is guarded and cleanup retires the panel", () => {
  assert.match(
    panel,
    /const requestGuard = useRef\(new ProjectHooksRequestGuard\(\)\)/,
  );
  assert.match(panel, /const guard = requestGuard\.current/);
  assert.match(panel, /guard\.activate\(\)/);
  assert.match(panel, /guard\.retire\(\)/);
  assert.match(
    panel,
    /const requestRevision = requestGuard\.current\.begin\(\)/g,
  );
  assert.match(
    panel,
    /if \(requestGuard\.current\.accepts\(requestRevision\)\) \{\s*setHooks\(next\)/g,
  );
  assert.match(
    panel,
    /if \(requestGuard\.current\.accepts\(requestRevision\)\) \{[\s\S]*setTrustState\(fallbackTrustState\);[\s\S]*setError\(errorMessage\(nextError\)\)/,
  );
});

test("mounted browser renders trusted enabled active foreground and background states", async (t) => {
  const require = createRequire(import.meta.url);
  let createElement: typeof import("react").createElement;
  let renderToStaticMarkup: typeof import("react-dom/server").renderToStaticMarkup;
  try {
    ({ createElement } = require("react") as typeof import("react"));
    ({ renderToStaticMarkup } = require("react-dom/server") as typeof import("react-dom/server"));
  } catch {
    t.skip("mounted frontend dependencies are unavailable on this host");
    return;
  }
  const render = (
    props: Parameters<typeof projectHookHandlerPresentation>[0],
  ) => {
    const presentation = projectHookHandlerPresentation(props);
    return renderToStaticMarkup(
      createElement(
        "div",
        { "data-eligible": presentation.eligible },
        createElement("span", { className: "state" }, presentation.state),
        createElement(
          "span",
          { className: "execution" },
          presentation.execution,
        ),
      ),
    );
  };

  const foreground = render({
    trusted: true,
    enabled: true,
    active: true,
    background: false,
  });
  assert.match(foreground, /data-eligible="true"/);
  assert.match(foreground, /Enabled preference, trusted and eligible/);
  assert.match(foreground, /foreground, may control supported pre-events/);

  const background = render({
    trusted: true,
    enabled: true,
    active: true,
    background: true,
  });
  assert.match(background, /data-eligible="true"/);
  assert.match(background, /background, inactive in this version/);

  for (const state of [
    { trusted: false, enabled: true, active: true, background: false },
    { trusted: true, enabled: false, active: true, background: false },
    { trusted: true, enabled: true, active: false, background: true },
  ]) {
    const inactive = render(state);
    assert.match(inactive, /data-eligible="false"/);
    assert.match(inactive, /not eligible/);
    assert.doesNotMatch(inactive, /trusted and eligible/);
  }
});
