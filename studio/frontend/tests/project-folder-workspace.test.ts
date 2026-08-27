// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  OPEN_PROJECT_FOLDER_ENDPOINT,
  buildOpenProjectFolderRequest,
  buildOpenProjectFolderRequestFromToken,
} from "../src/features/chat/api/project-folder-request.ts";
import {
  createProjectFolderPickerGuard,
  createProjectWorkspace,
  projectCreateDisabled,
  projectFolderPickerDisabled,
} from "../src/features/chat/components/new-project-workspace.ts";

function source(relative: string): string {
  return readFileSync(
    fileURLToPath(new URL(`../src/${relative}`, import.meta.url)),
    "utf8",
  );
}

test("folder-project creation sends only an opaque lease and project name", () => {
  const request = buildOpenProjectFolderRequest(
    "  signed.native.lease  ",
    "  Existing repository  ",
  );
  assert.equal(request.input, OPEN_PROJECT_FOLDER_ENDPOINT);
  assert.equal(request.init.method, "POST");
  assert.deepEqual(request.init.headers, {
    "Content-Type": "application/json",
  });
  assert.deepEqual(JSON.parse(String(request.init.body)), {
    nativePathLease: "signed.native.lease",
    name: "Existing repository",
  });
  assert.ok(!String(request.init.body).includes("rootPath"));
});

test("folder-project request construction rejects missing identity", () => {
  assert.throws(
    () => buildOpenProjectFolderRequest("", "Project"),
    /native project-folder lease is required/,
  );
  assert.throws(
    () => buildOpenProjectFolderRequest("signed.lease", "   "),
    /Project name is required/,
  );
});

test("the native token is consumed at commit time for an open-project lease", async () => {
  const consumed: [string, string][] = [];
  const request = await buildOpenProjectFolderRequestFromToken(
    "  opaque-picker-token  ",
    "Repository",
    (token, operation) => {
      consumed.push([token, operation]);
      return Promise.resolve({ nativePathLease: "signed.commit.lease" });
    },
  );

  assert.deepEqual(consumed, [["opaque-picker-token", "open-project"]]);
  assert.deepEqual(JSON.parse(String(request.init.body)), {
    nativePathLease: "signed.commit.lease",
    name: "Repository",
  });
  assert.ok(!String(request.init.body).includes("opaque-picker-token"));
  await assert.rejects(
    buildOpenProjectFolderRequestFromToken("", "Repository", () =>
      Promise.resolve({ nativePathLease: "unused" }),
    ),
    /native project-folder token is required/,
  );
});

test("managed and existing-folder projects use separate creation paths", async () => {
  const managedEvents: string[] = [];
  const managed = await createProjectWorkspace({
    name: "  Managed  ",
    workspaceMode: "managed",
    sources: ["source-a"],
    createManaged: (name) => {
      managedEvents.push(`managed:${name}`);
      return Promise.resolve({ id: "managed-1" });
    },
    openFolder: () =>
      Promise.reject(new Error("managed mode must not open a folder")),
    uploadSources: (projectId, sources) => {
      managedEvents.push(`sources:${projectId}:${sources.join(",")}`);
      return Promise.resolve();
    },
  });
  assert.equal(managed.project.id, "managed-1");
  assert.equal(managed.sourceUploadError, null);
  assert.deepEqual(managedEvents, [
    "managed:Managed",
    "sources:managed-1:source-a",
  ]);

  const folderEvents: string[] = [];
  const folder = await createProjectWorkspace({
    name: "  Existing repo  ",
    workspaceMode: "folder",
    folderToken: "opaque-token",
    sources: ["explicit-rag-source"],
    createManaged: () =>
      Promise.reject(
        new Error("folder mode must not create a managed workspace"),
      ),
    openFolder: (token, name) => {
      folderEvents.push(`folder:${token}:${name}`);
      return Promise.resolve({ id: "folder-1" });
    },
    uploadSources: (projectId, sources) => {
      folderEvents.push(`sources:${projectId}:${sources.join(",")}`);
      return Promise.resolve();
    },
  });
  assert.equal(folder.project.id, "folder-1");
  assert.equal(folder.sourceUploadError, null);
  assert.deepEqual(folderEvents, [
    "folder:opaque-token:Existing repo",
    "sources:folder-1:explicit-rag-source",
  ]);
});

test("folder selection and Create stay disabled until prerequisites hold", () => {
  assert.equal(
    projectFolderPickerDisabled({
      nativePathLeasesSupported: false,
      busy: false,
      pickingFolder: false,
    }),
    true,
  );
  assert.equal(
    projectFolderPickerDisabled({
      nativePathLeasesSupported: true,
      busy: false,
      pickingFolder: false,
    }),
    false,
  );
  assert.equal(
    projectCreateDisabled({
      name: "Repository",
      busy: false,
      pickingFolder: false,
      stagingSources: false,
      workspaceMode: "folder",
      folderToken: null,
    }),
    true,
  );
  assert.equal(
    projectCreateDisabled({
      name: "Repository",
      busy: false,
      pickingFolder: false,
      stagingSources: false,
      workspaceMode: "folder",
      folderToken: "opaque-token",
    }),
    false,
  );
  assert.equal(
    projectCreateDisabled({
      name: "Repository",
      busy: false,
      pickingFolder: true,
      stagingSources: false,
      workspaceMode: "managed",
      folderToken: null,
    }),
    true,
    "a pending native picker must not create a managed project",
  );
});

test("closing or replacing the dialog invalidates a pending folder picker", () => {
  const guard = createProjectFolderPickerGuard();
  const first = guard.begin();
  assert.equal(guard.isCurrent(first), true);

  guard.invalidate();
  assert.equal(guard.isCurrent(first), false);

  const second = guard.begin();
  assert.equal(guard.isCurrent(first), false);
  assert.equal(guard.isCurrent(second), true);
});

test("a folder picker that resolves after dialog close cannot refill the draft", async () => {
  const guard = createProjectFolderPickerGuard();
  let settlePicker: ((value: string) => void) | undefined;
  const picker = new Promise<string>((resolve) => {
    settlePicker = resolve;
  });
  let selectedFolder: string | null = null;
  const claim = guard.begin();
  const pendingSelection = picker.then((selected) => {
    if (guard.isCurrent(claim)) {
      selectedFolder = selected;
    }
  });

  guard.invalidate();
  settlePicker?.("late-repository");
  await pendingSelection;

  assert.equal(selectedFolder, null);
});

test("Create Project delegates folder interactions to the tested controller", () => {
  const dialog = source("features/chat/components/new-project-dialog.tsx");
  const api = source("features/chat/api/chat-api.ts");
  const history = source("features/chat/utils/chat-history-storage.ts");
  const hook = source("features/chat/hooks/use-chat-projects.ts");

  assert.ok(dialog.includes("pickNativeProjectFolder"));
  assert.ok(dialog.includes("useNativePathLeasesSupported"));
  assert.ok(dialog.includes("createNewProjectDialogController"));
  assert.ok(dialog.includes("controller.clickExistingFolder"));
  assert.ok(dialog.includes("controller.submit"));
  assert.match(
    dialog,
    /is the live working directory for every chat and code tool/,
  );
  assert.match(dialog, /It is not automatically indexed as a Source/);
  assert.match(dialog, /<fieldset[\s\S]*?<legend[^>]*>Project workspace/);
  assert.match(dialog, /aria-pressed=\{workspaceMode === "managed"\}/);
  assert.ok(dialog.includes("<ProjectSourceDropzone"));
  assert.match(
    api,
    /buildOpenProjectFolderRequestFromToken\(\s*nativePathToken,\s*name,\s*consumeNativePathToken/,
  );
  assert.match(
    history,
    /openStoredChatProjectFolder[\s\S]*openChatProjectFolder\(nativePathToken, name\)/,
  );
  assert.match(
    hook,
    /openChatProjectFromFolder[\s\S]*openStoredChatProjectFolder\(nativePathToken, name\)/,
  );
});

test("the picker returns an opaque token and never sends a renderer path", () => {
  const dialog = source("features/chat/components/new-project-dialog.tsx");
  const nativeApi = source("features/native-intents/api.ts");

  assert.ok(dialog.includes("controller.folderPickerDisabled"));
  assert.match(
    nativeApi,
    /invokeNative<NativeDocumentFolderSelection \| null>\(\s*"pick_native_project_folder"/,
  );
  assert.ok(!dialog.includes("folder.rootPath"));
  assert.ok(!dialog.includes("folder.path"));
});
