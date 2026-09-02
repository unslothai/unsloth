// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildChangeProjectFolderRequest,
  buildDisconnectProjectFolderRequest,
  buildOpenProjectFolderRequest,
} from "../src/features/chat/api/project-folder-request.ts";
import {
  ProjectWorkspaceMutationError,
  projectWorkspaceMutationShouldStayBusy,
} from "../src/features/chat/api/project-workspace-mutation.ts";
import { createNewProjectDialogController } from "../src/features/chat/components/new-project-dialog-controller.ts";
import { createProjectWorkspace } from "../src/features/chat/components/new-project-workspace.ts";
import type { ProjectRecord } from "../src/features/chat/types.ts";

function project(id = "project-1"): ProjectRecord {
  return {
    id,
    name: "Repository",
    workspaceKind: "folder",
    workspacePath: "/private/repository",
    workspaceAvailable: true,
    workspaceRevision: 2,
    archived: false,
    createdAt: 1,
    updatedAt: 1,
  };
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("project folder requests consume purpose-bound tokens without sending paths", async () => {
  const consumed: Array<[string, string]> = [];
  const consume = async (
    token: string,
    operation: "open-project" | "set-project-workspace",
  ) => {
    consumed.push([token, operation]);
    return { nativePathLease: "signed.lease" };
  };

  const opened = await buildOpenProjectFolderRequest(
    "opaque-token",
    " Repository ",
    consume,
  );
  const changed = await buildChangeProjectFolderRequest(
    "project/one",
    "replacement-token",
    7,
    consume,
  );
  const disconnected = buildDisconnectProjectFolderRequest("project/one", 8);

  assert.deepEqual(consumed, [
    ["opaque-token", "set-project-workspace"],
    ["replacement-token", "set-project-workspace"],
  ]);
  assert.equal(opened.input, "/api/chat/projects/open-folder");
  assert.deepEqual(JSON.parse(String(opened.init.body)), {
    nativePathLease: "signed.lease",
    name: "Repository",
  });
  assert.equal(
    changed.input,
    "/api/chat/projects/project%2Fone/workspace-folder",
  );
  assert.deepEqual(JSON.parse(String(changed.init.body)), {
    nativePathLease: "signed.lease",
    expectedWorkspaceRevision: 7,
  });
  assert.deepEqual(JSON.parse(String(disconnected.init.body)), {
    expectedWorkspaceRevision: 8,
  });
  for (const request of [opened, changed, disconnected]) {
    const encoded = String(request.init.body);
    assert.doesNotMatch(
      encoded,
      /private\/repository|opaque-token|replacement-token/,
    );
  }
});

test("revision conflicts keep workspace controls disabled until refreshed", () => {
  assert.equal(
    projectWorkspaceMutationShouldStayBusy(
      new ProjectWorkspaceMutationError(
        409,
        "Project workspace changed before this update completed.",
      ),
    ),
    true,
  );
  assert.equal(
    projectWorkspaceMutationShouldStayBusy(
      new ProjectWorkspaceMutationError(409, "A project tool is still running."),
    ),
    false,
  );
  assert.equal(
    projectWorkspaceMutationShouldStayBusy(
      new ProjectWorkspaceMutationError(500, "request failed"),
    ),
    false,
  );
  assert.equal(projectWorkspaceMutationShouldStayBusy(new Error("offline")), false);
});

test("a stale folder picker result cannot replace the selected workspace mode", async () => {
  const picked = deferred<{
    token: string;
    displayName: string;
    displayPath: string;
  } | null>();
  const controller = createNewProjectDialogController({
    nativePathLeasesSupported: true,
    pickFolder: () => picked.promise,
    createManaged: async () => project(),
    openFolder: async () => project(),
    uploadSources: async () => {},
    currentRoute: () => "/chat",
    onOpenChange: () => {},
    activateProject: () => {},
    navigateToProject: () => {},
    showError: () => {},
  });

  const pending = controller.clickExistingFolder();
  controller.clickManagedWorkspace();
  picked.resolve({
    token: "late-token",
    displayName: "late",
    displayPath: "/late",
  });
  await pending;

  assert.equal(controller.getState().workspaceMode, "managed");
  assert.equal(controller.getState().folder, null);
});

test("a failed folder create clears the consumed selection before retry", async () => {
  const errors: Array<[string, string | undefined]> = [];
  const controller = createNewProjectDialogController({
    nativePathLeasesSupported: true,
    pickFolder: async () => ({
      token: "one-shot-token",
      displayName: "repository",
      displayPath: "/repository",
    }),
    createManaged: async () => project(),
    openFolder: async () => {
      throw new Error("request failed");
    },
    uploadSources: async () => {},
    currentRoute: () => "/chat",
    onOpenChange: () => {},
    activateProject: () => {},
    navigateToProject: () => {},
    showError: (title, description) => errors.push([title, description]),
  });

  await controller.clickExistingFolder();
  assert.equal(controller.createDisabled(), false);
  await controller.submit();

  assert.equal(controller.getState().folder, null);
  assert.equal(controller.createDisabled(), true);
  assert.match(errors[0]?.[1] ?? "", /Choose the folder again/);
});

test("source uploads remain separate from durable project creation", async () => {
  const calls: string[] = [];
  const created = project("durable-project");

  const result = await createProjectWorkspace({
    name: "Repository",
    workspaceMode: "folder",
    folderToken: "opaque-token",
    sources: ["notes.pdf"],
    createManaged: async () => {
      throw new Error("wrong mode");
    },
    openFolder: async (token) => {
      calls.push(`open:${token}`);
      return created;
    },
    uploadSources: async (projectId, sources) => {
      calls.push(`sources:${projectId}:${sources.join(",")}`);
      throw new Error("source upload failed");
    },
  });

  assert.equal(result.project, created);
  assert.match(String(result.sourceUploadError), /source upload failed/);
  assert.deepEqual(calls, [
    "open:opaque-token",
    "sources:durable-project:notes.pdf",
  ]);
});
