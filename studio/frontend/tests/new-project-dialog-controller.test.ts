// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type NewProjectDialogDependencies,
  createNewProjectDialogController,
} from "../src/features/chat/components/new-project-dialog-controller.ts";
import type { ProjectRecord } from "../src/features/chat/types.ts";

interface TestSource {
  id: string;
}

function project(id: string, name = "Repository"): ProjectRecord {
  return {
    id,
    name,
    archived: false,
    createdAt: 1,
    updatedAt: 1,
  };
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function dependencies(
  overrides: Partial<NewProjectDialogDependencies<TestSource>> = {},
): NewProjectDialogDependencies<TestSource> {
  return {
    nativePathLeasesSupported: true,
    pickFolder: () => Promise.resolve(null),
    createManaged: (name) => Promise.resolve(project("managed-1", name)),
    openFolder: (_token, name) => Promise.resolve(project("folder-1", name)),
    uploadSources: () => Promise.resolve(),
    currentRoute: () => "/projects",
    onOpenChange: () => undefined,
    activateProject: () => undefined,
    navigateToProject: () => undefined,
    showError: () => undefined,
    ...overrides,
  };
}

test("clicking Use existing folder selects through the native picker", async () => {
  const picker = deferred<{ token: string; displayName: string } | null>();
  let pickerCalls = 0;
  const controller = createNewProjectDialogController<TestSource>(
    dependencies({
      pickFolder: () => {
        pickerCalls += 1;
        return picker.promise;
      },
    }),
  );

  const click = controller.clickExistingFolder();
  assert.equal(pickerCalls, 1);
  assert.deepEqual(controller.getState(), {
    name: "",
    sources: [],
    workspaceMode: "folder",
    folder: null,
    pickingFolder: true,
    busy: false,
    stagingSources: false,
  });

  picker.resolve({ token: "opaque-picker-token", displayName: "repo-name" });
  await click;

  assert.deepEqual(controller.getState().folder, {
    token: "opaque-picker-token",
    displayName: "repo-name",
  });
  assert.equal(controller.getState().name, "repo-name");
  assert.equal(controller.getState().pickingFolder, false);
  assert.equal(controller.createDisabled(), false);
});

test("native picker cancellation leaves folder creation safely disabled", async () => {
  const controller = createNewProjectDialogController<TestSource>(
    dependencies({ pickFolder: () => Promise.resolve(null) }),
  );

  await controller.clickExistingFolder();

  assert.equal(controller.getState().workspaceMode, "folder");
  assert.equal(controller.getState().folder, null);
  assert.equal(controller.getState().pickingFolder, false);
  assert.equal(controller.createDisabled(), true);
});

test("dialog Cancel invalidates a stale native picker result", async () => {
  const picker = deferred<{ token: string; displayName: string } | null>();
  const openChanges: boolean[] = [];
  const controller = createNewProjectDialogController<TestSource>(
    dependencies({
      pickFolder: () => picker.promise,
      onOpenChange: (open) => openChanges.push(open),
    }),
  );

  const click = controller.clickExistingFolder();
  controller.clickCancel();
  picker.resolve({ token: "stale-token", displayName: "stale-repo" });
  await click;

  assert.deepEqual(openChanges, [false]);
  assert.deepEqual(controller.getState(), {
    name: "",
    sources: [],
    workspaceMode: "managed",
    folder: null,
    pickingFolder: false,
    busy: false,
    stagingSources: false,
  });
});

test("folder Create submits once, then uploads only explicit Sources", async () => {
  const opened = deferred<ProjectRecord>();
  const events: string[] = [];
  const openCalls: Array<{ token: string; name: string }> = [];
  const uploaded: Array<{ projectId: string; sources: TestSource[] }> = [];
  const activeProjects: string[] = [];
  const navigatedProjects: string[] = [];
  const openChanges: boolean[] = [];
  const controller = createNewProjectDialogController<TestSource>(
    dependencies({
      pickFolder: () =>
        Promise.resolve({
          token: "opaque-folder-token",
          displayName: "selected-repo",
        }),
      createManaged: () => {
        throw new Error("folder Create must not create a managed workspace");
      },
      openFolder: (token, name) => {
        events.push("open-folder");
        openCalls.push({ token, name });
        return opened.promise;
      },
      uploadSources: (projectId, sources) => {
        events.push("upload-sources");
        uploaded.push({ projectId, sources });
        return Promise.resolve();
      },
      onOpenChange: (open) => openChanges.push(open),
      activateProject: (projectId) => activeProjects.push(projectId),
      navigateToProject: (projectId) => navigatedProjects.push(projectId),
    }),
  );

  await controller.clickExistingFolder();
  controller.setName("  Working copy  ");
  controller.setSources([{ id: "explicit-rag-source" }]);

  const firstClick = controller.submit();
  const secondClick = controller.submit();
  assert.equal(controller.getState().busy, true);
  assert.deepEqual(openCalls, [
    { token: "opaque-folder-token", name: "Working copy" },
  ]);

  opened.resolve(project("folder-project", "Working copy"));
  await Promise.all([firstClick, secondClick]);

  assert.deepEqual(events, ["open-folder", "upload-sources"]);
  assert.deepEqual(uploaded, [
    {
      projectId: "folder-project",
      sources: [{ id: "explicit-rag-source" }],
    },
  ]);
  assert.equal(
    JSON.stringify(uploaded).includes("opaque-folder-token"),
    false,
    "the selected workspace must not become an implicit RAG Source",
  );
  assert.deepEqual(openChanges, [false]);
  assert.deepEqual(activeProjects, ["folder-project"]);
  assert.deepEqual(navigatedProjects, ["folder-project"]);
  assert.equal(controller.getState().busy, false);
  assert.equal(controller.getState().workspaceMode, "managed");
  assert.deepEqual(controller.getState().sources, []);
});

test("source upload failure still opens the durable project", async () => {
  const activeProjects: string[] = [];
  const navigatedProjects: string[] = [];
  const errors: Array<{ title: string; description?: string }> = [];
  const controller = createNewProjectDialogController<TestSource>(
    dependencies({
      createManaged: (name) =>
        Promise.resolve(project("created-project", name)),
      uploadSources: () => Promise.reject(new Error("Upload interrupted.")),
      activateProject: (projectId) => activeProjects.push(projectId),
      navigateToProject: (projectId) => navigatedProjects.push(projectId),
      showError: (title, description) => errors.push({ title, description }),
    }),
  );
  controller.setName("Durable project");
  controller.setSources([{ id: "source-1" }]);

  await controller.submit();

  assert.deepEqual(activeProjects, ["created-project"]);
  assert.deepEqual(navigatedProjects, ["created-project"]);
  assert.deepEqual(errors, [
    {
      title: "Project created, but sources were not added",
      description:
        "Upload interrupted. Add the sources again from the project Sources tab.",
    },
  ]);
  assert.equal(controller.getState().busy, false);
});

test("onCreated owns follow-up behavior and receives route staleness", async () => {
  const created = deferred<ProjectRecord>();
  let route = "/projects";
  const callbacks: Array<{
    project: ProjectRecord;
    context: { stayedOnRoute: boolean };
  }> = [];
  const activeProjects: string[] = [];
  const navigatedProjects: string[] = [];
  const controller = createNewProjectDialogController<TestSource>(
    dependencies({
      createManaged: () => created.promise,
      currentRoute: () => route,
      onCreated: (createdProject, context) => {
        callbacks.push({ project: createdProject, context });
      },
      activateProject: (projectId) => activeProjects.push(projectId),
      navigateToProject: (projectId) => navigatedProjects.push(projectId),
    }),
  );
  controller.setName("Managed project");

  const submit = controller.submit();
  route = "/settings";
  created.resolve(project("managed-project", "Managed project"));
  await submit;

  assert.deepEqual(callbacks, [
    {
      project: project("managed-project", "Managed project"),
      context: { stayedOnRoute: false },
    },
  ]);
  assert.deepEqual(activeProjects, []);
  assert.deepEqual(navigatedProjects, []);
});

test("default follow-up does not navigate after the user changes routes", async () => {
  const created = deferred<ProjectRecord>();
  let route = "/projects";
  const activeProjects: string[] = [];
  const navigatedProjects: string[] = [];
  const controller = createNewProjectDialogController<TestSource>(
    dependencies({
      createManaged: () => created.promise,
      currentRoute: () => route,
      activateProject: (projectId) => activeProjects.push(projectId),
      navigateToProject: (projectId) => navigatedProjects.push(projectId),
    }),
  );
  controller.setName("Managed project");

  const submit = controller.submit();
  route = "/settings";
  created.resolve(project("managed-project", "Managed project"));
  await submit;

  assert.deepEqual(activeProjects, []);
  assert.deepEqual(navigatedProjects, []);
});
