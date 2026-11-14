// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const hook = readSrc("features/api-monitor/use-api-model-reload.ts");
const page = readSrc("features/api-monitor/api-monitor-page.tsx");

test("a served row reloads its checkpoint, not the public alias a kept slot is listed under", () => {
  assert.match(hook, /const target = model\.checkpoint === model\.name \? id : model\.checkpoint;[\s\S]*?id: target,/);
  assert.doesNotMatch(hook, /id: standalone \? model\.checkpoint : id,/);
  assert.match(page, /checkpoint: model\.checkpoint \?\? model\.id,/);
});

test("the Context fact finds requests logged as <id>:<quant>, and shows chat's pinnable chip", () => {
  assert.match(page, /modelIdsMatch\(monitorSelection\(entry\.model\)\.id, id\),/);
  assert.match(page, /entry\.event === "load" &&\s*entry\.status === "completed" &&/);
  assert.doesNotMatch(page, /modelIdsMatch\(entry\.model, id\)/);
  assert.match(page, /<ContextUsageChip\s+ringFirst=\{true\}\s+pinOnClick=\{true\}/);
});

test("an MLX precision folder reloads as a model with its folder, not as a GGUF quant", () => {
  assert.match(hook, /const variant = model\.quant \?\? ggufVariant;/);
  assert.match(hook, /kind: variant != null && !mlxFolder \? "gguf" : "model",/);
  // A kept row has no backend flag: a precision folder name on a non-GGUF repo is MLX too.
  assert.match(hook, /model\.isMlx \|\|\s*\(variant != null && isPrecisionFolderName\(variant\) && !\/gguf\/i\.test\(target\)\)/);
  assert.match(page, /const loadedIsMlx = useChatRuntimeStore\(\(s\) => s\.loadedIsMlx\);/);
  assert.match(page, /isMlx: index === activeIndex && loadedIsMlx === true,/);
});

test("a served NPU model reloads through the NPU loader, as the picker routes it", () => {
  assert.match(hook, /if \(isNpuModelId\(target\.id\)\) \{\s*await loadNpuModel\(target\.id, \{\s*forceReload: true,/);
  assert.match(page, /loadNpuModel: runtime\.loadNpuModel,/);
});

test("a row reload asks to stop only the reloaded model's chats, not the selected model's", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(runtime, /const touchesOnlySelected =\s*forceReload && !isExternalModelId\(modelId\) && loadedNow\.length > 1;/);
  assert.match(runtime, /touchesOnlySelected \? modelId : undefined,/);
  assert.doesNotMatch(runtime, /touchesOnlySelected \? \(paramsNow\.checkpoint/);
});

test("a file-picked primary reloads through its file grant, and not once the grant expired", () => {
  assert.match(hook, /model\.active && fileGrant\s*\?\s*grantedTarget\(fileGrant\)\s*:\s*servedTarget\(model\)/);
  assert.match(hook, /nativePathToken: resident\.nativePathToken \?\? undefined,/);
  assert.match(hook, /loadId: target\.loadId,\s*nativePathToken: target\.nativePathToken,\s*nativePathExpiresAtMs: target\.nativePathExpiresAtMs,/);
  // A primary with no grant (a reloaded tab) is judged by its label like a kept row.
  assert.match(hook, /model\.active && fileGrant\s*\?\s*residentHasRunSettings\(fileGrant\)\s*:\s*!isNativeFileLabel\(model\.checkpoint\)/);
  const residentHook = readSrc("features/api-monitor/use-resident-run-target.ts");
  const grantHook = residentHook.slice(residentHook.indexOf("export function useResidentFileGrant("), residentHook.indexOf("export function useResidentRunTarget("));
  assert.doesNotMatch(grantHook, /residentMatchesMonitor|residentHasRunSettings/);
  assert.match(page, /const fileGrant = useResidentFileGrant\(data\?\.active_model\);/);
  assert.match(page, /disabled=\{reloadBusy \|\| !reload\.canReload\(model\)\}/);
  assert.match(page, /served\.length > 0 &&\s*!served\.some\(\(model\) => reload\.canReload\(model\)\)/);
  assert.doesNotMatch(hook, /load\(servedTarget\(model\)\)/);
});

test("a kept file-picked row has no grant, so it cannot reload and Reload all skips it", () => {
  // /status lists a lease-backed load by its bare file name only.
  assert.match(hook, /:\s*!isNativeFileLabel\(model\.checkpoint\)/);
  assert.match(hook, /for \(const model of models\s*\.filter\(canReload\)/);
});

test("Reload all stops when a load is cancelled or a newer pick supersedes it", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.equal(runtime.match(/if \(!stopDecision\.proceed\) \{\s*modelLoadCancelEpoch \+= 1;/g)?.length, 2);
  assert.match(runtime, /modelLoadCancelEpoch \+= 1;\s*run\.abortController\.abort\(\);/);
  // A declined Hugging Face token prompt also resolves quietly, so it counts as a cancel too.
  assert.match(runtime, /if \(!preparedToken\.proceed\) \{\s*modelLoadCancelEpoch \+= 1;/);
  assert.match(hook, /const before = readModelLoadEpochs\(\);\s*await load\(rowTarget\(model\)\);/);
  assert.match(hook, /after\.cancels !== before\.cancels \|\|\s*after\.intent > before\.intent \+ 1/);
  assert.match(page, /readModelLoadEpochs: runtime\.readModelLoadEpochs,/);
});

test("row statistics match the served id, so an MLX folder row still finds its requests", () => {
  assert.match(page, /id: monitorSelection\(model\.id\)\.id,/);
  assert.match(page, /name: activeModel,\s*id: activeId,/);
  assert.match(page, /const \{ id \} = model;/);
  assert.doesNotMatch(page, /const id = monitorSelection\(model\.name\)\.id;/);
});
