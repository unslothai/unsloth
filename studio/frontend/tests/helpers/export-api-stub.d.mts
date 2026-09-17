// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Types for export-api-stub.mjs so a .ts test can read the recorded calls without `any`.

export declare const calls: { name: string; args: unknown[] }[];
export declare const responses: Map<string, unknown>;

export declare function resetStub(): void;

export declare const loadCheckpoint: (...args: unknown[]) => Promise<unknown>;
export declare const exportGGUF: (...args: unknown[]) => Promise<unknown>;
export declare const exportMerged: (...args: unknown[]) => Promise<unknown>;
export declare const exportLoRA: (...args: unknown[]) => Promise<unknown>;
export declare const cleanupExport: (...args: unknown[]) => Promise<unknown>;
export declare const cancelExport: (...args: unknown[]) => Promise<unknown>;
export declare const getExportStatus: (...args: unknown[]) => Promise<unknown>;

export declare function isRecoverableTransportError(): boolean;
