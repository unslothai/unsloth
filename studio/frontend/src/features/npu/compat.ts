// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No path aliases here, so node:test can import it directly.

/** One row of GET /api/npu/compat: a GGUF architecture and shape FastFlowLM runs. */
export interface NpuCompatShape {
  architecture: string;
  layers: number;
  hidden: number;
  experts: number;
  checkpoints: string[];
  names: string[];
  /** FastFlowLM's size keys ("1.7b", "e4b", "35b-a3b"), matched against repo names. */
  sizes: string[];
  runs: boolean;
  converts: boolean;
}

export interface NpuCompat {
  fastflowlm_known: boolean;
  converter_known: boolean;
  converter: {
    architectures: string[];
    /** general.basename prefixes the converter also matches. */
    basenames: string[];
    qwen35_shapes: number[][];
  } | null;
  shapes: NpuCompatShape[];
  architectures: { architecture: string; runs: boolean; converts: boolean }[];
}

export type NpuTag = "runs" | "convertRuns" | "convertOnly" | "unsupported";

export const NPU_TAG_TEXT: Record<NpuTag, string> = {
  runs: "Runs on NPU",
  convertRuns: "Convert, then runs",
  convertOnly: "Converts, won't run",
  unsupported: "Not supported",
};

/** A FastFlowLM catalog model: FastFlowLM ships it. */
export const NPU_CATALOG_TAG: NpuTag = "runs";

export interface NpuFit {
  /** null when the installed FastFlowLM or converter could not be read. */
  tag: NpuTag | null;
  note: string | null;
}

// A size key standing alone in a name: "4b" in "Qwen3-4B-Instruct", not in "Qwen3-14B".
function namesSize(name: string, size: string): boolean {
  for (let at = name.indexOf(size); at >= 0; at = name.indexOf(size, at + 1)) {
    const before = name[at - 1] ?? "";
    const after = name[at + size.length] ?? "";
    if (!/[a-z0-9.]/.test(before) && !/[a-z0-9]/.test(after)) return true;
  }
  return false;
}

const NAMED_SIZE = /(?:^|[^a-z0-9.])e?\d+(?:\.\d+)?b(?![a-z0-9])/;

function repoName(name: string): string {
  const first = name.trim().split(/\s+/)[0] ?? "";
  return first.slice(Math.max(first.lastIndexOf("/"), first.lastIndexOf("\\")) + 1);
}

function list(names: string[]): string {
  return names.length > 1
    ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`
    : (names[0] ?? "");
}

/** The tag for a GGUF model of this architecture; `name` is its repo id or file path. */
export function npuFit(
  compat: NpuCompat | null | undefined,
  architecture: string | null | undefined,
  name?: string | null,
): NpuFit {
  const arch = architecture?.trim().toLowerCase();
  if (!compat || !arch || !compat.fastflowlm_known || !compat.converter_known) {
    return { tag: null, note: null };
  }
  const lower = (name ?? "").toLowerCase();
  const shapes = compat.shapes.filter((s) => s.architecture === arch);
  const matched = shapes.filter((s) => s.sizes.some((size) => namesSize(lower, size)));
  const sized = matched.length > 0 || NAMED_SIZE.test(lower);
  const runNames = shapes.flatMap((s) => s.names.slice(0, 1));
  const converter = compat.converter;
  const converts =
    arch === "qwen35"
      ? lower.includes("omni") || (matched.length > 0 ? matched.some((s) => s.converts) : !sized)
      : !!converter &&
        (converter.architectures.includes(arch) ||
          converter.basenames.some((prefix) => repoName(lower).startsWith(prefix)));

  if (matched.length > 0) {
    return converts
      ? { tag: "convertRuns", note: null }
      : {
          tag: "runs",
          note: `FastFlowLM ships its own ${matched[0].names[0]}; download it from the NPU list. The Q4NX converter does not convert ${arch} models.`,
        };
  }
  if (shapes.length > 0 && !sized) {
    // No size in the name: only the converter's size check after download settles it.
    return converts
      ? { tag: "convertRuns", note: `Only these ${arch} sizes run on the NPU: ${list(runNames)}.` }
      : {
          tag: "unsupported",
          note: `The Q4NX converter does not convert ${arch} models; FastFlowLM ships ${list(runNames)}.`,
        };
  }
  if (converts) {
    return {
      tag: "convertOnly",
      note: shapes.length
        ? `FastFlowLM has no NPU kernels for this size. The ${arch} sizes it runs: ${list(runNames)}.`
        : `FastFlowLM has no NPU kernels for the ${arch} architecture.`,
    };
  }
  return {
    tag: "unsupported",
    note: `Neither FastFlowLM nor the Q4NX converter supports the ${arch} architecture.`,
  };
}
