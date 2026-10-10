// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelIniResponse } from "@/features/chat";

export function modelIniLocationLabel(
  ini: Pick<ModelIniResponse, "filename" | "location">,
): string {
  const name = ini.filename || "unsloth.ini";
  switch (ini.location) {
    case "variant_folder":
      return `${name} in this quant's folder`;
    case "repo_root":
      return `${name} in the repo root`;
    case "local_dir":
      return `${name} beside the model file`;
    default:
      return name;
  }
}

/** `--ctx-size 56000 --no-mmap` -> `ctx-size=56000, no-mmap`, with the slot count the INI moves to `n_parallel`. */
export function formatModelIniSettings(
  args: readonly string[],
  nParallel: number | null,
): string {
  const parts: string[] = [];
  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];
    if (!token.startsWith("-")) {
      continue;
    }
    const name = token.replace(/^-+/, "");
    const eq = name.indexOf("=");
    if (eq >= 0) {
      parts.push(`${name.slice(0, eq)}=${name.slice(eq + 1)}`);
      continue;
    }
    const next = args[i + 1];
    // A negative number is a value, not a flag.
    if (next !== undefined && (!next.startsWith("-") || /^-\d/.test(next))) {
      parts.push(`${name}=${next}`);
      i += 1;
    } else {
      parts.push(name);
    }
  }
  if (nParallel != null) {
    parts.push(`parallel=${nParallel}`);
  }
  return parts.join(", ");
}

/** Whether the row has anything to offer: hidden for non-GGUF targets and when the file is absent,
 *  unless the switch is still on, since it is the only way to turn a vanished file's setting off. */
export function shouldShowModelIniRow(
  ini: Pick<ModelIniResponse, "found"> | null | undefined,
  isGguf: boolean,
  isDiffusion: boolean,
  switchedOn = false,
): boolean {
  return isGguf && !isDiffusion && (ini?.found === true || switchedOn);
}
