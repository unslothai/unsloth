// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { OpenAIModel } from "../api/openai-models";
import { sameBaseModelId } from "../components/agent-command";
import { EXAMPLE_MODEL_REPO, EXAMPLE_MODEL_VARIANT } from "./example-model-id";

// The same example the Agents tab ships; named only while nothing is downloaded.
export const PLACEHOLDER_EXAMPLE_MODEL = `${EXAMPLE_MODEL_REPO}:${EXAMPLE_MODEL_VARIANT}`;

export type ExampleModelOption = {
  id: string;
  loaded: boolean;
  quants: string[];
};

// A checkpoint can be an on-disk load path, which /v1 never advertises. Mirrors _looks_like_path.
export function looksLikePath(id: string): boolean {
  return (
    id.startsWith("/") ||
    id.startsWith("~") ||
    id.startsWith(".") ||
    id.includes("\\") ||
    id.toLowerCase().endsWith(".gguf") ||
    (id.match(/\//g)?.length ?? 0) >= 2
  );
}

export function pinQuant(id: string, quant: string | undefined): string {
  return quant && !id.includes(":") ? `${id}:${quant}` : id;
}

export function splitPinnedQuant(model: string): {
  repo: string;
  quant: string | null;
} {
  // A repo id carries no colon, so the first one opens the quant. Split there, not on
  // the last: a variant qualified by its directory (`BF16/model-BF16`, what
  // _qualified_variant_name advertises) contains slashes and its own hyphens, and
  // rejecting those would silently drop the pin the user chose.
  const separator = model.indexOf(":");
  if (separator <= 0) {
    return { repo: model, quant: null };
  }
  const quant = model.slice(separator + 1);
  return quant
    ? { repo: model.slice(0, separator), quant }
    : { repo: model, quant: null };
}

// The models the picker offers: every chat model the server can serve, resident first.
export function exampleModelOptions(
  catalog: OpenAIModel[] | null,
): ExampleModelOption[] {
  if (catalog === null) return [];
  // Two rows can spell one repo differently (`org/Foo` from the Hub cache, `Org/Foo`
  // from LM Studio). Ids resolve case-insensitively here and in the server's alias
  // index, so a second spelling is not a second choice: offering it would show a row
  // that selects the first one, and name quants only the first copy can serve.
  const byIdentity = new Map<string, ExampleModelOption>();
  for (const m of catalog) {
    const identity = m.id.trim().toLowerCase();
    const existing = byIdentity.get(identity);
    if (existing) {
      // Residency belongs to the repo, so keep it wherever the rows disagree.
      existing.loaded = existing.loaded || m.loaded === true;
      continue;
    }
    byIdentity.set(identity, {
      id: m.id,
      loaded: m.loaded === true,
      quants: m.quants?.length ? m.quants : m.quant ? [m.quant] : [],
    });
  }
  const options = [...byIdentity.values()];
  return [
    ...options.filter((o) => o.loaded),
    ...options.filter((o) => !o.loaded),
  ];
}

export type FollowedModelInput = {
  catalog: OpenAIModel[] | null;
  autoSwitch: boolean;
  keylessOnly: boolean;
  checkpoint: string | null | undefined;
  ggufVariant: string | null | undefined;
};

// The model the examples name when nothing is picked: always an id /v1 resolves against.
export function followedExampleModel({
  catalog,
  autoSwitch,
  keylessOnly,
  checkpoint,
  ggufVariant,
}: FollowedModelInput): string | null {
  const usableCheckpoint =
    !!checkpoint &&
    !checkpoint.startsWith("external::") &&
    !looksLikePath(checkpoint);
  // Name something held here, with its quant to pin the file on disk.
  const fromCatalog = (): string | null => {
    const pick =
      catalog?.find((m) => m.loaded) ??
      (!keylessOnly && autoSwitch ? catalog?.[0] : undefined);
    if (!pick) {
      return null;
    }
    return pinQuant(pick.id, pick.quant);
  };
  // The store keeps a checkpoint across an idle unload and across the model being
  // deleted, so it only names a runnable model while the catalog still lists it:
  // resident, or downloaded with switching able to reload it. A null catalog means
  // /v1/models has not answered, which is not evidence against it.
  const entry = catalog?.find((m) => sameBaseModelId(m.id, checkpoint ?? ""));
  const backed =
    (!keylessOnly && catalog === null) ||
    (!!entry && (entry.loaded || (!keylessOnly && autoSwitch)));
  if (usableCheckpoint && checkpoint && backed) {
    if (checkpoint.includes(":")) {
      return checkpoint;
    }
    // Pin the quant the catalog advertises, not the stored one: membership proves the
    // repo, and the saved quant can name a file deleted while another quant remains.
    // Fall back to the store only before /v1/models answers.
    const quant = catalog === null ? ggufVariant : entry?.quant;
    return quant ? `${checkpoint}:${quant}` : checkpoint;
  }
  return fromCatalog();
}

export type ResolvedExampleModel = {
  // What the snippet names; null only while /v1 has not answered and nothing is resident.
  model: string | null;
  // The model the server would serve on its own, for the coding-agent verdict.
  followed: string | null;
  // The option the model select shows, null for a path-loaded or media model.
  option: ExampleModelOption | null;
  // False when the pick is downloaded but not resident and switching is off,
  // so the copied request would 404.
  servable: boolean;
  // Why the request would 404, so the warning names a remedy that works. A keyless
  // caller is refused a switch server-side, so the toggle is not one.
  blockedBy: "autoSwitchOff" | "keyless" | null;
  // Nothing downloaded: the snippet names the placeholder instead.
  placeholder: boolean;
};

// A downloaded model only answers when it is resident, or when switching can reload
// it. Keyless callers never get that switch, so the toggle is not their remedy.
function servability(
  option: ExampleModelOption,
  pinned: string | null,
  keylessOnly: boolean,
  autoSwitch: boolean,
): { servable: boolean; blockedBy: "autoSwitchOff" | "keyless" | null } {
  // `loaded` marks the repo, but only one quant of it is in memory, and the catalog
  // lists that one first. Pinning any other quant still needs a switch to serve, so
  // the resident shortcut only applies to the quant actually loaded.
  const residentQuant = option.quants[0] ?? null;
  if (
    option.loaded &&
    (pinned === null || residentQuant === null || pinned === residentQuant)
  ) {
    return { servable: true, blockedBy: null };
  }
  if (keylessOnly) {
    return { servable: false, blockedBy: "keyless" };
  }
  return autoSwitch
    ? { servable: true, blockedBy: null }
    : { servable: false, blockedBy: "autoSwitchOff" };
}

export function resolveExampleModel(
  input: FollowedModelInput & {
    picked: string | null;
    options: ExampleModelOption[];
  },
): ResolvedExampleModel {
  const { picked, options, catalog, autoSwitch, keylessOnly } = input;
  const followed = followedExampleModel(input);
  const optionFor = (id: string | null): ExampleModelOption | null =>
    id === null
      ? null
      : (options.find((o) => sameBaseModelId(o.id, id)) ?? null);
  if (picked !== null) {
    const { repo, quant } = splitPinnedQuant(picked);
    const option = optionFor(repo);
    // A pick the catalog no longer lists (deleted, or /v1 not answered yet) follows.
    if (option !== null) {
      const pinned =
        quant && option.quants.includes(quant) ? quant : option.quants[0];
      return {
        model: pinQuant(option.id, pinned),
        followed,
        option,
        ...servability(option, pinned, keylessOnly, autoSwitch),
        placeholder: false,
      };
    }
  }
  if (followed === null && catalog !== null) {
    // Nothing downloaded: name the example the Agents tab ships, so the request shape
    // is still visible, with the note saying this server does not have it.
    if (options.length === 0) {
      return {
        model: PLACEHOLDER_EXAMPLE_MODEL,
        followed: null,
        option: null,
        servable: false,
        blockedBy: null,
        placeholder: true,
      };
    }
    // Downloaded, but nothing resident and switching cannot reload it. Name the first
    // model held here rather than nothing: the snippet is the one the user wants once
    // it is loaded, and the warning below says why it will not run yet.
    const first = options[0];
    return {
      model: pinQuant(first.id, first.quants[0]),
      followed: null,
      option: first,
      ...servability(first, first.quants[0] ?? null, keylessOnly, autoSwitch),
      placeholder: false,
    };
  }
  return {
    model: followed,
    followed,
    option: optionFor(followed),
    servable: true,
    blockedBy: null,
    placeholder: false,
  };
}
