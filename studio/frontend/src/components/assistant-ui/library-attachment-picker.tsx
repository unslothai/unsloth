// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui } from "@assistant-ui/react";
import {
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { getAuthSessionEpoch } from "@/features/auth";
import {
  createLibraryComposerAttachment,
  fileKind,
  getLibrary,
  libraryItemFile,
  LibraryAttachmentTargetContext,
  LibraryFileTooLarge,
  type LibraryItem,
} from "@/features/library";
import { MAX_AUDIO_SIZE } from "@/lib/audio-utils";
import { MAX_VIDEO_SIZE } from "@/lib/video-utils";
import { useT } from "@/i18n";

function attachmentLimit(item: LibraryItem): number {
  const kind = fileKind(item);
  if (kind === "audio") return MAX_AUDIO_SIZE;
  if (kind === "video") return MAX_VIDEO_SIZE;
  return kind === "image" ||
    kind === "code" ||
    kind === "web" ||
    item.textOnly ||
    item.contentType.startsWith("text/")
    ? 20 * 1024 * 1024
    : 50 * 1024 * 1024;
}

export function LibraryAttachmentPicker({ onClose }: { onClose: () => void }) {
  const t = useT();
  const aui = useAui();
  const target = useContext(LibraryAttachmentTargetContext);
  const currentTarget = useRef(target);
  useLayoutEffect(() => {
    currentTarget.current = target;
  }, [target]);
  const [origin] = useState(() => ({ target, session: getAuthSessionEpoch() }));
  const mounted = useRef(true);
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const valid = useCallback(
    () =>
      mounted.current &&
      currentTarget.current === origin.target &&
      getAuthSessionEpoch() === origin.session,
    [origin],
  );

  useEffect(() => {
    mounted.current = true;
    let cancelled = false;
    void getLibrary()
      .then((library) => {
        if (!cancelled && getAuthSessionEpoch() === origin.session)
          setItems(library.items.filter((item) => !item.model));
      })
      .catch((cause: unknown) => {
        if (!cancelled)
          setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      mounted.current = false;
    };
  }, [origin.session]);

  useEffect(() => {
    if (target === null || target !== origin.target) onClose();
  }, [target, origin.target, onClose]);

  const attach = useRef<((item: LibraryItem) => Promise<boolean>) | null>(null);
  useEffect(() => {
    attach.current = createLibraryComposerAttachment<LibraryItem>({
      getTarget: () => currentTarget.current,
      getSession: getAuthSessionEpoch,
      isEditing: () => valid() && aui.composer().getState().isEditing,
      read: (item) => libraryItemFile(item, attachmentLimit(item)),
      add: (file) => aui.composer().addAttachment(file),
    });
  }, [aui, valid]);

  const results = useMemo(() => {
    const search = query.trim().toLocaleLowerCase();
    return items
      .filter((item) =>
        `${item.name} ${item.fileName ?? ""}`
          .toLocaleLowerCase()
          .includes(search),
      )
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, 50);
  }, [items, query]);

  const select = async (item: LibraryItem) => {
    if (pending || !valid()) return;
    setPending(true);
    setError(null);
    try {
      if (item.sizeBytes !== null && item.sizeBytes > attachmentLimit(item))
        throw new LibraryFileTooLarge(item.name);
      if ((await attach.current?.(item)) && valid()) onClose();
    } catch (cause) {
      if (valid())
        setError(
          cause instanceof LibraryFileTooLarge
            ? t("library.toast.tooLargeOne", { name: item.name })
            : cause instanceof Error
              ? cause.message
              : String(cause),
        );
    } finally {
      if (mounted.current) setPending(false);
    }
  };

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="gap-4 sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Attach from Library</DialogTitle>
          <DialogDescription>
            Choose a saved file to attach to this message.
          </DialogDescription>
        </DialogHeader>
        <Command shouldFilter={false}>
          <CommandInput
            placeholder="Search Library files..."
            value={query}
            onValueChange={setQuery}
            disabled={pending}
          />
          <CommandList aria-label="Library files">
            {loading || pending ? (
              <Spinner
                className="mx-auto my-4 size-5"
                label={pending ? "Attaching file" : "Loading Library"}
              />
            ) : null}
            {!loading && !pending ? (
              <CommandEmpty>No files found.</CommandEmpty>
            ) : null}
            {results.map((item) => (
              <CommandItem
                key={item.id}
                value={item.id}
                disabled={pending}
                onSelect={() => {
                  void select(item);
                }}
              >
                <span className="min-w-0 flex-1 truncate">{item.name}</span>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {t(
                    item.source === "generated"
                      ? "library.toolbar.generated"
                      : "library.toolbar.uploaded",
                  )}
                </span>
              </CommandItem>
            ))}
          </CommandList>
        </Command>
        {error ? (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
