


import { create } from "zustand";

type Resolver = (confirmed: boolean) => void;

/** What confirming does to the model: reload it, or leave none loaded. */
export type StopRunningChatsEffect = "reload" | "unload";

// One at a time: a new request declines any pending one so no promise leaks.
let pendingResolver: Resolver | null = null;

interface StopRunningChatsDialogStore {
  open: boolean;
  /** How many conversations the pending action would stop. */
  count: number;
  /** Titles of those conversations, when known, for the dialog body. */
  titles: string[];
  /** What the user is about to do, e.g. "Loading a different model". */
  action: string;
  /** The set includes an embeddings/completions/audio request, which is not a chat. */
  hasNonChat: boolean;
  /** Ejecting leaves no model loaded, so it must not be described as a reload. */
  effect: StopRunningChatsEffect;
  requestConfirm: (args: {
    count: number;
    titles?: string[];
    action?: string;
    hasNonChat?: boolean;
    effect?: StopRunningChatsEffect;
  }) => Promise<boolean>;
  resolve: (confirmed: boolean) => void;
}

export const useStopRunningChatsDialogStore =
  create<StopRunningChatsDialogStore>()((set) => ({
    open: false,
    count: 0,
    titles: [],
    action: "",
    hasNonChat: false,
    effect: "reload",
    requestConfirm: ({
      count,
      titles = [],
      action = "",
      hasNonChat = false,
      effect = "reload",
    }) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({ open: true, count, titles, action, hasNonChat, effect });
      }),
    resolve: (confirmed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({
        open: false,
        count: 0,
        titles: [],
        action: "",
        hasNonChat: false,
        effect: "reload",
      });
      resolver?.(confirmed);
    },
  }));
