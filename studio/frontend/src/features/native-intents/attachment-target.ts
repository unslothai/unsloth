


import { createContext, useContext } from "react";

export const NativeAttachmentTargetContext = createContext<string | null>(null);

export function useNativeAttachmentTargetKey(): string | null {
  return useContext(NativeAttachmentTargetContext);
}
