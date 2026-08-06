


import type { TauriUpdateController } from "@/hooks/use-tauri-update";
import { createContext, useContext } from "react";

export const TauriUpdateContext = createContext<TauriUpdateController | null>(
  null,
);

export function useTauriUpdateController(): TauriUpdateController | null {
  return useContext(TauriUpdateContext);
}
