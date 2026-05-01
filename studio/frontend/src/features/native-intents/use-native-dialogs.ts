import { toast } from "sonner";
import { pickNativeModel } from "./api";
import { useNativeIntentStore } from "./store";

export function useChooseNativeModel() {
  const addIntent = useNativeIntentStore((state) => state.addIntent);

  return async () => {
    try {
      const intent = await pickNativeModel();
      if (intent) addIntent(intent);
    } catch (error) {
      toast.error("Could not choose local model", {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };
}
