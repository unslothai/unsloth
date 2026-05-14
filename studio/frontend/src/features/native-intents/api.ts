import { isTauri } from "@/lib/api-base";
import type {
  NativeIntent,
  NativePathLeaseResponse,
  NativePathOperation,
} from "./types";

async function invokeNative<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri) {
    throw new Error("Native desktop features are only available in the Tauri app.");
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export async function drainNativeIntents(): Promise<NativeIntent[]> {
  if (!isTauri) return [];
  return invokeNative<NativeIntent[]>("drain_native_intents");
}

export async function pickNativeModel(): Promise<NativeIntent | null> {
  if (!isTauri) return null;
  return invokeNative<NativeIntent | null>("pick_native_model");
}

export async function registerNativeModelPath(path: string): Promise<NativeIntent> {
  return invokeNative<NativeIntent>("register_native_model_path", { path });
}

export async function consumeNativePathToken(
  token: string,
  operation: NativePathOperation,
): Promise<NativePathLeaseResponse> {
  return invokeNative<NativePathLeaseResponse>("consume_native_path_token", {
    token,
    operation,
  });
}

export async function revealPathToken(token: string): Promise<void> {
  return invokeNative<void>("reveal_path_token", { token });
}

export async function openPathToken(token: string): Promise<void> {
  return invokeNative<void>("open_path_token", { token });
}
