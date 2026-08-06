


import { toast } from "@/lib/toast";

export function toastSuccess(message: string): void {
  toast.success(message);
}

export function toastError(message: string, description?: string): void {
  toast.error(message, {
    description,
  });
}
