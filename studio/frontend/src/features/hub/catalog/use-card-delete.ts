


import { useDeleteConfirmAction } from "./use-delete-confirm-action";

type DeleteSuccessMessage = string | (() => string);

export function useCardDelete({
  action,
  resourceName,
  successMessage,
  onSuccess,
  onSettled,
}: {
  action: () => Promise<void>;
  resourceName: string;
  successMessage: DeleteSuccessMessage;
  onSuccess?: () => void;
  onSettled?: () => void;
}) {
  return useDeleteConfirmAction({
    action,
    successMessage,
    errorToast: (err) => ({
      title: `Failed to delete ${resourceName}`,
      description: err instanceof Error ? err.message : undefined,
    }),
    onSuccess,
    onSettled,
  });
}
