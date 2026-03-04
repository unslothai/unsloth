import type {
  CheckFormatResponse,
  LocalDatasetsResponse,
} from "../types/datasets";

type CheckDatasetFormatArgs = {
  datasetName: string;
  hfToken: string | null;
  subset?: string | null;
  split?: string | null;
  isVlm?: boolean;
};

export async function checkDatasetFormat({
  datasetName,
  hfToken,
  subset,
  split,
  isVlm,
}: CheckDatasetFormatArgs): Promise<CheckFormatResponse> {
  const res = await fetch("/api/datasets/check-format", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: datasetName,
      hf_token: hfToken || undefined,
      subset: subset || undefined,
      split: split || "train",
      is_vlm: !!isVlm,
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }

  return res.json();
}

export async function listLocalDatasets(): Promise<LocalDatasetsResponse> {
  const res = await fetch("/api/datasets/local");
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}
