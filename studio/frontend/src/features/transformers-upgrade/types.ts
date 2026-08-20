


/** Wire shape of `transformers_upgrade` from /api/inference/validate. */
export interface TransformersUpgradeInfo {
  /** config.json model_type unknown to installed transformers. */
  model_type: string;
  /** Latest transformers release on PyPI at check time. */
  pypi_version?: string | null;
  /** Latest PyPI release ships this model_type (installable after consent). */
  supported_in_pypi?: boolean;
  /** Only transformers main ships it (dev-only; not installable). */
  supported_in_main?: boolean;
}

export type TransformersUpgradePhase = "consent" | "installing" | "error";

/** What `/api/inference/transformers-upgrade-check` says about one model. */
export interface TransformersUpgradeCheck {
  /** Set when no installed transformers ships the architecture but a newer one does. */
  upgrade: TransformersUpgradeInfo | null;
  /** The model ships its own modeling code, so a declined install still has a path. */
  requiresTrustRemoteCode: boolean;
  /** The latest sidecar already routes this model. */
  latestTierActive: boolean;
  /** A run started now loads 16-bit, not bnb 4-bit: the latest sidecar forces it. */
  forces16Bit: boolean;
  /** Installing would permanently strand the resume this check was asked for: the
   *  checkpoint only resumes in the 4-bit load mode the current runtime still gives it.
   *  Only set when a resume run was named in the request. */
  installBreaksExactResume: boolean;
}

/** Which copy of a model to inspect, in the four fields the remote-code scan takes.
 *  Kept identical to that gate's arguments so both read the same config.json. */
export interface ModelCachePin {
  preferLocalCache?: boolean;
  modelLocalPath?: string | null;
  modelSnapshotPath?: string | null;
  modelSnapshotRepoId?: string | null;
}
