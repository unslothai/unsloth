use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopPreflightDisposition {
    NotInstalled,
    ManagedReady,
    ManagedStale,
    OwnedReady,
    OwnedStale,
    AttachedReady,
    ExternalConflict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesktopPreflightResult {
    pub disposition: DesktopPreflightDisposition,
    pub reason: Option<String>,
    pub port: Option<u16>,
    pub can_auto_repair: bool,
    pub managed_bin: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalBackendConflict {
    pub port: u16,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ManagedProbe {
    Missing,
    Ready { bin: PathBuf },
    Stale { bin: PathBuf, reason: String },
    /// The install could not be looked at, rather than found wanting: no binary
    /// path to report and nothing to repair until the profile is back.
    Unavailable { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum BackendProbe {
    Missing,
    Ready { port: u16 },
    Old { port: u16, reason: String },
    ExternalConflict { port: u16, reason: String },
    /// A backend answered here, but it is not adoptable and not provably ours:
    /// it reports no install id, so it may be a remote Studio behind a tunnel or
    /// an install that predates the id. Launching skips the port; a mutation
    /// refuses only when a live local process is attributable to this install,
    /// which is the one thing a health probe cannot tell us.
    Unrelated { port: u16, reason: String },
}
