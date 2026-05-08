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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum BackendProbe {
    Missing,
    Ready { port: u16 },
    Old { port: u16, reason: String },
    ExternalConflict { port: u16, reason: String },
}
