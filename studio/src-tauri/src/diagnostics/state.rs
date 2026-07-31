use crate::preflight::{DesktopPreflightDisposition, DesktopPreflightResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::phase_log::{PhaseLogHandle, PhaseLogWriter, SegmentOwner};
use super::{cap_string, new_id, now_ms, MAX_STATE_ITEMS};

pub type DiagnosticsState = Arc<Mutex<DiagnosticsSnapshot>>;

#[derive(Clone, Debug)]
pub struct DiagnosticsSnapshot {
    pub app_session_id: String,
    pub app_started_at_ms: u64,
    pub latest_preflight: Option<PreflightSummary>,
    pub install: Option<AttemptSummary>,
    pub install_history: Vec<AttemptSummary>,
    pub repair_groups: Vec<RepairGroupSummary>,
    pub update: Option<AttemptSummary>,
    pub backend: Option<BackendSummary>,
    pub auth: Option<AuthSummary>,
}

impl Default for DiagnosticsSnapshot {
    fn default() -> Self {
        Self {
            app_session_id: new_id("app"),
            app_started_at_ms: now_ms(),
            latest_preflight: None,
            install: None,
            install_history: Vec::new(),
            repair_groups: Vec::new(),
            update: None,
            backend: None,
            auth: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CappedStrings {
    pub items: Vec<String>,
    pub omitted: usize,
}

impl CappedStrings {
    fn push(&mut self, value: impl Into<String>) {
        if self.items.len() >= MAX_STATE_ITEMS {
            self.items.remove(0);
            self.omitted += 1;
        }
        self.items.push(value.into());
    }
}

#[derive(Clone, Debug)]
pub struct PreflightSummary {
    pub disposition: String,
    pub reason: Option<String>,
    pub port: Option<u16>,
    pub can_auto_repair: bool,
    pub managed_bin: Option<PathBuf>,
    pub recorded_at_ms: u64,
}

#[derive(Clone, Debug)]
pub struct AttemptSummary {
    pub flow: String,
    pub attempt_id: String,
    pub repair_group_id: Option<String>,
    pub child_type: Option<String>,
    pub log_segments: Vec<PathBuf>,
    pub started_at_ms: u64,
    pub ended_at_ms: Option<u64>,
    pub exit_status: Option<String>,
    pub intentional_stop: bool,
    pub last_error: Option<String>,
    pub elevation_packages: Vec<String>,
    pub steps: CappedStrings,
    pub progress_details: CappedStrings,
    pub diag_markers: CappedStrings,
}

impl AttemptSummary {
    pub(crate) fn new(
        flow: impl Into<String>,
        attempt_id: String,
        repair_group_id: Option<String>,
        child_type: Option<String>,
        started_at_ms: u64,
    ) -> Self {
        Self {
            flow: flow.into(),
            attempt_id,
            repair_group_id,
            child_type,
            log_segments: Vec::new(),
            started_at_ms,
            ended_at_ms: None,
            exit_status: None,
            intentional_stop: false,
            last_error: None,
            elevation_packages: Vec::new(),
            steps: CappedStrings::default(),
            progress_details: CappedStrings::default(),
            diag_markers: CappedStrings::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RepairGroupSummary {
    pub repair_group_id: String,
    pub children: Vec<AttemptSummary>,
    pub started_at_ms: u64,
    pub ended_at_ms: Option<u64>,
    pub final_status: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BackendSummary {
    pub session_id: Option<String>,
    pub backend_kind: String,
    pub log_segments: Vec<PathBuf>,
    pub requested_port: Option<u16>,
    pub reported_port: Option<u16>,
    pub generation: Option<u64>,
    pub started_at_ms: Option<u64>,
    pub ended_at_ms: Option<u64>,
    pub exit_status: Option<String>,
    pub intentional_stop: bool,
    pub terminal_reason: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AuthSummary {
    pub failed_at_ms: u64,
    pub stage: String,
    pub port: Option<u16>,
    pub message: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrontendSupportSnapshot {
    pub status: Option<String>,
    pub error: Option<String>,
    pub current_step_index: Option<i64>,
    pub progress_detail: Option<String>,
    pub elevation_packages: Option<Vec<String>>,
    pub last_ui_log_lines: Option<Vec<String>>,
    pub flow: Option<String>,
    pub update_phase: Option<String>,
    /// Frontend update progress can be numeric today; keep this tolerant of strings/numbers.
    pub update_progress: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct AttemptLog {
    pub flow: String,
    pub attempt_id: String,
    pub repair_group_id: Option<String>,
    #[allow(dead_code)]
    pub child_type: Option<String>,
    pub handle: PhaseLogHandle,
}

#[derive(Clone)]
pub struct BackendLog {
    pub session_id: String,
    pub handle: PhaseLogHandle,
}

pub fn new_diagnostics_state() -> DiagnosticsState {
    Arc::new(Mutex::new(DiagnosticsSnapshot::default()))
}

pub fn begin_install_attempt(diagnostics: &DiagnosticsState) -> AttemptLog {
    begin_attempt(diagnostics, "install", None, None)
}

pub fn begin_update_attempt(diagnostics: &DiagnosticsState) -> AttemptLog {
    begin_attempt(diagnostics, "update", None, None)
}

pub fn begin_repair_group(diagnostics: &DiagnosticsState) -> String {
    let repair_group_id = new_id("repair");
    with_snapshot(diagnostics, |snapshot| {
        if snapshot.repair_groups.len() >= MAX_STATE_ITEMS {
            snapshot.repair_groups.remove(0);
        }
        snapshot.repair_groups.push(RepairGroupSummary {
            repair_group_id: repair_group_id.clone(),
            children: Vec::new(),
            started_at_ms: now_ms(),
            ended_at_ms: None,
            final_status: None,
            last_error: None,
        });
    });
    repair_group_id
}

pub fn begin_repair_child(
    diagnostics: &DiagnosticsState,
    repair_group_id: &str,
    child_type: &str,
) -> AttemptLog {
    begin_attempt(
        diagnostics,
        "repair",
        Some(repair_group_id.to_string()),
        Some(child_type.to_string()),
    )
}

fn begin_attempt(
    diagnostics: &DiagnosticsState,
    flow: &str,
    repair_group_id: Option<String>,
    child_type: Option<String>,
) -> AttemptLog {
    let started_at_ms = now_ms();
    let attempt_id = new_id(child_type.as_deref().unwrap_or(flow));
    let owner = match (flow, repair_group_id.as_ref(), child_type.as_ref()) {
        ("install", None, _) => SegmentOwner::Install {
            attempt_id: attempt_id.clone(),
        },
        ("update", None, _) => SegmentOwner::Update {
            attempt_id: attempt_id.clone(),
        },
        (_, Some(group), Some(child)) => SegmentOwner::Repair {
            repair_group_id: group.clone(),
            attempt_id: attempt_id.clone(),
            child_type: child.clone(),
        },
        _ => SegmentOwner::Install {
            attempt_id: attempt_id.clone(),
        },
    };

    let summary = AttemptSummary::new(
        flow,
        attempt_id.clone(),
        repair_group_id.clone(),
        child_type.clone(),
        started_at_ms,
    );

    with_snapshot(diagnostics, |snapshot| {
        if let Some(group_id) = &repair_group_id {
            let group = ensure_repair_group(snapshot, group_id);
            if group.children.len() >= MAX_STATE_ITEMS {
                group.children.remove(0);
            }
            group.children.push(summary);
        } else if flow == "update" {
            snapshot.update = Some(summary);
        } else {
            if let Some(previous) = snapshot.install.take() {
                if snapshot.install_history.len() >= MAX_STATE_ITEMS {
                    snapshot.install_history.remove(0);
                }
                snapshot.install_history.push(previous);
            }
            snapshot.install = Some(summary);
        }
    });

    let writer = PhaseLogWriter::new(
        diagnostics.clone(),
        owner,
        if repair_group_id.is_some() {
            "repair"
        } else {
            flow
        },
        started_at_ms,
        None,
    );

    AttemptLog {
        flow: flow.to_string(),
        attempt_id,
        repair_group_id,
        child_type,
        handle: Arc::new(Mutex::new(writer)),
    }
}

pub fn finish_attempt(
    diagnostics: &DiagnosticsState,
    attempt: &AttemptLog,
    exit_status: Option<String>,
    intentional_stop: bool,
    last_error: Option<String>,
) {
    with_snapshot(diagnostics, |snapshot| {
        update_attempt_mut(snapshot, attempt, |summary| {
            if summary.ended_at_ms.is_none() {
                summary.ended_at_ms = Some(now_ms());
            }
            if exit_status.is_some() {
                summary.exit_status = exit_status.clone();
            }
            summary.intentional_stop |= intentional_stop;
            if last_error.is_some() {
                summary.last_error = last_error.clone();
            }
        });
    });
}

pub fn finish_repair_group(
    diagnostics: &DiagnosticsState,
    repair_group_id: &str,
    final_status: impl Into<String>,
    last_error: Option<String>,
) {
    let status = final_status.into();
    with_snapshot(diagnostics, |snapshot| {
        let group = ensure_repair_group(snapshot, repair_group_id);
        group.ended_at_ms = Some(now_ms());
        group.final_status = Some(status.clone());
        group.last_error = last_error.clone();
    });
}

pub fn record_step(diagnostics: &DiagnosticsState, attempt: &AttemptLog, step: &str) {
    record_attempt_string(diagnostics, attempt, step, |summary, value| {
        summary.steps.push(value)
    });
}

pub fn record_progress(diagnostics: &DiagnosticsState, attempt: &AttemptLog, progress: &str) {
    record_attempt_string(diagnostics, attempt, progress, |summary, value| {
        summary.progress_details.push(value)
    });
}

pub fn record_diag_marker(diagnostics: &DiagnosticsState, attempt: &AttemptLog, marker: &str) {
    record_attempt_string(diagnostics, attempt, marker, |summary, value| {
        summary.diag_markers.push(value)
    });
}

pub fn record_elevation_packages(
    diagnostics: &DiagnosticsState,
    attempt: &AttemptLog,
    packages: &[String],
) {
    let capped: Vec<String> = packages.iter().take(100).cloned().collect();
    with_snapshot(diagnostics, |snapshot| {
        update_attempt_mut(snapshot, attempt, |summary| {
            summary.elevation_packages = capped.clone();
        });
    });
}

fn record_attempt_string<F>(
    diagnostics: &DiagnosticsState,
    attempt: &AttemptLog,
    value: &str,
    mut update: F,
) where
    F: FnMut(&mut AttemptSummary, String),
{
    let capped = cap_string(value, 4096);
    with_snapshot(diagnostics, |snapshot| {
        update_attempt_mut(snapshot, attempt, |summary| {
            update(summary, capped.clone());
        });
    });
}

pub fn record_preflight(diagnostics: &DiagnosticsState, result: &DesktopPreflightResult) {
    let summary = PreflightSummary {
        disposition: disposition_label(&result.disposition).to_string(),
        reason: result.reason.clone(),
        port: result.port,
        can_auto_repair: result.can_auto_repair,
        managed_bin: result.managed_bin.clone(),
        recorded_at_ms: now_ms(),
    };
    with_snapshot(diagnostics, |snapshot| {
        snapshot.latest_preflight = Some(summary);
    });
    if matches!(
        result.disposition,
        DesktopPreflightDisposition::AttachedReady
    ) {
        if let Some(port) = result.port {
            record_attached_external_backend(diagnostics, port);
        }
    }
}

pub fn begin_backend_session(
    diagnostics: &DiagnosticsState,
    requested_port: u16,
    generation: u64,
) -> BackendLog {
    let session_id = new_id("backend");
    let started_at_ms = now_ms();
    with_snapshot(diagnostics, |snapshot| {
        snapshot.backend = Some(BackendSummary {
            session_id: Some(session_id.clone()),
            backend_kind: "managed".to_string(),
            log_segments: Vec::new(),
            requested_port: Some(requested_port),
            reported_port: None,
            generation: Some(generation),
            started_at_ms: Some(started_at_ms),
            ended_at_ms: None,
            exit_status: None,
            intentional_stop: false,
            terminal_reason: None,
            last_error: None,
        });
    });
    let writer = PhaseLogWriter::new(
        diagnostics.clone(),
        SegmentOwner::Backend {
            session_id: session_id.clone(),
        },
        "backend",
        started_at_ms,
        Some("managed".to_string()),
    );
    BackendLog {
        session_id,
        handle: Arc::new(Mutex::new(writer)),
    }
}

pub fn begin_adopted_backend_session(diagnostics: &DiagnosticsState, port: u16, generation: u64) {
    let session_id = new_id("adopted-backend");
    let started_at_ms = now_ms();
    with_snapshot(diagnostics, |snapshot| {
        snapshot.backend = Some(BackendSummary {
            session_id: Some(session_id.clone()),
            backend_kind: "adopted".to_string(),
            log_segments: Vec::new(),
            requested_port: Some(port),
            reported_port: Some(port),
            generation: Some(generation),
            started_at_ms: Some(started_at_ms),
            ended_at_ms: None,
            exit_status: None,
            intentional_stop: false,
            terminal_reason: None,
            last_error: None,
        });
    });
}

pub fn record_backend_start_failure(
    diagnostics: &DiagnosticsState,
    requested_port: Option<u16>,
    generation: Option<u64>,
    stage: &str,
    error: &str,
) {
    let error = cap_string(error, 4096);
    with_snapshot(diagnostics, |snapshot| {
        snapshot.backend = Some(BackendSummary {
            session_id: None,
            backend_kind: "managed".to_string(),
            log_segments: Vec::new(),
            requested_port,
            reported_port: None,
            generation,
            started_at_ms: Some(now_ms()),
            ended_at_ms: Some(now_ms()),
            exit_status: None,
            intentional_stop: false,
            terminal_reason: Some(stage.to_string()),
            last_error: Some(error.clone()),
        });
    });
}

pub fn record_backend_port(diagnostics: &DiagnosticsState, session_id: &str, port: u16) {
    with_snapshot(diagnostics, |snapshot| {
        if let Some(backend) = snapshot.backend.as_mut() {
            if backend.session_id.as_deref() == Some(session_id) {
                backend.reported_port = Some(port);
            }
        }
    });
}

pub fn record_backend_exit(
    diagnostics: &DiagnosticsState,
    session_id: &str,
    exit_status: Option<String>,
    intentional_stop: bool,
    terminal_reason: Option<String>,
) {
    with_snapshot(diagnostics, |snapshot| {
        if let Some(backend) = snapshot.backend.as_mut() {
            if backend.session_id.as_deref() != Some(session_id) {
                return;
            }
            if backend.ended_at_ms.is_none() {
                backend.ended_at_ms = Some(now_ms());
            }
            if exit_status.is_some() {
                backend.exit_status = exit_status.clone();
            }
            backend.intentional_stop |= intentional_stop;
            if backend.terminal_reason.is_none() {
                backend.terminal_reason = terminal_reason.clone().or_else(|| {
                    Some(if intentional_stop {
                        "intentional_stop".to_string()
                    } else {
                        "process_exit".to_string()
                    })
                });
            }
        }
    });
}

pub fn record_backend_watchdog(
    diagnostics: &DiagnosticsState,
    generation: u64,
    terminal_reason: &str,
) {
    with_snapshot(diagnostics, |snapshot| {
        if let Some(backend) = snapshot.backend.as_mut() {
            if backend.generation == Some(generation) && backend.terminal_reason.is_none() {
                backend.terminal_reason = Some(terminal_reason.to_string());
                backend.ended_at_ms = Some(now_ms());
            }
        }
    });
}

pub fn record_backend_intentional_stop(diagnostics: &DiagnosticsState) {
    with_snapshot(diagnostics, |snapshot| {
        if let Some(backend) = snapshot.backend.as_mut() {
            if matches!(
                backend.terminal_reason.as_deref(),
                Some("no_port_after_grace" | "unresponsive_health_check")
            ) {
                return;
            }
            backend.intentional_stop = true;
            if backend.terminal_reason.is_none() {
                backend.terminal_reason = Some("intentional_stop".to_string());
            }
        }
    });
}

pub fn record_attached_external_backend(diagnostics: &DiagnosticsState, port: u16) {
    with_snapshot(diagnostics, |snapshot| {
        let should_replace = snapshot
            .backend
            .as_ref()
            .map(|backend| backend.backend_kind != "managed" || backend.session_id.is_none())
            .unwrap_or(true);
        if should_replace {
            snapshot.backend = Some(BackendSummary {
                session_id: None,
                backend_kind: "attached_external".to_string(),
                log_segments: Vec::new(),
                requested_port: None,
                reported_port: Some(port),
                generation: None,
                started_at_ms: None,
                ended_at_ms: None,
                exit_status: None,
                intentional_stop: false,
                terminal_reason: None,
                last_error: None,
            });
        }
    });
}

pub fn record_auth_failure(
    diagnostics: &DiagnosticsState,
    stage: &str,
    port: Option<u16>,
    message: &str,
) {
    let message = cap_string(message, 4096);
    with_snapshot(diagnostics, |snapshot| {
        snapshot.auth = Some(AuthSummary {
            failed_at_ms: now_ms(),
            stage: stage.to_string(),
            port,
            message,
        });
    });
}

pub(crate) fn with_snapshot<F>(diagnostics: &DiagnosticsState, update: F)
where
    F: FnOnce(&mut DiagnosticsSnapshot),
{
    match diagnostics.lock() {
        Ok(mut snapshot) => update(&mut snapshot),
        Err(poisoned) => {
            let mut snapshot = poisoned.into_inner();
            update(&mut snapshot);
        }
    }
}

pub(crate) fn clone_snapshot(diagnostics: &DiagnosticsState) -> DiagnosticsSnapshot {
    match diagnostics.lock() {
        Ok(snapshot) => snapshot.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

fn ensure_repair_group<'a>(
    snapshot: &'a mut DiagnosticsSnapshot,
    repair_group_id: &str,
) -> &'a mut RepairGroupSummary {
    if let Some(index) = snapshot
        .repair_groups
        .iter()
        .position(|group| group.repair_group_id == repair_group_id)
    {
        return &mut snapshot.repair_groups[index];
    }
    snapshot.repair_groups.push(RepairGroupSummary {
        repair_group_id: repair_group_id.to_string(),
        children: Vec::new(),
        started_at_ms: now_ms(),
        ended_at_ms: None,
        final_status: None,
        last_error: None,
    });
    snapshot.repair_groups.last_mut().expect("just pushed")
}

fn update_attempt_mut<F>(snapshot: &mut DiagnosticsSnapshot, attempt: &AttemptLog, update: F)
where
    F: FnOnce(&mut AttemptSummary),
{
    if let Some(group_id) = &attempt.repair_group_id {
        if let Some(group) = snapshot
            .repair_groups
            .iter_mut()
            .find(|group| &group.repair_group_id == group_id)
        {
            if let Some(child) = group
                .children
                .iter_mut()
                .rev()
                .find(|child| child.attempt_id == attempt.attempt_id)
            {
                update(child);
            }
        }
    } else if attempt.flow == "update" {
        if let Some(summary) = snapshot.update.as_mut() {
            if summary.attempt_id == attempt.attempt_id {
                update(summary);
            }
        }
    } else if let Some(summary) = snapshot.install.as_mut() {
        if summary.attempt_id == attempt.attempt_id {
            update(summary);
        }
    }
}

pub(crate) fn record_segment_path(
    diagnostics: &DiagnosticsState,
    owner: &SegmentOwner,
    path: PathBuf,
) {
    with_snapshot(diagnostics, |snapshot| match owner {
        SegmentOwner::Install { attempt_id } => {
            if let Some(summary) = snapshot.install.as_mut() {
                if &summary.attempt_id == attempt_id && !summary.log_segments.contains(&path) {
                    summary.log_segments.push(path.clone());
                }
            }
            for summary in &mut snapshot.install_history {
                if &summary.attempt_id == attempt_id && !summary.log_segments.contains(&path) {
                    summary.log_segments.push(path.clone());
                }
            }
        }
        SegmentOwner::Update { attempt_id } => {
            if let Some(summary) = snapshot.update.as_mut() {
                if &summary.attempt_id == attempt_id && !summary.log_segments.contains(&path) {
                    summary.log_segments.push(path.clone());
                }
            }
        }
        SegmentOwner::Repair {
            repair_group_id,
            attempt_id,
            ..
        } => {
            if let Some(group) = snapshot
                .repair_groups
                .iter_mut()
                .find(|group| &group.repair_group_id == repair_group_id)
            {
                if let Some(child) = group
                    .children
                    .iter_mut()
                    .find(|child| &child.attempt_id == attempt_id)
                {
                    if !child.log_segments.contains(&path) {
                        child.log_segments.push(path.clone());
                    }
                }
            }
        }
        SegmentOwner::Backend { session_id } => {
            if let Some(backend) = snapshot.backend.as_mut() {
                if backend.session_id.as_deref() == Some(session_id)
                    && !backend.log_segments.contains(&path)
                {
                    backend.log_segments.push(path.clone());
                }
            }
        }
    });
}

pub(crate) fn forget_segment_path(
    diagnostics: &DiagnosticsState,
    owner: &SegmentOwner,
    path: &PathBuf,
) {
    with_snapshot(diagnostics, |snapshot| match owner {
        SegmentOwner::Install { attempt_id } => {
            if let Some(summary) = snapshot.install.as_mut() {
                if &summary.attempt_id == attempt_id {
                    summary.log_segments.retain(|segment| segment != path);
                }
            }
            for summary in &mut snapshot.install_history {
                if &summary.attempt_id == attempt_id {
                    summary.log_segments.retain(|segment| segment != path);
                }
            }
        }
        SegmentOwner::Update { attempt_id } => {
            if let Some(summary) = snapshot.update.as_mut() {
                if &summary.attempt_id == attempt_id {
                    summary.log_segments.retain(|segment| segment != path);
                }
            }
        }
        SegmentOwner::Repair {
            repair_group_id,
            attempt_id,
            ..
        } => {
            if let Some(group) = snapshot
                .repair_groups
                .iter_mut()
                .find(|group| &group.repair_group_id == repair_group_id)
            {
                if let Some(child) = group
                    .children
                    .iter_mut()
                    .find(|child| &child.attempt_id == attempt_id)
                {
                    child.log_segments.retain(|segment| segment != path);
                }
            }
        }
        SegmentOwner::Backend { session_id } => {
            if let Some(backend) = snapshot.backend.as_mut() {
                if backend.session_id.as_deref() == Some(session_id) {
                    backend.log_segments.retain(|segment| segment != path);
                }
            }
        }
    });
}

fn disposition_label(disposition: &DesktopPreflightDisposition) -> &'static str {
    match disposition {
        DesktopPreflightDisposition::NotInstalled => "not_installed",
        DesktopPreflightDisposition::ManagedReady => "managed_ready",
        DesktopPreflightDisposition::ManagedStale => "managed_stale",
        DesktopPreflightDisposition::OwnedReady => "owned_ready",
        DesktopPreflightDisposition::OwnedStale => "owned_stale",
        DesktopPreflightDisposition::AttachedReady => "attached_ready",
        DesktopPreflightDisposition::ExternalConflict => "external_conflict",
    }
}
