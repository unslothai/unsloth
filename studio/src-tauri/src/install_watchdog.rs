//! Keeps a slow install alive, and says what it is waiting on.
//!
//! Killing a download wastes every byte: uv restarts an interrupted one from zero
//! (astral-sh/uv#16934), so the budget marks pathology, not impatience.
//!
//! No silence-based stop: the markers miss the uv calls under `studio setup`, whose
//! output is captured rather than streamed, so quiet there does not mean stuck.

use std::time::{Duration, Instant};

/// Long enough that reaching it means something is wrong rather than slow.
pub const BACKSTOP_TIMEOUT: Duration = Duration::from_secs(12 * 60 * 60);
pub const REPORT_INTERVAL: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, PartialEq, Eq)]
pub enum Marker<'a> {
    Start { package: &'a str, size: &'a str },
    Done { package: &'a str },
}

/// `[TAURI:DL] torch 2.4GiB` / `[TAURI:DL_DONE] torch`, from install.sh and install.ps1.
pub fn parse_marker(line: &str) -> Option<Marker<'_>> {
    if let Some(rest) = line.strip_prefix("[TAURI:DL] ") {
        let (package, size) = rest.trim().split_once(' ')?;
        return (!package.is_empty() && !size.is_empty())
            .then_some(Marker::Start { package, size });
    }
    let package = line.strip_prefix("[TAURI:DL_DONE] ")?.trim();
    (!package.is_empty()).then_some(Marker::Done { package })
}

fn human_duration(elapsed: Duration) -> String {
    let minutes = elapsed.as_secs() / 60;
    if minutes < 60 {
        return format!("{} min", minutes);
    }
    match minutes % 60 {
        0 => format!("{} h", minutes / 60),
        rest => format!("{} h {} min", minutes / 60, rest),
    }
}

struct Download {
    package: String,
    size_text: String,
}

pub struct ProgressWatch {
    started: Instant,
    last_output: Instant,
    last_report: Instant,
    step: String,
    in_flight: Vec<Download>,
    backstop_timeout: Duration,
}

impl ProgressWatch {
    pub fn new(now: Instant) -> Self {
        Self {
            started: now,
            last_output: now,
            last_report: now,
            step: String::new(),
            in_flight: Vec::new(),
            backstop_timeout: BACKSTOP_TIMEOUT,
        }
    }

    #[cfg(test)]
    pub fn with_backstop(now: Instant, backstop: Duration) -> Self {
        Self { backstop_timeout: backstop, ..Self::new(now) }
    }

    pub fn note_output(&mut self, now: Instant) {
        self.last_output = now;
    }

    pub fn note_step(&mut self, step: &str) {
        self.step = step.to_string();
    }

    pub fn note_marker(&mut self, marker: &Marker<'_>, now: Instant) {
        match marker {
            Marker::Start { package, size } => {
                self.in_flight.retain(|d| d.package != *package);
                self.in_flight.push(Download {
                    package: (*package).to_string(),
                    size_text: (*size).to_string(),
                });
            }
            Marker::Done { package } => self.in_flight.retain(|d| d.package != *package),
        }
        self.note_output(now);
    }

    /// The message to fail with, once the run is long enough to be pathological.
    pub fn expired(&self, now: Instant) -> Option<String> {
        (now.duration_since(self.started) >= self.backstop_timeout).then(|| {
            let budget = human_duration(self.backstop_timeout);
            format!("Installation timed out after {budget}{}.", self.where_it_was())
        })
    }

    /// `None` until a full interval of silence, so a talkative install stays quiet.
    pub fn due_report(&mut self, now: Instant) -> Option<String> {
        if now.duration_since(self.last_output) < REPORT_INTERVAL
            || now.duration_since(self.last_report) < REPORT_INTERVAL
        {
            return None;
        }
        self.last_report = now;
        let elapsed = human_duration(now.duration_since(self.started));
        Some(match self.in_flight.first() {
            Some(download) => format!(
                "downloading {} ({}){} -- {} so far",
                download.package,
                download.size_text,
                self.where_it_was(),
                elapsed
            ),
            None => format!("still working{} -- {} so far", self.where_it_was(), elapsed),
        })
    }

    fn where_it_was(&self) -> String {
        match self.step.is_empty() {
            true => String::new(),
            false => format!(" during \"{}\"", self.step),
        }
    }
}

pub type WatchState = std::sync::Arc<std::sync::Mutex<ProgressWatch>>;

/// Returns true for a marker line, which the caller keeps off the UI.
pub fn note_progress(watch: &WatchState, text: &str) -> bool {
    let Ok(mut watch) = watch.lock() else {
        return false;
    };
    let now = Instant::now();
    if let Some(step) = text.strip_prefix("[TAURI:STEP] ") {
        watch.note_step(step);
    }
    match parse_marker(text) {
        Some(marker) => {
            watch.note_marker(&marker, now);
            true
        }
        None => {
            watch.note_output(now);
            false
        }
    }
}

/// `poll` yields the exit status and whether the stop was asked for, or None while running.
pub fn wait_with_watchdog(
    watch: &std::sync::Mutex<ProgressWatch>,
    mut poll: impl FnMut() -> Result<Option<(std::process::ExitStatus, bool)>, String>,
    mut report: impl FnMut(&str),
    mut stop: impl FnMut(),
) -> Result<(std::process::ExitStatus, bool), String> {
    loop {
        if let Some(exit) = poll()? {
            return Ok(exit);
        }

        let now = Instant::now();
        let (due, expired) = {
            let mut watch = watch.lock().map_err(|e| e.to_string())?;
            (watch.due_report(now), watch.expired(now))
        };
        if let Some(line) = due {
            report(&line);
        }
        if let Some(message) = expired {
            stop();
            return Err(message);
        }

        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    const HOUR: Duration = Duration::from_secs(3600);

    fn watch(now: Instant) -> ProgressWatch {
        ProgressWatch::with_backstop(now, 12 * HOUR)
    }

    /// Built rather than spawned: `cargo test` also runs on Windows, which has no /bin/sh.
    fn exit_status(code: i32) -> std::process::ExitStatus {
        #[cfg(unix)]
        return std::os::unix::process::ExitStatusExt::from_raw(code << 8);
        #[cfg(windows)]
        return std::os::windows::process::ExitStatusExt::from_raw(code as u32);
    }

    #[test]
    fn markers_are_parsed_and_anything_else_is_left_alone() {
        assert_eq!(
            parse_marker("[TAURI:DL] torch 2.4GiB"),
            Some(Marker::Start { package: "torch", size: "2.4GiB" })
        );
        assert_eq!(
            parse_marker("[TAURI:DL_DONE] torch"),
            Some(Marker::Done { package: "torch" })
        );
        for not_ours in [
            "[TAURI:STEP] Installing PyTorch",
            "Downloading torch (2.4GiB)",
            "[TAURI:DL] torch",
            "[TAURI:DL_DONE] ",
        ] {
            assert_eq!(parse_marker(not_ours), None, "{not_ours}");
        }
    }

    #[test]
    fn only_the_backstop_stops_an_install() {
        assert_eq!(BACKSTOP_TIMEOUT, 12 * HOUR, "the shipped budget is what these assert");
        let t0 = Instant::now();
        let mut watch = watch(t0);
        watch.note_step("Installing unsloth");
        // The reporting host needed ~3.7 h for its torch download and was killed at 2 h.
        assert_eq!(watch.expired(t0 + 4 * HOUR), None);
        assert_eq!(watch.expired(t0 + 11 * HOUR + Duration::from_secs(3599)), None);
        let message = watch.expired(t0 + 12 * HOUR).expect("the backstop fires");
        assert!(message.contains("12 h"), "{message}");
        assert!(message.contains("Installing unsloth"), "{message}");
    }

    #[test]
    fn a_silent_stretch_reports_once_per_interval_and_names_the_download() {
        let t0 = Instant::now();
        let mut watch = watch(t0);
        watch.note_step("Installing PyTorch");
        watch.note_marker(&Marker::Start { package: "torch", size: "2.4GiB" }, t0);

        // Output resets the window, so an install that talks never reports.
        watch.note_output(t0 + Duration::from_secs(299));
        assert_eq!(watch.due_report(t0 + Duration::from_secs(300)), None);
        let first = watch.due_report(t0 + Duration::from_secs(600)).expect("a report is due");
        assert!(first.contains("torch") && first.contains("2.4GiB"), "{first}");
        assert!(first.contains("Installing PyTorch") && first.contains("10 min"), "{first}");
        assert_eq!(watch.due_report(t0 + Duration::from_secs(601)), None);
        let second = watch.due_report(t0 + Duration::from_secs(900)).expect("a second report");
        assert!(second.contains("15 min"), "{second}");
    }

    #[test]
    fn a_landed_download_stops_being_named() {
        let t0 = Instant::now();
        let mut watch = watch(t0);
        watch.note_marker(&Marker::Start { package: "torch", size: "2.4GiB" }, t0);
        watch.note_marker(&Marker::Done { package: "torch" }, t0);
        let report = watch.due_report(t0 + Duration::from_secs(600)).expect("a report is due");
        assert!(!report.contains("torch"), "{report}");
        assert!(report.contains("still working"), "{report}");
    }

    #[test]
    fn note_progress_records_markers_and_steps_and_reports_which_is_which() {
        let now = Instant::now();
        let watch: WatchState = Arc::new(Mutex::new(watch(now)));
        assert!(note_progress(&watch, "[TAURI:DL] torch 2.4GiB"));
        assert!(!note_progress(&watch, "[TAURI:STEP] Installing PyTorch"));
        assert!(!note_progress(&watch, "  venv  creating environment"));
        // A plain line is output: it postpones the report even though it changes no state.
        assert_eq!(watch.lock().unwrap().due_report(Instant::now()), None);

        let later = now + Duration::from_secs(600);
        let report = watch.lock().unwrap().due_report(later).expect("a report is due");
        assert!(report.contains("torch") && report.contains("Installing PyTorch"), "{report}");
        assert!(note_progress(&watch, "[TAURI:DL_DONE] torch"));
        let after = watch
            .lock()
            .unwrap()
            .due_report(later + Duration::from_secs(600))
            .expect("a later report");
        assert!(!after.contains("torch"), "{after}");
    }

    #[test]
    fn the_wait_loop_reports_a_silence_then_stops_the_child_at_the_backstop() {
        let watch = Mutex::new(ProgressWatch::with_backstop(Instant::now() - 13 * HOUR, 12 * HOUR));
        let (mut reports, mut stops) = (Vec::new(), 0);
        let error = wait_with_watchdog(
            &watch,
            || Ok(None),
            |line| reports.push(line.to_string()),
            || stops += 1,
        )
        .expect_err("the backstop must end the wait");
        assert!(error.contains("12 h"), "{error}");
        assert_eq!(stops, 1);
        assert_eq!(reports.len(), 1, "{reports:?}");
        assert!(reports[0].contains("still working"), "{reports:?}");
    }

    #[test]
    fn the_wait_loop_returns_the_childs_own_exit_without_stopping_it() {
        let watch = Mutex::new(watch(Instant::now()));
        let mut stops = 0;
        let status = exit_status(3);
        let (returned, intentional) = wait_with_watchdog(
            &watch,
            || Ok(Some((status, false))),
            |_| panic!("an install that exits at once must not report"),
            || stops += 1,
        )
        .expect("the child's exit is the result");
        assert_eq!(returned.code(), Some(3));
        assert!(!intentional);
        assert_eq!(stops, 0);
    }
}
