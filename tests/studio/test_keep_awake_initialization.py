# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Exercise startup synchronization without building Tauri or touching OS inhibitors."""

from pathlib import Path
import subprocess


def test_commands_wait_for_keep_awake_restoration(tmp_path):
    source = (Path(__file__).resolve().parents[2] / "studio/src-tauri/src/keep_awake.rs").read_text(
        encoding = "utf-8"
    )
    start = source.find("fn wait_for_initialization()")
    helper = (
        source[start : source.index("\n}\n", start) + 3]
        if start >= 0
        else "fn wait_for_initialization() {}"
    )
    program = r"""
use std::sync::{Mutex, PoisonError, Arc, atomic::{AtomicBool, Ordering}};
static INITIALIZATION: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);
macro_rules! warn { ($($arg:tt)*) => { eprintln!($($arg)*) }; }
HELPER
fn main() {
    let held = Arc::new(AtomicBool::new(false));
    let restored = held.clone();
    *INITIALIZATION.lock().unwrap() = Some(std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        restored.store(true, Ordering::SeqCst);
    }));
    let readers: Vec<_> = (0..2).map(|_| {
        let held = held.clone();
        std::thread::spawn(move || {
            wait_for_initialization();
            assert!(held.load(Ordering::SeqCst), "read before startup restoration");
        })
    }).collect();
    for reader in readers { reader.join().unwrap(); }
    wait_for_initialization();
}
""".replace("HELPER", helper)
    rust = tmp_path / "startup.rs"
    rust.write_text(program)
    binary = tmp_path / "startup"
    subprocess.run(["rustc", "--edition=2021", str(rust), "-o", str(binary)], check = True)
    subprocess.run([str(binary)], check = True)
    getter = source.split("pub async fn get_keep_awake()", 1)[1].split("#[tauri::command]", 1)[0]
    setter = source.split("pub async fn set_keep_awake", 1)[1].split("#[cfg(windows)]", 1)[0]
    assert getter.index("wait_for_initialization();") < getter.index("Some(is_held())")
    assert setter.index("wait_for_initialization();") < setter.index("apply(enabled)")
