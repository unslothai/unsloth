# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Best-effort bytecode compile of the Studio environment, run by setup.sh / setup.ps1.

uv installs without .pyc, so the first launch compiled every module `import main` loads (about
twice the CPU of a later launch). setup runs this after a dependency pass changed the venv.

Best effort, never part of the install's outcome:
- only ADDS missing .pyc: a source that already has one (current or not) is left alone, so no
  existing file is ever modified. A re-run over a compiled tree finds nothing to do and writes nothing;
- exit 0 on every error, an unwritable directory, a missing interpreter or a timeout; Ctrl+C is
  passed on as SIGINT, as for any other setup step;
- prints nothing (setup prints the one progress line);
- skipped below _MIN_FREE_BYTES free, so it cannot take the space a later install step needs;
- a timeout (or Ctrl+C) kills the whole worker tree and removes exactly what this run created: the
  .pyc it was asked to write, their temp files (`<name>.pyc.<digits>`) and the `__pycache__`
  directories it made. The environment is then as it was before the step.

Skipped when UV_COMPILE_BYTECODE or UNSLOTH_STUDIO_COMPILE_BYTECODE is set to a false value.
UNSLOTH_STUDIO_COMPILE_BYTECODE_TIMEOUT bounds the whole compile in seconds (default 240, under the
desktop installer's 5 minute silence report).

Usage: python -I precompile_bytecode.py [DIR ...]   (no DIR: this interpreter's site-packages plus
the Studio packages that live outside it, i.e. an editable checkout)
"""

import os
import re
import shutil
import subprocess
import sys
import sysconfig

_FALSE = ("0", "false", "no", "off")
_DEFAULT_TIMEOUT = 240.0
_MIN_FREE_BYTES = 4 * 1024**3
_PACKAGES = ("studio", "unsloth", "unsloth_cli")
# Never walked inside a checkout: the Node tree, the desktop sources, the test suites.
_CHECKOUT_SKIP_DIRS = ("node_modules", "frontend", "src-tauri", "tests")
# importlib's _write_atomic writes "<path>.<id(path)>" and renames it over <path>.
_PARTIAL_PYC = re.compile(r"\.pyc\.\d+$")
_CHILD_FLAG = "--compile-child"


def _disabled():
    for name in ("UV_COMPILE_BYTECODE", "UNSLOTH_STUDIO_COMPILE_BYTECODE"):
        if os.environ.get(name, "").strip().lower() in _FALSE:
            return True
    return False


def _timeout():
    try:
        value = float(os.environ.get("UNSLOTH_STUDIO_COMPILE_BYTECODE_TIMEOUT", ""))
    except ValueError:
        return _DEFAULT_TIMEOUT
    return value if value > 0 else _DEFAULT_TIMEOUT


def _same(a, b):
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except Exception:
        return False


def _inside(path, roots):
    path = os.path.normcase(os.path.realpath(path))
    for root in roots:
        root = os.path.normcase(os.path.realpath(root))
        if path == root or path.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


def _enough_space(dirs):
    for path in dirs:
        try:
            if shutil.disk_usage(path).free < _MIN_FREE_BYTES:
                return False
        except Exception:
            return False
    return True


def default_dirs():
    """(site dirs, checkout dirs) for this interpreter."""
    paths = sysconfig.get_paths()
    site = []
    for key in ("purelib", "platlib"):
        path = paths.get(key)
        if path and os.path.isdir(path) and not _inside(path, site):
            site.append(path)
    checkout = []
    import importlib.util

    # Only what the venv itself resolves (site-packages, its .pth files, an editable finder): never
    # PYTHONPATH, the current directory or this script's directory, which can hold unrelated trees.
    blocked = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    blocked += [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if p and not any(_same(p, b) for b in blocked)]
    try:
        for name in _PACKAGES:
            try:
                spec = importlib.util.find_spec(name)
            except Exception:
                continue
            for path in list(getattr(spec, "submodule_search_locations", None) or []):
                if os.path.isfile(os.path.join(path, "__init__.py")) and not _inside(
                    path, site + checkout
                ):
                    checkout.append(path)
    finally:
        sys.path[:] = saved
    return site, checkout


def plan(site, checkout):
    """(sources lacking a .pyc, their .pyc paths, __pycache__ dirs that do not exist yet).

    Symlinked directories are not followed, as compileall does not follow them either."""
    import importlib.util

    sources, targets, new_dirs = [], [], set()

    def add(directory, names):
        for name in names:
            if not name.endswith(".py"):
                continue
            source = os.path.join(directory, name)
            try:
                target = importlib.util.cache_from_source(source)
            except Exception:
                continue
            if os.path.lexists(target):
                continue
            sources.append(source)
            targets.append(target)
            cache = os.path.dirname(target)
            if not os.path.isdir(cache):
                new_dirs.add(cache)

    for root in site:
        for current, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d != "__pycache__"]
            add(current, files)
    for root in checkout:
        for current, subdirs, files in os.walk(root):
            subdirs[:] = [
                d
                for d in subdirs
                if d != "__pycache__" and not d.startswith(".") and d not in _CHECKOUT_SKIP_DIRS
            ]
            add(current, files)
    return sources, targets, new_dirs


def _child_command(site, checkout):
    """The child re-plans from the same roots (a 0.15 s walk) rather than take 25k paths on stdin."""
    return (
        [sys.executable, "-I", os.path.abspath(__file__), _CHILD_FLAG]
        + ["site:" + d for d in site]
        + ["checkout:" + d for d in checkout]
    )


def _compile_child(args):
    """The worker side: compile what plan() finds, on all cores; never raises."""
    import functools
    import py_compile

    site = [a[len("site:") :] for a in args if a.startswith("site:")]
    checkout = [a[len("checkout:") :] for a in args if a.startswith("checkout:")]
    sources = plan(site, checkout)[0]
    job = functools.partial(py_compile.compile, doraise = False, quiet = 2)
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as pool:
            for _ in pool.map(job, sources, chunksize = 64):
                pass
        return
    except Exception:
        pass
    # No working process pool here (no sem_open, for one): one by one.
    for source in sources:
        try:
            job(source)
        except Exception:
            pass


def undo(targets, new_dirs):
    """Remove exactly what a cut-short run created."""
    caches = {}
    for target in targets:
        caches.setdefault(os.path.dirname(target), set()).add(os.path.basename(target))
        try:
            os.remove(target)
        except Exception:
            pass
    for cache, names in caches.items():
        try:
            leftovers = os.listdir(cache)
        except Exception:
            continue
        for name in leftovers:
            match = _PARTIAL_PYC.search(name)
            if match and name[: match.start() + 4] in names:
                try:
                    os.remove(os.path.join(cache, name))
                except Exception:
                    pass
    for cache in sorted(new_dirs, key = len, reverse = True):
        try:
            os.rmdir(cache)  # only when empty, i.e. nothing else wrote there meanwhile
        except Exception:
            pass


def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 30,
            )
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout = 30)
    except Exception:
        pass


def run(site, checkout, timeout):
    """Compile in a child process tree under one deadline. True when it finished in time."""
    kwargs = {}
    if os.name != "nt":
        # Its own process group, so a timeout can kill the pool workers too.
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(
            _child_command(site, checkout),
            stdin = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            **kwargs,
        )
    except Exception:
        return True  # nothing started, nothing written
    try:
        proc.wait(timeout = timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        return False
    except BaseException:
        _kill_tree(proc)
        raise
    return True


def _reraise_sigint():
    """Die of SIGINT, so the calling shell stops on Ctrl+C exactly as it does for any step."""
    if os.name != "nt":
        import signal
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            pass
    os._exit(130)


def main(argv):
    targets, new_dirs = [], set()
    try:
        if _disabled():
            return 0
        if argv:
            site, checkout = [d for d in argv if os.path.isdir(d)], []
        else:
            site, checkout = default_dirs()
        if not site + checkout or not _enough_space(site + checkout):
            return 0
        sources, targets, new_dirs = plan(site, checkout)
        if not sources:
            return 0
        if not run(site, checkout, _timeout()):
            undo(targets, new_dirs)
    except KeyboardInterrupt:
        try:
            undo(targets, new_dirs)
        except BaseException:
            pass
        _reraise_sigint()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == [_CHILD_FLAG]:
        try:
            _compile_child(sys.argv[2:])
        except BaseException:
            pass
        sys.exit(0)
    sys.exit(main(sys.argv[1:]))
