#!/opt/unsloth-venv/bin/python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""execute notebooks headlessly with one transformers version active per kernel."""

import argparse, ctypes, json, os, re, select, shutil, stat, struct, subprocess, sys, tempfile, threading, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import unsloth_nb_compat as compat
except Exception:
    compat = None

_MODEL_RE = re.compile(r"""from_pretrained\(\s*['"]([^'"]+)['"]""")
_MODEL_NAME_RE = re.compile(r"""model_name\s*=\s*['"]([^'"]+)['"]""")

# share the site-packages scanner so this path and the IPython hook select the same sidecar.
if compat is not None:
    _PIN_RE = compat._PIN_RE
    _INSTALL_RE = compat._INSTALL_RE
    _strip_comment = compat._strip_comment
    _live_source = compat._live_source
    _install_lines = compat._install_lines
    _pin_from = compat.pin_from
else:
    # compat is also what turns a scan result into a sidecar, so with it gone there is
    # nothing either half of the scan could select (see main, where sidecar stays None
    # and the tier lookup is skipped). Degrade to no scan rather than keep a second
    # copy of the rules here, which is the drift this move exists to prevent.
    _PIN_RE = _INSTALL_RE = None
    _strip_comment = _live_source = _install_lines = _pin_from = None


DEFAULT_FETCH_TIMEOUT = int(os.environ.get("UNSLOTH_NOTEBOOK_FETCH_TIMEOUT", "60") or 60)


def _load(path_or_url, fetch_timeout = None):
    """Parsed notebook. A URL is fetched with a socket timeout: --timeout only ever
    reached nbconvert, so a host that accepted the connection and then went quiet hung
    the run before a single cell had executed. This bounds each blocking socket
    operation, not the total transfer, which is what that failure needs; a server
    trickling bytes forever is a different problem and not one seen here."""
    if path_or_url.startswith(("http://", "https://")):
        if fetch_timeout is None:
            fetch_timeout = DEFAULT_FETCH_TIMEOUT
        with urllib.request.urlopen(  # nosec - user-provided nb
            path_or_url, timeout = fetch_timeout
        ) as r:
            data = r.read().decode()
        return json.loads(data)
    with open(path_or_url) as f:
        return json.load(f)


def _scan(nb):
    """(pinned_transformers, first_model_name); dead code must not count, as above."""
    pin = model = None
    if compat is None:
        return pin, model
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = _live_source("".join(cell.get("source", [])))
        if pin is None:
            # the shared helper, not _PIN_RE directly: the pattern now matches any
            # requirement name and the PEP 503 comparison that picks transformers out
            # of it lives with it, so reading the match here would drop that half
            pin = _pin_from(src)
        if model is None:
            m = _MODEL_RE.search(src) or _MODEL_NAME_RE.search(src)
            if m:
                model = m.group(1)
    return pin, model


def _makedirs_as_host(path):
    """Create `path` owned by the nearest existing ancestor. mkdir(2) uses the CALLER's
    uid/gid and only setgid carries down, so a new `--out sub/dir/` would be root-owned
    and _stage_metadata would then give the OUTPUT that owner too."""
    path = os.path.abspath(path)
    missing = []
    probe = path
    while not os.path.isdir(probe):
        missing.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    os.makedirs(path, exist_ok = True)
    if not missing:
        return
    try:
        anchor = os.stat(probe)
    except OSError:
        return
    for created in reversed(missing):
        try:
            os.chown(created, anchor.st_uid, anchor.st_gid)
        except (OSError, AttributeError):
            pass


def _stage_metadata(staged, dest):
    """Give the staged output the metadata the destination must end up with: mkstemp()
    creates 0600, nbconvert truncates that same inode, and os.replace carries it onto
    the destination. Best effort."""
    try:
        st = os.stat(dest)
    except OSError:
        # new output: the umask-derived mode a plain write would have produced
        try:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(staged, 0o666 & ~umask)
        except OSError:
            pass
        try:
            _dir = os.stat(os.path.dirname(os.path.abspath(dest)) or ".")
            os.chown(staged, _dir.st_uid, _dir.st_gid)
        except (OSError, AttributeError):
            pass
        return
    try:
        os.chmod(staged, stat.S_IMODE(st.st_mode))
    except OSError:
        pass
    try:
        os.chown(staged, st.st_uid, st.st_gid)
    except (OSError, AttributeError):
        pass


def _open_url_download(url):
    name = os.path.basename(urllib.parse.unquote(urllib.parse.urlsplit(url).path)) or "notebook"
    stem = name[: -len(".ipynb")] if name.endswith(".ipynb") else name
    n = 0
    while True:
        path = os.path.abspath(f"{stem}-{n}.ipynb" if n else f"{stem}.ipynb")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            n += 1
            continue
        try:
            parent = os.stat(os.path.dirname(path))
            os.fchown(fd, parent.st_uid, parent.st_gid)
        except (OSError, AttributeError):
            pass
        return path, fd


def _host_run_ids():
    uid = os.environ.pop("UNSLOTH_RUN_UID", None)
    gid = os.environ.pop("UNSLOTH_RUN_GID", None)
    if uid is None and gid is None:
        return None
    if uid is None or gid is None or not uid.isdigit() or not gid.isdigit():
        raise SystemExit("UNSLOTH_RUN_UID and UNSLOTH_RUN_GID must be non-negative integers")
    return int(uid), int(gid)


_IN_EVENT = struct.Struct("iIII")
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_IGNORED = 0x00008000
_IN_Q_OVERFLOW = 0x00004000
_IN_ISDIR = 0x40000000


class _OwnershipMonitor:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.device = os.stat(self.root).st_dev
        self.fd = -1
        self.watches = {}
        self.affected = set()
        self.recursive = set()
        self.failed = False
        self.overflowed = False
        self.stop_event = threading.Event()
        self.thread = None
        self.libc = ctypes.CDLL(None, use_errno = True)

    def _remember(self, path):
        path = os.path.abspath(path)
        prefix = self.root + os.sep
        if path != self.root and not path.startswith(prefix):
            return
        while path != self.root:
            self.affected.add(path)
            path = os.path.dirname(path)

    def _add_tree(self, root):
        try:
            same_device = os.stat(root, follow_symlinks = False).st_dev == self.device
        except OSError:
            self.failed = True
            return
        pending = [(root, same_device)]
        mask = _IN_ATTRIB | _IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_CREATE
        while pending:
            current, recurse = pending.pop()
            wd = self.libc.inotify_add_watch(self.fd, os.fsencode(current), mask)
            if wd < 0:
                self.failed = True
            else:
                self.watches[wd] = current
            if not recurse:
                continue
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if not entry.is_dir(follow_symlinks = False):
                            continue
                        try:
                            child_device = entry.stat(follow_symlinks = False).st_dev
                        except OSError:
                            self.failed = True
                            continue
                        pending.append((entry.path, child_device == self.device))
            except OSError:
                self.failed = True

    def start(self):
        try:
            self.libc.inotify_init1.argtypes = [ctypes.c_int]
            self.libc.inotify_init1.restype = ctypes.c_int
            self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            self.libc.inotify_add_watch.restype = ctypes.c_int
            self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        except (AttributeError, OSError):
            return False
        if self.fd < 0:
            return False
        self._add_tree(self.root)
        if self.failed or not self.watches:
            os.close(self.fd)
            self.fd = -1
            self.watches.clear()
            return False
        self.thread = threading.Thread(target = self._run, daemon = True)
        self.thread.start()
        return True

    def _drain(self):
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            except OSError:
                self.failed = True
                return
            if not data:
                return
            offset = 0
            while offset + _IN_EVENT.size <= len(data):
                wd, mask, _cookie, length = _IN_EVENT.unpack_from(data, offset)
                offset += _IN_EVENT.size
                raw_name = data[offset : offset + length].split(b"\0", 1)[0]
                offset += length
                if mask & _IN_Q_OVERFLOW:
                    self.failed = True
                    self.overflowed = True
                    continue
                parent = self.watches.get(wd)
                if mask & _IN_IGNORED:
                    self.watches.pop(wd, None)
                if parent is None or not raw_name:
                    continue
                path = os.path.join(parent, os.fsdecode(raw_name))
                self._remember(path)
                if mask & _IN_ISDIR and mask & (_IN_CREATE | _IN_MOVED_TO):
                    self.recursive.add(path)
                    self._add_tree(path)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.1)
            except OSError:
                self.failed = True
                return
            if ready:
                self._drain()

    def stop(self):
        if self.fd < 0:
            return set(), set()
        self.stop_event.set()
        self.thread.join()
        self._drain()
        os.close(self.fd)
        self.fd = -1
        return self.affected, self.recursive


def _restore_output_ownership(affected, recursive, uid, gid):
    paths = set(affected)
    for root in recursive:
        for parent, dirs, files in os.walk(root, followlinks = False):
            paths.add(parent)
            paths.update(os.path.join(parent, name) for name in dirs + files)
    for path in sorted(paths, key = len):
        try:
            if os.lstat(path).st_uid == 0:
                os.chown(path, uid, gid, follow_symlinks = False)
        except (OSError, TypeError):
            pass


def main():
    ap = argparse.ArgumentParser(prog = "unsloth-run")
    ap.add_argument("notebook")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type = int, default = 3600)
    ap.add_argument(
        "--fetch-timeout",
        dest = "fetch_timeout",
        type = int,
        default = DEFAULT_FETCH_TIMEOUT,
        help = "seconds a URL fetch may stall before giving up (default 60)",
    )
    ap.add_argument("--transformers", dest = "tf")
    args = ap.parse_args()

    nb = _load(args.notebook, fetch_timeout = args.fetch_timeout)
    pin, model = _scan(nb)
    want = args.tf or pin or (compat.tier_for_model(model) if compat else None)
    sidecar = compat.sidecar_for(want) if (compat and want) else None
    host_ids = _host_run_ids()

    tmp_files = []
    publish_from = None
    if args.out:
        out_path = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_path) or "."
        _makedirs_as_host(out_dir)
        if args.notebook.startswith(("http://", "https://")):
            # keep URL inputs beside --out so relative artifacts land in the output directory
            fd, src_path = tempfile.mkstemp(prefix = ".unsloth-run-in-", suffix = ".ipynb", dir = out_dir)
            with os.fdopen(fd, "w") as f:
                json.dump(nb, f)
            tmp_files.append(src_path)
        else:
            # nbconvert uses the input directory as the kernel cwd, so keep local inputs in place
            src_path = args.notebook
        fd, publish_from = tempfile.mkstemp(
            prefix = ".unsloth-run-out-", suffix = ".ipynb", dir = out_dir
        )
        os.close(fd)
        tmp_files.append(publish_from)
    elif args.notebook.startswith(("http://", "https://")):
        src_path, fd = _open_url_download(args.notebook)
        with os.fdopen(fd, "w") as f:
            json.dump(nb, f)
        out_path = src_path
    else:
        src_path = args.notebook
        out_path = src_path

    kernel_dir = os.path.dirname(os.path.abspath(src_path)) or "."
    ownership_monitor = _OwnershipMonitor(kernel_dir) if host_ids is not None else None
    if ownership_monitor is not None and not ownership_monitor.start():
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass
        raise SystemExit("unsloth-run could not establish a complete output ownership monitor")

    env = dict(os.environ)
    env["UNSLOTH_NB_SHIM"] = "1"
    # nested runs need a fresh marker to avoid overwriting or reusing the caller's transformers pin
    fd, marker = tempfile.mkstemp(prefix = ".unsloth-run-tfmarker-")
    os.close(fd)
    env["UNSLOTH_NB_TF_MARKER"] = marker
    tmp_files.append(marker)
    if want:
        open(marker, "w").write(want)
    if sidecar:
        env["PYTHONPATH"] = sidecar + os.pathsep + env.get("PYTHONPATH", "")
        print(f"[unsloth-run] transformers {want} -> sidecar {sidecar}")
    elif want:
        print(f"[unsloth-run] transformers {want}: no sidecar (using base venv's newest)")
    else:
        print("[unsloth-run] no transformers pin/model tier detected; using base venv")

    nbconvert_out = publish_from if publish_from is not None else out_path
    cmd = [
        "/opt/unsloth-venv/bin/jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={args.timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        src_path,
        "--output",
        os.path.basename(nbconvert_out),
        "--output-dir",
        os.path.dirname(os.path.abspath(nbconvert_out)) or ".",
    ]
    print(
        "[unsloth-run] executing:",
        os.path.basename(args.notebook.split("?")[0]) if args.out else os.path.basename(src_path),
    )
    ownership_overflow = False
    try:
        rc = subprocess.call(cmd, env = env)
        if rc == 0 and publish_from is not None:
            _stage_metadata(publish_from, out_path)
            try:
                os.replace(publish_from, out_path)
            except OSError:
                # bind-mounted output files return EBUSY from rename(2) and require writing the mounted inode
                try:
                    with open(publish_from, "rb") as staged, open(out_path, "wb") as live:
                        shutil.copyfileobj(staged, live)
                except OSError:
                    # preserve the result because the run may have taken hours
                    if publish_from in tmp_files:
                        tmp_files.remove(publish_from)
                    print(
                        f"[unsloth-run] could not publish to {out_path}; "
                        f"the executed notebook is at {publish_from}",
                        file = sys.stderr,
                    )
                    raise
    finally:
        if ownership_monitor is not None:
            affected, recursive = ownership_monitor.stop()
            _restore_output_ownership(affected, recursive, *host_ids)
            ownership_overflow = ownership_monitor.overflowed
            if ownership_monitor.failed:
                print(
                    "[unsloth-run] some output ownership events could not be monitored",
                    file = sys.stderr,
                )
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass
    if ownership_overflow:
        raise SystemExit("unsloth-run output ownership monitoring overflowed")
    sys.exit(rc)


if __name__ == "__main__":
    main()
