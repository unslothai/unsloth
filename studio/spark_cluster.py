# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA DGX Spark cluster detection and setup.

Two cabled Sparks are two independent hosts: ``device_count()`` is 1 on each, and the cable
is a RoCEv2 link. ``is_dgx_spark()`` gates every entry point; install.sh carries a twin.
The ConnectX-7 exposes each physical QSFP port as TWO PCIe functions (``rocep1s0f0`` and
``roceP2p1s0f0`` are the same port), and one subnet drives only one function, so each rail
needs its own /24 to reach the full link.
"""

from __future__ import annotations

import getpass
import glob
import ipaddress
import json
import os
import os.path as osp
import platform
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DGX_RELEASE = "/etc/dgx-release"
_DMI_PRODUCT = "/sys/class/dmi/id/product_name"
_SPARK_RE = re.compile(r"dgx[_ -]*spark", re.IGNORECASE)

_IS_SPARK_CACHE: Optional[bool] = None


def is_dgx_spark() -> bool:
    """True only on an NVIDIA DGX Spark. Cached; short-circuits before any file is opened."""
    global _IS_SPARK_CACHE
    if _IS_SPARK_CACHE is not None:
        return _IS_SPARK_CACHE

    result = False
    if platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"):
        for path in (_DGX_RELEASE, _DMI_PRODUCT):
            try:
                # Capped so a bad mount cannot make the gate expensive.
                with open(path, "r", encoding = "utf-8", errors = "replace") as handle:
                    if _SPARK_RE.search(handle.read(4096)):
                        result = True
                        break
            except OSError:
                continue

    _IS_SPARK_CACHE = result
    return result


# Callers match on this exact string; keep it single-sourced.
NOT_A_SPARK = "This machine is not a DGX Spark; nothing to do."


def _ssh_user() -> str:
    """The login to reach the peer as: `provision` mirrors the install as the same account."""
    for var in ("USER", "USERNAME", "LOGNAME"):
        value = os.environ.get(var)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:
        return "nvidia"


_IB_ROOT = Path("/sys/class/infiniband")
_NET_ROOT = Path("/sys/class/net")


def _int_or_none(text: str) -> Optional[int]:
    return int(text) if text.isdigit() else None


def _read(path: Path, limit: int = 256) -> str:
    try:
        with open(path, "r", encoding = "utf-8", errors = "replace") as handle:
            return handle.read(limit).strip()
    except OSError:
        return ""


def _rail_sort_key(name: str) -> Tuple[int, int, str]:
    """Order by (port, PCIe function): plain sorting puts ``roceP2p1s0f0`` (function 2) first,
    since uppercase P sorts first, and hands every caller the wrong primary rail."""
    match = re.search(r"s0f(\d+)$", name)
    port = int(match.group(1)) if match else 9
    function = 2 if name.startswith("roceP2p") else 1
    return (port, function, name)


def local_rails() -> List[Dict[str, Any]]:
    """Every RoCE device here: netdev, link state, IPv4s. Sysfs only, no fork. Usable =
    IB port ACTIVE *and* netdev carrier: the cable is seated and trained at the far end."""
    rails: List[Dict[str, Any]] = []
    if not _IB_ROOT.is_dir():
        return rails

    for dev in sorted(_IB_ROOT.iterdir(), key = lambda p: _rail_sort_key(p.name)):
        port = dev / "ports" / "1"
        state = _read(port / "state")
        phys = _read(port / "phys_state")
        netdev = ""
        gid_attr = dev / "ports" / "1" / "gid_attrs" / "ndevs" / "0"
        netdev = _read(gid_attr)
        if not netdev:
            try:
                dev_pci = (dev / "device").resolve()
                for candidate in _NET_ROOT.iterdir():
                    try:
                        if (candidate / "device").resolve() == dev_pci:
                            netdev = candidate.name
                            break
                    except OSError:
                        continue
            except OSError:
                pass

        carrier = _read(_NET_ROOT / netdev / "carrier") if netdev else ""
        mtu = _read(_NET_ROOT / netdev / "mtu") if netdev else ""
        rails.append(
            {
                "ib_device": dev.name,
                "netdev": netdev,
                "carrier_up_count": _int_or_none(_read(_NET_ROOT / netdev / "carrier_up_count"))
                if netdev
                else None,
                "ib_active": "ACTIVE" in state.upper(),
                "link_up": "LINKUP" in phys.upper().replace(" ", ""),
                "carrier": carrier == "1",
                "mtu": int(mtu) if mtu.isdigit() else None,
                "ipv4": _netdev_ipv4(netdev) if netdev else [],
            }
        )
    return rails


def _netdev_ipv4(netdev: str) -> List[str]:
    ip_bin = shutil.which("ip")
    if not ip_bin:
        return []
    try:
        out = subprocess.run(
            [ip_bin, "-4", "-o", "addr", "show", "dev", netdev],
            capture_output = True,
            text = True,
            timeout = 5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)/", out)


def cabled_rails() -> List[Dict[str, Any]]:
    return [r for r in local_rails() if r["ib_active"] and r["carrier"]]


_MDNS_TIMEOUT = 3.0
# Bounded: `status` runs while a job is starting. One TCP SYN to sshd per peer, no fork.
_PEER_PROBE_PORT = 22
_PEER_PROBE_TIMEOUT = 0.75


def _ipv4_sort_key(address: str) -> Tuple[int, int, int, int, int]:
    """Numeric IPv4 ordering, so .9 sorts before .10: lexical ordering silently reshuffles
    rank assignment once a cluster passes ten nodes. Non-IPv4 sorts last, never dropped."""
    parts = address.split(".")
    if len(parts) == 4 and all(p.isdigit() and len(p) <= 3 for p in parts):
        a, b, c, d = (int(p) for p in parts)
        if max(a, b, c, d) <= 255:
            return (0, a, b, c, d)
    return (1, 0, 0, 0, 0)


def peer_reachable(
    address: str,
    port: int = _PEER_PROBE_PORT,
    timeout: float = _PEER_PROBE_TIMEOUT,
) -> Optional[bool]:
    """Is this peer answering? A refused connection still proves the host is up, so it
    counts as reachable. ``None`` means could not tell, never "down"."""
    if not address:
        return None
    try:
        conn = socket.create_connection((address, port), timeout = timeout)
    except (socket.timeout, TimeoutError):
        return False
    except ConnectionRefusedError:
        return True
    except OSError:
        return False
    except Exception:
        return None
    try:
        conn.close()
    except OSError:
        pass
    return True


def discover_peers(timeout: float = _MDNS_TIMEOUT, check_reachable: bool = False) -> Dict[str, Any]:
    """A cabled RoCE rail proves a cable but cannot name the peer; mDNS is the reverse. Returns
    immediately off a Spark, so avahi-browse never runs on an ordinary Linux box."""
    if not is_dgx_spark():
        return {
            "is_spark": False,
            "cabled_rails": [],
            "cable_present": False,
            "configured": [],
            "mdns_peers": [],
            "peers": [],
            "n_peers": 0,
            "n_nodes": 1,
            "note": "not a DGX Spark; no discovery attempted",
        }
    rails = cabled_rails()
    result: Dict[str, Any] = {
        "is_spark": True,
        "cabled_rails": rails,
        "cable_present": bool(rails),
        "configured": [r for r in rails if r["ipv4"]],
        "mdns_peers": [],
    }
    if timeout > 0:
        result["mdns_peers"] = _mdns_spark_peers(timeout)
    peers = merge_peers(result["mdns_peers"], check_reachable = check_reachable)
    result["peers"] = peers
    result["n_peers"] = len(peers)
    # Nodes, not peers: the planner counts nodes.
    result["n_nodes"] = len(peers) + 1
    return result


def _planned_peers(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """The peers `setup` addressed, read back from the rail plans it saved.

    Setup writes `n_nodes`, `peer_rails` and `other_rails` but never a `peers` list, so a
    cluster configured as `setup --nodes 3 --switched` -- the switched fabric where mDNS does
    not answer, which is the case the config is meant to cover -- came back from discovery as
    two nodes and was planned and sized as two. The addresses are already here; only the key
    they were looked up under was missing."""
    out: List[Dict[str, str]] = []
    for plan in [config.get("peer_rails")] + list(config.get("other_rails") or []):
        if not isinstance(plan, list) or not plan:
            continue
        # One node per plan, from its first rail. The others are the same host on its second
        # PCIe function and would otherwise each count as another Spark.
        address = str((plan[0] or {}).get("address") or "")
        if address:
            # No hostname: `merge_peers` keys on the first dotted field of one, which would
            # make every planned address collapse into a single peer called `192`.
            out.append({"hostname": "", "address": address, "source": "config-rails"})
    return out


def configured_peers() -> List[Dict[str, str]]:
    """Peers pinned in the saved config. mDNS does not survive a switched fabric, so the
    config is a first-class source, not a fallback. Malformed entries are skipped."""
    out: List[Dict[str, str]] = []
    config = load_config()
    raw = config.get("peers")
    if not isinstance(raw, list) or not raw:
        return _planned_peers(config)
    for entry in raw:
        if isinstance(entry, str):
            out.append({"hostname": entry, "address": entry, "source": "config"})
        elif isinstance(entry, dict):
            address = str(entry.get("address") or entry.get("ip") or "")
            hostname = str(entry.get("hostname") or address)
            if address or hostname:
                out.append(
                    {"hostname": hostname, "address": address or hostname, "source": "config"}
                )
    return out


def merge_peers(
    mdns: Optional[List[Dict[str, str]]] = None, check_reachable: bool = False
) -> List[Dict[str, Any]]:
    """Every known peer, deduplicated and ordered by (address, hostname) so every node
    computes the same list: ``index`` is only usable as a node rank because of that."""
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in list(mdns or []) + configured_peers():
        hostname = str(entry.get("hostname", ""))
        address = str(entry.get("address", ""))
        key = hostname.split(".")[0].lower() or address
        if not key:
            continue
        prev = merged.get(key)
        if prev is None:
            merged[key] = {
                "hostname": hostname or address,
                "short": key,
                "address": address,
                "source": entry.get("source", "mdns"),
            }
        elif not prev["address"] or (":" in prev["address"] and ":" not in address):
            prev["address"] = address or prev["address"]
    # One host reached two ways is one node. A peer named by mDNS and the same peer named by
    # the saved rail plan key differently -- a hostname against an address -- and counting
    # both would report a two-Spark pair as three nodes and plan for hardware that is not
    # there. The named entry wins, since it carries the hostname.
    by_address: Dict[str, Dict[str, Any]] = {}
    for peer in sorted(merged.values(), key = lambda d: d["short"] == d["address"]):
        by_address.setdefault(peer["address"] or peer["short"], peer)
    peers = sorted(by_address.values(), key = lambda d: (_ipv4_sort_key(d["address"]), d["short"]))
    for index, peer in enumerate(peers):
        # This node is node 0; peers occupy 1..N-1.
        peer["index"] = index + 1
        peer["reachable"] = peer_reachable(peer["address"]) if check_reachable else None
    return peers


def _mdns_spark_peers(timeout: float) -> List[Dict[str, str]]:
    browse = shutil.which("avahi-browse")
    if not browse:
        return []
    me = socket.gethostname().split(".")[0].lower()
    try:
        proc = subprocess.run(
            [browse, "-a", "-t", "-r", "-p", "-k"],
            capture_output = True,
            text = True,
            timeout = timeout + 2,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    seen: Dict[str, Dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("="):
            continue
        parts = line.split(";")
        if len(parts) < 8:
            continue
        host, addr = parts[6], parts[7]
        short = host.split(".")[0].lower()
        if not short.startswith("spark-") or short == me:
            continue
        # Prefer a routable IPv4 over a link-local IPv6 for the same host.
        prev = seen.get(short)
        if prev is None or (":" in prev["address"] and ":" not in addr):
            seen[short] = {"hostname": host, "address": addr, "source": "mdns"}
    return sorted(seen.values(), key = lambda d: (_ipv4_sort_key(d["address"]), d["hostname"]))


# The QSFP hot-plug throttle cannot be inferred from sysfs. `carrier_up_count` is NOT a
# signal for it: a node at count=7 measured a full 97.97 Gb/s. Only a measurement settles it.
HOTPLUG_NOTE = (
    "If the link measures far below ~98 Gb/s per rail, the usual cause is the QSFP "
    "cable having been connected after boot, which can leave the ConnectX-7 throttled. "
    "Reboot both Sparks with the cable already plugged in, then leave the cabling alone."
)


def link_carrier_events(rails: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Optional[int]]:
    """Per-rail carrier_up_count, reported as a FACT and never a verdict; see HOTPLUG_NOTE."""
    rails = rails if rails is not None else cabled_rails()
    return {r["ib_device"]: r.get("carrier_up_count") for r in rails}


def _studio_root() -> Path:
    for var in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser()
    return Path.home() / ".unsloth" / "studio"


def config_path() -> Path:
    return _studio_root() / "spark_cluster.json"


def load_config() -> Dict[str, Any]:
    try:
        with open(config_path(), "r", encoding = "utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(config: Dict[str, Any]) -> bool:
    """False when the plan did not reach disk. Setup must not report success on that: the
    cluster stays unconfigured and the next invocation has forgotten the plan."""
    path = config_path()
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding = "utf-8") as handle:
            json.dump(config, handle, indent = 2, sort_keys = True)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        print(f"  could not save the cluster plan to {path}: {exc}")
        return False
    return True


def cluster_state() -> str:
    """``not_spark`` | ``no_cable`` | ``unconfigured`` | ``configured``. ``configured``
    needs a config AND rails still carrying IPv4, which is what makes a re-run safe."""
    if not is_dgx_spark():
        return "not_spark"
    rails = cabled_rails()
    if not rails:
        return "no_cable"
    config = load_config()
    # EVERY cabled rail, not any of them. `nccl_env` hands NCCL all of them through
    # NCCL_IB_MERGE_NICS, so a netplan that reached one PCIe function and not the other is
    # not a configured cluster: it is a job that fails, or runs at half the link. `any`
    # declared that state finished and made every later install and `spark up` skip setup.
    if config.get("enabled") and rails and all(r["ipv4"] for r in rails):
        return "configured"
    return "unconfigured"


def cluster_config_problems() -> List[str]:
    """Where the machine's rails and the saved plan disagree. Empty when they agree, and
    empty when there is no plan to compare against -- this reports drift, not absence."""
    problems: List[str] = []
    config = load_config()
    if not config.get("enabled"):
        return problems
    planned = {
        str(entry.get("netdev") or ""): entry
        for entry in (config.get("rails") or [])
        if isinstance(entry, dict)
    }
    current = cabled_rails()
    # The loop below can only see rails that are still there. A rail that loses carrier or
    # disappears drops out of `cabled_rails()` entirely, so it was compared against nothing:
    # the cluster still read as configured while `nccl_env` advertised one HCA and the link
    # ran at about half its bandwidth. Missing is drift too.
    live = {rail["netdev"] for rail in current}
    for netdev, entry in planned.items():
        if netdev and netdev not in live:
            problems.append(
                f"{netdev} is in the saved plan at {entry.get('address') or 'no address'} but "
                f"is not cabled or has no carrier now; the pair is running on the remaining "
                f"rails at roughly half the two-rail bandwidth"
            )
    for rail in current:
        entry = planned.get(rail["netdev"])
        if entry is None:
            problems.append(
                f"{rail['ib_device']} ({rail['netdev']}) is cabled but is not in the saved "
                f"plan; re-run `unsloth spark setup`"
            )
            continue
        if not rail["ipv4"]:
            problems.append(
                f"{rail['netdev']} has no address; the plan places it at {entry.get('address')}"
            )
        elif entry.get("address") and entry["address"] not in rail["ipv4"]:
            problems.append(
                f"{rail['netdev']} is at {', '.join(rail['ipv4'])} but the plan places it "
                f"at {entry['address']}"
            )
        want_mtu = entry.get("mtu")
        if want_mtu and rail["mtu"] and int(want_mtu) != rail["mtu"]:
            # Not cosmetic: MTU 9000 is what lifts the RoCE path MTU from 1024 to 4096.
            problems.append(
                f"{rail['netdev']} is at MTU {rail['mtu']}, the plan asks for {want_mtu}"
            )
    return problems


# One /24 per PCIe function: a single subnet drives only one function, half the link.
DEFAULT_SUBNETS = ("192.168.200", "192.168.201")
DEFAULT_MTU = 9000  # lifts the RoCE path MTU from 1024 to 4096


def nccl_env(rails: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """NCCL settings for GB10; every one is load-bearing, none is a tuning knob. GB10 has no
    GPUDirect RDMA, so ``NCCL_NET_GDR_LEVEL=0`` is required: unset, it hangs
    ``init_process_group``, as does GID index 0 (RoCEv1; RoCEv2's IPv4 GID is 3).
    ``NCCL_IB_MERGE_NICS=1`` is what drives both PCIe functions of the one physical port."""
    rails = rails if rails is not None else cabled_rails()
    hcas = ",".join(r["ib_device"] for r in rails) or "rocep1s0f0,roceP2p1s0f0"
    primary = next((r["netdev"] for r in rails if r["netdev"]), "enp1s0f0np0")
    return {
        "NCCL_SOCKET_IFNAME": primary,
        "GLOO_SOCKET_IFNAME": primary,
        "NCCL_IB_HCA": hcas,
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_MERGE_NICS": "1",
        "NCCL_CROSS_NIC": "1",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_IB_DISABLE": "0",
    }


def apply_nccl_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Fill the GB10 NCCL defaults into ``os.environ``; an existing value always wins."""
    target = env if env is not None else os.environ
    applied: Dict[str, str] = {}
    if not is_dgx_spark():
        return applied
    for key, value in nccl_env().items():
        if not target.get(key):
            target[key] = value
            applied[key] = value
    return applied


# ~92-97 Gb/s per rail is NVIDIA's reference; ~13-16 is the kernel-6.17/CX-7 firmware fault.
DEGRADED_GBPS = 40.0
EXPECTED_GBPS = 90.0


def link_health(
    peer_ip: str,
    ib_device: str,
    local_ip: str,
    seconds: int = 5,
    port: int = 18999,
) -> Dict[str, Any]:
    """Measure one rail end to end; {} when it cannot be measured. ib_write_bw needs a
    server here and a client on the peer, so this needs the pairing step's SSH keys."""
    if not shutil.which("ib_write_bw") or not shutil.which("ssh"):
        return {}
    server = None
    try:
        server = subprocess.Popen(
            [
                "ib_write_bw",
                "-d",
                ib_device,
                "-F",
                "-x",
                "3",
                "--report_gbits",
                "-D",
                str(seconds),
                "-s",
                "1048576",
                "-q",
                "4",
                "-p",
                str(port),
            ],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
        # Let the server bind before the client dials in.
        import time

        time.sleep(3)
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"{_ssh_user()}@{peer_ip}",
                f"ib_write_bw -d {ib_device} -F -x 3 --report_gbits -D {seconds} "
                f"-s 1048576 -q 4 -p {port} {local_ip}",
            ],
            capture_output = True,
            text = True,
            timeout = seconds + 60,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    finally:
        if server is not None:
            try:
                server.wait(timeout = seconds + 20)
            except Exception:
                server.kill()

    gbps = None
    for line in proc.stdout.splitlines():
        fields = line.split()
        # The results row starts with the message size in bytes.
        if len(fields) >= 4 and fields[0].isdigit():
            try:
                gbps = float(fields[3])
            except ValueError:
                continue
    if gbps is None:
        return {}
    return {
        "gbps": gbps,
        "degraded": gbps < DEGRADED_GBPS,
        "expected_gbps": EXPECTED_GBPS,
    }


_WC_MARKER = "Write combining is not supported"


def write_combining_broken() -> Optional[bool]:
    """True broken, False healthy, None *could not tell*. Ubuntu sets dmesg_restrict, so an
    unprivileged read fails and answering False would report a healthy link on exactly the
    machines carrying this fault (the mlx5 write-combining ceiling on Grace-class ARM64)."""
    dmesg = shutil.which("dmesg")
    if dmesg:
        try:
            proc = subprocess.run([dmesg], capture_output = True, text = True, timeout = 10)
            # dmesg exits 0 even when the buffer read is denied, so test output.
            if proc.stdout.strip():
                return _WC_MARKER in proc.stdout and "mlx5" in proc.stdout
        except (OSError, subprocess.SubprocessError):
            pass
    # kern.log survives dmesg_restrict.
    for log in ("/var/log/kern.log", "/var/log/dmesg"):
        try:
            with open(log, "r", encoding = "utf-8", errors = "replace") as handle:
                text = handle.read()
        except OSError:
            continue
        if "mlx5" in text:
            return _WC_MARKER in text
    return None


def pending_system_updates() -> List[str]:
    """DGX/ConnectX-relevant packages with an upgrade waiting. Unsloth never installs
    these itself: the remedy needs a reboot, which is not Unsloth's to take."""
    apt = shutil.which("apt")
    if not apt:
        return []
    try:
        out = subprocess.run(
            [apt, "list", "--upgradable"],
            capture_output = True,
            text = True,
            timeout = 60,
            env = {**os.environ, "LC_ALL": "C"},
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    wanted = ("dgx", "mlnx", "nvidia", "linux-image", "linux-nvidia", "firmware")
    held = _held_packages()
    names = []
    for line in out.splitlines():
        pkg = line.split("/", 1)[0].strip()
        if pkg and pkg not in held and any(token in pkg for token in wanted):
            names.append(pkg)
    return sorted(set(names))


def _held_packages() -> set:
    apt_mark = shutil.which("apt-mark")
    if not apt_mark:
        return set()
    try:
        out = subprocess.run(
            [apt_mark, "showhold"], capture_output = True, text = True, timeout = 30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def ota_status() -> Dict[str, Any]:
    """What ``nvidia-spark-ota-check`` says about this Spark. Returns {} when the tool is
    absent, which callers must not read as "no OTA available"."""
    tool = shutil.which("nvidia-spark-ota-check")
    if not tool:
        return {}
    try:
        proc = subprocess.run(
            [tool, "is-ota-available"],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        return json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def update_instructions() -> List[str]:
    """What to do about a degraded link; OTA state decides. The usual "dist-upgrade + reboot"
    advice changes nothing on a box already on the newest OTA, so check before spending one."""
    status = ota_status()
    if status.get("available"):
        name = status.get("name") or "a newer release"
        return [
            f"A newer DGX Spark OTA is available ({name}).",
            "Apply it with the DGX Dashboard -- NVIDIA's supported path, which keeps the",
            "firmware and driver paired. Raw `apt upgrade` can install a driver the on-box",
            "firmware rejects, which is a documented way to lose the GPU entirely.",
        ]
    if status:
        return [
            "This Spark is already on the newest validated OTA, so no update fixes this.",
            "The ~13 Gb/s ceiling is a platform-level mlx5 write-combining limitation on",
            "ARM64 Grace cores; the kernel fix is not in a released Spark kernel yet.",
            "Re-check later with: nvidia-spark-ota-check is-ota-available",
        ]
    return [
        "Check for a validated update first:  nvidia-spark-ota-check is-ota-available",
        "Apply any update through the DGX Dashboard rather than raw apt, so firmware and",
        "driver stay paired.",
    ]


# llama.cpp's RPC backend places layers on a remote device, so a model too large for one
# Spark splits across the pair. Training cannot: device_count() is 1 on each host, so
# `device_map="balanced"` has nothing to balance (unsloth/unsloth#4858).
RPC_DEFAULT_PORT = 50052

# The legacy `rpc-server` name predates the ggml- prefix; macOS carries a versioned dylib.
_RPC_SERVER_NAMES = ("ggml-rpc-server", "rpc-server", "ggml-rpc-server.exe", "rpc-server.exe")
# What Windows will actually run. `os.access(path, os.X_OK)` is not that test: on Windows it
# succeeds for ANY existing file, so it accepted a text file as a server binary. The extension
# is the real permission there, and it is a fixed set rather than PATHEXT because the only
# thing being resolved is a binary this project ships.
_WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".com", ".bat", ".cmd")


def _on_windows(windows: Optional[bool] = None) -> bool:
    return (os.name == "nt") if windows is None else bool(windows)


def _is_executable_file(path, windows: Optional[bool] = None) -> bool:
    """Whether this host would run `path`. POSIX means the exec bit; Windows means the suffix."""
    try:
        if not os.path.isfile(path):
            return False
        if _on_windows(windows):
            return os.path.splitext(str(path))[1].lower() in _WINDOWS_EXECUTABLE_SUFFIXES
        return os.access(path, os.X_OK)
    except OSError:
        return False


def rpc_server_names(windows: Optional[bool] = None) -> Tuple[str, ...]:
    """`_RPC_SERVER_NAMES` with the platform's own spelling first.

    The list is POSIX-shaped, extensionless names leading, and the search takes the first hit.
    On Windows that let a stray extensionless `ggml-rpc-server` sitting beside the real
    `ggml-rpc-server.exe` win, so the resolution could report success while returning the wrong
    file. Ordering by platform is what stops that; rejecting non-executables alone would not."""
    suffixed = tuple(
        n for n in _RPC_SERVER_NAMES if n.lower().endswith(_WINDOWS_EXECUTABLE_SUFFIXES)
    )
    plain = tuple(n for n in _RPC_SERVER_NAMES if n not in suffixed)
    return (suffixed + plain) if _on_windows(windows) else (plain + suffixed)


_RPC_LIB_NAMES = ("libggml-rpc.so", "libggml-rpc.dylib", "libggml-rpc.0.dylib", "ggml-rpc.dll")
_BUNDLE_SUBDIRS = (("build", "bin"), ("build", "bin", "Release"), ("bin",), ())


def llama_bundle_dir() -> Path:
    """The managed llama.cpp bundle, resolved as the installer resolves it. The default is
    deliberately NOT under the studio root -- the bundle sits one level up from the venv -- so
    ``_studio_root() / "llama.cpp"`` would make provision silently skip it."""
    override = (os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    root = _studio_root()
    try:
        is_default = root.resolve() == _DEFAULT_STUDIO_ROOT.resolve()
    except (OSError, ValueError, RuntimeError):
        is_default = root == _DEFAULT_STUDIO_ROOT
    if not is_default:
        return root / "llama.cpp"
    return Path.home() / ".unsloth" / "llama.cpp"


def _peer_dir_exists(peer_ip: str, user: str, remote_dir: str) -> bool:
    """Read-only `test -d` on the peer. Used only by the dry run, which must not write."""
    try:
        return (
            subprocess.run(
                [
                    "ssh",
                    "-n",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{user}@{peer_ip}",
                    "test",
                    "-d",
                    remote_dir,
                ],
                capture_output = True,
                timeout = 30,
            ).returncode
            == 0
        )
    except Exception:
        return False


def _find_in_bundle(
    root: Path,
    names: Tuple[str, ...],
    executable: bool = False,
) -> Optional[Path]:
    for parts in _BUNDLE_SUBDIRS:
        base = root.joinpath(*parts) if parts else root
        for name in names:
            candidate = base / name
            try:
                if not candidate.is_file():
                    continue
                if executable and not _is_executable_file(candidate):
                    continue
            except OSError:
                continue
            return candidate
    return None


def rpc_server_binary() -> Optional[str]:
    """Path to ggml-rpc-server, or None. Bundles from b10796-mix-659e406 onward ship it;
    earlier ones shipped only the client backend, so a source build or PATH is the fallback.
    build/bin and build/bin/Release lead ``_BUNDLE_SUBDIRS`` to match the installer."""
    roots = [llama_bundle_dir(), Path.home() / "src" / "llamacpp-rpc"]
    for root in roots:
        found = _find_in_bundle(root, rpc_server_names(), executable = True)
        if found is not None:
            return str(found)
    return shutil.which("ggml-rpc-server") or shutil.which("rpc-server")


_LLAMA_SERVER_NAMES = ("llama-server", "llama-server.exe")


def llama_server_names(windows: Optional[bool] = None) -> Tuple[str, ...]:
    """`_LLAMA_SERVER_NAMES` with the platform's own spelling first, as `rpc_server_names`."""
    suffixed = tuple(
        n for n in _LLAMA_SERVER_NAMES if n.lower().endswith(_WINDOWS_EXECUTABLE_SUFFIXES)
    )
    plain = tuple(n for n in _LLAMA_SERVER_NAMES if n not in suffixed)
    return (suffixed + plain) if _on_windows(windows) else (plain + suffixed)


def llama_server_binary() -> Optional[str]:
    """Path to llama-server, or None. Separate from `rpc_server_binary` because the replica
    layout runs only this: requiring the RPC server to locate it refused a deployment that
    never touches RPC."""
    for root in (llama_bundle_dir(), Path.home() / "src" / "llamacpp-rpc"):
        found = _find_in_bundle(root, llama_server_names(), executable = True)
        if found is not None:
            return str(found)
    return shutil.which("llama-server")


def _bundle_version(root: Path) -> str:
    """The release tag from BUILD_INFO.txt, or ``"unknown"``. Source builds have no such file,
    and that must read unknown rather than fail: an unknown is compared by library hash."""
    for parts in ((), ("build", "bin"), ("bin",)):
        base = root.joinpath(*parts) if parts else root
        text = _read(base / "BUILD_INFO.txt", 4096)
        if not text:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                if "version" in key.lower() and value.strip():
                    return value.strip()
            return line
    return "unknown"


def _file_md5(path: Path) -> Optional[str]:
    import hashlib

    digest = hashlib.md5()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def llama_bundle_identity(root: Optional[Path] = None) -> Dict[str, Any]:
    """The bundle tag and the RPC library's hash: two signals because each fails alone.
    BUILD_INFO.txt is missing from source builds; the md5 catches a bundle patched in
    place under an unchanged tag. Never raises; a missing bundle is ``present: False``."""
    root = Path(root) if root is not None else llama_bundle_dir()
    out: Dict[str, Any] = {
        "root": str(root),
        "present": False,
        "version": "unknown",
        "rpc_lib": None,
        "rpc_lib_md5": None,
        "rpc_server": None,
    }
    try:
        out["present"] = root.is_dir()
    except OSError:
        return out
    if not out["present"]:
        return out
    out["version"] = _bundle_version(root)
    lib = _find_in_bundle(root, _RPC_LIB_NAMES)
    if lib is not None:
        out["rpc_lib"] = str(lib)
        out["rpc_lib_md5"] = _file_md5(lib)
    server = _find_in_bundle(root, rpc_server_names(), executable = True)
    if server is not None:
        out["rpc_server"] = str(server)
    return out


def _peer_relative_path(path: Path) -> str:
    """``path`` as the peer should see it. The peer's home may differ (different username),
    so a path under our home is sent home-relative and expanded THERE."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return path.as_posix()


# Runs on the PEER under its own python3, so it must stay self-contained: no Unsloth
# checkout is assumed there. Mirrors llama_bundle_identity() field for field.
_BUNDLE_PROBE = """\
import hashlib, json, os
root = os.path.expanduser(os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or {root!r})
subs = (("build", "bin"), ("build", "bin", "Release"), ("bin",), ())
libs = {libs!r}
servers = {servers!r}
out = {{"root": root, "present": os.path.isdir(root), "version": "unknown",
       "rpc_lib": None, "rpc_lib_md5": None, "rpc_server": None}}
win = os.name == "nt"
exts = (".exe", ".com", ".bat", ".cmd")
def runnable(c):
    if win:
        return os.path.splitext(c)[1].lower() in exts
    return os.access(c, os.X_OK)
def order(names):
    # The peer decides for itself: it may not be the platform that launched this probe.
    suffixed = [n for n in names if n.lower().endswith(exts)]
    plain = [n for n in names if n not in suffixed]
    return (suffixed + plain) if win else (plain + suffixed)
def find(names, executable):
    for parts in subs:
        base = os.path.join(root, *parts) if parts else root
        for name in order(names):
            c = os.path.join(base, name)
            if os.path.isfile(c) and (not executable or runnable(c)):
                return c
    return None
if out["present"]:
    for parts in ((), ("build", "bin"), ("bin",)):
        p = os.path.join(root, *parts, "BUILD_INFO.txt")
        try:
            with open(p, "r", encoding = "utf-8", errors="replace") as fh:
                text = fh.read(4096)
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                if "version" in k.lower() and v.strip():
                    out["version"] = v.strip()
                    break
            out["version"] = line
            break
        break
    lib = find(libs, False)
    if lib:
        out["rpc_lib"] = lib
        h = hashlib.md5()
        with open(lib, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["rpc_lib_md5"] = h.hexdigest()
    out["rpc_server"] = find(servers, True)
print("UNSLOTH_BUNDLE " + json.dumps(out))
"""


def peer_llama_bundle_identity(peer_ip: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """``llama_bundle_identity()`` as evaluated ON THE PEER. None means "could not check",
    which callers must report as unverified rather than as matching."""
    if not peer_ip or not shutil.which("ssh"):
        return None
    import base64

    source = _BUNDLE_PROBE.format(
        root = _peer_relative_path(llama_bundle_dir()),
        libs = _RPC_LIB_NAMES,
        servers = _RPC_SERVER_NAMES,
    )
    blob = base64.b64encode(source.encode()).decode()
    user = _ssh_user()
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-n",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"{user}@{peer_ip}",
                f"echo {blob} | base64 -d | python3 -",
            ],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception:
        return None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("UNSLOTH_BUNDLE "):
            try:
                data = json.loads(line[len("UNSLOTH_BUNDLE ") :])
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


PROVISION_FIX = "run `unsloth spark provision` to copy this node's llama.cpp bundle to the peer"


def compare_llama_bundles(local: Dict[str, Any], peer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Will llama-server here and ggml-rpc-server there speak the same RPC protocol? The
    protocol is pinned by the build: b10796 speaks 6.0 where earlier bundles spoke 5.1,
    and llama-server then fails at load. ``ok`` may be None for "could not verify"."""
    out: Dict[str, Any] = {"ok": None, "problems": [], "notes": [], "local": local, "peer": peer}
    lv = str(local.get("version") or "unknown")
    if peer is None:
        out["notes"].append(
            "could not read the peer's llama.cpp bundle (ssh unavailable or the probe "
            "failed), so RPC protocol parity is UNVERIFIED. If llama-server fails at "
            f"load with 'RPC server version mismatch', {PROVISION_FIX}."
        )
        return out
    if not peer.get("present"):
        out["problems"].append(
            f"the peer has no llama.cpp bundle at {peer.get('root')} while this Spark "
            f"runs {lv}. Nothing there can answer an RPC connection. Fix: {PROVISION_FIX}."
        )
        out["ok"] = False
        return out
    pv = str(peer.get("version") or "unknown")
    lm, pm = local.get("rpc_lib_md5"), peer.get("rpc_lib_md5")
    if lv != pv and not (lv == "unknown" and pv == "unknown"):
        out["problems"].append(
            f"llama.cpp bundle mismatch: this Spark has {lv}, the peer has {pv}. The RPC "
            f"protocol is pinned by the build (b10796 speaks 6.0, earlier bundles 5.1), so "
            f"llama-server fails at load with 'RPC server version mismatch'. "
            f"Fix: {PROVISION_FIX}."
        )
    elif lm and pm and lm != pm:
        out["problems"].append(
            f"libggml-rpc differs between the nodes (md5 {lm[:12]} here, {pm[:12]} on the "
            f"peer) although both report {lv}; one of them was rebuilt or patched in "
            f"place. Fix: {PROVISION_FIX}."
        )
    elif lv == "unknown" and not (lm and pm):
        out["notes"].append(
            "neither bundle carries BUILD_INFO.txt and libggml-rpc could not be hashed "
            "on both nodes, so RPC protocol parity is UNVERIFIED."
        )
        return out
    out["ok"] = not out["problems"]
    if out["ok"]:
        out["notes"].append(
            f"llama.cpp bundles match on both nodes ({lv}"
            + (f", libggml-rpc md5 {lm[:12]}" if lm else "")
            + ")."
        )
    return out


# HELLO wire format, ggml/src/ggml-rpc/ggml-rpc.cpp at ggml-org/llama.cpp b10796 (RPC 6.0):
#   client -> uint8 RPC_CMD_HELLO(14), LE uint64 payload length, then conn_caps[24] zeroed
#   server -> LE uint64 body length, then uint8 major, minor, patch, pad, conn_caps[24]
# A 6.0 server closes without replying on a request length it does not recognise, so EOF
# before any reply means "listening, but not a 6.0 server" -- not the same as refused.
RPC_CMD_HELLO = 14
RPC_CONN_CAPS_SIZE = 24
RPC_HELLO_MAX_BODY = 4096


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def rpc_hello_probe_detail(
    host: str,
    port: int = RPC_DEFAULT_PORT,
    timeout: float = 2.0,
    read_timeout: float = 3.0,
) -> Dict[str, Any]:
    """Send one HELLO and classify the reply. Never raises, always bounded. ``closed`` means a
    listener hung up without replying, which is what a protocol-mismatched server does.

    ``timeout`` (nothing answered the CONNECT) and ``silent`` (it accepted, then never replied)
    are separate states, because they mean opposite things for a launch: the first is an
    unreachable or filtered host, the second is an OCCUPIED port a new server cannot bind."""
    import struct

    out: Dict[str, Any] = {"host": host, "port": port, "state": "refused", "version": None}
    try:
        sock = socket.create_connection((host, port), timeout = timeout)
    except (socket.timeout, TimeoutError):
        out["state"] = "timeout"
        return out
    except Exception:
        return out
    try:
        sock.settimeout(read_timeout)
        payload = bytes(RPC_CONN_CAPS_SIZE)
        sock.sendall(bytes([RPC_CMD_HELLO]) + struct.pack("<Q", len(payload)) + payload)
        header = _recv_exact(sock, 8)
        if header is None:
            out["state"] = "closed"
            return out
        (length,) = struct.unpack("<Q", header)
        if length < 3 or length > RPC_HELLO_MAX_BODY:
            out["state"] = "garbled"
            return out
        body = _recv_exact(sock, length)
        if body is None:
            out["state"] = "closed"
            return out
        out["state"] = "ok"
        out["version"] = (body[0], body[1], body[2])
        return out
    except (socket.timeout, TimeoutError):
        out["state"] = "silent"
        return out
    except Exception:
        out["state"] = "garbled"
        return out
    finally:
        try:
            sock.close()
        except OSError:
            pass


def rpc_hello_probe(
    host: str,
    port: int = RPC_DEFAULT_PORT,
    timeout: float = 2.0,
) -> Optional[Tuple[int, int, int]]:
    return rpc_hello_probe_detail(host, port, timeout = timeout)["version"]


def rpc_protocol_preflight(peer_ip: str, port: int = RPC_DEFAULT_PORT) -> Dict[str, Any]:
    """Bundle identity (works before anything runs) plus a live HELLO against whatever
    already listens (catches a stale server from an older bundle). A refused connection is
    the normal pre-launch state. ``ok`` False means CONFIRMED mismatch, None unverified."""
    result = compare_llama_bundles(llama_bundle_identity(), peer_llama_bundle_identity(peer_ip))
    result["peer_rpc"] = peer_live = rpc_hello_probe_detail(peer_ip, port)
    result["local_rpc"] = local_live = rpc_hello_probe_detail("127.0.0.1", port)
    seen: Dict[str, Tuple[int, int, int]] = {}
    for where, live in (("the peer", peer_live), ("this Spark", local_live)):
        state, version = live["state"], live["version"]
        # A split runs ggml-rpc-server on the PEER and points the local llama-server at
        # `peer:port`; nothing local binds that port or speaks the protocol. Refusing the
        # deployment over an unrelated local listener rejected a launch that would have
        # worked, so what is found here is reported and does not block.
        keep = result["problems"] if where == "the peer" else result["notes"]
        if state == "ok":
            seen[where] = version
            result["notes"].append(
                f"a ggml-rpc-server on {where} ({live['host']}:{port}) answers HELLO with "
                f"RPC protocol {version[0]}.{version[1]}.{version[2]}."
            )
        elif state == "closed":
            keep.append(
                f"something listens on {where} at {live['host']}:{port} but closed the "
                f"connection on an RPC 6.0 HELLO without answering. That is what an older "
                f"(5.x) ggml-rpc-server does, and llama-server would report 'RPC server "
                f"version mismatch'. Stop it, then {PROVISION_FIX} and start the one from "
                f"the current bundle."
            )
        elif state == "garbled":
            keep.append(
                f"the listener on {where} at {live['host']}:{port} is not a ggml-rpc-server "
                f"(its HELLO reply was malformed). Free the port or pick another with "
                f"--rpc-port."
            )
        elif state == "silent":
            # Held by something, so a new server cannot bind it -- the one outcome this
            # preflight exists to prevent. It used to fall through with `refused`, which is
            # the normal pre-launch state, and the command printed a launch on that port.
            keep.append(
                f"something on {where} accepted a connection at {live['host']}:{port} and "
                f"never answered the HELLO. The port is held, so a new ggml-rpc-server "
                f"cannot bind it. Stop that process, or pick another port with --rpc-port."
            )
    if len(seen) == 2 and seen["the peer"] != seen["this Spark"]:
        a, b = seen["this Spark"], seen["the peer"]
        result["problems"].append(
            f"RPC protocol mismatch between the running servers: this Spark speaks "
            f"{a[0]}.{a[1]}.{a[2]}, the peer {b[0]}.{b[1]}.{b[2]}. Fix: {PROVISION_FIX}, "
            f"then restart both servers."
        )
    if result["problems"]:
        result["ok"] = False
    return result


def peer_address_of(addr: str) -> Optional[str]:
    """The OTHER endpoint of a two-node rail, given this node's address on it.

    Setup assigns `NODE_BASE_OCTET + node_index`, so node 0 is `.12` and node 1 is `.13`.
    Always adding one is right only on node 0: run from the second Spark it returned `.14`,
    a host that does not exist, and doctor, provisioning, serving and training launched from
    there all aimed at it. The direction has to come from which endpoint this node is."""
    head, _, last = addr.rpartition(".")
    if not head:
        return None
    try:
        index = int(last) - NODE_BASE_OCTET
    except ValueError:
        return None
    if index < 0:
        return None
    # Two-node rail: the peer is the other one of the pair, whichever end this is.
    return f"{head}.{NODE_BASE_OCTET + (1 if index == 0 else 0)}"


def peer_ip_for(rails: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """The peer's address on the first configured rail."""
    rails = rails if rails is not None else cabled_rails()
    for rail in rails:
        for addr in rail.get("ipv4", []):
            peer = peer_address_of(addr)
            if peer:
                return peer
    return None


def rpc_cluster_plan(port: int = RPC_DEFAULT_PORT) -> Dict[str, Any]:
    if not is_dgx_spark():
        # Answer before rail discovery's sysfs walk and `ip` fork; the answer cannot change.
        return {
            "ok": False,
            "problems": ["not a DGX Spark"],
            "rpc_server": None,
            "local_ip": None,
            "peer_ip": None,
            "port": port,
            "rpc_arg": None,
        }
    binary = rpc_server_binary()
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    problems = []
    if not is_dgx_spark():
        problems.append("not a DGX Spark")
    if binary is None:
        problems.append(
            f"no ggml-rpc-server binary in {llama_bundle_dir()} (bundles from "
            f"b10796-mix-659e406 onward ship it; update the llama.cpp prebuilt, or build "
            f"llama.cpp with -DGGML_RPC=ON)"
        )
    if peer is None or local is None:
        problems.append("no configured peer rail (run `unsloth spark setup`)")
    return {
        "ok": not problems,
        "problems": problems,
        "rpc_server": binary,
        "local_ip": local,
        "peer_ip": peer,
        "port": port,
        # Remote first: llama.cpp fills RPC devices in the given order; local-first starves it.
        "rpc_arg": f"{peer}:{port},127.0.0.1:{port}" if peer else None,
    }


# NVIDIA numbers the first node .12 of each rail subnet; the cap is honesty, not space.
NODE_BASE_OCTET = 12
MAX_PLANNABLE_NODES = 240


def rail_plan_report(
    rails: Optional[List[Dict[str, Any]]] = None,
    node_index: int = 0,
    n_nodes: int = 2,
    switched: bool = False,
) -> Dict[str, Any]:
    """The addressing plan, or an explicit refusal -- never a wrong plan. A flat /24 per PCIe
    function is right for a cabled pair and for a switch (``switched=True``) but wrong for 3+
    Sparks in a chain: nodes sharing no cable land on one subnet and every route black-holes.
    A plausible netplan that does not work is worse than a refusal, so N>2 refuses."""
    rails = rails if rails is not None else cabled_rails()
    problems: List[str] = []
    notes: List[str] = []
    n_nodes = max(1, int(n_nodes))
    if node_index < 0 or node_index >= n_nodes:
        problems.append(
            f"node_index {node_index} is outside 0..{n_nodes - 1} for a {n_nodes}-node cluster"
        )
    if n_nodes > MAX_PLANNABLE_NODES:
        problems.append(
            f"{n_nodes} nodes does not fit one /24 starting at .{NODE_BASE_OCTET}; "
            f"Unsloth will not plan more than {MAX_PLANNABLE_NODES}"
        )
    if n_nodes > 2 and not switched:
        problems.append(
            f"{n_nodes} Sparks cannot be cabled point-to-point the way two are. This flat "
            f"one-/24-per-rail plan is only correct if all {n_nodes} nodes share a "
            f"switched RoCE fabric. Re-run with switched=True (`--switched`) if they do. "
            f"If instead they are daisy-chained, each cable needs its own subnet and "
            f"Unsloth will not guess your cabling -- a wrong netplan is worse than none."
        )
    if not rails:
        problems.append("no cabled ConnectX rail found on this node")
    # A Spark's ConnectX presents TWO PCIe functions, and one /24 drives one function: with a
    # single rail the pair runs at about half the advertised bandwidth. That is a working
    # cluster, so it is not refused, but it was not reported either -- the plan applied, the
    # cluster read as configured, and `nccl_env` advertised the one HCA without comment.
    degraded = bool(rails) and len(rails) < len(DEFAULT_SUBNETS)
    if degraded:
        notes.append(
            f"DEGRADED: {len(rails)} of {len(DEFAULT_SUBNETS)} ConnectX rail functions are "
            f"cabled and up, so this plan addresses one. One subnet drives one function, so "
            f"expect roughly half the two-rail bandwidth. Check the second cable and that "
            f"both PCIe functions appear in `ibv_devices`."
        )
    if n_nodes > 2 and switched:
        notes.append(
            "Assuming a switched RoCE fabric with all rails in the same broadcast domain, "
            "jumbo frames enabled on every switch port (MTU 9000), and PFC/ECN configured. "
            "None of that is verified from here."
        )
    plan: List[Dict[str, str]] = []
    if not problems:
        for slot, rail in enumerate(rails[: len(DEFAULT_SUBNETS)]):
            plan.append(
                {
                    "ib_device": rail["ib_device"],
                    "netdev": rail["netdev"],
                    "address": f"{DEFAULT_SUBNETS[slot]}.{NODE_BASE_OCTET + node_index}",
                    "prefix": "24",
                    "mtu": str(DEFAULT_MTU),
                }
            )
    return {
        "ok": not problems,
        "problems": problems,
        "notes": notes,
        "plan": plan,
        "degraded": degraded,
        "node_index": node_index,
        "n_nodes": n_nodes,
        "switched": switched,
    }


def rail_plan(
    rails: Optional[List[Dict[str, Any]]] = None,
    node_index: int = 0,
    n_nodes: int = 2,
    switched: bool = False,
) -> List[Dict[str, str]]:
    """One /24 per PCIe function, node N on .12+N. Two subnets are not redundancy: one subnet
    drives one function and caps a pair near 100 Gb/s. Returns ``[]``, never a half plan."""
    return rail_plan_report(rails, node_index, n_nodes, switched)["plan"]


def netplan_yaml(plan: List[Dict[str, str]]) -> str:
    """A netplan drop-in for the rail addressing. An empty plan renders as comments, not an
    empty ``ethernets:`` map: netplan accepts that and silently configures nothing."""
    if not plan:
        return (
            "# No addressing plan was produced -- nothing to apply.\n"
            "# Run `unsloth spark setup` and read the refusal it prints; writing this\n"
            "# file with no `ethernets:` entries would configure nothing while looking\n"
            "# like it had.\n"
        )
    lines = ["network:", "  version: 2", "  renderer: NetworkManager", "  ethernets:"]
    for entry in plan:
        lines += [
            f"    {entry['netdev']}:",
            f"      addresses: [{entry['address']}/{entry['prefix']}]",
            "      dhcp4: no",
            f"      mtu: {entry['mtu']}",
        ]
    return "\n".join(lines) + "\n"


def _print_manual_steps(
    plan: List[Dict[str, str]],
    peer_plan: List[Dict[str, str]],
    *,
    extra_plans: Optional[List[List[Dict[str, str]]]] = None,
) -> None:
    def emit(where: str, entries: List[Dict[str, str]]) -> None:
        print(f"\n  Run these on {where}:")
        print("    sudo tee /etc/netplan/40-unsloth-cx7.yaml >/dev/null <<'EOF'")
        print(netplan_yaml(entries), end = "")
        # Column 0: <<'EOF' only ends on an unindented terminator, and an indented one makes the
        # heredoc swallow the chmod and netplan apply lines into the file it was writing.
        print("EOF")
        print("    sudo chmod 600 /etc/netplan/40-unsloth-cx7.yaml && sudo netplan apply")

    emit("THIS Spark", plan)
    others = [peer_plan] + list(extra_plans or [])
    for index, entries in enumerate(others, start = 1):
        label = "the PEER Spark" if len(others) == 1 else f"Spark node {index}"
        emit(label, entries)


# Healthy measures ~21.6 GB/s here and the fault state ~3.0, so 8.0 sits far from both.
NCCL_DEGRADED_GBPS = 8.0
NCCL_HEALTHY_GBPS = 15.0

POWER_CYCLE_ADVICE = """\
  This is almost always the DGX Spark power-delivery fault. Fix it in this order:

    1. FULLY POWER CYCLE BOTH SPARKS -- shut down, unplug the power for ~30 seconds,
       then boot. A REBOOT IS NOT ENOUGH; the fault survives one.
    2. Leave the QSFP cable connected the whole time. Hot-plugging it after boot is a
       separate fault that also produces a slow link and needs another reboot.

  Measured here before and after a power cycle: 3.0 GB/s -> 21.6 GB/s, a 7x difference,
  with every setting identical. Raw RDMA looked healthy (24.5 GB/s) in BOTH states, which
  is why only a real NCCL collective detects this."""


def nccl_bandwidth(
    peer_ip: str,
    local_ip: str,
    mb: int = 1024,
    timeout: int = 90,
) -> Optional[float]:
    """Real NCCL all-reduce bus bandwidth in GB/s, or None. Shells out to torchrun rather
    than importing torch, so importing this module stays free on every other platform."""
    if not shutil.which("ssh"):
        return None
    # Random port: a fixed one collides with a dead run still holding the socket.
    import random

    port = random.randint(29600, 29999)
    # Both sides run the probe by absolute path: `-m studio.spark_nccl_probe` would need
    # Unsloth importable at the same path on both nodes, and the peer may have no venv.
    local_probe = osp.join(osp.dirname(osp.abspath(__file__)), "spark_nccl_probe.py")
    if not osp.exists(local_probe):
        return None
    remote_probe = "/tmp/spark_nccl_probe.py"
    user = _ssh_user()
    ssh_opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    try:
        subprocess.run(
            ["scp", *ssh_opts, local_probe, f"{user}@{peer_ip}:{remote_probe}"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 30,
            check = True,
        )
    except Exception:
        return None

    env = " ".join(f"{k}={v}" for k, v in nccl_env().items())

    def _common(trun: str) -> str:
        return (
            f"{env} SPARK_PROBE_MB={mb} {trun} --nnodes=2 --nproc_per_node=1 "
            f"--master_addr={local_ip} --master_port={port}"
        )

    # The peer sources `activate`, so the bare name is right there. Rank 0 runs from whatever
    # shell the user is in, where it is not: this ran no local rank at all, and doctor reported
    # the pair as unverified with no bandwidth on a healthy link.
    common = _common("torchrun")
    local_common = _common(managed_torchrun())
    # Non-interactive ssh has no venv on PATH: without this, torchrun is missing on the
    # peer, so it never starts and the local side hangs at the rendezvous.
    activate = venv_activate_sh()
    # The pid is recorded so every exit below can stop this rank. setsid makes it a process
    # group leader, so the negative kill takes the torchrun children with it.
    peer_cmd = (
        f"setsid nohup bash -c '[ -f {activate} ] && . {activate}; "
        f"exec env {common} --node_rank=1 {remote_probe}' "
        f"> {_NCCL_PROBE_LOG} 2>&1 < /dev/null & "
        f"echo $! > {_NCCL_PROBE_PID}"
    )
    try:
        started = subprocess.run(
            ["ssh", *ssh_opts, f"{user}@{peer_ip}", peer_cmd],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 30,
        )
    except Exception:
        stop_peer_nccl_probe(peer_ip, user, ssh_opts)
        return None
    # The result was discarded, and there is no `check = True`, so ssh 255 (auth or routing)
    # looked like a launch. Rank 0 then waited out the whole rendezvous timeout before doctor
    # said only "could not measure". `run_pipeline` already checks this for its peer launch.
    if started.returncode != 0:
        stop_peer_nccl_probe(peer_ip, user, ssh_opts)
        return None
    import time

    # Every path from here on stops the peer rank. Without this a local timeout, a torchrun
    # that never starts, or an exit during the collective left it detached in rendezvous or
    # NCCL timeout handling, still holding the 1 GiB probe buffer, and the next provisioning
    # run correctly refuses the peer as busy on the strength of a probe nobody is using.
    try:
        time.sleep(4)  # let the peer's rendezvous come up before we dial in
        try:
            out = subprocess.run(
                f"env {local_common} --node_rank=0 {shlex.quote(local_probe)}",
                shell = True,
                capture_output = True,
                text = True,
                timeout = timeout,
            )
        except Exception:
            return None
        for line in (out.stdout or "").splitlines():
            if line.startswith("SPARK_NCCL_BUSBW"):
                try:
                    return float(line.split()[1])
                except (IndexError, ValueError):
                    return None
        return None
    finally:
        stop_peer_nccl_probe(peer_ip, user, ssh_opts)


_NCCL_PROBE_LOG = "/tmp/spark_nccl_probe.log"
_NCCL_PROBE_PID = "/tmp/spark_nccl_probe.pid"


def stop_peer_by_pidfile(peer_ip: str, user: str, ssh_opts, pid_file: str) -> bool:
    """Stop a detached rank on the peer, by the pid it recorded for us.

    By pid, never by name: a pattern kill on a shared machine can take out something else that
    happens to match. A dead pid makes this a no-op, so it is safe on the success path too."""
    command = (
        f"p=$(cat {pid_file} 2>/dev/null); "
        f'if [ -n "$p" ]; then kill -TERM -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; fi; '
        f"rm -f {pid_file}"
    )
    try:
        subprocess.run(
            ["ssh", *ssh_opts, f"{user}@{peer_ip}", command],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 20,
        )
        return True
    except Exception:
        return False


def stop_peer_nccl_probe(peer_ip: str, user: str, ssh_opts) -> bool:
    return stop_peer_by_pidfile(peer_ip, user, ssh_opts, _NCCL_PROBE_PID)


def diagnose_link(busbw: Optional[float]) -> Dict[str, Any]:
    if busbw is None:
        return {
            "verdict": "unknown",
            "busbw": None,
            "summary": "could not measure NCCL bandwidth (needs a peer and torchrun)",
            "advice": "",
        }
    if busbw < NCCL_DEGRADED_GBPS:
        return {
            "verdict": "degraded",
            "busbw": busbw,
            "summary": (
                f"NCCL all-reduce is {busbw:.1f} GB/s -- far below the "
                f"~21 GB/s this link should reach. TRAINING WILL BE SLOW."
            ),
            "advice": POWER_CYCLE_ADVICE,
        }
    if busbw < NCCL_HEALTHY_GBPS:
        return {
            "verdict": "suspect",
            "busbw": busbw,
            "summary": (
                f"NCCL all-reduce is {busbw:.1f} GB/s. Healthy is ~21 GB/s, so "
                f"this is workable but below par."
            ),
            "advice": POWER_CYCLE_ADVICE,
        }
    return {
        "verdict": "healthy",
        "busbw": busbw,
        "summary": f"NCCL all-reduce is {busbw:.1f} GB/s -- healthy.",
        "advice": "",
    }


def python_dev_headers(peer_ip: Optional[str] = None) -> Dict[str, Any]:
    """CPython dev headers, here and on the peer. The failure is invisible: Triton JIT-builds a
    shim, and without Python.h a head rank blocks 601 s and dies with `DistStoreError: 1/2
    clients joined`. rsync copies packages but NOT the system headers they need."""
    import sysconfig

    out: Dict[str, Any] = {"local": False, "peer": None, "include": ""}
    inc = sysconfig.get_paths().get("include", "")
    out["include"] = inc
    out["local"] = bool(inc) and osp.exists(osp.join(inc, "Python.h"))
    if peer_ip and shutil.which("ssh"):
        user = _ssh_user()
        # Ask the PEER's own interpreter: Triton builds against whichever Python that
        # node's venv runs. base64 so the probe survives ssh -> bash -lc -> python -c.
        import base64

        probe = (
            "import sysconfig, os\n"
            "p = os.path.join(sysconfig.get_paths()['include'], 'Python.h')\n"
            "print('yes' if os.path.isfile(p) else 'no')\n"
        )
        b64 = base64.b64encode(probe.encode()).decode()
        activate = venv_activate_sh()
        remote = f"[ -f {activate} ] && . {activate}; " f"echo {b64} | base64 -d | python3 -"
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{user}@{peer_ip}",
                    remote,
                ],
                capture_output = True,
                text = True,
                timeout = 25,
            )
            answer = [l for l in (r.stdout or "").strip().splitlines() if l in ("yes", "no")]
            out["peer"] = (answer[-1] == "yes") if answer else None
        except Exception:
            out["peer"] = None
    return out


# Caches that must exist on BOTH nodes. A node missing one silently rebuilds from scratch,
# which presents as a HANG rather than an error (observed: 17+ min in CUDA-graph capture).
SHARED_CACHES = (
    "~/.cache/flashinfer",
    "~/.cache/vllm/flashinfer_autotune_cache",
    "~/.cache/vllm/torch_compile_cache",
)


def cache_symmetry(peer_ip: str) -> Dict[str, Optional[bool]]:
    """Only caches present LOCALLY are flagged: one neither node has is merely cold."""
    out: Dict[str, Optional[bool]] = {}
    if not shutil.which("ssh"):
        return out
    user = _ssh_user()
    for c in SHARED_CACHES:
        local = osp.expanduser(c)
        if not osp.isdir(local):
            continue
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{user}@{peer_ip}",
                    f"test -d {c} && echo yes || echo no",
                ],
                capture_output = True,
                text = True,
                timeout = 20,
            )
            out[c] = (r.stdout or "").strip() == "yes"
        except Exception:
            out[c] = None
    return out


def classify_cuda_state(cuinit: Optional[int], smi_ok: bool) -> str:
    """`ok` | `dead-engine` | `cuda-error` | `unknown`, from cuInit's return.

    Only 100 is the fault the probe is named for, and only 100 is what a reboot clears. Any
    nonzero code used to be called `dead-engine` and labelled CUDA_ERROR_NO_DEVICE, which put
    a driver and userspace mismatch -- and the synthesised -1 that means libcuda.so.1 would
    not even load -- under a reboot-only diagnosis that does not apply to either."""
    if cuinit == 0:
        return "ok"
    if not smi_ok or cuinit is None:
        return "unknown"
    if cuinit == 100:
        return "dead-engine"
    if cuinit < 0:
        # -1 is ours, from the probe's own except: libcuda.so.1 is missing or unloadable.
        return "unknown"
    return "cuda-error"


def cuda_health(peer_ip: Optional[str] = None) -> Dict[str, Any]:
    """A GPU that enumerates but whose compute engine is dead: nvidia-smi looks fine, cuInit(0)
    returns 100 and dmesg shows NVRM 0xbadf5600. A REBOOT clears it; a module reload does not,
    and a power cycle is NOT required. Not the power-delivery fault, which caps NCCL instead."""
    probe = (
        "import ctypes\n"
        "try:\n"
        "    r = ctypes.CDLL('libcuda.so.1').cuInit(0)\n"
        "except Exception:\n"
        "    r = -1\n"
        "print(r)\n"
    )
    out: Dict[str, Any] = {"local": None, "peer": None}

    _classify = classify_cuda_state

    smi = shutil.which("nvidia-smi") is not None
    if smi:
        try:
            r = subprocess.run(
                [sys.executable, "-c", probe], capture_output = True, text = True, timeout = 60
            )
            code = int((r.stdout or "-1").strip().splitlines()[-1])
        except Exception:
            code = None
        # A wedged driver is the case this command exists for, and there nvidia-smi is what hangs
        # or refuses to run, so an unguarded probe would abort doctor with a traceback.
        try:
            smi_ok = (
                subprocess.run(["nvidia-smi", "-L"], capture_output = True, timeout = 30).returncode
                == 0
            )
        except Exception:
            smi_ok = False
        out["local"] = {"cuinit": code, "state": _classify(code, smi_ok)}

    if peer_ip and shutil.which("ssh"):
        import base64

        b64 = base64.b64encode(probe.encode()).decode()
        act = venv_activate_sh()
        cmd = (
            f"[ -f {act} ] && . {act}; nvidia-smi -L >/dev/null 2>&1 && echo SMI_OK || echo SMI_BAD; "
            f"echo {b64} | base64 -d | python3 -"
        )
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{_ssh_user()}@{peer_ip}",
                    cmd,
                ],
                capture_output = True,
                text = True,
                timeout = 90,
            )
            lines = [l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
            smi_ok = "SMI_OK" in lines
            code = next((int(l) for l in reversed(lines) if l.lstrip("-").isdigit()), None)
            out["peer"] = {"cuinit": code, "state": _classify(code, smi_ok)}
        except Exception:
            out["peer"] = {"cuinit": None, "state": "unknown"}
    return out


def _cmd_doctor() -> int:
    if not is_dgx_spark():
        print("This machine is not a DGX Spark; nothing to check.")
        return 0
    info = discover_peers()
    if not info["cable_present"]:
        print("No QSFP cable detected. Nothing to measure.")
        return 0
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    if not peer or not local:
        print("No configured peer rail yet -- run `unsloth spark setup` first.")
        return 1

    # Every definite fault found below, so the exit code says what the output said. A dead
    # GPU or a missing Python.h was printed and then dropped: the return came from the NCCL
    # verdict alone, a dead engine usually measures `unknown`, and `unknown` returned 0. A
    # script could therefore read a node that cannot initialise CUDA as a healthy pair.
    blockers: List[str] = []

    # Cheap preflight first: a missing header wastes 10 minutes on a silent hang later.
    hdr = python_dev_headers(peer)
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}-dev"
    for where, ok in (("this Spark", hdr["local"]), ("the peer", hdr["peer"])):
        if ok is False:
            blockers.append(f"Python.h is missing on {where}")
            print(f"  MISSING Python.h on {where}.")
            print("    Without it Triton cannot build its cuda_utils shim. Locally that")
            print("    reads as 'Model architectures [...] failed to be inspected'; on a")
            print("    worker it makes the head rank hang for 601s and report only")
            print("    'DistStoreError: 1/2 clients joined'. Neither names the header.")
            print("")
            print("    Fix, preferred -- a uv-managed CPython ships its own headers, so")
            print("    this class of failure cannot recur:")
            print("        uv python install 3.12")
            print(f"        uv venv --python 3.12 && . .venv/bin/activate")
            print(f"    Or, if you must use the system interpreter: apt install {ver}")
            print("")
        elif ok is None:
            print(f"  could not check Python.h on {where} (ssh unavailable)\n")

    for where, info in cuda_health(peer).items():
        if not info:
            continue
        if info["state"] == "dead-engine":
            blockers.append(f"the GPU on {where} does not initialise")
            print(
                f"  GPU NOT USABLE on {where}: nvidia-smi works but cuInit returns "
                f"{info['cuinit']} (CUDA_ERROR_NO_DEVICE)."
            )
            print("    The GPU enumerates but its compute engine is not responding. Check")
            print("    `dmesg | grep NVRM` for 'Possible bad register read ... 0xbadf5600'.")
            print("    FIX: REBOOT that node. A module reload does not clear it, and a")
            print("    power cycle is not required -- a plain reboot is faster and works.")
            print("    (This is NOT the power-delivery fault, which leaves CUDA healthy and")
            print("     instead caps NCCL bandwidth. Same symptom class, opposite remedy.)")
            print("")
        elif info["state"] == "cuda-error":
            blockers.append(f"CUDA does not initialise on {where} (cuInit {info['cuinit']})")
            print(
                f"  CUDA NOT USABLE on {where}: nvidia-smi works but cuInit returns "
                f"{info['cuinit']}. That is not the 100 (CUDA_ERROR_NO_DEVICE) hardware "
                f"fault a reboot clears; the usual cause is a driver and userspace "
                f"mismatch. Check `nvidia-smi` against the installed CUDA runtime.\n"
            )
        elif info["state"] == "unknown":
            print(f"  could not determine GPU health on {where}\n")

    for cache, present in cache_symmetry(peer).items():
        if present is False:
            print(f"  CACHE ASYMMETRY: {cache} exists here but NOT on the peer.")
            print("    The peer will rebuild it from scratch on first run. That is not an")
            print("    error and produces no message -- it looks like a hang, for many")
            print("    minutes. Copy it over the fast link instead:")
            print(f"      rsync -a {cache}/ {peer}:{cache}/")
            print("")

    # A mismatched llama.cpp bundle kills a layer split at load; only ask when we have one.
    bundle_bad = False
    local_bundle = llama_bundle_identity()
    if local_bundle["present"]:
        bundles = compare_llama_bundles(local_bundle, peer_llama_bundle_identity(peer))
        for problem in bundles["problems"]:
            bundle_bad = True
            print(f"  LLAMA.CPP BUNDLE MISMATCH: {problem}")
            print("")
        for note in bundles["notes"]:
            print(f"  llama.cpp: {note}")
            print("")

    if bundle_bad:
        blockers.append("the llama.cpp bundles do not match")

    print(f"Measuring NCCL all-reduce {local} <-> {peer} (takes ~30s)...")
    result = diagnose_link(nccl_bandwidth(peer, local))
    print("")
    print(f"  {result['summary']}")
    if result["advice"]:
        print("")
        print(result["advice"])
    if result["verdict"] == "unknown":
        # Not a pass. The distributed health of the pair was not established, and a dead
        # engine is one of the reasons the measurement comes back unknown.
        blockers.append("the NCCL bandwidth could not be measured, so the link is unverified")
    elif result["verdict"] != "healthy":
        blockers.append(f"the link measured {result['verdict']}")
    if blockers:
        print("")
        print("  doctor FAILED:")
        for blocker in blockers:
            print(f"    - {blocker}")
        return 1
    return 0


def _cmd_status(benchmark: bool = False) -> int:
    if not is_dgx_spark():
        # Explicit, so this guard does not depend on cluster_state()'s internal ordering.
        print("state: not_spark")
        print(NOT_A_SPARK)
        return 0
    state = cluster_state()
    print(f"state: {state}")
    if state == "not_spark":
        print(NOT_A_SPARK)
        return 0
    info = discover_peers()
    print(f"cable present: {info['cable_present']}")
    for rail in info["cabled_rails"]:
        ips = ", ".join(rail["ipv4"]) or "no IPv4"
        print(f"  {rail['ib_device']:<14} {rail['netdev']:<16} mtu={rail['mtu']}  {ips}")
    for peer in info["mdns_peers"]:
        print(f"  peer seen: {peer['hostname']} at {peer['address']}")
    for problem in cluster_config_problems():
        print(f"  PLAN DRIFT: {problem}")
    events = link_carrier_events(info["cabled_rails"])
    shown = ", ".join(f"{k}={v}" for k, v in events.items() if v is not None)
    if shown:
        print(f"\n  link events: carrier_up_count {shown}")

    # Only a measurement can say whether the link is throttled (see HOTPLUG_NOTE).
    peer_ip = None
    for rail in info["configured"]:
        for addr in rail["ipv4"]:
            # Same resolution as peer_ip_for, not a second copy of the increment: this path had
            # the identical off-by-one and aimed the benchmark at a nonexistent host on node 1.
            peer_ip = peer_address_of(addr)
            break
        if peer_ip:
            break
    if benchmark and peer_ip:
        # Every configured rail, not the first one. NCCL_IB_MERGE_NICS hands the job both
        # PCIe functions, so measuring rails[0] and declaring the pair healthy said nothing
        # about the second one, which can be unaddressed, misconfigured or degraded and still
        # take half the traffic.
        degraded = False
        for rail in info["configured"]:
            local_ip = rail["ipv4"][0]
            rail_peer = peer_address_of(local_ip)
            if not rail_peer:
                print(f"  {rail['ib_device']}: no peer address derivable from {local_ip}")
                continue
            print(f"  measuring {rail['ib_device']} against {rail_peer} ...")
            health = link_health(rail_peer, rail["ib_device"], local_ip)
            if not health:
                print("  (could not measure; is ib_write_bw running on the peer?)")
            elif health["degraded"]:
                degraded = True
                print(
                    f"  MEASURED {health['gbps']:.2f} Gb/s on {rail['ib_device']} -- well "
                    f"below ~{health['expected_gbps']:.0f} Gb/s per rail."
                )
            else:
                print(f"  MEASURED {health['gbps']:.2f} Gb/s on {rail['ib_device']} -- healthy.")
        if degraded:
            print(f"  {HOTPLUG_NOTE}")
    elif peer_ip:
        print("  (run `unsloth spark status --benchmark` to measure the link;")
        print("   carrier counters alone cannot tell a throttled link from a healthy one)")
    return 0


def _cmd_env() -> int:
    if not is_dgx_spark():
        return 0
    for key, value in nccl_env().items():
        print(f"export {key}={value}")
    return 0


# Copying beats installing on the peer: far faster here, and a copy cannot drift. The venv
# path is RESOLVED because UNSLOTH_STUDIO_HOME moves it, and a hardcoded one copies a stale
# venv while still reporting "Peer now matches this node".
_DEFAULT_STUDIO_ROOT = Path.home() / ".unsloth" / "studio"


def provision_paths() -> Tuple[Tuple[str, str], ...]:
    # The llama.cpp bundle is NOT inside the venv (see llama_bundle_dir): skip it and the
    # peer keeps an older bundle, which llama-server rejects at load as a version mismatch.
    return (
        (str(_studio_root() / "unsloth_studio"), "Unsloth venv"),
        (str(llama_bundle_dir()), "llama.cpp prebuilt"),
        ("~/.cache/flashinfer", "FlashInfer JIT cache"),
        ("~/.cache/vllm/flashinfer_autotune_cache", "vLLM FlashInfer autotune cache"),
        ("~/.cache/vllm/torch_compile_cache", "vLLM torch.compile cache"),
    )


def peer_destination(path: str) -> str:
    """Where `path` should land on the peer: home-relative when it is under OUR home.

    The launch activates `$HOME/.unsloth/studio/unsloth_studio/bin/activate` on the peer, and
    `_BUNDLE_PROBE` expands `$HOME` there too. Provisioning sent the venv to our own expanded
    absolute path, so on a pair whose homes differ it wrote outside the peer's home -- or
    failed to -- and the launch then looked for the environment somewhere it had never been
    put. `_peer_relative_path` is the same mapping the bundle probe already uses."""
    expanded = osp.expanduser(path)
    return _peer_relative_path(Path(expanded))


_VENV_MARKERS = ("bin/activate", "bin/activate.csh", "bin/activate.fish", "pyvenv.cfg")


def venv_relocate_script(old_prefix: str, new_prefix: str) -> str:
    """Shell that repairs a venv copied to a different absolute path.

    A virtual environment is not relocatable on its own: `bin/activate` and `pyvenv.cfg`
    record `VIRTUAL_ENV` as an absolute path and every console script (`torchrun`, `unsloth`)
    carries an absolute shebang. Copied to a peer whose home differs, those all still point at
    the first Spark, so the remote rank exits before rendezvous with nothing that names the
    cause. Only the venv's own prefix is rewritten, and only in its own files."""
    old = shlex.quote(old_prefix.rstrip("/"))
    new = shlex.quote(new_prefix.rstrip("/"))
    files = " ".join(shlex.quote(m) for m in _VENV_MARKERS)
    return (
        "set -eu\n"
        f"v={new}\n"
        f"old={old}\n"
        '[ -d "$v" ] || exit 0\n'
        '[ "$v" = "$old" ] && exit 0\n'
        f"for f in {files}; do\n"
        '  [ -f "$v/$f" ] && sed -i "s|$old|$v|g" "$v/$f" || true\n'
        "done\n"
        # Console scripts only: a shebang is the first line and no binary in bin/ has one.
        'for f in "$v"/bin/*; do\n'
        '  [ -f "$f" ] || continue\n'
        '  head -c 2 "$f" 2>/dev/null | grep -q "^#!" || continue\n'
        '  sed -i "1s|$old|$v|" "$f" || true\n'
        "done\n"
    )


def _relocate_peer_venv(peer_ip: str, user: str, local: str, remote: str) -> str:
    """Run `venv_relocate_script` on the peer. Returns an error string, or "" on success."""
    script = venv_relocate_script(local, _peer_path(remote))
    try:
        res = subprocess.run(
            ["ssh", *_SSH_OPTS, f"{user}@{peer_ip}", "bash", "-s"],
            input = script,
            capture_output = True,
            text = True,
            timeout = 300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)[:200]
    return "" if res.returncode == 0 else (res.stderr or "").strip()[:200]


def venv_activate() -> str:
    """The peer's `activate`. Left as the literal `$HOME/...` so it expands on the PEER, whose
    home may differ; only a custom UNSLOTH_STUDIO_HOME forces an absolute path."""
    root = _studio_root()
    if root == _DEFAULT_STUDIO_ROOT:
        return "$HOME/.unsloth/studio/unsloth_studio/bin/activate"
    return str(root / "unsloth_studio" / "bin" / "activate")


def venv_activate_sh() -> str:
    """`venv_activate()` as ONE shell word, for the fragments that interpolate it.

    A custom UNSLOTH_STUDIO_HOME containing a space was returned bare into
    `[ -f {act} ] && . {act}`, so the test and the source command each split it into several
    words: the peer venv was not activated, and the launch failed on a `torchrun` that is not
    on a non-interactive SSH PATH. Double quotes rather than shlex.quote, because every call
    site sits inside `bash -c '...'` where a single-quoted word would end the outer quoting,
    and because `$HOME` in the default form still has to expand on the peer."""
    path = venv_activate()
    escaped = path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    if not path.startswith("$HOME"):
        escaped = escaped.replace("$", "\\$")
    return f'"{escaped}"'


# Below this a process is a CUDA context and scratch, not somebody's job.
PEER_BUSY_MIB = 96


def peer_gpu_busy(peer_ip: str, timeout: int = 25) -> Dict[str, Any]:
    """Is the peer's GPU holding someone's work? FAILS CLOSED: this gates a destructive
    ``rsync --delete``, and the probe is least likely to answer exactly when a job IS running,
    so "could not tell" must read as BUSY. Returns {"busy", "known", "processes", "reason"}."""
    out: Dict[str, Any] = {"busy": True, "known": False, "processes": [], "reason": ""}
    if not peer_ip:
        out["reason"] = "no peer address"
        return out
    if not shutil.which("ssh"):
        out["reason"] = "ssh unavailable, so the peer's GPU state cannot be checked"
        return out
    user = _ssh_user()
    # RC separates "nvidia-smi ran and listed nothing" (idle) from "it did not run"
    # (unknown); without it both are an empty string and idle is the fail-open mistake.
    remote = (
        "nvidia-smi --query-compute-apps=pid,used_gpu_memory "
        "--format=csv,noheader,nounits; echo RC=$?"
    )
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"{user}@{peer_ip}",
                remote,
            ],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception as exc:
        out["reason"] = f"could not reach the peer to check its GPU ({str(exc)[:80]})"
        return out
    lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
    rc = next((l[3:] for l in lines if l.startswith("RC=")), None)
    if rc is None or rc != "0":
        out["reason"] = f"nvidia-smi did not run on the peer (rc={rc!r}); treating the GPU as BUSY"
        return out
    # A row that cannot be read is not an absent process. nvidia-smi reports `[N/A]` for
    # used_memory in real situations, and skipping such a row and then declaring the peer idle
    # inverts this function's whole contract: provisioning would overwrite a venv the process
    # behind that row is running out of. Unreadable means unverifiable, which means busy.
    unreadable = []
    for line in lines:
        if line.startswith("RC=") or line.lower().startswith("pid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            unreadable.append(line)
            continue
        mem = parts[1].replace("MiB", "").replace("MB", "").strip()
        try:
            mib = int(float(mem))
        except ValueError:
            unreadable.append(line)
            continue
        if mib >= PEER_BUSY_MIB:
            out["processes"].append({"pid": int(parts[0]), "used_mib": mib})
    if unreadable:
        out["known"] = False
        out["busy"] = True
        out["reason"] = (
            f"{len(unreadable)} process row(s) from nvidia-smi could not be read "
            f"({unreadable[0][:80]!r}); treating the GPU as BUSY"
        )
        return out
    out["known"] = True
    out["busy"] = bool(out["processes"])
    out["reason"] = (
        f"{len(out['processes'])} compute process(es) resident"
        if out["busy"]
        else "no compute processes on the peer GPU"
    )
    return out


# rsync over ssh is bound by ssh's CPU, well below the disk floor, so `provision` runs an
# ephemeral `rsync --daemon` on the peer with FAST_MAX_WORKERS clients over disjoint subsets.
# SECURITY: the bytes travel UNENCRYPTED. Acceptable ONLY because the rail is a
# point-to-point cable with no other host on it AND the daemon is locked to it: bound to the
# peer's rail address, `hosts allow` the one local rail address, a random one-shot user and
# secret that never reach a command line, dying with the command, and refused unless the peer
# is a private address inside the local rail's /24.
# SYMLINKS: without root the daemon cannot chroot, and a non-chrooted module rewrites absolute
# symlink targets, so the workers carry REGULAR FILES ONLY and the ssh rsync runs last as the
# finaliser for symlinks, directory metadata and --delete.
FAST_ENV = "UNSLOTH_SPARK_PROVISION_FAST"
FAST_MAX_WORKERS = 4
FAST_PORT_RANGE = (40000, 59999)
# A worker idle this long is retried on a fresh connection: with four streams, roughly one
# flow in ten is blackholed in both directions with TCP in RTO backoff at 51 s, and a retry
# is a new 5-tuple. 30 s is long against a legitimate pause and short against a stall.
FAST_IO_TIMEOUT = 30
FAST_WORKER_ATTEMPTS = 3
# Backstop only: the daemon is stopped from a `finally`; this covers a SIGKILL here.
FAST_DAEMON_MAX_SECONDS = 4 * 3600
_FAST_UP_MARKER = "UNSLOTH_DAEMON_UP"
_FAST_LEFT_MARKER = "UNSLOTH_RSYNC_LEFT"
_FAST_ALIVE_MARKER = "UNSLOTH_DAEMON_ALIVE"


def _ssh_argv(user: str, peer_ip: str) -> List[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=8",
        f"{user}@{peer_ip}",
    ]


def _peer_path(path: str) -> str:
    """`~/x` as `$HOME/x`: a quoted `~` does not expand, and it must expand on the PEER,
    whose home may differ from ours."""
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + path[2:]
    return path


def _unquoted_heredoc(text: str) -> str:
    """Escape for an unquoted heredoc so that only the deliberate `$HOME` expands."""
    out = text.replace("\\", "\\\\").replace("$", "\\$").replace("`", "\\`")
    return out.replace("\\$HOME/", "$HOME/").replace("\\$HOME\n", "$HOME\n")


def fast_port() -> int:
    lo, hi = FAST_PORT_RANGE
    return lo + secrets.randbelow(hi - lo + 1)


def rail_local_address(peer_ip: str, rails: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Our address on the rail that carries `peer_ip`: the one in the same /24."""
    try:
        peer = ipaddress.ip_address(peer_ip)
    except ValueError:
        return None
    rails = rails if rails is not None else cabled_rails()
    for rail in rails:
        for addr in rail.get("ipv4", []):
            try:
                net = ipaddress.ip_network(f"{addr}/24", strict = False)
            except ValueError:
                continue
            if peer in net and addr != peer_ip:
                return addr
    return None


def fast_path_decision(
    peer_ip: str,
    no_fast: bool = False,
    env: Optional[Dict[str, str]] = None,
    local_ip: Optional[str] = None,
) -> Dict[str, Any]:
    """May bulk bytes go through the unencrypted rail daemon? {"ok", "reason", "local_ip"}.
    Every refusal names its reason so `provision` can print why it fell back to ssh. Order
    is cheapest-first: the flag and platform gates precede the rail lookup, which forks `ip`."""
    env = os.environ if env is None else env

    def no(reason: str) -> Dict[str, Any]:
        return {"ok": False, "reason": reason, "local_ip": local_ip}

    if no_fast:
        return no("--no-fast")
    flag = env.get(FAST_ENV, "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return no(f"{FAST_ENV}={env.get(FAST_ENV)}")
    if platform.system() != "Linux":
        return no("not Linux")
    if not is_dgx_spark():
        return no("not a DGX Spark")
    if local_ip is None:
        local_ip = rail_local_address(peer_ip)
    if not local_ip:
        return no("no local address on the peer's rail")
    try:
        peer = ipaddress.ip_address(peer_ip)
        local = ipaddress.ip_address(local_ip)
    except ValueError:
        return no("peer or local address is not an IP address")
    if peer.version != 4 or local.version != 4:
        return no("rail addresses are not IPv4")
    if not peer.is_private or not local.is_private:
        return no(f"{peer_ip} is not a private address")
    if peer == local:
        return no("peer address is our own")
    if local not in ipaddress.ip_network(f"{peer_ip}/24", strict = False):
        return no(f"{local_ip} and {peer_ip} are not in the same /24")
    # Sharing a /24 is not the same as being alone on a wire. After `setup --nodes N --switched`
    # every node shares these subnets, so the same-subnet test above still passes while the
    # fabric carries other hosts. The SECURITY note on FAST_ENV is explicit that the plaintext
    # transfer is acceptable ONLY because the rail is a point-to-point cable with nothing else
    # on it, and `hosts allow` plus a one-shot secret restrict access without providing any
    # confidentiality against another participant. So the persisted plan decides, and anything
    # other than a two-node direct rail falls back to ssh.
    config = load_config()
    if config.get("switched"):
        return no("the rails are switched, not point-to-point; bytes would cross in plaintext")
    try:
        configured_nodes = int(config.get("n_nodes") or 0)
    except (TypeError, ValueError):
        configured_nodes = 0
    if configured_nodes > 2:
        return no(
            f"{configured_nodes} nodes share these rails, so they are not point-to-point; "
            f"bytes would cross in plaintext"
        )
    return {"ok": True, "reason": "direct rail", "local_ip": local_ip}


def rsync_daemon_config(
    modules: Dict[str, str],
    bind_ip: str,
    port: int,
    hosts_allow: str,
    auth_user: str,
    work_dir: str,
) -> str:
    """rsyncd.conf for the ephemeral daemon. `modules` maps name -> path ON THE PEER, one
    per destination root, writable only by `auth_user` from the one `hosts_allow` address."""
    lines = [
        f"address = {bind_ip}",
        f"port = {port}",
        f"pid file = {work_dir}/rsyncd.pid",
        f"lock file = {work_dir}/rsyncd.lock",
        f"log file = {work_dir}/rsyncd.log",
        "use chroot = no",
        "munge symlinks = no",
        "reverse lookup = no",
        f"max connections = {FAST_MAX_WORKERS}",
        # Twice the client's, so a stuck flow is the client's call to drop and retry.
        f"timeout = {2 * FAST_IO_TIMEOUT}",
    ]
    for name, path in modules.items():
        lines += [
            "",
            f"[{name}]",
            f"    path = {path}",
            "    read only = no",
            f"    hosts allow = {hosts_allow}",
            f"    auth users = {auth_user}",
            f"    secrets file = {work_dir}/rsyncd.secrets",
        ]
    return "\n".join(lines) + "\n"


def daemon_files_script(
    config: str, auth_user: str, secret: str, dest_paths: List[str], work_dir: str
) -> str:
    """The file-writing half of the peer script: a 700 work dir, the 600 secrets file, the
    config, and the module destination roots. The secret travels inside the ssh session
    (the script is stdin) and lands only in that 600 file; it is never an argument."""
    q = _unquoted_heredoc
    mkdirs = " ".join(f'"{q(p)}"' for p in dest_paths)
    return (
        "set -eu\n"
        "umask 077\n"
        f"d='{work_dir}'\n"
        'mkdir -m 700 "$d"\n'
        "cat > \"$d/rsyncd.secrets\" <<'UNSLOTH_EOF'\n"
        f"{auth_user}:{secret}\n"
        "UNSLOTH_EOF\n"
        'chmod 600 "$d/rsyncd.secrets"\n'
        'cat > "$d/rsyncd.conf" <<UNSLOTH_EOF\n'
        f"{q(config)}"
        "UNSLOTH_EOF\n"
        f"mkdir -p {mkdirs}\n"
    )


def daemon_setup_script(
    config: str, auth_user: str, secret: str, dest_paths: List[str], work_dir: str, port: int
) -> str:
    """The bash that runs on the peer (`ssh bash -s`, script on stdin). Reports the daemon up
    only once its pid file exists AND the port listens, so a bind failure is an error here
    rather than a hang later."""
    return daemon_files_script(config, auth_user, secret, dest_paths, work_dir) + (
        f"setsid timeout {FAST_DAEMON_MAX_SECONDS} rsync --daemon --no-detach "
        '--config="$d/rsyncd.conf" </dev/null >/dev/null 2>&1 &\n'
        'echo $! > "$d/rsyncd.pgid"\n'
        "_listening() { command -v ss >/dev/null 2>&1 || return 0; "
        f'ss -Hltn "sport = :{port}" 2>/dev/null | grep -q .; }}\n'
        "for i in $(seq 1 50); do\n"
        '  if [ -s "$d/rsyncd.pid" ] && kill -0 "$(cat "$d/rsyncd.pid")" 2>/dev/null && _listening; then\n'
        f'    echo "{_FAST_UP_MARKER} $(cat "$d/rsyncd.pid")"; exit 0\n'
        "  fi\n"
        "  sleep 0.1\n"
        "done\n"
        'echo "rsync daemon did not come up" >&2\n'
        'cat "$d/rsyncd.log" >&2 2>/dev/null || true\n'
        'kill -TERM -- -"$(cat "$d/rsyncd.pgid")" 2>/dev/null || true\n'
        'rm -rf "$d"\n'
        "exit 1\n"
    )


def daemon_stop_script(work_dir: str) -> str:
    """Kill the daemon (its whole process group, so in-flight receivers go too), remove
    the temp dir, and report what `pgrep -x rsync` still sees on the peer."""
    return (
        f"d='{work_dir}'\n"
        'pid=$(cat "$d/rsyncd.pid" 2>/dev/null || true)\n'
        'pgid=$(cat "$d/rsyncd.pgid" 2>/dev/null || true)\n'
        '[ -n "$pgid" ] && kill -TERM -- -"$pgid" 2>/dev/null || true\n'
        '[ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true\n'
        'for i in $(seq 1 25); do [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done\n'
        'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then\n'
        '  kill -KILL "$pid" 2>/dev/null || true\n'
        '  [ -n "$pgid" ] && kill -KILL -- -"$pgid" 2>/dev/null || true\n'
        "  sleep 0.2\n"
        "fi\n"
        'rm -rf "$d"\n'
        f'[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "{_FAST_ALIVE_MARKER} $pid"\n'
        f"echo \"{_FAST_LEFT_MARKER} $(pgrep -x rsync | tr '\\n' ' ')\"\n"
        "exit 0\n"
    )


def start_peer_rsync_daemon(
    peer_ip: str,
    local_ip: str,
    user: str,
    modules: Dict[str, str],
    timeout: int = 60,
) -> Dict[str, Any]:
    """Start the ephemeral daemon on the peer; raises RuntimeError when it did not. The
    secret lives only in the returned dict and in the peer's 600 file."""
    port = fast_port()
    auth_user = "unsloth-" + secrets.token_hex(4)
    secret = secrets.token_urlsafe(24)
    work_dir = "/tmp/unsloth-provision-" + secrets.token_hex(6)
    config = rsync_daemon_config(modules, peer_ip, port, local_ip, auth_user, work_dir)
    script = daemon_setup_script(config, auth_user, secret, list(modules.values()), work_dir, port)

    # Every failure below is cleaned up from here, because the caller has no descriptor to
    # clean up with: an ssh that times out or loses its reply AFTER the peer ran the detached
    # `setsid ... rsync --daemon` left a writable daemon, its credentials file and its temp
    # directory alive for the four-hour backstop, against the promise that they die with the
    # command. `work_dir` is known before the connection is made, which is what makes the
    # cleanup possible at all.
    def _abandon() -> None:
        stop_peer_rsync_daemon(
            {"ssh_user": user, "peer_ip": peer_ip, "work_dir": work_dir, "pid": None}
        )

    try:
        proc = subprocess.run(
            _ssh_argv(user, peer_ip) + ["bash", "-s"],
            input = script,
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except BaseException:
        _abandon()
        raise
    out = proc.stdout or ""
    up = next((l for l in out.splitlines() if l.startswith(_FAST_UP_MARKER)), None)
    if proc.returncode != 0 or up is None:
        err = (proc.stderr or "").strip()[:200] or f"rc={proc.returncode}"
        _abandon()
        raise RuntimeError(f"rsync daemon did not start on {peer_ip}: {err}")
    pid = _int_or_none(up.split(None, 1)[1].strip()) if " " in up else None
    return {
        "peer_ip": peer_ip,
        "local_ip": local_ip,
        "ssh_user": user,
        "port": port,
        "auth_user": auth_user,
        "secret": secret,
        "work_dir": work_dir,
        "pid": pid,
        "modules": dict(modules),
    }


def stop_peer_rsync_daemon(daemon: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """Stop the daemon and remove its temp dir; never raises. Returns {"stopped", "left",
    "error"}, where `left` is every rsync pid still on the peer, reported as a FACT: another
    may legitimately be running, but ours must not be, and "stopped" is False if it is."""
    out: Dict[str, Any] = {"stopped": False, "left": [], "error": ""}
    try:
        proc = subprocess.run(
            _ssh_argv(daemon["ssh_user"], daemon["peer_ip"]) + ["bash", "-s"],
            input = daemon_stop_script(daemon["work_dir"]),
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception as exc:
        out["error"] = str(exc)[:200]
        return out
    text = proc.stdout or ""
    alive = any(l.startswith(_FAST_ALIVE_MARKER) for l in text.splitlines())
    for line in text.splitlines():
        if line.startswith(_FAST_LEFT_MARKER):
            out["left"] = [int(p) for p in line.split()[1:] if p.isdigit()]
    out["stopped"] = proc.returncode == 0 and not alive
    if proc.returncode != 0:
        out["error"] = (proc.stderr or "").strip()[:200] or f"rc={proc.returncode}"
    elif alive:
        out["error"] = f"daemon pid {daemon.get('pid')} survived SIGKILL"
    return out


def _regular_files(root: str) -> List[Tuple[int, str]]:
    """(size, relative path) of every regular file under `root`; symlinks and special
    files are left to the ssh finaliser (see the daemon notes above)."""
    files: List[Tuple[int, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = osp.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                files.append((st.st_size, osp.relpath(full, root)))
    return files


def provision_work_split(
    root: str,
    max_workers: int = FAST_MAX_WORKERS,
    files: Optional[List[Tuple[int, str]]] = None,
) -> List[List[str]]:
    """Disjoint, byte-balanced subsets of the regular files under `root`: largest file onto
    the lightest worker, so a sharded GGUF spreads and small files fill in around it. One
    file is one worker, because rsync cannot split a single file across streams."""
    files = _regular_files(root) if files is None else list(files)
    if not files:
        return []
    n = max(1, min(max_workers, len(files)))
    files.sort(reverse = True)
    buckets: List[List[str]] = [[] for _ in range(n)]
    loads = [0] * n
    for size, rel in files:
        i = loads.index(min(loads))
        buckets[i].append(rel)
        loads[i] += size
    return [b for b in buckets if b]


def _fast_bulk_copy(
    local_root: str,
    daemon: Dict[str, Any],
    module: str,
    timeout: int = 3600,
) -> Tuple[bool, str, int, int]:
    """Push the regular files under `local_root` through the daemon with parallel
    workers. Returns (ok, error, bytes, workers, retries)."""
    files = _regular_files(local_root)
    buckets = provision_work_split(local_root, files = files)
    total = sum(size for size, _ in files)
    if not buckets:
        return True, "", 0, 0, 0
    url = f"rsync://{daemon['auth_user']}@{daemon['peer_ip']}:{daemon['port']}/{module}/"
    # The secret goes through the environment: never on a command line, never in `ps`.
    env = dict(os.environ)
    env["RSYNC_PASSWORD"] = daemon["secret"]
    with tempfile.TemporaryDirectory(prefix = "unsloth-provision-") as tmp:
        cmds = []
        for i, bucket in enumerate(buckets):
            listing = osp.join(tmp, f"files{i}")
            with open(listing, "wb") as fh:
                fh.write(b"\0".join(p.encode("utf-8", "surrogateescape") for p in bucket) + b"\0")
            # -W whole files: the delta algorithm checksums on both ends, slower than this wire.
            cmds.append(
                [
                    "rsync",
                    "-aW",
                    "--from0",
                    f"--files-from={listing}",
                    "--contimeout=15",
                    f"--timeout={FAST_IO_TIMEOUT}",
                    local_root + "/",
                    url,
                ]
            )

        def run(cmd: List[str]) -> Tuple[Any, int]:
            attempts = 0
            while True:
                attempts += 1
                proc = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout, env = env)
                if proc.returncode == 0 or attempts >= FAST_WORKER_ATTEMPTS:
                    return proc, attempts - 1

        with ThreadPoolExecutor(max_workers = len(cmds)) as pool:
            outcomes = list(pool.map(run, cmds))
    retries = sum(r for _, r in outcomes)
    bad = [p for p, _ in outcomes if p.returncode != 0]
    if bad:
        err = (bad[0].stderr or "").strip()[:200] or f"rc={bad[0].returncode}"
        return False, err, total, len(cmds), retries
    return True, "", total, len(cmds), retries


def _raise_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def provision_peer(
    peer_ip: str,
    dry_run: bool = False,
    delete: bool = False,
    force: bool = False,
    no_fast: bool = False,
) -> Dict[str, Any]:
    """Copy the environment and warm caches to the peer. rsync rather than reinstall: identical
    bytes, no dependency drift, at link speed. Bulk bytes go through the rail daemon when
    ``fast_path_decision`` allows it, and the ssh rsync is always the finaliser."""
    results: Dict[str, Any] = {
        "copied": [],
        "skipped": [],
        "failed": [],
        "dry_run": dry_run,
        "refused": "",
        "peer_gpu": None,
        "delete": delete,
        # "errors" are paths that fell back to ssh mid-way, "retries" counts worker
        # reconnects after a stalled flow, "timings" is (label, mode, bytes, seconds, workers).
        "fast": {
            "used": False,
            "reason": "",
            "errors": [],
            "stop": None,
            "port": None,
            "retries": 0,
        },
        "timings": [],
    }
    if not shutil.which("rsync") or not shutil.which("ssh"):
        results["failed"].append(("rsync/ssh", "not installed"))
        return results
    # A dry run reads nothing on the peer and writes nothing, so it needs no gate.
    if not dry_run:
        gpu = peer_gpu_busy(peer_ip)
        results["peer_gpu"] = gpu
        if gpu["busy"] and not force:
            held = (
                ", ".join(f"pid {p['pid']} ({p['used_mib']} MiB)" for p in gpu["processes"])
                or "state unknown"
            )
            results["refused"] = (
                f"peer {peer_ip} is BUSY or unverifiable ({gpu['reason']}: {held}). "
                f"Refusing to overwrite a venv a running job may be executing out of. "
                f"Wait for the job, or pass force=True if you are certain."
            )
            return results
    user = _ssh_user()
    targets = []
    for path, label in provision_paths():
        local = osp.expanduser(path)
        if not osp.isdir(local):
            results["skipped"].append((label, "not present locally"))
            continue
        targets.append((path, label, local))

    daemon: Optional[Dict[str, Any]] = None
    fast = results["fast"]
    if targets and not dry_run:
        decision = fast_path_decision(peer_ip, no_fast = no_fast)
        if decision["ok"]:
            # peer_destination, not the raw path: an absolute path under OUR home is not a
            # place on the peer, whose home may differ, and the launch looks for the venv
            # under the PEER's $HOME.
            modules = {
                f"m{i}": _peer_path(peer_destination(path))
                for i, (path, _, _) in enumerate(targets)
            }
            try:
                daemon = start_peer_rsync_daemon(peer_ip, decision["local_ip"], user, modules)
                fast["port"] = daemon["port"]
            except Exception as exc:
                fast["reason"] = f"daemon did not start ({str(exc)[:160]}); using ssh"
        else:
            fast["reason"] = decision["reason"]

    # The daemon must not outlive this call, and SIGTERM's default disposition would skip
    # the `finally`. Only the main thread may set handlers; elsewhere `finally` alone does.
    prev_term = None
    if daemon is not None:
        try:
            prev_term = signal.signal(signal.SIGTERM, _raise_interrupt)
        except (ValueError, OSError):
            prev_term = None
    try:
        for i, (path, label, local) in enumerate(targets):
            started = time.monotonic()
            mode, moved, workers = "ssh", 0, 0
            if daemon is not None:
                ok, err, moved, workers, retries = _fast_bulk_copy(local, daemon, f"m{i}")
                fast["retries"] += retries
                if ok:
                    mode = "fast"
                    fast["used"] = True
                else:
                    # The ssh pass below resumes whatever the workers left; only time is lost.
                    fast["errors"].append((label, err))
            # rsync creates only the LAST component of the destination, so a brand-new peer
            # with no ~/.unsloth/studio fails; create the parent remotely first.
            remote = peer_destination(path)
            remote_parent = _peer_path(osp.dirname(remote))
            # `--rsync-path` runs on the PEER before rsync starts, so `--dry-run` does not
            # suppress it: a dry run was creating the directories it was only meant to report.
            # Under a dry run the wrapper is dropped, and a missing parent is reported as
            # something the real run would create rather than as a failure, since with nothing
            # on the far side there is also nothing for rsync to compare against.
            if dry_run:
                if not _peer_dir_exists(peer_ip, user, remote_parent):
                    results["skipped"].append((label, f"would create {remote_parent}"))
                    results["timings"].append(
                        (label, mode, moved, time.monotonic() - started, workers),
                    )
                    continue
                rsync_path = "rsync"
            else:
                rsync_path = f'mkdir -p "{remote_parent}" && rsync'
            # --delete is OFF by default: a stale file costs disk, while deleting a live one
            # takes a running interpreter out from under a job mid-flight.
            cmd = [
                "rsync",
                "-a",
                "--rsync-path",
                rsync_path,
                "-e",
                "ssh -o BatchMode=yes -o StrictHostKeyChecking=no",
                local + "/",
                f"{user}@{peer_ip}:{remote}/",
            ]
            if delete:
                cmd.insert(2, "--delete")
            if dry_run:
                cmd.insert(1, "--dry-run")
            try:
                r = subprocess.run(cmd, capture_output = True, text = True, timeout = 3600)
                if r.returncode == 0:
                    results["copied"].append((label, remote))
                    # A venv that landed at a different absolute path carries the first
                    # Spark's `VIRTUAL_ENV` and console-script shebangs, so `torchrun` there
                    # runs -- or fails to run -- out of a path that does not exist on the
                    # peer. Repaired in place; a no-op when the two paths agree.
                    if not dry_run and label == "Unsloth venv":
                        error = _relocate_peer_venv(peer_ip, user, local, remote)
                        if error:
                            results["failed"].append((f"{label} (relocation)", error))
                else:
                    results["failed"].append((label, (r.stderr or "").strip()[:200]))
            except Exception as exc:
                results["failed"].append((label, str(exc)[:200]))
            results["timings"].append(
                (label, mode, moved, time.monotonic() - started, workers),
            )
    finally:
        if daemon is not None:
            fast["stop"] = stop_peer_rsync_daemon(daemon)
            if not fast["stop"]["stopped"]:
                results["failed"].append(
                    ("rsync daemon on the peer", fast["stop"]["error"] or "did not stop"),
                )
            if prev_term is not None:
                try:
                    signal.signal(signal.SIGTERM, prev_term)
                except (ValueError, OSError):
                    pass
    return results


def _gb_per_s(nbytes: int, seconds: float) -> str:
    return f"{nbytes / 1e9 / seconds:.2f} GB/s" if seconds > 0 and nbytes else "n/a"


def _cmd_provision(
    dry_run: bool = False,
    delete: bool = False,
    force: bool = False,
    no_fast: bool = False,
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to provision.")
        return 0
    peer = peer_ip_for()
    if not peer:
        print("No configured peer. Run `unsloth spark setup` first.")
        return 1
    print(
        f"Provisioning peer {peer} over the ConnectX link" f"{' (dry run)' if dry_run else ''}..."
    )
    print("  Copying rather than installing: HuggingFace measures ~20 KB/s from these")
    print("  boxes while this link does ~1 GB/s, and copying cannot drift.")
    try:
        res = provision_peer(peer, dry_run = dry_run, delete = delete, force = force, no_fast = no_fast)
    except KeyboardInterrupt:
        print("\n  Interrupted. The rsync daemon on the peer, if any, has been stopped.")
        return 130
    if res["refused"]:
        print(f"  REFUSED: {res['refused']}")
        print("  (nothing was copied, nothing was deleted)")
        return 1
    fast = res["fast"]
    if fast["used"]:
        print(
            f"  fast path: rsync daemon on {peer}:{fast['port']}, unencrypted over the direct "
            f"rail cable, up to {FAST_MAX_WORKERS} workers; ssh finalises each path"
        )
    elif not dry_run and fast["reason"]:
        print(f"  ssh path: {fast['reason']}")
    if fast.get("retries"):
        print(f"  note    {fast['retries']} worker reconnect(s) after a stalled flow on the rail")
    for label, why in fast["errors"]:
        print(f"  note    {label}: fast path failed ({why}); finished over ssh")
    timings = {t[0]: t for t in res["timings"]}
    for label, path in res["copied"]:
        detail = ""
        if label in timings:
            _, mode, moved, seconds, workers = timings[label]
            if moved:
                detail = (
                    f"  {moved / 2**30:.1f} GiB in {seconds:.1f} s, {_gb_per_s(moved, seconds)}"
                    f" ({mode}, {workers} workers)"
                )
            elif not dry_run:
                detail = f"  {seconds:.1f} s ({mode})"
        print(f"  ok      {label} ({path}){detail}")
    for label, why in res["skipped"]:
        print(f"  skip    {label}: {why}")
    for label, why in res["failed"]:
        print(f"  FAILED  {label}: {why}")
    stop = fast["stop"]
    if stop is not None:
        if stop["left"]:
            print(f"  note    rsync still running on the peer (not ours): pids {stop['left']}")
        elif stop["stopped"]:
            print("  daemon  stopped on the peer; `pgrep -x rsync` there finds nothing")
    if res["failed"]:
        return 1
    print("\n  Peer now matches this node. Verify with: unsloth doctor")
    return 0


# 128 GiB physical minus ~6.3 GiB firmware-reserved.
SPARK_USABLE_GIB = 121.69
# KV cache, compute buffers and fragmentation beyond the weights. Context dominates and
# grows fast, so this is deliberately not tight.
SERVE_OVERHEAD_GIB = 8.0


_GGUF_SHARD_RE = re.compile(r"^(?P<series>.+)-\d{5}-of-\d{5}\.gguf$")


def _weight_group(filename: str) -> Optional[str]:
    """Which mutually exclusive artifact a weight file belongs to, or None if it is not one.

    Shards of one model share a group and are summed; alternatives get their own group and
    must never be. `a-00001-of-00003.gguf` and `a-00002-of-00003.gguf` are one model,
    `a-Q4_K_M.gguf` and `a-Q8_0.gguf` are two, and `model.safetensors` beside
    `pytorch_model.bin` is one model written twice."""
    if filename.endswith(".gguf"):
        match = _GGUF_SHARD_RE.match(filename)
        return f"gguf:{match.group('series')}" if match else f"gguf:{filename[:-5]}"
    if filename.endswith(".safetensors"):
        return "safetensors"
    if filename.endswith(".bin"):
        return "bin"
    return None


def _size_one_group(files: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Total bytes of the ONE model in `files` [(name, path)], or why there is not exactly one.

    Summing everything was wrong in both directions. A directory of alternative quantizations
    reported their combined size, so a model that fits each Spark was planned as too large for
    the pair; a repo carrying both safetensors and the legacy .bin counted the same weights
    twice. Neither is recoverable by the caller, because a plausible number is indistinguishable
    from a right one, so an ambiguous directory is refused rather than guessed at."""
    groups: Dict[str, int] = {}
    seen: set = set()
    for name, full in files:
        group = _weight_group(name)
        if group is None:
            continue
        try:
            real = osp.realpath(full)
            if real in seen:
                continue
            seen.add(real)
            groups[group] = groups.get(group, 0) + osp.getsize(real)
        except OSError:
            pass
    # The same weights in two formats: transformers reads the safetensors, so does Unsloth.
    if "safetensors" in groups:
        groups.pop("bin", None)
    if not groups:
        return {"bytes": None, "why": "no weight files found"}
    if len(groups) > 1:
        names = ", ".join(sorted(groups))
        return {
            "bytes": None,
            "why": (
                f"{len(groups)} alternative models are present ({names}); name one file so "
                f"the size is the model that will actually load"
            ),
        }
    return {"bytes": next(iter(groups.values())), "why": ""}


def _hf_snapshot_dir(root: str) -> Optional[str]:
    """The snapshot `refs/main` points at, which is the one that loads. Walking the whole
    `models--...` tree also counted superseded revisions, overstating an updated repo."""
    ref = osp.join(root, "refs", "main")
    try:
        with open(ref, "r", encoding = "utf-8") as handle:
            commit = handle.read().strip()
    except OSError:
        commit = ""
    if commit:
        snapshot = osp.join(root, "snapshots", commit)
        if osp.isdir(snapshot):
            return snapshot
    snapshots = osp.join(root, "snapshots")
    try:
        entries = [osp.join(snapshots, d) for d in os.listdir(snapshots)]
    except OSError:
        return None
    dirs = [d for d in entries if osp.isdir(d)]
    return dirs[0] if len(dirs) == 1 else None


def model_size_report(target: str) -> Dict[str, Any]:
    """`{"gib": float|None, "why": str}`: the size of the ONE model `target` names, or why
    that cannot be answered. None rather than a guess: a wrong size produces confidently
    wrong deployment advice, and the caller has no way to tell the two apart."""
    path = osp.expanduser(target)
    if osp.isfile(path):
        # Naming `model-00001-of-00002.gguf` names the whole series: llama.cpp opens the
        # first shard and loads the rest beside it. Reporting one shard's size made a model
        # that needs a split look like one that fits, and the load is where that is found out.
        match = _GGUF_SHARD_RE.match(osp.basename(path))
        if match:
            series = sorted(
                glob.glob(osp.join(osp.dirname(path) or ".", f"{match.group('series')}-*.gguf"))
            )
            expected = int(osp.basename(path).rsplit("-of-", 1)[1].split(".")[0])
            if len(series) != expected:
                return {
                    "gib": None,
                    "why": (
                        f"{target} is shard 1 of {expected} and {len(series)} are present; "
                        f"the missing shards would not load either"
                    ),
                }
            try:
                return {"gib": sum(osp.getsize(f) for f in series) / 2**30, "why": ""}
            except OSError as exc:
                return {"gib": None, "why": f"could not read the shard series: {exc}"}
        try:
            return {"gib": osp.getsize(path) / 2**30, "why": ""}
        except OSError as exc:
            return {"gib": None, "why": f"could not read {path}: {exc}"}
    root = path if osp.isdir(path) else None
    if root is None:
        # A repo id: read the HF cache rather than the network, which is ~20 KB/s here.
        cache = osp.expanduser("~/.cache/huggingface/hub")
        slug = "models--" + target.replace("/", "--")
        repo = osp.join(cache, slug)
        if not osp.isdir(repo):
            return {"gib": None, "why": f"{target} is not a file, a directory or in the HF cache"}
        root = _hf_snapshot_dir(repo)
        if root is None:
            return {
                "gib": None,
                "why": f"the HF cache for {target} has no single current snapshot to size",
            }
    files = []
    for dirpath, _, names in os.walk(root):
        files.extend((name, osp.join(dirpath, name)) for name in names)
    result = _size_one_group(files)
    if result["bytes"] is None:
        return {"gib": None, "why": f"{target}: {result['why']}"}
    return {"gib": result["bytes"] / 2**30, "why": ""}


def model_size_gib(target: str) -> Optional[float]:
    """Best-effort size of a model, from a file, a directory, or the HF cache."""
    return model_size_report(target)["gib"]


# GGUF value types, by the spec's numbering. Only the fixed-width ones need a size here;
# strings and arrays carry their own length.
_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_UNPACK = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _gguf_read_value(
    handle,
    value_type: int,
    depth: int = 0,
):
    """One GGUF value. Arrays of fixed-width elements are seeked over rather than read: the
    tokenizer arrays are the bulk of the header and none of them is wanted here."""
    if value_type in _GGUF_FIXED:
        raw = handle.read(_GGUF_FIXED[value_type])
        if len(raw) < _GGUF_FIXED[value_type]:
            raise ValueError("truncated GGUF header")
        return struct.unpack(_GGUF_UNPACK[value_type], raw)[0]
    if value_type == 8:
        (length,) = struct.unpack("<Q", handle.read(8))
        return handle.read(length).decode("utf-8", "replace")
    if value_type == 9:
        if depth:
            raise ValueError("nested GGUF array")
        element, count = struct.unpack("<IQ", handle.read(12))
        if element in _GGUF_FIXED and count > 64:
            handle.seek(_GGUF_FIXED[element] * count, 1)
            return []
        return [_gguf_read_value(handle, element, depth + 1) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {value_type}")


def gguf_metadata(path: str) -> Dict[str, Any]:
    """The GGUF key/value header, or `{}` when `path` is not a readable GGUF.

    Only the header is read; tensor data is never touched. Long fixed-width arrays come back
    empty because nothing here wants them, and reading a 150k-entry token list to discard it
    would put seconds into a planning command."""
    try:
        with open(osp.expanduser(path), "rb") as handle:
            magic, version = struct.unpack("<4sI", handle.read(8))
            if magic != b"GGUF" or version not in (2, 3):
                return {}
            _, kv_count = struct.unpack("<QQ", handle.read(16))
            out: Dict[str, Any] = {}
            for _ in range(min(kv_count, 4096)):
                (key_len,) = struct.unpack("<Q", handle.read(8))
                key = handle.read(key_len).decode("utf-8", "replace")
                (value_type,) = struct.unpack("<I", handle.read(4))
                out[key] = _gguf_read_value(handle, value_type)
            return out
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return {}


def gguf_kv_bytes_per_token(meta: Dict[str, Any]) -> Optional[float]:
    """Bytes of KV cache one token costs, or None when the header does not say.

    K and V, every layer, at the f16 default llama-server uses unless told otherwise. None
    rather than a default: the whole point of the number is to decide whether a context fits,
    and a made-up one decides it wrongly with no way for the caller to tell."""
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    layers = meta.get(f"{arch}.block_count")
    heads_kv = meta.get(f"{arch}.attention.head_count_kv")
    key_len = meta.get(f"{arch}.attention.key_length")
    value_len = meta.get(f"{arch}.attention.value_length")
    if key_len is None or value_len is None:
        embedding = meta.get(f"{arch}.embedding_length")
        heads = meta.get(f"{arch}.attention.head_count")
        if not isinstance(embedding, int) or not isinstance(heads, int) or heads <= 0:
            return None
        key_len = value_len = embedding // heads
    if not isinstance(key_len, int) or not isinstance(value_len, int):
        return None
    per_layer = float(key_len + value_len) * 2.0  # f16
    if isinstance(heads_kv, list) and heads_kv:
        # Hybrid stacks give head_count_kv per layer, and a layer with 0 has no KV cache
        # at all, so multiplying by a single head count would invent memory it never uses.
        return sum(float(h) * per_layer for h in heads_kv)
    if not isinstance(layers, int) or not isinstance(heads_kv, int):
        return None
    return float(layers) * float(heads_kv) * per_layer


def serving_kv_gib_per_user(model: str, ctx: int) -> Dict[str, Any]:
    """`{"gib": float|None, "why": str}` -- the KV one user's full context costs.

    A model whose weights fit but whose KV does not is the case the fit classes cannot see on
    their own: at 8192 tokens and 16 slots a 27B-class GGUF wants tens of GiB of cache, far
    past the flat `SERVE_OVERHEAD_GIB` allowance, so it was planned as fitting and then died
    at load."""
    path = osp.expanduser(model)
    if not (osp.isfile(path) and path.endswith(".gguf")):
        return {"gib": None, "why": "KV per user is only read from a GGUF header"}
    per_token = gguf_kv_bytes_per_token(gguf_metadata(path))
    if per_token is None:
        return {"gib": None, "why": f"{model} does not declare the dimensions KV size needs"}
    return {"gib": per_token * max(1, int(ctx)) / 2**30, "why": ""}


# Llama-3.3-70B fp8 on two Sparks against one. TP is the ONLY axis that makes a single
# request faster; PP moves tokens through more silicon without shortening one token's
# critical path, so its TPOT is FLAT and PP buys capacity, never latency; replicas change
# per-request latency not at all.
TP_SPEEDUP_2 = {1: 2.09, 2: 2.13, 4: 2.10, 8: 1.97}
PP_SPEEDUP_2 = {1: 1.08, 2: 1.11, 4: 1.09, 8: 1.07}
TP_TPOT_MS_2 = (332.7, 162.4)
PP_TPOT_MS_2 = (320.0, 320.0)
# Splitting a model that ALREADY FITS: still a flat loss on a llama.cpp whose RPC backend
# predates ggml-org/llama.cpp#18626, because without it the backend advertises neither async
# nor events, so ggml_backend_sched refuses to pipeline and the halves run serially.
LAYER_SPLIT_FITTING_SPEEDUP = 0.92

# WITH #18626 the answer becomes a function of prompt length, because what overlaps is
# prefill: decode cannot improve, since a split moves the same weight bytes per token.
# Below ~256 tokens splitting costs a few percent; above ~1024 it wins.
LAYER_SPLIT_ASYNC_RPC_SPEEDUP = {
    128: {1: 0.94, 4: 0.95, 8: 0.95},
    256: {1: 0.98, 4: 1.00, 8: 1.00},
    512: {1: 0.96, 4: 1.05, 8: 1.07},
    1024: {1: 1.02, 4: 1.12, 8: 1.17},
    2048: {1: 1.07, 4: 1.23, 8: 1.29},
    4096: {1: 1.11, 4: 1.35, 8: 1.45},
}
LAYER_SPLIT_BREAK_EVEN_TOKENS = 256


def layer_split_speedup(
    prompt_tokens = None,
    concurrency = 1,
    async_rpc = False,
):
    """End-to-end speedup from splitting a model that already fits on one node. `async_rpc`
    stays False unless the build carries #18626. Returns the nearest measured row at or below
    `prompt_tokens`: these are measured points, not a fitted curve, so nothing is interpolated."""
    if not async_rpc:
        return LAYER_SPLIT_FITTING_SPEEDUP
    if prompt_tokens is None:
        return None  # genuinely unknown; the caller must not guess
    rows = sorted(LAYER_SPLIT_ASYNC_RPC_SPEEDUP)
    key = rows[0]
    for r in rows:
        if prompt_tokens >= r:
            key = r
    by_c = LAYER_SPLIT_ASYNC_RPC_SPEEDUP[key]
    near = min(by_c, key = lambda c: abs(c - max(1, int(concurrency))))
    return by_c[near]


# Aggregate DECODE tok/s on two Sparks against one; TOPOLOGY_MEASUREMENT names the run. A
# different question from LAYER_SPLIT_ASYNC_RPC_SPEEDUP, which includes prefill. A layer
# split never speeds up decode at any user count: it is a capacity and prefill feature. Two
# replicas win throughput once model plus KV fits on one node and 8+ users are concurrent;
# below 8 a second copy buys little, and at 1 user only tensor parallel helps.
TOPOLOGY_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL on llama.cpp b10796, two DGX Sparks, 2026-09-04, uncapped clocks"
)
REPLICAS_DECODE_SPEEDUP = {
    512: {1: 1.00, 2: 1.13, 4: 1.13, 8: 1.30, 16: 1.75, 32: 1.91},
    2048: {1: 1.01, 2: 1.38, 4: 1.30, 8: 1.81, 16: 1.99, 32: 2.38},
}
LAYER_SPLIT_DECODE_SPEEDUP = {
    512: {1: 0.95, 2: 0.99, 4: 0.92, 8: 0.85, 16: 0.94, 32: 1.01},
    2048: {1: 0.94, 2: 0.96, 4: 0.88, 8: 0.96, 16: 1.06, 32: 1.12},
}
LAYER_SPLIT_DECODE_ONLY_SPEEDUP = 0.95
LAYER_SPLIT_PREFILL_SPEEDUP = (1.7, 1.85)

# llama-server ``--pipeline-groups N`` (unslothai/llama.cpp PR #187): N contexts from one
# model, so one group's batch runs on the peer's layers while the other's runs here. Measured
# with the RPC device first, which is a precondition, not a detail.
PIPELINE_GROUPS_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL, llama-server --pipeline-groups 2 with --device RPC0,CUDA0, two DGX "
    "Sparks, 2026-09-05, uncapped clocks, two repeats"
)
# rows: (one Spark, split with one context, split with two groups), decode tok/s
PIPELINE_GROUPS_DECODE_TOKS = {
    32: (116, 100, 130),
    64: (139, 117, 157),
    128: (150, 124, 170),
}
PIPELINE_GROUPS_SPLIT_SPEEDUP = {32: 1.12, 64: 1.13, 128: 1.13}  # two groups over one Spark
PIPELINE_GROUPS_SPLIT_SPEEDUP_RANGE = (1.12, 1.13)  # 32 to 128 rows
PIPELINE_GROUPS_OVER_ONE_CONTEXT = {32: 1.31, 64: 1.34, 128: 1.37}  # two groups over one context
PIPELINE_GROUPS_OVER_ONE_CONTEXT_RANGE = (1.31, 1.37)
PIPELINE_GROUPS_GPU_UTIL = (0.45, 0.78)  # per node, without and with the groups
# MTP self speculation on ONE Spark, a no-op for a GGUF with no nextn_predict_layers. Depth 3
# is the mixed-traffic choice: 8 is faster at one user and much slower at eight. Draft models
# and n-gram speculation are NOT defaults, both winning at one user and losing from four.
# Greedy output under MTP is not byte identical to the baseline, and that is not a defect:
# the baseline is not batch invariant itself.
MTP_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL and Qwen3.5-4B-MTP-UD-Q4_K_XL, llama-server b10796 --spec-type "
    "draft-mtp --spec-draft-n-max 3, one DGX Spark, 2026-09-05, 1690 MHz, npp 128 / ntg 256"
)
MTP_SPEEDUP_27B = {1: 2.61, 4: 1.87, 8: 1.59}  # users -> aggregate decode over no speculation
MTP_SPEEDUP_4B = {1: 2.04, 4: 1.67, 8: 1.46}
MTP_SPEEDUP_27B_LONG_OUTPUT = {1: 2.81, 8: 2.01}  # ntg 1024
MTP_ACCEPTANCE_27B = {1: 0.88, 8: 0.74}
MTP_ACCEPTANCE_4B = {1: 0.88, 8: 0.72}
MTP_DRAFT_N_MAX = 3
MTP_SMALL_MODEL_B = 8.0  # below this many B parameters the 4B table is the closer estimate
# Depth 3 came from the 1-to-8-user table above and is the WORST of the three at every row
# count measured on a split; below 32 rows nothing was measured there. Acceptance is a function
# of the depth alone and flat in rows, so this is not the drafter working less well at width,
# it is the draft tokens widening a batch already past the cheap point of the per-token curve.
# Every cell is a TWO-GROUP split, and a one-context split gets the same depth: its rows sit in
# one step rather than halved across two, so the batch is WIDER and a narrower draft is the
# only safe extrapolation.
MTP_DRAFT_N_MAX_X_ROWS_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL split over two DGX Sparks, llama-server --pipeline-groups 2 "
    "--kv-unified --tensor-split 0.5,0.5 --spec-type draft-mtp, unslothai/llama.cpp PR #187 "
    "a1dd7c5e8, both nodes pinned at 1690 MHz, 2026-09-07, npp 128 / ntg 256, forward and "
    "reversed legs per cell"
)
# concurrent rows -> {n-max: decode tok/s}, leg mean. 0 is the drafter off.
MTP_DRAFT_N_MAX_X_ROWS_TOKS = {
    32: {0: 146.8, 1: 152.8, 2: 162.9, 3: 155.9},
    64: {0: 186.3, 1: 171.7, 2: 166.9, 3: 146.3},
    128: {0: 212.7, 1: 164.1, 2: 138.7, 3: 132.2},
}
MTP_DRAFT_N_MAX_ACCEPTANCE = {1: 0.87, 2: 0.78, 3: 0.69}  # n-max -> acceptance, flat in rows
MTP_DRAFT_N_MAX_BY_ROWS = {32: 2, 64: 1}


def mtp_draft_n_max(users: Optional[int] = None) -> int:
    """The ``--spec-draft-n-max`` depth measured best at this many concurrent rows, and
    ``MTP_DRAFT_N_MAX`` below the lowest measured row count, where the one-Spark table lives.
    No fitting: the largest measured key applies from there up."""
    if users is None:
        return MTP_DRAFT_N_MAX
    rows = max(1, int(users or 1))
    depth = MTP_DRAFT_N_MAX
    for key in sorted(MTP_DRAFT_N_MAX_BY_ROWS):
        if rows >= key:
            depth = MTP_DRAFT_N_MAX_BY_ROWS[key]
    return depth


# Whether a drafter pays on a split AT ALL, which the depth rule above does not answer. LAYER
# SPLIT only. Percentages come from the unrounded leg means, so recomputing them from the
# rounded table lands within 0.2 points.
SPLIT_MTP_BEST_DEPTH_VS_OFF_PCT = {32: 11.0, 64: -7.9, 128: -22.9}
SPLIT_MTP_OFF_ROWS = 64
"""Concurrent rows at or above which a two-Spark layer split runs with the drafter OFF.

The measured boundary is somewhere between 32 and 64 rows, and 33 to 63 is INTERPOLATED:
nothing in between was measured. The two points that bound it, on the 27B layer split with
two pipeline groups, ``--kv-unified`` and ``--tensor-split 0.5,0.5``, both nodes at 1690 MHz
(``MTP_DRAFT_N_MAX_X_ROWS_MEASUREMENT``):

* 32 rows, where the best depth (n-max 2) is worth **+11.0 %** over the drafter off,
  162.9 against 146.8 tok/s, so speculation stays ON there;
* 64 rows, where the best depth (n-max 1) **costs 7.9 %**, 171.7 against 186.3 tok/s, and
  every deeper draft costs more; at 128 rows the best depth costs 22.9 %.

The constant is put at 64, the lower of the two measured points at which speculation loses,
so no interpolated row count is ever served a rule that was not measured to help there.

This is NOT ``GROUPS_X_MTP_CROSSOVER_ROWS`` (16), which answers a different question -- two
pipeline groups against one context, both with the drafter on -- and does not move.
"""


def split_mtp_wins(users: Optional[int] = None) -> bool:
    """Whether a two-Spark layer split at this many concurrent rows should run a drafter AT
    ALL, at the depth ``mtp_draft_n_max`` returns: True below ``SPLIT_MTP_OFF_ROWS``.

    Layer split only; one Spark and two replicas are governed by the one-Spark MTP table, where
    speculation is a large win, and no cell of this matrix was measured on either. ``users`` of
    None keeps the drafter, which is what every topology did before this rule existed."""
    if users is None:
        return True
    return max(1, int(users or 1)) < SPLIT_MTP_OFF_ROWS


def split_mtp_note() -> str:
    """Whether a layer split speculates at all, and at what depth, for reasons and the
    ``spark plan`` text."""
    lo, hi, top = 32, SPLIT_MTP_OFF_ROWS, max(MTP_DRAFT_N_MAX_X_ROWS_TOKS)
    cells = MTP_DRAFT_N_MAX_X_ROWS_TOKS

    def _best(rows: int) -> Tuple[int, float, float]:
        drafting = {depth: toks for depth, toks in cells[rows].items() if depth}
        depth = max(drafting, key = lambda d: drafting[d])
        return depth, drafting[depth], SPLIT_MTP_BEST_DEPTH_VS_OFF_PCT[rows]

    lo_depth, lo_toks, lo_pct = _best(lo)
    hi_depth, hi_toks, hi_pct = _best(hi)
    _top_depth, _top_toks, top_pct = _best(top)
    return (
        f"A layer split speculates only below {SPLIT_MTP_OFF_ROWS} concurrent rows. Swept on "
        f"the pair at {lo} / {hi} / {top} rows against the same split with no drafter, the "
        f"best draft depth is worth {lo_pct:+.1f} percent at {lo} rows "
        f"({lo_toks:.1f} against {cells[lo][0]:.1f} tok/s at n-max {lo_depth}) but "
        f"{hi_pct:+.1f} percent at {hi} ({hi_toks:.1f} against {cells[hi][0]:.1f} at n-max "
        f"{hi_depth}) and {top_pct:+.1f} percent at {top}, because the draft tokens widen a "
        f"batch that is already past the cheap point of the per-token curve. So below "
        f"{SPLIT_MTP_OFF_ROWS} rows the split asks for --spec-type draft-mtp at the depth "
        f"measured best for the row count (n-max {mtp_draft_n_max(1)} under {lo} rows, "
        f"{mtp_draft_n_max(lo)} from {lo}) and at or above {SPLIT_MTP_OFF_ROWS} it asks for "
        f"no drafter and no draft flags at all. Nothing between {lo + 1} and "
        f"{SPLIT_MTP_OFF_ROWS - 1} rows was measured, so the boundary sits on the lower of "
        f"the two measured points at which speculation loses. This is the split only: one "
        f"Spark and two replicas keep MTP, which is a win at every user count measured there."
    )


# Pipeline groups AND speculative decoding on the same layer split: unslothai/llama.cpp
# PR #187 (a1dd7c5e8) gives every group its own speculative state, so the pair is accepted.
# It is still refused with --mmproj, --control-vector and --sleep-idle-seconds, and --parallel
# must stay a multiple of N. --kv-unified is on every cell, and is why this table was
# re-measured: the first version left it out and so described a launch the product never
# makes. Both nodes were PINNED, because the unpinned attempt spanned three clock states and
# no two arms were comparable. The winner flips between 8 and 32 rows with nothing measured in
# between, so GROUPS_X_MTP_CROSSOVER_ROWS is the geometric midpoint.
GROUPS_X_MTP_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL, llama-server --kv-unified --pipeline-groups 2 --spec-type "
    "draft-mtp with --device RPC0,CUDA0, unslothai/llama.cpp PR #187 a1dd7c5e8, two DGX "
    "Sparks both pinned at 1700 MHz, 2026-09-06, npp 128 / ntg 256, --parallel 32, two "
    "repeats in opposite arm order"
)
# concurrent rows -> (split 1 context, + MTP, split 2 groups, + MTP) decode tok/s, repeat mean
GROUPS_X_MTP_DECODE_TOKS = {
    8: (51.4, 87.8, 49.7, 84.8),
    32: (97.2, 112.1, 139.5, 152.5),
}
# NOT re-measured with --kv-unified or under the pin, so not directly comparable with the rest.
GROUPS_X_MTP_ONE_SPARK_MTP_TOKS = {8: 85.6, 32: 83.8}
GROUPS_X_MTP_ACCEPTANCE = {8: (0.75, 0.74), 32: (0.70, 0.71)}  # (one context, two groups)
GROUPS_X_MTP_OVER_MTP_ONLY = {8: 0.97, 32: 1.36}  # both over one context with MTP
GROUPS_X_MTP_OVER_GROUPS_ONLY = {8: 1.71, 32: 1.09}  # both over two groups alone
# The geometric mean of the measured 8 and 32; nothing between them was measured.
GROUPS_X_MTP_CROSSOVER_ROWS = 16

# Sized with --parallel and -c to the offered concurrency, so every slot keeps the same context
# at every point. Two pipeline groups win at EVERY concurrency here, including 8 rows, where an
# earlier table had them losing: that table held --parallel at 32 and sent only 8 clients, so
# the SIZING decided it and not the topology. No cell used a drafter.
# The gain grows with the rows faster than a fixed-cost model predicts, because decode step
# time here is not c0 + c1*b: two groups of 64 sit at a cheaper point on that curve than one
# context of 128.
SPLIT_GROUPS_ROWS_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL, llama-server --kv-unified --cache-ram 0 -fa on --device RPC0,CUDA0 "
    "-sm layer, --parallel R with -c 512*R, unslothai/llama.cpp PR #187 a1dd7c5e8, two DGX "
    "Sparks both pinned at 1700 MHz, 2026-09-06, npp 128 / ntg 256, no speculation, two passes "
    "in opposite arm order"
)
# concurrent rows -> (one context, two groups) decode tok/s, mean of the two passes
SPLIT_GROUPS_ROWS_TOKS = {
    8: (50.9, 68.0),
    16: (75.1, 100.1),
    32: (97.5, 143.4),
    64: (112.9, 174.3),
    128: (116.6, 201.2),
}
# The lowest measured point rather than a crossing: two groups won at every point.
SPLIT_GROUPS_MIN_ROWS = 8

# With no --tensor-split, llama.cpp divides the layers by each device's FREE MEMORY at load
# time, so the boundary moves with whatever else the nodes hold and is not reproducible between
# two loads. An explicit even split lands one block past the middle, where the two GPUs' busy
# fractions come out equal, because the split indexes n_layer + 1 slots and the last is the
# output block.
SPLIT_TENSOR_SPLIT_EVEN = "0.5,0.5"
SPLIT_TENSOR_SPLIT_MEASURED_PEER_BLOCKS = {27: 192.1, 30: 199.7, 33: 206.4, 34: 201.2, 36: 192.5}

# SPLIT_GROUPS_ROWS_TOKS alone says "use 128 rows", and that is wrong twice over: TTFT rises
# faster than throughput, with the knee between 32 and 64 rows, and asking for MORE slots than
# the load offers is not free either -- an oversized server loses about a quarter of the
# throughput AND a third of the median TTFT. So the rows track the offered concurrency, capped
# at the last point whose p90 TTFT is inside 14 s.
SPLIT_ROWS_TTFT_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL, llama-server --kv-unified --cache-ram 0 -fa on --device RPC0,CUDA0 "
    "-sm layer --tensor-split 0.5,0.5 --pipeline-groups 2, --parallel R with -c 512*R, two DGX "
    "Sparks both pinned at 1700 MHz, 2026-09-06, npp 128 / ntg 256, no speculation, R clients "
    "arriving at once so the TTFT is a saturating burst and not a steady state"
)
# concurrent rows -> (decode tok/s, TTFT median seconds, TTFT p90 seconds), slots = offered load
SPLIT_ROWS_TTFT = {
    8: (66.8, 1.73, 2.00),
    16: (100.1, 2.95, 3.51),
    32: (143.3, 4.60, 5.61),
    64: (182.7, 9.07, 13.17),
    128: (211.1, 18.11, 26.33),
}
# Largest rows setting whose measured p90 TTFT a person waiting on a reply can be asked to accept.
SPLIT_ROWS_INTERACTIVE_MAX = 64
# Largest rows setting measured at all. Only for offline or batch work.
SPLIT_ROWS_THROUGHPUT_MAX = 128
# Smallest rows setting measured; below this nothing has been run.
SPLIT_ROWS_MIN = 8
# offered -> (tok/s ratio, TTFT median ratio) of a 128-slot server against one sized to that load.
SPLIT_ROWS_OVERSIZED_SLOTS = {8: (0.773, 1.09), 32: (0.770, 1.38), 128: (1.0, 1.0)}


def split_rows_for_users(users: int, interactive: bool = True) -> int:
    """Rows (``--parallel``) a layer split should ask for, for ``users`` offered concurrency.

    Tracks the offered load rather than rounding up, because oversizing the slot count costs
    throughput and TTFT both (``SPLIT_ROWS_OVERSIZED_SLOTS``). Clamped to the measured range,
    and to ``SPLIT_ROWS_INTERACTIVE_MAX`` unless the caller says nobody is waiting.
    """
    cap = SPLIT_ROWS_INTERACTIVE_MAX if interactive else SPLIT_ROWS_THROUGHPUT_MAX
    return max(SPLIT_ROWS_MIN, min(cap, int(users or 1)))


def split_rows_ttft_s(rows: int) -> float:
    """Measured p90 TTFT in seconds at the nearest measured rows point at or below ``rows``."""
    points = sorted(SPLIT_ROWS_TTFT)
    key = points[0]
    for point in points:
        if rows >= point:
            key = point
    return SPLIT_ROWS_TTFT[key][2]


REPLICAS_MIN_USERS = 8
REPLICAS_FEW_USERS_SPEEDUP = 1.13  # 2 to 4 users, prompt 512
TOPOLOGIES = ("single", "replicas", "layer_split")


def _measured_cell(table: Dict[int, Dict[int, float]], prompt_tokens: int, users: int) -> float:
    """Nearest measured row at or below ``prompt_tokens``, nearest user count. No fitting."""
    rows = sorted(table)
    key = rows[0]
    for row in rows:
        if prompt_tokens >= row:
            key = row
    by_users = table[key]
    near = min(by_users, key = lambda u: (abs(u - max(1, int(users))), u))
    return by_users[near]


def replicas_speedup(prompt_tokens: int = 512, users: int = 1) -> float:
    return _measured_cell(REPLICAS_DECODE_SPEEDUP, prompt_tokens, users)


def layer_split_decode_speedup(prompt_tokens: int = 512, users: int = 1) -> float:
    return _measured_cell(LAYER_SPLIT_DECODE_SPEEDUP, prompt_tokens, users)


def pipeline_groups_note() -> str:
    """What a layer split delivers with and without pipeline groups, for reasons and the
    ``spark plan`` text. The flag is added only when the bundle has it, so both halves are
    stated."""
    low, high = PIPELINE_GROUPS_SPLIT_SPEEDUP_RANGE
    clow, chigh = PIPELINE_GROUPS_OVER_ONE_CONTEXT_RANGE
    rows = sorted(PIPELINE_GROUPS_SPLIT_SPEEDUP)
    decode = LAYER_SPLIT_DECODE_SPEEDUP[512]
    return (
        f"With pipeline groups (llama-server --pipeline-groups 2, added when the bundle "
        f"has it, RPC device first so the output layer stays local) the pair measured "
        f"{clow:.2f}x to {chigh:.2f}x of the one-context split and {low:.2f}x to {high:.2f}x "
        f"of one Spark at {rows[0]} to {rows[-1]} concurrent rows with both GPUs near "
        f"{PIPELINE_GROUPS_GPU_UTIL[1] * 100:.0f} percent; without them expect "
        f"{min(decode.values()):.2f}x to {max(decode.values()):.2f}x on decode and "
        f"{LAYER_SPLIT_PREFILL_SPEEDUP[0]:.1f}x to {LAYER_SPLIT_PREFILL_SPEEDUP[1]:.2f}x "
        f"on prefill."
    )


def mtp_speedup(users: int = 1, model_size_b: Optional[float] = None) -> float:
    """MTP self speculation's aggregate decode gain at the nearest measured user count, for a
    GGUF that ships the head. It multiplies whatever the topology gives; a GGUF without the head
    is 1.0, which the caller knows and this function does not."""
    small = model_size_b is not None and float(model_size_b) < MTP_SMALL_MODEL_B
    table = MTP_SPEEDUP_4B if small else MTP_SPEEDUP_27B
    _users, value = _nearest_concurrency(table, max(1, int(users or 1)))
    return value


def mtp_note() -> str:
    """What MTP self speculation delivers, for reasons and the ``spark plan`` text. Asked for
    only when the header has the head and the bundle has --spec-type, so the no-op case is
    stated too."""
    a, b = MTP_SPEEDUP_27B, MTP_SPEEDUP_4B
    return (
        f"A GGUF that ships its own MTP head (Qwen3.5-4B-MTP, Qwen3.8-27B) self-speculates "
        f"(llama-server --spec-type draft-mtp --spec-draft-n-max {MTP_DRAFT_N_MAX}, asked for "
        f"when the header has nextn_predict_layers and the bundle has the flag): measured on "
        f"one Spark {a[1]:.2f}x / {a[4]:.2f}x / {a[8]:.2f}x aggregate decode at 1 / 4 / 8 "
        f"users on the 27B and {b[1]:.2f}x / {b[4]:.2f}x / {b[8]:.2f}x on the 4B, on top of "
        f"what the topology gives; a no-op for a GGUF without the head. Draft models and "
        f"n-gram speculation are single-user tricks on this pair (about 2x at one user, a "
        f"loss from 4) and stay off. Those figures are ONE Spark, and they govern the single "
        f"and replicas topologies; a two-Spark layer split has its own swept table, runs the "
        f"depth measured best for its row count and turns the drafter off entirely from "
        f"{SPLIT_MTP_OFF_ROWS} rows up (split_mtp_note)."
    )


def groups_x_mtp_wins(users: int) -> bool:
    """Whether a two-Spark layer split at this many concurrent rows should run pipeline groups
    AND speculative decoding rather than one context with speculation. Below the crossover the
    two are within 3 percent of each other, so choosing wrong there costs little.

    This answers GROUPS against ONE CONTEXT and nothing else; whether the split runs a drafter
    at all is ``split_mtp_wins`` / ``SPLIT_MTP_OFF_ROWS``, a separate measurement. The serving
    side also needs a build that accepts the two flags together (PR #187)."""
    return max(1, int(users or 1)) >= GROUPS_X_MTP_CROSSOVER_ROWS


def groups_x_mtp_note() -> str:
    """What a layer split gets from pipeline groups and speculative decoding together, for
    reasons and the ``spark plan`` text.

    Two measurements are stated separately: ``GROUPS_X_MTP_CROSSOVER_ROWS`` for groups against
    one context, and ``SPLIT_MTP_OFF_ROWS`` (``split_mtp_note``) for whether a drafter runs at
    all. So a split at 8 rows is one context with MTP, at 32 rows two groups with MTP, and at
    64 rows and up two groups with no drafter."""
    hi, lo = 32, 8
    both_hi = GROUPS_X_MTP_DECODE_TOKS[hi][3]
    mtp_hi = GROUPS_X_MTP_DECODE_TOKS[hi][1]
    both_lo = GROUPS_X_MTP_DECODE_TOKS[lo][3]
    mtp_lo = GROUPS_X_MTP_DECODE_TOKS[lo][1]
    return (
        f"A layer split can run pipeline groups and speculative decoding together on a "
        f"llama-server that gives each group its own speculative state (unslothai/llama.cpp "
        f"PR #187): measured on the pair at {hi} concurrent rows {both_hi:.1f} tok/s against "
        f"{mtp_hi:.1f} for one context with MTP and "
        f"{GROUPS_X_MTP_DECODE_TOKS[hi][2]:.1f} for two groups alone "
        f"({GROUPS_X_MTP_OVER_MTP_ONLY[hi]:.2f}x and "
        f"{GROUPS_X_MTP_OVER_GROUPS_ONLY[hi]:.2f}x), but at {lo} rows {both_lo:.1f} against "
        f"{mtp_lo:.1f} ({GROUPS_X_MTP_OVER_MTP_ONLY[lo]:.2f}x), because two groups halve the "
        f"rows per group. So from {GROUPS_X_MTP_CROSSOVER_ROWS} rows up a split asks for the "
        f"groups and below that for one context; the speculation in that sentence is the "
        f"drafter it runs below {SPLIT_MTP_OFF_ROWS} rows, since from {SPLIT_MTP_OFF_ROWS} "
        f"rows up the split keeps the groups and drops the drafter entirely "
        f"(split_mtp_note). The combination is still refused "
        f"with --mmproj, --control-vector and --sleep-idle-seconds, and --parallel stays a "
        f"multiple of the group count."
    )


def recommend_topology(
    model_bytes: float,
    kv_bytes_per_user: float,
    users: int,
    prompt_tokens: int,
    per_node_free_bytes: float,
    prefill_heavy: bool = False,
) -> Dict[str, Any]:
    """Which of single / replicas / layer_split to serve a GGUF with, and why. Pure. Memory is
    checked with every concurrent user's KV included, so a model that fits alone but not with
    its users' KV goes to replicas or, failing that, to a split. ``layer_split`` is never
    chosen for a model that FITS unless the caller says the work is prefill-heavy."""
    users = max(1, int(users or 1))
    prompt_tokens = max(1, int(prompt_tokens or 512))
    model_bytes = max(0.0, float(model_bytes or 0))
    kv_each = max(0.0, float(kv_bytes_per_user or 0))
    free = float(per_node_free_bytes or 0)
    single_need = model_bytes + kv_each * users
    # Full users, not half. A replica node runs its own complete server, and the launcher hands
    # each one the same --parallel and context as the primary while the router declares the
    # slots on both, so every node allocates KV for the whole user count. Budgeting half was
    # optimistic in exactly the window this branch exists to rescue -- full-user KV does not fit
    # one node, half-user KV does -- and on 121.69 GiB shared between CPU and GPU that is an
    # OOM rather than a slowdown. Pricing it honestly declines a throughput optimisation
    # instead, which is the right way round.
    replica_need = model_bytes + kv_each * users
    fits_model = model_bytes <= free
    out: Dict[str, Any] = {
        "topology": "single",
        "reason": "",
        "speedup": None,
        "prefill_speedup": None,
        "pipeline_groups_speedup": None,
        # The planner states the single-Spark measurement; the serving side checks the header.
        "mtp_speedup": mtp_speedup(users),
        "mtp_note": mtp_note(),
        # Layer split only: the rows-against-drafter matrix was measured on the split alone.
        "split_mtp": None,
        "split_mtp_note": None,
        "fits_one_node": fits_model,
        "fits_any_topology": True,
        "users": users,
        "prompt_tokens": prompt_tokens,
        "single_node_bytes": single_need,
        "replica_node_bytes": replica_need,
        "per_node_free_bytes": free,
        "measured_on": TOPOLOGY_MEASUREMENT,
    }
    gib = 2**30
    # A split halves the weights and the KV across the pair, so what it can hold is bounded by
    # the two nodes together. Above that no topology fits, and recommending one anyway spends
    # the whole load, which for a model this size is minutes of transfer over the rail, before
    # llama-server runs out of memory. Saying so up front is the only useful answer.
    pair_free = free * 2.0  # this planner answers for the pair; there is no third node
    pair_need = model_bytes + kv_each * users
    if pair_need > pair_free:
        out.update(
            topology = "single",
            fits_any_topology = False,
            reason = (
                f"the model ({model_bytes / gib:.1f} GiB) with KV for {users} users needs "
                f"{pair_need / gib:.1f} GiB, against {pair_free / gib:.1f} GiB across both "
                f"Sparks: no two-node topology holds it. Use a smaller quant, fewer users "
                f"or a shorter context."
            ),
        )
        return out
    if not fits_model:
        out.update(
            topology = "layer_split",
            prefill_speedup = LAYER_SPLIT_PREFILL_SPEEDUP,
            pipeline_groups_speedup = PIPELINE_GROUPS_SPLIT_SPEEDUP_RANGE,
            reason = (
                f"the model ({model_bytes / gib:.1f} GiB) does not fit in one node's "
                f"{free / gib:.1f} GiB, so a layer split across both Sparks is the only way "
                f"to run it. " + pipeline_groups_note() + " " + split_mtp_note()
            ),
            split_mtp = split_mtp_wins(users),
            split_mtp_note = split_mtp_note(),
        )
        return out
    if single_need > free:
        # No replicas branch here, and that is the point: a replica node runs its own full
        # server, and the launcher gives it the same context and --parallel as the primary
        # while the router declares the slots on both, so a replica needs exactly what a single
        # node needs. Replicas buy throughput for a model that already fits, never capacity.
        out.update(
            topology = "layer_split",
            prefill_speedup = LAYER_SPLIT_PREFILL_SPEEDUP,
            pipeline_groups_speedup = PIPELINE_GROUPS_SPLIT_SPEEDUP_RANGE,
            reason = (
                f"the model fits, but model plus KV for {users} users "
                f"({single_need / gib:.1f} GiB) exceeds one node's {free / gib:.1f} GiB, and a "
                f"replica is no smaller because each one holds a full copy and KV for every "
                f"user. Only a layer split, which spreads the KV with the layers, has the "
                f"room. Capacity, not speed: " + pipeline_groups_note() + " " + split_mtp_note()
            ),
            split_mtp = split_mtp_wins(users),
            split_mtp_note = split_mtp_note(),
        )
        return out
    if prefill_heavy and users < REPLICAS_MIN_USERS:
        out.update(
            topology = "layer_split",
            speedup = layer_split_decode_speedup(prompt_tokens, users),
            prefill_speedup = LAYER_SPLIT_PREFILL_SPEEDUP,
            reason = (
                f"you asked for prefill-heavy long-prompt work at {users} users. A layer "
                f"split measured {LAYER_SPLIT_PREFILL_SPEEDUP[0]:.1f}x to "
                f"{LAYER_SPLIT_PREFILL_SPEEDUP[1]:.2f}x on prefill, which is the only "
                f"reason to split a model that fits; its decode is "
                f"{layer_split_decode_speedup(prompt_tokens, users):.2f}x, so time to first "
                f"token improves and tokens per second do not. Chat-shaped traffic should "
                f"stay on one node. At {users} rows the split still speculates: the drafter "
                f"only goes off from {SPLIT_MTP_OFF_ROWS} rows up."
            ),
            split_mtp = split_mtp_wins(users),
            split_mtp_note = split_mtp_note(),
        )
        return out
    fit_note = (
        f"the model fits on one node with KV for {users} users "
        f"({single_need / gib:.1f} of {free / gib:.1f} GiB)"
        if kv_each
        else f"the model fits on one node ({single_need / gib:.1f} of {free / gib:.1f} GiB, "
        f"KV not counted)"
    )
    if users >= REPLICAS_MIN_USERS:
        gain = replicas_speedup(prompt_tokens, users)
        out.update(
            topology = "replicas",
            speedup = gain,
            reason = (
                f"{fit_note}, and at {users} concurrent "
                f"users two replicas measured {gain:.2f}x aggregate decode "
                f"(1.30x at 8, 1.75x at 16, 1.91x at 32 users, prompt 512). A layer split "
                f"measured {layer_split_decode_speedup(prompt_tokens, users):.2f}x here, so "
                f"never split a model that fits for throughput."
                + (
                    " Prefill-heavy work does not change this at this many users: the "
                    "replicas still win end to end (1.81x against 0.96x at prompt 2048, "
                    "8 users)."
                    if prefill_heavy
                    else ""
                )
            ),
        )
        return out
    few = replicas_speedup(prompt_tokens, users)
    out.update(
        topology = "single",
        speedup = 1.0,
        reason = (
            f"{fit_note}, "
            f"and {users} concurrent user{'s' if users != 1 else ''} cannot use a second "
            f"one: two replicas measured {few:.2f}x at this concurrency (1.00x at 1 user, "
            f"{REPLICAS_FEW_USERS_SPEEDUP:.2f}x at 2 to 4) for the cost of a full second "
            f"copy, and a layer split measured "
            f"{layer_split_decode_speedup(prompt_tokens, users):.2f}x. Leave the second "
            f"Spark idle, or use it for something else. Replicas start paying at "
            f"{REPLICAS_MIN_USERS} users."
            + (
                " At 1 user the only measured win is vLLM tensor parallel "
                f"({TP_SPEEDUP_2[1]:.2f}x), which llama.cpp cannot do."
                if users == 1
                else ""
            )
        ),
    )
    return out


REPLICA_AGGREGATE_PER_NODE = 1.0  # n replicas -> ~n x aggregate, 1.0x per request
# GPipe pipeline parallel, M=4 microbatches, 2 nodes: 3024 vs 2032 tok/s single-node.
TRAIN_PP_SPEEDUP_2 = 1.49

# Data parallel against pipeline parallel on the same rows and loss, the control being the same
# file at WORLD_SIZE=1. This inverts the old guidance: the number that made a split of a model
# that FITS look like a throughput win came from a control on a node missing the
# linear-attention fast path. The schedules did not change; the denominator did.
TRAIN_MEASUREMENT = (
    "2026-09-06, unsloth/Qwen3.5-2B and -9B, LoRA r=16, seq 512, global batch 64, M=32, "
    "20 steps, seed 3407, both Sparks uncapped at ~2400 MHz"
)
# `one_spark` is None where no control ran in the SAME clock state: a control taken while the
# thermal guard capped the node mid-cell is not a control.
TRAIN_DP_VS_PP_TOKS: Dict[str, Dict[str, Optional[int]]] = {
    "unsloth/Qwen3.5-2B": {
        "one_spark": 2613,
        "ddp": 5203,
        "dualpipev": 4128,
        "1f1b": 4106,
        "fsdp": 3276,
    },
    "unsloth/Qwen3.5-9B": {
        "one_spark": 850,
        "ddp": 1741,
        "dualpipev": 1525,
        "1f1b": 1505,
        "fsdp": 927,
    },
}
TRAIN_DP_SPEEDUP = {"unsloth/Qwen3.5-2B": 1.99, "unsloth/Qwen3.5-9B": 2.05}
TRAIN_PP_SPEEDUP = {"unsloth/Qwen3.5-2B": 1.58, "unsloth/Qwen3.5-9B": 1.79}
TRAIN_DP_SPEEDUP_RANGE = (1.99, 2.05)
TRAIN_PP_SPEEDUP_RANGE = (1.58, 1.79)
# No single-Spark control needed, so this ratio survives when a control is thrown away.
TRAIN_DP_OVER_PP = {"unsloth/Qwen3.5-2B": 1.26, "unsloth/Qwen3.5-9B": 1.14}
# DDP holds the whole model per node against PP's half, which is why the rule is size-gated.
TRAIN_PEAK_GIB = {"unsloth/Qwen3.5-2B": (10.16, 9.58), "unsloth/Qwen3.5-9B": (28.60, 24.13)}
# FSDP cuts resident memory but gathers every layer on every microbatch, landing BELOW both
# DDP and the pipeline schedules.
TRAIN_FSDP_SPEEDUP = {"unsloth/Qwen3.5-2B": 1.25, "unsloth/Qwen3.5-9B": 1.09}
# Which schedule when a split is forced. The two are a tie-break on models that FIT, but a
# split is only RECOMMENDED for a model that does NOT, and there the tie breaks hard: at 70B
# dualpipev never completed a step, out of memory on rank 0, while 1f1b did. Structural, not a
# bug: the V layout co-locates the FIRST and LAST stages on one rank, which then carries the
# embedding, the LM head, the loss and the deepest in-flight set at once. The co-located hop it
# trades that for buys nothing either, the link being idle at seq 512 with grad checkpointing.
TRAIN_PP_SCHEDULE = "1f1b"
TRAIN_PP_SCHEDULE_MARGIN = 0.005  # dualpipev over 1f1b ON MODELS THAT FIT; below noise

# M sets the fill/drain bubble, which shrinks as 2M/(M+1), and the microbatch size B/M. Swept
# at a FIXED global batch it is MONOTONE with no interior knee on both models, right up to one
# row per microbatch, where the axis ends because B rows cannot be cut into more than B pieces.
# At 70B the best M is also the CHEAPEST in memory, and the harness default of 4 was the WORST
# point on that curve.
TRAIN_PP_MICROBATCHES_RULE = "one row per microbatch: --microbatches equal to the global batch"
TRAIN_PP_MICROBATCHES_SWEPT = {
    # model -> global batch -> {M: tok/s}, best-of/mean where a point was measured twice
    "unsloth/Qwen3.5-9B": {64: {4: 1108, 8: 1239, 16: 1343, 32: 1436, 64: 1473}},
    "unsloth/Llama-3.3-70B-Instruct": {16: {2: 111, 4: 133, 8: 142, 16: 161}},
}
# Raising the GLOBAL batch saturates almost at once and the control is FLAT: a single Spark
# training the 9B is bandwidth-bound.
TRAIN_PP_GLOBAL_BATCH_SWEPT = {64: (1481, 847), 128: (1502, 849), 256: (1508, 849)}
# Against a control swept over the same axis, not pinned at somebody else's M.
TRAIN_PP_BEST_SPEEDUP = 1.776

# The earlier speedups divided by a control that was ITSELF splitting batch 64 into 32 pieces,
# so the denominator was handicapped like the numerator; the ratio barely moves, so they are
# inflated by a couple of percent and the ordering is untouched. NOT rewritten here: this block
# did not re-measure DDP or FSDP, and a half-pinned table would be worse than a consistent one.
TRAIN_SPEEDUP_INFLATION_FROM_UNSWEPT_CONTROL = 0.03
# The recorded M=32 sat at the top of that curve by luck; moving off it either way costs.

# `dualpipev_gib` is None because the arm never reached a step it could report a peak for, and
# recording the number it died at would imply it ran.
TRAIN_PP_70B = {
    "model": "unsloth/Llama-3.3-70B-Instruct",
    "settings": "seq 512, global batch 16, M=16 (one row per microbatch), LoRA r=16, 6 steps, "
    "--shard-load --grad-checkpoint, both Sparks pinned at 1690 MHz",
    "1f1b_toks": 161,
    "1f1b_s_per_step": 50.94,
    "1f1b_steps": 6,
    "1f1b_microbatches": 16,
    "1f1b_loss_at_last_step": 12.1359,
    "1f1b_peak_gib": (68.45, 68.61),
    "1f1b_toks_by_microbatches": {2: 111, 4: 133, 8: 142, 16: 161},
    "1f1b_peak_gib_by_microbatches": {
        2: (75.15, 76.88),
        4: (71.49, 72.17),
        8: (69.46, 69.80),
        16: (68.45, 68.61),
    },
    "dualpipev_toks": None,
    "dualpipev_peak_gib": None,
    # Settled by exhaustion, not inferred from two failed cells: ScheduleDualPipeV at pp=2 has
    # num_stages=4, so M>=4 is a hard refusal and raising M only puts more microbatches in
    # flight, leaving batch 4 with M=4 the smallest configuration the V layout admits. It
    # failed too.
    "dualpipev_outcome": "out of memory on rank 0 at global batch 16, at batch 8, and at "
    "batch 4 with M=4 -- the smallest configuration the V layout admits, so no microbatch "
    "setting makes it viable at this size",
    "dualpipev_rank0_weights_gib": 68.06,
    "dualpipev_rank1_weights_gib": 64.14,
    "link_mb_moved": 1386,
    # Measured on the M=8 cell, and bytes per step do not depend on M, so it carries over.
    "link_measured_over_steps": 10,
    "link_measured_s_per_step": 55.96,
    "link_busbw_gbs": 20.31,
}


def plan_training(
    size_gib: Optional[float],
    n_nodes: int = 2,
    *,
    model: str = "<model>",
) -> Dict[str, Any]:
    """Which axis to TRAIN on, from model size and node count. Pure, measured.

    * A model that fits on one Spark trains data parallel: one whole model per node and
      the LoRA gradients averaged (`unsloth spark train --data-parallel`). Measured
      against a pipeline split of the same model on the same rows
      (TRAIN_DP_VS_PP_TOKS): the split pays a fill/drain bubble and holds the two halves
      in lockstep, the replicas do not. The price is memory: the whole model per node
      instead of half.
    * A model that does not fit trains layer split (`--layer-split --shard-load
      --grad-checkpoint`), with TRAIN_PP_SCHEDULE. There is no other option; DP would
      need the whole model on each node. No speedup is claimed for that case: it is a
      capacity feature. The schedule is a near-tie on models that fit, but at the sizes
      this branch actually fires for it is not a tie at all -- dualpipev could not train
      a 70B on the pair at any batch tried, so TRAIN_PP_SCHEDULE is 1f1b (TRAIN_PP_70B).

    ``fits`` uses the serving budget (SPARK_USABLE_GIB minus SERVE_OVERHEAD_GIB), which
    is deliberately looser than `training_memory_estimate`; that function is the one to
    run before a big job, this one only picks the axis.
    """
    budget = SPARK_USABLE_GIB - SERVE_OVERHEAD_GIB
    nodes = max(1, int(n_nodes or 1))
    out: Dict[str, Any] = {
        "size_gib": size_gib,
        "budget_gib": budget,
        "n_nodes": nodes,
        "measurement": TRAIN_MEASUREMENT,
    }
    if size_gib is None:
        out.update(
            axis = None,
            fits_one_node = None,
            schedule = None,
            speedup = None,
            measured = False,
            commands = [],
            recommendation = "could not determine model size; not guessing",
        )
        return out
    fits = size_gib <= budget
    out["fits_one_node"] = fits
    env = 'eval "$(unsloth spark env)"   # GB10 NCCL settings; NCCL_NET_GDR_LEVEL=0 is mandatory'
    if nodes < 2:
        out.update(
            axis = "single" if fits else "none",
            schedule = None,
            speedup = 1.0 if fits else None,
            measured = fits,
            commands = [f"unsloth train --model {_q(model)}"] if fits else [],
            recommendation = (
                f"{size_gib:.1f} GiB fits on this Spark ({budget:.0f} GiB budget); train "
                f"as usual. Pair a second Spark for {TRAIN_DP_SPEEDUP_RANGE[0]:.2f}x to "
                f"{TRAIN_DP_SPEEDUP_RANGE[1]:.2f}x."
                if fits
                else f"{size_gib:.1f} GiB does NOT fit on one Spark ({budget:.0f} GiB "
                f"budget). Pair a second Spark and layer-split it."
            ),
        )
        return out
    if fits:
        lo, hi = TRAIN_DP_SPEEDUP_RANGE
        plo, phi = TRAIN_PP_SPEEDUP_RANGE
        over_pp = min(TRAIN_DP_OVER_PP.values())
        out.update(
            axis = "data-parallel",
            schedule = None,
            speedup = lo,
            speedup_over_pipeline = over_pp,
            measured = True,
            commands = [env, f"unsloth spark train --data-parallel {_q(model)} --run"],
            recommendation = (
                f"{size_gib:.1f} GiB fits on one Spark ({budget:.0f} GiB budget): train it "
                f"DATA PARALLEL, one whole model per Spark. Measured {lo:.2f}x to {hi:.2f}x "
                f"over one Spark against {plo:.2f}x to {phi:.2f}x for a layer split of the "
                f"same model on the same rows, and DDP beat the best schedule by "
                f"{over_pp:.2f}x or more on every model tried. The split costs a fill/drain "
                f"bubble and buys nothing here; it is for capacity. Budget the WHOLE model "
                f"per node, not half."
            ),
        )
        return out
    out.update(
        axis = "pipeline-parallel",
        schedule = TRAIN_PP_SCHEDULE,
        speedup = None,
        # `measured` is about the SPEEDUP and stays False: there is no control to divide by.
        measured = False,
        schedule_measured = True,
        schedule_evidence = TRAIN_PP_70B,
        commands = [
            env,
            # NOT optional: the trainer default of 4 is the worst point on the M curve.
            f"unsloth spark train --layer-split {_q(model)} --shard-load --grad-checkpoint "
            f"--schedule {TRAIN_PP_SCHEDULE} --batch {TRAIN_PP_70B['1f1b_microbatches']} "
            f"--microbatches {TRAIN_PP_70B['1f1b_microbatches']} --run",
        ],
        recommendation = (
            f"{size_gib:.1f} GiB does NOT fit on one Spark ({budget:.0f} GiB budget): "
            f"layer-split it with --shard-load --grad-checkpoint, schedule "
            f"{TRAIN_PP_SCHEDULE}. A whole-model replica per node is impossible at this "
            f"size, so this is the only two-Spark axis -- it buys capacity, not speed. "
            f"Use one row per microbatch: --microbatches equal to the global batch. That is "
            f"measured, not a default -- at {TRAIN_PP_70B['model']} the M sweep at global "
            f"batch 16 reads "
            + ", ".join(
                f"{t} tok/s at M={m}"
                for m, t in sorted(TRAIN_PP_70B["1f1b_toks_by_microbatches"].items())
            )
            + f", so the trainer's default of 4 costs about 20 percent, and M=16 is also the "
            f"smallest in memory. "
            f"Do NOT substitute dualpipev here: at {TRAIN_PP_70B['model']} it ran out of "
            f"memory on rank 0 at global batch 16, again at batch 8, and again at batch 4 "
            f"with M=4 -- the smallest configuration a 4-stage V layout admits, so there is "
            f"no microbatch setting that rescues it, while "
            f"{TRAIN_PP_SCHEDULE} completed the same work at "
            f"{TRAIN_PP_70B['1f1b_peak_gib'][0]:.1f}/{TRAIN_PP_70B['1f1b_peak_gib'][1]:.1f} "
            f"GiB per rank and {TRAIN_PP_70B['1f1b_toks']} tok/s. The V layout co-locates the "
            f"first and last stages on one rank, so that rank carries the embedding, the LM "
            f"head, the loss and the deepest in-flight set together "
            f"({TRAIN_PP_70B['dualpipev_rank0_weights_gib']:.2f} GiB of weights against rank "
            f"1's {TRAIN_PP_70B['dualpipev_rank1_weights_gib']:.2f}, the difference being the "
            f"LM head), which is affordable at 2B and 9B and is not at 70B. "
            f"Run `unsloth spark estimate` first."
        ),
    )
    return out


INTENTS = ("latency", "throughput", "capacity")


def _nearest_concurrency(table: Dict[int, float], concurrency: int) -> Tuple[int, float]:
    key = min(table, key = lambda c: (abs(c - concurrency), c))
    return key, table[key]


def expected_gain(
    axis: str,
    n_nodes: int,
    concurrency: int = 1,
    prompt_tokens: int = 512,
) -> Dict[str, Any]:
    """What to expect from an axis at N nodes. Every constant here was measured at exactly
    TWO Sparks, so ``measured`` is part of the answer rather than a scaled number for N>2
    presented as if it had been benchmarked."""
    out: Dict[str, Any] = {
        "axis": axis,
        "n_nodes": n_nodes,
        "concurrency": concurrency,
        "speedup": None,
        "measured": False,
        "note": "",
    }
    if n_nodes <= 1 or axis in ("none", "single", None):
        out["speedup"] = 1.0
        out["measured"] = True
        out["note"] = "one node: nothing is distributed, so there is nothing to gain."
        return out
    if axis == "tensor-parallel":
        c, value = _nearest_concurrency(TP_SPEEDUP_2, concurrency)
        if n_nodes == 2:
            out.update(
                speedup = value,
                measured = True,
                note = (
                    f"measured {value:.2f}x at concurrency {c} on 2 Sparks; "
                    f"median TPOT {TP_TPOT_MS_2[0]:.1f}ms -> {TP_TPOT_MS_2[1]:.1f}ms."
                ),
            )
        else:
            out.update(
                speedup = None,
                measured = False,
                note = (
                    f"TP=2 measured {value:.2f}x; TP={n_nodes} across Sparks is "
                    f"NOT measured here. Every token needs an all-reduce over the "
                    f"RoCE link, and that cost grows with the node count while the "
                    f"per-node work shrinks, so expect clearly sublinear scaling and "
                    f"benchmark before believing a number."
                ),
            )
        return out
    if axis in ("pipeline-parallel", "layer-split"):
        c, value = _nearest_concurrency(PP_SPEEDUP_2, concurrency)
        out.update(
            speedup = value if n_nodes == 2 else None,
            measured = n_nodes == 2,
            note = (
                f"PP=2 measured {value:.2f}x end-to-end with median TPOT FLAT at "
                f"~{PP_TPOT_MS_2[0]:.0f}ms -- i.e. NO latency benefit. Pipelining "
                f"buys capacity: it is how a model too large for one Spark runs at "
                f"all, not how a request gets faster."
            ),
        )
        return out
    if axis == "replicas":
        if n_nodes == 2:
            gain = replicas_speedup(prompt_tokens or 512, concurrency)
            out.update(
                speedup = 1.0,
                measured = True,
                aggregate = gain,
                note = (
                    f"{gain:.2f}x AGGREGATE decode at {concurrency} concurrent, "
                    f"{prompt_tokens or 512} prompt tokens (measured at prompt 512: 1.00x "
                    f"at 1, 1.13x at 2 to 4, 1.30x at 8, 1.75x at 16, 1.91x at 32 users), "
                    f"1.00x per request. Two copies pay from {REPLICAS_MIN_USERS} users "
                    f"up; below that the second Spark is idle money."
                ),
            )
        else:
            out.update(
                speedup = 1.0,
                measured = False,
                aggregate = float(n_nodes) * REPLICA_AGGREGATE_PER_NODE,
                note = (
                    f"up to ~{n_nodes}x AGGREGATE throughput, 1.00x per request; only two "
                    f"replicas were measured here (1.91x at 32 users). Independent copies "
                    f"never make one request faster; they let you serve more at once."
                ),
            )
        return out
    if axis == "layer-split-fitting":
        out.update(
            speedup = LAYER_SPLIT_DECODE_ONLY_SPEEDUP,
            measured = True,
            note = (
                f"decode measured {LAYER_SPLIT_DECODE_ONLY_SPEEDUP:.2f}x, and 0.85x to "
                f"1.01x across 1 to 32 users: a layer split never speeds up decode for a "
                f"model that fits. It is a capacity feature and a prefill feature "
                f"({LAYER_SPLIT_PREFILL_SPEEDUP[0]:.1f}x to "
                f"{LAYER_SPLIT_PREFILL_SPEEDUP[1]:.2f}x prefill), so it pays only for "
                f"prefill-heavy long-prompt work at few users. For {REPLICAS_MIN_USERS} or "
                f"more users two replicas measured 1.30x to 1.91x instead."
            ),
        )
        return out
    if axis == "training-pipeline":
        out.update(
            speedup = TRAIN_PP_SPEEDUP_2 if n_nodes == 2 else None,
            measured = n_nodes == 2,
            note = (
                f"GPipe with M=4 microbatches measured {TRAIN_PP_SPEEDUP_2:.2f}x on 2 "
                f"Sparks (3024 vs 2032 tok/s). Bubbles, not bandwidth, are the ceiling -- "
                f"and for a model that FITS this axis is the wrong one entirely: see "
                f"`plan_training`, where data parallel measured "
                f"{TRAIN_DP_SPEEDUP_RANGE[0]:.2f}x against the best schedule's "
                f"{TRAIN_PP_SPEEDUP_RANGE[0]:.2f}x on the same rows."
            ),
        )
        return out
    out["note"] = f"unknown axis {axis!r}; no measurement to report."
    return out


def _nodes_needed(size_gib: float, budget: float) -> int:
    if budget <= 0:
        return 1
    count = int(size_gib / budget)
    if count * budget < size_gib - 1e-9:
        count += 1
    return max(1, count)


def _q(model: str) -> str:
    """A model path as one shell word, for a command line a human is meant to paste.

    These strings are copied into a terminal, so a checkpoint path with a space in it was
    splitting into two arguments and a path with a glob character was expanding against the
    current directory. Display strings elsewhere in this module are deliberately left bare;
    only the things that are meant to be RUN are quoted.
    """
    return shlex.quote(str(model))


def _serve_commands(
    axis: str,
    n_nodes: int,
    model: str = "<model>",
) -> List[str]:
    env = 'eval "$(unsloth spark env)"   # GB10 NCCL settings; NCCL_NET_GDR_LEVEL=0 is mandatory'
    # These lines exist to be pasted into a shell, so a local checkpoint path with a space or a
    # metacharacter has to survive the paste. `spark serve` already quotes; `spark plan` printed
    # the same paths bare and split them into several arguments. The placeholder stays bare so
    # the example still reads as a placeholder.
    qmodel = model if model == "<model>" else shlex.quote(model)
    if axis == "tensor-parallel":
        return [
            env,
            "ray start --head --port=6379            # on THIS Spark",
            "ray start --address=<this-spark>:6379   # on each of the other Sparks",
            f"vllm serve {qmodel} --tensor-parallel-size {n_nodes} "
            f"--distributed-executor-backend ray",
        ]
    # llama.cpp opens a GGUF FILE. A cached repo id or a safetensors directory is accepted by
    # `spark plan --model`, and emitting a llama.cpp command for one produced a plan whose
    # last step asks llama-server to open a directory as a GGUF. Those go to vLLM instead.
    gguf = model.endswith(".gguf") or model == "<model>"
    server = llama_server_binary() or "llama-server"
    if axis == "replicas":
        # spark_lb takes backends as positional, space-separated tokens.
        # The backends are on 8081 and the front door on 8080. Both on 8080 meant that the
        # load balancer -- which defaults to 8080 and is meant to run on one of the Sparks --
        # could not bind, because that node's own engine already held the port.
        backends = " ".join(
            f"{DEFAULT_SUBNETS[0]}.{NODE_BASE_OCTET + i}:8081" for i in range(n_nodes)
        )
        if not gguf:
            return [
                env,
                f"vllm serve {qmodel} --host 0.0.0.0 --port 8081     # run on EACH Spark",
                f"python -m studio.spark_lb --port 8080 {backends}     # one front door",
            ]
        # The real server, on the port the load balancer is told about. This used to say
        # `unsloth spark serve`, which prints a recipe rather than launching anything, and
        # the recipe it prints binds different ports -- so every backend the line below
        # advertised was closed.
        return [
            env,
            f"{shlex.quote(server)} -m {qmodel} -ngl 999 --host 0.0.0.0 --port 8081     # run on EACH Spark",
            f"python -m studio.spark_lb --port 8080 {backends}     # one front door",
        ]
    if axis in ("pipeline-parallel", "layer-split"):
        if not gguf:
            return [
                env,
                f"vllm serve {qmodel} --pipeline-parallel-size {n_nodes} "
                f"--distributed-executor-backend ray",
            ]
        return [
            env,
            # This one stays a `spark serve`: the RPC split needs the peer's bundle probed and
            # the protocol checked before a launch is worth printing, which is what that
            # command does. It PRINTS those commands; it does not start them.
            f"unsloth spark serve --model {qmodel} --engines 1   # prints the RPC split launch",
            f"# or, with vLLM:  vllm serve {qmodel} --pipeline-parallel-size {n_nodes} "
            f"--distributed-executor-backend ray",
        ]
    if axis == "single":
        if not gguf:
            return [f"vllm serve {qmodel} --host 0.0.0.0 --port 8080"]
        return [f"{shlex.quote(server)} -m {qmodel} -ngl 999 --host 0.0.0.0 --port 8080"]
    return []


def plan_deployment(
    size_gib: Optional[float],
    two_sparks: Optional[bool] = None,
    *,
    n_nodes: Optional[int] = None,
    intent: str = "throughput",
    concurrency: int = 1,
    model: str = "<model>",
    prompt_tokens: Optional[int] = None,
    prefill_heavy: bool = False,
    kv_gib_per_user: float = 0.0,
) -> Dict[str, Any]:
    """Recommend a topology AND an axis from model size, node count and intent. ``topology`` is
    a MEMORY-FIT class keeping its historical vocabulary; ``axis`` is the orthogonal question
    of which parallelism to use. A 70B that fits on one node is ``single-or-replicas`` with
    axis ``tensor-parallel`` for latency or ``replicas`` for throughput."""
    budget = SPARK_USABLE_GIB - SERVE_OVERHEAD_GIB
    if n_nodes is None:
        nodes = 1 if two_sparks is None else (2 if two_sparks else 1)
    else:
        try:
            nodes = max(1, int(n_nodes))
        except (TypeError, ValueError):
            nodes = 1
    if intent not in INTENTS:
        intent = "throughput"

    out: Dict[str, Any] = {
        "size_gib": size_gib,
        "budget_gib": budget,
        "n_nodes": nodes,
        "cluster_gib": nodes * budget,
        "intent": intent,
        "concurrency": concurrency,
    }

    # Never guess: wrong advice is indistinguishable from right advice until the run fails.
    if size_gib is None:
        out.update(
            topology = "unknown",
            axis = None,
            fits = None,
            commands = [],
            command = "",
            expected = expected_gain("none", 1, concurrency),
            recommendation = "",
            summary = "could not determine model size; not guessing",
        )
        return out

    fits_one = size_gib <= budget
    min_nodes = _nodes_needed(size_gib, budget)
    out["min_nodes"] = min_nodes
    out["fits_one_node"] = fits_one

    if nodes < 2:
        out.update(
            topology = "single",
            fits = fits_one,
            axis = "single" if fits_one else "none",
            summary = (
                f"{size_gib:.1f} GiB fits on this Spark ({budget:.0f} GiB budget)"
                if fits_one
                else f"{size_gib:.1f} GiB does NOT fit on one Spark ({budget:.0f} GiB "
                f"budget) -- pair a second one, or use a smaller quant"
            ),
            expected = expected_gain("none", 1, concurrency),
            commands = _serve_commands("single" if fits_one else "none", 1, model),
        )
        out["recommendation"] = (
            f"Serve it on this Spark. With one node there is no axis to choose."
            if fits_one
            else f"This needs at least {min_nodes} Sparks at {budget:.0f} GiB each, or a "
            f"smaller quant. One Spark cannot run it at any speed."
        )
        out["command"] = "\n".join(out["commands"])
        return out

    # KV counted here, not only in `recommend_topology`: a layer split spreads the KV with the
    # layers, so what has to fit across the pair is model plus every user's KV. Weights alone
    # let a model that needs more than the pair holds through as `layer-split`, and `serve`
    # then printed an RPC launch that OOMs during load. The per-node classes below keep their
    # weight-only meaning; `recommend_topology` prices KV for those.
    split_need = size_gib + max(0.0, float(kv_gib_per_user or 0)) * max(1, int(concurrency or 1))
    out["split_need_gib"] = split_need
    if split_need > nodes * budget:
        topology = "too-large"
    elif not fits_one:
        topology = "layer-split"
    elif size_gib * 2 <= budget:
        # Historical meaning, kept: two engines fit side by side on ONE node.
        topology = "replicas"
    else:
        topology = "single-or-replicas"
    out["topology"] = topology
    out["fits"] = topology != "too-large"
    if topology != "too-large":
        out["serving"] = recommend_topology(
            size_gib * 2**30,
            max(0.0, float(kv_gib_per_user or 0)) * 2**30,
            concurrency,
            prompt_tokens or 512,
            budget * 2**30,
            prefill_heavy = prefill_heavy,
        )

    # `summary` answers ONLY "what fits where"; every statement about which axis to use lives
    # in `recommendation`. Overlapping them reads as the tool contradicting itself.
    copies = int(budget // size_gib) if size_gib > 0 else 0
    if topology == "replicas":
        out["summary"] = (
            f"{size_gib:.1f} GiB against a {budget:.0f} GiB budget per node: TWO copies "
            f"fit side by side on a single Spark, so all {nodes} nodes have room to spare."
            if copies < 3
            else f"{size_gib:.1f} GiB against a {budget:.0f} GiB budget per node: {copies} "
            f"copies fit on each of your {nodes} Sparks."
        )
    elif topology == "single-or-replicas":
        out["summary"] = (
            f"{size_gib:.1f} GiB fits on ONE Spark ({budget:.0f} GiB budget), but a second "
            f"copy does not fit beside it on the same node. Each of your {nodes} Sparks "
            f"can hold exactly one copy."
        )
    elif topology == "layer-split":
        out["summary"] = (
            f"{size_gib:.1f} GiB exceeds one Spark's {budget:.0f} GiB, so it cannot run on "
            f"a single node. It fits across {min_nodes} of your {nodes} "
            f"({nodes * budget:.0f} GiB total), which means it has to be sharded somehow."
        )
    else:
        need_nodes = _nodes_needed(split_need, budget)
        # Say which of the two it was. "34 GiB exceeds 243 GiB" reads as a bug when the
        # weights fit and it is the KV for the requested concurrency that does not.
        what = (
            f"{size_gib:.1f} GiB of weights plus KV for {concurrency} " f"({split_need:.1f} GiB)"
            if split_need > size_gib
            else f"{size_gib:.1f} GiB"
        )
        out["summary"] = (
            f"{what} exceeds all {nodes} Sparks together ({nodes * budget:.0f} "
            f"GiB usable). At least {need_nodes} nodes would be needed, or a smaller quant."
        )

    if topology == "too-large":
        need_nodes = _nodes_needed(split_need, budget)
        out.update(axis = "none", expected = expected_gain("none", 1, concurrency), commands = [])
        out["recommendation"] = (
            f"No topology helps: {split_need:.1f} GiB does not fit in {nodes} x "
            f"{budget:.0f} GiB. Add nodes until you have {need_nodes}, quantise smaller, or "
            f"serve fewer concurrent users."
        )
    elif not fits_one:
        # It must be sharded to run at all; PP/layer-split is the fallback if TP is absent.
        shard_nodes = min(nodes, max(2, min_nodes))
        out.update(
            axis = "tensor-parallel",
            axis_nodes = shard_nodes,
            expected = expected_gain("tensor-parallel", shard_nodes, concurrency),
            commands = _serve_commands("tensor-parallel", shard_nodes, model),
        )
        out["fallback_axis"] = "pipeline-parallel"
        out["fallback_expected"] = expected_gain("pipeline-parallel", shard_nodes, concurrency)
        out["recommendation"] = (
            f"This model cannot run on one Spark, so it must be sharded. Use TENSOR "
            f"parallel across {shard_nodes} nodes: at TP=2 that measured 2.09x a single "
            f"Spark and halved median TPOT (332.7ms -> 162.4ms), so it is both the way to "
            f"run this at all and the fastest way. Pipeline/layer split is the fallback "
            f"when your engine cannot TP across hosts (llama.cpp RPC): it measured 1.08x "
            f"with FLAT TPOT -- capacity only."
        )
    elif intent == "latency":
        out.update(
            axis = "tensor-parallel",
            axis_nodes = nodes,
            expected = expected_gain("tensor-parallel", nodes, concurrency),
            commands = _serve_commands("tensor-parallel", nodes, model),
        )
        out["recommendation"] = (
            f"TENSOR parallel across {nodes} Sparks. It is the only axis that makes a "
            f"single request faster: TP=2 measured 2.09x with median TPOT 332.7ms -> "
            f"162.4ms. Do NOT use pipeline parallel for this -- its TPOT is flat at "
            f"~320ms -- and do NOT layer-split a model that fits: its decode measured "
            f"0.85x to 1.01x across 1 to 32 users, never a win."
        )
    elif intent == "throughput" and (out.get("serving") or {}).get("topology") != "replicas":
        # `recommend_topology` already decided this, with the concurrency and the KV in hand.
        # Hard-coding replicas for every throughput intent contradicted it out loud: at the
        # default concurrency of 1 it returns `single` and says the second copy buys nothing,
        # and for prefill-heavy work it can return a split -- and the plan then told the user
        # to spend a second copy of the weights on a layout its own measured policy rejects.
        chosen = (out.get("serving") or {}).get("topology") or "single"
        axis = "pipeline-parallel" if chosen == "layer_split" else "single"
        axis_nodes = nodes if chosen == "layer_split" else 1
        out.update(
            axis = "none" if axis == "single" else axis,
            axis_nodes = axis_nodes,
            expected = expected_gain(axis, axis_nodes, concurrency, prompt_tokens or 512),
            commands = _serve_commands(
                "layer-split" if chosen == "layer_split" else "single", axis_nodes, model
            ),
        )
        out["recommendation"] = (out.get("serving") or {}).get("reason", "") or (
            f"One Spark: {size_gib:.1f} GiB fits, and a second copy is not worth its memory "
            f"at this concurrency."
        )
    elif intent == "throughput":
        gain = replicas_speedup(prompt_tokens or 512, concurrency)
        out.update(
            axis = "replicas",
            axis_nodes = nodes,
            expected = expected_gain("replicas", nodes, concurrency, prompt_tokens or 512),
            commands = _serve_commands("replicas", nodes, model),
        )
        out["recommendation"] = (
            f"REPLICAS: one independent server per Spark, {nodes} in total, behind "
            f"`python -m studio.spark_lb`. Two replicas measured {gain:.2f}x aggregate "
            f"decode at {concurrency} concurrent (1.30x at 8, 1.75x at 16, 1.91x at 32 "
            f"users; only 1.00x to 1.13x below {REPLICAS_MIN_USERS}, where one Spark is as "
            f"good and the second copy is wasted). It does not make any single request "
            f"faster -- if that is what you want, ask for intent=latency and use tensor "
            f"parallel instead. Never layer-split a model that fits for throughput: "
            f"decode measured 0.85x to 1.01x."
        )
    else:  # capacity, and it already fits
        out.update(
            axis = "none",
            axis_nodes = 1,
            expected = expected_gain("none", 1, concurrency),
            commands = _serve_commands("single", 1, model),
        )
        out["recommendation"] = (
            f"For capacity, MORE SPARKS WILL NOT HELP YOU HERE: {size_gib:.1f} GiB already "
            f"fits in one node's {budget:.0f} GiB. Serve it on a single Spark. The extra "
            f"nodes are worth using only for throughput (replicas, 1.30x to 1.91x "
            f"aggregate decode at 8 to 32 users) or for latency (tensor parallel, 2.09x "
            f"measured at 2 nodes)."
        )
    out["command"] = "\n".join(out.get("commands") or [])
    return out


def _cmd_plan(
    model: str,
    intent: str = "throughput",
    nodes: Optional[int] = None,
    concurrency: int = 1,
    prompt_tokens: int = 512,
    prefill_heavy: bool = False,
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to plan.")
        return 0
    if nodes is None:
        info = discover_peers(timeout = 0.0)
        nodes = max(info.get("n_nodes", 1), 2 if peer_ip_for() else 1)
    size = model_size_gib(model)
    plan = plan_deployment(
        size,
        n_nodes = nodes,
        intent = intent,
        concurrency = concurrency,
        model = model,
        prompt_tokens = prompt_tokens,
        prefill_heavy = prefill_heavy,
    )
    print(f"  model     : {model}")
    print(f"  size      : " + (f"{size:.1f} GiB" if size else "unknown (not cached locally)"))
    print(f"  Sparks    : {nodes}")
    print(f"  intent    : {intent}")
    if concurrency != 1 or prompt_tokens != 512:
        print(f"  traffic   : {concurrency} concurrent, {prompt_tokens} prompt tokens")
    print(f"  topology  : {plan['topology']}")
    if plan.get("axis"):
        print(f"  axis      : {plan['axis']}")
    print("")
    print(f"  {plan['summary']}")
    if plan.get("recommendation"):
        print("")
        print(f"  {plan['recommendation']}")
    serving = plan.get("serving")
    if serving:
        print("")
        print(f"  llama.cpp : {serving['topology']}")
        print(f"              {serving['reason']}")
        print(f"              (measured on {serving['measured_on']})")
        if serving.get("mtp_note"):
            print(f"  MTP       : {serving['mtp_note']}")
        if serving.get("split_mtp_note"):
            print(f"  split MTP : {serving['split_mtp_note']}")
    exp = plan.get("expected") or {}
    if exp.get("note"):
        print("")
        label = "measured" if exp.get("measured") else "NOT measured at this node count"
        speed = f"{exp['speedup']:.2f}x" if exp.get("speedup") is not None else "unknown"
        print(f"  expected  : {speed} ({label})")
        print(f"              {exp['note']}")
    if plan.get("commands"):
        print("")
        print("  Run:")
        for line in plan["commands"]:
            print(f"    {line}")
    return 0


def _cmd_peers(check: bool = True) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; no peers to look for.")
        return 0
    info = discover_peers(check_reachable = check)
    print(f"  cable present : {info['cable_present']}")
    print(f"  nodes         : {info['n_nodes']} (this Spark + {info['n_peers']} peer(s))")
    if not info["peers"]:
        print("")
        print("  No peer Sparks discovered. mDNS only sees peers that advertise, so a")
        print('  switched cluster may need them written down: add a "peers" list of')
        print(f'  {{"hostname": ..., "address": ...}} to {config_path()}.')
        return 0
    for peer in info["peers"]:
        state = {True: "reachable", False: "UNREACHABLE", None: "not probed"}[peer["reachable"]]
        print(
            f"    node {peer['index']}  {peer['short']:<16} {peer['address']:<18} "
            f"{state}  ({peer['source']})"
        )
    # mDNS answers with whatever interface advertised, usually Wi-Fi, not the 200 Gb/s rail.
    rail_peer = peer_ip_for()
    if rail_peer:
        print("")
        print(f"  rail peer     : {rail_peer} (from the addressing plan, not from mDNS)")
        print("    Use THIS address for NCCL/ray/rsync. An address discovered over mDNS is")
        print("    typically the Wi-Fi one and would not touch the ConnectX link at all.")
    if any(p["reachable"] is False for p in info["peers"]):
        print("")
        print("  An unreachable peer will not fail loudly in a distributed launch: the head")
        print("  rank blocks at the rendezvous for 601s and then reports only")
        print("  'DistStoreError: N/M clients joined'. Fix reachability first.")
    return 0


# Measured on GB10, identical weights and one GEMM shape, only the kernel changing. No
# single "best kernel" exists: the ranking INVERTS with batch size, worth up to 6.2x, and the
# crossover sits between M=32 and M=256, matching the roofline knee at M~436.
NVFP4_KERNELS = {
    "decode": {
        "backend": "marlin",
        "flag": "--linear-backend marlin --moe-backend marlin",
        "why": (
            "fastest measured at M=1 (429 us). At decode batch sizes every format runs at "
            "94-106% of achievable memory bandwidth, so 4-bit activations cannot help -- "
            "and Marlin's 16-bit compute costs nothing it was not already paying in stalls."
        ),
    },
    "prefill": {
        "backend": "flashinfer_cutlass",
        "flag": "--linear-backend flashinfer_cutlass",
        "why": (
            "309 TF/s against Marlin's 50 TF/s at M=4096 -- a 6.2x difference. Prefill is "
            "compute-bound, which is the one regime where FP4's 3.3x arithmetic advantage "
            "over BF16 is reachable."
        ),
        # NVFP4_FINDINGS.md sections 12, 33 and 37. The earlier "b12x 1.6x at long prompts"
        # claim was refuted by the 4096-token study, so b12x is not a serving recommendation.
        "note": (
            "validated on vLLM 0.28.0; on vLLM main the same override also selects the FP8 "
            "ScaledMM kernel for the FP8 layers of Unsloth's mixed checkpoints and fails at "
            "init; there, leave auto. flashinfer_b12x only for offline batch prefill, about 5%."
        ),
    },
}


def recommend_kernels(workload: str = "mixed") -> Dict[str, Any]:
    """Kernel choice for an NVFP4 model on GB10, by workload. The fastest decode kernel is
    the slowest prefill kernel; vLLM auto-selects well for decode, so the actionable case is
    prefill-heavy work, where the explicit flag is worth multiples."""
    out: Dict[str, Any] = {"workload": workload}
    if workload in NVFP4_KERNELS:
        out.update(NVFP4_KERNELS[workload])
        return out
    out.update(
        backend = "marlin for decode, flashinfer_cutlass for prefill",
        flag = "(choose per workload; see `unsloth spark kernels --workload prefill`)",
        why = (
            "The crossover is at roughly 256 tokens per forward pass. Chat-style decode "
            "wants Marlin; RAG, summarisation and long-prompt workloads want CUTLASS."
        ),
    )
    return out


def _cmd_kernels(workload: str = "mixed") -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; these measurements do not apply.")
        return 0
    rec = recommend_kernels(workload)
    print(f"  workload : {rec['workload']}")
    print(f"  kernel   : {rec['backend']}")
    print(f"  flag     : {rec['flag']}")
    print("")
    print(f"  {rec['why']}")
    if rec.get("note"):
        print(f"  Note: {rec['note']}")
    print("")
    print("  Measured on this hardware (same weights, same shape, kernel varied):")
    print("    kernel        acts   M=1        M=4096")
    print("    marlin        A16    429 us     29257 us /  50 TF")
    print("    fi_cutlass    A4     447 us      4727 us / 309 TF")
    print("    vllm_cutlass  A4     486 us      4511 us / 324 TF")
    print("    bf16          A16   1544 us     15339 us /  95 TF")
    print("")
    print("  Also on GB10: pin `nvidia-cutlass-dsl==4.6.2` -- 4.7.0 fails b12x with an")
    print("  internal DSL compiler error, disabling the kernel family built for this GPU.")
    print("  b12x itself is an offline batch-prefill kernel (about 5% there, nothing for")
    print("  serving); the serving default is auto.")
    return 0


def model_dimensions(target: str) -> Optional[Dict[str, int]]:
    """`{"layers", "hidden", "vocab"}` for `target`, or None when they cannot be read.

    The training estimate needs these and had defaults for them, so every model was sized as
    an 80-layer, 8192-hidden, 128256-vocabulary one. The vocabulary is the term that decides
    the answer: the last stage holds the fp32 logits, and `mb_rows * seq * vocab * 4` is
    usually the largest single tensor in the step. A model with a bigger vocabulary got an
    `OK` and then exhausted the node."""
    path = osp.expanduser(target)
    if osp.isfile(path) and path.endswith(".gguf"):
        meta = gguf_metadata(path)
        arch = meta.get("general.architecture")
        if isinstance(arch, str):
            layers = meta.get(f"{arch}.block_count")
            hidden = meta.get(f"{arch}.embedding_length")
            vocab = meta.get(f"{arch}.vocab_size")
            if all(isinstance(v, int) and v > 0 for v in (layers, hidden, vocab)):
                return {"layers": layers, "hidden": hidden, "vocab": vocab}
        return None

    root = path if osp.isdir(path) else None
    if root is None:
        cache = osp.expanduser("~/.cache/huggingface/hub")
        repo = osp.join(cache, "models--" + target.replace("/", "--"))
        root = _hf_snapshot_dir(repo) if osp.isdir(repo) else None
    if root is None:
        return None
    try:
        with open(osp.join(root, "config.json"), "r", encoding = "utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    # Multimodal configs put the decoder's own numbers under `text_config`, and it is the
    # decoder that this estimate is about.
    inner = config.get("text_config")
    source = inner if isinstance(inner, dict) else config
    layers = source.get("num_hidden_layers", config.get("num_hidden_layers"))
    hidden = source.get("hidden_size", config.get("hidden_size"))
    vocab = source.get("vocab_size", config.get("vocab_size"))
    if not all(isinstance(v, int) and v > 0 for v in (layers, hidden, vocab)):
        return None
    return {"layers": layers, "hidden": hidden, "vocab": vocab}


def training_memory_estimate(
    size_gib: float,
    world: int,
    batch: int,
    microbatches: int,
    seq: int,
    # No defaults. They were one 70B's shape, and a caller that forgot them got that shape
    # for every model, silently; the only caller now reads them from the checkpoint.
    hidden: int,
    layers: int,
    vocab: int,
    checkpointed: bool = True,
) -> Dict[str, Any]:
    """Per-node memory for a layer-split training step, before it is attempted. The failure
    it prevents is severe: a 70B arm spent an hour materialising, exhausted the node on
    activations and left it UNREACHABLE over ssh (userspace could no longer fork), needing a
    power cycle. Deliberately pessimistic: it refuses the impossible, not the last GiB."""
    weights = size_gib / world
    # Adam: fp32 master + two moments. LoRA trains a tiny fraction; a full finetune pays all.
    optimizer_full = weights * 6.0
    optimizer_lora = weights * 0.02
    mb_rows = max(batch // max(microbatches, 1), 1)
    own_layers = max(layers // world, 1)
    # Residual stream per layer per in-flight microbatch, bf16.
    per_layer = mb_rows * seq * hidden * 2 / 2**30
    live_microbatches = microbatches if not checkpointed else min(microbatches, 2)
    activations = per_layer * own_layers * live_microbatches
    if not checkpointed:
        activations *= 4.0  # attention + MLP intermediates kept for the backward pass
    # The LAST stage also holds logits, computed in fp32, and for a large vocabulary that one
    # tensor dominates. Ignoring it is why a naive estimate says a run fits and the final
    # stage is the one that runs out.
    logits_gib = (mb_rows * seq * vocab * 4 / 2**30) * live_microbatches * 2
    activations_last_stage = activations + logits_gib
    budget = SPARK_USABLE_GIB - 6.0  # driver, CUDA context, fragmentation
    # Size the answer on the WORST stage, which is the last one.
    worst_activations = max(activations, activations_last_stage)
    total_lora = weights + optimizer_lora + worst_activations
    total_full = weights + optimizer_full + worst_activations
    return {
        "weights_gib": weights,
        "activations_gib": worst_activations,
        "logits_gib": logits_gib,
        "optimizer_lora_gib": optimizer_lora,
        "optimizer_full_gib": optimizer_full,
        "total_lora_gib": total_lora,
        "total_full_gib": total_full,
        "budget_gib": budget,
        "fits_lora": total_lora <= budget,
        "fits_full": total_full <= budget,
        "tokens_per_microbatch": mb_rows * seq,
    }


def _cmd_estimate(
    model: str, batch: int, microbatches: int, seq: int, full_finetune: bool, checkpointed: bool
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to estimate.")
        return 0
    sized = model_size_report(model)
    size = sized["gib"]
    if size is None:
        print(f"Cannot size {model}: {sized['why']}. Refusing to guess.")
        return 1
    # No architecture, no verdict. The defaults are one 70B's shape, and applying them to
    # every checkpoint is what let a larger vocabulary print OK and then run out of memory.
    dims = model_dimensions(model)
    if dims is None:
        print(f"  model      : {model}  ({size:.1f} GiB)")
        print("")
        print("  CANNOT ESTIMATE: this model's layer count, hidden size and vocabulary could")
        print("  not be read (no config.json, and no GGUF header that declares them). The")
        print("  logits term alone is `batch/microbatches * seq * vocab * 4` bytes, so a")
        print("  guessed vocabulary decides the answer. Point --model at the checkpoint")
        print("  directory or the cached repo id instead of a bare name.")
        return 1
    world = 2 if peer_ip_for() else 1
    est = training_memory_estimate(
        size,
        world,
        batch,
        microbatches,
        seq,
        hidden = dims["hidden"],
        layers = dims["layers"],
        vocab = dims["vocab"],
        checkpointed = checkpointed,
    )
    fits = est["fits_full"] if full_finetune else est["fits_lora"]
    total = est["total_full_gib"] if full_finetune else est["total_lora_gib"]
    mode = "full finetune" if full_finetune else "LoRA"

    print(f"  model      : {model}  ({size:.1f} GiB)")
    print(
        f"  shape      : {dims['layers']} layers, hidden {dims['hidden']}, "
        f"vocab {dims['vocab']}"
    )
    print(f"  stages     : {world}")
    print(
        f"  per node   : weights {est['weights_gib']:.1f} + "
        f"activations {est['activations_gib']:.1f} + optimizer "
        f"{(est['optimizer_full_gib'] if full_finetune else est['optimizer_lora_gib']):.1f}"
        f" = {total:.1f} GiB"
    )
    print(f"  budget     : {est['budget_gib']:.1f} GiB per node")
    print(
        f"  microbatch : {est['tokens_per_microbatch']} tokens"
        + (
            ""
            if est["tokens_per_microbatch"] >= 436
            else "  <-- BELOW the ~436-token crossover; the split cannot speed this up"
        )
    )
    print("")
    if fits:
        print(f"  OK: {mode} should fit, with " f"{est['budget_gib'] - total:.1f} GiB headroom.")
        return 0
    print(
        f"  WILL NOT FIT: {mode} needs {total:.1f} GiB against a "
        f"{est['budget_gib']:.1f} GiB budget."
    )
    print("  Reduce --batch or --seq, raise --microbatches, or add --grad-checkpoint.")
    if not checkpointed:
        print("  Gradient checkpointing alone would cut activations roughly 8x here.")
    return 1


def _consented(assume_yes: bool, prompt: str) -> bool:
    """Explicit yes, an interactive yes, or no. Never a default yes. The failure that
    motivated this was an `rsync --delete` of the studio venv onto a peer running a job out
    of it, reached by calling the setup entry point. No TTY and no flag means no, so no
    automation can trip it by accident."""
    if assume_yes:
        return True
    try:
        watching = sys.stdout.isatty()
    except (AttributeError, ValueError):
        watching = False
    if not watching:
        return False  # nobody is there to answer; the answer is no
    try:
        if sys.stdin.isatty():
            return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (AttributeError, ValueError, EOFError, KeyboardInterrupt, OSError):
        return False
    # stdout is a terminal but stdin is not: `curl ... | sh`, where the script occupies
    # stdin. Read /dev/tty as install.sh does; if that fails it is a real "no terminal".
    try:
        with open("/dev/tty", "r", encoding = "utf-8") as tty:
            print(f"{prompt} [y/N] ", end = "", flush = True)
            return (tty.readline() or "").strip().lower() in ("y", "yes")
    except (OSError, EOFError, KeyboardInterrupt):
        return False


def _cmd_setup(
    assume_yes: bool = False,
    n_nodes: int = 2,
    switched: bool = False,
    dry_run: bool = False,
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to do.")
        return 0
    rails = cabled_rails()
    if not rails:
        print("No cabled ConnectX rail found. Connect the QSFP cable between the two")
        print("Sparks (use the SAME physical port on both), then re-run:")
        print("    unsloth spark setup")
        return 1

    n_nodes = max(2, int(n_nodes))
    report = rail_plan_report(rails, node_index = 0, n_nodes = n_nodes, switched = switched)
    if not report["ok"]:
        # Refuse rather than emit a netplan that looks right and routes nowhere.
        print(f"Cannot plan addressing for {n_nodes} Sparks:")
        for problem in report["problems"]:
            print(f"  - {problem}")
        return 1
    plan = report["plan"]
    peer_plan = rail_plan(rails, node_index = 1, n_nodes = n_nodes, switched = switched)
    extra_plans = [
        rail_plan(rails, node_index = i, n_nodes = n_nodes, switched = switched)
        for i in range(2, n_nodes)
    ]
    for note in report["notes"]:
        print(f"  NOTE: {note}")
    print(
        "Detected a cabled second Spark on:" if n_nodes == 2 else f"Planning {n_nodes} Sparks on:"
    )
    for entry in plan:
        print(
            f"  {entry['ib_device']:<14} {entry['netdev']:<16} -> {entry['address']}/24 mtu {entry['mtu']}"
        )

    _print_manual_steps(plan, peer_plan, extra_plans = extra_plans)
    print("\n  NOTE: " + HOTPLUG_NOTE)

    print("\n  Then verify with:")
    print(f"    ping -c3 {peer_plan[0]['address']}")
    print("    unsloth spark status")

    # Provisioning WRITES TO ANOTHER MACHINE, so it is never automatic: printing the plan is
    # free, performing it needs a yes.
    peer_now = peer_ip_for()
    changes = [f"rewrite {config_path()} with the plan above"]
    if peer_now:
        for path, label in provision_paths():
            if osp.isdir(osp.expanduser(path)):
                changes.append(f"rsync {path}/ -> {peer_now}:{path}/  ({label})")
    print("\n  This command would then:")
    for change in changes:
        print(f"    - {change}")
    if peer_now:
        print(f"    (the peer's GPU is checked first; a busy or unverifiable {peer_now}")
        print("     is refused, because a job may be running out of that venv)")

    if dry_run:
        print("\n  --dry-run: nothing was written here and nothing was sent to the peer.")
        if peer_now:
            print("  Preview the file list with: unsloth spark provision --dry-run")
        return 0
    if not _consented(assume_yes, "\n  Apply this plan and provision the peer?"):
        print("\n  Not applied. Nothing was written here and nothing was sent to the peer.")
        print("  Re-run with --yes to apply, or --dry-run to see it again.")
        return 0

    provision_failures = False
    if peer_now:
        print(f"\n  Peer {peer_now} -- copying environment and caches over the ConnectX link:")
        res = provision_peer(peer_now)
        if res["refused"]:
            print(f"    REFUSED: {res['refused']}")
            print("    Nothing was copied. Re-run `unsloth spark provision` when it is idle.")
            provision_failures = True
        for label, _ in res["copied"]:
            print(f"    ok      {label}")
        for label, why in res["failed"]:
            print(f"    FAILED  {label}: {why}")
        if res["failed"]:
            print("    Re-run later with: unsloth spark provision")
            provision_failures = True
    else:
        print("\n  Once the peer is reachable, run: unsloth spark provision")

    saved = save_config(
        {
            "enabled": True,
            "planned": True,
            "n_nodes": n_nodes,
            "switched": switched,
            "rails": plan,
            "peer_rails": peer_plan,
            "other_rails": extra_plans,
            "nccl_env": nccl_env(rails),
        }
    )
    if not saved:
        return 1
    print(f"\nSaved plan to {config_path()}")
    # A copy that was attempted and failed leaves the peer without the environment, so the
    # installer and `spark up` must not read this as a finished setup. An unreachable peer is
    # different: that is the documented "provision it later" path and stays a success.
    if provision_failures:
        return 1
    # The netplan above is PRINTED, never applied: this command does not write /etc/netplan
    # or run `netplan apply`, because reconfiguring a machine's networking as a side effect
    # of a plan is not something it may do unasked. What it must not do either is return
    # success while the rails still carry no address -- `spark up --yes` then failed its own
    # `cluster_state() == "configured"` check immediately after setup said it was done, with
    # nothing naming the step in between.
    current = cabled_rails()
    if not current or not all(r["ipv4"] for r in current):
        print("\n  NOT YET CONFIGURED: the rails still have no address, because the netplan")
        print("  above has to be applied by you. Run the numbered commands on both Sparks,")
        print("  then re-run `unsloth spark setup` to provision the peer.")
        return 1
    return 0


def _cmd_serve(
    model: str,
    port: int = 8080,
    rpc_port: int = RPC_DEFAULT_PORT,
    ctx: int = 8192,
    engines: int = 2,
    slots: int = 16,
) -> int:
    """Serve a GGUF across BOTH Sparks, using the layout that actually wins. A single split
    engine never decodes faster than one Spark: the nodes take turns on the graph, and a
    single autoregressive stream cannot be pipelined because token t+1 depends on token t.
    Two independent engines give the pair data-independent work, so split only when the model
    does not fit. Bundles are compared before a split is printed."""
    if not is_dgx_spark():
        print(NOT_A_SPARK)
        return 0
    peer_ip = peer_ip_for()
    if peer_ip is None:
        print(
            "  cannot serve across both Sparks: no configured peer rail (run `unsloth spark setup`)"
        )
        return 1
    engines = max(1, engines)
    # These lines are printed to be pasted into a shell. A checkpoint path with a space or a
    # shell metacharacter was interpolated bare, so the pasted command split it into several
    # arguments or ran part of it. The pipeline command generator already quotes.
    qmodel = shlex.quote(model)

    # Decide the topology from the model: the rule is not guessable, and the right answer
    # flips at the point where two copies stop fitting.
    sized = model_size_report(model)
    size = sized["gib"]
    if size is None:
        # Falling through printed a two-engine split recipe for a model whose size was never
        # established, which is the one thing this command must not do.
        print(f"  cannot plan a deployment: {sized['why']}")
        return 1
    kv = serving_kv_gib_per_user(model, ctx)
    advice = plan_deployment(
        size,
        two_sparks = True,
        concurrency = slots,
        kv_gib_per_user = kv["gib"] or 0.0,
    )
    # The fit class answers "how many copies fit on ONE node", which is not the launch
    # topology: `single-or-replicas` means each Spark holds exactly one copy, so two replicas
    # DO fit across the pair. Reading it as a split forced the 0.92x layout onto the range
    # where replicas measured 1.30x to 1.91x. `serving` is the recommendation that already
    # weighs concurrency and KV; use it, and keep the fit class for what does not fit at all.
    serving = advice.get("serving") or {}
    topology = serving.get("topology") or advice["topology"]
    if advice["topology"] == "too-large":
        print(f"  {advice['summary']}")
        return 1

    if topology in ("replicas", "single"):
        # Independent replicas never touch the wire during decode: each Spark runs at full
        # local memory bandwidth, coordinated only by a request-level round-robin. No RPC is
        # involved, so nothing here may require the RPC server to be present.
        server = llama_server_binary()
        if server is None:
            print(f"  no llama-server in {llama_bundle_dir()}; reinstall the llama.cpp bundle")
            return 1
        bin_dir = Path(server).parent
        peer_bin_dir = _peer_relative_path(bin_dir)
        local_port, peer_port = port + 1, port + 2
        print(f"  model    : {model}  ({size:.1f} GiB)")
        if kv["gib"]:
            # `--ctx-size` is llama-server's SHARED pool, not a per-slot budget: with `-np N`
            # the slots divide it. The planner prices `ctx` per user, so the emitted commands
            # ask for `ctx * slots` and the two now describe the same deployment. Checked
            # against the managed build's own help, which says `--kv-unified-per-slot N`
            # sizes the shared pool to `n_parallel*N` -- the same arithmetic, spelled with a
            # flag a stock llama.cpp does not have.
            print(
                f"  kv       : {kv['gib']:.2f} GiB per user at {ctx} tokens each, "
                f"{kv['gib'] * slots:.2f} GiB for {slots} slots"
            )
        else:
            print(f"  kv       : not counted -- {kv['why']}")
        if topology == "single":
            print("  topology : ONE SPARK -- the second buys nothing at this concurrency")
            print("")
            print(f"  {serving.get('reason', '')}")
            print("")
            print(f"     {shlex.quote(str(bin_dir))}/llama-server -m {qmodel} \\")
            print(f"         -ngl 999 --ctx-size {ctx * slots} -np {slots} -cb -ub 512 \\")
            print(f"         --host 0.0.0.0 --port {port}")
            return 0
        print("  topology : INDEPENDENT REPLICAS -- one full model per Spark, no RPC")
        print("")
        print("  One copy per Spark, and for a model that fits this beats every split layout:")
        print("  a layer split never speeds up decode (0.85x to 1.01x measured from 1 to 32")
        print("  users), while two replicas measured 1.30x at 8 users, 1.75x at 16 and")
        print("  1.91x at 32. Below 8 concurrent users one Spark is as good as two.")
        print("")
        print("  1. This Spark:")
        print(f"     {shlex.quote(str(bin_dir))}/llama-server -m {qmodel} \\")
        print(f"         -ngl 999 --ctx-size {ctx * slots} -np {slots} -cb -ub 512 \\")
        print(f"         --host 0.0.0.0 --port {local_port}")
        print("")
        print(f"  2. The peer ({peer_ip}) -- the model must exist there; copy it over the")
        print("     ConnectX link rather than downloading (444 MB/s vs ~20 KB/s internet):")
        print(f"     rsync -a <model.gguf> {peer_ip}:<path>")
        # Double quotes inside the outer single quotes ssh is given: a custom
        # UNSLOTH_STUDIO_HOME with a space made the peer shell split the executable path,
        # and shlex.quote here would end that outer quoting instead.
        peer_server_cmd = f'"{peer_bin_dir}/llama-server"'
        print(f"     ssh {peer_ip} '{peer_server_cmd} -m <path> \\")
        print(f"         -ngl 999 --ctx-size {ctx * slots} -np {slots} -cb -ub 512 \\")
        print(f"         --host 0.0.0.0 --port {peer_port}'")
        print("")
        print("  3. Round-robin front end:")
        print(
            f"     python -m studio.spark_lb --port {port} "
            f"127.0.0.1:{local_port} {peer_ip}:{peer_port}"
        )
        print("")
        print(f"  Clients talk to port {port}. Nothing crosses the wire during decode.")
        return 0

    # A split, and only now is the RPC server a requirement.
    plan = rpc_cluster_plan(rpc_port)
    if not plan["ok"]:
        for problem in plan["problems"]:
            print(f"  cannot layer-split across both Sparks: {problem}")
        return 1
    binary = plan["rpc_server"]
    bin_dir = Path(binary).parent
    peer_bin_dir = _peer_relative_path(bin_dir)
    if engines > 1:
        # Two split engines need two full copies of the weights, so obeying --engines 2 here
        # would OOM mid-load.
        print(f"  {model} is {size:.1f} GiB -- two copies do not fit across the pair.")
        print("  Forcing --engines 1 (layer split). This buys capacity, not speed.")
        print("")
        engines = 1

    # Both nodes must speak the same RPC protocol, and the build pins it; check before
    # printing a launch that would fail at load.
    preflight = rpc_protocol_preflight(peer_ip, rpc_port)
    for note in preflight["notes"]:
        print(f"  note: {note}")
    if preflight["problems"]:
        for problem in preflight["problems"]:
            print(f"  RPC PROTOCOL: {problem}")
        print("")
        print("  Not printing a launch that would fail at load. Fix the above and re-run.")
        return 1
    print("")

    # The peer's own answer, not ours mapped onto it. A source build lives at a path the
    # peer was never given -- provisioning copies the managed bundle and nothing else -- so
    # deriving the remote command from the local layout emitted an ssh line for a file that
    # is not there, after a preflight that had just reported everything fine.
    peer_server = (preflight.get("peer") or {}).get("rpc_server")
    if peer_server:
        # Already absolute in the PEER's filesystem, whose home may not be ours.
        peer_command = peer_server
    else:
        peer_command = f"{peer_bin_dir}/{Path(binary).name}"
        print("  note: the peer's bundle could not be probed, so the command below assumes it")
        print("        keeps its ggml-rpc-server where this Spark does.")
        print("")

    print(f"  model   : {model}")
    print(f"  engines : {engines} (each layer-split across both Sparks)")
    print(f"  peer    : {peer_ip}")
    print("")
    print(f"  1. Start {engines} rpc-server(s) on the peer, one per engine:")
    for i in range(engines):
        # Bound to the rail, not 0.0.0.0. ggml-rpc-server has no authentication and executes
        # graphs on whoever connects, and llama-server reaches it only over `peer_ip`, so
        # every other interface the peer has -- LAN, Wi-Fi -- was exposure with no purpose.
        # Double quotes, not shlex.quote: the whole remote command already sits inside the
        # single quotes ssh is given, where a single-quoted word would end that quoting.
        quoted_server = '"' + peer_command.replace("\\", "\\\\").replace('"', '\\"') + '"'
        print(f"     ssh {peer_ip} '{quoted_server} -H {peer_ip} -p {rpc_port + i} -c'")
    print("")
    print(f"  2. Start {engines} llama-server(s) on this Spark:")
    # Resolved on its own: an rpc-server found in a source tree says nothing about where
    # llama-server is, and the two need not share a directory.
    local_server = llama_server_binary() or f"{bin_dir}/llama-server"
    for i in range(engines):
        print(f"     {shlex.quote(local_server)} -m {qmodel} \\")
        print(f"         --rpc {peer_ip}:{rpc_port + i} -ngl 999 --ctx-size {ctx * slots} \\")
        print(f"         -np {slots} -cb -ub 512 --host 127.0.0.1 --port {port + 1 + i}")
    print("")
    if engines > 1:
        ends = " ".join(f"127.0.0.1:{port + 1 + i}" for i in range(engines))
        print("  3. Put the round-robin front end in front of them:")
        print(f"     python -m studio.spark_lb --port {port} {ends}")
        print("")
        print(f"  Clients then talk to a single endpoint on port {port}.")
    print("")
    print("  Notes that change the numbers materially:")
    print("    * `-ub 512` -- the ubatch optimum INVERTS under a split (512 beats 1024).")
    print("    * CUDA_SCALE_LAUNCH_QUEUES=4x -- noise on one node, +4.2% when split.")
    print("    * Do NOT set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1: measured -22% decode on")
    print("      GB10. It swaps cudaMalloc for cudaMallocManaged, and the memory-sizing")
    print("      benefit it is usually wanted for is already active via prop.integrated.")
    print("    * A model that FITS on one Spark is faster served by two independent")
    print("      single-node servers than by any cross-node split.")
    print("    * Restart engines between benchmark runs: llama-server at -np 32 degrades")
    print("      2.6x across successive load bursts against the same instance.")
    return 0


# device_map="balanced" does NOT work here: it splits layers across GPUs inside one process,
# and each Spark is a separate host with a single GB10. DDP over torchrun does work and buys
# throughput, not capacity (the model must still fit on one Spark). FSDP would buy capacity
# but Unsloth does not support it (#4858). For capacity, use llama.cpp RPC for inference.


def _not_a_spark_plan(what: str) -> Dict[str, Any]:
    """The refusal every *_launch_plan returns off a Spark, shaped like a failed plan so
    callers need no special case. Guarded here, not only in the CLI, because these are
    importable and the installer calls them directly rather than through `main()`."""
    return {
        "ok": False,
        "problems": [f"not a DGX Spark, so there is no peer to {what}"],
        "env": {},
        "node0": None,
        "node1": None,
        "peer_ip": None,
        "local_ip": None,
    }


def train_launch_plan(script: str, port: int = 29500) -> Dict[str, Any]:
    if not is_dgx_spark():
        return _not_a_spark_plan("train against")
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    if not peer or not local:
        return {"ok": False, "problems": ["no configured peer rail (run `unsloth spark setup`)"]}
    base = f"torchrun --nnodes=2 --nproc_per_node=1 --master_addr={local} " f"--master_port={port}"
    # Quoted, as the pipeline launcher next door already does: these are printed to be pasted,
    # and a script path with a space split into several arguments.
    qscript = shlex.quote(script)
    return {
        "ok": True,
        "problems": [],
        "env": nccl_env(),
        "node0": f"{base} --node_rank=0 {qscript}",
        "node1": f"{base} --node_rank=1 {qscript}",
        "peer_ip": peer,
        "local_ip": local,
    }


def pipeline_launch_plan(
    model: str,
    port: int = 29500,
    *,
    extra: str = "",
) -> Dict[str, Any]:
    """torchrun commands for a layer-split (pipeline-parallel) run. Unlike `train_launch_plan`
    (DDP, which replicates and needs the model to fit on ONE Spark) this splits the decoder
    stack, and is the only way to train a model larger than a single Spark."""
    if not is_dgx_spark():
        return _not_a_spark_plan("split a model across")
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    if not peer or not local:
        return {"ok": False, "problems": ["no configured peer rail (run `unsloth spark setup`)"]}
    base = f"torchrun --nnodes=2 --nproc_per_node=1 --master_addr={local} " f"--master_port={port}"
    # Quoted: both commands are printed for a shell and `--run` feeds them to one, so a local
    # checkpoint path with a space would otherwise split into several arguments.
    target = f"-m studio.spark_pipeline --model {shlex.quote(model)}"
    if extra:
        target = f"{target} {extra}"
    return {
        "ok": True,
        "problems": [],
        "env": nccl_env(),
        "node0": f"{base} --node_rank=0 {target}",
        "node1": f"{base} --node_rank=1 {target}",
        "peer_ip": peer,
        "local_ip": local,
    }


def managed_torchrun() -> str:
    """The managed venv's `torchrun`, as one shell word, or the bare name if it is not there.

    The installer puts only `~/.local/bin/unsloth` on PATH -- a symlink to a console script,
    which does not add the venv's `bin` to anything -- so `torchrun` is not on the PATH of an
    ordinary post-install shell, nor of a non-interactive ssh login on the peer. Anything that
    runs or PRINTS a torchrun command has to resolve it or say `. activate` first."""
    torchrun = _studio_root() / "unsloth_studio" / "bin" / "torchrun"
    return shlex.quote(str(torchrun)) if torchrun.exists() else "torchrun"


def _local_launch(command: str) -> str:
    """Run rank 0 out of the managed venv; the peer command sources `activate` for the same
    reason."""
    trun = managed_torchrun()
    if trun != "torchrun" and command.startswith("torchrun "):
        return f"{trun} {command[len('torchrun '):]}"
    return command


_PEER_STAGE_PID = "/tmp/unsloth_pp_stage1.pid"
# The detached rank's exit status. "The process is gone" is not "the process succeeded": rank 1
# can fail after the last barrier, `save_pretrained` on the peer being the obvious way, and
# that looked exactly like a clean finish.
_PEER_STAGE_RC = "/tmp/unsloth_pp_stage1.rc"
_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no")


def peer_home(peer_ip: str, user: str) -> Optional[str]:
    """The peer's home directory, absolute. Asked once and reused: the peer's account need
    not be ours, so nothing here may assume the two homes have the same path."""
    try:
        out = subprocess.run(
            ["ssh", "-n", *_SSH_OPTS, f"{user}@{peer_ip}", 'printf %s "$HOME"'],
            capture_output = True,
            text = True,
            timeout = 30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    home = (out.stdout or "").strip()
    return home if out.returncode == 0 and home.startswith("/") else None


def _rsync_to_peer(
    local: str,
    remote: str,
    peer_ip: str,
    user: str,
    dereference: bool = False,
) -> Optional[str]:
    """Copy `local` (file or directory) to the absolute `remote` on the peer. Returns an
    error string, or None on success.

    `dereference` sends what the links point AT. Archive mode implies `--links`, which copies
    a symlink as a symlink, and a Hugging Face snapshot directory is almost entirely symlinks
    into `../../blobs` -- outside the directory being copied. Staging one without this put
    broken links on the peer and rank 1 failed loading the model."""
    parent = osp.dirname(remote.rstrip("/")) or "/"
    source = local.rstrip("/") + "/" if osp.isdir(local) else local
    destination = remote.rstrip("/") + "/" if osp.isdir(local) else remote
    cmd = [
        "rsync",
        "-a",
        *(["--copy-links"] if dereference else []),
        "--rsync-path",
        f"mkdir -p {shlex.quote(parent)} && rsync",
        "-e",
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=no",
        source,
        f"{user}@{peer_ip}:{destination}",
    ]
    try:
        res = subprocess.run(cmd, capture_output = True, text = True, timeout = 7200)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)[:200]
    return None if res.returncode == 0 else (res.stderr or "").strip()[:200]


def stage_run_inputs(
    command: str, peer_ip: str, user: str, home: str
) -> Tuple[str, List[str], List[str]]:
    """Put rank 1's inputs on the peer and point its command at them.

    Rank 1 runs over SSH from its own `$HOME`, and provisioning copies the venv, the
    llama.cpp bundle and the kernel caches -- not the dataset and not the checkpoint. The
    same `--data` and `--model` values were handed to both ranks, so unless the user had
    independently created them at the identical peer path rank 1 died in `open(args.data)`
    or in the tokenizer load while rank 0 sat in the rendezvous. Returns the rewritten
    command, what was staged, and what failed."""
    staged: List[str] = []
    failed: List[str] = []
    root = f"{home.rstrip('/')}/.unsloth/spark_inputs"
    tokens = shlex.split(command)
    for i, token in enumerate(tokens[:-1]):
        if token not in ("--data", "--model"):
            continue
        value = tokens[i + 1]
        local = osp.expanduser(value)
        if osp.exists(local):
            remote = f"{root}/{osp.basename(local.rstrip('/'))}"
            # Dereferenced: a snapshot directory is symlinks into ../../blobs.
            error = _rsync_to_peer(local, remote, peer_ip, user, dereference = True)
            if error:
                failed.append(f"{token} {value}: {error}")
            else:
                tokens[i + 1] = remote
                staged.append(f"{token} -> {remote}")
            continue
        if token != "--model":
            continue
        # A repo id: copy the cache entry rather than making the peer download it again at
        # internet speed over a link that moves 444 MB/s.
        cached = osp.expanduser(
            osp.join("~/.cache/huggingface/hub", "models--" + value.replace("/", "--"))
        )
        if not osp.isdir(cached):
            continue
        remote = f"{home.rstrip('/')}/.cache/huggingface/hub/{osp.basename(cached)}"
        # The cache entry is copied whole, blobs included, so its links stay internal and
        # valid on the peer; dereferencing here would duplicate every blob.
        error = _rsync_to_peer(cached, remote, peer_ip, user)
        if error:
            failed.append(f"{value} (HF cache): {error}")
        else:
            staged.append(f"{value} -> {remote}")
    return shlex.join(tokens), staged, failed


def wait_for_peer_stage(
    peer_ip: str,
    user: str,
    pid_file: str,
    timeout: int = 3600,
    rc_file: str = _PEER_STAGE_RC,
) -> Dict[str, Any]:
    """Block until the detached peer rank has exited, or the timeout.

    The local `torchrun` returning says only that RANK 0 finished serialising. Rank 1 is
    detached and `spark_pipeline` has no barrier after `save_pretrained`, so collecting on
    rank 0's exit could rsync a stage that was still being written and report success on a
    truncated checkpoint."""
    # Exit 0 only when the recorded status is 0. Waiting for the pid to disappear and calling
    # that success accepted a rank that died after the final barrier, and if the peer's save
    # directory still held a `stage1` from an earlier run the rsync then reported a complete
    # checkpoint assembled from a new local stage and stale peer weights. 2 means "gone but
    # failed", 3 "gone and never said", 1 "still running".
    command = (
        f"p=$(cat {pid_file} 2>/dev/null); "
        f'if [ -n "$p" ]; then for i in $(seq 1 {max(1, timeout)}); do '
        f'kill -0 "$p" 2>/dev/null || break; sleep 1; done; '
        f'kill -0 "$p" 2>/dev/null && exit 1; fi; '
        # A short grace period for the status file rather than reading it once. The recorded
        # pid IS the shell that writes it -- `setsid` execs rather than forks from a
        # non-interactive ssh shell, which was checked on this pair (`ps` shows the recorded
        # pid running the inner `bash -c`) -- but on a host where it did fork, reading once
        # would call every successful run a failure. Waiting a few seconds cannot turn a real
        # failure into a pass, since the file is only written with the true status.
        f"for i in $(seq 1 30); do "
        f"[ -f {rc_file} ] && break; sleep 1; done; "
        f"rc=$(cat {rc_file} 2>/dev/null); "
        f'if [ -z "$rc" ]; then exit 3; fi; '
        f'[ "$rc" = "0" ] || exit 2; exit 0'
    )
    try:
        res = subprocess.run(
            ["ssh", "-n", *_SSH_OPTS, f"{user}@{peer_ip}", command],
            capture_output = True,
            timeout = timeout + 60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "why": f"could not ask the peer whether its rank finished: {exc}"}
    why = {
        0: "",
        1: "the peer rank has not exited",
        2: "the peer rank exited NONZERO; its stage is incomplete or was never written",
        3: "the peer rank is gone but recorded no exit status, so it did not finish normally",
    }.get(res.returncode, f"the peer status check exited {res.returncode}")
    return {"ok": res.returncode == 0, "why": why}


def collect_stage_outputs(
    save_dir: str,
    peer_ip: str,
    user: str,
    home: str,
    ranks: Optional[List[int]] = None,
) -> Optional[str]:
    """Bring the PEER's own `stageN/` directories back into the local save directory.

    `spark merge` reads every stage from ONE local directory and says it needs no second
    Spark, while rank 1 wrote its stage on the peer and nothing fetched it. A successful run
    therefore left no mergeable checkpoint on either machine.

    Only the peer's own stages, named explicitly. Copying the whole remote root would bring
    back a stale `stage0` -- from an earlier run with the roles reversed, say -- over the
    freshly trained local one, and the metadata and adapter config can still agree, so the
    merge would accept a checkpoint combining two different runs."""
    local = osp.expanduser(save_dir)
    remote = save_dir if osp.isabs(save_dir) else f"{home.rstrip('/')}/{save_dir}"
    os.makedirs(local, exist_ok = True)
    wanted = [f"stage{rank}" for rank in (ranks if ranks is not None else [1])]
    cmd = [
        "rsync",
        "-a",
        "-e",
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=no",
        *[f"{user}@{peer_ip}:{remote.rstrip('/')}/{name}" for name in wanted],
        local.rstrip("/") + "/",
    ]
    try:
        res = subprocess.run(cmd, capture_output = True, text = True, timeout = 7200)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)[:200]
    return None if res.returncode == 0 else (res.stderr or "").strip()[:200]


def _is_data_parallel(command: str) -> bool:
    """Whether a launch command is a data-parallel run rather than a pipeline one."""
    return "--data-parallel" in shlex.split(command)


def _save_dir_of(command: str) -> str:
    tokens = shlex.split(command)
    for i, token in enumerate(tokens[:-1]):
        if token == "--save":
            return tokens[i + 1]
    return ""


def run_pipeline(plan: Dict[str, Any], log_peer: str = "/tmp/unsloth_pp_stage1.log") -> int:
    """Actually launch a layer-split run on both Sparks. Without `ssh -f` the launcher is
    held open and the head rank never starts; without the peer log its errors are lost, since
    the head only ever reports `DistStoreError: 1/2 clients joined`."""
    user = _ssh_user()
    ssh_opts = list(_SSH_OPTS)
    activate = venv_activate_sh()
    env = "; ".join(f"export {k}={v}" for k, v in plan["env"].items())

    # Rank 1's inputs, before it is started. It runs from its own $HOME and provisioning
    # copies the venv, the bundle and the kernel caches, so the dataset and the checkpoint
    # are not there under any name unless they are put there.
    node1 = plan["node1"]
    home = peer_home(plan["peer_ip"], user)
    if home is None:
        print("  could not read the peer's home directory over ssh; not launching")
        return 1
    node1, staged, failed = stage_run_inputs(node1, plan["peer_ip"], user, home)
    for note in staged:
        print(f"  staged  {note}")
    if failed:
        for note in failed:
            print(f"  FAILED to stage {note}")
        print("  Not launching: rank 1 would fail on a missing input while rank 0 waits out")
        print("  the rendezvous timeout, which reports only 'DistStoreError: 1/2 clients'.")
        return 1
    # `cd $HOME`, not the local cwd: provisioning copies the venv and the caches, never the
    # project directory, so the same absolute path need not exist on the peer.
    #
    # The pid is recorded, as the NCCL probe does, so a local failure or a Ctrl-C can stop
    # rank 1. Nothing did: it was launched under `setsid nohup` and then abandoned, so it sat
    # in rendezvous or a collective holding its model and CUDA context until the distributed
    # timeout, which makes the next provisioning run refuse the peer as busy and the next
    # launch collide with a job nobody is watching. `setsid` makes it a process group leader,
    # so the negative kill takes torchrun's children with it.
    # Quoted ONCE, here, rather than wrapped in hand-written single quotes: `node1` comes back
    # from `stage_run_inputs` already `shlex.join`ed, so a staged path with a space carries its
    # own quotes, and those closed the outer `bash -c '...'` early. Rank 1 then ran a fragment
    # or nothing, and rank 0 reported only the rendezvous timeout.
    # The peer's own stage directory from an EARLIER run, removed before this one starts.
    # Left in place it is indistinguishable from what this run is about to write, and a rank
    # that dies before writing anything then leaves a `stage1` the collection would bring back
    # beside a fresh local `stage0`. Only the stage this launch owns, under the run's own
    # `--save` directory, and never the local one.
    stale = _save_dir_of(node1)
    if stale:
        remote_save = stale if osp.isabs(stale) else f"{home.rstrip('/')}/{stale}"
        target = f"{remote_save.rstrip('/')}/stage1"
        cleared = subprocess.run(
            ["ssh", "-n", *ssh_opts, f"{user}@{plan['peer_ip']}", f"rm -rf {shlex.quote(target)}"],
            capture_output = True,
            timeout = 60,
        )
        if cleared.returncode == 0:
            print(f"  cleared any earlier {target} on the peer")
        else:
            print(f"  note: could not clear {target} on the peer; a stale stage there would")
            print("        be collected as if this run had written it")
    #
    # Not `exec`: the status has to be recorded after the stage exits, and a group leader is
    # already what `setsid` gives us, so the negative kill still takes the children.
    inner = f"[ -f {activate} ] && . {activate}; {env}; {node1}; " f"echo $? > {_PEER_STAGE_RC}"
    remote = (
        f'cd "$HOME" && rm -f {_PEER_STAGE_RC} {_PEER_STAGE_PID}; '
        f"setsid nohup bash -c {shlex.quote(inner)} "
        f"> {shlex.quote(log_peer)} 2>&1 < /dev/null & "
        f"echo $! > {_PEER_STAGE_PID}"
    )
    try:
        peer = subprocess.run(
            [
                "ssh",
                "-f",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                f"{user}@{plan['peer_ip']}",
                remote,
            ],
            timeout = 60,
            check = False,
        )
    except Exception as exc:
        print(f"  could not start the peer stage: {exc}")
        return 1
    # `ssh -f` backgrounds only AFTER authentication, so auth/routing failures land here. Without
    # this rank 1 never exists and rank 0 just waits out the rendezvous timeout.
    if peer.returncode != 0:
        print(f"  could not start the peer stage: ssh exited {peer.returncode}")
        return 1
    print(f"  peer stage started; its log is {log_peer} on {plan['peer_ip']}")

    time.sleep(6)  # let the peer reach the rendezvous first
    child_env = dict(os.environ)
    child_env.update({k: str(v) for k, v in plan["env"].items()})
    try:
        rc = subprocess.run(_local_launch(plan["node0"]), shell = True, env = child_env).returncode
    except BaseException:
        # Ctrl-C included, which is the common way this ends.
        stop_peer_by_pidfile(plan["peer_ip"], user, ssh_opts, _PEER_STAGE_PID)
        raise
    if rc != 0:
        # Only on failure. A successful run leaves rank 1 writing its own stage under --save,
        # and killing it there would truncate the half of the checkpoint it owns.
        print(f"  local stage exited {rc}; stopping the peer stage")
        stop_peer_by_pidfile(plan["peer_ip"], user, ssh_opts, _PEER_STAGE_PID)
        return rc

    # Rank 1 wrote `DIR/stage1` on the PEER. `spark merge` reads every stage from one local
    # directory and says it needs no second Spark, so without this a run that trained
    # correctly still left no mergeable checkpoint on either machine.
    save_dir = _save_dir_of(plan["node0"])
    if save_dir and _is_data_parallel(plan["node0"]):
        # Data parallel has no stages. Every rank holds the whole model, rank 0 writes the
        # complete checkpoint or adapter and rank 1 deliberately writes nothing, so there is no
        # `DIR/stage1` on the peer and never will be. Collecting it anyway made every
        # SUCCESSFUL data-parallel save end in an rsync failure and a nonzero exit, after the
        # whole training run had already finished correctly -- the worst possible moment to
        # report a failure that is not one. The rank 0 save is the final output.
        print(f"  data parallel: rank 0 wrote the whole checkpoint to {save_dir}")
        print("  nothing to collect from the peer; no merge step is needed")
        return rc
    if save_dir:
        print("  waiting for the peer stage to finish writing ...")
        finished = wait_for_peer_stage(plan["peer_ip"], user, _PEER_STAGE_PID)
        if not finished["ok"]:
            print(f"  {finished['why']}.")
            print(f"  Not collecting: its stage is under {save_dir} on {plan['peer_ip']}, and")
            print("  merging a stale or half-written stage produces a checkpoint that loads")
            print("  and trains worse. Its log is above; re-run once it is fixed.")
            return 1
        print(f"  collecting the peer's stages into {save_dir} ...")
        error = collect_stage_outputs(save_dir, plan["peer_ip"], user, home)
        if error:
            print(f"  FAILED to collect the peer's stages: {error}")
            print(f"  They are under {save_dir} on {plan['peer_ip']}; copy them before merging.")
            return 1
        print(f"  all stages are now in {save_dir}; merge with `unsloth spark merge`")
    return rc


def _cmd_pipeline(
    model: str,
    port: int = 29500,
    extra: str = "",
    run: bool = False,
) -> int:
    if not is_dgx_spark():
        print(NOT_A_SPARK)
        return 0
    plan = pipeline_launch_plan(model, port, extra = extra)
    if not plan["ok"]:
        for problem in plan["problems"]:
            print(f"  cannot launch: {problem}")
        return 1

    data_parallel = "--data-parallel" in extra.split()
    # A model that fits on one Spark trains faster there than split across two.
    size = model_size_gib(model)
    if data_parallel and size is not None:
        budget = SPARK_USABLE_GIB - SERVE_OVERHEAD_GIB
        if size > budget:
            print(
                f"  NOTE: {model} is {size:.1f} GiB and does NOT fit on one Spark "
                f"({budget:.0f} GiB budget); a data-parallel replica holds the whole model."
            )
            print("        Use --layer-split with --shard-load for it instead.")
            print("")
            if run:
                # A note is not enough once --run is on the command line. Every rank builds the
                # COMPLETE model before DDP or FSDP wraps it, so this cannot satisfy the
                # documented requirement that the model fit one Spark, and --fsdp does not
                # rescue it: sharding happens after construction, not during. Continuing spends
                # the whole load on both nodes to arrive at an out-of-memory that is already
                # known here.
                print("  Not launching: --data-parallel needs the model to fit one Spark.")
                return 1
    elif size is not None:
        budget = SPARK_USABLE_GIB - SERVE_OVERHEAD_GIB
        if size <= budget:
            print(
                f"  NOTE: {model} is {size:.1f} GiB and fits on ONE Spark "
                f"({budget:.0f} GiB budget)."
            )
            print("        A layer split buys capacity, not speed. Consider")
            print("        `--data-parallel` for throughput instead.")
            print("")

    if run:
        return run_pipeline(plan)
    if data_parallel:
        print("  Two-Spark data-parallel training (throughput, not capacity -- one whole")
        print("  model per Spark; the LoRA gradients are averaged across the pair).")
    else:
        print("  Two-Spark layer-split training (capacity, not throughput -- this is how a")
        print("  model too large for one Spark gets trained; add --shard-load for those).")
    print("")
    _print_launch(plan)
    return 0


def _print_launch(plan: Dict[str, Any]) -> None:
    """The two commands, with the one line that makes them runnable.

    They start with a bare `torchrun`, which is in the managed venv and not on the PATH of an
    ordinary post-install shell (the installer exposes only the `unsloth` shim), so a user who
    followed these exactly got `torchrun: command not found` on both nodes. Activation is
    printed with them rather than assumed."""
    activate = venv_activate_sh()
    print("  Export on BOTH nodes, and activate the managed environment:")
    for key, value in plan["env"].items():
        print(f"    export {key}={value}")
    print(f"    . {activate}")
    print("")
    print(f"  On this Spark ({plan['local_ip']}):")
    print(f"    {plan['node0']}")
    print(f"  On the peer ({plan['peer_ip']}):")
    print(f"    {plan['node1']}")
    print("")


def _cmd_train(script: str, port: int = 29500) -> int:
    if not is_dgx_spark():
        print(NOT_A_SPARK)
        return 0
    plan = train_launch_plan(script, port)
    if not plan["ok"]:
        for problem in plan["problems"]:
            print(f"  cannot launch: {problem}")
        return 1
    print("  Two-Spark DDP training (throughput, not capacity -- the model must")
    print("  still fit on ONE Spark; use `unsloth spark serve` to split a model).")
    print("")
    _print_launch(plan)
    print("  Both nodes must have the SAME Unsloth/torch versions and the same")
    print("  optional kernels installed, or the ranks disagree on which path to run.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog = "unsloth spark", description = __doc__)
    parser.add_argument(
        "command",
        nargs = "?",
        default = "status",
        choices = (
            "status",
            "setup",
            "env",
            "detect",
            "serve",
            "train",
            "doctor",
            "provision",
            "plan",
            "kernels",
            "estimate",
            "peers",
        ),
    )
    parser.add_argument("--model", default = "", help = "GGUF path for `serve`")
    parser.add_argument("--script", default = "", help = "training script for `train`")
    parser.add_argument("--port", type = int, default = 8080)
    # The public `unsloth spark serve` has always exposed --rpc-port, and the preflight
    # names it when the default port is taken; without it registered here, passing it
    # died in argparse before reaching `_cmd_serve`.
    parser.add_argument(
        "--rpc-port",
        type = int,
        default = RPC_DEFAULT_PORT,
        help = "base port for the peer's RPC servers (`serve`)",
    )
    parser.add_argument("--ctx", type = int, default = 8192)
    parser.add_argument("--yes", "-y", action = "store_true")
    parser.add_argument(
        "--run",
        action = "store_true",
        help = "launch the layer-split run on both Sparks, not just print it",
    )
    parser.add_argument("--batch", type = int, default = 8)
    parser.add_argument("--microbatches", type = int, default = 4)
    parser.add_argument("--seq", type = int, default = 512)
    parser.add_argument("--full-finetune", action = "store_true")
    parser.add_argument("--grad-checkpoint", action = "store_true")
    parser.add_argument(
        "--workload",
        default = "mixed",
        choices = ("decode", "prefill", "mixed"),
        help = "which regime to optimise the kernel choice for",
    )
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help = "show exactly what would be written or copied, and do none "
        "of it (works for `setup` and `provision`)",
    )
    parser.add_argument(
        "--rsync-delete",
        action = "store_true",
        help = "also delete files on the peer that are absent here. OFF by "
        "default: a stale extra file is far cheaper than deleting one "
        "a running job is executing.",
    )
    parser.add_argument(
        "--force",
        action = "store_true",
        help = "provision even though the peer GPU looks busy (or could not "
        "be checked). You are asserting no job is running there.",
    )
    parser.add_argument(
        "--no-fast",
        action = "store_true",
        help = "copy over ssh only. By default `provision` moves bulk bytes through an "
        "ephemeral rsync daemon on the peer, UNENCRYPTED over the direct rail cable, "
        "which is a point-to-point link between the two Sparks with no other host on "
        "it. The daemon is bound to the peer's rail address, accepts only this node's "
        "rail address with a one-shot random secret, and is stopped when the command "
        "ends. ssh then finalises every path. Measured: ssh ~0.25 GB/s, fast path at "
        f"the disk floor. Same as {FAST_ENV}=0.",
    )
    parser.add_argument(
        "--engines",
        type = int,
        default = 2,
        help = "independent engines to run; >1 is what beats a single Spark",
    )
    parser.add_argument("--slots", type = int, default = 16, help = "server slots per engine")
    parser.add_argument(
        "--layer-split",
        default = "",
        help = "model to train split across the Sparks (capacity, not speed)",
    )
    parser.add_argument(
        "--pipeline-args", default = "", help = "extra flags passed through to studio.spark_pipeline"
    )
    parser.add_argument("--master-port", type = int, default = 29500)
    parser.add_argument(
        "--benchmark",
        action = "store_true",
        help = "measure the link with ib_write_bw (needs perftest on both nodes)",
    )
    parser.add_argument(
        "--intent",
        default = "throughput",
        choices = INTENTS,
        help = "what you want from the cluster; it decides the axis, "
        "not just whether the model fits",
    )
    parser.add_argument(
        "--nodes", type = int, default = None, help = "how many Sparks to plan for (default: discovered)"
    )
    parser.add_argument(
        "--concurrency",
        type = int,
        default = 1,
        help = "requests in flight, for the expected-speedup number",
    )
    parser.add_argument(
        "--prompt-tokens",
        type = int,
        default = 512,
        help = "typical prompt length, for the replicas/layer-split decision in `plan`",
    )
    parser.add_argument(
        "--prefill-heavy",
        action = "store_true",
        help = "the work is prefill-heavy long-prompt work (RAG, documents); the only "
        "case where `plan` will layer-split a model that fits",
    )
    parser.add_argument(
        "--switched",
        action = "store_true",
        help = "all Sparks share a switched RoCE fabric; required to plan "
        "addressing for more than two",
    )
    parser.add_argument("--no-probe", action = "store_true", help = "do not TCP-probe peers in `peers`")
    args = parser.parse_args(argv)

    if args.command == "detect":
        # Machine-readable, for install.sh: exit 0 only when a peer is cabled.
        print(json.dumps({"is_spark": is_dgx_spark(), "state": cluster_state()}))
        return 0 if cluster_state() in ("unconfigured", "configured") else 1
    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "estimate":
        if not args.model:
            print("estimate needs --model")
            return 2
        return _cmd_estimate(
            args.model,
            args.batch,
            args.microbatches,
            args.seq,
            args.full_finetune,
            args.grad_checkpoint,
        )
    if args.command == "kernels":
        return _cmd_kernels(args.workload)
    if args.command == "plan":
        if not args.model:
            print("plan needs --model <path-or-repo-id>")
            return 2
        return _cmd_plan(
            args.model,
            intent = args.intent,
            nodes = args.nodes,
            concurrency = args.concurrency,
            prompt_tokens = args.prompt_tokens,
            prefill_heavy = args.prefill_heavy,
        )
    if args.command == "peers":
        return _cmd_peers(check = not args.no_probe)
    if args.command == "provision":
        return _cmd_provision(
            dry_run = args.dry_run,
            delete = args.rsync_delete,
            force = args.force,
            no_fast = args.no_fast,
        )
    if args.command == "env":
        return _cmd_env()
    if args.command == "serve":
        if not args.model:
            print("serve needs --model <path-to.gguf>")
            return 2
        return _cmd_serve(
            args.model,
            port = args.port,
            rpc_port = args.rpc_port,
            ctx = args.ctx,
            engines = args.engines,
            slots = args.slots,
        )
    if args.command == "train":
        if args.layer_split:
            # argparse records --grad-checkpoint, and forwarding only --pipeline-args meant
            # the stage process never saw it: the flag parsed, printed nothing, and did
            # nothing. Appended rather than assumed, so an explicit one in --pipeline-args
            # is not duplicated.
            extra = args.pipeline_args or ""
            if args.grad_checkpoint and "--grad-checkpoint" not in extra:
                extra = f"{extra} --grad-checkpoint".strip()
            return _cmd_pipeline(args.layer_split, port = args.master_port, extra = extra, run = args.run)
        if not args.script:
            print("train needs --script <train.py>, or --layer-split <model>")
            return 2
        return _cmd_train(args.script, port = args.master_port)
    if args.command == "setup":
        return _cmd_setup(
            assume_yes = args.yes,
            n_nodes = args.nodes if args.nodes else 2,
            switched = args.switched,
            dry_run = args.dry_run,
        )
    return _cmd_status(benchmark = args.benchmark)


if __name__ == "__main__":
    raise SystemExit(main())
