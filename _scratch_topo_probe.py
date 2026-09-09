"""Scratch: adversarial inputs against _probe_nvlink_topology. Not a test file."""
import sys, types, subprocess
sys.path.insert(0, "studio/backend")
from core.inference.llama_cpp import LlamaCppBackend as B

def parse(text, rc=0):
    real = subprocess.run
    subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=rc, stdout=text, stderr="")
    try:
        return B._probe_nvlink_topology()
    finally:
        subprocess.run = real

def show(name, text, rc=0):
    m = parse(text, rc)
    if m is None:
        print(f"{name}: None")
        return
    ids = sorted({i for p in m for i in p})
    labels = sorted({v for v in m.values()})
    print(f"{name}: {len(m)} pairs, gpus={ids}, labels={labels}")

# 1. Single GPU box
show("single_gpu", (
    "\t\x1b[4mGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \t0-23\t0\t\tN/A\n"
    "\nLegend:\n\n  X    = Self\n"
))

# 2. 16-GPU box: GPU10..GPU15 must not collide with GPU1
hdr = "\t\x1b[4m" + "\t".join(f"GPU{i}" for i in range(16)) + "\tCPU Affinity\x1b[0m\n"
rows = ""
for i in range(16):
    cells = ["  X  " if j == i else "NV18" for j in range(16)]
    rows += f"GPU{i}\t" + "\t".join(cells) + "\t0-95\n"
show("16gpu_all_nvlink", hdr + rows + "\nLegend:\n")

# 3. NIC row BEFORE the GPU rows
show("nic_row_first", (
    "\t\x1b[4mGPU0\tGPU1\tNIC0\tCPU Affinity\x1b[0m\n"
    "NIC0\tSYS\tSYS\t X \t\n"
    "GPU0\t X \tNV18\tSYS\t0-23\n"
    "GPU1\tNV18\t X \tSYS\t0-23\n"
    "\nLegend:\n"
))

# 4. Partial NVLink: two bridged islands (4x A100 PCIe with bridges)
show("two_islands", (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\x1b[0m\n"
    "GPU0\t X \tNV12\tPHB\tPHB\t0-23\n"
    "GPU1\tNV12\t X \tPHB\tPHB\t0-23\n"
    "GPU2\tPHB\tPHB\t X \tNV12\t24-47\n"
    "GPU3\tPHB\tPHB\tNV12\t X \t24-47\n"
    "\nLegend:\n"
))

# 5. No ANSI escapes at all (piped through a filter), no trailing NUMA columns
show("plain_no_ansi", (
    "        GPU0    GPU1    CPU Affinity\n"
    "GPU0     X      NV12    0-23\n"
    "GPU1    NV12     X      0-23\n"
))

# 6. Non-zero exit
show("rc_nonzero", "whatever", rc=9)

# 7. Truncated row (fewer cells than columns)
show("truncated_row", (
    "\tGPU0\tGPU1\tGPU2\tCPU Affinity\n"
    "GPU0\t X \tNV12\n"
    "GPU1\tNV12\t X \tNV12\t0-23\n"
    "GPU2\tNV12\tNV12\t X \t0-23\n"
))

# 8. Missing GPU row entirely
show("missing_row", (
    "\tGPU0\tGPU1\tCPU Affinity\n"
    "GPU0\t X \tNV12\t0-23\n"
))

# 9. MIG-style extra indented rows after the GPU rows
show("mig_rows", (
    "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\x1b[0m\n"
    "GPU0\t X \tNV12\t0-23\n"
    "GPU1\tNV12\t X \t0-23\n"
    "\n"
    "MIG-GPU-abc/1/0\n"
    "\nLegend:\n"
))

# 10. Empty stdout
show("empty", "")

# 11. Legend absent, legend body follows directly
show("no_legend_word", (
    "\tGPU0\tGPU1\tCPU Affinity\n"
    "GPU0\t X \tNV12\t0-23\n"
    "GPU1\tNV12\t X \t0-23\n"
    "\n"
    "  X    = Self\n"
    "  NV#  = Connection traversing a bonded set of # NVLinks\n"
))

# 12. Lowercase nv label (hypothetical locale/version)
show("lowercase_nv", (
    "\tGPU0\tGPU1\tCPU Affinity\n"
    "GPU0\t X \tnv12\t0-23\n"
    "GPU1\tnv12\t X \t0-23\n"
))
