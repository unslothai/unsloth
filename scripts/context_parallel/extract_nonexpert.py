"""Extract all non-expert tensors of a sharded safetensors repo from ModelScope.

Reads each shard's header, then fetches only the byte ranges of tensors whose
name does not contain '.experts.', and streams them into one safetensors file
with pwrite (never holds more than one tensor in memory).

usage: extract_nonexpert.py <repo_id> <out_dir>
"""
import json, os, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

repo, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
API = f'https://www.modelscope.cn/api/v1/models/{repo}/repo/files'
URL = f'https://www.modelscope.cn/models/{repo}/resolve/master/'
OUT = os.path.join(out_dir, 'model-nonexpert.safetensors')
WORKERS = 16


def rng(url, a, b):
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=60) as r:
                buf = r.read()
            if len(buf) == b - a + 1:
                return buf
        except Exception:
            pass
        time.sleep(1 + attempt)
    raise RuntimeError(f'range failed {url} {a}-{b}')


files = json.loads(urllib.request.urlopen(API, timeout=30).read().decode())['Data']['Files']
# small side files (config, tokenizer, index, ...) are copied whole
for f in files:
    p = f['Path']
    if f['Type'] != 'blob' or p.endswith('.safetensors') or p.startswith(('figures/', '.git')):
        continue
    dst = os.path.join(out_dir, p)
    if os.path.exists(dst) and os.path.getsize(dst) == f['Size']:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as o:
        o.write(rng(URL + p, 0, f['Size'] - 1) if f['Size'] else b'')
    print('[side]', p, f['Size'], flush=True)

shards = sorted(f['Path'] for f in files if f['Path'].endswith('.safetensors'))
items = []  # (shard, name, dtype, shape, abs_start, nbytes)
for s in shards:
    n = struct.unpack('<Q', rng(URL + s, 0, 7))[0]
    h = json.loads(rng(URL + s, 8, 8 + n - 1))
    for k, v in h.items():
        if k == '__metadata__' or '.experts.' in k:
            continue
        a, b = v['data_offsets']
        items.append((s, k, v['dtype'], v['shape'], 8 + n + a, b - a))
    print('[hdr]', s, flush=True)

header, off = {}, 0
for s, k, dt, shape, _, nb in items:
    header[k] = {'dtype': dt, 'shape': shape, 'data_offsets': [off, off + nb]}
    off += nb
hb = json.dumps(header, separators=(',', ':')).encode()
hb += b' ' * ((8 - len(hb) % 8) % 8)
base = 8 + len(hb)
total = off
print(f'{len(items)} tensors, {total/2**30:.2f} GiB -> {OUT}', flush=True)

fd = os.open(OUT, os.O_RDWR | os.O_CREAT)
os.ftruncate(fd, base + total)
os.pwrite(fd, struct.pack('<Q', len(hb)) + hb, 0)

done = [0]
t0 = time.time()


def work(it, dst):
    s, k, dt, shape, src, nb = it
    CH = 64 << 20
    for p in range(0, nb, CH):
        e = min(nb, p + CH)
        os.pwrite(fd, rng(URL + s, src + p, src + e - 1), dst + p)
    done[0] += nb
    return nb


offs = [header[it[1]]['data_offsets'][0] + base for it in items]
with ThreadPoolExecutor(WORKERS) as ex:
    futs = [ex.submit(work, it, o) for it, o in zip(items, offs)]
    for i, f in enumerate(futs):
        f.result()
        if i % 100 == 0:
            print(f'  {done[0]/2**30:.2f}/{total/2**30:.2f} GiB, {done[0]/2**20/(time.time()-t0):.1f} MiB/s', flush=True)
os.close(fd)
print(f'=== Finished {repo} ({total/2**30:.2f} GiB) ===', flush=True)
