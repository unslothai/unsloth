# Serving one GGUF across several machines

Studio can split a GGUF that does not fit one machine over any number of machines through
llama.cpp RPC: four DGX Sparks, a few Macs, a Linux box with NVIDIA or AMD GPUs, or a mix.
This machine runs `llama-server`; every other machine runs `ggml-rpc-server` and holds a share
of the layers.

Paired DGX Sparks need none of this: they are found over the QSFP cable and served as before.
The cluster path below only turns on when you configure it, and then it replaces the two-Spark
path for every load. If you switch a paired Spark over to a cluster, list its peer as `managed` (or
start its `ggml-rpc-server` yourself on another port): the two-Spark worker on the peer stops
when the first cluster load replaces the two-Spark model.

## 1. Start a worker on each machine

```bash
ggml-rpc-server -H <this machine's IP on the shared network> -p 50052
```

The binary ships in the Studio llama.cpp bundle (`~/.unsloth/llama.cpp/build/bin/`). It exposes
every accelerator on that machine (CUDA, ROCm, Metal, Vulkan). Use the same llama.cpp build on
every machine: llama.cpp refuses a worker whose RPC protocol differs, and Studio names it.

`ggml-rpc-server` has no authentication. Bind it to a private network you trust, never to a
public address.

## 2. Tell Studio about the workers

Either an environment variable when starting Studio:

```bash
UNSLOTH_CLUSTER_NODES="10.0.0.2,10.0.0.3:50052,mac-studio.local" unsloth studio
```

or `<studio home>/cluster.json` (default `~/.unsloth/studio/cluster.json`):

```json
{"nodes": [
  {"host": "10.0.0.2"},
  {"host": "10.0.0.3", "rpc_port": 50052},
  {"host": "10.0.0.4", "ssh_user": "me", "managed": true}
]}
```

A node with `"managed": true` (or every node, with `UNSLOTH_CLUSTER_MANAGED=1`) has its
`ggml-rpc-server` started by Studio over non-interactive ssh, as the two-Spark path does
(from a Linux or macOS coordinator; on Windows, start the workers yourself).
`UNSLOTH_SPARK_SERVING=0` turns both paths off.

## What happens on a load

1. Studio checks which workers answer, then asks its own `llama-server --rpc ... --list-devices`
   what each one exposes and how much memory is free there.
2. A model that fits this machine is served here, as without a cluster.
3. A larger one is split by layer: `--rpc` lists the workers, `--device` lists their devices
   with this machine's GPU last (so the logits stay local), and `--tensor-split` weights each
   device by its usable free memory. Projectors and drafters are charged to (and pinned on) this
   machine; LoRA adapters and control vectors sit beside their layers, so they are spread with
   the model.
4. A load no set of devices can hold is refused with the numbers, instead of failing mid-load.

A worker that does not answer, or a managed one that cannot be started, is left out and named
in the status; the load is planned over the rest.

`UNSLOTH_CLUSTER_TOPOLOGY=layer_split` splits even a model that would fit here (useful to test
a cluster); `=single` never splits.

The owner-only `GET /api/inference/spark/status` shows the nodes, the devices each one exposed,
the plan and the exact arguments used.

## Limits

- A split is bounded by the network. Over plain Ethernet each token crosses the wire once per
  worker, so a split is only chosen when the model does not fit this machine.
- Copies of the model on several machines (replicas) are only available for two paired Sparks.
- A worker that restarts loses its share of the model; reload the model afterwards.
- A split turns automatic speculative decoding off, since an automatically chosen drafter is
  not in the plan. Ask for a mode explicitly (or pass a drafter) to keep it; the drafter and the
  projector are pinned to this machine's GPU.
