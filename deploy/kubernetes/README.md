# Production Kubernetes Deployment for Unsloth Studio

This directory provides a production-grade Kubernetes deployment manifest for Unsloth Studio with multi-GPU acceleration, large-model memory optimizations, and auto-compaction.

## Features Included
- **GPU Acceleration**: Configured with `runtimeClassName: nvidia` and multi-GPU allocation (`nvidia.com/gpu`).
- **High-Throughput Shared Memory**: 32Gi IPC shared memory (`emptyDir` on RAM) for multi-worker PyTorch and FlashAttention.
- **P2P & DMA Optimizations**: NCCL buffer scaling (`NCCL_BUFFSIZE=8388608`) and direct NVL/SYS P2P direct engine configuration.
- **Auto-Compaction & Truncation**: Default `UNSLOTH_CONTEXT_OVERFLOW="truncate_oldest"` to prevent context ceiling crashes.
- **Persistent Storage Volumes**: Decoupled persistent volumes for workspace, studio configuration/database, and Hugging Face weights cache.
