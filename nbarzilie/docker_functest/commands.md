# Docker Functional Test Commands

## Start Interactive SQSH Shell

This command starts a live shell inside the SQSH container, mounts the Hugging Face
cache and logs directory, and redirects runtime/JIT caches away from the container
overlay into `/logs`.

```bash
srun -A network_research_advdev \
     -p interactive \
     -t 2:00:00 \
     --gpus-per-node=8 \
     --cpus-per-task=32 \
     --mem=0 \
     --container-image=$MY/sqshs/sglang-nixl-functest.sqsh \
     --container-workdir=/workspace/sglang \
     --container-mounts=$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs \
     --pty bash -lc '
       mkdir -p /logs/{tmp,xdg-cache,sglang-cache,triton-cache,torchinductor-cache,nv-cache,tvm-ffi-cache} && \
       export HF_HOME=/root/.cache/huggingface \
              XDG_CACHE_HOME=/logs/xdg-cache \
              SGLANG_CACHE_DIR=/logs/sglang-cache \
              TRITON_CACHE_DIR=/logs/triton-cache \
              TORCHINDUCTOR_CACHE_DIR=/logs/torchinductor-cache \
              TVM_FFI_CACHE_DIR=/logs/tvm-ffi-cache \
              CUDA_CACHE_PATH=/logs/nv-cache \
              TMPDIR=/logs/tmp \
              LOG_DIR=/logs && \
       exec bash
     '
```

## Run PD NIXL Smoke

Run this inside the interactive container shell.

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
KEEP_ALIVE=0 \
MODEL_PATH=Qwen/Qwen3-8B \
SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
DISAGG_IB_DEVICES=mlx5_0,mlx5_1 \
PREFILL_EXTRA_ARGS="--disable-cuda-graph --mem-fraction-static 0.55" \
DECODE_EXTRA_ARGS="--disable-cuda-graph --mem-fraction-static 0.55" \
run_qwen3_pd_nixl.sh
```

## Logs

```bash
tail -300 /logs/pd_prefill.log
tail -300 /logs/pd_decode.log
tail -300 /logs/pd_router.log
```
