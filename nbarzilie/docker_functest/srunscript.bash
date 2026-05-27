srun \
  -p interactive \
  --container-image=$MY/sqshs/sglang-nixl-functest.sqsh \
  --container-workdir=/workspace/sglang \
  --container-mounts=$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs \
  --time=02:00:00 \
  --pty bash
