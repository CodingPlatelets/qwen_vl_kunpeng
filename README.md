### for vllm self compiled for 2 kunpeng aarch64 cpus and four a100 gpus

```SHELL
source /root/data/vllm/.venv/bin/activate && \\
cd /root/data/vllm && \\
rm -rf build/ dist/ *.egg-info && \\
export CUDA_HOME=/usr/local/cuda-12.6 && \\
export TORCH_CUDA_ARCH_LIST="8.0" && \\
export VLLM_TARGET_DEVICE="cuda" && \\
export MAX_JOBS=64 && \\
pip wheel . --no-deps --no-build-isolation --wheel-dir dist/ 2>&1 | tee build_wheel.log
```

