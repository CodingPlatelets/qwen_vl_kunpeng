### for vllm self compiled for 2 kunpeng aarch64 cpus and four a100 gpus

```SHELL
source /root/data/vllm/.venv/bin/activate && \\ncd /root/data/vllm && \\nrm -rf build/ dist/ *.egg-info && \\nexport CUDA_HOME=/usr/local/cuda-12.6 && \\nexport TORCH_CUDA_ARCH_LIST="8.0" && \\nexport VLLM_TARGET_DEVICE="cuda" && \\nexport MAX_JOBS=64 && \\npip wheel . --no-deps --no-build-isolation --wheel-dir dist/ 2>&1 | tee build_wheel.log
```
