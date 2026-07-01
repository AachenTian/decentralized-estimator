#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# 先装项目通用依赖
python -m pip install -r requirements.txt

# 再装固定版本的 GPU JAX。
# CUDA 13 或 CUDA 12 由 H100 节点实际 driver 决定。
python -m pip install --upgrade "jax[cuda13]==0.6.2"

# 最后忽略 JaxMARL 的旧 JAX upper-bound metadata，
# 保留你本地已经验证过的 JAX 0.6.2。
python -m pip install --no-deps "jaxmarl==0.1.0"