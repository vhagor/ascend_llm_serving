#!/usr/bin/env bash
set -euo pipefail

# Edit the qwen vLLM configuration directly here.
qwen_config=$(cat <<'QWEN_CONFIG'
{
  "tensor-parallel-size": 1,
  "dtype": "bfloat16",
  "max-model-len": 10240,
  "max-num-seqs": 1,
  "max-num-batched-tokens": 8192,
  "gpu-memory-utilization": 0.9,
  "additional-config": {
    "enable_cpu_binding": false,
    "ascend_log_path": "/tmp/vllm-ascend-logs"
  }
}
QWEN_CONFIG
)

# Edit the deepseek vLLM configuration directly here.
deepseek_config=$(cat <<'DEEPSEEK_CONFIG'
{
  "tensor-parallel-size": 8,
  "data-parallel-size": 1,
  "dtype": "bfloat16",
  "max-model-len": 102400,
  "max-num-seqs": 32,
  "max-num-batched-tokens": 8192,
  "gpu-memory-utilization": 0.9,
  "quantization": "ascend",
  "enable-expert-parallel": true,
  "tokenizer-mode": "deepseek_v4",
  "block-size": 128,
  "additional-config": {
    "enable_cpu_binding": false,
    "ascend_log_path": "/tmp/vllm-ascend-logs"
  }
}
DEEPSEEK_CONFIG
)

: "${HELLOHPC_MODEL_PATH:?HELLOHPC_MODEL_PATH is set by the platform}"
: "${HELLOHPC_SERVICE_BASE_URL:?HELLOHPC_SERVICE_BASE_URL is set by the platform}"
: "${HELLOHPC_MODEL_ID:?HELLOHPC_MODEL_ID is set by the platform}"
: "${HELLOHPC_CASE_ID:?HELLOHPC_CASE_ID is set by the platform}"
: "${HELLOHPC_STAGE:?HELLOHPC_STAGE is set by the platform}"

: "${HELLOHPC_SERVICE_LOG:?HELLOHPC_SERVICE_LOG is set by the platform}"
: "${HELLOHPC_ASSIGNED_NPU_COUNT:?HELLOHPC_ASSIGNED_NPU_COUNT is set by the platform}"

case "$HELLOHPC_CASE_ID:$HELLOHPC_STAGE:$HELLOHPC_MODEL_ID" in
    qwen-performance:stage1:Qwen3-14B) profile=qwen; resolved_config="$qwen_config" ;;
    dpsk-stage2:stage2:DeepSeek-V4-Flash-0731-w8a8) profile=deepseek; resolved_config="$deepseek_config" ;;
    *) printf '%s\n' "Unsupported platform task, stage, or model combination." >&2; exit 2 ;;
esac

# Vendor environment scripts may reference unset variables.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# ABI detection only needs CPU PyTorch; avoid importing NPU plugins before mapping.
torch_cxx_abi=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
source /usr/local/Ascend/nnal/atb/set_env.sh "--cxx_abi=$torch_cxx_abi"
set -u

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
# Prefer local temporary storage; use the platform's temporary directory if unavailable.
if ! runtime_dir=$(mktemp -d /var/tmp/inference-runtime.XXXXXX); then
    runtime_dir=$(mktemp -d "${TMPDIR:-/tmp}/inference-runtime.XXXXXX")
fi
export TMPDIR="$runtime_dir"
export VLLM_NO_USAGE_STATS=1
export VLLM_CONFIG_ROOT="$runtime_dir/vllm-config"
export XDG_CONFIG_HOME="$runtime_dir/config"
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm-cache"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-cache"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-cache"
export HF_HOME="${TMPDIR:-/tmp}/huggingface"
export TORCH_HOME="${TMPDIR:-/tmp}/torch-cache"
export ASCEND_WORK_PATH="${TMPDIR:-/tmp}/ascend-work"
export ASCEND_CACHE_PATH="$runtime_dir/ascend-cache"
export TEST_DATA_ROOT_PATH="$runtime_dir/tvm-test-data"
mkdir -p "$XDG_CONFIG_HOME" "$VLLM_CONFIG_ROOT" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$ASCEND_WORK_PATH" "$ASCEND_CACHE_PATH"

# Preserve runtime visibility, including an explicitly empty device list.
# Only translate physical IDs when the platform supplies no runtime visibility.
if [[ ! -v ASCEND_RT_VISIBLE_DEVICES && -n ${ASCEND_VISIBLE_DEVICES:-} ]]; then
    logical_devices=$(python3 - <<'PY'
import ctypes
import os
import sys

physical_ids = [int(value) for value in os.environ["ASCEND_VISIBLE_DEVICES"].split(",")]
if len(physical_ids) != int(os.environ["HELLOHPC_ASSIGNED_NPU_COUNT"]):
    raise ValueError("Assigned NPU count does not match platform visibility")
driver = ctypes.CDLL("libascend_hal.so")
logical_ids = []
for physical in physical_ids:
    logical = ctypes.c_uint()
    actual_physical = ctypes.c_uint()
    if driver.drvDeviceGetIndexByPhyId(physical, ctypes.byref(logical)) != 0:
        raise RuntimeError(f"Assigned physical NPU {physical} is unavailable")
    if driver.drvDeviceGetPhyIdByIndex(logical.value, ctypes.byref(actual_physical)) != 0:
        raise RuntimeError("Cannot verify the NPU mapping")
    if actual_physical.value != physical:
        raise RuntimeError("NPU mapping does not preserve platform assignment")
    logical_ids.append(str(logical.value))
print(f"NPU physical IDs {physical_ids} map to logical IDs {logical_ids}", file=sys.stderr)
print(",".join(logical_ids))
PY
)
    export ASCEND_RT_VISIBLE_DEVICES="$logical_devices"
fi

endpoint=$(python3 - <<'PY'
import ipaddress
import os
from urllib.parse import urlsplit

url = urlsplit(os.environ["HELLOHPC_SERVICE_BASE_URL"])
if url.scheme != "http" or not url.hostname:
    raise ValueError("Expected an HTTP loopback service URL")
if url.hostname != "localhost" and not ipaddress.ip_address(url.hostname).is_loopback:
    raise ValueError("Expected an HTTP loopback service URL")
print(url.hostname, url.port or 80)
PY
)
read -r service_host service_port <<< "$endpoint"

config_path="$runtime_dir/vllm-config.yaml"
printf '%s\n' "$resolved_config" > "$config_path"
printf 'Selected inference profile: %s\n%s\n' "$profile" "$resolved_config"

nohup vllm serve "$HELLOHPC_MODEL_PATH" \
    --config "$config_path" \
    --served-model-name "$HELLOHPC_MODEL_ID" \
    --host "$service_host" \
    --port "$service_port" \
    > "$HELLOHPC_SERVICE_LOG" 2>&1 < /dev/null &
