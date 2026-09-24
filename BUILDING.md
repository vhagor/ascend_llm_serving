# Inference container build, configuration, and validation

The local `ascend_llm` directory is the source of truth. Its competition code,
data, and submission files are mirrored to `~/ascend_llm` on the cluster.
Synchronize changed files before each build or test. No hash comparison step is used.
Generated SIF files, runtime caches, and full logs are not source files.

## Pinned image

- Tag: `quay.io/ascend/vllm-ascend:v0.26.0rc1`
- Platform: `linux/arm64`, Ascend A2 / 910B
- ARM64 manifest: `sha256:7265a3a08bba26b9376a4d577015ac8d459675fb3c1eec89aef6d9a19ac219d9`
- Ubuntu 22.04; Python 3.12.13; CANN 9.1.0
- vLLM 0.26.0+empty; vLLM-Ascend 0.26.0rc1
- PyTorch 2.10.0+cpu; torch-npu 2.10.0.post4; Triton-Ascend 3.2.2

The `empty` and `cpu` suffixes are expected for this plugin-based NPU stack.
The image records installed packages in
`/opt/inference/requirements.freeze.txt`. The %test section checks framework
and dependency versions using package metadata without loading the NPU driver.

## Build on an allocated compute node

After SSH login, allocate a single NPU and enter the node using `srun --pty bash`.
Use node-local temporary storage for the build: extracting and scanning the
image on the shared NFS directory was extremely slow in the first attempt.
Check the Slurm memory limit as well: `/tmp` on ascend6 is tmpfs and therefore
charges build files to job memory. A failed build can leave a root filesystem
behind when read-only package directories prevent cleanup. Do not accumulate
failed build trees across retries. A clean build previously fit the default
48-GiB allocation, while retaining a 16-GiB failed tree caused a confirmed OOM
during SIF compression. The later permission-denied cleanup error masked it.
On ascend6, host `/var/tmp` is local disk. The final fork-based build used it
and passed in 15m03s without OOM; it is the recommended build path on this node.

```bash
cd ~/ascend_llm
build_dir=$(mktemp -d /var/tmp/qwen-stage0.XXXXXX)
/nfs/bin/hellohpc test --case stage0 --set "sif=$build_dir/image.sif"
mkdir -p .hellohpc-local
cp "$build_dir/image.sif" .hellohpc-local/image.sif
```

Inspect Stage0 status before copying its output. On 2026-09-23, the node-local
build was accepted in 18 minutes 2 seconds. The resulting SIF was
6,078,517,248 bytes, below the 20 GiB limit. No model weights are embedded.

## Run Qwen

Use `npu-smi info` to identify the physical NPU assigned by Slurm. This allocation
did not populate the Ascend visibility variables, and the test tool otherwise
defaults to physical NPU 0. Supply the assigned physical ID explicitly:

```bash
/nfs/bin/hellohpc test --case stage1 --set devices=6
```

The value `6` is an example from the tested allocation, not a fixed requirement.
The submission script uses platform-provided visibility and the driver mapping
APIs to translate assigned physical IDs into container-local logical IDs. It
checks the reverse mapping before starting the service.

Runtime compiler temporary files and caches use a private directory under
`/var/tmp`, which was a writable, executable tmpfs in the competition container.
The platform's `/tmp` bind was NFS-backed; Triton compilation failed there during
temporary directory cleanup. The service log still uses `HELLOHPC_SERVICE_LOG`.

## Scope and observed limitations

- The launch script selects separate Qwen and DeepSeek task profiles.
- The DeepSeek profile is a starting configuration, not a validated deployment.
- DeepSeek and eight-NPU execution have not been tested.
- Driver 25.2.3 successfully executed a basic NPU tensor operation with this image.
- Apptainer 1.1.6 refuses the evaluator's HOME environment override. The new
  build patches Ascend profiling to respect XDG_CONFIG_HOME without changing
  HOME. This requires rebuilding the SIF. The evaluator warning itself remains.
- Automatic Ascend CPU binding is explicitly disabled: the evaluator does not
  mount the host npu-smi executable. Slurm CPU affinity remains in effect.
- Safe compiler fallbacks and model generation-config notices are not hidden.
- Do not edit the supplied evaluator or embed evaluation data in the SIF.

## Manual task configuration

Edit qwen_config or deepseek_config at the top of submission/start.sh. Each
heredoc contains the exact JSON configuration passed to vLLM as a YAML file
(JSON is valid YAML). The platform task selects the corresponding block.
There is no menu, preview command, custom parameter validator, or automatic
parallelism calculation.

| Setting | Qwen | DeepSeek |
| --- | --- | --- |
| tensor-parallel-size | 1 | 8 |
| data-parallel-size | default | 1 |
| dtype | bfloat16 | bfloat16 |
| max-model-len | 10240 | 102400 |
| max-num-seqs | 1 | 20 |
| max-num-batched-tokens | 8192 | 8192 |
| gpu-memory-utilization | 0.90 | 0.90 |
| quantization | model default | ascend |
| enable-expert-parallel | default | true |
| tokenizer-mode | default | deepseek_v4 |

Set parallelism explicitly to match the assigned resources. When changing DP,
update TP yourself; no auto value is supported. Add other vLLM configuration
keys directly to the chosen block. For top-level boolean switches that must
be disabled, use the corresponding negative flag, such as
"no-enable-prefix-caching": true; this vLLM version skips false values in its
YAML-to-CLI conversion. The nested additional-config.enable_cpu_binding=false
value is passed as part of the object and remains valid.

Keep writable runtime paths in additional-config, including ascend_log_path.
The defaults disable automatic CPU binding and use /tmp/vllm-ascend-logs.
Model path, served model name, host, and port are supplied separately by the
platform launcher.

Synchronize start.sh to the cluster and restart the stage after editing. These
parameter changes do not require rebuilding the SIF. The script saves the
selected configuration beside the service log as service.config.json.

Qwen requires room for about 8192 input and 1024 output tokens. The DeepSeek
profile covers 98 Ki input plus 128 output tokens but remains unvalidated on
eight NPUs. D1, D2, and GSM8K use the same configuration throughout Stage2.

## Fork sources and precompiled editable installation

Stage0 fetches `hpc_dev` from these repositories:

- https://github.com/vhagor/vllm (based on v0.26.0)
- https://github.com/vhagor/vllm-ascend (based on v0.26.0rc1)

The branches start at the releases matching the precompiled base image. Release
branches contain backports not present on main, so the exact release commits
were selected with user confirmation. Local and cluster checkouts live under
`ascend_llm/frameworks/` and track the corresponding fork branches.

Source trees stay at `/vllm-workspace/vllm` and
`/vllm-workspace/vllm-ascend` inside the image, including Git metadata. The build
updates origin to the fork, fetches hpc_dev, switches to that branch, and
reinstalls both projects editable without resolving dependencies:

```bash
VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 \
  python3 -m pip install --no-deps --no-build-isolation -e /vllm-workspace/vllm
COMPILE_CUSTOM_KERNELS=0 SOC_VERSION=ascend910b1 SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0rc1 \
  python3 -m pip install --no-deps --no-build-isolation -e /vllm-workspace/vllm-ascend
```

Precompiled artifacts come from the pinned official A2 image and remain in the
source directories during checkout. Ascend's COMPILE_CUSTOM_KERNELS=0 skips
compilation; it does not download or copy binaries by itself. vLLM uses the
empty backend for this NPU plugin setup, rather than CUDA wheel extraction.
The base CANN, torch, torch-npu, and Triton versions remain unchanged. Build-only
helpers setuptools-rust 1.12.0 and semantic-version 2.10.0 are installed without
resolving or upgrading the framework dependency stack.

The base repositories originally track release tags only. The recipe changes
the origin fetch specification to remote branches before setting up hpc_dev
tracking. The recipe performs a direct Git fetch and checkout, followed by
editable installation. There are no custom download retries, DNS overrides,
editable-location checks, native-file assertions, or source manifest generation.
The requested pip freeze inventory and %test version checks are retained. Standard command failures stop the build.
The recipe retains the profiling configuration fix and owner-write permissions
needed to clean up Ascend package directories after an interrupted build.

Python-only tuning can reuse these artifacts when native interfaces remain
compatible. Changes to native operators or the dependency/ABI stack require
rebuilding the matching binaries. A newer hpc_dev tip is fetched on each build;
the retained Git repository identifies the source actually used. The package version
remains tied to the selected release compatibility stack.

Both complete source repositories remain in the SIF. Editable installation
resolves imports to those source paths; a SIF itself remains read-only. Commit
and push changes to hpc_dev and rebuild Stage0 to include them in a submission.

Upstream reference:
https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc1/Dockerfile

## Runtime validation

Validate changes by running the selected stage on allocated Ascend hardware.
The obsolete menu and profile-preview tests have been removed with those tools.

## Edit framework sources

Both repositories and Git metadata remain inside the read-only SIF. Edit the
working copies under frameworks/, commit and push to the corresponding hpc_dev
branches, synchronize the project, and rebuild Stage0. Local uncommitted edits
do not alter an existing SIF. The scripts directory and source export helper
have been removed.

For an independently prepared writable development mount, preserve the matching
native artifacts and mount each repository at its original /vllm-workspace path.
Changes to native operators or their interfaces require recompilation.

## Validated fork image (2026-09-23)

Both user forks on hpc_dev are now built and tested. Stage0 run
`d86fe402122d480da6ff7438aedb56ea` passed in 15m03s, producing a
6,078,726,144-byte SIF. Both projects passed editable-location and version
checks, retained native artifacts, and preserved their source trees and Git
metadata. Stage1 run `062545b633254cf39b081bf07891c7f8` passed in 6m46s on
ascend6, physical NPU 2, port 18149.

| Metric | Result |
| --- | --- |
| Qwen score | 30/30 |
| Formal output throughput | 32.93422708 tokens/s |
| p99 TTFT | 7.10298623 s |
| p99 TPOT | 28.03110557 ms/token |
| Formal requests | 5/5 |
| Mathematics correctness | 1/1 |

Remote images: `.hellohpc-local/qwen-hpc-dev.sif` and the default
`.hellohpc-local/image.sif` reference this validated build. Earlier baselines
remain preserved under their versioned names. Evidence is synchronized under
`logs/forks-resume-2026-09-23/`. Only the latest successful validation
evidence is retained; old baselines and failed-attempt logs were cleaned up. The
service log contains zero ERROR, traceback, permission-denied, or Bindcpus
matches. The nonfatal vendor probe, Ascend parameter fallback, compiler safety
fallback, and model generation-config notices remain visible.

DeepSeek Stage2 was not run. These are local self-test results, not an OJ
submission. No source or binary hash comparison was added.

## Local recipe update (2026-09-24)

Simplified the recipe to direct fork fetch/checkout, required build dependencies,
editable installation with retained native artifacts, and the existing profiling
permission fix. Removed special DNS handling, custom retries/timeouts, metadata
labels, editable-location/native-file assertions, and source manifest.
Restored pip freeze and the %test version checks at the user's request.

This revision has not been built or evaluated; the results above describe the
previous recipe. The user will run the full workflow. This revision was synchronized
to the cluster on 2026-09-24; no build or evaluation was started.

## Manual launcher and cleanup update (2026-09-24)

start.sh contains two directly editable task configurations with explicit TP
values. Removed render_config(), the preview/menu tooling, the entire scripts
directory, and obsolete tests. The project was synchronized to the cluster on 2026-09-24.
No build or evaluation was run for these edits; the successful image and results
above describe the earlier validated revision.

## Git version control

The main repository uses branch main and origin
https://github.com/vhagor/ascend_llm_serving.git. The two repositories under
frameworks/ are Git submodules with hpc_dev configured as their update branch.
The parent repository records a specific commit for each submodule.

After the initial commit has been pushed, a fresh checkout can use:

```bash
git clone --recurse-submodules https://github.com/vhagor/ascend_llm_serving.git
cd ascend_llm_serving
git submodule update --init --recursive
```

Commit and push framework changes in their own repositories, then commit the
updated submodule pointers in the parent repository. A normal submodule update
uses the recorded commits; git submodule update --remote explicitly advances
to the configured branch. The container recipe still fetches the latest hpc_dev
branch from each fork and does not read the parent repository's submodule pins.

Local run evidence is stored under logs/, which is ignored by Git. The current
project and Git/submodule metadata were synchronized to the cluster via SCP on
2026-09-24, including the validation-to-logs rename. No initial commit or push
has been made. Existing cluster SIF/runtime artifacts remain available.
