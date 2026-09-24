# Local Git setup: 2026-09-24

- Initialized the main repository on main; origin is https://github.com/vhagor/ascend_llm_serving.git.
- Registered frameworks/vllm and frameworks/vllm-ascend as submodules, with hpc_dev as their update branch. Existing checkouts and commits are preserved.
- Renamed validation/ to logs/ and ignored /logs/.
- No commit or push has been made. The project, staged Git metadata, submodule layout, and logs directory are synchronized to the cluster via SCP on 2026-09-24. Cluster-only SIF/runtime artifacts are retained.

# Project state: 2026-09-24

- Build uses the user forks on hpc_dev, retains precompiled artifacts, and installs both frameworks editable.
- pip freeze and %test version checks are retained. Custom DNS handling, retries, source manifest generation, and extra installation assertions are removed.
- Edit qwen_config and deepseek_config directly in submission/start.sh. Default TP is 1 and 8; render_config and menu/preview tooling are removed.
- The entire scripts directory and obsolete tests are removed.
- Only logs/forks-resume-2026-09-23 successful-run evidence is retained. Old baselines, failed-attempt logs, stale scratch paths, and test caches are removed.
- Project files are synchronized to ~/ascend_llm on the cluster as of 2026-09-24. No build or evaluation is run for these edits.

## Last validated image

- Remote image: .hellohpc-local/qwen-hpc-dev.sif; default image.sif references the same image.
- Stage0: accepted in 15m03s, size 6,078,726,144 bytes.
- Qwen3-14B Stage1: 30/30, 32.93422708 output tokens/s, mathematics 1/1, duration 6m46s.
- Hardware: ascend6, physical NPU 2. Allocation was released after the run.
- vLLM 0.26.0+empty, vLLM-Ascend 0.26.0rc1, torch 2.10.0+cpu, torch-npu 2.10.0.post4, triton-ascend 3.2.2.
- These results apply to the earlier image, not the current unexecuted recipe/launcher changes. DeepSeek remains unvalidated.

## Cleanup backup

Remote pre-cleanup backup: ~/ascend_llm-before-cleanup-20260924-102453.tar.gz.
Existing SIF images were retained. Fifteen obsolete remote run directories and
eight loose legacy logs were removed; the latest successful Stage0/Stage1 runs
remain available. Framework Git metadata was preserved during source sync.
