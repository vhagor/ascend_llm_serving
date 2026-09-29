# Ascend LLM Serving 性能报告

截至 2026-09-28，DeepSeek 八卡 baseline 得分为 15.1901/70，正确性和延迟门槛均通过。DSpark 7 组合配置的吞吐更高，但 D1 的 p99 TTFT 超过门槛，总分降至 13.7283/70。两轮还改变了其他配置，不能把分数差单独归因于 DSpark。

正式 baseline 来自作业 7926；作业 7813 提供早期抽样诊断。D1 的 p99 TPOT 距 74 ms 门槛只有 5.62% 余量。调度中的请求数已接近客户端并发上限，单独增大 `max-num-seqs` 缺少收益依据。旧诊断的设备 trace 曾在本地通过 CANN 解析恢复，相关原始数据现已清理；后续补采的 baseline 和 DSpark timeline 保存在 [logs/timeline](logs/timeline/)。

## 当前归档与项目状态

`logs` 只保留 [服务日志](logs/service/) 31 份、[评测日志及报告](logs/evaluation/) 35 份、[镜像构建日志](logs/build/) 2 份，以及 [最终 timeline](logs/timeline/) 32 份。文件名包含日期与框架版本。baseline 与 DSpark 各有 D1/D2 rank0 至 rank7 的 16 份 Perfetto；分类整理时，移动前后的 SHA-256 均通过核对。原始采集、CANN 导出、解析脚本、验证 JSON 和硬件快照已按要求删除。下文涉及这些材料的旧路径只说明当时的分析来源。

Qwen3-14B 已完成服务调通和本地正确性验证，用户确认过 OJ 通过。本地 [Stage1 报告](logs/evaluation/2026-09-23_vllm-0.26.0_vllm-ascend-0.26.0rc1_qwen3-14b_stage1-smoke_report.json)记录数学题 1/1、输出吞吐 32.934 token/s、p99 TTFT 7.103 s、p99 TPOT 28.031 ms；这些是本地成绩，未代替 OJ 记录。更早的 2026-09-24 状态记录还载有旧 Qwen SIF 的 Stage0 在 15 分 03 秒内通过、镜像大小 6,078,726,144 字节，以及 Stage1 30/30、历时 6 分 46 秒；这些数字属于当时的镜像，不代表后来重编的版本。

项目已建立 SIF 构建、启动和 `hellohpc` 自测流程，vLLM 与 vLLM-Ascend 源码以可编辑安装方式保留，并纳入 submodule 管理。WSL 的 GitHub SSH 路径曾通过本地 Clash 代理走 443 端口完成推送。vLLM-Ascend 的 release 分支 `hpc_dev` 已推送提交 `cff0538`；基于官方 main 提交 `9df651d0b` 的 `feat/dspark-local-argmax-main` 分支已推送 `b85954bb0`，尚未创建 PR。该 main 分支的 24 项 CPU 与源码隔离测试及 Ruff 检查通过，完整仓库测试未在本地通过依赖加载阶段。

当前 [启动脚本](submission/start.sh)设置 DSpark 5、local argmax 开启、FlashComm1 关闭、token 预算 16384、`scheduler-reserve-full-isl=false` 和共享专家 DP。它晚于本文 DSpark 7 的正式评测与 profile，尚无对应的八卡 NPU 正确性、分数或性能数据。KV cache 8-bit 改造也未完成。2026-09-28 最近一次集群检查时，登录节点仍可访问，但账号对应的 contest 分区不可用；作业 15744 曾在分配节点前出现 NODE_FAIL。之后没有恢复评测任务，远端控制器是否停止未得到直接确认，最近队列检查没有本账号的运行或排队作业。

## 1. 环境与统计范围

| 项目 | 本次运行 |
| --- | --- |
| 节点 / 队列 | ascend9 / contest-full |
| NPU / 互联 | 8 × Ascend 910B3，每卡 64 GiB HBM / HCCS |
| CPU / 作业内存 | 192 核 Kunpeng-920，4 socket / 768 GiB |
| 驱动 / CANN | 25.2.3 / 9.1.0 |
| vLLM / vLLM-Ascend | 0.26.0 / 0.26.0rc1 |
| torch / torch-npu | 2.10.0+cpu / 2.10.0.post4 |
| 模型 / 量化 | DeepSeek-V4-Flash-0731-w8a8 / ascend W8A8 |
| 并行 | TP=8、DP=1、EP 开启；1 个 EngineCore、8 个 Worker |
| 精度 / KV block | bfloat16 / 128 token |
| 最大上下文 / 最大序列数 | 102400 / 32 |
| 每步 token 预算 / 显存比例 | 8192 / 0.90 |
| Chunked prefill / Prefix caching | 均开启 |
| 投机解码 / 动态 EPLB / CPU binding | 均未开启 |

硬件、环境版本和启动配置取自归档前的运行快照；[正式 baseline 服务日志](logs/service/2026-09-26_vllm-0.26.0_vllm-ascend-0.26.0rc1_deepseek-v4-flash-0731-w8a8_baseline_output__logs__service.log)仍可查阅。

正式成绩来自完整评测，调度和资源细节主要来自抽样 profile。抽样种子为 20260925，包含 D1 的 20 个请求和 D2 四个 session 的六轮请求，共 44 个，不重复 GSM；它沿用原始请求体和输出限制。抽样的缓存历史、请求补入过程与完整评测不同，因此吞吐不能直接对比。

## 2. 完整 baseline 结果

| 指标 | D1 | D2 |
| --- | ---: | ---: |
| 完成正式请求 | 60/60 | 144/144 |
| 客户端并发上限 | 20 | 4 |
| 正式阶段耗时 | 856.26 秒 | 1390.88 秒 |
| 输出吞吐 | 287.014 token/s | 13.252 token/s |
| p99 TTFT / 门槛 | 83.487 / 96.4 秒 | 54.815 / 60.5 秒 |
| p99 TPOT / 门槛 | 69.839 / 74 ms/token | 343.196 / 380 ms/token |
| TPOT 剩余余量，相对门槛 | 5.62% | 9.69% |
| 数学正确性 | 3/3 | 3/3 |
| 得分 | 5.9833/30 | 9.2068/40 |

Stage2 合计 15.1901/70，属于本地自测成绩。此次没有重跑 Stage1，流程总分显示的 15.19/100 没有计入 Qwen 成绩。

GSM 完成 1319/1319，答对 1276 题、答错 43 题，正确率 96.74%，没有截断。正确数高于 1212 题的门槛 64 题，比参考正确数 1275 多 1 题。本轮达到正确性门槛；一次结果无法说明每次都会优于参考。

作业从 05:23:09 运行至 06:37:31，历时 74 分 22 秒。evaluator 生命周期为 4440.40 秒，比包含外围准备和清理的 Slurm 作业略短。模型就绪约耗时 739.33 秒，D1 预热约 340.24 秒，GSM 阶段约 1112.79 秒。

正式吞吐包含 prefill、请求补入和收尾；服务日志里的瞬时 decode 吞吐只反映其中一段。后续调优需要同时守住 D1 TPOT 门槛。

证据：[完整报告](logs/evaluation/2026-09-26_vllm-0.26.0_vllm-ascend-0.26.0rc1_deepseek-v4-flash-0731-w8a8_baseline_output__report.json)、[hellohpc 输出](logs/evaluation/2026-09-26_vllm-0.26.0_vllm-ascend-0.26.0rc1_deepseek-v4-flash-0731-w8a8_baseline_hellohpc.log)。

## 3. 实际 batch 与 token 长度

抽样 profile 在 scheduler 更新已计算 token 数之前记录请求，共有 5001 个非空调度步。表中 batch 是每步实际调度的请求数，不是客户端并发数或八卡请求数之和；图捕获的 padding 也不计入。各列按调度步等权平均，反映的是抽样运行。

| 阶段 | 调度步数 | 平均请求数/步 | 纯 decode 步平均请求数 | batch 内请求平均完整输入 token | 含 prefill 的步数 | 含 prefill 步的平均 prefill token 总数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 4181 | 19.61 | 19.79 | 32768 | 86 | 7620.47 |
| D2 第 1 轮 | 128 | 4.00 | 4.00 | 2048 | 1 | 8192.00 |
| D2 第 2 轮 | 144 | 3.66 | 3.80 | 32768 | 17 | 7710.12 |
| D2 第 3 轮 | 144 | 3.66 | 3.80 | 65536 | 17 | 7710.12 |
| D2 第 4 轮 | 144 | 3.66 | 3.80 | 98304 | 17 | 7710.12 |
| D2 第 5 轮 | 130 | 3.94 | 3.96 | 99328 | 2 | 2048.00 |
| D2 第 6 轮 | 130 | 3.94 | 3.96 | 100352 | 2 | 4096.00 |

完整输入包含已缓存前缀；本步 prefill 只计该步安排计算的 prompt 部分，并在请求之间共享 8192 的预算。二者不能混用。

这 5001 步中，19 步为纯 prefill，123 步混合 prefill/decode，4859 步为纯 decode。baseline 没有开启投机解码，纯 decode 步的有效 token 数与请求数相同：D1 平均 19.79 token/步，D2 各轮约 3.80 至 4.00 token/步。第一枚输出可以在 prefill 中产生，所以步数不能直接换算为完整输出长度。

“平均 decode 长度”的三种口径如下：

| 口径 | D1 | D2 |
| --- | ---: | ---: |
| 正式 baseline 每请求完整输出 | 4096 token | 128 token |
| profile 中 batch 内请求截至该步已生成 token 的均值，再对步平均 | 2046.26 | 各轮 60.92–62.51 |
| profile 纯 decode 步处理的有效 token 总数均值 | 19.79 | 各轮 3.80–4.00 |

已生成长度是累计进度，不是最终回答长度。完整 baseline 中 GSM 每请求平均输入 85.95 token、输出 98.77 token，输出范围 34–541。

有效 batch 接近客户端供给上限：D1 为 20，D2 为 4，引擎上限则为 32。增加 `max-num-seqs` 不会增加客户端请求。出现 Waiting 时，还需区分 prefill 预算、KV 分配和异步执行造成的限制。

本节的逐步记录、统计 JSON 和复算脚本均已按归档要求删除；上表保留清理前计算的结果。

## 4. Chunked prefill、前缀复用和 KV cache

服务日志记录了 chunked prefill 和 prefix caching；抽样逐步记录的最大 token 数为 8192。D1 的 86 个含 prefill 步中，82 个还包含 decode 请求。

| D2 轮次 | 四个请求完整 prompt 总量 | 实际调度 prefill token 总量 |
| --- | ---: | ---: |
| 第 1 轮 | 8192 | 8192 |
| 第 2 轮 | 131072 | 131072 |
| 第 3 轮 | 262144 | 131072 |
| 第 4 轮 | 393216 | 131072 |
| 第 5 轮 | 397312 | 4096 |
| 第 6 轮 | 401408 | 8192 |

第 3 轮以后，实际调度的 prefill token 明显少于完整 prompt，符合前缀缓存复用的预期。这些数字衡量调度工作量，没有给出 vLLM 的缓存命中率。完整 24-session 评测的缓存竞争也与该抽样不同。

每卡权重加载内存增量 36.1983 GiB，可用 KV 空间约 16.65 GiB，图捕获约 0.67 GiB。引擎报告 192218 个 KV token、102400 长度下最大并发估算 1.88x；这不是所有上下文长度下只能运行 1.88 个请求的固定限制。

正式 baseline 服务日志快照为：

| 阶段 | 样本数 | Running 平均/最大 | Waiting 最大 | KV cache usage 平均/最大 |
| --- | ---: | ---: | ---: | ---: |
| D1 | 85 | 18.81 / 20 | 17 | 41.79% / 80.7% |
| D2 | 139 | 3.33 / 4 | 3 | 54.45% / 66.4% |
| GSM | 112 | 19.39 / 20 | 0 | 4.07% / 4.7% |

快照没有显示 KV 总容量长期耗尽。瞬时分配失败或某个 KV 组的局部限制仍需用分配失败、抢占和等待原因确认。KV cache usage 衡量缓存占用，与驱动报告的 HBM 容量占用口径不同。

清理前的文件头统计得到权重张量 293.014 GiB，其中 MTP 为 21.857 GiB；主加载器跳过 MTP。八卡加载增量为 289.586 GiB，比主模型磁盘张量的 271.157 GiB 多 18.429 GiB。差额尚未逐张量归因，不能全部算作 KV cache。

## 5. Core、HBM 与通信负载

### 5.1 正式 baseline 的 AI Core 利用率

按正式请求开始/结束时间划分设备快照。两次运行同在 ascend9，用 profile 中 wall-clock 与 monotonic 的差值对齐；每卡 D1 有 76 个样本，D2 有 123 个样本。以下为等权样本均值，不是连续时间积分。

| NPU | D1 AI Core | D2 AI Core |
| --- | ---: | ---: |
| 0 | 39.99% | 50.57% |
| 1 | 41.13% | 52.91% |
| 2 | 41.13% | 51.86% |
| 3 | 39.25% | 51.20% |
| 4 | 40.71% | 52.35% |
| 5 | 40.42% | 53.03% |
| 6 | 40.50% | 51.41% |
| 7 | 40.28% | 50.62% |

各卡均值接近，未见长期明显离群。短时专家失衡仍需逐层检查。D2 的 Core 利用率高于 D1，输出吞吐却更低；两者的输入长度、prefill 占比和 batch 不同。`AICore(%)` 也不是峰值 FLOPS 达成率。

这组数值来自正式运行时的设备快照；原始快照已按归档要求删除。

### 5.2 名义通信上限与实测带宽

华为的 [Atlas 800I A2 材料](https://e.huawei.com/marketingcloud/pep/asset/20000001/Material/744857c25fe4438b97c85ed15fd79d1a/M3T1A669N1162432897309413660/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88R24_0-%E5%BD%A9%E9%A1%B5_25H2_.pdf)给出 HCCS Full Mesh 双向互联 392 GB/s。本报告暂以它作为单 NPU 聚合双向带宽的参考，按对称发送/接收各 196 GB/s、七链路均分则单链路单向 28 GB/s 计算。ascend9 的整机规格、聚合口径和实际链路速率尚未核实，也没有独立峰值测试。因此表中的百分比是相对这个假设的比值。

以下来自 profile 抽样重放，排除模型启动阶段；D1 每卡 11 个样本、D2 每卡 7 个样本，单次 HCCS 窗口 1000 ms，间隔约 30 秒。仍包含 prefill、decode 和 profiler 控制/导出停顿，不是稳定 decode 专项测试。

| NPU | D1 Tx / Rx（GB/s） | D1 双向参考比值 | D2 Tx / Rx（GB/s） | D2 双向参考比值 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.988 / 5.016 | 2.55% | 3.336 / 3.336 | 1.70% |
| 1 | 3.691 / 3.687 | 1.88% | 3.326 / 3.326 | 1.70% |
| 2 | 3.952 / 3.947 | 2.02% | 3.409 / 3.409 | 1.74% |
| 3 | 3.661 / 3.657 | 1.87% | 3.349 / 3.349 | 1.71% |
| 4 | 3.780 / 3.776 | 1.93% | 3.420 / 3.420 | 1.74% |
| 5 | 3.964 / 3.960 | 2.02% | 3.416 / 3.416 | 1.74% |
| 6 | 3.647 / 3.643 | 1.86% | 3.482 / 3.482 | 1.78% |
| 7 | 3.697 / 3.693 | 1.88% | 3.463 / 3.463 | 1.77% |

计算式为 `(Tx + Rx) / 392 × 100%`，Tx/Rx 取 `npu-smi info -t hccs-bw` 的 `total`。卡间采样不严格同步，链路编号不能直接当作远端 rank；全机 Tx+Rx 相加会对传输重复计数。

同一 profile 采样中，各卡 HBM 带宽利用率均值：D1 16.27%–17.36%、D2 8.14%–8.86%；Core 均值分别为 30.36%–31.18%、13.43%–15.14%。这些低于正式 baseline 的 Core 数据，不能当成 baseline 退化证据：诊断过程含采集停顿、请求数量与缓存历史也不同。

短窗口平均带宽远低于名义上限。高频小消息、collective 同步、专家 dispatch/combine 和计算通信重叠不足仍可能增加延迟。软件通信后端是 HCCL，硬件互联是 HCCS；链路平均 GB/s 无法单独解释关键路径。

本节的 HCCS 与 HBM 原始采样、分阶段统计 JSON 已清理；表中保留归档前的样本均值。

## 6. Torch profile 产物和 CPU 调度分析

早期采样作业从 06:37:31 运行至 07:00:04，耗时 22 分 33 秒；44 个请求全部完成，8 次采集窗口都收到 start/stop 成功响应。当时的原始产物约为 6.22 GB（5.79 GiB），现已清理。下面保留清理前确认的采集范围。

| 产物 | 状态 |
| --- | --- |
| rank 0–7 原始采集 | 当时每 rank 8 份，共 64 份；原始文件已清理 |
| EngineCore CPU trace | 当时解析 8 份摘要；原始文件已清理 |
| 前端 CPU trace | 原始文件已清理 |
| 调度记录 | 5001 个非空 step；统计保留在本报告 |
| 进程树、容器/宿主 PID 映射 | 原始记录已清理 |
| NPU 算子与 stream 导出 | 当时完成 CANN 解析；早期导出已清理 |
| HCCL 和 AI Core 数据 | 当时导出通信、逐算子与管线统计；汇总见第 9 节 |

profile 的 EngineCore 容器 PID 为 191，Worker rank 0–7 的容器 PID 依次为 230、262、300、379、438、496、555、614。它们是八个 Worker 共享一个 EngineCore，不是八个独立 EngineCore；baseline 的 PID 不同。

早期诊断服务设置 `profiler-config.max_iterations=12`，Worker 达到上限便停止设备采集。D1 decode 的 HTTP 窗口约 15.34 秒，EngineCore 标注 215 个 step，而 rank0 的 NPU trace 从 `PROFILING_ENABLE` 到 `PROFILING_DISABLE` 只有约 0.803 秒。因此当时的算子与通信统计只覆盖开头约 12 个 Worker step。后续完整 timeline 补采已另行完成，见第 13 节。

早期解析连续遇到两处故障：torch-npu 禁止 daemon Worker 启动解析子进程；登录节点镜像内的原生 `msprof` 又在驱动平台初始化时失败，直接错误为 `Init platform by driver faild!`，CANN export 随后报 `ERR04005`。最初五个 `trace_view.json` 为空或不完整。后来在本地 x86_64 CPU 上调用官方 msprof 的 Python 解析入口，恢复了设备导出，并逐份检查 JSON 与非空设备事件。相关原始解析日志已清理。

已有 EngineCore CPU trace 可以作有限分析。D1 第二个窗口（约 06:53:20 起）的标注统计：

| 标注 | 次数 | 累计跨度 | 平均每次 |
| --- | ---: | ---: | ---: |
| step_with_batch_queue | 215 | 15279.99 ms | 71.07 ms |
| scheduler.schedule | 215 | 352.35 ms | 1.64 ms |
| scheduler.update_from_output | 215 | 89.25 ms | 0.42 ms |
| executor.execute_model | 215 | 50.54 ms | 0.24 ms |
| executor.sample_tokens | 215 | 28.56 ms | 0.13 ms |

`scheduler.schedule` 的累计跨度约为外层 step 的 2.31%，这个窗口没有显示调度函数本身占据主要 step 跨度。其余部分不能全部归为 NPU 计算：外层包含异步等待和同步，标注有嵌套关系；主机 `execute_model` 返回时间也不代表设备执行时间。该结果不足以支持“切换 V2 runner 或开启 CPU binding 就能大幅提速”。

当时从集群同步了 baseline 报告、日志、硬件记录、EngineCore CPU trace、进程映射、调度和利用率数据，以及八卡共 64 份原始 NPU 采样。本地原始采集后来已按归档要求清理。集群当时的原始目录为：

```text
/nfs/home/acct-stu/stu1649/ascend_llm/logs/stage2-2026-09-25/profile/output/logs/torch-profile/
```

修复前的 CPU 摘要曾把空的设备 trace 也列入覆盖范围，因此那份索引已经失效。后续补采的最终 Perfetto 已合并 EngineCore 标注、Worker Python 调用和 NPU stream；早期诊断文件没有补出当时未采集的 Python 栈。

## 7. 后续验证顺序

| 优先级 | 建议 | 验证目标 |
| --- | --- | --- |
| 1，已完成补采 | 修正设备 profiler 的 12-step 自动停止，采集固定短窗口 | 八卡 NPU 与 EngineCore 的完整 timeline 已保存；性能归因仍需控制采集扰动 |
| 2 | D2 长上下文 indexer/稀疏注意力专项 | 在相同 batch、固定 32K/64K/96K 输入下比较单步时间和算子耗时；先确认 indexer 在关键路径上，再考虑源码中的 kernel/tiling 与缓存优化 |
| 3 | D1 decode 的 MoE 与 collective 专项 | 对齐 GroupedMatmul、QuantBatchMatmul 与约 87 次/step 的 AllReduce；判定通信等待和数据搬运是否限制 TPOT |
| 4 | 对比 prefill token 预算 | 以 8192 为对照，小样本比较 4096、16384；检查 D1 TPOT、D2 TTFT、吞吐与 KV 占用 |
| 5 | 对投机解码做单变量 A/B | 已验证 DSpark 7 token 的组合配置可运行，但 D1 TTFT 未过门槛；下一步固定其他参数，记录接受长度、draft/verify 耗时、KV 占用及正确性 |
| 暂不优先 | 增大 max-num-seqs、直接提高显存比例、切换 V2 | 请求供给低于序列上限；KV 瓶颈未证实；缺少当前组合的 V2 性能对照 |

prefill 预算涉及 TTFT 与 token 间延迟的权衡，官方文档提供一般方向，不能代替本模型实测：[vLLM 调优说明](https://docs.vllm.ai/en/v0.18.0/configuration/optimization/)。第 10 节的完整评测证明当前镜像能够运行 DSpark 7 token，但同轮更改了多个参数，尚不能分离 DSpark 的单独收益。[vLLM 投机解码说明](https://docs.vllm.ai/en/v0.20.2/features/spec_decode.html)用于解释机制，具体开关以本地 0.26.0/0.26.0rc1 源码为准。

后续实验应逐次改变一个因素，使用相同请求集、并发、缓存预热方式和输出限制。profiler 运行不计正式分数；小样本改善后，再做完整数学题、GSM 和 p99 门槛验证。第 10 节记录的是组合配置的实测差值。

## 8. 早期数据与资源状态

下列命令记录了当时的统计方法。本地脚本和原始数据现已清理，当前工作区不能直接重跑它：

```bash
python3 logs/stage2-2026-09-25/summarize-performance.py
```

作业 7926、7813 均已完成，八卡于 07:00:04 释放。08:14 检查账号队列为空，后续只在登录节点解析文件，未重新申请 NPU。

未完成的旧 baseline 和单独的旧报告已清理。本报告第 2 节的正式成绩及第 5 节的 Core 统计取自完整运行。

## 9. 本地离线解析修复与设备热点补充

当时在本地解析了全部 64 份 CANN/NPU 采集，rank0–7 各 8 份，失败 0。检查包括导出完成、JSON 可读取，以及存在非空 Ascend Hardware 事件和 stream。解析使用官方 msprof `26.0.0` 分支的 Python 入口及独立 Python 3.10.12 环境，不依赖本地 NPU 驱动；没有重新申请 NPU 或重跑模型。

这批原始数据可解析。前述报错发生在解析流程，不能据此判断计算节点或模型推理故障。

早期 `PROF_*/mindstudio_profiler_output/` 曾包含 timeline JSON、逐算子 CSV 和通信汇总，现已删除。归档前还生成了 rank0 的 D1 合并、D1 prefill、D1 decode 和 D2 第 1 轮视图。D1 合并视图中间有 193.2 秒未采集区间；三个原窗口的调度标注分别覆盖 2/2、215/215、103/103 个 CPU 步。早期 Worker 使用 `with_stack=False`、`with_modules=False`，离线合并无法补出未采集的 Python 调用栈。后续补采的完整 timeline 采用了不同采集设置，见第 13 节。

D1 opening 与 D1 decode 的八卡算子累计耗时占比显示不同热点：前者最高为 SparseAttnSharedkv 20.10%、ScatterNdUpdateV2 16.22%、HcPre 14.98%；后者最高为 GroupedMatmulSwigluQuant 14.88%、QuantBatchMatmulV3 8.16%、HcPre 7.90%。这些是各采集窗口内的算子耗时构成，不能当作端到端阶段耗时或单步关键路径。D1 opening 的 EngineCore CPU 标注仅覆盖两个 prefill 调度步；逐步瓶颈还需在 Perfetto 中对齐对应的 stream 片段。

D1 合并视图已修复 Perfetto 的重叠 slice 导入告警，v58.2 验证 283,495 个 slices 且无 trace health issues。它按实际调度步标记阶段：已采集的 opening 两步为 prefill，后续 215 步为 decode；中间 193.2 秒没有 profile 数据，不能推断其中是否发生 mixed batch。vLLM 在启用 chunked prefill 时允许一个 batch 同时含 prefill 与 decode，本次按采集窗口命名不代表两者必须分开执行。

### 9.1 D2 长上下文算子随长度变化

八卡设备采集开头约 12 个 Worker step 中，D2 第 2–4 轮四个请求的完整 prompt 分别约为 32K、64K、96K token。三轮 `VllmQuantLightningIndexer` 均记录 3,360 次，单次平均设备算子耗时从约 2.11 ms 升至 5.49 ms、8.83 ms，在各窗口算子累积耗时中的占比为 8.1% → 18.2% → 25.8%。调用次数相同而单次耗时随上下文增长，说明长上下文 indexer 是优先验证的 D2 热点。`SparseAttnSharedkv` 在三轮均约占 20%–21%。这些是算子耗时，存在跨卡、跨 stream 重叠，不能直接当作 TTFT 或 TPOT 的归因比例。

### 9.2 D1 AI Core 利用率的可解释部分

正式 baseline 的 76 个八卡平均采样点：均值约 40.43%、中位数 41.13%；去掉三个低于 20% 的采样点后均值仍约 41.65%。因此 40% 不是仅被少量请求间空档拉低。`npu-smi` 的 AICore(%) 是占用率指标，不等于算力 FLOPS 达成率；也不能直接用 `100%-40%` 当作可恢复的性能空间。[华为 npu-smi 字段说明](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/250RC1/re/npu/npusmi_014.html)

在 rank 0 的 D1 decode 设备采集里，以首个到最后一个非等待算子为边界，活动区间约 607.60 ms。跨所有 stream 合并时，约 469.59 ms（77.3%） 有至少一个非等待算子；约 113.11 ms（18.6%） 只有 `EVENT_WAIT`/`NOTIFY_WAIT` 等等待事件；约 24.90 ms（4.1%） 两类事件都没有。这是仅覆盖约 12 个 Worker step 的事件并集，不是 AICore(%) 的另一种计算：非等待算子可以只用部分 Core，等待也可与其他 stream 的计算重叠。等待事件尚未逐条归因到 HCCL、stream 同步或其他依赖。

在该 decode 样本中，`scheduler.schedule` 平均约 1.64 ms，而外层 step 平均约 71.07 ms；CPU 调度本身不像首要阻塞。`GroupedMatmulSwigluQuant` 与 `GroupedMatmul` 的 `aic_mte2_ratio` 分别约 0.758、0.809，`aic_mac_ratio` 约 0.110、0.126；MTE2 是 GM 到 AI Core 的数据搬运指令。这提示算子内部的数据搬运和 Cube 计算比例值得检查，但不是全卡 HBM 带宽饱和的证明。[华为 PipeUtilization 字段说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/optools/Operatordevelopmenttools/docs/en/user_guide/msopprof_performance_data.md)

将 NVIDIA NVML `GPU-Util` 的“采样期间至少一个 kernel 在执行”定义近似应用到 D1 decode 的八卡设备 trace：以每卡第一个至最后一个非等待设备算子为活动窗口，长度均约 608 ms；至少一个非等待算子在运行的时间比例为 76.2%–77.6%，八卡平均约 77.1%。如果把 profiler enable/disable 的初始化和收尾也放进分母，比例为 55.3%–63.8%，平均约 59.6%；后者会被采集控制开销稀释，不代表正式 baseline 的利用率。两种口径都是 trace 事件并集，不是 `npu-smi AICore(%)`，也不是 FLOPS 利用率。[NVIDIA NVML 定义](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlUtilization__t.html)

活动窗口中每卡约 17.5%–20.2% 是只有 `EVENT_WAIT`/`NOTIFY_WAIT`、没有非等待算子的时间；另约 3.4%–5.2% 两类事件都没有。把 `Communication` 轨道的 `hcom_*` collective 同步到同一时钟后，它们在每卡占约 80.8–97.0 ms，即活动窗口的 13.3%–15.9%，并全部落在“没有非等待算子”的区间。八卡平均 HCCL 窗口约 89.1 ms/608.1 ms = 14.6%，占无算子空档约 63.8%。所以当前短采样不能排除通信等待；HCCL 事件与空档重合不等于证明全部等待都是链路传输，具体还需区分 collective 内同步、远端 rank 等待和数据搬运。扣除 HCCL 窗口后，无算子空档仍平均约 50 ms/608 ms，Worker 侧 CPU launch 与其他 stream 依赖还没有足够事件覆盖来归因。

### 9.3 D1 decode 采样的算子热点

取每个 rank 的第二个采集窗口，即 D1 decode 窗口，合计八卡 `op_statistic` 中同一 OP Type 的耗时。以下占比的分母是该表所有算子的累计耗时，不是墙钟时间，不包括另表的 HCCL 通信累计时间。

| 算子类型 | 八卡累计算子耗时占比 |
| --- | ---: |
| GroupedMatmulSwigluQuant | 14.88% |
| QuantBatchMatmulV3 | 8.16% |
| HcPre | 7.90% |
| Compressor | 7.49% |
| GroupedMatmul | 7.44% |
| MatMulV2 | 6.93% |
| TransposeBatchMatMul | 5.48% |
| TensorMove | 4.47% |

热点分散在多个算子，不能用一个算子解释所有损耗。GroupedMatmulSwigluQuant 和 GroupedMatmul 值得优先检查：rank 0 同一窗口中，按调用等权平均的 `aic_mte2_ratio` 分别约为 0.758、0.809，`aic_mac_ratio` 分别约为 0.110、0.126。这提示数据搬运与小 batch 下的计算效率值得进一步研究；这些是算子执行内部的管线比例，不是整卡 HBM 带宽利用率，也不足以单独证明内存带宽已饱和。

### 9.4 D1 decode 窗口的通信与 rank 对比

| rank | 设备事件时间跨度 | 累计算子耗时 | 累计通信耗时 | AllReduce 次数 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 803.23 ms | 494.80 ms | 89.44 ms | 1044 |
| 1 | 845.10 ms | 492.51 ms | 93.35 ms | 1044 |
| 2 | 779.11 ms | 495.62 ms | 83.21 ms | 1044 |
| 3 | 752.66 ms | 497.70 ms | 91.10 ms | 1044 |
| 4 | 828.17 ms | 496.90 ms | 80.79 ms | 1044 |
| 5 | 732.21 ms | 492.94 ms | 82.98 ms | 1044 |
| 6 | 764.57 ms | 491.00 ms | 94.86 ms | 1044 |
| 7 | 793.51 ms | 488.02 ms | 96.95 ms | 1044 |

各 rank 的累计算子耗时接近，但采集边界和设备时间跨度并不完全一致，不能简单把跨度最大的 rank 判为慢卡。通信累计时间也不能与算子时间直接相加：事件可跨 stream 重叠，通信融合算子可能另有计算路径。

rank 0 的通信统计中，AllReduce 累计约 88.31 ms、1044 次，AllGather 约 1.12 ms、12 次。结合前述低平均 GB/s，下一步更值得检查频繁 collective 的等待与计算通信重叠，而不能仅从链路带宽占比判断通信是否重要。精确关键路径、重叠率和整个 D1/D2 的平均占比仍未量化。

## 10. DSpark 配置的完整 Stage2 评测

本轮使用与 baseline 相同的 `/nfs/home/acct-stu/stu1649/image.sif`，开启 DSpark、7 个 speculative token，并同时调整 KV block size、图捕获、HCCL 和若干 Ascend 参数。两轮均为未开启 profiler 的完整 Stage2 评测。下表比较的是整套配置，不能将差值单独归因于 DSpark。

| 指标 | Baseline：D1 | DSpark 配置：D1 | Baseline：D2 | DSpark 配置：D2 |
| --- | ---: | ---: | ---: | ---: |
| 完成正式请求 | 60/60 | 60/60 | 144/144 | 144/144 |
| 正式阶段耗时 | 856.26 s | 590.27 s | 1390.88 s | 1188.01 s |
| 输出吞吐 | 287.014 token/s | 416.350 token/s（+45.06%） | 13.252 token/s | 15.515 token/s（+17.08%） |
| p99 TTFT / 门槛 | 83.487 / 96.4 s | 160.119 / 96.4 s，未通过 | 54.815 / 60.5 s | 46.688 / 60.5 s |
| p99 TPOT / 门槛 | 69.839 / 74 ms | 47.188 / 74 ms | 343.196 / 380 ms | 276.844 / 380 ms |
| 数学正确性 | 3/3 | 3/3 | 3/3 | 3/3 |
| 阶段得分 | 5.9833/30 | 0/30 | 9.2068/40 | 13.7283/40 |

整轮得分从 15.1901/70 降至 13.7283/70，下降 9.62%。D1 吞吐提高、TPOT 缩短，但 p99 TTFT 比 baseline 增加 91.79%，超过 96.4 秒门槛，该阶段计零分。D2 的吞吐提高 17.08%，TTFT 缩短 14.83%，TPOT 缩短 19.33%，且通过门槛。这套配置未提高最终得分。

GSM8K 完成 1319/1319，正确 1274 题、截断 0 题，超过至少 1212 题的正确性门槛；与 baseline 的 1276 题相比少 2 题。这说明本轮正确性达标，但不能据此认为 DSpark 对每道题的输出都没有影响。

两轮启动配置的关键差异：`block-size` 从 128 改为 32；新增 DSpark 7 token、`FULL_DECODE_ONLY` 图捕获、FlashComm1、共享专家与多流参数、NPU 图参数、HCCL 环境变量和前缀缓存保留间隔。要判断 D1 TTFT 回退来自投机验证、prefill 与 decode 混排、图捕获尺寸，还是其他改动，需要固定其余参数逐项 A/B；目前没有单独的 DSpark 收益测量。

证据：[DSpark 完整评测报告](logs/evaluation/2026-09-26_vllm-0.26.0_vllm-ascend-0.26.0rc1_deepseek-v4-flash-0731-w8a8_dspark7-evaluation_report.json)、[baseline 完整评测报告](logs/evaluation/2026-09-26_vllm-0.26.0_vllm-ascend-0.26.0rc1_deepseek-v4-flash-0731-w8a8_baseline_output__report.json)。当时的启动脚本快照已随旧运行目录清理；当前脚本包含后续未评测的改动，不能用于复现这两轮配置。

## 11. DSpark 八卡等待依赖分析（2026-09-28）

### 11.1 结论与适用范围

分析对象为 `profile-safe-stage2` 的 D1、D2 八卡原始 CANN timeline、修正后的 Perfetto 产物及独立 Worker Python 调用记录。对六个到达时间差较大的 ReduceScatter，逐一检查全部八个 rank；这是长尾案例分析，不是随机样本的全程归因。

这些案例中的长等待与各 rank 的主机进度差有关：上一轮草稿生成、下一轮输入准备和 attention metadata 构建结束的时间不同，目标模型进入 collective 的时间随之错开。先到的卡等待其他卡，计算 stream 又等待通信 stream。现有证据不足以将整段等待算作链路传输或专家计算。

同时发现完整采集开启时，相同负载特征的调度间隔明显变长。当前重型 profile 适合定位路径，不能直接作为正式评测中的等待占比或潜在加速比。此前仅凭此 trace 排除 CPU 瓶颈的判断不成立。

### 11.2 等待在哪里

D1 rank0 的约 30 秒采集窗口：

| 轨道 | EVENT_WAIT 累计 | 与 HCCL 窗口重叠 | 与其他计算重叠 |
| --- | ---: | ---: | ---: |
| stream2 | 15.113 s | 14.364 s（等待的 95.0%） | 0.732 s |
| stream16 | 13.586 s | 9.451 s（等待的 69.6%） | 4.110 s |

各 stream 的等待不可相加为整卡空闲。计算采用 AI_CORE、AI_VECTOR_CORE、MIX_AIC、MIX_AIV、AI_CPU 事件时间并集，不是 Core 利用率计数器；HCCL 包含同步等待。

采集开始后约 14.966 s，stream2 的一次 44.145 ms WAIT 与 stream11 的 ReduceScatter 几乎完全重叠。通信结束后 stream11 RECORD，stream2 恢复 Tile。CANN 中该 WAIT 对应 `aclrtStreamWaitEvent`，CPU API 自身仅 7.65 us。此处支持等待通信完成，不能把整段等待算成 CPU 提交 WAIT 的耗时。

采集开始后约 2.300 s，stream16 的 110.747 ms WAIT 期间有约 95.179 ms 通信和 15.424 ms 其他计算；末尾依次出现 stream11 AllGather 结束、stream2 DynamicQuant 与 RECORD、stream16 DynamicQuant。支持 stream16 等待上游分支准备输入。导出没有直接提供被等待的 event handle，这些关联是强时序证据，尚不是所有 event 的标识级配对。

### 11.3 晚到 rank 的时间去向

下表时间范围是“最早 rank 进入此 collective”到“最晚 rank 进入”；计算与此前通信均为最晚 rank 在该区间内的事件并集。

| 负载 / collective 后缀 | 最晚 rank | 到达差 | 晚到卡计算 | 晚到卡此前 HCCL |
| --- | ---: | ---: | ---: | ---: |
| D1 / 346_1 | 6 | 61.386 ms | 1.696 ms | 0.000 ms |
| D1 / 346_2 | 3 | 56.738 ms | 2.512 ms | 0.225 ms |
| D1 / 3386_1 | 5 | 47.940 ms | 5.749 ms | 0.000 ms |
| D2 / 4570_1 | 6 | 60.294 ms | 3.019 ms | 0.000 ms |
| D2 / 0_8 | 0 | 60.074 ms | 1.845 ms | 0.000 ms |
| D2 / 346_21 | 5 | 56.699 ms | 1.465 ms | 0.088 ms |

因此，这六个窗口里，不能用“晚到卡一直在执行更重的计算”解释数十毫秒差异。无计算或 HCCL 覆盖的时间仍可能包括拷贝、同步和运行时活动，并非全部是 CPU 有效计算。

D1 `hcom_reduceScatter__503_346_1` 的 rank6，以上述最早到达为相对零点：

| Worker 阶段 | 起止时间 |
| --- | ---: |
| 上一轮 `sample_tokens` 返回 | +4.17 ms |
| `_update_states` | +5.23 ～ +9.57 ms |
| `_prepare_inputs` | +9.70 ～ +21.86 ms |
| `_build_attention_metadata` | +23.23 ～ +41.96 ms |
| `ACLGraphWrapper.__call__` | +44.17 ～ +61.31 ms |
| 设备进入 collective | +61.39 ms |

图提交阶段 CANN `aclmdlRIExecuteAsync` 本身跨度约 16.65 ms；这是主机 API 跨度，不等于 CPU 一直在做有效计算，也可能包含运行时阻塞。

同一次通信，rank1 的 `sample_tokens` 在 -58.1 ms 已返回，rank6 则到 +4.2 ms 才返回。两者 `_prepare_inputs` 都约 12 ms，metadata 构建都约 19 至 20 ms。相当一部分到达差在这两项准备工作开始之前就已形成。

Worker 栈把该 `sample_tokens` 进一步定位到 `propose_draft_token_ids` → `AscendSpecDecodeBaseProposer._run_merged_draft` → `DSparkDeepseekV4ForCausalLM.forward`，其中包含 DSA 与 MoE 分段调用。D2 rank0 的另一案例同样出现在草稿生成返回后，再进行约 14.68 ms 输入准备与 24.45 ms metadata 构建。这里的 `sample_tokens` 包含草稿模型推理，不能当成纯粹的随机采样 CPU 操作。

这些证据定位到草稿阶段至目标模型下一步之间的主机提交进度差。尚未进一步分离 Python 运行、线程调度、CANN 阻塞、采集回调开销分别占多少，也未证明 DSpark 本身在无 profiler 时有同样严重的错位。

### 11.4 采集扰动不可忽略

从同一次诊断服务的 scheduler 日志，对相同请求数、调度 token 数、prefill token 数及各请求完整 prompt 长度进行匹配。统计连续调度记录的墙钟间隔，排除开关状态跨界和超过 10 秒的间隔；这不是设备单 step 耗时，异步调度也可能影响间隔。

| 负载特征 | 未开启 profile 的间隔中位数 | 开启 profile 的间隔中位数 | 比值 | 样本数 off/on |
| --- | ---: | ---: | ---: | ---: |
| 17 请求，8000 tokens，其中 prefill 7872，各 prompt 32768 | 846.63 ms | 2323.89 ms | 2.74× | 28 / 9 |
| 4 请求，32 tokens，纯 decode，各 prompt 98304 | 72.97 ms | 174.22 ms | 2.39× | 66 / 37 |
| 3 请求，24 tokens，纯 decode，各 prompt 98304 | 73.67 ms | 179.29 ms | 2.43× | 62 / 40 |

这是显著的开关状态关联，不是严格随机 A/B：输出进度、缓存状态和运行时间仍可能不同，不能将全部倍率归因到 profiler。但它足以否定“完整采集对性能没有明显影响”的前提。

采集代码对 Worker 使用 `sys.setprofile` / `threading.setprofile`，记录框架 Python 调用，回调包含共享锁、时钟读取和二进制写入；EngineCore 同时开启 CPU profiler 与调用栈。Worker 的 return 回调即使未命中所关注函数，也会获取锁并查询 active 表。由此带来的 GIL、线程竞争和写入扰动属于合理机制解释，尚未单独量化，不能声称已证明某一把锁就是唯一根因。

### 11.5 后续验证顺序

1. 保留现有完整 timeline 用于结构定位；性能定量另做短窗口轻量采集，关闭 Worker Python 回调和 EngineCore 完整栈，保留设备与必要的少量阶段标记。
2. 对齐相同请求 / token / 上下文长度，检查关闭重型采集后：调度步间隔、collective 到达差、stream2 等待占比是否同步下降。
3. 若主机进度差仍明显，优先验证 Worker CPU 绑定、线程竞争、输入 / block table 准备和 metadata 更新；不能把扩大 batch 或专家均衡视为已证实的修复。
4. 若轻量采集仍显示同层专家 GMM 时长跨 rank 明显不同，再推进专家负载均衡。若所有 rank 近乎同时进入 collective 但通信仍长，再研究通信算法与带宽。
5. 组合实验后来确定使用 `scheduler-reserve-full-isl=false`、16384 token 预算和共享专家 DP；当前启动脚本又将投机长度改为 5，见第 12 节。这些改动尚未完成八卡正式评测，也没有证据表明它们能解决上述主机进度差。

证据限制：跨卡时间依赖 CANN 导出时钟；同一 collective 的结束时间存在毫秒级差异，不对亚毫秒先后做精确因果断言。collective 按完整名称及 count / dtype 匹配；所选六例为长尾样本，不代表总体均值。所有推测均需轻量采集验证。

这些数据由归档前的八卡 CANN 导出和 Worker 调用记录统计。依赖分析 JSON、统计脚本与原始 CANN 文件现已清理；可查看的最终 trace 保存在 [timeline 目录](logs/timeline/) 的 `dspark7` 文件中。

### 11.6 草稿路径与 baseline 对照补充

本轮采集对应的 DSpark 路径在 `frameworks/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py` 设置 `self.use_cuda_graph = False`。因此目标模型配置 `FULL_DECODE_ONLY` 不代表草稿模型也通过完整 ACLGraph replay 执行。Worker trace 中可见 `_run_merged_draft`、`DSparkDeepseekV4ForCausalLM.forward` 下的 DSA 与 MoE 调用，与该路径一致。

这给出合理的机制解释：草稿模型虽然层数少，但其逐算子主机调用、前后处理和输入 metadata 更新会暴露主机开销；各 rank 的提交进度差会在后续目标模型的 collective 处转化为等待。它是当前证据支持的路径解释，不能仅凭配置断言所有差异由 eager 引起。

使用同样负载签名匹配方法检查未开启 DSpark 的 baseline：

| 负载 | 未采集间隔中位数 | 采集间隔中位数 | 样本数 off/on |
| --- | ---: | ---: | ---: |
| D1，20 请求，20 decode tokens，各 prompt 32768 | 47.43 ms | 48.34 ms | 432 / 622 |
| D2，4 请求，4 decode tokens，各 prompt 98304 | 33.41 ms | 36.19 ms | 190 / 8 |
| D2，4 请求，8192 tokens，prefill 8189，各 prompt 98304 | 1111.62 ms | 2223.92 ms | 20 / 4 |

baseline 的部分 decode 分组变化较小，prefill 与 DSpark 分组的变化明显。这与不同执行路径对主机回调的敏感程度不同相符，但两轮不是严格同条件实验，D2 baseline 的部分采集样本也少，不能据此计算跨版本加速比。原始统计 JSON 已清理。

组合评测曾同步远端并提交 Slurm 作业 15744。提交时配置包含 DSpark 7、`max-num-seqs=24`、`gpu-memory-utilization=0.92` 及三项合并修改，使用已有 `image.sif`，没有修改 `build.def`。作业最初为 `PENDING (Priority)`，之后在分配节点前出现 `NODE_FAIL`；没有产生该组合的正式评测结果。旧产物目录和控制器记录已清理，最新启动脚本也已修改，见第 12 节。

### 11.7 4096 token 预算的适用范围核查

按 DSpark D1/D2 实际约 30 秒采集窗口、调度步等权统计。这里的 scheduled tokens 是单步安排的 token positions；包含投机验证位置，不是最终接受的输出 token 数，也不是八卡相加。

| 负载 / 步类型 | 步数 | 平均请求数 | 平均 scheduled tokens / step | token 中位数 |
| --- | ---: | ---: | ---: | ---: |
| D1 全部 | 17 | 16.65 | 5833.88 | 8000 |
| D1 mixed | 14 | 16.71 | 7056.00 | 8000 |
| D1 decode | 3 | 16.33 | 130.67 | 128 |
| D2 全部 | 85 | 3.46 | 364.33 | 32 |
| D2 mixed | 7 | 3.86 | 4118.86 | 4120 |
| D2 decode | 78 | 3.42 | 27.38 | 24 |

D1 有 12/17 步实际调度 8000 tokens，因此 4096 会真实限制这些大 prefill / mixed 步，可能增加 chunk 数、主机准备次数和同步次数；不能称为确定的性能优化。D2 mixed 步的 prefill 恰为 4096，但总预算还包含其他请求的 decode / 验证位置（通常另加 24），因此全局 max-num-batched-tokens=4096 也不是完全无影响。

纯 decode 样本中，D1 每步约 131 tokens、D2 每步约 27 tokens，均远低于 4096；这些低 token batch 不能归因于将预算设为 4096。它们的每请求调度位置为 8，与本轮 7 个投机 token 配置相符，但不代表每请求实际接受 8 个输出 token。

这是受采集扰动影响的短窗口，不是完整评测平均。4096 只适合作为 TTFT/TPOT 权衡实验，不能作为已验证的默认值。组合配置后来改为 16384；4096 未执行正式评测，原始调度统计文件已清理。

### 11.8 组合评测更新为 16384

DeepSeek 的 `max-num-batched-tokens` 后来改为 16384，并保留 `scheduler-reserve-full-isl=false` 和 `enable_shared_expert_dp=true`。作业 15744 排队期间曾同步这版配置，但随后在分配节点前 `NODE_FAIL`，没有产生评测结果。当前脚本进一步采用 DSpark 5 和 local argmax；同样没有八卡正式成绩。后续验证需要同时检查 D1 TTFT、D1/D2 TPOT、显存和正确性门槛。

## 12. Ascend DSpark local argmax 实现与验证状态

针对 DeepSeek-V4 DSpark，新增 [本地 argmax 实现](frameworks/vllm-ascend/vllm_ascend/spec_decode/dspark_local_argmax.py)。每个草稿位置沿用原有 Markov bias 计算，再在各 rank 的词表分片上取最大值；跨 TP rank 只收集 FP32 分数和 token ID 候选，按全词表 argmax 的顺序处理并列最大值，并屏蔽 padding 词表项。Markov 权重和完整 bias 投影保持原路径。它只支持普通 TP 词表分片；独立 LMHead TP group 会显式报错，词表大小还需落在 FP32 可精确表示的 token ID 范围内。

release 分支 `hpc_dev` 的实现已推送为 `cff0538`。随后将代码迁移到基于官方 main 提交 `9df651d0b` 的 `feat/dspark-local-argmax-main`，推送为 `b85954bb0`；当前 `frameworks/vllm-ascend` 工作树位于该分支。新版适配了接口及动态验证长度更新，尚未创建上游 PR。此前临时使用的 `submission/python/sitecustomize.py` 覆盖目录已删除；用户重编镜像后，启动脚本从镜像内获取源码。`build.def` 未因这项改动而修改。

release 实现曾通过本地 CPU/Gloo 的 TP1/2/8、FP32/BF16、三类输入共 18 组检查，覆盖 padding、跨 rank 并列最大值、大词表 ID 和连续七步 Markov 修正；独立主分支测试后来通过 24 项 CPU 与源码隔离检查及 Ruff 检查。完整仓库测试在本地未通过依赖加载阶段。上述结果都不是八卡 NPU/HCCL 的端到端正确性或性能验证。

当前 [启动脚本](submission/start.sh)配置 DSpark 5、`use_local_argmax_reduction=true`、`enable_flashcomm1=false`，同时设置 16384 token 预算、`scheduler-reserve-full-isl=false` 和共享专家 DP。它不同于第 10 节的 DSpark 7 正式评测配置，也没有新的 NPU 分数。每个草稿位置仍需一次小 collective；减少收集的字节数不等于降低了端到端延迟，更不能据此认定已消除 stream WAIT。

## 13. 最终 Perfetto 文件与设备空档核查（2026-09-28）

已在本地完成无 DSpark 补采 baseline 的最终转换，没有重新申请 NPU 或运行评测。[timeline 目录](logs/timeline/) 中的 `no-dspark` 文件为 D1/D2 各 rank0 至 rank7，共 16 份、7.84 GiB。另有 16 份 `dspark7` 修正版，设备窗口约 30.000 至 30.047 秒，归档前验证记录显示设备事件时长错误为 0。每份文件保留同一 rank 的合并 timeline，包含已有 Worker Python 调用记录、EngineCore、CANN、设备 stream、通信及计数器数据；prefill 与 decode 没有拆成两个文件。EngineCore 数据在各 rank 文件中重复，不应跨文件相加。

| 验证项 | 最终结果 |
| --- | ---: |
| 完成文件数 | 16/16 |
| 设备采集窗口 | 30.053–30.723 秒 |
| 校验设备事件数 | 32,392,052 |
| 设备时长误差 | 0，按纳秒精确匹配 |
| 已标注设备起始时间误差 | 0 |
| Perfetto 导入错误 | 0 |
| 导入 slice / counter / flow 总数 | 353,201,528 / 480,735 / 1,390,296 |

以上总数用于文件完整性核对，包含各文件重复的主机记录，不代表去重后的系统活动量。完整 trace 的主机时间跨度可能超过约 30 秒设备窗口。

此前失败来自同一 stream 的部分原始设备区间交叉重叠：按单一栈编码 begin/end 会配错结束事件，改变 Notify Wait、Write Value、SDMA 等事件的时长。最终版将重叠区间放到附加展示行，保留原始起止时间、属性和物理 stream 归属；没有删除 WAIT，也没有放宽验证阈值。`Stream N [overlap K]` 是同一物理 Stream N 的展示分行，分析利用率应取区间并集，不能把这些行的时长直接相加。

原有 flow 数据保持不变，但没有重新分配到附加展示行，因此箭头本身不足以证明某个重叠设备事件的等待依赖。这次转换修复不产生新的吞吐或性能瓶颈结论。

可用本地 `trace_processor --httpd` 直接打开最终 `.pb.gz`，无需解压。两套共 32 份最终 Perfetto 在分类整理时再次通过 SHA-256 核对。按照最新归档要求，pending、合并 JSON、原始采集、CANN 导出、算子表、验证 JSON 和其他中间产物均已删除。文件名标出日期、框架版本、DSpark 状态、D1/D2 与 rank。

### 13.1 D1 rank0 的跨 stream bubble 核查

针对上述无 DSpark baseline，从原始 CANN 设备事件重新计算区间并集，裁剪到 `capture_started` 至 `capture_window_finished` 的 30.061942 秒窗口，避免把单条 stream 的空白或跨 stream 重叠重复计数。

| 事件范围 | 时间并集 | 窗口覆盖率 |
| --- | ---: | ---: |
| Stream 664、667 的计算事件 | 22.066074 s | 73.40% |
| 全部 stream 的计算事件 | 23.169650 s | 77.07% |
| 全部计算加 AI_CPU、SDMA、MEMCPY_ASYNC | 24.653124 s | 82.01% |
| 全部正时长 Hardware 事件，包含等待和控制事件 | 29.911168 s | 99.50% |

计算口径为 AI_CORE、AI_VECTOR_CORE、MIX_AIC、MIX_AIV。664 与 667 的计算累计时长和并集非常接近，支持用户观察到的交替互补执行；但仍未覆盖整个窗口。

按第三行口径，没有这些工作事件覆盖的时间共 5.408818 秒（17.99%），分散成大量短间隙。达到 1 ms 的间隙仅 11 段、共 53.213 ms，占整个窗口约 0.18%；最长一段约 31.430 ms。约 99% 的累计间隙时间来自不足 1 ms 的间隙，缩小 timeline 后容易显得连续。最长间隙位于采集窗口起点后 28.379183–28.410613 秒；该偏移不是 UI 的全 trace 零点。

包含等待后，真正没有任何正时长 Hardware 记录覆盖的时间仅 0.150774 秒（0.50%）。因此这份 trace 没有大量长时间、所有 stream 都空白的 bubble；存在的是累计可观的细碎计算空档，其中多数仍有等待或控制记录。不能把 17.99% 全部解释为物理设备彻底空闲，也不能据此直接归因于 CPU 或 HCCL。以上为单 rank 的采集窗口统计，不是八卡平均或硬件利用率采样。

上述结果来自清理前的原始 CANN 数据；原始数据、统计 JSON 和复现脚本已按归档要求删除，最终 timeline 仍在 `logs/timeline/`。

### 13.2 DSpark D1 rank0 的相同口径核查

对 `profile-safe-stage2` 的 D1 rank0 原始 CANN 数据执行相同区间并集分析。该采集使用 DSpark 7、FULL_DECODE_ONLY，不是后续未验证的 DSpark 5 配置。窗口为 30.035692 秒。

| 口径 | 无 DSpark D1 rank0 | DSpark D1 rank0 |
| --- | ---: | ---: |
| 所有 stream 计算覆盖率 | 77.07% | 23.91% |
| 加入 AI_CPU、SDMA、MEMCPY_ASYNC | 82.01% | 24.17% |
| 包含等待及控制的全部 Hardware 覆盖率 | 99.50% | 72.94% |
| 无上述计算/搬运的 ≥1 ms 间隙 | 11 段 / 0.053 s | 4,647 段 / 18.794 s |

DSpark 的主要计算 stream 是 2（计算并集 6.051 s）和 16（1.074 s），二者并集为 6.822 s；另有 Stream 13 和部分 126–136 stream 的计算。纳入全部 stream 后仍存在大量毫秒级无计算/搬运区间，不能用两条主 stream 的交替执行解释掉。最长约 44.590 ms，位于本次采集起点后 14.965632–15.010222 秒。等待和控制记录不等于有效计算，约 27.06% 的窗口甚至没有正时长 Hardware 事件覆盖；这仍不是硬件利用率采样。

这个对照不能用来定量归因 DSpark 的真实开销：baseline D1 采集主要为纯 decode（约 619 个调度步），DSpark D1 为 mixed/prefill 占多数的窗口（17 步），而且完整调用采集在 DSpark 上扰动显著。已有同配置调度步对照中，DSpark 的 16/17 请求、8000 token 步在采集期间的中位时长约为非采集期间的 2.77/2.74 倍。因此这些 gap 是该采集里的真实事件空档，但不能直接推断正常服务也只有 24% 计算/搬运覆盖率，或全部由通信造成。

上述结果来自清理前的原始 CANN 数据和采集扰动对照；统计 JSON 与原始数据已按归档要求删除，最终 timeline 仍在 `logs/timeline/`。
