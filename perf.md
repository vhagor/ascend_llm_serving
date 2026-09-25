# DeepSeek 八卡推理性能评估（阶段性结果）

数据日期：2026-09-25，时区 Asia/Shanghai。本文基于作业 7554 的第三次启动，即修复缓存目录后的运行。**D1、D2 完整完成；GSM 未完成，因此尚无有效的完整 Stage2 分数。** 后续重跑结果不包含在本报告中。

## 1. 环境与推理配置

| 项目 | 本次运行 |
| --- | --- |
| 节点 | ascend3 |
| NPU | 8 × Ascend 910B3，每卡 64 GiB HBM |
| 卡间拓扑 | 八卡之间均为 HCCS |
| 主机资源分配 | 192 CPU，768 GiB 内存 |
| 驱动 / CANN | 25.2.3 / 9.1.0 |
| 框架 | vLLM 0.26.0，vLLM-Ascend 0.26.0rc1 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post4 |
| 模型 | DeepSeek-V4-Flash-0731-w8a8 |
| 并行 | TP=8，DP=1，启用 EP；1 个 EngineCore、8 个 Worker |
| 数据类型 / 量化 | bfloat16 / ascend W8A8 |
| 最大上下文 | 102400 token |
| 最大序列数 | 32 |
| 每步 token 预算 | 8192 |
| 显存利用率配置 | 0.90 |
| KV block size | 128 |
| Chunked prefill / Prefix caching | 均开启 |
| 投机解码 / CPU binding | 均未开启 |

日志中每卡权重约 36.20 GiB，KV cache 约 16.65 GiB，图缓存约 0.66 GiB。这些是框架内存统计，不能与驱动报告的总 HBM 占用直接等同。

更准确地说，日志中的“Loading model weights”是加载前后框架分配内存的增量，不是磁盘权重字节数。虽然日志写的是 GB，源码使用 `2**30` 换算，实际单位是 GiB。核对 74 个 safetensors 文件头后：

| 统计口径 | 大小 |
| --- | ---: |
| safetensors 文件总大小，含文件头 | 293.026 GiB（314.634 GB） |
| 全部张量数据 | 293.014 GiB |
| 其中 `mtp.*` 投机预测权重 | 21.857 GiB |
| 扣除 MTP 的主模型张量数据 | 271.157 GiB |
| 日志设备内存增量，8 × 36.1983 GiB | 289.586 GiB |

当前未开启投机解码，主模型加载器明确跳过 `mtp.*`。同时，TP/EP 不是把所有张量机械地除以 8：模型使用了 `ReplicatedLinear`，还有量化权重布局转换、scale 副本及模型初始化缓冲区。因此文件总大小与八卡加载内存不会严格相等。八卡加载增量比主模型文件张量多约 18.429 GiB；**尚未逐张量核算这部分，不能将其全部归因于某一个因素**。KV cache 和图捕获在该加载统计之后，不能用来解释这项差额。

证据：[文件头字节统计](logs/stage2-2026-09-25/weight-accounting.json)、[模型加载器及 MTP 跳过逻辑](frameworks/vllm-ascend/vllm_ascend/models/deepseek_v4.py)、[加载阶段内存统计](frameworks/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)、[内存差值计算](frameworks/vllm/vllm/utils/mem_utils.py)。

证据：[服务日志](logs/stage2-2026-09-25/baseline/output/logs/service.log)、[运行配置快照](logs/stage2-2026-09-25/baseline/submission/start.sh)、[硬件记录](logs/stage2-2026-09-25/baseline/hardware/)。后续重跑开始后，远端本轮目录会归档为 `baseline-attempt3-allocation-timeout/`；本文相对链接指向此次同步到本地的快照。

## 2. 性能与正确性

| 指标 | D1 正式 | D2 正式 |
| --- | ---: | ---: |
| 完成请求 | 60/60 | 144/144 |
| 客户端并发上限 | 20 | 4 |
| 阶段耗时 | 862.96 秒 | 1385.33 秒 |
| 输出吞吐 | 284.79 token/s | 13.31 token/s |
| p99 TTFT | 83.64 秒 | 55.04 秒 |
| TTFT SLA | 96.40 秒 | 60.50 秒 |
| p99 TPOT | 70.59 毫秒/token | 344.73 毫秒/token |
| TPOT SLA | 74.00 毫秒/token | 380.00 毫秒/token |
| 数学题 | 3/3，通过 | 3/3，通过 |

两项负载的数学题和 p99 延迟门槛均符合要求。正式吞吐包含整个正式阶段耗时，不能用服务日志里稳定 decode 时约 414–420 token/s 的瞬时吞吐替代 D1 正式吞吐。

GSM 已完成 **555/1319**，其中 **537 正确、18 错误**，已完成子集正确率 **96.76%**；报告记录截断数为 0。门槛要求完整测试至少 **1212/1319** 正确，剩余 764 题至少还需答对 675 题。子集结果没有显示明显精度异常，但不能证明完整 GSM 门槛通过，也不能直接外推最终正确数。

`report.json` 中的 GSM `accuracy=0.4071` 是 `537/1319`，包含未完成题目的总分母；本文 96.76% 是 `537/555`，两者口径不同。本次 Qwen 未重新测试。

本轮因 Slurm 分配即将到期而主动发送 SIGTERM，让 evaluator 保存中间报告。报告中的 `failed`、`traffic-failure` 和零分反映整轮未完成，不代表 D1/D2 数学题错误或模型崩溃。**本报告不发布完整 Stage2 baseline 分数。**

证据：[完整请求报告](logs/stage2-2026-09-25/baseline/output/report.json)、[阶段汇总](logs/stage2-2026-09-25/progress-summary.json)、[评分规则](README.md)。

## 3. NPU 通信带宽与利用率

### 3.1 理想上限及计算口径

华为 Atlas 800I A2 官方材料给出 HCCS Full Mesh **双向互联带宽 392 GB/s**。本报告将其作为 A2 八卡形态下的**单 NPU 七链路聚合名义参考上限**，按对称全双工推导：

| 参考口径 | 理想值 |
| --- | ---: |
| 单链路双向合计 | 56 GB/s |
| 单链路单方向 | 28 GB/s |
| 单 NPU 七链路发送合计 | 196 GB/s |
| 单 NPU 七链路接收合计 | 196 GB/s |
| 单 NPU 七链路 Tx+Rx 合计 | 392 GB/s |

**这是规格参考及其推导，不是 ascend3 实测峰值。** 本次已确认 910B3 和八卡 HCCS 拓扑，但没有取得该节点整机型号、链路协商速率或独立 P2P/HCCL 峰值测试。因而下表百分比均为“相对上述名义上限的估算”，不能认定为已校准的物理链路利用率。若节点实际链路规格不同，必须更换分母后重算。

资料：[华为大模型推理解决方案官方彩页](https://e.huawei.com/marketingcloud/pep/asset/20000001/Material/744857c25fe4438b97c85ed15fd79d1a/M3T1A669N1162432897309413660/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88R24_0-%E5%BD%A9%E9%A1%B5_25H2_.pdf)。本次检索可见官方索引中的 392 GB/s 双向规格，但 PDF 全文抓取超时；单 NPU 聚合口径仍应结合整机白皮书复核。

### 3.2 实测通信与计算利用率

采样发生在 **22:21，D1 稳定解码阶段**。三轮采样起始时间分别为 22:21:07.821、22:21:13.650、22:21:19.466；每轮各卡并行执行查询，每卡 HCCS 采样窗口为 1000 ms，下表为三次等权平均。各卡窗口接近但不是严格同步，不能代表整个 D1，更不能代表 D2/GSM。

HCCS 查询命令为：

```bash
npu-smi info -t hccs-bw -i 0 -c 0 -time 1000
npu-smi info -t usages -i 0 -c 0
```

官方文档将 Tx/Rx 定义为指定芯片各 HCCS 链路的发送/接收带宽，`total` 为链路总和，单位 GB/s。[命令说明](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A2/A2npu/npusmi_0079.html)

| NPU | AI Core | AI Vector | HBM 带宽利用率 | HCCS Tx GB/s | HCCS Rx GB/s | 估算双向带宽利用率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 42.00% | 46.00% | 26.00% | 3.081 | 3.140 | 1.587% |
| 1 | 42.00% | 46.67% | 26.00% | 0.548 | 0.539 | 0.277% |
| 2 | 41.67% | 46.33% | 25.67% | 0.548 | 0.540 | 0.277% |
| 3 | 40.33% | 46.67% | 25.33% | 0.546 | 0.538 | 0.277% |
| 4 | 41.00% | 46.00% | 25.67% | 0.548 | 0.539 | 0.277% |
| 5 | 39.67% | 46.00% | 25.67% | 0.548 | 0.539 | 0.277% |
| 6 | 39.67% | 46.00% | 25.67% | 0.546 | 0.538 | 0.277% |
| 7 | 41.67% | 45.67% | 26.00% | 0.545 | 0.536 | 0.276% |

计算式：双向利用率 = `(Tx + Rx) / 392 × 100%`；若单独计算发送或接收利用率，分母分别使用 196 GB/s。不能把单方向流量直接除以双向上限。卡 0 的 Tx、Rx 利用率约为 1.57%、1.60%。

采样时各卡 HBM 容量占用约 93%。**显存容量占用、HBM 带宽利用率、AI Core 利用率是三个不同指标**；AI Core 利用率也不等于峰值 FLOPS 达成率。

卡 0 的计数器流量明显高于其他卡，但目前没有 trace 证明是哪个通信算子或哪种通信拓扑导致。不能把链路编号直接当作远端 rank；将全机 Tx+Rx 相加还会重复统计发送端和接收端，不宜据此报告全机有效业务流量。

原始数据：[三轮八卡利用率与 HCCS 采样](logs/stage2-2026-09-25/baseline/hardware/live-utilization-hccs.json)、[全程设备采样](logs/stage2-2026-09-25/baseline/hardware/npu-samples.log)。未运行独立通信压测，避免改变 baseline 负载。

### 3.3 能得出的结论

八卡的粗粒度 Core 利用率较均衡，约 40%–42%。短窗口平均通信带宽相对名义上限较低，**尚无链路持续饱和的证据，但也不能排除通信延迟瓶颈**。频繁小消息、rank 等待、CPU 下发间隙、算子粒度及计算通信重叠，都需要 trace 才能区分。不能用“低平均 GB/s”推断“通信开销低”，也不能根据 40% 利用率推断有 2.5 倍性能提升空间。

## 4. 实际并发、batch size 与 token 长度

### 4.1 是否测过真实 batch size？

**没有采集逐个 scheduler step 的真实 batch size。** 已有的是客户端在途并发记录，以及约每 10 秒一次的 vLLM `Running/Waiting` 日志。`Running` 是调度器运行请求池的快照，不等于该次模型 forward 真正参与的序列数；prefill/decode 混合时尤其不能混用。

对明确落在各正式阶段内的日志子窗口统计如下。平均值仅是日志样本均值，不是逐步 batch 平均，也不是完整阶段的时间加权平均：

| 日志子窗口 | 样本数 | Running 范围 | Running 样本均值 | Waiting 最大值 |
| --- | ---: | ---: | ---: | ---: |
| D1，22:08:00 ≤ t < 22:21:00 | 78 | 12–20 | 19.37 | 8 |
| D2，22:23:00 ≤ t < 22:44:00 | 126 | 1–4 | 3.26 | 3 |

客户端 D1 并发上限为 20，D2 为 4；`max-num-seqs=32` 只是引擎上限，不会自动补齐到 32。TP=8 表示八卡共同执行这批请求，不能乘成 160 或 32 个独立请求。

在所有请求都做普通 decode、无投机解码时，每个活跃序列通常贡献一个新 token；例如 4 个序列可对应约 4 个有效 decode token/step。这是执行机制说明，**不是本轮逐步实测值**。图捕获可能使用 padding 后的形状，kernel 张量维度也不能直接当作有效 batch size。

### 4.2 已测得的请求级 token 长度

以下均值从 `report.json` 中已成功完成请求的 `prompt_tokens` 和 `completion_tokens` 计算，不使用字符数，也不将输出上限当成实际输出。它们是**每请求均值，不是每 batch 均值**。

| 请求集合 | 样本数 | 平均输入 token | 平均输出 token | 输出 token 范围 |
| --- | ---: | ---: | ---: | ---: |
| D1 正式 | 60 | 32768.00 | 4096.00 | 4096–4096 |
| D2 正式 | 144 | 66389.33 | 128.00 | 128–128 |
| GSM 已完成子集 | 555 | 84.97 | 98.33 | 36–291 |

D2 各轮分别有 24 个请求，输入 token 数依次为 **2048、32768、65536、98304、99328、100352**，每次输出 128 token。GSM 输出上限是 1024，但已完成子集实际平均只输出 98.33 token。

“每个 batch 的平均输入长度”还需明确是参与请求的完整 prompt 长度，还是该步实际执行的 prefill chunk 长度；“decode 长度”也需区分完整回答长度与该步 decode token 数。**本轮没有 batch 成员和逐步 token 记录，因此无法计算这些 batch 级指标。** 将请求均值乘以客户端并发，不能还原真实混合 batch。

### 4.3 Chunked prefill 已开启

服务日志第 19 行明确记录：

```text
Chunked prefill is enabled with max_num_batched_tokens=8192.
```

EngineCore 初始化日志也记录 `enable_chunked_prefill=True`、`enable_prefix_caching=True`。因此 32K–98Ki prompt 可以跨多个调度步处理，单步 token 总预算为 8192，并与该步其他请求的 token 共享预算；不是每个请求都能独占 8192。Prefix cache 命中又会减少需要重新计算的输入，不能简单用完整 prompt 长度除以 8192 推算精确步数。

源码依据：[调度器](frameworks/vllm/vllm/v1/core/sched/scheduler.py) 中 `token_budget`、`num_scheduled_tokens` 和 `max_num_running_reqs`；[压测请求池](dpsk_benchmark.py) 中 `run_pool`。当前未开启 `enable_logging_iteration_details`，不能从现有日志补出逐步 batch。

## 5. 耗时与后续采集范围

本轮排除前两次排错后耗时 **63 分 12 秒**：冷启动约 11分59秒、D1 预热 5分43秒、D1 正式 14分23秒、D2 正式 23分05秒、部分 GSM 8分钟。按已完成 GSM 的速度线性估计，完整流程约 74 分钟；建议预留 75–80 分钟，不含排队。这不是完整运行的实测耗时。

后续先完成完整 baseline 并确认正确性，再进行抽样 profiling：固定随机种子选择 20 个 D1 请求和 4 个 D2 session 的六轮，共 44 个请求，保持原始输入/输出长度和并发，不重复完整 GSM。当前报告的数据快照中尚无 profiler 产物。

要回答 batch 和瓶颈问题，后续仍需采集：

1. EngineCore 每步参与请求 ID、有效序列数、prefill/decode token 数和完整 prompt 长度，按 step ID 关联。
2. 八个 Worker 的实际执行 token 数、padding 后形状、rank/PID/设备映射及 stream。
3. CPU 下发、NPU 算子、HCCL 通信、同步等待和重叠时间；仅凭 kernel trace 不一定能恢复请求的语义长度。
4. 在规格确认后，另行测量独立 P2P/HCCL 可达峰值，区分名义带宽、实测可达带宽和业务平均带宽。

现有 profiling 脚本准备了 EngineCore CPU 标注与八卡 profiler，**尚未证明能导出上述所有逐步 batch 字段**；这些字段需要额外的 scheduler 元数据采集。不能将计划采集的内容写成本次已经测得的结果。
