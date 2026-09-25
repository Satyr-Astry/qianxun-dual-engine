# DeepSeek-V4.1-Flash 可移植亮点分析（面向"12 GB 单机提速"）

> 论文：**DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression**，arXiv **2609.19969**（2026-09-17 提交），
> 官方技术报告 + HF 模型卡。**本文所有论文侧事实均来自实证抓取**（arXiv abs 页 + 模型卡 + 架构解读），非记忆。
> 本机侧数字均出自本项目实测（见 `N7N8_报告.md` / `N9_报告.md`）。

---

## 一、为什么这篇对我们特别重要：**它的动机就是我们的瓶颈**

论文摘要原话（逐字）：
> "The widespread adoption of long-horizon agents has made model workloads increasingly input-heavy.
> **Although prior work has substantially reduced the cost of long-context computation, prefill remains
> computationally expensive**, and large KV caches continue to strain HBM and SSD capacity and
> data-transfer bandwidth."

我们的实测瓶颈（同一件事的两个面）：
| 本机现象 | 实测 |
|---|---|
| **prefill 昂贵**（首 token 慢） | Agent 系统提示 10K token ⇒ **首 token ≈18 s**（prefill 555 tok/s） |
| **大 KV 压榨容量** | 64K 上下文要 `q4/q4 KV`（3264 MiB）+ **16 层 FFN 卸到 CPU** 才装得下 ⇒ decode 掉到 10.5 tok/s |
| **带宽是硬墙** | 短上下文 decode 46.5 tok/s = **418 GB/s = 504 GB/s 峰值的 83%** |

⇒ 论文的解法方向，正好是我们需要的。

---

## 二、论文侧关键事实（已实证）

| # | 机制 | 论文数字 |
|---|---|---|
| 1 | **CED（Causal Encoder-Decoder）**：40 层 = 20 层因果编码器 + 20 层解码器；**解码器的全局 KV 由编码器末层隐状态投影而来**，不再由每个解码层自己产生 | **prefill 只激活 8B/token，decode 16B/token**（非对称激活） |
| 2 | **CSA2**：每层静态指定 **Full / Reindex / Reuse** 三模式之一，**跨层共享主 KV 与 indexer K**，并**复用 Top-K 稀疏索引** | — |
| 3 | **Hierarchical Sparse Indexer**：后续索引层只在"首个 Full 层给出的候选池"里做索引 | **索引成本与上下文长度解耦** |
| 4 | **FP4 KV 缓存**（E2M1，每 16 通道一个 E4M3 scale） | 全局 KV **890 字节/token**（≈ V4-Flash 的 1/4） |
| 5 | **SWA Bounded Replay**：只重放最近 `n_win` 个 token 来**重建**缺失的 SWA KV，不必把 SWA KV 持久化到 SSD | 持久 KV 降到 V4-Flash 的 **~1/8** |
| 6 | **DSpark 投机解码**：半自回归 draft + **置信度调度的验证** | — |
| 7 | **Engram 条件记忆**：196B 参数，**按 token 查表稀疏访问** | — |
| 8 | Single-Pass mHC（Mega-mHC 核）/ Muon / MoE 1 shared + 384 routed（激活 6） | 45T token 多模态预训练 |

---

## 三、逐条可移植性（★= 对本机提速的性价比）

### ★★★★★ 1. CED 的「非对称激活」思想 —— **最大杠杆，且我们有现成的验证工装**
**论文的洞见**：prefill 慢是因为"每个解码层都要算自己的 KV"；CED 让 KV 由编码器末层**投影一次**，
⇒ **prefill 与 decode 的激活量彻底解耦**（8B vs 16B）。

**我们的对应迁移（不需要重训整个世界）**：**跨层 KV 共享**
- 本机现状：48 层各自算 KV ⇒ 每 token **102 KiB（q8）/ 51 KiB（q4）**
- 若按 **每 4 层共享一份 KV**（第 4/8/…/48 层算，其余层复用）：KV 直接 **÷4**
- 代入我们的显存菜单：
  - 64K 上下文 KV：3264 MiB → **816 MiB**
  - 于是 **16 层 FFN 可以搬回 GPU**（省 2066 MiB）仍有富余 ⇒ 总占用 ≈ 8148 + 816 + 325 = **9289 MiB**
  - ⇒ **回到实测"健康区"**：prefill **~2000 tok/s**、decode **~37.8 tok/s**
  - ⇒ **首 token 18 s → ~5 s，decode 10.5 → ~38 tok/s（3.6×）**

**代价与做法**：跨层共享 KV 会让质量下降（各层不再有自己的 KV），需要用**轻量适配**补回来 ——
而我们**已经有这套工装**：`code/lora_readout_sft.py`（读出侧 LoRA）+ `tools/memcore_harness.py`（六项检查）。
⇒ **N11 实验**：在 PyTorch 侧做"4 层一组共享 KV + LoRA 适配"，用 `memcore_harness` 量化它是否满足"功能等价"判据。
（llama.cpp 目前**不支持**跨层 KV 共享，所以落地要等 kernel 或走我们自己的推理栈 —— 这条要在文档里写清。）

### ★★★★★ 2. SWA Bounded Replay —— 就是我们计划书里的「重算代替搬运」
论文用"只重放最近 `n_win` 个 token"来重建 SWA KV，**避免把 KV 持久化到 SSD**。
这与我们计划书的 **KVPR（KV Paging by Recomputation）** 是同一个思想，而且给出了**上界规则**：
> **只有滑窗部分需要重放，全局部分靠"投影/压缩"一次拿到。**

⇒ 直接写进我们的设计：重算预算 = `n_win`，不随上下文线性增长。
⇒ **可测**：本机 decode vs "重算前 n_win 个 token"的耗时对比（llama.cpp 不给这个开关，但在我们自己的栈里可测）。

### ★★★★☆ 3. CSA2 的三模式（Full/Reindex/Reuse）+ 索引成本解耦
"每层静态选一种模式、跨层复用主 KV 与索引 K、复用 Top-K 索引" —— 本质是**用层间的冗余换显存与算力**。
对我们的启示（= 我们架构书"分层记忆体 + 写入门控"的同构解）：
- 不必每层都"全注意力"：**一部分层做全量、一部分层复用**（静态分配，无需训练即可先验证）
- **索引成本与上下文解耦**：这是我们 T12（评估侧记忆检索）必须遵守的原则，否则检索层会变成新的瓶颈

### ★★★★☆ 4. FP4 KV（890 字节/token）—— **思想可用，内核不可用**
对比：我们的 q4/q4 KV = **51 KiB/token**，论文 **890 B/token** ⇒ **约 57× 更小**
（注意口径不同：他们全局 KV 是压缩后的稀疏结构 + 384 专家 MoE）。
- **能用的部分**：我们已经把 KV 压到 llama.cpp 允许的最低（`q4_0`），**每 32 元素一个 scale**；
- **不能用的部分**：`FP4(E2M1) + 每 16 通道 E4M3 scale` 需要专用内核 ——
  llama.cpp build 10938 **没有** FP4 KV 路径。
⇒ 结论写死：**这条是"内核决定一切"（我们已记过：量化 ≠ 提速，内核差 3.59×）**，
除非有 FP4 KV 内核，否则我们在 KV 上已接近本机可行下限。

### ★★★☆☆ 5. DSpark：置信度调度的投机解码 —— 可移植到我们的 MTP 线
我们的 N8-S1 已自训 MTP 头（α₁ 0.34→0.44）。论文加的是：
**draft 半自回归 + 按 draft 置信度决定验证多少**。
- llama.cpp 的 `--spec-draft-n-max` 是**固定值**；我们可以**先测**"接受率随 draft 长度的曲线"，
  再给一个**自适应调度**（置信度高就多投机，低就少投机）—— 这在服务层就能实现（虽然改不了 llama.cpp 内部，
  但可以在我们的栈里验证收益，并作为"机制结论"写进论文规划）。
⇒ **N8-S2 实验**：测 α₁(1..k) 曲线 + 置信度阈值 → 换算成期望加速比。

### ★★★☆☆ 6. Engram 条件记忆 —— 我们"主动知识库"的同行背书
196B 参数、**按 token 查表稀疏访问** ⇒ 与我们"分层记忆体 / 主动知识库 / 静默态"的取向一致，
可作为**引文锚点**（写进 `论文规划.md` 的相关工作）。

### ★☆☆☆☆ 7/8. mHC、Muon、多模态 ViT
- **mHC / Muon**：训练侧（收敛与稳定性），**不提升推理速度**；对"自学习轴"有参考价值，与本次目标无关。
- **视觉编码器**：与我们的文本提速无关。

---

## 四、落到我们项目上的结论（可直接执行的顺序）

| 优先 | 动作 | 预期（基于本机实测表换算） | 依赖 |
|---|---|---|---|
| **1** | **跨层 KV 共享 + 读出侧 LoRA 适配**（CED 思想的朴素实现） | 首 token **18 s → ~5 s**、decode **10.5 → ~38 tok/s** | 我们自己的推理栈（llama.cpp 不支持） |
| **2** | **SWA Bounded Replay 式重算**（重算窗口 = n_win，不随上下文增长） | 省掉持久 KV 搬运，长上下文 decode 不再塌 | 同上 |
| **3** | **投机解码的置信度调度** | 在 α₁=0.44 基础上把有效加速从 ~1.3× 往 **1.5×+** 推 | llama.cpp 固定 draft，需自测曲线 |
| 4 | 层间"全量/复用"静态分配（CSA2 三模式的朴素版） | 进一步省 KV，质量需量化校验 | 工装已有 |
| ⛔ | FP4 KV / 更大位宽压缩 | **不做**：无内核支持（内核决定一切） | — |

> **一句话**：这篇论文替我们**确认了方向**（KV 结构 > KV 位宽；prefill 与 decode 的激活该解耦），
> 而**最可移植的那一条（CED → 跨层 KV 共享）恰好能用我们已有的 LoRA + harness 工装验证**。
