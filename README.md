# 千寻 · 双引擎推理栈（Qianxun Dual-Engine Lab）

> **在 12 GB 显存的消费级显卡上，把一个小模型的「上下文容量 / 速度 / 自学习」三轴推到它能到的位置**
> —— 并且把**跑不通的路也如实记录下来**。
>
> A hands-on lab notebook for squeezing a 14B local model on a single 12 GB GPU:
> measured performance laws, a layer-streaming loader, a readout-side LoRA that fixes
> tool calling, and an honest negative-result library.

**定位**：这是一个**架构设计与机制验证平台**，不是"又一个可用模型"。
产出物是三样：① 可复现工装 ② 负面结果库 ③ 机制论文的素材。

---

## 硬件与环境（全部数字都出自这台机器）

| 项 | 值 |
|---|---|
| GPU | **RTX 4070 Ti 12 GB**（物理 12282 MiB，`504 GB/s` 标称带宽） |
| CPU / 内存 | AMD Ryzen 7 5700X3D 8C/16T / **16 GB DDR4** |
| OS | Windows 10（bash/MSYS 环境） |
| 推理后端 | **llama.cpp build 10938**（`llama-server`） |
| 底座模型 | 本机 `huihui_ai/deepseek-r1-abliterated:14b`（Q4_K_M **8.37 GiB**，48 层，d=5120，FFN=13824，40/8 头，ctx _131072_） |
| Python | 3.11 + torch 2.11 / transformers 5.16 / peft 0.20 / bitsandbytes 0.50 |

---

## ⭐ 主要成果

### 1. 本机性能「三档定律」：一个数字预测 20–30× 的性能悬崖

**加载期余量**（`12282 − 探针实测占用`）单变量就能预测一切：

| 区间 | 余量 | prefill | decode |
|---|---|---|---|
| 🟢 **健康区** | **≥ 1.4 GiB** | **1500–2200 tok/s** | **37.8–44 tok/s** |
| ⚠️ **悬崖区** | 0.6–0.8 GiB | **49–73 tok/s**（塌 20–30×） | 15–25 |
| ⛔ **灾难区** | 负（超额） | — | **4.73** |

实测数据（全部真喂长文）：

| 组 | 配置 | 加载期余量 | prefill | decode |
|---|---|---|---|---|
| **Q6** | q8 KV, ctx **24576**, 不卸载 | **1434** | **2202** | **37.8** |
| B | q8 KV, ctx 8192 | ~3.8 GiB | 1038 | 44.0 |
| Q2 | q8 KV, ctx 32768 + 末 8 层 FFN 下 CPU | 1578 | 1538 | 19.5 |
| Q1 | q8 KV, ctx 32768, 不卸载 | 617 | 73.1 | 25.5 |
| Q5 | 同上 + 小批量 `-b 512 -ub 256` | 617 | **72.6（证伪：无效）** | 26.9 |
| Q4 | q8 + `-ot`, ctx 40960 | 762 | 48.8 | 15.4 |
| C | q8 KV, ctx 40960, 不卸载 | **−247** | — | **4.73** |

> **最坑的一条**：显存**超额约 247 MiB** 就让 decode 塌一个数量级，
> 而 **llama.cpp 全程不报错**（`--probe` 只保证"能加载"，**不保证 prefill 不溢出**）。

**结论**：那 20–30× 的悬崖是**显存预算分配问题，不是算法问题**——先治预算，白拿 **5.4×**。

### 2. 读出侧 LoRA：把「只会聊天」的本地模型改成「会调用工具」

问题：本机 r1-14B 接入 agent 框架后**只吐裸 JSON 或散文**，从不产生规范的
`<tool_call>{...}</tool_call>` ⇒ 工具调用**完全不可用**。

**12 GB 卡上唯一可行的训练路线**：冻结主干，只训**末 4 层 + lm_head** 的 LoRA
（依据是自己实验的结论：**读出侧才是瓶颈**，见 `docs/负面结果库.md`）。

| 环节 | 实测 |
|---|---|
| SFT 数据 | 1200 条合成（正例 872 含 `<tool_call>` / 负例 328 教"不该调"），格式自检 0 违规 |
| 隐状态缓存 | 618 条，流式跑前 44 层，**27.6 s/16 条** |
| 训练 | r=16，loss 3.234 → 0.000，**峰值显存仅 4.25 GiB** |
| 导出 | PEFT → GGUF **16.5 MB / 58 张量** |
| **验收** | **6/6 行为正确；真 `tool_calls` 4 次（改造前基线 0 次）** |

验收明细（4 正例 + 2 负例）：

| 探针 | 结果 |
|---|---|
| 读一下 F:/…/设定.txt | ✅ `read_file {"path": "..."}` 完全正确 |
| 列一下 E:/models 下面有什么 | ✅ `list_dir {"path": "E:/models"}` |
| 跑一下 nvidia-smi 看看显卡占用 | ✅ `terminal {"command": "nvidia-smi"}` |
| 联网搜一下 2026 年最新 MoE 模型 | ✅ `web_search {"query": "..."}` |
| 你好呀，在吗？ | ✅ 不调工具 |
| 1+1 等于几？ | ✅ 不调工具 |

**已知副作用（如实记录）**：618 条模板数据 + 1000 步 ⇒ 过拟合，
**非工具回答出现退化重复**（`'等于 2 不对等于 2 不对…'`）。已实测 LoRA 强度扫描：

| scale | 工具调用 | 退化重复 |
|---|---|---|
| 0.5 / 0.7 / 1.0 | ✅ 全部 6/6、4 次真 `tool_calls` | ⚠️ 全部仍有（0.5 时变成"正确内容重复"） |

⇒ ①工具调用**对强度不敏感**，降档无必要；②**降强度只能减轻不能消除**重复 ⇒ **根因在数据与训练**，
不在强度。下一轮修法（已记入 `docs/N9_报告.md`）：只训正例 / 按留出 loss 早停 / 混入通用语料 /
推理侧 `repeat_penalty` 兜底。

**第二轮实验（v2 = 只训正例）— 假设被证伪，但很有价值**

| 版本 | 数据 / 步数 | 行为正确 | 真 `tool_calls` | 症状 |
|---|---|---|---|---|
| **v1** | 正例+负例 618 条 / 1000 步 | **6/6** | 4 | 闲聊退化重复 |
| v2 | 只正例 388 条 / 300 步 / lr 1e-4 | 5/6 | 5 | **误触发**（"1+1" 去调 `terminal calc 1+1`）+ 仍退化 |

⇒ ①**只训正例治不了重复，还丢了精度**；②退化重复**与负例无关**——它是**读出侧 LoRA
在小数据下破坏原生"停止/多样性"行为**的结果；③**v1 是两版里更好的那个**，v2 仅作对照保留。

**这条负面结果比成功更有信息量**：读出侧（末 4 层 + lm_head）**能学会"输出形态"**
（0 → 4~5 次真 `tool_calls`，工具名与参数全对），但**"保住通用行为不变"它改不动** ——
那要靠更大规模、更混杂的数据，或**全层微调**（本机 12 GB 做不到）。这正是"稳定干活"的真门槛。

⚠️ **真实的验收结论（比上面的探针更重要）**：在**真实 Hermes 环境**里它**还不能用** ——
`hermes chat -q "列一下 E:/models 目录里有什么"` 会**反复调 `read_file`（连 12 次）**、9 分钟不收敛。
原因：探针只给 1~4 个工具而真实提示有 **22 个**，且 SFT 数据用的是 **7 个自编的"玩具工具"**。
⇒ **「能发出 `tool_calls`」≠「能用」**；下一轮必须**照抄真实工具集**造数据 + 加**多轮结果回流**样本。
详见 `docs/N9_报告.md` 第三轮与 `docs/负面结果库.md` NR-11。

**两个 Windows 特有的坑（都踩过，值得写下来）**
1. `--lora-scaled "E:/…/adapter.gguf:0.7"` **必然报错**：llama.cpp 按**第一个冒号**切 `FNAME:SCALE`，
   而盘符 `E:` 就是这个冒号 ⇒ 改用运行时 `POST /lora-adapters`（还省掉每次重启 75 s）。
2. **stale listener**：Windows 下多个 `llama-server` 能同时 bind 同一端口，**最先启动的一直在应答** ——
   症状是"重启三次毫无变化、`/health` 秒回 200"；`/props` 看 `chat_template` 指纹才能验明正身。

### 3. 14B 在 PyTorch 侧的可行路径：逐层流式加载

bitsandbytes **4bit/8bit 加载 14B 在本机必 segfault**（EXIT=139，三次复现）——
根因是 28 GB bf16 权重先落主机内存 > 16 GB RAM，**Windows 下表现为 segfault 而不是 MemoryError**。

替代方案（已跑通）：**只开当前层需要的 safetensors 分片、用完即关**，复用同一层模块：

| 指标 | 值 |
|---|---|
| 整模型 48 层前向 | **30.3 s** |
| 峰值显存 | **1.98 GiB** |
| 保真度校验 | 对 prompt「用一句话解释什么是 KV 缓存：」的 top-1 = **`'KV'`** ✅ |

配套工具：`tools/gguf_to_hf.py`（GGUF → bf16 分片，分块流式，27.51 GiB / 6 分片）。

### 4. 自训 MTP 头（投机解码用 draft）

本基底**没有 MTP 头**，而投机解码是唯一"一次读权重、多产出 token"的免费加速。

| 版本 | 配置 | **α₁（留出集）** |
|---|---|---|
| v1 | lr 3e-4 恒定 / 600 步 | 0.3396 |
| **v2** | **lr 1e-4 + warmup + 余弦 / 2000 步** | **0.4445** |
| 平凡基线（复读） | — | 0.0111 |

> **只改学习率调度就 +0.105 绝对（+31% 相对）** ⇒「实现未过关」≠「原理证伪」。
> 未达 0.6 判据，主因是数据量（82 K token）≪ 参数量（327 M）。

### 5. 引擎本体：投机解码与带宽天花板

| 组 | 配置 | decode |
|---|---|---|
| S2 | 无投机 | 40.4 tok/s |
| **S1** | `--spec-type ngram-mod` | **46.5 tok/s（+15.2%）** |

**46.5 tok/s ⇒ 418 GB/s = 504 GB/s 峰值的 83%**：短上下文解码**已把带宽打满**。
⇒ 调参到头了，**唯一出路是"每 token 少读字节"**（更强的 draft / 记忆层替 FFN / 更激进量化）。

---

## 🚀 快速开始

```bash
# ① 环境探针：先算清显存预算，再决定参数（最重要的习惯）
python tools/n7_n8_bench.py --probe

# ② 按「三档定律」选配置（示例：16K 内日常，prefill 2202 / decode 37.8）
llama-server -m <your-14b.gguf> -ngl 99 -c 24576 -ctk q8_0 -ctv q8_0

# ③ 量化检查工装（六项：结构等价 / 活性 / 回滚 / 梯度 / 主干不变 / 噪声地板）
python tools/memcore_harness.py --selftest

# ④ 工具调用微调全流程（数据 → 缓存 → 读出侧 LoRA → GGUF → 验收）
python code/gen_tool_sft.py --n 1200
python code/lora_readout_sft.py --stage cache --layers-from 44
python code/lora_readout_sft.py --stage train --steps 1000 --lr 2e-4
python tools/serve_with_lora.py --convert --start --test
```

---

## 📁 目录

```
docs/    机制与实测报告（中文）
  计划书.md                    权威设计书（含算法架构 + 程序实况 + 缺陷清单 + 变更日志）
  N7N8_报告.md                 本机性能七轮实测（三档定律原始数据）
  N8_报告.md                   MTP 头 + 逐层流式加载（含五个必踩坑）
  N9_报告.md                   读出侧 LoRA 工具调用（含验收明细与副作用）
  Hermes接入_自检报告.md        把本地模型接进 agent 框架的实测与局限
  负面结果库.md                NR-1… 诚实记录（本项目最有价值的部分之一）
  论文规划.md                  可主张的机制结论 C1–C6
  *_论述.md                    静默态 / 长上下文资源控制 / 自学习闭环 三篇设计论述

tools/   工装（可直接用）
  n7_n8_bench.py                llama-server 计时 + --probe 显存预算探针
  memcore_harness.py            架构改造六项检查 + 组装件
  gguf_to_hf.py                 GGUF → HF bf16（分块流式，防 OOM）
  serve_with_lora.py            LoRA 服务 + 工具调用自动验收（含强度扫描）
  start_r1_for_hermes.bat       一键把本地模型接进 agent 框架
  test_r1_endpoint.py           端点体检（含工具调用能力探测）

code/    实验代码
  lora_readout_sft.py           读出侧 LoRA 微调（cache / train / eval 三段）
  gen_tool_sft.py               工具调用 SFT 数据生成器
  mtp_n8s1.py                   逐层流式加载 + MTP 头训练
  memstate_n2c/n2d/n2e.py       记忆分支三臂对照 / 放大 / 动态写入（负面结果的来源）
```

---

## 📉 负面结果库（精选，详见 `docs/负面结果库.md`）

诚实记录比成功更有价值。已收录：

- **冻结主干 + 小架构件写不进记忆**：记忆分支净贡献 **−1.0pp**（n=128×2 种子放大后归零；
  曾出现的 9.4% 被证明是 **n=32 单种子噪声**）。对标 Memory Layers(1T token) / Sparse Upcycling(~50% 预算)，
  业界**无"冻结主干 + 小模块"成功先例**。
- **"必须 bit-exact 才算改造成功"是过度要求**（撤销）：自我进化系统只需**功能等价**
  （全位置 logits 相对差 ≤ tol）。
- **本机 14B 的 bitsandbytes 量化路径不通**（3 次 segfault，含根因分析）。
- **小批量（`-b/-ub`）救不了 prefill 悬崖**（72.6 vs 73.1，证伪）。
- **免训练投机（ngram）只有 +15%**：抄写类文本尚且如此 ⇒ 想要 1.5× 必须自训 draft。
- **harness 首轮自测抓出 3 个真缺陷**（元组返回 / 过滤器漏 `.branch` / 基线指纹重复拼 prompt）。
- **"测不出来"和"测出来是零"在日志上长得一模一样**：一次无效测量把
  **`连接被拒`** 伪装成"**0/6 行为正确**"（服务没起来 + 残留实例应答，双根因见 NR-10）⇒
  已加固为"测完先自检指纹（`/props` + 实例数），失败样本不计入分母"。

---

## ⚠️ 诚实的局限

1. **不是可用模型**：这是机制验证平台；读出侧 LoRA 会破坏一部分通用文本生成能力（退化重复）。
2. **14B 全层微调在本机做不到**（12 GB 显存 + 16 GB 内存），这是"稳定干活"的下一道门槛。
3. **工具调用验收样本量小**（6 条探针），只证明"方向成立"，不代表生产可用。
4. 部分报告含作者本机绝对路径（`F:/DESKTOP/...`），作为实验记录保留原样。

---

## 许可

MIT（见 `LICENSE`）。引用其中的实测数字请注明本仓库与实测环境。

---

*"A model that only chats is a toy; a model that calls tools is an engine.
The interesting part was not making it work — it was writing down every way it didn't."*
