# 本机 r1-14B 接入 Hermes · 自检报告

> 目标：让主人能在 Hermes 里直接试本机 `huihui_ai/deepseek-r1-abliterated:14b` 的效果。
> 实测时间：2026-09-25。**已跑通**（一次完整 Hermes 问答，无 fallback）。

---

## 一、已完成的接线

| 项 | 值 |
|---|---|
| 服务 | `llama-server` @ `http://127.0.0.1:8710/v1`，alias **`r1-14b-local`** |
| Hermes provider | `providers.r1-local`（已写入 `E:/HERMES/config.yaml`，**主网关仍是 deepseek 未动**） |
| 启动脚本 | `tools/start_r1_for_hermes.bat`（双击即可） |
| 用法 A（CLI，已验证） | `hermes chat -q "..." --model r1-14b-local --provider r1-local` |
| 用法 B（桌面） | 会话里输入 `/model` → 选 provider `r1-local` / model `r1-14b-local` |

⚠️ **CLI 必须同时给 `--model` 和 `--provider`**：只给 `--model` 时 Hermes 会把这个名字发给
主 provider（deepseek）→ 404 → **静默 fallback 到 deepseek-v4-pro**（日志里只有一行提示，很容易误判成功）。

---

## 二、验证证据（三层）

1. **端点层**：`/v1/models` → `['r1-14b-local']`；中文问答正常；
   推理题正确（水池题答出 12/5 小时 = 2 小时 24 分）。
2. **服务层**：`/props` → `n_ctx = 65536`、`chat_template` 长度 2506（含 Qwen2.5 工具段）。
3. **Hermes 端到端**：`hermes chat -q "用一句话介绍你自己是谁。" --model r1-14b-local --provider r1-local`
   → 27 s 返回，**无 fallback 警告**，且回答跟着 Hermes 的人格走（自称"小悠"）。

---

## 三、为什么必须是这条配置（关键发现）

### 发现 1：Hermes 硬要求 ≥64K 上下文
`agent/agent_init.py::_enforce_minimum_context`：
```python
if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not _allow_lmstudio_explicit_below_floor:
    raise ValueError(f"Model ... has a context window of {_ctx:,} ... below the minimum 64,000 ...")
```
- 我们第一次用 `-c 32768` → **Hermes 直接拒绝初始化**。
- 逃逸开关 `_allow_lmstudio_explicit_below_floor` **只对 `provider == "lmstudio"` 生效**，
  自定义 provider（`r1-local`）用不了。
- ❌ **不要**用 `model.context_length: 32768` 骗过关：那是**全局**值，会连带把主模型的压缩阈值也压到 32K。

### 发现 2：64K 在 12 GB 卡上的唯一可行组合
| 配置 | 权重 | KV@64K | compute | 合计 | 余量 | 结论 |
|---|---|---|---|---|---|---|
| q8/q8 + 卸 8 层 | 7115 | 6144 | 325 | 13584 | **−1302** | ❌ 装不下 |
| q8/q8 + 卸 16 层 | 6082 | 6144 | 325 | 12551 | −269 | ❌ |
| q8/q4 + 卸 16 层 | 6082 | 4896 | 325 | 11303 | 979 | ⚠️ 悬崖区 |
| **q4/q4 + 卸 16 层** ✅ | **6082** | **3264** | **325** | **9671** | **~1660**（扣桌面 ~900） | ✅ **健康区** |

⇒ 采纳最后一行：`-c 65536 -ctk q4_0 -ctv q4_0 -ot "blk\.(3[2-9]|4[0-7])\.ffn.*=CPU"`。
（只卸 8 层时余量约 700 ⇒ 落进 prefill 悬崖，**首答要等几分钟**；卸 16 层后首答 27 s。）

---

## 四、诚实的局限

| # | 局限 | 证据 |
|---|---|---|
| 1 | **工具调用不可用** ⇒ 在 Hermes 里只能当**纯聊天模型**，不能干活 | 三次实测：`--jinja` 原生模板 → 完全不吐 `tool_calls`；换 Qwen2.5 工具模板 → 模型**确实会发调用**，但输出 `{"name": "get_weather", "arguments": {...}}` **缺 `<tool_call>` 包裹标签**，llama.cpp 解析器拒绝提取 |
| 2 | 回答里会漏 `</think>` 标签 | R1 原生模板痕迹与 Qwen2.5 模板混用 |
| 3 | KV 量化到 **q4_0**（为塞下 64K） | 有质量损失，长对话更明显 |
| 4 | 速度 ~**13 tok/s**（q4/q4 + 16 层卸载）；q8/q8 配置下才是 21–31 tok/s | `/completion` timings |
| 5 | 桌面应用常驻约 900 MiB 显存，**会吃掉一档余量** | 服务独立跑时余量 ~1660，实测 10619/12282 |

**结论**：适合"试试本地模型的味道"（聊天、跟着人格走、数据不出本机）；
**不适合**取代 deepseek 做主网关（无工具调用 + 64K 是硬挤出来的）。

---

## 五、运维

```bash
# 启动（或双击 tools/start_r1_for_hermes.bat）
# 停止
taskkill /F /IM llama-server.exe

# ★ 验明正身（最容易被骗的一步）：确认只有一个实例、且模板/n_ctx 是新的
netstat -ano | grep ":8710.*LISTENING"
tasklist | grep -i llama
# /props 看 chat_template 指纹与 n_ctx
```

**最坑的坑（已记进技能）**：Windows 下 llama-server **多实例可同时 bind 同一端口**（SO_REUSEADDR），
**最先启动的那个一直应答** ⇒ "重启了三次其实一直在跟旧实例说话"（症状：重启后 `/health` 秒回 200、
新改动毫无效果）。另外 MSYS 下 `taskkill //F` 会被当字面量而**杀不掉进程**，要用**单斜杠** `taskkill /F`。
