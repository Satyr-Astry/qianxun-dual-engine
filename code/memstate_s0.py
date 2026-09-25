"""
忆态架构（MemState）· S0 结构等价改写 —— 原型与自验证
================================================================
目的（N1 闸门）：证明「用现有参数装进新架构骨架」在结构上是**无损**的。

做法：
  1) 加载 Qwen2.5-1.5B-Instruct（fp32，本机 HF 缓存已有）作为「慢主干」
  2) 改造：
       · 3 个 MLP 层 → MemoryFFN = 原 MLP 输出 + gate · 记忆层输出（gate 零初始化）
       · 记忆层 = 乘积键 top-k 查表（跨层共享同一记忆池，Memory Layers 式）
       · q_proj 挂零初始化 LoRA（B=0）
       · 路由打分器（DSA 式 indexer）计算但不生效（仅记录）
  3) 验证等价：同一批 prompt，greedy 生成逐 token 比对 + logits 最大绝对差
  4) 活体测试：把 gate 从 0 抬到 0.05 → 输出必须发生变化（证明分支真的接进了计算图，不是死代码）
  5) 可训性：冻结主干，只训 gate + 记忆输出投影 + LoRA-B，看 loss 是否下降
  6) 统计：新增参数量/占比、记忆池字节数、访存对照（dense FFN vs 记忆查表）

运行：/e/HARNESS/selfgrow/venv/Scripts/python.exe code/memstate_s0.py
"""
import os, sys, time, json, math, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
MEM_ENTRIES = 65536          # 记忆条目数（乘积键：256 × 256）
MEM_HALF_KEY = 256           # 每半键数量
MEM_KEY_DIM = 128            # 半键维度
MEM_TOPK = 8                 # 每 token 取 k 条
REPLACE_STRIDE = 8           # stride 8、居中取 3 层（Memory Layers 甜点）
LORA_R = 16

PROMPTS = [
    "用一句话解释什么是 KV 缓存。",
    "The capital of France is",
    "写一个 Python 函数，返回斐波那契数列前 n 项。",
    "如果 3x + 7 = 22，那么 x 等于",
]


# ────────────────────────────── 记忆层（乘积键 top-k 查表） ──────────────────────────────
class ProductKeyMemory(nn.Module):
    """跨层共享的记忆池：键用乘积键（两个半键集），值是可训练条目。"""

    def __init__(self, d_model, n_entries=MEM_ENTRIES, half=MEM_HALF_KEY, kd=MEM_KEY_DIM, topk=MEM_TOPK):
        super().__init__()
        assert half * half == n_entries, "本原型用 half×half = n_entries 的乘积键"
        self.half, self.kd, self.topk, self.d = half, kd, topk, d_model
        self.K1 = nn.Parameter(torch.randn(half, kd) / math.sqrt(kd))   # 半键 A
        self.K2 = nn.Parameter(torch.randn(half, kd) / math.sqrt(kd))   # 半键 B
        self.V = nn.Parameter(torch.zeros(n_entries, d_model))          # 值（条目）
        self.q_proj = nn.Linear(d_model, 2 * kd, bias=False)            # query → 两个半键
        self.w_out = nn.Linear(d_model, d_model, bias=False)            # 记忆输出投影
        nn.init.zeros_(self.w_out.weight)

    def forward(self, h):
        B, T, D = h.shape
        q = self.q_proj(h).view(B, T, 2, self.kd)           # [B,T,2,kd]
        q1, q2 = q[..., 0, :], q[..., 1, :]
        s1 = q1 @ self.K1.t()                               # [B,T,half]
        s2 = q2 @ self.K2.t()
        t1 = s1.topk(min(self.topk, self.half), dim=-1).indices
        t2 = s2.topk(min(self.topk, self.half), dim=-1).indices
        # 候选 = 两侧 top-k 的笛卡尔积（乘积键检索，从不实例化全量键矩阵）
        cand = (t1.unsqueeze(-1) * self.half + t2.unsqueeze(-2)).reshape(B, T, -1)      # [B,T,k*k]
        # 用候选的精确分数（用拼接的半键做近似内积）选中最终 top-k 条目
        k_cat = torch.cat([self.K1.repeat_interleave(self.half, 0),
                           self.K2.repeat(self.half, 1)], dim=-1)                        # [n_entries, 2kd]
        q_cat = self.q_proj(h)                                                           # [B,T,2kd]
        k_sel = k_cat[cand]                                                              # [B,T,k*k,2kd]
        score = (q_cat.unsqueeze(-2) @ k_sel.transpose(-1, -2)).squeeze(-2)              # [B,T,k*k]
        topk = min(self.topk, score.shape[-1])
        idx = score.topk(topk, dim=-1).indices                                           # [B,T,k]
        sel = cand.gather(-1, idx)                                                       # [B,T,k]
        w = F.softmax(score.gather(-1, idx), dim=-1)
        val = self.V[sel]                                                                # [B,T,k,D]
        out = (w.unsqueeze(-1) * val).sum(-2)
        return self.w_out(out), sel


# ────────────────────────────── 改造后的 MLP（记忆分支 + 零门控） ──────────────────────────────
class MemoryFFN(nn.Module):
    """原 MLP（慢权重，冻结保留） + gate · 记忆层输出。gate 零初始化 ⇒ S0 严格等价。"""

    def __init__(self, orig_mlp, mem):
        super().__init__()
        self.orig = orig_mlp
        self.mem = mem
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        base = self.orig(x)
        m, _ = self.mem(x)
        return base + self.gate * m          # gate=0 → 与 base 逐位相同


# ────────────────────────────── 零初始化 LoRA（挂在 q_proj 上） ──────────────────────────────
class ZeroLoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=LORA_R):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # 零初始化 ⇒ 等价

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


# ────────────────────────────── 路由打分器（DSA 式 indexer，S0 只计算不生效） ──────────────────────────────
class IndexerScorer(nn.Module):
    def __init__(self, d_model, n_heads=8, head_dim=128):
        super().__init__()
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, head_dim, bias=False)
        self.weights = nn.Linear(d_model, n_heads, bias=False)

    @torch.no_grad()
    def forward(self, h):
        q = self.wq(h)                                  # [B,T,H*D]
        k = self.wk(h)                                  # [B,T,D]
        return q, k


# ────────────────────────────── 改造整体 ──────────────────────────────
def rewrite(model):
    """把 3 个 MLP 换成 MemoryFFN（共享记忆池），并给 q_proj 挂零 LoRA。返回统计。"""
    layers = model.model.layers
    n = len(layers)
    center = n // 2
    idxs = [center + i * REPLACE_STRIDE // 2 for i in (-1, 0, 1)]
    idxs = [i for i in idxs if 0 <= i < n][:3]

    d_model = model.config.hidden_size
    mem = ProductKeyMemory(d_model)
    stats = {"n_layers": n, "replaced_layers": idxs, "d_model": d_model}

    for i in idxs:
        layers[i].mlp = MemoryFFN(layers[i].mlp, mem)
    # 路由打分器（每层一个，S0 不生效）
    scorers = nn.ModuleList([IndexerScorer(d_model) for _ in range(n)])
    # 零 LoRA（挂在每个注意力层的 q_proj）
    n_lora = 0
    for layer in layers:
        layer.self_attn.q_proj = ZeroLoRA(layer.self_attn.q_proj)
        n_lora += 1
    stats["n_lora"] = n_lora

    def count(module):
        return sum(p.numel() for p in module.parameters())

    trunk = count(model)
    new_mem = count(mem)
    new_lora = sum(count(l.self_attn.q_proj) - count(l.self_attn.q_proj.base) for l in layers)
    new_scorer = count(scorers)
    stats.update(
        trunk_params=trunk - new_mem - new_lora,     # 扣除新加的
        new_mem=new_mem, new_lora=new_lora, new_scorer=new_scorer,
        mem_bytes=mem.V.numel() * 4 + (mem.K1.numel() + mem.K2.numel()) * 4,
    )
    return mem, scorers, stats


@torch.no_grad()
def greedy(model, tok, prompt, n_new=48):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return out[0].tolist()


@torch.no_grad()
def first_logits(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    return model(ids).logits[0, -1].float().cpu()


def main():
    t0 = time.time()
    print("=" * 78)
    print("忆态架构 S0：结构等价改写原型")
    print(f"device={DEV}  dtype={DT}  torch={torch.__version__}")
    print("=" * 78)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)                      # 主干=慢权重，先冻结
    print(f"[load] 主干参数 {sum(p.numel() for p in model.parameters())/1e6:.1f}M，"
          f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    # ── 1) 基线输出
    base_tokens = {p: greedy(model, tok, p) for p in PROMPTS}
    base_logits = {p: first_logits(model, tok, p) for p in PROMPTS}

    # ── 2) 结构等价改写
    mem, scorers, stats = rewrite(model)
    model.to(DEV)
    new_total = sum(p.numel() for p in model.parameters())
    print(f"\n[rewrite] 替换 MLP 层 = {stats['replaced_layers']}（共 {stats['n_layers']} 层，共享 1 个记忆池）")
    print(f"          记忆池：{MEM_ENTRIES} 条目 × d={stats['d_model']}，"
          f"体积 {stats['mem_bytes']/2**20:.1f} MiB")
    print(f"          新增参数：记忆池 {stats['new_mem']/1e6:.2f}M + LoRA {stats['new_lora']/1e6:.2f}M"
          f" + 路由打分器 {stats['new_scorer']/1e6:.2f}M")
    print(f"          改造后总参数 {new_total/1e6:.1f}M")

    # ── 3) 等价性验证（S0 闸门）
    print("\n" + "-" * 78)
    print("【S0 闸门】逐 token 一致率 + logits 最大绝对差")
    print("-" * 78)
    tok_hit = tok_all = 0
    max_logit_diff = 0.0
    rows = []
    for p in PROMPTS:
        new = greedy(model, tok, p)
        hit = sum(1 for a, b in zip(base_tokens[p], new) if a == b)
        tok_all += len(new); tok_hit += hit
        d = (base_logits[p] - first_logits(model, tok, p)).abs().max().item()
        max_logit_diff = max(max_logit_diff, d)
        rows.append((p[:22], len(new), hit, hit / len(new), d))
        print(f"  {p[:22]:24s} 生成 {len(new):3d} token  一致 {hit:3d}  ({hit/len(new)*100:6.2f}%)  "
              f"logits最大差 {d:.3e}")
    agree = tok_hit / tok_all
    print(f"\n  ⇒ 总一致率 {agree*100:.2f}%   logits 最大绝对差 {max_logit_diff:.3e}")
    verdict = "✅ 通过（结构无损）" if agree >= 0.98 and max_logit_diff == 0.0 else "❌ 未通过"
    print(f"  ⇒ 判定：{verdict}（门槛：一致率≥98% 且 logits 差为 0）")

    # ── 4) 活体测试：把门打开，输出必须改变
    print("\n" + "-" * 78)
    print("【活体测试】gate 0 → 0.05（证明记忆分支真的在计算图里，不是死代码）")
    print("-" * 78)
    with torch.no_grad():
        for layer in model.model.layers:
            if isinstance(layer.mlp, MemoryFFN):
                layer.mlp.gate.fill_(0.05)
    live_rows = []
    for p in PROMPTS[:2]:
        new = greedy(model, tok, p)
        hit = sum(1 for a, b in zip(base_tokens[p], new) if a == b) / len(new)
        live_rows.append((p[:22], hit))
        print(f"  {p[:22]:24s} 与基线一致率 {hit*100:6.2f}%（应显著下降 = 分支已生效）")
    with torch.no_grad():
        for layer in model.model.layers:
            if isinstance(layer.mlp, MemoryFFN):
                layer.mlp.gate.zero_()

    # ── 5) 可训性：只训 gate + 记忆输出投影 + LoRA-B
    print("\n" + "-" * 78)
    print("【可训性】冻结主干，只训架构件（gate + w_out + LoRA.B），20 步")
    print("-" * 78)
    trainable = []
    for layer in model.model.layers:
        if isinstance(layer.mlp, MemoryFFN):
            layer.mlp.gate.requires_grad_(True); trainable.append(layer.mlp.gate)
            layer.mlp.mem.w_out.weight.requires_grad_(True); trainable.append(layer.mlp.mem.w_out.weight)
        layer.self_attn.q_proj.B.requires_grad_(True); trainable.append(layer.self_attn.q_proj.B)
    n_train = sum(t.numel() for t in trainable)
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    text = ("机器学习模型的推理速度主要受显存带宽限制。把知识放在可寻址的记忆表里，"
            "就能让参数规模与计算开销解耦。" * 3 +
            "The quick brown fox jumps over the lazy dog. Memory layers store "
            "associations in trainable key value tables." * 3)
    ids = tok(text, return_tensors="pt").input_ids[:, :256].to(DEV)
    model.train()
    losses = []
    for step in range(20):
        out = model(ids, labels=ids)
        loss = out.loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == 19:
            gates = [l.mlp.gate.item() for l in model.model.layers if isinstance(l.mlp, MemoryFFN)]
            print(f"  step {step:2d}  loss {loss.item():.4f}  gate={['%.4f' % g for g in gates]}")
    model.eval()
    print(f"  ⇒ loss {losses[0]:.4f} → {losses[-1]:.4f}  下降 {losses[0]-losses[-1]:.4f}"
          f"（{'✅ 可训' if losses[-1] < losses[0] else '❌ 未下降'}）")
    print(f"  ⇒ 可训参数 {n_train/1e6:.3f}M （主干 {stats['trunk_params']/1e6:.1f}M 冻结）")

    # 训练后又把门关掉，是否恢复等价（可回滚性）
    with torch.no_grad():
        for l in model.model.layers:
            if isinstance(l.mlp, MemoryFFN):
                l.mlp.gate.zero_()
            l.self_attn.q_proj.B.zero_()
    back = sum(1 for a, b in zip(base_tokens[PROMPTS[1]], greedy(model, tok, PROMPTS[1])) if a == b) / len(base_tokens[PROMPTS[1]])
    print(f"\n【可回滚性】训练后把 gate/LoRA-B 归零 → 与基线一致率 {back*100:.2f}%（应回到 100%）")

    # ── 6) 访存对照（把 1632× 的论证在原型尺度上验证）
    d = stats['d_model']
    ffn_layers_shape = 8960  # Qwen2.5-1.5B intermediate_size 近似
    dense_bytes = 3 * d * ffn_layers_shape * 0.5          # Q4 近似 0.5B/param
    mem_bytes = MEM_TOPK * d * 2                          # fp16 取 k 条
    print("\n" + "-" * 78)
    print("【访存对照·原型尺度】dense FFN vs 记忆查表（每 token 每层）")
    print("-" * 78)
    print(f"  dense FFN 需读 ≈ {dense_bytes/2**10:.0f} KiB（Q4）")
    print(f"  记忆查表 k={MEM_TOPK} 需读 ≈ {mem_bytes/2**10:.0f} KiB（fp16）")
    print(f"  ⇒ 差距 ≈ {dense_bytes/mem_bytes:.0f}×")

    # ── 产出 JSON
    res = {"device": DEV, "stats": stats, "agree": agree, "max_logit_diff": max_logit_diff,
           "verdict": verdict, "liveness": live_rows,
           "train_loss_first": losses[0], "train_loss_last": losses[-1],
           "trainable_params": n_train, "rollback_agree": back,
           "ffn_bytes_proto": dense_bytes, "mem_read_bytes_proto": mem_bytes,
           "elapsed_s": time.time() - t0}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s0_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[out] 结果已写入 {out_path}")
    print(f"[time] 总耗时 {time.time()-t0:.1f}s  峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("=" * 78)
    return res


if __name__ == "__main__":
    main()
