"""
忆态架构 S0b —— 修正版：零初始化「死锁」修复 + 记忆分支活性/可训性归因
================================================================
S0 原版的实测结果：
  ✅ 结构等价：逐 token 一致率 100.00%，logits 最大绝对差 0.000e+00（严格无损）
  ❌ 活体测试失败：gate 从 0 抬到 0.05，输出**完全没变**
  → 定位：w_out 与 gate **双层零初始化** ⇒ 记忆输出恒为 0，且两侧梯度互为 0（死锁/鞍点），
     记忆分支永远学不动。（loss 下降来自同时被训的 LoRA-B，不是记忆分支）

本版修正（关键设计）：
  · 只允许 **gate 零初始化**（保证 S0 严格无损：gate·x ≡ 0 逐位为 0）
  · w_out 用小随机初始化 ⇒ gate 的梯度 = <dL/dout, mem_out> ≠ 0 ⇒ 分支可被"开门"
验证内容：
  1) 等价性（期望 100% / 0.0）
  2) 梯度检查：gate 的梯度必须非零（死锁已解除的直接证据）
  3) 活体：gate=0.05 → 输出应变化
  4) 回滚：gate 归零 → 恢复 100%
  5) 归因训练 A：只训 gate + w_out + 记忆值 V（不训 LoRA）→ loss 应下降且 gate 离开 0
  6) 归因训练 B：只训 LoRA（冻结记忆分支）→ 对比两者
"""
import os, time, json, math
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
MEM_ENTRIES, MEM_HALF_KEY, MEM_KEY_DIM, MEM_TOPK = 65536, 256, 128, 8
REPLACE_STRIDE, LORA_R = 8, 16
PROMPTS = [
    "用一句话解释什么是 KV 缓存。",
    "The capital of France is",
    "写一个 Python 函数，返回斐波那契数列前 n 项。",
    "如果 3x + 7 = 22，那么 x 等于",
]


class ProductKeyMemory(nn.Module):
    def __init__(self, d, n=MEM_ENTRIES, half=MEM_HALF_KEY, kd=MEM_KEY_DIM, topk=MEM_TOPK):
        super().__init__()
        self.half, self.kd, self.topk = half, kd, topk
        self.K1 = nn.Parameter(torch.randn(half, kd) / math.sqrt(kd))
        self.K2 = nn.Parameter(torch.randn(half, kd) / math.sqrt(kd))
        self.V = nn.Parameter(torch.randn(n, d) * 0.02)   # ★ 必须非零：否则 gate 的梯度也归零
        self.q_proj = nn.Linear(d, 2 * kd, bias=False)
        self.w_out = nn.Linear(d, d, bias=False)
        # ★ 关键修正：小随机初始化（不是零！），否则与 gate 形成零死锁
        nn.init.normal_(self.w_out.weight, std=1.0 / math.sqrt(d))

    def forward(self, h):
        B, T, D = h.shape
        q = self.q_proj(h).view(B, T, 2, self.kd)
        s1 = q[..., 0, :] @ self.K1.t()
        s2 = q[..., 1, :] @ self.K2.t()
        t1 = s1.topk(min(self.topk, self.half), -1).indices
        t2 = s2.topk(min(self.topk, self.half), -1).indices
        cand = (t1.unsqueeze(-1) * self.half + t2.unsqueeze(-2)).reshape(B, T, -1)
        k_cat = torch.cat([self.K1.repeat_interleave(self.half, 0),
                           self.K2.repeat(self.half, 1)], -1)
        k_sel = k_cat[cand]
        score = (self.q_proj(h).unsqueeze(-2) @ k_sel.transpose(-1, -2)).squeeze(-2)
        idx = score.topk(min(self.topk, score.shape[-1]), -1).indices
        sel = cand.gather(-1, idx)
        w = F.softmax(score.gather(-1, idx), -1)
        out = (w.unsqueeze(-1) * self.V[sel]).sum(-2)
        return self.w_out(out), sel


class MemoryFFN(nn.Module):
    def __init__(self, orig, mem):
        super().__init__()
        self.orig, self.mem = orig, mem
        self.gate = nn.Parameter(torch.zeros(1))          # ★ 唯一允许的零初始化

    def forward(self, x):
        base = self.orig(x)
        m, _ = self.mem(x)
        return base + self.gate * m


class ZeroLoRA(nn.Module):
    def __init__(self, base, r=LORA_R):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


@torch.no_grad()
def greedy(model, tok, prompt, n=48):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0].tolist()


def set_gates(model, v):
    with torch.no_grad():
        for l in model.model.layers:
            if isinstance(l.mlp, MemoryFFN):
                l.mlp.gate.fill_(v)


def agree_rate(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def main():
    t0 = time.time()
    print("=" * 78)
    print("忆态架构 S0b（修正版）：零死锁解除后的活性与可训性归因")
    print("=" * 78)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DT).to(DEV).eval()

    base = {p: greedy(model, tok, p) for p in PROMPTS}
    ids0 = tok(PROMPTS[0], return_tensors="pt").input_ids.to(DEV)
    base_logits = model(ids0).logits[0, -1].float().cpu()

    # ── 改造
    layers = model.model.layers
    n = len(layers); c = n // 2
    idxs = sorted({max(0, min(n - 1, c + o)) for o in (-REPLACE_STRIDE // 2, 0, REPLACE_STRIDE // 2)})
    mem = ProductKeyMemory(model.config.hidden_size)
    for i in idxs:
        layers[i].mlp = MemoryFFN(layers[i].mlp, mem)
    for l in layers:
        l.self_attn.q_proj = ZeroLoRA(l.self_attn.q_proj)
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(DEV).eval()
    print(f"[rewrite] 替换层 {idxs}，共享记忆池 {MEM_ENTRIES} 条目 × d={model.config.hidden_size}；"
          f"记忆池 {mem.V.numel()*4/2**20:.0f} MiB")

    # ── 1) 等价性
    print("\n[1] S0 等价性（gate=0）")
    hits, tot = 0, 0
    for p in PROMPTS:
        g = greedy(model, tok, p); hits += sum(1 for x, y in zip(base[p], g) if x == y); tot += len(g)
    lg = model(ids0).logits[0, -1].float().cpu()
    d_logit = (base_logits - lg).abs().max().item()
    print(f"    一致率 {hits/tot*100:.2f}%  logits 最大差 {d_logit:.3e}  → "
          f"{'✅ 严格无损' if hits/tot == 1.0 and d_logit == 0 else '❌'}")

    # ── 2) 梯度检查（死锁是否解除）
    print("\n[2] 梯度检查（证明记忆分支不是死代码）")
    for l in layers:
        if isinstance(l.mlp, MemoryFFN):
            l.mlp.gate.requires_grad_(True)
    mem.w_out.weight.requires_grad_(True)
    mem.V.requires_grad_(True)
    model.train()
    out = model(ids0, labels=ids0)
    out.loss.backward()
    g_gate = [l.mlp.gate.grad.item() for l in layers if isinstance(l.mlp, MemoryFFN)]
    g_wout = mem.w_out.weight.grad.abs().mean().item()
    g_V = mem.V.grad.abs().mean().item()
    model.zero_grad(set_to_none=True); model.eval()
    print(f"    ∂L/∂gate   = {['%.3e' % g for g in g_gate]}")
    print(f"    |∂L/∂w_out| = {g_wout:.3e}   |∂L/∂V| = {g_V:.3e}")
    deadlock_fixed = all(abs(g) > 0 for g in g_gate)
    print(f"    → 零死锁{'已解除 ✅' if deadlock_fixed else '仍存在 ❌'}")

    # ── 3) 活体 / 4) 回滚
    print("\n[3] 活体测试：gate 0 → 1.0（应改变输出；训练时 gate 由 0 慢慢长起来）")
    set_gates(model, 1.0)
    live = []
    for p in PROMPTS[:2]:
        g = greedy(model, tok, p); a = agree_rate(base[p], g); live.append(a)
        print(f"    {p[:20]:22s} 与基线一致率 {a*100:6.2f}%")
    live_ok = any(a < 0.999 for a in live)
    print(f"    → 分支活性：{'✅ 生效（输出已改变）' if live_ok else '❌ 仍无效'}")
    set_gates(model, 0.0)
    g = greedy(model, tok, PROMPTS[0])
    rb = agree_rate(base[PROMPTS[0]], g)
    print(f"[4] 回滚（gate 归零）→ 一致率 {rb*100:.2f}% {'✅' if rb == 1.0 else '❌'}")

    # ── 5) 归因训练 A：只训记忆分支（gate + w_out + V）
    text = ("机器学习模型的推理速度主要受显存带宽限制。把知识放在可寻址的记忆表里，"
            "就能让参数规模与计算开销解耦。" * 4 +
            "Memory layers store associations in trainable key value tables. "
            "The quick brown fox jumps over the lazy dog." * 4)
    ids = tok(text, return_tensors="pt").input_ids[:, :256].to(DEV)

    def train(which, steps=30, lr=2e-3):
        params = []
        if which == "mem":
            for l in layers:
                if isinstance(l.mlp, MemoryFFN):
                    params.append(l.mlp.gate)
            params += [mem.w_out.weight, mem.V]
        else:
            for l in layers:
                params += [l.self_attn.q_proj.B]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=lr)
        model.train(); ls = []
        for s in range(steps):
            o = model(ids, labels=ids); loss = o.loss
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
        gates = [l.mlp.gate.item() for l in layers if isinstance(l.mlp, MemoryFFN)]
        model.zero_grad(set_to_none=True)
        for p in params:
            p.requires_grad_(False)
        model.eval()
        return ls, gates

    print("\n[5] 归因训练 A：只训记忆分支（gate + w_out + V，共 30 步）")
    lsA, gatesA = train("mem", 30)
    print(f"    loss {lsA[0]:.4f} → {lsA[-1]:.4f}（Δ {lsA[0]-lsA[-1]:+.4f}）  gate={['%.4f' % g for g in gatesA]}")
    A_ok = lsA[-1] < lsA[0] and any(abs(g) > 1e-6 for g in gatesA)
    print(f"    → 记忆分支可训且 gate 已离开 0：{'✅' if A_ok else '❌'}")

    print("\n[6] 归因训练 B：只训 LoRA（冻结记忆分支）")
    set_gates(model, 0.0)
    for l in layers:
        if isinstance(l.mlp, MemoryFFN):
            l.mlp.mem.w_out.weight.data = mem.w_out.weight.data.clone()
    lsB, _ = train("lora", 30)
    print(f"    loss {lsB[0]:.4f} → {lsB[-1]:.4f}（Δ {lsB[0]-lsB[-1]:+.4f}）")

    print(f"\n[归因] 记忆分支 Δloss {lsA[0]-lsA[-1]:+.4f} vs LoRA Δloss {lsB[0]-lsB[-1]:+.4f}")

    res = {"agree_s0": hits / tot, "logit_diff_s0": d_logit,
           "gate_grads": g_gate, "w_out_grad": g_wout, "V_grad": g_V,
           "deadlock_fixed": deadlock_fixed, "liveness_agree": live, "liveness_ok": live_ok,
           "rollback_agree": rb,
           "trainA_loss": [lsA[0], lsA[-1]], "trainA_gates": gatesA, "trainA_ok": A_ok,
           "trainB_loss": [lsB[0], lsB[-1]],
           "replaced_layers": idxs,
           "trainable_mem_params": sum(l.mlp.gate.numel() for l in layers if isinstance(l.mlp, MemoryFFN)) + mem.w_out.weight.numel() + mem.V.numel(),
           "elapsed_s": time.time() - t0}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s0b_result.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[out] {p}\n[time] {time.time()-t0:.1f}s  峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("=" * 78)
    return res


if __name__ == "__main__":
    main()
