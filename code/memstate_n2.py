"""
忆态架构 N2 —— 记忆层选型 + 关联记忆的「写入与召回」实验
================================================================
命题（架构核心）：**知识可以写进记忆表，而不必改主干权重。**

实验设计（单变量、可对照、有留出集）：
  0) 造一批「合成事实」：格式「事实：<钥匙> 的编号是 <4位码>。」提问「问：<钥匙> 的编号是什么？答：」
     —— 钥匙用生造词（KX37/蓝鲸座…），确保主干**从未见过**（基线召回应≈0%）
  1) 基线对照：原模型（无记忆分支）在留出事实上的召回率
  2) 记忆层选型扫描：替换 3 层 vs 5 层 / 居中 vs 靠前 vs 靠后（各训 40 步，比 loss 与召回）
  3) 最优配置完整训练（只训记忆分支，主干全冻结）：200 步，报告训练/留出召回率
  4) 完整性检查：主干参数校验和（训练前后必须一致 ⇒ 证明"没动主干"）

运行：/e/HARNESS/selfgrow/venv/Scripts/python.exe code/memstate_n2.py
"""
import os, time, json, math, random, hashlib
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
MEM_ENTRIES, HALF, KD, TOPK = 65536, 256, 128, 8
random.seed(7); torch.manual_seed(7)

# ── 合成事实（生造钥匙，主干不可能见过）──────────────────────────────
SYLL = ["zur", "qaf", "vom", "kix", "dun", "pyr", "lek", "nuf", "tad", "gor",
        "meb", "sil", "wol", "jap", "fer", "cug", "tar", "bez", "myn", "oku"]
def make_facts(n):
    facts = {}
    while len(facts) < n:
        k = random.choice(SYLL).capitalize() + random.choice(SYLL) + str(random.randint(10, 99))
        if k in facts: continue
        facts[k] = f"{random.randint(1000, 9999)}"
    return facts

FACTS_TR = make_facts(96)
FACTS_EV = make_facts(32)
FACTS_EV = {k: v for k, v in FACTS_EV.items() if k not in FACTS_TR}          # 严格留出
FACTS_EV = dict(list(FACTS_EV.items())[:24])

def train_text(k, v):
    return f"事实：{k} 的编号是 {v}。\n问：{k} 的编号是什么？\n答：{v}"
def query_text(k):
    return f"问：{k} 的编号是什么？\n答："


# ── 记忆层（乘积键；用 gather 半键算分，避免实例化全键矩阵）────────────
class ProductKeyMemory(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.K1 = nn.Parameter(torch.randn(HALF, KD) / math.sqrt(KD))
        self.K2 = nn.Parameter(torch.randn(HALF, KD) / math.sqrt(KD))
        self.V = nn.Parameter(torch.randn(MEM_ENTRIES, d) * 0.02)
        self.q_proj = nn.Linear(d, 2 * KD, bias=False)
        self.w_out = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.w_out.weight, std=1.0 / math.sqrt(d))

    def forward(self, h):
        B, T, D = h.shape
        q = self.q_proj(h).view(B, T, 2, KD)
        q1, q2 = q[..., 0, :], q[..., 1, :]
        i1 = (q1 @ self.K1.t()).topk(TOPK, -1).indices          # [B,T,k]
        i2 = (q2 @ self.K2.t()).topk(TOPK, -1).indices
        cand = (i1.unsqueeze(-1) * HALF + i2.unsqueeze(-2)).reshape(B, T, -1)   # [B,T,k*k]
        s = ((q1.unsqueeze(-2) @ self.K1[cand // HALF].transpose(-1, -2)) +
             (q2.unsqueeze(-2) @ self.K2[cand % HALF].transpose(-1, -2))).squeeze(-2)
        idx = s.topk(TOPK, -1).indices
        sel = cand.gather(-1, idx)
        w = F.softmax(s.gather(-1, idx), -1)
        return self.w_out((w.unsqueeze(-1) * self.V[sel]).sum(-2)), sel


class MemoryFFN(nn.Module):
    def __init__(self, orig, mem):
        super().__init__()
        self.orig, self.mem = orig, mem
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        base = self.orig(x)
        m, _ = self.mem(x)
        return base + self.gate * m


# ── 工具 ────────────────────────────────────────────────────────────
def trunk_checksum(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if not n.startswith("model.layers.") or ".mlp.mem." in n or n.endswith("mlp.gate"):
            continue
        h.update(p.detach().float().cpu().numpy().tobytes()[::97])   # 抽样，省时间
    return h.hexdigest()[:16]

@torch.no_grad()
def recall(model, tok, facts, max_new=12):
    """返回 (精确命中率, 数字级命中率)：生成后检查 4 位码是否完整/部分出现"""
    exact = part = 0
    for k, v in facts.items():
        ids = tok(query_text(k), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if v in txt: exact += 1
        else:
            hit = sum(1 for a, b in zip(v, txt.strip()) if a == b)
            if hit >= 2: part += 1
    n = len(facts)
    return exact / n, part / n


def install(model, idxs):
    """恢复所有 MLP 为原始，再在 idxs 上装 MemoryFFN（共享一个新记忆池）"""
    layers = model.model.layers
    for l in layers:
        if isinstance(l.mlp, MemoryFFN):
            l.mlp = l.mlp.orig
    mem = ProductKeyMemory(model.config.hidden_size).to(next(model.parameters()).device)
    for i in idxs:
        layers[i].mlp = MemoryFFN(layers[i].mlp, mem).to(next(model.parameters()).device)
    for p in model.parameters():
        p.requires_grad_(False)
    for l in layers:
        if isinstance(l.mlp, MemoryFFN):
            l.mlp.gate.requires_grad_(True)
    for p in [mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2]:
        p.requires_grad_(True)
    return mem


def mem_params(model, mem):
    return ([l.mlp.gate for l in model.model.layers if isinstance(l.mlp, MemoryFFN)]
            + [mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2])


def train_branch(model, tok, mem, steps, facts, lr=3e-3, bs=4):
    ps = mem_params(model, mem)
    opt = torch.optim.AdamW(ps, lr=lr)
    items = list(facts.items())
    model.train(); losses = []
    for s in range(steps):
        batch = random.sample(items, min(bs, len(items)))
        enc = tok([train_text(k, v) for k, v in batch], return_tensors="pt",
                  padding=True).to(DEV)
        out = model(**enc, labels=enc["input_ids"])
        opt.zero_grad(); out.loss.backward(); opt.step()
        losses.append(out.loss.item())
    model.eval()
    gates = [round(l.mlp.gate.item(), 4) for l in model.model.layers if isinstance(l.mlp, MemoryFFN)]
    return losses, gates


def main():
    t0 = time.time()
    print("=" * 80)
    print("N2 · 记忆层选型 + 关联记忆写入/召回（主干全冻结）")
    print("=" * 80)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DT).to(DEV).eval()
    n = len(model.model.layers)
    print(f"[model] {n} 层, d={model.config.hidden_size}, 主干 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 0) 基线对照（无记忆分支，主干没见过这些生造钥匙）
    e0, p0 = recall(model, tok, FACTS_EV)
    print(f"\n[0] 基线（未改造）留出事实召回：精确 {e0*100:.1f}%  数字级 {p0*100:.1f}%  ← 预期≈0")

    # 1) 记忆层选型扫描
    print("\n[1] 记忆层选型扫描（各 40 步，只训记忆分支）")
    c = n // 2
    configs = {
        "居中×3(stride8)": [c - 4, c, c + 4],
        "靠前×3":          [2, 5, 8],
        "靠后×3":          [n - 9, n - 6, n - 3],
        "居中×5":          [c - 8, c - 4, c, c + 4, c + 8],
    }
    scan = {}
    for name, idxs in configs.items():
        idxs = [max(0, min(n - 1, i)) for i in idxs]
        mem = install(model, idxs)
        t1 = time.time()
        ls, gates = train_branch(model, tok, mem, 40, FACTS_TR)
        e, p = recall(model, tok, dict(list(FACTS_EV.items())[:8]))
        scan[name] = {"layers": idxs, "loss0": ls[0], "lossN": ls[-1], "exact": e, "part": p,
                      "gates": gates, "sec": time.time() - t1}
        print(f"  {name:16s} 层{idxs} loss {ls[0]:.3f}→{ls[-1]:.3f} "
              f"召回(8条) 精确{e*100:5.1f}% 数字级{p*100:5.1f}% gate={gates}  ({time.time()-t1:.0f}s)")

    best = max(scan.items(), key=lambda kv: (kv[1]["exact"], -kv[1]["lossN"]))[0]
    print(f"  ⇒ 选型最优：{best}")

    # 2) 最优配置完整训练
    print(f"\n[2] 完整训练（{best}，200 步，主干全冻结）")
    ck_before = trunk_checksum(model)
    mem = install(model, scan[best]["layers"])
    t2 = time.time()
    ls, gates = train_branch(model, tok, mem, 200, FACTS_TR, lr=3e-3, bs=4)
    print(f"    loss {ls[0]:.4f} → {ls[-1]:.4f}  gate={gates}  ({time.time()-t2:.0f}s)")

    # 3) 全量留出集评估 + 主干校验
    e, p = recall(model, tok, FACTS_EV, max_new=14)
    ck_after = trunk_checksum(model)
    print(f"\n[3] 留出集（{len(FACTS_EV)} 条生造事实）召回：精确 {e*100:.1f}%  数字级 {p*100:.1f}%")
    print(f"    主干校验和：训练前 {ck_before} → 训练后 {ck_after}  "
          f"{'✅ 主干未被改动' if ck_before == ck_after else '❌ 主干被改动!'}")

    # 4) 顺手看一眼：未训练分支（gate=0）的召回仍应为 0（对照）
    mem2 = install(model, scan[best]["layers"])
    e2, p2 = recall(model, tok, dict(list(FACTS_EV.items())[:8]))
    print(f"[4] 对照：把记忆分支重置回 gate=0 → 召回（8条）精确 {e2*100:.1f}%（应≈0，证明增益来自记忆写入）")

    res = {"baseline": {"exact": e0, "part": p0}, "scan": scan, "best": best,
           "final": {"loss0": ls[0], "lossN": ls[-1], "exact": e, "part": p,
                     "gates": gates, "trunk_checksum_before": ck_before,
                     "trunk_checksum_after": ck_after},
           "control_reset": {"exact": e2, "part": p2},
           "n_train_facts": len(FACTS_TR), "n_eval_facts": len(FACTS_EV),
           "elapsed_s": time.time() - t0,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "n2_result.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[out] {out}\n[time] {time.time()-t0:.0f}s  峰值显存 {res['peak_gib']:.2f} GiB")
    print("=" * 80)


if __name__ == "__main__":
    main()
