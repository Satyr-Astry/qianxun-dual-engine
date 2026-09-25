"""
忆态架构 N2c —— 决定性对照：是「记忆架构不行」还是「任务/预算不足」？
================================================================
N2 / N2b 的结论：
  · 只训记忆分支 + 冻结主干 → 查询式（强制走记忆）召回 **0%**（3 个注入点均如此）
  · 加 LM-head LoRA → loss 大降（4.51→1.24）但召回仍 0% exact / 12.5% digit
    ⇒ 嫌疑：**读出侧（下游冻结层 + LM head）从未被训练去"读"注入的记忆向量**

本实验（三臂单变量 + 一个被修好的仪表）：
  · 仪表修复：主干不变性改用**张量级 max-abs-diff**（不用抽样 hash）
  · Arm 1 「记忆 + 读出侧」：记忆分支（居中×3）+ LM-head LoRA + 末 4 层 LoRA
  · Arm 2 「纯读出侧对照」：只有 LM-head LoRA + 末 4 层 LoRA（**无记忆分支**）
       → 若 Arm 2 也记不住 ⇒ 是任务/预算不足；若 Arm 2 记住了而 Arm 1 没有 ⇒ 记忆分支是拖累
  · Arm 3 复现「只训记忆分支（N2b 最优配置 B）」作为同批次基准
评估：训练键 32 条 + 留出键 16 条（负对照）的精确/数字级召回

运行：/e/HARNESS/selfgrow/venv/Scripts/python.exe code/memstate_n2c.py
"""
import os, time, json, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
MEM_ENTRIES, HALF, KD, TOPK = 65536, 256, 128, 8
STEPS, BS, LR = 250, 4, 3e-3
random.seed(11); torch.manual_seed(11)

SYLL = ["zur","qaf","vom","kix","dun","pyr","lek","nuf","tad","gor",
        "meb","sil","wol","jap","fer","cug","tar","bez","myn","oku"]
def make_facts(n, seed):
    rnd = random.Random(seed); out = {}
    while len(out) < n:
        k = rnd.choice(SYLL).capitalize() + rnd.choice(SYLL) + str(rnd.randint(10, 99))
        if k in out: continue
        out[k] = f"{rnd.randint(1000, 9999)}"
    return out
TR, EV = make_facts(96, 21), make_facts(48, 22)
EV = dict(list({k: v for k, v in EV.items() if k not in TR}.items())[:16])
TRS = dict(list(TR.items())[:32])

q_text  = lambda k: f"问：{k} 的编号是什么？\n答："
q_train = lambda k, v: f"问：{k} 的编号是什么？\n答：{v}"


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
        B, T, _ = h.shape
        q = self.q_proj(h).view(B, T, 2, KD)
        q1, q2 = q[..., 0, :], q[..., 1, :]
        i1 = (q1 @ self.K1.t()).topk(TOPK, -1).indices
        i2 = (q2 @ self.K2.t()).topk(TOPK, -1).indices
        cand = (i1.unsqueeze(-1) * HALF + i2.unsqueeze(-2)).reshape(B, T, -1)
        s = ((q1.unsqueeze(-2) @ self.K1[cand // HALF].transpose(-1, -2)) +
             (q2.unsqueeze(-2) @ self.K2[cand % HALF].transpose(-1, -2))).squeeze(-2)
        idx = s.topk(TOPK, -1).indices
        sel = cand.gather(-1, idx)
        w = F.softmax(s.gather(-1, idx), -1)
        return self.w_out((w.unsqueeze(-1) * self.V[sel]).sum(-2)), sel


class MemoryFFN(nn.Module):
    def __init__(self, orig, mem):
        super().__init__()
        self.orig, self.mem, self.gate = orig, mem, nn.Parameter(torch.zeros(1))
    def forward(self, x):
        base = self.orig(x); m, _ = self.mem(x)
        return base + self.gate * m


class LoRA(nn.Module):
    def __init__(self, base, r=16):
        super().__init__()
        self.base, self.r = base, r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


@torch.no_grad()
def recall(model, tok, facts, max_new=10):
    exact = part = 0
    for k, v in facts.items():
        ids = tok(q_text(k), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if v in txt: exact += 1
        elif sum(1 for a, b in zip(v, txt.strip()) if a == b) >= 2: part += 1
    return exact / len(facts), part / len(facts)


def snapshot_trunk(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if n.startswith("model.layers.") and ".mlp.mem." not in n and "mlp.gate" not in n
            and ".A" not in n and ".B" not in n}


def trunk_diff(snap, model):
    worst = 0.0; n_worst = ""
    for n, p in model.named_parameters():
        if n in snap:
            d = (p.detach() - snap[n]).abs().max().item()
            if d > worst: worst, n_worst = d, n
    return worst, n_worst


def strip(model):
    """还原所有改造（记忆分支 + LoRA）"""
    for l in model.model.layers:
        if isinstance(l.mlp, MemoryFFN):
            l.mlp = l.mlp.orig
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            m = getattr(l.self_attn if attr in ("q_proj","k_proj","v_proj","o_proj") else l.mlp, attr, None)
            if isinstance(m, LoRA): 
                if attr in ("q_proj","k_proj","v_proj","o_proj"): l.self_attn.__setattr__(attr, m.base)
                else: l.mlp.__setattr__(attr, m.base)
    if isinstance(model.lm_head, LoRA):
        model.lm_head = model.lm_head.base
    for p in model.parameters():
        p.requires_grad_(False)


def build(model, use_mem, use_readout, n_last=4, gate_warm=0.0):
    strip(model)
    dev = next(model.parameters()).device
    layers = model.model.layers; n = len(layers); c = n // 2
    ps = []
    if use_mem:
        mem = ProductKeyMemory(model.config.hidden_size).to(dev)
        for i in [c - 4, c, c + 4]:
            layers[i].mlp = MemoryFFN(layers[i].mlp, mem).to(dev)
            with torch.no_grad(): layers[i].mlp.gate.fill_(gate_warm)
            layers[i].mlp.gate.requires_grad_(True); ps.append(layers[i].mlp.gate)
        for p in (mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2):
            p.requires_grad_(True); ps.append(p)
    if use_readout:
        model.lm_head = LoRA(model.lm_head).to(dev)
        ps += [model.lm_head.A, model.lm_head.B]
        for i in range(n - n_last, n):
            for attr in ("q_proj", "v_proj"):
                sub = getattr(layers[i].self_attn, attr)
                lora = LoRA(sub).to(dev)
                setattr(layers[i].self_attn, attr, lora)
                ps += [lora.A, lora.B]
    return ps


def train(model, tok, ps, facts, steps=STEPS):
    opt = torch.optim.AdamW(ps, lr=LR)
    items = list(facts.items()); model.train(); ls = []
    for s in range(steps):
        b = random.sample(items, BS)
        enc = tok([q_train(k, v) for k, v in b], return_tensors="pt", padding=True).to(DEV)
        out = model(**enc, labels=enc["input_ids"])
        opt.zero_grad(); out.loss.backward(); opt.step(); ls.append(out.loss.item())
    model.eval(); return ls


def main():
    t0 = time.time()
    print("=" * 88)
    print("N2c · 三臂对照：读出侧是否才是关键？")
    print("=" * 88)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DT).to(DEV).eval()

    e0, _ = recall(model, tok, TRS); h0, _ = recall(model, tok, EV)
    print(f"[0] 基线（未改造）：训练键 精确{e0*100:.1f}% / 留出键 精确{h0*100:.1f}%")

    arms = [
        ("Arm1 记忆分支+读出侧(LMhead+末4层LoRA)", True,  True,  0.05),
        ("Arm2 纯读出侧对照(无记忆分支)",          False, True,  0.0),
        ("Arm3 只训记忆分支(N2b最优B复现)",        True,  False, 0.05),
    ]
    res = {}
    for name, use_mem, use_read, gw in arms:
        strip(model)
        snap = snapshot_trunk(model)
        ps = build(model, use_mem, use_read, gate_warm=gw)
        n_ps = sum(p.numel() for p in ps)
        ls = train(model, tok, ps, TR)
        ex_tr, pt_tr = recall(model, tok, TRS)
        ex_ev, pt_ev = recall(model, tok, EV)
        worst, wname = trunk_diff(snap, model)
        gates = [round(l.mlp.gate.item(), 3) for l in model.model.layers if isinstance(l.mlp, MemoryFFN)]
        res[name] = dict(loss0=ls[0], lossN=ls[-1], ex_train=ex_tr, pt_train=pt_tr,
                         ex_hold=ex_ev, pt_hold=pt_ev, trainable=n_ps,
                         trunk_maxdiff=worst, trunk_worst_param=wname, gates=gates)
        print(f"\n[{name}]  可训 {n_ps/1e6:.2f}M 参数")
        print(f"   loss {ls[0]:.3f} → {ls[-1]:.3f}" + (f"   gate={gates}" if gates else ""))
        print(f"   训练键召回：精确 {ex_tr*100:5.1f}%  数字级 {pt_tr*100:5.1f}%")
        print(f"   留出键召回：精确 {ex_ev*100:5.1f}%  数字级 {pt_ev*100:5.1f}%")
        print(f"   主干张量最大改动 = {worst:.3e}  ({wname or '无'})  "
              f"{'✅ 主干严格未变' if worst == 0 else '❌ 主干被改动'}")

    json.dump({"baseline": {"train": e0, "hold": h0}, "arms": res,
               "elapsed_s": time.time() - t0,
               "peak_gib": torch.cuda.max_memory_allocated()/2**30},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "n2c_result.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[time] {time.time()-t0:.0f}s  峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("=" * 88)


if __name__ == "__main__":
    main()
