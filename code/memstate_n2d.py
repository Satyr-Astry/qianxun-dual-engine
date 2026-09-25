"""
忆态架构 N2d —— 把「记忆分支到底有没有用」钉死
================================================================
继承 N2b/N2c 结论：①查询式训练（强制走记忆）②必须含读出侧 ③留出键只作负对照
④主干不变验证用张量级 max-abs-diff

本轮改进：
  · 训练键 128 条、留出键 32 条（负对照）；主指标 = 训练键 exact 召回
  · 多种子（2 个种子跑决定性对子）+ 多深度注入 + 难度刻度（2位/4位/6位码）
  · **每轮从磁盘重载模型**（彻底避免臂间污染，代价仅 ~5s/轮）
  · 显存用 **nvidia-smi 独立 1Hz 采样**（不再信 torch 的统计口径）

矩阵（每组都有读出侧 = LM-head LoRA + 末 4 层 q/v LoRA）：
  R1 记忆分支居中×3            seed0 / seed1
  R2 无记忆分支（纯读出侧对照）  seed0 / seed1      ← 决定性对照
  R3 多深度注入（每 4 层一个）   seed0
  R4 记忆分支 + 2位码（更易）   seed0

运行：/e/HARNESS/selfgrow/venv/Scripts/python.exe code/memstate_n2d.py
"""
import os, time, json, math, random, subprocess, threading, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, STEPS, BS, LR = torch.float32, 200, 4, 3e-3
MEM_ENTRIES, HALF, KD, TOPK = 65536, 256, 128, 8
N_TR, N_EV, EVAL_TR, EVAL_EV = 128, 32, 48, 24

SYLL = ["zur","qaf","vom","kix","dun","pyr","lek","nuf","tad","gor",
        "meb","sil","wol","jap","fer","cug","tar","bez","myn","oku"]

def make_facts(n, seed, digits=4):
    rnd = random.Random(seed); out = {}
    while len(out) < n:
        k = rnd.choice(SYLL).capitalize() + rnd.choice(SYLL) + str(rnd.randint(10, 99))
        if k in out: continue
        out[k] = str(rnd.randint(10 ** (digits - 1), 10 ** digits - 1))
    return out

q_text  = lambda k: f"问：{k} 的编号是什么？\n答："
q_train = lambda k, v: f"问：{k} 的编号是什么？\n答：{v}"


# ── nvidia-smi 采样线程（独立口径）
class SMI(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self._stop_flag = False; self.peak = 0
    def run(self):
        while not self._stop_flag:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
                self.peak = max(self.peak, int(o.splitlines()[0]))
            except Exception:
                pass
            time.sleep(1.0)


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
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


@torch.no_grad()
def recall(model, tok, facts, max_new=10):
    ex = pt = 0
    for k, v in facts.items():
        ids = tok(q_text(k), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        if v in txt: ex += 1
        elif sum(1 for a, b in zip(v, txt) if a == b) >= max(2, len(v) // 2): pt += 1
    return ex / len(facts), pt / len(facts)


def load(seed):
    torch.manual_seed(seed); random.seed(seed)
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DT).to(DEV).eval()
    return m


def build(model, use_mem, mem_style="center3", readout_k=4, gate_warm=0.05):
    layers = model.model.layers; n = len(layers); c = n // 2
    dev = next(model.parameters()).device
    ps = []
    if use_mem:
        mem = ProductKeyMemory(model.config.hidden_size).to(dev)
        idxs = ([c - 4, c, c + 4] if mem_style == "center3"
                else list(range(2, n - 1, 4)))
        for i in idxs:
            layers[i].mlp = MemoryFFN(layers[i].mlp, mem).to(dev)
            with torch.no_grad(): layers[i].mlp.gate.fill_(gate_warm)
            layers[i].mlp.gate.requires_grad_(True); ps.append(layers[i].mlp.gate)
        for p in (mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2):
            p.requires_grad_(True); ps.append(p)
    else:
        idxs = []
    model.lm_head = LoRA(model.lm_head).to(dev); ps += [model.lm_head.A, model.lm_head.B]
    for i in range(n - readout_k, n):
        for attr in ("q_proj", "v_proj"):
            lo = LoRA(getattr(layers[i].self_attn, attr)).to(dev)
            setattr(layers[i].self_attn, attr, lo); ps += [lo.A, lo.B]
    for p in model.parameters():
        if not any(p is q for q in ps):
            p.requires_grad_(False)
    return ps, idxs, n


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if n.startswith("model.layers.") and ".mlp.mem." not in n
            and "mlp.gate" not in n and ".A" not in n and ".B" not in n}


def maxdiff(snap, model):
    w = 0.0
    for n, p in model.named_parameters():
        if n in snap:
            w = max(w, (p.detach() - snap[n]).abs().max().item())
    return w


def run(name, use_mem, seed, mem_style="center3", digits=4, readout_k=4, tok=None):
    t0 = time.time()
    TR = make_facts(N_TR, 1000 + seed, digits)
    EV = {k: v for k, v in make_facts(N_EV, 2000 + seed, digits).items() if k not in TR}
    EV = dict(list(EV.items())[:EVAL_EV])
    TRS = dict(list(TR.items())[:EVAL_TR])

    model = load(seed)
    snap = snapshot(model)
    ps, idxs, n = build(model, use_mem, mem_style, readout_k)
    n_ps = sum(p.numel() for p in ps)
    opt = torch.optim.AdamW(ps, lr=LR)
    items = list(TR.items()); model.train(); ls = []
    for s in range(STEPS):
        b = random.sample(items, BS)
        enc = tok([q_train(k, v) for k, v in b], return_tensors="pt", padding=True).to(DEV)
        out = model(**enc, labels=enc["input_ids"])
        opt.zero_grad(); out.loss.backward(); opt.step(); ls.append(out.loss.item())
    model.eval()
    ex_tr, pt_tr = recall(model, tok, TRS)
    ex_ev, pt_ev = recall(model, tok, EV)
    md = maxdiff(snap, model)
    gates = [round(l.mlp.gate.item(), 3) for l in model.model.layers if isinstance(l.mlp, MemoryFFN)]
    r = dict(name=name, use_mem=use_mem, seed=seed, mem_style=mem_style, digits=digits,
             layers=idxs, trainable=n_ps, loss0=ls[0], lossN=ls[-1],
             ex_train=ex_tr, pt_train=pt_tr, ex_hold=ex_ev, pt_hold=pt_ev,
             trunk_maxdiff=md, gates=gates, sec=time.time() - t0)
    print(f"  {name:34s} loss {ls[0]:.2f}→{ls[-1]:.2f}  训练键 exact {ex_tr*100:5.1f}% "
          f"(数字级 {pt_tr*100:5.1f}%)  留出 exact {ex_ev*100:4.1f}%  主干diff {md:.1e}  "
          f"可训 {n_ps/1e6:.2f}M  {r['sec']:.0f}s", flush=True)
    del model, opt, ps; gc.collect(); torch.cuda.empty_cache()
    return r


def main():
    t0 = time.time()
    print("=" * 100); print("N2d · 放大统计 / 多深度 / 难度刻度（每轮重载模型，独立显存口径）"); print("=" * 100)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR); tok.pad_token = tok.eos_token
    smi = SMI(); smi.start()

    plan = [
        ("R1 记忆居中×3",        True,  "center3", 4),
        ("R2 无记忆(纯读出侧)",   False, "center3", 4),
        ("R3 多深度注入(每4层)",  True,  "multi4",  4),
        ("R4 记忆+2位码(更易)",   True,  "center3", 2),
    ]
    results = []
    for name, um, style, dg in plan:
        for seed in (0, 1) if name.startswith(("R1", "R2")) else (0,):
            results.append(run(f"{name} s{seed}", um, seed, style, dg, tok=tok))
        print("  " + "-" * 92, flush=True)

    def agg(prefix):
        rs = [r for r in results if r["name"].startswith(prefix)]
        ex = [r["ex_train"] for r in rs]
        return (sum(ex) / len(ex), min(ex), max(ex), len(rs))
    print("\n" + "=" * 100)
    print("汇总（训练键 exact 召回，>0 才算“写进去且读得出来”）")
    line = []
    for p in ("R1", "R2", "R3", "R4"):
        m, lo, hi, k = agg(p)
        line.append(f"{p}: 均值{m*100:5.1f}% [{lo*100:.1f}-{hi*100:.1f}] (n={k})")
    print("  " + "  |  ".join(line))
    r1 = agg("R1"); r2 = agg("R2")
    print(f"\n  记忆分支净贡献（R1−R2）= {(r1[0]-r2[0])*100:+.1f} pp"
          f"  {'✅ 方向为正' if r1[0] > r2[0] else '❌ 无正贡献'}")
    print(f"  主干不变性：所有轮 max-diff 最大值 = {max(r['trunk_maxdiff'] for r in results):.1e}")
    smi._stop_flag = True
    print(f"  显存（nvidia-smi 独立采样峰值）= {smi.peak} MiB / 12282 MiB")
    json.dump({"results": results, "peak_smi_mib": smi.peak, "elapsed_s": time.time() - t0},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "n2d_result.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[time] {time.time()-t0:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
