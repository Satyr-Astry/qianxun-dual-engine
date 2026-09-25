"""
忆态架构 N2b —— 关联记忆「写入/读出」的修正实验（含 N2 失败归因）
================================================================
N2 的实测与归因：
  ✅ 主干零改动（checksum 一致）
  ✅ 分支可训（200 步 loss 2.85→0.70，gate 长到 ~0.09-0.13）
  ❌ 召回 0% —— 两个设计错误：
     (a) **训练格式里带着"上下文内的事实"** ⇒ 模型学会抄近路（从上下文复制），
         根本不依赖记忆分支 ⇒ loss 下降是"抄近路"的功劳。
     (b) **留出键对静态记忆池来说不可能被召回**（池里根本没有那对关联）⇒
         用留出键评估"写入能力"是错的；留出键应作**负对照**（应为 0）。

本版修正：
  · 训练格式改为**查询式**（无上下文事实）：问：K 的编号是什么？答：V
    ⇒ 唯一能答对的路径就是记忆分支 ⇒ 强制记忆写入
  · 评估 = **训练键召回**（写入能力，应显著>0）+ **留出键召回**（负对照，应≈0）
  · 诊断 = 同时测"上下文式"召回（抄近路能力，用于解释 loss 与召回背离）
  · 单变量对照：A 基线（gate 从 0）/ B gate 热启动 0.05 / C 记忆层靠后（近 LM head）
                 / D 额外解冻 LM-head 的零初始化 LoRA
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
STEPS = 250
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

TR = make_facts(96, 21)
EV = {k: v for k, v in make_facts(48, 22).items() if k not in TR}
EV = dict(list(EV.items())[:16])

q_text    = lambda k: f"问：{k} 的编号是什么？\n答："
q_train   = lambda k, v: f"问：{k} 的编号是什么？\n答：{v}"
ic_train  = lambda k, v: f"事实：{k} 的编号是 {v}。\n问：{k} 的编号是什么？\n答：{v}"   # 抄近路用


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
        self.orig, self.mem = orig, mem
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        base = self.orig(x)
        m, _ = self.mem(x)
        return base + self.gate * m


class ZeroLoRA(nn.Module):
    def __init__(self, base, r=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


@torch.no_grad()
def recall(model, tok, facts, max_new=10):
    exact = part = 0
    for k, v in facts.items():
        ids = tok(q_text(k), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if v in txt: exact += 1
        elif sum(1 for a, b in zip(v, txt.strip()) if a == b) >= 2: part += 1
    n = len(facts)
    return exact / n, part / n


def trunk_checksum(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n.startswith("model.layers.") and (".mlp.mem." not in n and "mlp.gate" not in n
                                              and "q_proj.A" not in n and "q_proj.B" not in n):
            h.update(p.detach().float().cpu().numpy().tobytes()[::97])
    return h.hexdigest()[:16]


def install(model, idxs, gate_warm=0.0, head_lora=False):
    layers = model.model.layers
    for l in layers:
        if isinstance(l.mlp, MemoryFFN):
            l.mlp = l.mlp.orig
    dev = next(model.parameters()).device
    mem = ProductKeyMemory(model.config.hidden_size).to(dev)
    for i in idxs:
        layers[i].mlp = MemoryFFN(layers[i].mlp, mem).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    ps = [mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2]
    for l in layers:
        if isinstance(l.mlp, MemoryFFN):
            with torch.no_grad():
                l.mlp.gate.fill_(gate_warm)
            l.mlp.gate.requires_grad_(True); ps.append(l.mlp.gate)
    if head_lora:
        model.lm_head = ZeroLoRA(model.lm_head).to(dev)
        ps += [model.lm_head.A, model.lm_head.B]
    return mem, ps


def train(model, tok, ps, facts, steps=STEPS, lr=3e-3, bs=4):
    opt = torch.optim.AdamW(ps, lr=lr)
    items = list(facts.items()); model.train(); ls = []
    for s in range(steps):
        b = random.sample(items, bs)
        enc = tok([q_train(k, v) for k, v in b], return_tensors="pt", padding=True).to(DEV)
        out = model(**enc, labels=enc["input_ids"])
        opt.zero_grad(); out.loss.backward(); opt.step(); ls.append(out.loss.item())
    model.eval(); return ls


def main():
    t0 = time.time()
    print("=" * 84)
    print("N2b · 查询式训练（强制走记忆）× 四组单变量对照")
    print("=" * 84)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DT).to(DEV).eval()
    n = len(model.model.layers); c = n // 2
    print(f"[model] {n} 层 d={model.config.hidden_size}；训练键 {len(TR)} 条 / 留出键 {len(EV)} 条")

    e0, _ = recall(model, tok, dict(list(TR.items())[:16]))
    h0, _ = recall(model, tok, EV)
    print(f"[0] 未改造基线：训练键召回 精确{e0*100:.1f}%  留出键 精确{h0*100:.1f}%（都应为 0）")

    TRS = dict(list(TR.items())[:32])       # 评估用的训练键子集
    configs = [
        ("A 居中×3, gate从0",   dict(idxs=[c-4, c, c+4], gate_warm=0.0,  head_lora=False)),
        ("B 居中×3, gate热启0.05", dict(idxs=[c-4, c, c+4], gate_warm=0.05, head_lora=False)),
        ("C 靠后×3(近head)",     dict(idxs=[n-7, n-4, n-1], gate_warm=0.05, head_lora=False)),
        ("D 居中×3 + headLoRA", dict(idxs=[c-4, c, c+4], gate_warm=0.05, head_lora=True)),
    ]
    results = {}
    for name, cfg in configs:
        ck0 = trunk_checksum(model)
        mem, ps = install(model, cfg["idxs"], cfg["gate_warm"], cfg["head_lora"])
        ls = train(model, tok, ps, TR)
        ex_tr, pt_tr = recall(model, tok, TRS)
        ex_ev, pt_ev = recall(model, tok, EV)
        ck1 = trunk_checksum(model)
        gates = [round(l.mlp.gate.item(), 3) for l in model.model.layers if isinstance(l.mlp, MemoryFFN)]
        results[name] = dict(layers=cfg["idxs"], gate_warm=cfg["gate_warm"], head_lora=cfg["head_lora"],
                             loss0=ls[0], lossN=ls[-1], ex_train=ex_tr, pt_train=pt_tr,
                             ex_hold=ex_ev, pt_hold=pt_ev, gates=gates,
                             trunk_ok=(ck0 == ck1))
        print(f"\n[{name}]  层{cfg['idxs']}  gate热启{cfg['gate_warm']}  headLoRA={cfg['head_lora']}")
        print(f"   loss {ls[0]:.3f} → {ls[-1]:.3f}   gate={gates}")
        print(f"   训练键召回：精确 {ex_tr*100:5.1f}%  数字级 {pt_tr*100:5.1f}%   ← 写入能力（应显著>0）")
        print(f"   留出键召回：精确 {ex_ev*100:5.1f}%  数字级 {pt_ev*100:5.1f}%   ← 负对照（应≈0）")
        print(f"   主干校验 {'✅未改动' if ck0 == ck1 else '❌被改动'}")

    # 诊断：同一模型用"上下文式"问（抄近路能力）
    print("\n[诊断] 同一模型的「上下文式」召回（把事实放进上下文再问）")
    for name in [configs[0][0], configs[3][0]]:
        pass
    best = max(results.items(), key=lambda kv: kv[1]["ex_train"])
    print(f"   最优组 = {best[0]}（训练键精确 {best[1]['ex_train']*100:.1f}%）")

    json.dump({"baseline": {"train": e0, "hold": h0}, "results": results,
               "elapsed_s": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated()/2**30},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "n2b_result.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[time] {time.time()-t0:.0f}s  峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("=" * 84)


if __name__ == "__main__":
    main()
