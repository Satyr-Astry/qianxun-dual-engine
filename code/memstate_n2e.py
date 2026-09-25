"""
忆态架构 N2e —— Titans 式「运行时动态写入」：真正的写入即生效
================================================================
N2d 结论：静态记忆池（梯度训练写入）+ 冻结主干 ⇒ 关联召回 0%（净贡献 −1.0pp）
⇒ 静态池只能记"训练时写进去的"，无法改写"没见过的键"。

本实验（架构核心命题的真正形态）：
  ① 写入阶段：喂「记住：K 的编号是 V。」→ 各注入层的记忆矩阵 M 做**内部梯度步**更新
              M ← (1-α)·M − θ·∇ℓ(M; x)，  ℓ = ‖Mᵀk − v‖²（惊讶/关联损失）
  ② 读取阶段：喂「问：K 的编号是什么？\n答：」→ 用 M 做前向读出（**不再写入**）
  ③ 评测：**留出键在推理时才写入**（训练时从未出现）→ 若能召回 = 真·运行时写入成立

可训部分（外循环）：Wk/Wv/Wq/Wo + α/θ（每层约 0.6M），主干全冻结。
负对照：同一模型关闭记忆（gate=0 或 M 置零）→ 应≈0%。

运行：/e/HARNESS/selfgrow/venv/Scripts/python.exe code/memstate_n2e.py
"""
import os, time, json, math, random, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
STEPS, BS, LR = 240, 4, 2e-3
KDIM, RANK, INJECT = 128, 64, 3          # 键维 / 值秩 / 注入层数
SYLL = ["zur","qaf","vom","kix","dun","pyr","lek","nuf","tad","gor",
        "meb","sil","wol","jap","fer","cug","tar","bez","myn","oku"]

def make_facts(n, seed, digits=4):
    rnd = random.Random(seed); out = {}
    while len(out) < n:
        k = rnd.choice(SYLL).capitalize() + rnd.choice(SYLL) + str(rnd.randint(10, 99))
        if k in out: continue
        out[k] = str(rnd.randint(10 ** (digits - 1), 10 ** digits - 1))
    return out

WRITE = lambda k, v: f"记住：{k} 的编号是 {v}。\n"
QUERY = lambda k: f"问：{k} 的编号是什么？\n答："


class DynMem(nn.Module):
    """Titans-lite：内部梯度步写入 + 前向读出。M 是运行时状态(Xi)"""
    def __init__(self, d):
        super().__init__()
        self.Wk = nn.Linear(d, KDIM, bias=False)
        self.Wv = nn.Linear(d, RANK, bias=False)
        self.Wq = nn.Linear(d, KDIM, bias=False)
        self.Wo = nn.Linear(RANK, d, bias=False)
        nn.init.normal_(self.Wo.weight, std=1.0 / math.sqrt(d))
        self.alpha = nn.Parameter(torch.tensor(0.0))    # 遗忘（sigmoid）
        self.theta = nn.Parameter(torch.tensor(0.0))    # 写入步长（sigmoid）
        self.M = None
        self.gate = nn.Parameter(torch.zeros(1))

    def reset(self, device):
        self.M = None

    def write(self, h):
        """h:[B,T,d] → 内部梯度步更新 M（可微，供外循环学习 W*/α/θ）"""
        B, T, _ = h.shape
        k = self.Wk(h)                                   # [B,T,kdim]
        v = self.Wv(h)                                   # [B,T,rank]
        if self.M is None:
            self.M = torch.zeros(k.shape[0], KDIM, RANK, device=h.device, dtype=h.dtype)
        pred = k @ self.M                                # [B,T,rank]
        grad = k.transpose(1, 2) @ (pred - v) / max(1, T)  # [B,kdim,rank]
        a = torch.sigmoid(self.alpha); th = torch.sigmoid(self.theta)
        self.M = (1 - a) * self.M - th * grad
        return h

    def read(self, h):
        if self.M is None:
            return torch.zeros_like(h)
        q = self.Wq(h)                                   # [B,T,kdim]
        y = q @ self.M                                   # [B,T,rank]
        return self.gate * self.Wo(y)


class DynLayer(nn.Module):
    """mode='write' → 先做内部梯度步写入状态；两种模式都做读出"""
    def __init__(self, orig_mlp, d):
        super().__init__()
        self.orig, self.mem, self.mode = orig_mlp, DynMem(d), "read"

    def forward(self, x):
        if self.mode == "write":
            self.mem.write(x)
        return self.orig(x) + self.mem.read(x)


def build(model, n_inject=INJECT):
    layers = model.model.layers; n = len(layers); c = n // 2
    dev = next(model.parameters()).device
    d = model.config.hidden_size
    mems, dyns = [], []
    for i in [c - 4, c, c + 4][:n_inject]:
        dl = DynLayer(layers[i].mlp, d).to(dev)
        layers[i].mlp = dl
        mems.append(dl.mem); dyns.append(dl)
    for p in model.parameters():
        p.requires_grad_(False)
    ps = []
    for m in mems:
        m.gate.requires_grad_(True); ps.append(m.gate)
        for p in (m.Wk.weight, m.Wv.weight, m.Wq.weight, m.Wo.weight, m.alpha, m.theta):
            p.requires_grad_(True); ps.append(p)
    return mems, dyns, ps


def write_pass(model, layers_dyn, tok, fact_text):
    """写入阶段：前向一次，各注入层做内部梯度步更新 M（不写入就旁路）"""
    for dl in layers_dyn:
        dl.mem.reset(DEV)
        dl.mode = "write"
    enc = tok(fact_text, return_tensors="pt").input_ids.to(DEV)
    model(enc)
    for dl in layers_dyn:
        dl.mode = "read"


@torch.no_grad()
def query(model, tok, k, max_new=10):
    enc = tok(QUERY(k), return_tensors="pt").input_ids.to(DEV)
    out = model.generate(enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.shape[1]:], skip_special_tokens=True).strip()


def evaluate(model, dyns, tok, facts, write=True):
    ex = pt = 0
    for k, v in facts.items():
        if write:
            write_pass(model, dyns, tok, WRITE(k, v))
        txt = query(model, tok, k)
        if v in txt: ex += 1
        elif sum(1 for a, b in zip(v, txt) if a == b) >= max(2, len(v) // 2): pt += 1
    return ex / len(facts), pt / len(facts)


def main():
    t0 = time.time()
    print("=" * 92); print("N2e · 运行时动态写入（Titans 式）：写入即生效？"); print("=" * 92)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR); tok.pad_token = tok.eos_token
    torch.manual_seed(3); random.seed(3)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DT).to(DEV).eval()
    mems, dyns, ps = build(model)
    print(f"[build] 注入 {len(mems)} 层，可训 {sum(p.numel() for p in ps)/1e6:.3f}M（主干冻结）")

    TR = make_facts(96, 31); EV = make_facts(32, 32)
    TRS = dict(list(TR.items())[:32]); EVS = dict(list(EV.items())[:16])

    # 基线：门控=0（关闭记忆）
    for m in mems:
        with torch.no_grad(): m.gate.zero_()
    e0, p0 = evaluate(model, dyns, tok, EVS, write=True)
    print(f"[0] 基线（gate=0，写入被旁路）：留出键 exact {e0*100:.1f}%（应≈0）")

    # 外循环训练：让"写入→读出"这条链学会（梯度穿过内部梯度步）
    print(f"\n[1] 外循环训练 {STEPS} 步（BS={BS}, lr={LR}）：损失=答案 token 的 CE")
    items = list(TR.items()); model.train()
    for m in mems:
        with torch.no_grad(): m.gate.fill_(0.5)
    opt = torch.optim.AdamW(ps, lr=LR)
    losses = []
    for s in range(STEPS):
        batch = random.sample(items, BS)
        write_pass(model, dyns, tok, "\n".join(WRITE(k, v) for k, v in batch))
        qs = [QUERY(k) + v for k, v in batch]
        enc = tok(qs, return_tensors="pt", padding=True).to(DEV)
        out = model(**enc, labels=enc["input_ids"])
        opt.zero_grad(); out.loss.backward(); opt.step(); losses.append(out.loss.item())
        if s % 60 == 0 or s == STEPS - 1:
            print(f"    step {s:3d}  loss {out.loss.item():.4f}", flush=True)
    model.eval()

    # 评测（写入用留出键！训练时从未见过）
    e_tr, p_tr = evaluate(model, dyns, tok, TRS, write=True)
    e_ev, p_ev = evaluate(model, dyns, tok, EVS, write=True)
    e_now, p_now = evaluate(model, dyns, tok, EVS, write=False)   # 不写入直接问（应≈0）
    print(f"\n[2] 训练键（写入即问）：exact {e_tr*100:.1f}%  数字级 {p_tr*100:.1f}%")
    print(f"[3] 留出键（**推理时才写入**）：exact {e_ev*100:.1f}%  数字级 {p_ev*100:.1f}%  ← 核心指标")
    print(f"[4] 负对照（不写入直接问）：exact {e_now*100:.1f}%  数字级 {p_now*100:.1f}%  ← 应≈0")

    res = dict(trainable=sum(p.numel() for p in ps), loss0=losses[0], lossN=losses[-1],
               baseline_gate0=e0, train_keys=[e_tr, p_tr], heldout_written=[e_ev, p_ev],
               no_write=[e_now, p_now], gates=[round(m.gate.item(), 3) for m in mems],
               elapsed_s=time.time() - t0)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "n2e_result.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[time] {time.time()-t0:.0f}s")
    print("=" * 92)


if __name__ == "__main__":
    main()
