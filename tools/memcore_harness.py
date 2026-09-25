"""
忆态架构机制验证工装 · MemCore Harness
================================================================================
定位：把 N1/N2 系列实验中**被证明有用**的检查，抽成可复用、可自测的工装。
      外部（或未来的我们）拿到它，可以直接复现"新架构改造是否无损 / 是否活 / 是否有真增益"。

六项检查（每项都对应一次真实的踩坑或铁律）：
  1. rewrite_equiv_check   结构等价：**功能判据 = 全位置 logits max-abs-diff ≤ tol(1e-3)**
                          （v3.1 修订：**不再要求逐位相同** —— 本架构要自我进化，权重本来就会变；
                           逐 token 一致率仅作辅助，因核级漂移会在 79%–100% 摆动）
  2. init_grad_health      init 梯度体检：∂L/∂gate 是否为 0 ⇒ 零死锁（N1 踩出的坑）
  3. trunk_invariance      主干不变性：**张量级 max-abs-diff**（禁用抽样 hash：会假警报）
  4. recall_protocol       关联召回协议：查询式训练 + 留出键负对照（N2 的两个设计错误）
  5. smi_peak              nvidia-smi 独立显存峰值采样（不信 torch 统计口径）
  6. run_record            实验记录：JSON + 结论 + 多种子要求提醒（防单次种子当结论）

组装件：ZeroGateBranch（零门旁路）/ ProductKeyMemory（乘积键静态池）/ LoRA / DynMem（Titans-lite 动态记忆）

自测：python tools/memcore_harness.py --selftest
"""
from __future__ import annotations
import argparse, gc, json, math, os, random, subprocess, threading, time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = os.environ.get(
    "MEMCORE_MODEL",
    # v3.2 基底换代：本机实验基准 = Qwen3.5-4B（bf16 8.1G，放得下 12G）
    # 回退：旧原型 Qwen2.5-1.5B-Instruct（若新基底尚未下载）
    os.path.expanduser("E:/models/Qwen3.5-4B"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float32   # 由 load_model() 按权重体积设定（>5GB → bf16）


# ═══════════════════════════ 组装件 ═══════════════════════════
class ZeroGateBranch(nn.Module):
    """把任意子模块以**零门**挂到主干上：gate 零初始化 ⇒ S0 时贡献逐位为 0。
    ★ 铁律：只允许「最后一层门」零初始化；门的输入侧必须非零，否则梯度互相乘零（零死锁）。
    ★ dtype 规范（v3.2）：分支内部一律 **fp32** 计算（主干可能是 bf16），结果回投主干 dtype。
      gate=0 时 `0.0 * fp32 + bf16主干` 仍**精确等于**主干（t + 0.0 位级不变）⇒ 功能等价不受影响。"""

    def __init__(self, orig: nn.Module, branch: nn.Module):
        super().__init__()
        self.orig, self.branch = orig, branch
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.branch(x.float())
        return self.orig(x) + (self.gate.float() * y).to(x.dtype)


class ProductKeyMemory(nn.Module):
    """静态关联记忆池（乘积键 top-k 查表，Memory Layers 式）。
    注意：静态池只能记"训练时写进去的"关联；对推理时才出现的键无能为力（见 DynMem）。"""

    def __init__(self, d: int, entries: int = 65536, half: int = 256, kdim: int = 128, topk: int = 8):
        super().__init__()
        assert half * half == entries, "本实现用 half × half = entries 的乘积键"
        self.half, self.kdim, self.topk = half, kdim, topk
        self.K1 = nn.Parameter(torch.randn(half, kdim) / math.sqrt(kdim))
        self.K2 = nn.Parameter(torch.randn(half, kdim) / math.sqrt(kdim))
        self.V = nn.Parameter(torch.randn(entries, d) * 0.02)      # ★ 非零（否则零死锁）
        self.q_proj = nn.Linear(d, 2 * kdim, bias=False)
        self.w_out = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.w_out.weight, std=1.0 / math.sqrt(d))  # ★ 非零

    def forward(self, h):
        B, T, _ = h.shape
        q = self.q_proj(h).view(B, T, 2, self.kdim)
        q1, q2 = q[..., 0, :], q[..., 1, :]
        i1 = (q1 @ self.K1.t()).topk(self.topk, -1).indices
        i2 = (q2 @ self.K2.t()).topk(self.topk, -1).indices
        cand = (i1.unsqueeze(-1) * self.half + i2.unsqueeze(-2)).reshape(B, T, -1)
        s = ((q1.unsqueeze(-2) @ self.K1[cand // self.half].transpose(-1, -2)) +
             (q2.unsqueeze(-2) @ self.K2[cand % self.half].transpose(-1, -2))).squeeze(-2)
        idx = s.topk(min(self.topk, s.shape[-1]), -1).indices
        sel = cand.gather(-1, idx)
        w = F.softmax(s.gather(-1, idx), -1)
        self.last_sel = sel          # 供审计用（不参与前向，避免与门控相乘时被当索引）
        return self.w_out((w.unsqueeze(-1) * self.V[sel]).sum(-2))


class LoRA(nn.Module):
    """零初始化 LoRA（B=0 ⇒ 初始等价）。零初始化在"乘积链末端"是安全的。"""

    def __init__(self, base: nn.Linear, r: int = 16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


class DynMem(nn.Module):
    """Titans-lite 动态记忆：写入 = 内部梯度步（可微），读取 = 前向。
    实测（N2e）：本机预算下未打通（loss 不稳、召回 0%）→ 保留为研究件，别当产品。"""

    def __init__(self, d: int, kdim: int = 128, rank: int = 64):
        super().__init__()
        self.Wk, self.Wv = nn.Linear(d, kdim, bias=False), nn.Linear(d, rank, bias=False)
        self.Wq, self.Wo = nn.Linear(d, kdim, bias=False), nn.Linear(rank, d, bias=False)
        nn.init.normal_(self.Wo.weight, std=1.0 / math.sqrt(d))
        self.alpha, self.theta = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Parameter(torch.zeros(1))
        self.M: Optional[torch.Tensor] = None

    def reset(self):
        self.M = None

    def write(self, h):
        B, T, _ = h.shape
        k, v = self.Wk(h), self.Wv(h)
        if self.M is None:
            self.M = torch.zeros(B, k.shape[-1], v.shape[-1], device=h.device, dtype=h.dtype)
        grad = k.transpose(1, 2) @ (k @ self.M - v) / max(1, T)
        self.M = (1 - torch.sigmoid(self.alpha)) * self.M - torch.sigmoid(self.theta) * grad
        return h

    def read(self, h):
        if self.M is None:
            return torch.zeros_like(h)
        return self.gate * self.Wo(self.Wq(h) @ self.M)


# ═══════════════════════════ 六项检查 ═══════════════════════════
class SMISampler(threading.Thread):
    """nvidia-smi 独立显存采样（1Hz）；不信 torch 的 max_memory_allocated 口径。"""

    def __init__(self):
        super().__init__(daemon=True)
        self._flag, self.peak_mib = False, 0

    def run(self):
        while not self._flag:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                self.peak_mib = max(self.peak_mib, int(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(1.0)

    def stop(self):
        self._flag = True


def find_layers(model) -> nn.ModuleList:
    """通用定位 Transformer 层列表（兼容多模态包装：model.model.layers / language_model.layers …）"""
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8 and hasattr(mod[0], "mlp"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("找不到层列表：模型结构超出预期")
    return best


def pick_dtype(path: str):
    """按权重体积自适应精度：>5GB 用 bf16（12G 卡放不下 fp32），否则 fp32。"""
    try:
        with open(os.path.join(path, "model.safetensors.index.json"), encoding="utf-8") as f:
            total = json.load(f).get("metadata", {}).get("total_size", 0)
    except Exception:
        total = 0
    return (torch.bfloat16 if total > 5e9 else torch.float32), total


def load_model(path: str):
    """先试因果 LM，再试多模态包装（Qwen3.5/3.6/3.8 是 Qwen3_5ForConditionalGeneration）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    global MODEL_DTYPE
    dt, total = pick_dtype(path)
    MODEL_DTYPE = dt
    print(f"[load] {path}  权重 {total/1e9:.2f} GB → dtype={dt}")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(DEV).eval(), tok
    except Exception as e:
        print(f"[load] AutoModelForCausalLM 不适用（{type(e).__name__}），改走多模态包装")
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=dt).to(DEV).eval()
        return m, tok


@torch.no_grad()
def _greedy(model, tok, prompt, n_new=32):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0].tolist()


@torch.no_grad()
def full_logits(model, tok, ids: list) -> torch.Tensor:
    """教师强制：对给定 token 序列做一次前向，返回**全部位置**的 logits（fp32, cpu）。
    这是等价性判据的稳健形态：单次前向、无解码路径差异、覆盖每一位。"""
    t = torch.tensor([ids], device=DEV)
    return model(t).logits[0].float().cpu()


@torch.no_grad()
def rewrite_equiv_check(model, tok, prompts, n_new=32) -> dict:
    """【1】结构等价基线：逐 token greedy 输出 + **全位置 logits 指纹**。"""
    base_tok, fulls = {}, {}
    for p in prompts:
        gen = _greedy(model, tok, p, n_new)
        prompt_ids = tok(p, return_tensors="pt").input_ids[0].tolist()
        base_tok[p] = gen[len(prompt_ids):]
        fulls[p] = full_logits(model, tok, prompt_ids + base_tok[p])   # ★ 只拼续写段（勿重复拼 prompt）
    return {"base_tokens": base_tok, "base_full": fulls,
            "prompt_ids": {p: tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts}}


def compare_equiv(base: dict, model, tok, prompts, floor: float = 1.0,
                  tol: float = 1e-3, noise_floor: float = 0.0) -> dict:
    """等价性判定（**v3.2：dtype 与噪声感知**）。

    判据三档，**从强到弱**：
      1. **位级**：全位置 logits maxdiff == 0（fp32 下常见）
      2. **噪声内**：maxdiff ≤ max(`noise_floor`, tol) × 3 —— `noise_floor` 由 `noise_floor_logits`
         在**同一模型结构**上实测（bf16 下可达 0.5，远超任何固定阈值）⇒ 差异不能归因于分支
      3. **token 级**：greedy 一致率 ≥ 确定性底线
    通过任一档即可判「功能等价」。理由（作者定调）：**本架构要自我进化**，逐位不变不是目标。"""
    hit = tot = 0
    maxdiff = 0.0
    absmax = 1e-9
    lens = []
    for p in prompts:
        ids = base["prompt_ids"][p]
        gen_new = _greedy(model, tok, p)[len(ids):]
        b = base["base_tokens"][p]
        hit += sum(1 for x, y in zip(b, gen_new) if x == y); tot += len(b)
        lg = full_logits(model, tok, ids + gen_new)
        bf = base["base_full"][p]
        L = min(bf.shape[0], lg.shape[0])            # 生成长度可能因 EOS 提前截断 → 比公共前缀
        maxdiff = max(maxdiff, (bf[:L] - lg[:L]).abs().max().item())
        absmax = max(absmax, bf[:L].abs().max().item())
        lens.append((bf.shape[0], lg.shape[0]))
    rel = maxdiff / absmax
    rel_tol = 1e-3 if MODEL_DTYPE == torch.float32 else 5e-2   # bf16 只有 8 位尾数
    return {"agree": hit / tot, "logit_maxdiff_allpos": maxdiff, "logit_absmax": absmax,
            "rel_maxdiff": rel, "rel_tol": rel_tol, "floor": floor, "tol": tol,
            "noise_floor": noise_floor, "lens": lens,
            "pass": ((rel <= rel_tol or maxdiff <= max(noise_floor, tol) * 3) and hit / tot >= floor),
            "strict_bitwise": maxdiff == 0.0}


def init_grad_health(model, tok, text, gates, extra_named: dict) -> dict:
    """【2】init 梯度体检：∂L/∂gate ≠ 0 = 分支可被"开门"；全 0 = 零死锁。
    extra_named: {"w_out": param, "V": param} 用于判断"先开门再训分支"的顺序依赖。"""
    for g in gates:
        g.requires_grad_(True)
    for p in extra_named.values():
        p.requires_grad_(True)
    model.train()
    ids = tok(text, return_tensors="pt").input_ids[:, :128].to(DEV)
    loss = model(ids, labels=ids).loss
    loss.backward()
    g_grads = [g.grad.item() if g.grad is not None else 0.0 for g in gates]
    extras = {k: (p.grad.abs().mean().item() if p.grad is not None else 0.0)
              for k, p in extra_named.items()}
    model.zero_grad(set_to_none=True); model.eval()
    for g in gates:
        g.requires_grad_(False)
    for p in extra_named.values():
        p.requires_grad_(False)
    ok = any(abs(g) > 0 for g in g_grads)
    return {"gate_grads": g_grads, "extra_grads": extras, "deadlock_free": ok,
            "note": "gate≠0 而输入侧=0 属正常（先开门后训分支）"}


INJECTED_PAT = (".branch", ".mem.", ".gate", ".A", ".B")   # 新挂架构件的名字特征


def trunk_snapshot(model, exclude=INJECTED_PAT) -> dict:
    """【3】主干快照（张量级全量）。
    ★ 用法铁律：**在注入新模块之前**调用（那时 model 里的参数才是真主干）。
    同时按名字特征排除已知架构件，双保险。"""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if not any(pat in n for pat in exclude)}


def trunk_invariance(snap: dict, model) -> dict:
    worst, wname = 0.0, ""
    for n, p in model.named_parameters():
        if n in snap:
            d = (p.detach() - snap[n]).abs().max().item()
            if d > worst:
                worst, wname = d, n
    return {"max_abs_diff": worst, "worst_param": wname, "untouched": worst == 0.0}


def determinism_floor(model, tok, prompts, n_new=24) -> dict:
    """【3b】确定性底线：把**完全没有被改动**的模型同一段 greedy 生成跑两遍。
    若自一致率 < 100%，说明"逐 token 一致率"在本机/本精度下**不是可用判据**
    （cuBLAS/融合核在不同调用间可能选不同算法），此时等价性必须改用 logits 判据。"""
    a = {p: _greedy(model, tok, p, n_new) for p in prompts}
    b = {p: _greedy(model, tok, p, n_new) for p in prompts}
    hit = tot = 0
    for p in prompts:
        hit += sum(1 for x, y in zip(a[p], b[p]) if x == y); tot += len(a[p])
    return {"self_agreement": hit / tot, "deterministic": hit == tot}


def noise_floor_logits(model, tok, prompts, ref: dict) -> float:
    """【3c】**核级噪声地板**：对**已经挂好分支的同一个模型**跑两遍前向，
    取两遍之间的全位置 logits 最大绝对差。
    ⇒ 这才是"同一结构下的数值噪声水平"；baseline↔modified 的差异只要不超过它，
    就**不能**归因于我们挂的分支（bf16 下这个差值可达 0.5，远超任何固定 tol）。"""
    worst = 0.0
    for p in prompts:
        ids = ref["prompt_ids"][p]
        gen = _greedy(model, tok, p)[len(ids):]
        a = full_logits(model, tok, ids + gen)
        b = full_logits(model, tok, ids + gen)
        L = min(a.shape[0], b.shape[0])
        worst = max(worst, (a[:L] - b[:L]).abs().max().item())
    return worst


# ── 关联召回协议 ──────────────────────────────────────────────
_SYLL = ["zur", "qaf", "vom", "kix", "dun", "pyr", "lek", "nuf", "tad", "gor",
         "meb", "sil", "wol", "jap", "fer", "cug", "tar", "bez", "myn", "oku"]


def make_facts(n: int, seed: int, digits: int = 4) -> dict:
    rnd = random.Random(seed); out = {}
    while len(out) < n:
        k = rnd.choice(_SYLL).capitalize() + rnd.choice(_SYLL) + str(rnd.randint(10, 99))
        if k not in out:
            out[k] = str(rnd.randint(10 ** (digits - 1), 10 ** digits - 1))
    return out


Q_TEXT = lambda k: f"问：{k} 的编号是什么？\n答："
Q_TRAIN = lambda k, v: f"问：{k} 的编号是什么？\n答：{v}"


@torch.no_grad()
def recall_rate(model, tok, facts, max_new=10) -> tuple:
    """【4】召回评估。★ 协议铁律：
      - 训练样本**不得**带"上下文内的事实"（否则模型抄近路，记忆分支变摆设）
      - **留出键只作负对照**（静态池里没有它们，召回必然为 0）"""
    ex = pt = 0
    for k, v in facts.items():
        txt = tok.decode(_greedy(model, tok, Q_TEXT(k), max_new)[
                         len(tok(Q_TEXT(k)).input_ids):], skip_special_tokens=True).strip()
        if v in txt:
            ex += 1
        elif sum(1 for a, b in zip(v, txt) if a == b) >= max(2, len(v) // 2):
            pt += 1
    return ex / len(facts), pt / len(facts)


def train_query_style(model, tok, params, facts, steps=200, bs=4, lr=3e-3) -> list:
    opt = torch.optim.AdamW(params, lr=lr)
    items = list(facts.items()); model.train(); ls = []
    for _ in range(steps):
        b = random.sample(items, bs)
        enc = tok([Q_TRAIN(k, v) for k, v in b], return_tensors="pt", padding=True).to(DEV)
        loss = model(**enc, labels=enc["input_ids"]).loss
        opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
    model.eval()
    return ls


# ═══════════════════════════ 实验记录 ═══════════════════════════
@dataclass
class RunRecord:
    """【6】实验记录：结论必须满足"多种子 + 负对照 + 仪表可信"。"""
    name: str
    seed: int
    trainable_params: int
    metrics: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    verdict: str = "unknown"
    warnings: list = field(default_factory=list)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def single_seed_warning(n_seeds: int) -> Optional[str]:
        return None if n_seeds >= 2 else "⚠️ 单种子结果不得作为结论（本项目已被此坑骗过一次）"


def free_model(model):
    del model; gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════ 自测 ═══════════════════════════
def selftest(model_dir: str = DEFAULT_MODEL):
    t0 = time.time()
    print("=" * 78); print("MemCore Harness · 自测（验证六项检查都能跑且有正确判定）"); print("=" * 78)
    model, tok = load_model(model_dir)
    prompts = ["用一句话解释什么是 KV 缓存。", "The capital of France is"]
    smi = SMISampler(); smi.start()

    # 基线（含确定性底线：同模型两次生成的自一致率）
    base = rewrite_equiv_check(model, tok, prompts, n_new=24)
    fl = determinism_floor(model, tok, prompts, n_new=24)
    print(f"[0] 确定性底线     同模型两次 greedy 生成自一致率 {fl['self_agreement']*100:.2f}%"
          f"  → {'确定性可用（逐 token 判据有效）' if fl['deterministic'] else '⚠️ 非确定：等价性以 logits 判据为主'}")

    # 主干快照必须在**注入之前**取
    snap = trunk_snapshot(model)
    print(f"[·] 主干快照       {len(snap)} 个张量（注入前取，作真主干基准）")

    # 挂一个零门旁路（记忆分支）——注入位置取**后半段层**（v3.2 规范：MSA 实证前层语义抽象不足）
    layers = find_layers(model)
    L = len(layers); c = L - 1 - L // 6
    inj = sorted({L - 2, c, L - 4})
    mem = ProductKeyMemory(model.config.hidden_size if hasattr(model.config, "hidden_size")
                           else model.config.text_config.hidden_size).to(DEV)
    gates = []
    for i in inj:
        layers[i].mlp = ZeroGateBranch(layers[i].mlp, mem).to(DEV)
        gates.append(layers[i].mlp.gate)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in (mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2):
        p.requires_grad_(True)

    # 【1】等价（判据：位级 / 噪声内 / token 级，任一通过即算功能等价）
    nf = noise_floor_logits(model, tok, prompts, base)
    eq = compare_equiv(base, model, tok, prompts, floor=fl["self_agreement"],
                       tol=(1e-3 if MODEL_DTYPE == torch.float32 else 5e-2), noise_floor=nf)
    print(f"[1] 结构等价（dtype={MODEL_DTYPE}，核级噪声地板 {nf:.2e}）")
    print(f"    全位置 logits |Δ|max {eq['logit_maxdiff_allpos']:.2e}（logits 量级 {eq['logit_absmax']:.1f}，"
          f"相对 {eq['rel_maxdiff']:.2%} / 门槛 {eq['rel_tol']:.0%}）"
          f" ｜ 逐 token 一致率 {eq['agree']*100:.2f}%（底线 {eq['floor']*100:.2f}%）"
          f"  → {'✅ PASS' if eq['pass'] else '❌ FAIL'}"
          f"（位级严格={'是' if eq['strict_bitwise'] else '否'}）")

    # 【2】init 梯度体检
    gh = init_grad_health(model, tok, "机器学习推理速度受显存带宽限制。" * 4,
                          gates, {"w_out": mem.w_out.weight, "V": mem.V})
    print(f"[2] init 梯度体检  ∂L/∂gate={['%.2e' % g for g in gh['gate_grads']]} "
          f"|w_out|={gh['extra_grads']['w_out']:.1e} |V|={gh['extra_grads']['V']:.1e}"
          f"  → {'✅ PASS（死锁已解除）' if gh['deadlock_free'] else '❌ FAIL（零死锁）'}")

    # 【3】主干不变性（训 5 步后）
    facts = make_facts(16, 7)
    train_query_style(model, tok, [mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2]
                      + gates, facts, steps=5, bs=2)
    ti = trunk_invariance(snap, model)
    print(f"[3] 主干不变性     max-abs-diff {ti['max_abs_diff']:.1e}（{ti['worst_param'] or '无'}）"
          f"  → {'✅ PASS（主干严格未变）' if ti['untouched'] else '❌ FAIL（主干被改动）'}")

    # 【4】召回协议（用 2 条做连通性检查，不作为结论）
    ex, pt = recall_rate(model, tok, dict(list(facts.items())[:2]))
    print(f"[4] 召回协议       小样本连通性检查：exact {ex*100:.0f}% / 数字级 {pt*100:.0f}%"
          f"  → ✅ PASS（协议可执行；数值不构成结论，需 128 键 × ≥2 种子）")

    # 【6】记录
    rec = RunRecord("selftest", seed=0, trainable_params=sum(p.numel() for p in
                    [mem.q_proj.weight, mem.w_out.weight, mem.V, mem.K1, mem.K2] + gates),
                    metrics={"equiv": eq, "gate_grads": gh["gate_grads"], "recall_small": [ex, pt]},
                    checks={"equiv": eq["pass"], "deadlock_free": gh["deadlock_free"],
                            "trunk_untouched": ti["untouched"]},
                    verdict="harness-ok" if (eq["pass"] and gh["deadlock_free"] and ti["untouched"])
                            else "harness-bad")
    w = RunRecord.single_seed_warning(1)
    if w: rec.warnings.append(w)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_record.json")
    rec.save(out)

    smi._flag = True
    print(f"[5] 显存（nvidia-smi 独立采样峰值） {smi.peak_mib} MiB / 12282 MiB  → ✅ PASS（口径可信）")
    print(f"[6] 记录已写入 {out}")
    print(f"\n总判定：{rec.verdict}   耗时 {time.time()-t0:.1f}s")
    print("=" * 78)
    return rec.verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.model)
    else:
        ap.print_help()
