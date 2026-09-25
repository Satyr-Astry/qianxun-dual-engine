import sys, torch, json
sys.argv=['x']
exec(open('code/mtp_n8s1.py').read().split('if __name__')[0])
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(MODEL_DIR)
trunk=StreamQwen2()
lm=load_small()["lm_head.weight"]
prompt="用一句话解释什么是 KV 缓存："
ids=tok(prompt).input_ids
print("[保真] prompt token 数", len(ids), flush=True)
h=trunk.forward_hidden(torch.tensor(ids).unsqueeze(0))
lg=(trunk.normed(h)[0,-1].float() @ lm.float().t())
top=lg.topk(8)
print("[保真] 主干下一 token top-8:", flush=True)
for v,i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"    {i:>7d}  {v:8.3f}  {tok.decode([i])!r}", flush=True)
