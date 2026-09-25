import sys, torch
sys.argv=['x']
src=open('code/mtp_n8s1.py').read().split('if __name__')[0]
exec(src)
def p(s): print(s, flush=True)
t=StreamQwen2(); p("A: 基础张量就绪")
pos=torch.arange(8, device=t.dev).unsqueeze(0)
dummy=t.embed[:1,:1].new_zeros(1,1,1)
cos,sin=t.rope(dummy,pos); p(f"B: rope {cos.shape} {sin.shape}")
ids=torch.tensor([[100,200,300,400,500,600,700,800]])
h=t.embed[ids.to(t.dev)]; p(f"C: embed {tuple(h.shape)}")
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from safetensors import safe_open
import os
handles={s: safe_open(os.path.join(t.md,s), framework="pt") for s in set(t.wmap.values())}
p("D: 句柄就绪")
layer=Qwen2DecoderLayer(t.cfg, layer_idx=0).to(t.dev, t.dt); p(f"E: 层创建 参数 {sum(x.numel() for x in layer.parameters())/1e6:.1f}M")
sd={k[len("model.layers.0."):]: handles[s].get_tensor(k).to(t.dev,t.dt) for s,k in t.layer_keys[0]}
p(f"F: sd {len(sd)} 张量")
layer.load_state_dict(sd, strict=True); p("G: 权重装载 OK")
with torch.no_grad():
    out=layer(h, attention_mask=None, position_ids=pos, past_key_value=None, use_cache=False, position_embeddings=(cos,sin))
p(f"H: 前向 OK {tuple(out[0].shape) if isinstance(out,tuple) else tuple(out.shape)}")
