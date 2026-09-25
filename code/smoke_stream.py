import sys, torch, time
sys.argv=['x']
exec(open('code/mtp_n8s1.py').read().split('if __name__')[0])
t=StreamQwen2()
ids=torch.tensor([[100,200,300,400,500,600,700,800]])
h=t.forward_hidden(ids)
print("[冒烟] h", tuple(h.shape), h.dtype, "显存", torch.cuda.max_memory_allocated()/2**30)
hn=t.normed(h)
print("[冒烟] normed", tuple(hn.shape), "| 前3个值", hn[0,0,:3].tolist())
