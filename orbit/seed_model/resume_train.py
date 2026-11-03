"""Continue ORBIT Seed training in short, resumable CPU batches."""
import argparse, json, math, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm
from safetensors.torch import load_file, save_file
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=100); a=p.parse_args()
random.seed(73421); torch.manual_seed(73421); torch.set_num_threads(3); torch.set_num_interop_threads(1)
c=json.loads((ROOT/'config.json').read_text()); sp=spm.SentencePieceProcessor(model_file=str(ROOT/'tokenizer.model'))
class Attn(nn.Module):
 def __init__(self,d,h,drop): super().__init__(); self.h=h; self.hd=d//h; self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d); self.rd=nn.Dropout(drop)
 def forward(self,x):
  b,t,d=x.shape; q,k,v=self.qkv(x).split(d,dim=2); q=q.view(b,t,self.h,self.hd).transpose(1,2); k=k.view(b,t,self.h,self.hd).transpose(1,2); v=v.view(b,t,self.h,self.hd).transpose(1,2)
  z=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.rd(self.proj(z.transpose(1,2).contiguous().view(b,t,d)))
class Block(nn.Module):
 def __init__(self,d,h,drop): super().__init__(); self.ln1=nn.LayerNorm(d); self.attn=Attn(d,h,drop); self.ln2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d),nn.Dropout(drop))
 def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.mlp(self.ln2(x))
class GPT(nn.Module):
 def __init__(self):
  super().__init__(); d=c['dim']; self.block_size=c['block_size']; self.token_emb=nn.Embedding(c['vocab_size'],d); self.pos_emb=nn.Embedding(c['block_size'],d); self.drop=nn.Dropout(c['dropout']); self.blocks=nn.Sequential(*[Block(d,c['heads'],c['dropout']) for _ in range(c['layers'])]); self.ln_f=nn.LayerNorm(d); self.lm_head=nn.Linear(d,c['vocab_size'],bias=False); self.lm_head.weight=self.token_emb.weight
 def forward(self,x,y=None):
  if x.size(1)>self.block_size: x=x[:,-self.block_size:]
  pos=torch.arange(x.size(1)); h=self.drop(self.token_emb(x)+self.pos_emb(pos)[None,:,:]); logits=self.lm_head(self.ln_f(self.blocks(h))); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y[:,-x.size(1):].reshape(-1)) if y is not None else None; return logits,loss

def pack(path):
 docs=[x.strip() for x in path.read_text(encoding='utf-8').split('\n\n') if x.strip()]; ids=[]
 for d in docs: ids.extend(sp.encode(d,out_type=int)); ids.append(sp.eos_id())
 return torch.tensor(ids,dtype=torch.long)
tr=pack(ROOT/'train_corpus.txt'); va=pack(ROOT/'validation_corpus.txt')
model=GPT(); model.load_state_dict(load_file(str(ROOT/'model.safetensors'))); opt=torch.optim.AdamW(model.parameters(),lr=1.5e-4,betas=(0.9,0.95),weight_decay=0.1)
statepath=ROOT/'optimizer_state.pt'; current=0
if statepath.exists():
 st=torch.load(statepath,map_location='cpu',weights_only=False); opt.load_state_dict(st['optimizer']); current=st['step']
else:
 log_path=ROOT/'training_log.json'
 if log_path.exists():
  existing=json.loads(log_path.read_text()); current=max((row.get('step',0) for row in existing),default=0)
 print(f'Starting continuation from step {current} with a fresh optimizer.',flush=True)
B=8; S=112

def batch(data):
 starts=torch.randint(0,len(data)-S-1,(B,)); x=torch.stack([data[s:s+S] for s in starts]); y=torch.stack([data[s+1:s+S+1] for s in starts]); return x,y
@torch.no_grad()
def evaluate():
 model.eval(); vals=[]
 for _ in range(8): x,y=batch(va); _,l=model(x,y); vals.append(l.item())
 model.train(); return sum(vals)/len(vals)
model.train(); start=time.time(); avg=0
for i in range(a.steps):
 step=current+i+1; lr=1.5e-4*(0.35+0.65*0.5*(1+math.cos(math.pi*min(step,1600)/1600)))
 for pg in opt.param_groups: pg['lr']=lr
 x,y=batch(tr); _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); avg+=loss.item()
val=evaluate(); train_avg=avg/max(a.steps,1)
save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(ROOT/'model.safetensors'))
torch.save({'step':current+a.steps,'optimizer':opt.state_dict()},statepath)
logp=ROOT/'training_log.json'; log=json.loads(logp.read_text()) if logp.exists() else []
log.append({'step':current+a.steps,'chunk_train_loss':round(train_avg,4),'val_loss':round(val,4),'chunk_seconds':round(time.time()-start,1)})
logp.write_text(json.dumps(log,indent=2)); print({'step':current+a.steps,'chunk_train_loss':round(train_avg,4),'val_loss':round(val,4),'seconds':round(time.time()-start,1)},flush=True)
