"""Train ORBIT Seed from random initialization. CPU-friendly PyTorch GPT."""
from __future__ import annotations
import json, math, random, time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
SEED = 73421
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(3)
torch.set_num_interop_threads(1)

# Keep examples intact when making the validation split.
raw = (ROOT/'corpus.txt').read_text(encoding='utf-8')
docs = [x.strip() for x in raw.split('\n\n') if x.strip()]
random.shuffle(docs)
cut = max(1, int(len(docs)*0.94))
train_docs, val_docs = docs[:cut], docs[cut:]
(ROOT/'train_corpus.txt').write_text('\n\n'.join(train_docs), encoding='utf-8')
(ROOT/'validation_corpus.txt').write_text('\n\n'.join(val_docs), encoding='utf-8')

spm.SentencePieceTrainer.train(
    input=str(ROOT/'train_corpus.txt'), model_prefix=str(ROOT/'tokenizer'),
    vocab_size=2000, model_type='bpe', character_coverage=1.0,
    user_defined_symbols=['<|user|>','<|assistant|>','<|end|>'],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    hard_vocab_limit=False, max_sentence_length=12000,
    num_threads=3, input_sentence_size=0, shuffle_input_sentence=False,
)
sp = spm.SentencePieceProcessor(model_file=str(ROOT/'tokenizer.model'))

# Pack documents with explicit end-of-conversation markers.
def encode_docs(items):
    ids=[]
    for doc in items:
        ids.extend(sp.encode(doc, out_type=int))
        ids.append(sp.eos_id())
    return torch.tensor(ids, dtype=torch.long)

train_ids = encode_docs(train_docs)
val_ids = encode_docs(val_docs)
print(f'Tokenizer: {sp.get_piece_size()} pieces; train={len(train_ids):,} tokens; validation={len(val_ids):,} tokens', flush=True)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__(); self.heads=heads; self.head_dim=dim//heads
        self.qkv=nn.Linear(dim, 3*dim); self.proj=nn.Linear(dim, dim)
        self.attn_dropout=dropout; self.resid_dropout=nn.Dropout(dropout)
    def forward(self, x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).split(c, dim=2)
        q=q.view(b,t,self.heads,self.head_dim).transpose(1,2)
        k=k.view(b,t,self.heads,self.head_dim).transpose(1,2)
        v=v.view(b,t,self.heads,self.head_dim).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v, attn_mask=None,
             dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True)
        y=y.transpose(1,2).contiguous().view(b,t,c)
        return self.resid_dropout(self.proj(y))

class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__(); self.ln1=nn.LayerNorm(dim); self.attn=CausalSelfAttention(dim,heads,dropout)
        self.ln2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); x=x+self.mlp(self.ln2(x)); return x

class OrbitGPT(nn.Module):
    def __init__(self, vocab, block_size=192, dim=192, layers=4, heads=4, dropout=0.08):
        super().__init__(); self.block_size=block_size
        self.token_emb=nn.Embedding(vocab,dim); self.pos_emb=nn.Embedding(block_size,dim)
        self.drop=nn.Dropout(dropout); self.blocks=nn.Sequential(*[Block(dim,heads,dropout) for _ in range(layers)])
        self.ln_f=nn.LayerNorm(dim); self.lm_head=nn.Linear(dim,vocab,bias=False)
        self.lm_head.weight=self.token_emb.weight
        self.apply(self._init)
        for name,p in self.named_parameters():
            if name.endswith('proj.weight'): nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*layers))
    def _init(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,mean=0.0,std=0.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,mean=0.0,std=0.02)
    def forward(self,idx, targets=None):
        b,t=idx.shape
        if t>self.block_size: idx=idx[:,-self.block_size:]; t=self.block_size
        pos=torch.arange(t,device=idx.device)
        x=self.drop(self.token_emb(idx)+self.pos_emb(pos)[None,:,:])
        x=self.ln_f(self.blocks(x)); logits=self.lm_head(x)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets[:,-t:].reshape(-1)) if targets is not None else None
        return logits,loss

CONFIG={'model_name':'ORBIT Seed','version':'0.1','architecture':'decoder-only causal transformer','vocab_size':sp.get_piece_size(),'block_size':160,'dim':128,'layers':3,'heads':4,'dropout':0.06,'seed':SEED,'parameters':None}
model=OrbitGPT(CONFIG['vocab_size'],CONFIG['block_size'],CONFIG['dim'],CONFIG['layers'],CONFIG['heads'],CONFIG['dropout'])
CONFIG['parameters']=sum(p.numel() for p in model.parameters())
print(f"Model: {CONFIG['parameters']:,} parameters", flush=True)

BATCH=8; SEQ=112; STEPS=1600; EVAL_EVERY=200; LR=3e-4; WARMUP=100

def get_batch(data):
    max_start=len(data)-SEQ-1
    starts=torch.randint(0,max_start,(BATCH,))
    x=torch.stack([data[s:s+SEQ] for s in starts])
    y=torch.stack([data[s+1:s+SEQ+1] for s in starts])
    return x,y

@torch.no_grad()
def estimate_loss():
    model.eval(); out={}
    for name,data in [('train',train_ids),('val',val_ids)]:
        vals=[]
        for _ in range(8):
            x,y=get_batch(data); _,loss=model(x,y); vals.append(loss.item())
        out[name]=sum(vals)/len(vals)
    model.train(); return out

opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
best=float('inf'); start=time.time(); log=[]
for step in range(1,STEPS+1):
    if step<=WARMUP: lr=LR*step/WARMUP
    else: lr=LR*0.15+0.85*LR*0.5*(1+math.cos(math.pi*(step-WARMUP)/(STEPS-WARMUP)))
    for pg in opt.param_groups: pg['lr']=lr
    x,y=get_batch(train_ids); _,loss=model(x,y)
    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step % EVAL_EVERY == 0 or step==1:
        metrics=estimate_loss(); elapsed=time.time()-start
        row={'step':step,'train_loss':round(metrics['train'],4),'val_loss':round(metrics['val'],4),'seconds':round(elapsed,1)}; log.append(row)
        print(row, flush=True)
        if metrics['val'] < best:
            best=metrics['val']
            save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(ROOT/'model.safetensors'))
            (ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
            (ROOT/'training_log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    if step % 400 == 0:
        print(f'checkpoint at step {step}; elapsed {time.time()-start:.1f}s',flush=True)

print(f'Training complete in {time.time()-start:.1f}s; best validation loss {best:.4f}',flush=True)
