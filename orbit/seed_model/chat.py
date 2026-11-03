"""Run a trained ORBIT Seed model locally on CPU."""
from pathlib import Path
import argparse, json, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(3)

class CausalSelfAttention(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__(); self.heads=heads; self.head_dim=dim//heads; self.qkv=nn.Linear(dim,3*dim); self.proj=nn.Linear(dim,dim); self.resid_dropout=nn.Dropout(dropout)
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).split(c,dim=2)
        q=q.view(b,t,self.heads,self.head_dim).transpose(1,2); k=k.view(b,t,self.heads,self.head_dim).transpose(1,2); v=v.view(b,t,self.heads,self.head_dim).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.resid_dropout(self.proj(y.transpose(1,2).contiguous().view(b,t,c)))
class Block(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__(); self.ln1=nn.LayerNorm(dim); self.attn=CausalSelfAttention(dim,heads,dropout); self.ln2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim),nn.Dropout(dropout))
    def forward(self,x): return x+self.mlp(self.ln2(x+self.attn(self.ln1(x))))
class OrbitGPT(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c['dim']; self.block_size=c['block_size']; self.token_emb=nn.Embedding(c['vocab_size'],d); self.pos_emb=nn.Embedding(self.block_size,d); self.drop=nn.Dropout(c['dropout']); self.blocks=nn.Sequential(*[Block(d,c['heads'],c['dropout']) for _ in range(c['layers'])]); self.ln_f=nn.LayerNorm(d); self.lm_head=nn.Linear(d,c['vocab_size'],bias=False); self.lm_head.weight=self.token_emb.weight
    def forward(self,idx):
        if idx.size(1)>self.block_size: idx=idx[:,-self.block_size:]
        pos=torch.arange(idx.size(1),device=idx.device); x=self.drop(self.token_emb(idx)+self.pos_emb(pos)[None,:,:]); return self.lm_head(self.ln_f(self.blocks(x)))

def top_p_sample(logits,temperature=0.85,top_p=0.9,top_k=40):
    logits=logits/ max(temperature,1e-5)
    if top_k>0:
        kth=torch.topk(logits,min(top_k,logits.numel())).values[-1]
        logits[logits<kth]=-float('inf')
    sorted_logits, sorted_idx=torch.sort(logits,descending=True)
    probs=torch.softmax(sorted_logits,dim=-1); cumulative=probs.cumsum(dim=-1)
    remove=cumulative-probs>top_p; sorted_logits[remove]=-float('inf')
    probs=torch.softmax(sorted_logits,dim=-1)
    ix=torch.multinomial(probs,1)
    return sorted_idx[ix]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--temperature',type=float,default=0.78); ap.add_argument('--top-p',type=float,default=0.9); ap.add_argument('--top-k',type=int,default=40); ap.add_argument('--max-new-tokens',type=int,default=100); ap.add_argument('--one-shot',type=str,default=None); args=ap.parse_args()
    c=json.loads((ROOT/'config.json').read_text()); sp=spm.SentencePieceProcessor(model_file=str(ROOT/'tokenizer.model'))
    model=OrbitGPT(c); model.load_state_dict(load_file(str(ROOT/'model.safetensors'))); model.eval()
    history=[]
    def answer(user):
        history.append(('user',user))
        prompt='\n'.join(f'<|{role}|> {msg}' for role,msg in history)+'\n<|assistant|>'
        ids=sp.encode(prompt,out_type=int)
        if len(ids)>c['block_size']-args.max_new_tokens: ids=ids[-(c['block_size']-args.max_new_tokens):]
        x=torch.tensor([ids],dtype=torch.long)
        generated=[]
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                logits=model(x)[0,-1,:]
                nxt=top_p_sample(logits,args.temperature,args.top_p,args.top_k).item()
                if nxt in {sp.eos_id(), sp.piece_to_id('<|end|>'), sp.piece_to_id('<|user|>')}: break
                generated.append(nxt); x=torch.cat([x,torch.tensor([[nxt]])],dim=1)
        reply=sp.decode(generated).strip()
        reply=re.split(r'<\|(?:user|assistant)\|>',reply)[0].strip()
        history.append(('assistant',reply))
        return reply
    if args.one_shot is not None:
        print(answer(args.one_shot)); return
    print('ORBIT Seed 0.1 — local CPU chat. Type /exit to quit, /reset to clear context.')
    while True:
        try: user=input('\nYou: ').strip()
        except (EOFError,KeyboardInterrupt): print(); break
        if user.lower() in {'/exit','/quit'}: break
        if user.lower()=='/reset': history.clear(); print('Context cleared.'); continue
        if not user: continue
        print('\nORBIT: '+answer(user))
if __name__=='__main__': main()
