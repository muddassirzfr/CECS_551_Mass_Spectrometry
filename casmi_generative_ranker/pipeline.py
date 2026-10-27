"""Retrieval-free CASMI spectrum -> SMILES -> LightGBM pipeline.
Run `python pipeline.py --help`. Requires train.parquet and sample_submission.csv for actual data.
"""
import argparse, json, math, random
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from tokenizers import Tokenizer, models, trainers, processors
from molecular_preprocessing import (
 POLICY, RDKIT_VERSION, apply_preprocessing, build_preprocessing,
 canonical_candidate, load_preprocessing, require_competition_rdkit,
)
RDLogger.DisableLog('rdApp.*')

PAD, BOS, EOS, UNK = 0, 1, 2, 3
FEATURES = ['logprob','logprob_per_token','mass_error_ppm','mass_error_da','length','n_atoms','n_rings','n_hetero','n_spectra','n_peaks','peak_entropy','candidate_frequency','best_logprob','adduct_known']

def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def amp(args):
 return torch.autocast(device_type='cuda',dtype=torch.bfloat16 if args.bf16 else torch.float16,enabled=args.device.startswith('cuda'))

def canonical(smiles):
 return canonical_candidate(smiles)

def neutral_mass(precursor,adduct):
 # Single-charge common adducts only; others intentionally get no mass constraint.
 offsets={'[M+H]+':1.007276466812,'[M-H]-':-1.007276466812,'[M+Na]+':22.989218,'[M+NH4]+':18.033823,'[M+K]+':38.963158}
 return float(precursor)-offsets[adduct] if adduct in offsets else None

def peaks(row,max_peaks):
 mz=np.asarray(row['ms2_mzs'],dtype=np.float32)
 ints=np.asarray(row['ms2_normalized_intensities'],dtype=np.float32)
 ok=np.isfinite(mz)&np.isfinite(ints)&(mz>0)&(ints>0)
 mz,ints=mz[ok],ints[ok]
 if len(mz):
  ix=np.argsort(-ints)[:max_peaks-1];mz,ints=mz[ix],ints[ix];ints=ints/max(float(ints.max()),1e-9)
 mz=np.r_[float(row['precursor_mz']),mz].astype(np.float32)
 ints=np.r_[2.,ints].astype(np.float32)
 return mz,ints

class Spectra(Dataset):
 def __init__(self,df,max_peaks):self.df=df.reset_index(drop=True);self.max_peaks=max_peaks
 def __len__(self):return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i];m,z=peaks(r,self.max_peaks)
  return {'mz':m,'int':z,'smiles':r.get('normalized_smiles',None),'id':str(r.get('molecule_id',i)),'adduct':str(r.get('adduct','')),'precursor':float(r['precursor_mz'])}

def collate(items,tokenizer=None):
 b=len(items);n=max(len(x['mz']) for x in items)
 mz=torch.zeros(b,n);ints=torch.zeros(b,n);mask=torch.zeros(b,n,dtype=torch.bool)
 for i,x in enumerate(items):
  k=len(x['mz']);mz[i,:k]=torch.from_numpy(x['mz']);ints[i,:k]=torch.from_numpy(x['int']);mask[i,:k]=True
 out={'mz':mz,'ints':ints,'mask':mask,'meta':items}
 if tokenizer is not None:
  seq=[tokenizer.encode(x['smiles']).ids for x in items];t=torch.full((b,max(map(len,seq))),PAD,dtype=torch.long)
  for i,s in enumerate(seq):t[i,:len(s)]=torch.tensor(s)
  out['tokens']=t
 return out

class PeakEmbedding(nn.Module):
 def __init__(self,d):
  super().__init__();freq=2*math.pi/torch.logspace(-2,3,d//2);self.register_buffer('freq',freq)
  self.mlp=nn.Sequential(nn.Linear(d+1,d),nn.GELU(),nn.Linear(d,d))
 def forward(self,mz,ints):
  a=mz.unsqueeze(-1)*self.freq;return self.mlp(torch.cat([a.sin(),a.cos(),ints.unsqueeze(-1)],-1))

class Generator(nn.Module):
 def __init__(self,vocab,d=384,heads=8,enc_layers=4,dec_layers=4,ff=1536,dropout=.1,max_len=256,norm_first=True):
  super().__init__();self.settings=dict(vocab=vocab,d=d,heads=heads,enc_layers=enc_layers,dec_layers=dec_layers,ff=ff,dropout=dropout,max_len=max_len,norm_first=norm_first)
  # Pre-LN layers need a final LayerNorm on each stack; they train stably from scratch at higher learning rates.
  self.peak=PeakEmbedding(d);self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,heads,ff,dropout,batch_first=True,activation='gelu',norm_first=norm_first),enc_layers,norm=nn.LayerNorm(d) if norm_first else None,enable_nested_tensor=False)
  self.emb=nn.Embedding(vocab,d,padding_idx=PAD);self.pos=nn.Embedding(max_len,d)
  self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d,heads,ff,dropout,batch_first=True,activation='gelu',norm_first=norm_first),dec_layers,norm=nn.LayerNorm(d) if norm_first else None)
  self.head=nn.Linear(d,vocab)
 @property
 def max_len(self):return self.pos.num_embeddings
 def encode(self,b):return self.encoder(self.peak(b['mz'],b['ints']),src_key_padding_mask=~b['mask'])
 def decode(self,t,mem,mask):
  n=t.shape[1]
  if n>self.max_len:raise ValueError(f'token sequence length {n} exceeds model max_len {self.max_len}')
  x=self.emb(t)+self.pos(torch.arange(n,device=t.device))[None,:,:]
  causal=torch.ones(n,n,device=t.device,dtype=torch.bool).triu(1)
  z=self.decoder(x,mem,tgt_mask=causal,tgt_key_padding_mask=(t==PAD),memory_key_padding_mask=~mask)
  return self.head(z)
 def forward(self,b,reduction='mean'):
  mem=self.encode(b);t=b['tokens'];logits=self.decode(t[:,:-1],mem,b['mask'])
  return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),t[:,1:].reshape(-1),ignore_index=PAD,reduction=reduction)
 @torch.inference_mode()
 def sample(self,b,n_samples=32,max_tokens=0,temperature=1.,chunk=8):
  self.eval();mem=self.encode(b);out=[]
  # The BOS token occupies one position, so at most max_len-1 tokens can be generated.
  max_tokens=self.max_len-1 if max_tokens<=0 else min(max_tokens,self.max_len-1)
  for j in range(0,n_samples,chunk):
   n=min(chunk,n_samples-j);m=mem.repeat_interleave(n,0);mask=b['mask'].repeat_interleave(n,0)
   ids=torch.full((m.shape[0],1),BOS,dtype=torch.long,device=m.device);scores=torch.zeros(m.shape[0],device=m.device);lengths=torch.zeros_like(scores);done=torch.zeros(m.shape[0],dtype=torch.bool,device=m.device)
   for _ in range(max_tokens):
    logits=self.decode(ids,m,mask)[:,-1,:].float();logp=F.log_softmax(logits,-1)
    logits=logits/temperature;logits[:,PAD]=-1e9;logits[:,BOS]=-1e9;logits[:,UNK]=-1e9
    nxt=torch.multinomial(F.softmax(logits,-1),1).squeeze(1);nxt=torch.where(done,torch.full_like(nxt,EOS),nxt)
    scores+=torch.where(done,0.,logp.gather(1,nxt[:,None]).squeeze(1));lengths+=(~done).float()
    done|=nxt==EOS;ids=torch.cat([ids,nxt[:,None]],1)
    if done.all():break
   out.extend([(ids[k].cpu().tolist(),float(scores[k]),int(lengths[k])) for k in range(ids.shape[0])])
  return [out[i*n_samples:(i+1)*n_samples] for i in range(len(b['meta']))] if n_samples<=chunk else _rearrange_samples(out,len(b['meta']),n_samples,chunk)

def _rearrange_samples(out,b,n,chunk):
 # Sampling is performed in chunks, with each chunk grouped by input spectrum.
 grouped=[[] for _ in range(b)];p=0
 for start in range(0,n,chunk):
  k=min(chunk,n-start)
  for i in range(b):grouped[i].extend(out[p+i*k:p+(i+1)*k])
  p+=b*k
 return grouped

def train_tokenizer(df,path,vocab):
 tk=Tokenizer(models.BPE(unk_token='[UNK]'));tk.pre_tokenizer=None
 strings=sorted(df.normalized_smiles.unique())
 tr=trainers.BpeTrainer(vocab_size=vocab,special_tokens=['<pad>','<s>','</s>','[UNK]'],initial_alphabet=sorted(set(''.join(strings))))
 tk.train_from_iterator(strings,trainer=tr)
 tk.post_processor=processors.TemplateProcessing(single='<s> $A </s>',special_tokens=[('<s>',BOS),('</s>',EOS)])
 tk.save(str(path));return tk

def token_lengths(tk,smiles):
 # Lengths include BOS and EOS; each unique SMILES is tokenized once.
 s=smiles.astype(str);u=pd.unique(s).tolist()
 return s.map(dict(zip(u,(len(e.ids) for e in tk.encode_batch(u)))))

def split_structures(df,seed,holdout):
 from sklearn.model_selection import train_test_split
 if 'competition_identity' not in df:raise ValueError('Normalize molecular identities before splitting')
 keys=np.sort(df.competition_identity.unique());a,b=train_test_split(keys,test_size=holdout,random_state=seed)
 return df[df.competition_identity.isin(a)].copy(),df[df.competition_identity.isin(b)].copy()

def move(b,device):return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}

def warmup_cosine(warmup,total):
 def f(step):
  if step<warmup:return (step+1)/warmup
  return .5*(1+math.cos(math.pi*min((step-warmup)/max(total-warmup,1),1.)))
 return f

@torch.no_grad()
def validation_loss(model,dl,args):
 model.eval();total=0.;count=0
 for batch in dl:
  batch=move(batch,args.device)
  with amp(args):total+=float(model(batch,reduction='sum'))
  count+=int((batch['tokens'][:,1:]!=PAD).sum())
 model.train();return total/max(count,1)

def save_generator(model,path,args):torch.save({'model':model.state_dict(),'settings':model.settings,'max_peaks':args.max_peaks,'normalization_policy':POLICY,'rdkit_version':RDKIT_VERSION},path)

def fit_generator(args):
 p=Path(args.output);p.mkdir(parents=True,exist_ok=True)
 preprocessing=Path(args.preprocessing) if args.preprocessing else p/'preprocessing'
 if not args.preprocessing:
  build_preprocessing(args.train,preprocessing,args.seed,args.holdout,args.val_frac,args.preprocessing_workers)
 mapping,report=load_preprocessing(preprocessing)
 print('Reusing identity partitions:',report['partitions'],flush=True)
 source=pd.read_parquet(args.train)
 if len(source)!=report['rows_scanned']:raise ValueError('Input row count differs from preprocessing; regenerate the manifest')
 df=apply_preprocessing(source,mapping)
 df=df.dropna(subset=['ms2_mzs','ms2_normalized_intensities','precursor_mz'])
 # Use the single identity-level manifest; never split raw supplied keys again.
 a=df[df.split=='Training'];val=df[df.split=='Validation'];b=df[df.split=='Ranker holdout']
 sets=[set(part.competition_identity) for part in (a,val,b)]
 if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Identity leakage between partitions')
 (p/'normalization_policy.json').write_text(json.dumps({'policy':POLICY,'rdkit_version':RDKIT_VERSION,'preprocessing':str(preprocessing.resolve())},indent=2))
 if args.limit:a=a.sample(min(args.limit,len(a)),random_state=args.seed)
 if args.val_max:val=val.sample(min(args.val_max,len(val)),random_state=args.seed)
 tk=train_tokenizer(a,p/'tokenizer.json',args.vocab)
 # Sequences longer than the positional table cannot be trained on or generated; drop them rather than truncate into invalid SMILES.
 def drop_long(part,name):
  keep=token_lengths(tk,part.normalized_smiles)<=args.max_len
  print(f'{name}: dropped {int((~keep).sum())} of {len(part)} spectra with SMILES longer than {args.max_len} tokens',flush=True)
  return part[keep]
 a=drop_long(a,'train');val=drop_long(val,'val')
 if a.empty:raise ValueError(f'No training spectra left after the --max-len {args.max_len} filter; raise --max-len.')
 a.to_parquet(p/'generator_train.parquet');val.to_parquet(p/'generator_val.parquet');b.to_parquet(p/'ranker_holdout.parquet')
 model=Generator(tk.get_vocab_size(),args.d,args.heads,args.enc_layers,args.dec_layers,args.ff,max_len=args.max_len,norm_first=not args.post_ln).to(args.device)
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
 coll=partial(collate,tokenizer=tk)
 dl=DataLoader(Spectra(a,args.max_peaks),batch_size=args.batch,shuffle=True,collate_fn=coll,num_workers=args.workers)
 vdl=DataLoader(Spectra(val,args.max_peaks),batch_size=args.batch,shuffle=False,collate_fn=coll,num_workers=args.workers)
 total_steps=args.epochs*math.ceil(len(dl)/args.accum);warmup=min(args.warmup,max(1,total_steps//10))
 sched=torch.optim.lr_scheduler.LambdaLR(opt,warmup_cosine(warmup,total_steps))
 print(f'train spectra={len(a)} val spectra={len(val)} ranker holdout spectra={len(b)} optimizer steps={total_steps} warmup={warmup}',flush=True)
 scaler=torch.amp.GradScaler('cuda',enabled=args.device.startswith('cuda') and not args.bf16)
 best=math.inf;opt_step=0;log=open(p/'train_log.jsonl','a')
 def checkpoint(epoch,train_loss):
  nonlocal best
  vl=validation_loss(model,vdl,args) if len(val) else math.nan
  save_generator(model,p/'generator_last.pt',args)
  improved=len(val)==0 or vl<best
  if improved:best=vl if len(val) else best;save_generator(model,p/'generator.pt',args)
  rec={'epoch':epoch,'opt_step':opt_step,'lr':sched.get_last_lr()[0],'train_loss':train_loss,'val_loss':vl,'best_val_loss':best,'saved_best':improved}
  print(json.dumps(rec),flush=True);log.write(json.dumps(rec)+'\n');log.flush()
 for epoch in range(args.epochs):
  model.train();total=0.;steps=0;opt.zero_grad(set_to_none=True)
  for step,batch in enumerate(dl):
   batch=move(batch,args.device)
   with amp(args):
    loss=model(batch)/args.accum
   scaler.scale(loss).backward()
   total+=float(loss.detach())*args.accum;steps+=1
   if (step+1)%args.accum==0 or step+1==len(dl):
    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);sched.step();opt_step+=1
    if args.eval_every and opt_step%args.eval_every==0:checkpoint(epoch+1,total/max(steps,1))
  checkpoint(epoch+1,total/max(steps,1))
 log.close()
 print('Saved best generator (generator.pt), last generator (generator_last.pt), tokenizer, and structure-disjoint splits to',p)

def load_generator(folder,device):
 require_competition_rdkit()
 p=Path(folder);ck=torch.load(p/'generator.pt',map_location='cpu',weights_only=False)
 if ck.get('normalization_policy')!=POLICY or ck.get('rdkit_version')!=RDKIT_VERSION:
  raise ValueError('Generator uses a different or missing molecular normalization policy; regenerate splits/tokenizer and retrain')
 m=Generator(**ck['settings']);m.load_state_dict(ck['model']);m.to(device).eval()
 return m,Tokenizer.from_file(str(p/'tokenizer.json')),ck['max_peaks']

def decode_smiles(tk,ids):
 if EOS in ids:ids=ids[:ids.index(EOS)]
 ids=[i for i in ids if i not in (PAD,BOS)]
 return tk.decode(ids,skip_special_tokens=True).replace(' ','')

def build_candidates(df,model,tk,max_peaks,args):
 loader=DataLoader(Spectra(df,max_peaks),batch_size=args.infer_batch,shuffle=False,collate_fn=collate,num_workers=args.workers)
 records={};done=0
 for nb,batch in enumerate(loader,1):
  batch=move(batch,args.device)
  with amp(args):samples=model.sample(batch,args.samples,args.max_tokens,args.temperature,args.sample_chunk)
  for meta,seqs in zip(batch['meta'],samples):
   group=records.setdefault(meta['id'],{'rows':[],'candidates':{},'n_samples':0});group['rows'].append(meta);group['n_samples']+=len(seqs)
   for ids,score,n in seqs:
    parsed=canonical(decode_smiles(tk,ids))
    if parsed is None:continue
    sm,ikey,mol=parsed
    old=group['candidates'].get(ikey)
    if old is None:group['candidates'][ikey]={'smiles':sm,'mol':mol,'scores':[score],'lengths':[n]}
    else:old['scores'].append(score);old['lengths'].append(n)
  done+=len(batch['meta'])
  if args.progress_every and nb%args.progress_every==0:print(f'sampled {done}/{len(df)} spectra',flush=True)
 return records

def features(group,entry):
 rows=group['rows'];mol=entry['mol'];mass=Descriptors.ExactMolWt(mol)
 errors=[];known=0
 for r in rows:
  target=neutral_mass(r['precursor'],r['adduct'])
  if target is not None and target>0:errors.append(abs(mass-target));known=1
 err=min(errors) if errors else 1e3
 peaks_count=np.mean([len(x['mz'])-1 for x in rows]);entropy=[]
 for r in rows:
  v=np.asarray(r['int'][1:]);v=v/v.sum() if v.sum()>0 else v
  entropy.append(float(-(v*np.log(v+1e-12)).sum()))
 scores=entry['scores'];lengths=entry['lengths'];best=max(scores)
 # Frequency is a fraction of all samples drawn for the molecule, so it is comparable however many spectra a molecule has.
 freq=len(scores)/max(group['n_samples'],1)
 return [best,max(s/max(l,1) for s,l in zip(scores,lengths)),err/max(mass,1)*1e6,err,len(entry['smiles']),mol.GetNumHeavyAtoms(),rdMolDescriptors.CalcNumRings(mol),sum(a.GetAtomicNum() not in (1,6) for a in mol.GetAtoms()),len(rows),peaks_count,float(np.mean(entropy)),freq,best,known]

def candidate_frame(groups,truth=None):
 data=[]
 for gid,group in groups.items():
  truth_key=truth.get(gid) if truth else None
  for key,entry in group['candidates'].items():
   row={'molecule_id':gid,'smiles':entry['smiles'],'inchikey14':key,'label':int(key==truth_key) if truth is not None else -1}
   row.update(zip(FEATURES,features(group,entry)));data.append(row)
 return pd.DataFrame(data,columns=['molecule_id','smiles','inchikey14','label']+FEATURES)

def ranking_metrics(frame,truth,score,k=25):
 # truth covers every evaluated molecule, so molecules with no valid candidates count as misses.
 if not truth:return {}
 hits={}
 if not frame.empty:
  f=frame.sort_values(['molecule_id',score],ascending=[True,False]).copy();f['rank']=f.groupby('molecule_id').cumcount()+1
  hits=f[f.label==1].groupby('molecule_id')['rank'].min().to_dict()
 ranks=np.array([hits.get(g,np.inf) for g in truth],dtype=float)
 return {f'mrr@{k}':float(np.where(ranks<=k,1/ranks,0.).mean()),'top1':float((ranks<=1).mean()),'top10':float((ranks<=10).mean()),f'top{k}':float((ranks<=k).mean()),'in_any_candidate':float(np.isfinite(ranks).mean()),'molecules':len(ranks)}

def positive_groups(frame):return frame[frame.molecule_id.isin(frame.groupby('molecule_id').label.max().loc[lambda x:x>0].index)]

def train_lambdarank(frame,seed):
 import lightgbm as lgb
 frame=positive_groups(frame).sort_values('molecule_id');sizes=frame.groupby('molecule_id',sort=True).size().tolist()
 # Single-threaded: LightGBM's OpenMP runtime can segfault once PyTorch's has run in the same process (seen on macOS); ranker data is small.
 ranker=lgb.LGBMRanker(objective='lambdarank',n_estimators=250,learning_rate=.04,num_leaves=15,min_child_samples=5,random_state=seed,n_jobs=1,verbosity=-1)
 ranker.fit(frame[FEATURES],frame.label,group=sizes);return ranker

def fit_ranker(args):
 from sklearn.model_selection import train_test_split
 p=Path(args.output);df=pd.read_parquet(p/'ranker_holdout.parquet');df['inchikey14']=df['inchikey14'].astype(str)
 keys=df.inchikey14.unique()
 # Sampling every holdout spectrum takes days at full scale; use a random subset of structures and spectra.
 if args.ranker_limit and len(keys)>args.ranker_limit:keys=np.random.default_rng(args.seed).choice(keys,args.ranker_limit,replace=False)
 df=df[df.inchikey14.isin(keys)]
 if args.spectra_per_structure:df=df.sample(frac=1,random_state=args.seed).groupby('inchikey14').head(args.spectra_per_structure)
 df=df.copy();df['molecule_id']=df['inchikey14'] # group spectra by structure for evaluation
 tr_keys,ev_keys=train_test_split(np.sort(keys),test_size=args.ranker_eval_frac,random_state=args.seed)
 print(f'ranker structures={len(keys)} (train={len(tr_keys)} eval={len(ev_keys)}) spectra={len(df)} samples/spectrum={args.samples}',flush=True)
 m,tk,max_peaks=load_generator(p,args.device)
 groups=build_candidates(df,m,tk,max_peaks,args);frame=candidate_frame(groups,{k:k for k in keys})
 frame.to_parquet(p/'ranker_candidates.parquet',index=False)
 tr=frame[frame.molecule_id.isin(tr_keys)];ev=frame[frame.molecule_id.isin(ev_keys)].copy();ev_truth={k:k for k in ev_keys}
 report={'structures':len(keys),'train_structures':len(tr_keys),'eval_structures':len(ev_keys),'spectra':len(df),'samples_per_spectrum':args.samples,
  'valid_unique_candidates_per_structure':float(frame.groupby('molecule_id').size().reindex(keys,fill_value=0).mean()),
  'candidate_recall_all':float(frame.groupby('molecule_id').label.max().reindex(keys,fill_value=0).mean()),
  'eval':{f'generator_{s}':ranking_metrics(ev,ev_truth,s) for s in ('logprob','logprob_per_token','candidate_frequency')}}
 n_pos=int(positive_groups(tr).molecule_id.nunique());report['train_structures_with_positive']=n_pos
 def write():
  (p/'ranker_metrics.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
 if n_pos<args.min_positive_groups:
  write();raise RuntimeError(f'Only {n_pos} ranker-train structures have a correct generated candidate (need {args.min_positive_groups}). Improve generator recall before training the ranker; metrics written to ranker_metrics.json.')
 ranker=train_lambdarank(tr,args.seed)
 if not ev.empty:ev['rank_score']=ranker.predict(ev[FEATURES])
 report['eval']['lightgbm']=ranking_metrics(ev,ev_truth,'rank_score')
 # Eval numbers above come from the train-only ranker; the saved model is refit on every labelled structure.
 if not args.no_refit:ranker=train_lambdarank(frame,args.seed)
 report['saved_ranker_refit_on_all']=not args.no_refit
 (p/'ranker_normalization_policy.json').write_text(json.dumps({'policy':POLICY,'rdkit_version':RDKIT_VERSION},indent=2))
 ranker.booster_.save_model(str(p/'ranker.txt'));write()
 print('Saved ranker to',p/'ranker.txt')

def order_candidates(frame,folder):
 import lightgbm as lgb
 if frame.empty:return frame
 ranker_path=Path(folder)/'ranker.txt'
 if ranker_path.exists():
  policy_path=Path(folder)/'ranker_normalization_policy.json'
  if not policy_path.exists() or json.loads(policy_path.read_text())!={'policy':POLICY,'rdkit_version':RDKIT_VERSION}:
   raise ValueError('Ranker uses a different or missing molecular normalization policy; retrain it')
  ranker=lgb.Booster(model_file=str(ranker_path));frame['rank_score']=ranker.predict(frame[FEATURES],num_threads=1)
 else:
  print('WARNING: ranker.txt missing; sorting by generator scores only')
  frame['rank_score']=frame['logprob_per_token']
 return frame.sort_values(['molecule_id','rank_score'],ascending=[True,False])

def submit(args):
 p=Path(args.output);df=pd.read_parquet(args.test)
 if 'molecule_id' not in df:raise ValueError('test data must contain molecule_id')
 m,tk,max_peaks=load_generator(p,args.device)
 groups=build_candidates(df,m,tk,max_peaks,args)
 frame=order_candidates(candidate_frame(groups),p)
 frame.to_csv(p/'ranked_candidates_debug.csv',index=False)
 sample=pd.read_csv(args.sample_submission)
 print('Sample submission columns:',sample.columns.tolist())
 # Only write when an unambiguous supported wide format is detected.
 idcol='molecule_id'
 predcols=[c for c in sample.columns if c!=idcol]
 if idcol not in sample or len(predcols)!=25:
  raise ValueError('Unknown sample_submission schema: inspect sample_submission.csv and adapt writer. Candidate debug CSV saved.')
 preds={str(k):v.smiles.head(25).tolist() for k,v in frame.groupby('molecule_id',sort=False)}
 for i,row in sample.iterrows():
  values=preds.get(str(row[idcol]),[])
  for j,c in enumerate(predcols):sample.at[i,c]=values[j] if j<len(values) else ''
 sample.to_csv(p/'submission.csv',index=False)
 print('Wrote',p/'submission.csv','rows',len(sample))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preprocess','train','ranker','submit']);ap.add_argument('--train');ap.add_argument('--test');ap.add_argument('--sample-submission');ap.add_argument('--output',default='artifacts');ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--preprocessing',help='existing preprocessing directory; its saved identity partitions are reused')
 ap.add_argument('--preprocessing-workers',type=int,default=4,help='processes for normalizing unique molecules')
 ap.add_argument('--holdout',type=float,default=.1);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--vocab',type=int,default=512);ap.add_argument('--d',type=int,default=384);ap.add_argument('--heads',type=int,default=8);ap.add_argument('--enc-layers',type=int,default=4);ap.add_argument('--dec-layers',type=int,default=4);ap.add_argument('--ff',type=int,default=1536);ap.add_argument('--max-peaks',type=int,default=128);ap.add_argument('--batch',type=int,default=32);ap.add_argument('--accum',type=int,default=4);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--workers',type=int,default=0);ap.add_argument('--bf16',action='store_true')
 ap.add_argument('--max-len',type=int,default=160,help='max SMILES tokens incl. BOS/EOS; longer training SMILES are dropped');ap.add_argument('--lr',type=float,default=3e-4,help='peak learning rate');ap.add_argument('--warmup',type=int,default=2000,help='warmup optimizer steps (capped at 10%% of training)');ap.add_argument('--post-ln',action='store_true',help='use the original post-LayerNorm layers instead of pre-LN')
 ap.add_argument('--val-frac',type=float,default=.02,help='fraction of generator-train structures held out for validation loss');ap.add_argument('--val-max',type=int,default=5000,help='max validation spectra (0 = all)');ap.add_argument('--eval-every',type=int,default=2000,help='validate/checkpoint every N optimizer steps (0 = epoch end only)')
 ap.add_argument('--infer-batch',type=int,default=8);ap.add_argument('--samples',type=int,default=100);ap.add_argument('--sample-chunk',type=int,default=100);ap.add_argument('--max-tokens',type=int,default=0,help='0 = model max_len-1');ap.add_argument('--temperature',type=float,default=1.);ap.add_argument('--progress-every',type=int,default=50,help='print sampling progress every N batches (0 = off)')
 ap.add_argument('--ranker-limit',type=int,default=2000,help='holdout structures to sample for the ranker (0 = all; slow)');ap.add_argument('--spectra-per-structure',type=int,default=5,help='max spectra per ranker structure (0 = all)');ap.add_argument('--ranker-eval-frac',type=float,default=.3,help='fraction of ranker structures held out to measure MRR@25');ap.add_argument('--min-positive-groups',type=int,default=2);ap.add_argument('--no-refit',action='store_true',help='save the train-split ranker instead of refitting on all ranker structures')
 args=ap.parse_args();seed_all(args.seed)
 if args.command=='preprocess':build_preprocessing(args.train,args.output,args.seed,args.holdout,args.val_frac,args.preprocessing_workers)
 elif args.command=='train':fit_generator(args)
 elif args.command=='ranker':fit_ranker(args)
 else:submit(args)
if __name__=='__main__':main()
