"""CSDI phase mechanisms: matched observations and absolute-time diffusion noise."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np

def write(p,d):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.root/'scripts'));import torch,bench_common as C,bench_models as M
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert (a.root/'cluster/control/hold_station_training.json').exists();assert C.SPLIT['val']==[2023]
    out=a.out/f's{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    F=C.load_year(2023,'ARS');net,ck=M.load(str(a.root/'models/comp/seeds'/f'csdi_best_s{a.seed}.pt'),'cuda')
    assert M.P==5 and M.NCH==1 and net.n_samples==1
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz') as z:
        starts=z['L1440_win0'].astype(int);gaps=z['L1440_gap0'].astype(int);true=z['L1440_true'][:,180:1620].astype(float)
    ids=list(dict.fromkeys([52]+np.random.default_rng(20261031).choice(1024,15,False).tolist()));B=len(ids);L=1440
    modes=['natural_native','natural_absolute','common_absolute','rounded_absolute','embedding_absolute']
    sampling=list(range(9101,9109));rows=[];tests=json.loads((out/'tests.json').read_text()) if (out/'tests.json').exists() else {};baseline=[]
    for i in ids:
        ix=np.arange(starts[i]+4,starts[i]+C.W);v=F[ix]
        visible=np.isfinite(v)&~((ix>=starts[i]+gaps[i])&(ix<starts[i]+gaps[i]+L))
        baseline.append({int(j):float(y) for j,y in zip(ix[visible],v[visible])})
    # Align ALL stochastic increments to absolute minutes, not token/channel indices.
    # W+4 spans every shifted window; independent noise is generated for each draw and reverse step.
    def sampler(X,mask,shift,seed,absolute,embedding=False):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cond=mask[:,:,:M.POUT].permute(0,2,1);obs=X[:,:,:M.POUT].permute(0,2,1)*cond;side=net._side(X,cond)
        if embedding:
            f=net.feat_emb[(torch.arange(5,device=X.device)+shift)%5].t()[None,:,:,None].expand(B,-1,5,M.WB)
            side=torch.cat([side[:,:4],f,cond.unsqueeze(1)],dim=1)
        def noise():
            if not absolute:return torch.randn_like(obs)
            timeline=torch.randn((B,C.W+4),device=X.device)
            return timeline[:,shift:shift+C.W].reshape(B,M.WB,5).permute(0,2,1)
        current=noise()
        for t in range(net.body.n_diffusion_steps-1,-1,-1):
            din=torch.cat([(cond*obs).unsqueeze(1),((1-cond)*current).unsqueeze(1)],dim=1)
            epsilon=net.body.diff_model(din,side,torch.tensor([t],device=X.device))
            c1=1/net.body.alpha_hat[t]**.5;c2=(1-net.body.alpha_hat[t])/(1-net.body.alpha[t])**.5
            current=c1*(current-c2*epsilon)
            if t>0:current+=((1-net.body.alpha[t-1])/(1-net.body.alpha[t])*net.body.beta[t])**.5*noise()
        return (obs*cond+current*(1-cond)).permute(0,2,1)
    # Check absolute noise coupling independently, including the five-channel packing.
    torch.manual_seed(99);n0=torch.randn((B,C.W+4),device='cuda')
    n4=n0[:,4:4+C.W].reshape(B,M.WB,5).permute(0,2,1).permute(0,2,1).reshape(B,C.W)
    assert torch.equal(n0[:,4:C.W],n4[:,:C.W-4]);tests['absolute_noise_alignment']=True
    for mode in modes:
        for shift in range(5):
            path=out/f'{mode}_shift{shift}.json'
            if path.exists():rows.extend(json.loads(path.read_text())['rows']);continue
            feats=[];centers=[];gaplist=[];observations=[]
            for i in ids:
                st=starts[i]+shift;gp=gaps[i]-shift;x=F[st:st+C.W].copy();x[gp:gp+L]=np.nan
                if mode.startswith(('common','rounded','embedding')):
                    absolute_idx=st+np.arange(C.W);x[(absolute_idx<starts[i]+4)|(absolute_idx>=starts[i]+C.W)]=np.nan
                if mode=='rounded_absolute':
                    x[(gp//5)*5:((gp+L+4)//5)*5]=np.nan
                ft=M.featurize(x,int(st),ck['scale']);feats.append(ft);centers.append(ft[2]);gaplist.append(gp)
                observations.append({int(st+j):float(v) for j,v in enumerate(x) if np.isfinite(v)})
            if mode=='common_absolute':
                assert observations==baseline,'Common-support observations differ'
            X=torch.from_numpy(np.stack([f[0] for f in feats])).cuda();mask=torch.from_numpy(np.stack([f[1] for f in feats])).cuda()
            if mode=='natural_native' and shift==0:
                with torch.no_grad():
                    torch.manual_seed(9101);torch.cuda.manual_seed_all(9101);original=net(X,mask)
                    copied=sampler(X,mask,0,9101,False)
                    delta=float(abs(original-copied).max());tests['sampler_vs_original_max_error']=delta
                    assert delta<1e-4,delta
                    write(out/'tests.json',tests)
            samples=[]
            for seed in sampling:
                with torch.no_grad():yy=sampler(X,mask,shift,seed,mode!='natural_native',mode=='embedding_absolute').cpu().numpy().reshape(B,C.W)
                predictions=[(yy[j]*ck['scale']+centers[j])[gp:gp+L] for j,gp in enumerate(gaplist)]
                samples.append(predictions)
            samples=np.asarray(samples,np.float32);assert np.isfinite(samples).all()
            with open(out/f'{mode}_shift{shift}.npz.tmp','wb') as f:np.savez_compressed(f,pred=samples,ids=ids,sampling_seeds=sampling)
            (out/f'{mode}_shift{shift}.npz.tmp').replace(out/f'{mode}_shift{shift}.npz')
            rr=[]
            for j,i in enumerate(ids):
                pr=samples[:,j].astype(float);t=true[i];dp=(pr[:,5:]-pr[:,:-5])/5;dr=(t[5:]-t[:-5])/5;k=int(np.argmax(abs(dr)))
                loc=np.argmax(abs(dp),axis=1)
                rr.append(dict(mode=mode,shift=shift,window=i,training_seed=a.seed,true_rate=float(dr[k]),true_peak_index=k,
                    gap_phase=int(gaplist[j]%5),center=float(centers[j]),mae=abs(pr-t).mean(1).tolist(),
                    max_rate=abs(dp).max(1).tolist(),interior_max_rate=abs(dp[:,10:-10]).max(1).tolist(),
                    max_rate_index=loc.tolist(),capture_pct=(100*dp[:,k]/dr[k]).tolist(),
                    first5_error=abs(pr[:,:5]-t[:5]).mean(1).tolist()))
            write(path,dict(rows=rr));rows.extend(rr);print(a.seed,mode,shift,'done',flush=True)
    tests['common_support_identical']=True
    write(out/'complete.json',dict(complete=True,rows=rows,tests=tests,seeds=sampling,ids=ids,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation={'natural_native':'Same tensor RNG, actual shifted context',
        'natural_absolute':'Noise matched to absolute minutes, actual shifted context',
        'common_absolute':'Same absolute observed values and noise at all phases; block grouping and time encoding still change',
        'rounded_absolute':'Same common outer support, additionally hide observed minutes in partially missing gap-boundary blocks',
        'embedding_absolute':'Same common support and noise; cyclic feature-embedding permutation to preserve the original absolute minute phase'},
        limits=['Targeted pathology plus 15 fixed random windows; not population prevalence',
                'Mask rounding and embedding permutation are diagnostic interventions, not validated fixes; no retraining',
                'Absolute noise coupling changes joint cross-phase samples but each phase retains independent standard-normal noise']))
if __name__=='__main__':main()
