"""Observation-preserving CSDI alignment; paired validation-only inference."""
import argparse,json,sys,time,hashlib,os
from pathlib import Path
import numpy as np

ARMS=['original','padding_control','padding_aligned']
SEEDS=[1234,2345,3456,4567,5678]

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp.json')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(path)

def setup(root):
    sys.path.insert(0,str(root/'scripts'));import torch,bench_common as C,bench_models as M
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert C.SPLIT['val']==[2023] and (root/'cluster/control/hold_station_training.json').exists()
    return torch,C,M

def encoded(M,net,inp,starts,gaps,scale,arm):
    B=len(inp);W=inp.shape[1];N=W if arm=='original' else W+5
    pads=np.zeros(B,dtype=int) if arm!='padding_aligned' else (-np.asarray(gaps))%5
    # Only sequence length changes. All learned weights and five feature channels remain fixed.
    M.C.W=N;M.set_patch(5);net.L=N//5
    xs=[];ms=[];centers=[]
    for j,pad in enumerate(pads):
        x=np.full(N,np.nan,np.float32);x[pad:pad+W]=inp[j]
        assert np.array_equal(x[pad:pad+W],inp[j],equal_nan=True)
        assert np.isfinite(x).sum()==np.isfinite(inp[j]).sum()
        ft=M.featurize(x,int(starts[j]-pad),scale);xs.append(ft[0]);ms.append(ft[1]);centers.append(ft[2])
        if arm=='padding_aligned':assert (gaps[j]+pad)%5==0
    return np.stack(xs),np.stack(ms),np.asarray(centers),pads

def sample(torch,net,X,mask,noise=None,offset=None):
    cond=mask[:,:,:5].permute(0,2,1);obs=X[:,:,:5].permute(0,2,1)*cond;side=net._side(X,cond)
    B,K,T=obs.shape;N=K*T;index=None if offset is None else offset[:,None]+torch.arange(N,device=X.device)[None,:]
    def draw(k):
        if noise is None:return torch.randn_like(obs)
        return noise[k].gather(1,index).reshape(B,T,K).permute(0,2,1)
    current=draw(0)
    for t in range(net.body.n_diffusion_steps-1,-1,-1):
        inp=torch.cat([(cond*obs).unsqueeze(1),((1-cond)*current).unsqueeze(1)],dim=1)
        eps=net.body.diff_model(inp,side,torch.tensor([t],device=X.device))
        c1=1/net.body.alpha_hat[t]**.5;c2=(1-net.body.alpha_hat[t])/(1-net.body.alpha[t])**.5
        current=c1*(current-c2*eps)
        if t>0:current+=((1-net.body.alpha[t-1])/(1-net.body.alpha[t])*net.body.beta[t])**.5*draw(net.body.n_diffusion_steps-t)
    return (obs*cond+current*(1-cond)).permute(0,2,1)

def noise_field(torch,ids,L,draw,W,steps):
    # Per-window streams make the pairing invariant to batch size, job order and restarts.
    arr=[]
    for i in ids:
        seed=20261101+int(L)*10000019+int(i)*1009+int(draw)*1000000007
        gen=torch.Generator(device='cpu');gen.manual_seed(seed)
        arr.append(torch.randn((steps,W+10),generator=gen))
    return torch.stack(arr,dim=1).cuda()

def run_batch(torch,M,net,ck,F,starts,gaps,ids,L,draw,deadline=float('inf'),test=False):
    W=8640;starts=np.asarray(starts,dtype=int);gaps=np.asarray(gaps,dtype=int)
    inp=np.array([F[st:st+W] for st in starts],np.float32)
    for j,gp in enumerate(gaps):inp[j,gp:gp+L]=np.nan
    fields=noise_field(torch,ids,L,draw,W,net.body.n_diffusion_steps);pred=[];checks={};native_center=None;prepared=[]
    for arm in ARMS:
        x,m,center,pads=encoded(M,net,inp,starts,gaps,ck['scale'],arm)
        if native_center is None:native_center=center
        else:assert np.array_equal(center,native_center),'Padding changed normalization'
        X=torch.from_numpy(x).cuda();mask=torch.from_numpy(m).cuda();offset=torch.from_numpy(5-pads).cuda()
        if test:
            index=offset[:,None]+torch.arange(x.shape[1]*5,device='cuda')[None,:]
            packed=fields[0].gather(1,index)
            for j,pad in enumerate(pads):assert torch.equal(packed[j,pad:pad+W],fields[0,j,5:5+W])
        if time.time()>deadline:raise SystemExit(75)
        net.L=x.shape[1]
        with torch.no_grad():
            if test and arm=='original':
                torch.manual_seed(101);torch.cuda.manual_seed_all(101);original=net(X,mask)
                torch.manual_seed(101);torch.cuda.manual_seed_all(101);copied=sample(torch,net,X,mask)
                delta=float(abs(original-copied).max());assert delta<1e-4;checks['native_sampler_max_difference']=delta
            y=sample(torch,net,X,mask,fields,offset).cpu().numpy().reshape(len(ids),-1)
        result=[]
        for j,(gp,pad) in enumerate(zip(gaps,pads)):
            full=y[j,pad:pad+W]*ck['scale']+center[j]
            restored=np.where(np.isfinite(inp[j]),inp[j],full)
            assert np.array_equal(restored[np.isfinite(inp[j])],inp[j,np.isfinite(inp[j])])
            result.append(restored[gp:gp+L].astype(np.float32))
        pred.append(np.asarray(result));prepared.append((x,m,pads))
    pred=np.asarray(pred);assert np.isfinite(pred).all()
    if test:
        for j,gp in enumerate(gaps):
            if gp%5==0:assert np.array_equal(pred[1,j],pred[2,j]),'Identical padding arms differ'
        checks.update(observations_preserved=True,normalization_preserved=True,absolute_noise_preserved=True,aligned_boundaries=bool(L%5==0))
    return pred,checks

def metrics(pred,true,margin,L):
    import analyze_geomag_events as E
    ref=E.reference(true,margin,L);ans={}
    for arm in range(3):
        for draw in range(pred.shape[1]):
            pr=np.asarray(pred[arm,draw],float);met=E.features(pr,ref);t=ref['t']
            met['first5_mae']=abs(pr[:,:5]-t[:,:5]).mean(1);met['last5_mae']=abs(pr[:,-5:]-t[:,-5:]).mean(1)
            met['edge5_mae']=(met['first5_mae']+met['last5_mae'])/2
            if L>5:
                dp=(pr[:,5:]-pr[:,:-5])/5;dt=(t[:,5:]-t[:,:-5])/5;strength=abs(dt).max(1)
                met['spike5x']=(abs(dp).max(1)>5*np.maximum(strength,1e-6)).astype(float)
                met['max_rate_error']=abs(abs(dp).max(1)-strength)
            for key,val in met.items():
                if key not in ans:ans[key]=np.empty((3,pred.shape[1],len(pr)))
                ans[key][arm,draw]=val
    return ans

def evaluate(a):
    torch,C,M=setup(a.root);W=8640
    protocol=json.loads((a.out/'protocol.json').read_text());deadline=protocol['deadline_unix']
    if time.time()>deadline:raise SystemExit(75)
    dest=a.out/a.stage/f's{a.seed}_L{a.length}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    net,ck=M.load(str(a.root/'models/comp/seeds'/f'csdi_best_s{a.seed}.pt'),'cuda');F=C.load_year(2023,'ARS')
    if a.stage=='primary':ids=np.arange(1024);draws=[0,1]
    else:ids=np.array(protocol['secondary_window_ids']);draws=list(range(2,8))
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz') as z:
        starts=z[f'L{a.length}_win0'][ids].astype(int);gaps=z[f'L{a.length}_gap0'][ids].astype(int)
        true=z[f'L{a.length}_true'][ids].copy();margin=int(z['margin'])
    file=dest/'predictions.npy';shape=(3,len(draws),len(ids),a.length)
    if file.exists():pred=np.lib.format.open_memmap(file,mode='r+');assert pred.shape==shape
    else:pred=np.lib.format.open_memmap(file,mode='w+',dtype=np.float32,shape=shape)
    marker=dest/'progress.json';done=set(json.loads(marker.read_text())['batches']) if marker.exists() else set()
    t0=time.time()
    for b in range(0,len(ids),16):
        if b in done:continue
        if time.time()>deadline:raise SystemExit(75)
        stop=min(b+16,len(ids))
        for j,d in enumerate(draws):
            pr,_=run_batch(torch,M,net,ck,F,starts[b:stop],gaps[b:stop],ids[b:stop],a.length,d,deadline)
            pred[:,j,b:stop]=pr
        pred.flush();done.add(b);write(marker,dict(batches=sorted(done),windows_done=sum(min(16,len(ids)-k) for k in done),total_windows=len(ids)))
        if b%128==0:print(a.stage,a.seed,a.length,stop,'/',len(ids),'seconds',round(time.time()-t0),flush=True)
    met=metrics(pred,true,margin,a.length)
    with open(dest/'metrics.npz.tmp','wb') as f:np.savez_compressed(f,ids=ids,minutes=starts+gaps,**met)
    (dest/'metrics.npz.tmp').replace(dest/'metrics.npz')
    if a.stage=='secondary':
        primary=a.out/'primary'/f's{a.seed}_L{a.length}'
        assert (primary/'complete.json').exists()
        pp=np.load(primary/'predictions.npy',mmap_mode='r')[:,:,ids]
        med=np.median(np.concatenate([pp,np.asarray(pred)],axis=1),axis=1)[:,None]
        mm=metrics(med,true,margin,a.length)
        with open(dest/'median8_metrics.npz','wb') as f:np.savez_compressed(f,ids=ids,minutes=starts+gaps,**mm)
    write(dest/'complete.json',dict(complete=True,stage=a.stage,seed=a.seed,length=a.length,windows=len(ids),draws=draws,seconds=time.time()-t0))

def pilot(a):
    torch,C,M=setup(a.root);a.out.mkdir(parents=True,exist_ok=True)
    net,ck=M.load(str(a.root/'models/comp/seeds'/'csdi_best_s1234.pt'),'cuda');F=C.load_year(2023,'ARS');records=[]
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz') as z:
        for L in [5,1440]:
            starts=z[f'L{L}_win0'];gaps=z[f'L{L}_gap0'];selected=[int(np.flatnonzero(gaps%5==phase)[0]) for phase in range(5)]
            if L==1440:selected=list(dict.fromkeys([52]+selected))
            selected+= [i for i in range(1024) if i not in selected][:16-len(selected)];ids=np.array(selected)
            for draw in [0,1]:
                torch.cuda.synchronize();t=time.time()
                pred,checks=run_batch(torch,M,net,ck,F,starts[ids],gaps[ids],ids,L,draw,test=draw==0)
                torch.cuda.synchronize();seconds=time.time()-t
                records.append(dict(length=L,draw=draw,seconds=seconds,checks=checks,case52_first5=pred[:,0,:5].tolist() if L==1440 else None))
    write(a.out/'pilot.json',dict(complete=True,records=records,seconds_per_3arm_batch=float(np.median([r['seconds'] for r in records if r['draw']==1])),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(records),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=['pilot','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1234);p.add_argument('--length',type=int,default=1440);p.add_argument('--stage',choices=['primary','secondary'],default='primary')
    a=p.parse_args();globals()[a.job](a)
if __name__=='__main__':main()
