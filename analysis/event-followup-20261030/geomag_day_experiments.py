"""Isolated ARS-only exploratory experiment jobs, with resumable per-job outputs."""
import argparse,datetime,hashlib,json,os,random,sys,time,types
from pathlib import Path
import numpy as np
_deadlines={}

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def save_npz(path,**data):
    tmp=str(path)+'.tmp'
    with open(tmp,'wb') as f:np.savez_compressed(f,**data)
    os.replace(tmp,path)

def deadline(out):
    p=out/'protocol.json'
    if str(p) not in _deadlines:_deadlines[str(p)]=json.loads(p.read_text())['deadline_unix'] if p.exists() else float('inf')
    if time.time()>_deadlines[str(p)]:
        raise SystemExit('Campaign time limit reached; saved checkpoints can be resumed explicitly')

def loss_terms(pred,Y,A):
    base=((pred-Y).abs()*A).sum()/A.sum().clamp(min=1)
    yy=Y.reshape(Y.shape[0],-1);pp=pred.reshape(pred.shape[0],-1);aa=A.reshape(A.shape[0],-1)
    valid=aa.unfold(1,6,1).amin(-1)
    delta=((pp[:,5:]-pp[:,:-5])-(yy[:,5:]-yy[:,:-5])).abs()
    inc=(delta*valid).sum()/valid.sum().clamp(min=1)
    return base,inc

def setup(root):
    sys.path.insert(0,str(root/'scripts'))
    import torch,bench_common as C,bench_models as M
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert C.SPLIT_NAME=='comp' and C.SPLIT['val']==[2023]
    assert (root/'cluster/control/hold_station_training.json').exists()
    return torch,C,M

def prepare(a):
    torch,C,M=setup(a.root)
    # Verify target masking and units independently of the network.
    yy=torch.arange(20,dtype=torch.float32).reshape(1,4,5);aa=torch.zeros_like(yy);aa.reshape(-1)[5:15]=1
    pp=yy.clone();pp.reshape(-1)[:5]+=100
    b,d=loss_terms(pp,yy,aa);assert b.item()==0 and d.item()==0
    b,d=loss_terms(yy+3,yy,aa);assert b.item()==3 and d.item()==0
    b,d=loss_terms(yy*2,yy,aa);assert d.item()==5
    b,d=loss_terms(yy*2,yy,aa*0);assert b.item()==0 and d.item()==0
    write(a.out/'loss_tests.json',dict(masking=True,constant_offset=True,increment_units=True,empty_mask=True))
    from check_geomag_mechanisms import episodes
    F=C.load_year(2023,'ARS');rate,ok,threshold,lookup,events=episodes(F,.95,30)
    rng=np.random.default_rng(20261031);cases=[];selection=[]
    # Same input window for every placement of a given event and length.
    for L in [240,1440]:
        eligible=[]
        for eid,e in enumerate(events):
            peak=e['peak'];st=peak-C.W//2
            if st<0 or st+C.W>len(F):continue
            # All candidate gaps have known truth, so comparisons stay strictly paired.
            if not np.isfinite(F[peak-L:peak+L+6]).all():continue
            if np.mean(~np.isfinite(F[st:st+C.W]))>C.MAX_REAL:continue
            eligible.append(eid)
        # Uniform sample of distinct episodes, independent of model predictions.
        chosen=sorted(rng.choice(eligible,min(96,len(eligible)),replace=False).tolist())
        for eid in chosen:
            e=events[eid];peak=e['peak'];st=peak-C.W//2
            for side in ['left','right']:
                for distance in [5,15,30,60,120,L//2-5]:
                    gs=peak-distance if side=='left' else peak+5+distance-L
                    gp=gs-st;points=e['points']
                    left=bool(((points>=st)&(points+5<gs)).any())
                    right=bool(((points>=gs+L)&(points+5<st+C.W)).any())
                    assert gp>=C.CTX and gp+L<=C.W-C.CTX
                    assert gs<=peak and peak+5<gs+L
                    cases.append(dict(event=eid,peak=peak,start=st,gap=gp,length=L,side=side,distance=distance,
                                      left_visible=left,right_visible=right,block=peak//(14*1440)))
        selection.append(dict(length=L,eligible=len(eligible),selected=len(chosen)))
    write(a.out/'paired_cases.json',dict(cases=cases,selection=selection,threshold=threshold,
        design='Fixed six-day input window per episode; fixed gap length; move gap around the same canonical peak. Random distinct episodes, no model-based selection.',
        primary='Signed five-minute increment at the fixed canonical peak; timing and amplitude errors. Equal weight per event.',
        limitation='Gap placement also changes other missing values. This tests mask placement, not a pure causal effect of a binary visibility label.'))
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz') as z:
        ids=np.sort(np.random.default_rng(20261031).choice(len(z['L1440_win0']),256,replace=False))
    write(a.out/'validation_subset.json',dict(ids=ids.tolist(),seed=20261031,n=256))
    print('Prepared',len(cases),'paired cases',selection,flush=True)

def scalar_prediction(M,net,ck,source,F,rows):
    inps=[];starts=[]
    for c in rows:
        st,gp,L=c['start'],c['gap'],c['length'];inp=source[st:st+8640].copy();inp[gp:gp+L]=np.nan
        if inp.ndim==2:inp[~np.isfinite(F[st:st+8640])]=np.nan
        inps.append(inp);starts.append(st)
    ans=M.fill_many(net,inps,starts,ck['scale'],'cuda',bs=8)
    return [np.sqrt((x*x).sum(1)) if x.ndim==2 else x for x in ans]

def paired(a,checkpoint=None,label=None):
    torch,C,M=setup(a.root)
    import bench_run as R
    label=label or a.name
    dest=a.out/'paired'/label;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    cases=json.loads((a.out/'paired_cases.json').read_text())['cases'];F=C.load_year(2023,'ARS')
    if a.limit:cases=cases[:a.limit]
    if checkpoint is not None:
        net,ck=M.load(str(checkpoint),'cuda');source=C.load_year_comp(2023,'ARS') if M.NCH==3 else F
    elif label=='seadec':net=None;source=F
    else:
        net,ck=M.load(str(a.root/'models/comp/seeds'/f'{label}.pt'),'cuda');source=C.load_year_comp(2023,'ARS') if M.NCH==3 else F
    rows=[]
    for b in range(0,len(cases),8):
        deadline(a.out);batch=cases[b:b+8];part=dest/f'part_{b:05d}.json'
        if part.exists():rows.extend(json.loads(part.read_text()));continue
        if net is None:
            preds=[]
            for c in batch:
                x=F[c['start']:c['start']+C.W].copy();x[c['gap']:c['gap']+c['length']]=np.nan
                preds.append(R.METHODS['seadec'](x))
        else:preds=scalar_prediction(M,net,ck,source,F,batch)
        rr=[]
        for c,p in zip(batch,preds):
            st,gp,L=c['start'],c['gap'],c['length'];t=F[st+gp:st+gp+L].astype(float);pr=p[gp:gp+L]
            assert np.isfinite(pr).all()
            dr=(t[5:]-t[:-5])/5;dp=(pr[5:]-pr[:-5])/5;k=c['peak']-(st+gp)
            assert 0<=k<len(dp)
            row=dict(c,mae=float(abs(pr-t).mean()),d5_mae=float(abs(dp-dr).mean()),true_rate=float(dr[k]),
                     capture_pct=float(100*dp[k]/dr[k]),peak_rate_error=float(abs(dp[k]-dr[k])),
                     own_max_rate=float(abs(dp).max()),timing_error=float(abs(np.argmax(abs(dp))-k)))
            rr.append(row)
        write(part,rr);rows.extend(rr)
        if b%128==0:print('Paired',label,b,'/',len(cases),flush=True)
    write(dest/'complete.json',dict(complete=True,label=label,rows=rows,checkpoint=str(checkpoint)))

def evaluate(a,checkpoint,label):
    torch,C,M=setup(a.root);import analyze_geomag_events as E
    dest=a.out/'evaluation'/label;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    net,ck=M.load(str(checkpoint),'cuda');F=C.load_year(2023,'ARS');source=F
    ids=np.array(json.loads((a.out/'validation_subset.json').read_text())['ids']);rows=[]
    if a.limit:ids=ids[:a.limit]
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz') as d:
        for L in map(int,d['lengths']):
            deadline(a.out);metricfile=dest/f'L{L}.json'
            if metricfile.exists():rows.extend(json.loads(metricfile.read_text()));continue
            true=d[f'L{L}_true'][ids];margin=int(d['margin']);starts=d[f'L{L}_win0'][ids];gaps=d[f'L{L}_gap0'][ids]
            rr=[dict(start=int(st),gap=int(gp),length=L) for st,gp in zip(starts,gaps)];pred=[]
            for b in range(0,len(rr),8):
                deadline(a.out)
                ans=scalar_prediction(M,net,ck,source,F,rr[b:b+8])
                pred.extend(x[c['gap']:c['gap']+L] for x,c in zip(ans,rr[b:b+8]))
            pred=np.asarray(pred,np.float32);ref=E.reference(true,margin,L);met=E.features(pred,ref)
            save_npz(dest/f'L{L}.npz',pred=pred,ids=ids,**met)
            row=dict(length=L,n=len(ids),mae=float(met['mae'].mean()))
            for key in ['d1_mae','d5_mae','peak_value_error','d5_timing']:
                if key in met:row[key]=float(np.nanmean(met[key]))
            if 'd5_capture' in met:row['d5_capture_pct']=float(100*met['d5_capture'].mean()/ref['d5_max'].mean())
            write(metricfile,[row]);rows.append(row);print('Evaluated',label,L,row['mae'],flush=True)
    write(dest/'complete.json',dict(complete=True,label=label,rows=rows,checkpoint=str(checkpoint),subset='256 fixed original validation windows per length'))

def train(a):
    torch,C,M=setup(a.root);import bench_train as T
    dest=a.out/'training'/a.name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    initial=a.root/'models/comp/seeds'/f'{a.arch}_best_s{a.seed}.pt'
    cfg=json.loads(initial.with_suffix('.json').read_text())
    cfg.update(steps=a.steps,lr=1e-5 if a.arch=='moment' else 1e-4)
    write(dest/'config.json',cfg)
    recipe=dict(initial=str(initial),arch=a.arch,seed=a.seed,steps=a.steps,increment_loss_weight=a.weight,
        objective='masked MAE + weight * masked five-minute increment MAE; both in training normalized units; all six minutes must be artificial-gap targets',
        primary_checkpoint='fixed final step',secondary_checkpoint='best validation MAE',precision='FP32; TF32 disabled',
        train_years=C.SPLIT['train'],validation_years=C.SPLIT['val'],station='ARS',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    recipepath=dest/'recipe.json'
    if recipepath.exists():assert json.loads(recipepath.read_text())==recipe,'Cannot change resumed experiment'
    else:write(recipepath,recipe)
    best=dest/'best.pt';final=dest/'final.pt'
    if not final.exists():
        initial_ck=torch.load(initial,map_location='cpu',weights_only=False);original_build=M.build;created=[]
        def factory(name,**kw):
            if initial_ck.get('extra') is not None:kw['_extra']=initial_ck['extra']
            net=original_build(name,**kw);net.load_state_dict(initial_ck['state']);created.append(net)
            def objective(self,X,mask,Y,A):
                deadline(a.out)
                pred=self(X,mask);base,inc=loss_terms(pred,Y,A)
                if a.weight==0:return base
                return base+a.weight*inc
            net.custom_loss=types.MethodType(objective,net)
            return net
        M.build=factory
        sys.argv=['bench_train','--arch',a.arch,'--cfg',str(dest/'config.json'),'--out',str(best),'--seed',str(a.seed),'--steps',str(a.steps),'--every','500']
        if a.steps<500:sys.argv[-1]=str(a.steps)
        try:T.main()
        finally:M.build=original_build
        net=created[-1]
        vals=T.build_valset(C.load_split('val',code='ARS'),initial_ck['scale'],n_per_L=getattr(M.ARCH[a.arch],'VAL_PER_L',T.VAL_PER_L))
        final_val=T.validate(net,vals,initial_ck['scale'],'cuda')
        M.save(str(final),net,a.arch,initial_ck['scale'],initial_ck['kw'],a.steps,final_val)
        write(dest/'final_score.json',dict(step=a.steps,val_mae=final_val))
        del vals
        del created,net,initial_ck;torch.cuda.empty_cache()
    evaluate(a,final,a.name)
    if not a.smoke:paired(a,final,a.name)
    write(dest/'complete.json',dict(complete=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),recipe=recipe))

def csdi(a):
    torch,C,M=setup(a.root)
    name=f'csdi_best_s{a.seed}';dest=a.out/'csdi'/f'{a.mode}_s{a.seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    net,ck=M.load(str(a.root/'models/comp/seeds'/f'{name}.pt'),'cuda');F=C.load_year(2023,'ARS');summary=[]
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz') as z:
        for L in ([1440] if a.mode=='phase' else [240,1440,4320]):
            ids=([52]+np.random.default_rng(20261031).choice(1024,15,False).tolist()) if a.mode=='phase' else np.random.default_rng(20261031).choice(1024,64,False).tolist()
            if a.limit:ids=ids[:a.limit]
            starts=z[f'L{L}_win0'];gaps=z[f'L{L}_gap0'];true=z[f'L{L}_true'][:,int(z['margin']):int(z['margin'])+L]
            for shift in (range(5) if a.mode=='phase' else [0]):
                for b in range(0,len(ids),8):
                    deadline(a.out);chunk=ids[b:b+8];file=dest/f'L{L}_shift{shift}_batch{b}.npz';mf=file.with_suffix('.json')
                    if mf.exists():summary.extend(json.loads(mf.read_text()));continue
                    cases=[dict(start=int(starts[i])+shift,gap=int(gaps[i])-shift,length=L) for i in chunk]
                    assert all(c['gap']>=0 and c['start']+C.W<=len(F) for c in cases)
                    allsamples=[];ns=8 if a.mode=='phase' else 32
                    if a.smoke:ns=2
                    for s in range(ns):
                        deadline(a.out);torch.manual_seed(8001+s);torch.cuda.manual_seed_all(8001+s)
                        ans=scalar_prediction(M,net,ck,F,F,cases)
                        allsamples.append([v[c['gap']:c['gap']+L] for c,v in zip(cases,ans)])
                    pred=np.asarray(allsamples,np.float32);assert np.isfinite(pred).all()
                    save_npz(file,pred=pred,ids=chunk,sampling_seeds=np.arange(8001,8001+ns),shift=shift)
                    rr=[]
                    for j,i in enumerate(chunk):
                        t=true[i].astype(float);samples=pred[:,j].astype(float);dr=(t[5:]-t[:-5])/5;dp=(samples[:,5:]-samples[:,:-5])/5;k=int(np.argmax(abs(dr)))
                        lo,hi=np.quantile(samples,[.05,.95],axis=0);med=np.median(samples,axis=0);sd=np.sort(samples,axis=0)
                        # Empirical CRPS = E|X-y| - .5 E|X-X'|, exact ensemble formula.
                        coef=(2*np.arange(1,ns+1)-ns-1)[:,None]
                        crps=abs(samples-t).mean(0)-(coef*sd).sum(0)/(ns*ns)
                        maxloc=np.argmax(abs(dp),axis=1);maxrate=np.max(abs(dp),axis=1)
                        rr.append(dict(length=L,window=int(i),shift=shift,peak_phase=int((cases[j]['gap']+k)%5),
                            true_rate=float(dr[k]),capture_pct=(100*dp[:,k]/dr[k]).tolist(),max_rate=maxrate.tolist(),
                            max_rate_phase=((cases[j]['gap']+maxloc)%5).tolist(),distance_to_boundary=np.minimum(maxloc,L-6-maxloc).tolist(),
                            sample_mae=abs(samples-t).mean(1).tolist(),median_mae=float(abs(med-t).mean()),
                            coverage90=float(((t>=lo)&(t<=hi)).mean()),width90=float((hi-lo).mean()),crps=float(crps.mean()),
                            peak_event_probability=float(((abs(maxloc-k)<=30)&(maxrate>=.5*abs(dr[k]))&(np.sign(dp[np.arange(ns),maxloc])==np.sign(dr[k]))).mean())))
                    write(mf,rr);summary.extend(rr);print('CSDI',a.mode,a.seed,L,shift,b,flush=True)
    write(dest/'complete.json',dict(complete=True,rows=summary,mode=a.mode,training_seed=a.seed,
        limits=['Phase shifts change at most four minutes at each context edge; same absolute artificial gap and truth',
                'Repeated sampling is diagnostic; primary benchmark remains single-sample',
                'Random controls estimate behaviour on this fixed validation subset only']))

def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=['prepare','paired','train','csdi'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--name',default='seadec');p.add_argument('--arch',default='unet');p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--weight',type=float,default=0);p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--mode',choices=['phase','uncertainty'],default='phase');p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);deadline(a.out)
    globals()[a.job](a)
if __name__=='__main__':main()
