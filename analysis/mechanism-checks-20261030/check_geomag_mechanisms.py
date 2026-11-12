"""Validation-only representation and event-visibility controls."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np

def write(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def controls(root,out,C,M,E):
    import torch
    F=C.load_year(2023,'ARS');XYZ=C.load_year_comp(2023,'ARS')
    refpath=root/'data/comp/dump_seadec_ARS_comp_val.npz'
    class Oracle(torch.nn.Module):
        def __init__(self,target,revin):super().__init__();self.target=torch.from_numpy(target);self.revin=revin
        def forward(self,X,mask):
            y=self.target
            if not self.revin:return y
            B=X.shape[0];x=X[:,:,:M.POUT].reshape(B,C.W,M.NCH).permute(0,2,1).reshape(B*M.NCH,C.W)
            m=mask[:,:,:M.POUT].reshape(B,C.W,M.NCH).permute(0,2,1).reshape(B*M.NCH,C.W)
            cnt=m.sum(1,keepdim=True).clamp(min=1);mu=(x*m).sum(1,keepdim=True)/cnt
            sd=torch.sqrt((((x-mu)*m)**2).sum(1,keepdim=True)/cnt)+1e-5
            yt=y.reshape(B,C.W,M.NCH).permute(0,2,1).reshape(B*M.NCH,C.W)
            yt=((yt-mu)/sd).reshape(B*M.NCH,-1,8).reshape(B*M.NCH,C.W)*sd+mu
            return yt.reshape(B,M.NCH,C.W).permute(0,2,1).reshape(B,M.WB,M.POUT)
    rows=[]
    with np.load(refpath,allow_pickle=False) as d:
        margin=int(d['margin'])
        for L in map(int,d['lengths']):
            starts=d[f'L{L}_win0'];gaps=d[f'L{L}_gap0'];true=d[f'L{L}_true']
            er=E.reference(true,margin,L);sel=er['gap_act']>np.quantile(er['gap_act'],.75)
            for nch,source,label in [(1,F,'F'),(3,XYZ,'XYZ')]:
                cfg=json.loads((root/'models/comp/seeds'/('csdi_best_xyz_s1234.json' if nch==3 else 'csdi_best_s1234.json')).read_text())
                scale=np.array(cfg['scale'],np.float32) if nch==3 else float(cfg['scale'][0])
                M.set_patch(5);M.set_channels(nch)
                for kind,revin in [('common_pack5',False),('moment_pack8_revin',True)]:
                    collected=[];channel_max=0.;norm_offset=[]
                    for b in range(0,len(starts),16):
                        inps=[];targets=[];originals=[]
                        for st,gp in zip(starts[b:b+16],gaps[b:b+16]):
                            st=int(st);gp=int(gp);truth=source[st:st+C.W].copy();inp=truth.copy()
                            inp[gp:gp+L]=np.nan
                            if nch==3:inp[~np.isfinite(F[st:st+C.W])]=np.nan
                            _,_,center=M.featurize(inp,st,scale)
                            target=np.nan_to_num((truth-center)/scale,nan=0).astype(np.float32).reshape(M.WB,M.POUT)
                            inps.append(inp);targets.append(target);originals.append(truth)
                        answer=M.fill_many(Oracle(np.stack(targets),revin),inps,list(map(int,starts[b:b+16])),scale,'cpu',bs=16)
                        for j,(pr,truth) in enumerate(zip(answer,originals)):
                            gp=int(gaps[b+j]);pr=pr.astype(np.float32);t=truth[gp:gp+L];p=pr[gp:gp+L]
                            assert np.isfinite(t).all() and np.isfinite(p).all()
                            channel_max=max(channel_max,float(abs(p.astype(float)-t).max()))
                            if nch==3:
                                mag=np.sqrt((p.astype(float)**2).sum(1));tm=np.sqrt((t.astype(float)**2).sum(1))
                                norm_offset.append(float(abs(tm-er['t'][b+j]).mean()))
                            else:mag=p
                            collected.append(mag.astype(np.float32))
                    pred=np.array(collected).astype(float);f=E.features(pred,er)
                    for sub,mask in [('all',np.ones(len(starts),bool)),('gap_high25',sel)]:
                        row=dict(input=label,control=kind,length=L,subset=sub,n=int(mask.sum()),
                            channel_max_error=channel_max,mae=float(f['mae'][mask].mean()),
                            xyz_norm_vs_scalar_mae=float(np.mean(norm_offset)) if norm_offset else 0.)
                        for k in ['d1_mae','d5_mae','d15_mae','peak_amp_error','peak_value_error']:
                            if k in f:row[k]=float(np.nanmean(f[k][mask]))
                        if 'd5_capture' in f:row['d5_capture_pct']=float(100*f['d5_capture'][mask].mean()/er['d5_max'][mask].mean())
                        if 'peak_capture' in f:row['peak_capture_pct']=float(100*np.nanmean(f['peak_capture'][mask])/np.nanmean(er['peak_amp'][mask]))
                        rows.append(row)
            write(out/'representation.json',dict(rows=rows,complete=False))
            print('Representation checked L='+str(L),flush=True)
    result=dict(complete=True,rows=rows,scope='Truth oracle bypasses learned layers; not an imputation baseline',
        findings=['Common patching preserves five separate minute values, no temporal averaging',
            'MOMENT reshapes those values into eight-minute vectors and returns eight outputs per patch',
            'Oracle checks normalization, layout, output decoding and storage; it does not prove that learned layers preserve events'])
    write(out/'representation.json',result)

def episodes(F,quantile,merge):
    rate=(F[5:].astype(float)-F[:-5].astype(float))/5
    # No derivative across a real missing point, including the three interior minutes.
    ok=np.convolve(np.isfinite(F).astype(int),np.ones(6,int),mode='valid')==6
    active=np.flatnonzero(ok&(abs(rate)>np.quantile(abs(rate[ok]),quantile)))
    threshold=float(np.quantile(abs(rate[ok]),quantile))
    cuts=np.r_[0,np.flatnonzero(np.diff(active)>merge)+1,len(active)]
    lookup=np.full(len(rate),-1,np.int32);entries=[]
    for i,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        points=active[a:b];peak=int(points[np.argmax(abs(rate[points]))])
        lookup[int(points[0]):int(points[-1])+1]=i
        entries.append(dict(start=int(points[0]),end=int(points[-1]+5),peak=peak,points=points))
    return rate,ok,threshold,lookup,entries

def event_comparison(net,base,mask,event_ids,event_blocks,reps=2000):
    ids=np.unique(event_ids[mask]);count=len(ids)
    if not count:return None
    n=np.stack([net[:,mask&(event_ids==i)].mean(1) for i in ids],axis=1)
    b=np.array([base[mask&(event_ids==i)].mean() for i in ids])
    nm=float(n.mean());bm=float(b.mean());row=dict(events=count,windows=int(mask.sum()),net=nm,baseline=bm,
        gain_pct=100*(1-nm/bm) if bm>1e-12 else None)
    blocks=np.array([event_blocks[int(i)] for i in ids]);levels,inv=np.unique(blocks,return_inverse=True)
    row['blocks']=len(levels)
    if count<5 or len(levels)<4:return row
    rng=np.random.default_rng(20261030);w=rng.multinomial(len(levels),np.ones(len(levels))/len(levels),size=reps)
    sw=rng.multinomial(5,np.ones(5)/5,size=reps)/5
    counts=np.bincount(inv,minlength=len(levels));den=w@counts
    sums=np.array([np.bincount(inv,weights=x,minlength=len(levels)) for x in n])
    bs=np.bincount(inv,weights=b,minlength=len(levels))
    delta=((w@sums.T)/den[:,None]*sw).sum(1)-(w@bs)/den
    row['difference_ci95']=np.quantile(delta,[.025,.975]).tolist()
    return row

def visibility(root,out,C):
    F=C.load_year(2023,'ARS')
    with np.load(root/'cluster/analysis/event-diagnostics-20261030/event_windows.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    focus=['unet_F','moment_F','saits_F','timesnet_F','csdi_F','unet_XYZ','moment_XYZ']
    labels={};rows=[];counts=[];matched=[];protocols=[]
    with np.load(root/'data/comp/dump_seadec_ARS_comp_val.npz',allow_pickle=False) as d:
        margin=int(d['margin'])
        for q,merge in [(.95,30),(.95,15),(.95,60),(.99,30)]:
            tag=f'q{int(q*100)}_merge{merge}';rate,ok,threshold,lookup,events=episodes(F,q,merge)
            eb={i:e['peak']//(14*1440) for i,e in enumerate(events)}
            protocols.append(dict(tag=tag,threshold_nT_per_min=threshold,episodes=len(events)))
            for L in map(int,d['lengths']):
                if L<60:continue
                truth=d[f'L{L}_true'][:,margin:margin+L].astype(float)
                starts=d[f'L{L}_win0'].astype(int);gps=d[f'L{L}_gap0'].astype(int);gs=starts+gps;ge=gs+L
                rel=np.argmax(abs((truth[:,5:]-truth[:,:-5])/5),axis=1);absolute=gs+rel
                ids=lookup[absolute];category=np.full(len(starts),'no_strong_event',dtype='<U24');canonical=np.zeros(len(starts),bool)
                for j,eid in enumerate(ids):
                    if eid<0 or not ok[absolute[j]] or abs(rate[absolute[j]])<=threshold:continue
                    e=events[int(eid)];points=e['points']
                    left=bool(((points>=starts[j])&(points+5<gs[j])).any())
                    right=bool(((points>=ge[j])&(points+5<starts[j]+C.W)).any())
                    category[j]='both_visible' if left and right else 'left_visible' if left else 'right_visible' if right else 'fully_hidden'
                    canonical[j]=gs[j]<=e['peak'] and e['peak']+5<ge[j]
                labels[f'{tag}_category_{L}']=category;labels[f'{tag}_event_{L}']=ids;labels[f'{tag}_canonical_{L}']=canonical
                valid=canonical&(ids>=0)&(category!='no_strong_event')
                hi=data[f'true_gap_act_{L}']>np.quantile(data[f'true_gap_act_{L}'],.75)
                groups=dict(fully_hidden=valid&(category=='fully_hidden'),partly_visible=valid&(category!='fully_hidden'))
                for sub,mask in groups.items():
                    counts.append(dict(protocol=tag,length=L,visibility=sub,windows=int(mask.sum()),events=len(np.unique(ids[mask]))))
                    for g in focus:
                        for metric in ['mae','peak_value_error','d5_mae','d5_max_error','d5_timing']:
                            net=data[f'{g}_{metric}_{L}'];base=data[f'base_{metric}_{L}']
                            sel=mask&np.isfinite(base)&np.isfinite(net).all(0)
                            result=event_comparison(net,base,sel,ids,eb)
                            if result:rows.append(dict(protocol=tag,length=L,visibility=sub,group=g,metric=metric,**result))
                        # Equal weight per distinct event, also for captured fraction.
                        strength=data[f'true_d5_max_{L}'];capture=data[f'{g}_d5_capture_{L}']/strength[None,:]*100
                        basecap=data[f'base_d5_capture_{L}']/strength*100
                        result=event_comparison(capture,basecap,mask,ids,eb)
                        if result:rows.append(dict(protocol=tag,length=L,visibility=sub,group=g,metric='d5_capture_pct',**result))
                # Same event and same length in both categories: exploratory paired contrast.
                common=np.intersect1d(ids[groups['fully_hidden']],ids[groups['partly_visible']])
                for g in ['unet_F','moment_F']:
                    capture=data[f'{g}_d5_capture_{L}']/data[f'true_d5_max_{L}'][None,:]*100
                    differences=[]
                    for eid in common:
                        h=capture[:,groups['fully_hidden']&(ids==eid)].mean(1)
                        v=capture[:,groups['partly_visible']&(ids==eid)].mean(1)
                        differences.append(v-h)
                    matched.append(dict(protocol=tag,length=L,group=g,events=len(common),
                        visible_minus_hidden_capture_pp=float(np.mean(differences)) if differences else None))
            write(out/'visibility.json',dict(complete=False,protocols=protocols,counts=counts,rows=rows,matched=matched))
            print('Visibility checked '+tag,flush=True)
    np.savez_compressed(out/'visibility_labels.npz',**labels)
    write(out/'visibility.json',dict(complete=True,protocols=protocols,counts=counts,rows=rows,matched=matched,
        definition='5-minute-rate episodes above annual validation q95; active samples separated by <=30 min merged; q95 merge15/60 and q99 merge30 sensitivity',
        primary_eligibility='Strongest gap change exceeds threshold and canonical episode peak is fully within the gap',
        weighting='Equal weight per distinct episode within each length/category; five seeds; 14-day-block bootstrap',
        limitations=['Retrospective local-variability episodes, not a storm catalogue',
            'Visible means a threshold-exceeding rate in the same connected episode is actually observed within the six-day input',
            'Episode visibility is not random; severity and duration can confound category comparisons',
            'Same-event contrasts may have very small support; not causal evidence',
            'CIs are pointwise exploratory; no cross-station or test reads']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.root/'scripts'));sys.path.insert(0,str(a.out))
    import bench_common as C,bench_models as M,analyze_geomag_events as E
    import torch
    torch.set_num_threads(2)
    assert C.SPLIT['val']==[2023]
    hold=a.root/'cluster/control/hold_station_training.json';assert hold.exists()
    write(a.out/'cpu_recipe.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),validation='ARS 2023 only',
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hold_sha256=hashlib.sha256(hold.read_bytes()).hexdigest()))
    if not (a.out/'representation.json').exists() or not json.loads((a.out/'representation.json').read_text()).get('complete'):controls(a.root,a.out,C,M,E)
    if not (a.out/'visibility.json').exists() or not json.loads((a.out/'visibility.json').read_text()).get('complete'):visibility(a.root,a.out,C)
    assert hold.exists();write(a.out/'cpu_complete.json',dict(complete=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('CPU CHECKS COMPLETE',flush=True)

if __name__=='__main__':main()
