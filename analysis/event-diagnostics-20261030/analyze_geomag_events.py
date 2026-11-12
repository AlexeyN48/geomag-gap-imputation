"""Exploratory, validation-only event diagnostics of already saved predictions.

No training, no test reads, no changes to the main experimental pipeline.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ARCHS='dlinear nbeatsx nhits tide tsmixerx segrnn unet timesnet saits imputeformer csdi moment'.split()
SEEDS=[1234,2345,3456,4567,5678]
FOCUS=['unet_F','moment_F','saits_F','timesnet_F','csdi_F','unet_XYZ','moment_XYZ']
ERRORS=['mae','range_error','peak_amp_error','peak_value_error','peak_timing',
        'd1_mae','d5_mae','d15_mae','d5_max_error','d5_timing']
METRICS=ERRORS+['peak_capture','peak_pred_amp','peak_hit25_30','peak_hit50_60',
                'd5_mse','d5_pred_energy','d5_cross','d5_center_cross','d5_center_pred_energy',
                'd5_capture','d5_pred_max','d5_hit50_30','d5_hit50_60']
PRIMARY=['mae','peak_amp_error','peak_value_error','peak_timing','d5_mae','d5_max_error','d5_timing']
SUBSETS=['all','gap_high25','gap_low25','gap_top10','large_peak25']

def smooth(x,width=15):
    s=np.concatenate([np.zeros((len(x),1)),np.cumsum(x,axis=1)],axis=1)
    return (s[:,width:]-s[:,:-width])/width

def reference(full,margin,L):
    t=full[:,margin:margin+L].astype(float)
    assert np.isfinite(t).all()
    left=full[:,margin-60:margin].astype(float);right=full[:,margin+L:margin+L+60].astype(float)
    eligible=(np.isfinite(left).sum(1)>=30)&(np.isfinite(right).sum(1)>=30)
    lm=np.nanmedian(left,axis=1);rm=np.nanmedian(right,axis=1)
    line=lm[:,None]+(rm-lm)[:,None]*(np.arange(L)[None,:]+30.5)/(L+60)
    r=dict(t=t,line=line,eligible=eligible,gap_act=abs(np.diff(t,axis=1)).mean(1),
           range=np.ptp(t,axis=1))
    if L>=60:
        residual=smooth(t-line)
        # Exclude unsupported context only for peak diagnostics.
        safe=np.where(np.isfinite(residual),residual,0)
        idx=np.argmax(abs(safe),axis=1)
        peak=safe[np.arange(len(t)),idx]
        r.update(peak_index=idx,peak_value=peak,peak_amp=np.where(eligible,abs(peak),np.nan))
    for lag in [1,5,15]:
        if L>lag:r[f'd{lag}']=(t[:,lag:]-t[:,:-lag])/lag
    if L>5:
        d=r['d5'];idx=np.argmax(abs(d),axis=1);peak=d[np.arange(len(t)),idx]
        r.update(d5_index=idx,d5_value=peak,d5_max=abs(peak),d5_energy=(d*d).mean(1),
                 d5_center_energy=((d-d.mean(1,keepdims=True))**2).mean(1))
    return r

def features(pred,r):
    t=r['t'];n,L=t.shape;ix=np.arange(n)
    f=dict(mae=abs(pred-t).mean(1),range_error=abs(np.ptp(pred,axis=1)-r['range']))
    if L>=60:
        pr=smooth(pred-r['line']);safe=np.where(np.isfinite(pr),pr,0)
        pi=np.argmax(abs(safe),axis=1);pv=safe[ix,pi]
        at=safe[ix,r['peak_index']];amp=abs(pv);ta=r['peak_amp'];dt=abs(pi-r['peak_index'])
        hit=(np.sign(pv)==np.sign(r['peak_value']))&(abs(amp-ta)<=.25*ta)&(dt<=30)
        hitwide=(np.sign(pv)==np.sign(r['peak_value']))&(abs(amp-ta)<=.5*ta)&(dt<=60)
        pf=dict(peak_amp_error=abs(amp-ta),peak_value_error=abs(at-r['peak_value']),peak_timing=dt.astype(float),
                peak_capture=np.sign(r['peak_value'])*at,peak_pred_amp=amp,
                peak_hit25_30=hit.astype(float),peak_hit50_60=hitwide.astype(float))
        f.update({k:np.where(r['eligible'],v,np.nan) for k,v in pf.items()})
    for lag in [1,5,15]:
        if L<=lag:continue
        pd=(pred[:,lag:]-pred[:,:-lag])/lag;td=r[f'd{lag}']
        f[f'd{lag}_mae']=abs(pd-td).mean(1)
        if lag==5:
            pi=np.argmax(abs(pd),axis=1);pv=pd[ix,pi];at=pd[ix,r['d5_index']]
            amp=abs(pv);dt=abs(pi-r['d5_index']);ta=r['d5_max']
            pc=pd-pd.mean(1,keepdims=True);tc=td-td.mean(1,keepdims=True)
            hit=(np.sign(pv)==np.sign(r['d5_value']))&(abs(amp-ta)<=.5*ta)&(dt<=30)
            hitwide=(np.sign(pv)==np.sign(r['d5_value']))&(abs(amp-ta)<=.5*ta)&(dt<=60)
            f.update(d5_mse=((pd-td)**2).mean(1),d5_pred_energy=(pd*pd).mean(1),d5_cross=(pd*td).mean(1),
                     d5_center_cross=(pc*tc).mean(1),d5_center_pred_energy=(pc*pc).mean(1),
                     d5_max_error=abs(amp-ta),d5_capture=np.sign(r['d5_value'])*at,d5_pred_max=amp,
                     d5_timing=dt.astype(float),d5_hit50_30=hit.astype(float),d5_hit50_60=hitwide.astype(float))
    return f

def collect(root,out):
    cache=out/'event_windows.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        return data,json.loads((out/'sources.json').read_text())
    baseline=root/'data/comp/dump_seadec_ARS_comp_val.npz'
    ref={};rr={};data={};sources=[]
    def record(p):
        st=p.stat();sources.append(dict(path=str(p.relative_to(root)),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    record(baseline)
    with np.load(baseline,allow_pickle=False) as d:
        assert str(d['setname'])=='ARS_comp_val'
        lengths=[int(x) for x in d['lengths']];margin=int(d['margin'])
        for L in lengths:
            ref[L]={k:d[f'L{L}_{k}'] for k in ['true','win0','gap0','block','act']}
            r=reference(ref[L]['true'],margin,L);rr[L]=r
            for k in ['gap_act','range','peak_amp','d5_max','d5_energy','d5_center_energy','eligible']:
                if k in r:data[f'true_{k}_{L}']=r[k]
            for k in ['block','win0','gap0','act']:data[f'{k}_{L}']=ref[L][k]
            pred=d[f'L{L}_pred'][:,margin:margin+L].astype(float)
            for k,v in features(pred,r).items():data[f'base_{k}_{L}']=v.astype(np.float32)
    data['lengths']=np.array(lengths);data['seeds']=np.array(SEEDS)
    groups=[];incomplete=[]
    for a in ARCHS:
        for mode,suffix in [('F',''),('XYZ','_xyz')]:
            paths=[root/f'data/comp/seeds/dump_{a}_best{suffix}_s{s}_ARS_comp_val.npz' for s in SEEDS]
            if not all(p.exists() for p in paths):
                incomplete.append(dict(arch=a,input=mode,available=sum(p.exists() for p in paths)));continue
            group=f'{a}_{mode}';groups.append(group)
            for path in paths:
                record(path)
                with np.load(path,allow_pickle=False) as d:
                    assert str(d['setname'])=='ARS_comp_val' and int(d['margin'])==margin
                    assert list(d['lengths'])==lengths
                    for L in lengths:
                        for k,v in ref[L].items():assert np.array_equal(d[f'L{L}_{k}'],v,equal_nan=True),(path.name,L,k)
                        pred=d[f'L{L}_pred'][:,margin:margin+L].astype(float)
                        assert np.isfinite(pred).all()
                        f=features(pred,rr[L])
                        for k,v in f.items():data.setdefault(f'{group}_{k}_{L}',[]).append(v.astype(np.float32))
        print('Collected event metrics: '+a,flush=True)
    data['groups']=np.array(groups)
    data={k:np.array(v) for k,v in data.items()}
    meta=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),
              validation='ARS 2023; existing predictions, 1024 matched windows per length',sources=sources,
              incomplete=incomplete,station_training_hold=(root/'cluster/control/hold_station_training.json').exists())
    np.savez_compressed(cache,**data);(out/'sources.json').write_text(json.dumps(meta,indent=2))
    return data,meta

def design(blocks,reps=2000,span=2):
    levels,inv=np.unique((blocks%100)//span,return_inverse=True)
    rng=np.random.default_rng(20261031+span)
    w=rng.multinomial(len(levels),np.ones(len(levels))/len(levels),size=reps)
    sw=rng.multinomial(5,np.ones(5)/5,size=reps)/5
    return inv,levels,w,sw

def paired(net,base,selection,des):
    inv,levels,w,sw=des
    sel=selection&np.isfinite(base)&np.isfinite(net).all(0)
    count=np.bincount(inv[sel],minlength=len(levels))
    den=w@count;valid=den>0
    def sums(x):return np.array([np.bincount(inv[sel],weights=v[sel],minlength=len(levels)) for v in np.atleast_2d(x)])
    nb=((w@sums(net).T)[valid]/den[valid,None]*sw[valid]).sum(1)
    bb=(w@sums(base)[0])[valid]/den[valid]
    nm=float(net[:,sel].mean());bm=float(base[sel].mean());diff=nb-bb
    sd=float(np.std(net[:,sel].mean(1),ddof=1))
    gain=100*(1-nm/bm) if bm>1e-12 else None
    row=dict(n=int(sel.sum()),blocks=int((count>0).sum()),net=nm,baseline=bm,
             difference=nm-bm,ci95=np.quantile(diff,[.025,.975]).tolist(),gain_pct=gain,
             seed_sd=sd,seeds_better=int((net[:,sel].mean(1)<bm).sum()))
    return row,diff

def analyze(data,meta,reps):
    rows=[];diag=[];sens=[];family=[];thresholds=[]
    lengths=[int(x) for x in data['lengths']];groups=list(data['groups'])
    for L in lengths:
        a=data[f'true_gap_act_{L}'];q25,q75,q90=np.quantile(a,[.25,.75,.9])
        masks=dict(all=np.ones(len(a),bool),gap_high25=a>q75,gap_low25=a<=q25,gap_top10=a>q90)
        if L>=60:
            peak=data[f'true_peak_amp_{L}'];pq=np.nanquantile(peak,.75);masks['large_peak25']=peak>pq
        else:pq=None
        thresholds.append(dict(length=L,activity_q25=float(q25),activity_q75=float(q75),activity_q90=float(q90),
                               peak_q75=float(pq) if pq is not None else None,peak_context_eligible=int(data[f'true_eligible_{L}'].sum())))
        des=design(data[f'block_{L}'],reps,2);des28=design(data[f'block_{L}'],reps,4)
        for g in groups:
            for sub,sel in masks.items():
                for k in ERRORS:
                    if f'{g}_{k}_{L}' not in data:continue
                    row,boot=paired(data[f'{g}_{k}_{L}'],data[f'base_{k}_{L}'],sel,des)
                    row.update(group=g,length=L,subset=sub,metric=k);rows.append(row)
                    if L>=60 and sub in ['all','gap_high25'] and k in PRIMARY:
                        assert len(boot)==reps
                        family.append((len(rows)-1,boot))
                    if g in ['unet_F','moment_F'] and L in [480,1440,4320] and sub=='gap_high25' and k in PRIMARY:
                        sr,_=paired(data[f'{g}_{k}_{L}'],data[f'base_{k}_{L}'],sel,des28)
                        sr.update(group=g,length=L,subset=sub,metric=k);sens.append(sr)
                if L<=5:continue
                # Ratio of summed strengths, not average of unstable per-window ratios.
                for name in [g,'base']:
                    def arr(k):return np.atleast_2d(data[f'{name}_{k}_{L}'])[:,sel]
                    def mean(k):return arr(k).mean(1)
                    te=data[f'true_d5_energy_{L}'][sel].mean()
                    tce=data[f'true_d5_center_energy_{L}'][sel].mean()
                    pce=mean('d5_center_pred_energy')
                    corr=np.divide(mean('d5_center_cross'),np.sqrt(pce*tce),out=np.full_like(pce,np.nan),where=pce>1e-20)
                    row=dict(group=g,method=name,subset=sub,length=L,
                        d5_skill_zero=float(1-mean('d5_mse').mean()/te),
                        d5_correlation=float(np.nanmean(corr)) if np.isfinite(corr).any() else None,
                        d5_peak_capture_pct=float(100*mean('d5_capture').mean()/data[f'true_d5_max_{L}'][sel].mean()),
                        d5_peak_amplitude_pct=float(100*mean('d5_pred_max').mean()/data[f'true_d5_max_{L}'][sel].mean()),
                        d5_hit50_30_pct=float(100*mean('d5_hit50_30').mean()),
                        d5_hit50_60_pct=float(100*mean('d5_hit50_60').mean()))
                    strength=data[f'true_d5_max_{L}'][sel]
                    ratios=100*arr('d5_capture')/strength[None,:]
                    keep=strength<=np.quantile(strength,.9)
                    row.update(d5_capture_median_pct=float(np.median(ratios)),
                        d5_capture_iqr_pct=np.quantile(ratios,[.25,.75]).tolist(),
                        d5_capture_per_seed_pct=(100*mean('d5_capture')/strength.mean()).tolist(),
                        d5_capture_excluding_strongest10_pct=float(100*arr('d5_capture')[:,keep].mean()/strength[keep].mean()),
                        true_d5_max_mean=float(strength.mean()))
                    if L>=60:
                        amp=np.nanmean(data[f'true_peak_amp_{L}'][sel])
                        row.update(peak_capture_pct=float(100*np.nanmean(arr('peak_capture'))/amp),
                            peak_amplitude_pct=float(100*np.nanmean(arr('peak_pred_amp'))/amp),
                            peak_hit25_30_pct=float(100*np.nanmean(arr('peak_hit25_30'))),
                            peak_hit50_60_pct=float(100*np.nanmean(arr('peak_hit50_60'))))
                    diag.append(row)
        print('Analyzed event metrics: '+str(L),flush=True)
    boots=np.array([b for _,b in family]).T;centers=np.array([rows[i]['difference'] for i,_ in family])
    sd=boots.std(0,ddof=1);valid=sd>1e-12
    critical=float(np.quantile(np.max(abs((boots[:,valid]-centers[valid])/sd[valid]),axis=1),.95))
    for j,(i,_) in enumerate(family):
        lo=centers[j]-critical*sd[j];hi=centers[j]+critical*sd[j]
        rows[i].update(simultaneous_ci95=[float(lo),float(hi)],supported_win=bool(hi<0),supported_loss=bool(lo>0))
    return dict(meta=meta,thresholds=thresholds,rows=rows,diagnostics=diag,sensitivity28=sens,
                bootstrap=dict(reps=reps,block_days=14,seed=20261033,seed_resampling=True,
                    family_size=len(family),critical=critical,scope='All groups; lengths>=60; all/high25; seven primary error metrics'),
                definitions=dict(peak='Largest absolute 15-minute-smoothed departure from a line joining medians of 60 visible minutes on either side; smoothing fully inside gap',
                    peak_amplitude_error='Absolute difference of maximum absolute departures, regardless of timing',
                    peak_value_error='Absolute error at true strongest departure time',
                    peak_timing='Absolute time difference of strongest absolute departures; sign mistakes are assessed separately by joint hit',
                    peak_capture='100 * sum(sign(true_peak)*prediction_at_true_peak) / sum(abs(true_peak)); no prediction ensemble',
                    derivative='(F[t+lag]-F[t])/lag for lag 1,5,15; both points strictly within the hidden gap',
                    d5_capture='Signed predicted 5-minute rate at the strongest true rate, divided by aggregate true strength',
                    d5_correlation='Pooled within-window centered 5-minute-rate correlation, averaged over training seeds',
                    d5_skill_zero='1 - derivative MSE / true derivative energy; zero means constant-field derivative predictor',
                    joint_peak='Correct sign, peak amplitude within 25%, peak time within 30 minutes; sensitivity 50%/60 minutes',
                    joint_rate='Correct sign, maximum 5-minute-rate magnitude within 50%, timing within 30 minutes; sensitivity 60 minutes'),
                limitations=['Exploratory validation analysis after model selection, not independent confirmation',
                    'F magnitude diagnostics only; XYZ inputs are also scored on F, not full vector',
                    'Detrended excursions can include regular daily variation; no external storm/event catalogue',
                    'Repeated/near-tied maxima make timing ambiguous; event counts represent overlapping windows',
                    'Largest excursion metrics exclude first/last seven minutes due to 15-minute smoothing',
                    'True activity/peak strengths define retrospective strata; unavailable for a real hidden interval',
                    'Hit tolerances are diagnostic choices, not established geophysical event criteria',
                    '14-day block bootstrap and 28-day sensitivity reduce but do not remove all temporal dependence',
                    'Diagnostics such as capture and hit rates are descriptive, without simultaneous confidence claims'])

def plots(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    idx={(r['group'],r['length'],r['subset'],r['metric']):r for r in result['rows']}
    dx={(r['group'],r['method'],r['length'],r['subset']):r for r in result['diagnostics']}
    ls=[60,120,240,480,720,1000,1440,2160,2880,4320]
    ticks=['1ч','2ч','4ч','8ч','12ч','16ч40м','24ч','36ч','48ч','72ч']
    colors=['#008781','#B14880','#DB951B','#5965B1']
    models=['unet_F','moment_F','saits_F','timesnet_F'];names=['UNet','MOMENT','SAITS','TimesNet']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    specs=[('mae','Ошибка уровня F'),('peak_value_error','Ошибка в момент истинного пика'),
           ('d5_mae','Ошибка изменений за 5 минут'),('d5_max_error','Ошибка максимальной скорости изменения')]
    for ax,(metric,title) in zip(axes.flat,specs):
        for g,c,name in zip(models,colors,names):
            rr=[idx[(g,L,'gap_high25',metric)] for L in ls]
            ax.plot(ls,[r['gain_pct'] for r in rr],'-o',ms=4,color=c,label=name)
        ax.set_xscale('log');ax.set_xticks(ls,ticks,rotation=40);ax.axhline(0,color='black',lw=1);ax.grid(alpha=.16)
        ax.set_title(title);ax.set_ylabel('Снижение ошибки против сезонного метода, %');ax.legend(fontsize=8)
    fig.suptitle('Верхние 25% по изменчивости внутри пропуска\nАрти, валидация 2023 · вход F · среднее пяти запусков',fontsize=14)
    fig.savefig(out/'event_skill.png',dpi=180);fig.savefig(out/'event_skill.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,key,title in zip(axes,['peak_capture_pct','d5_peak_capture_pct'],['Амплитуда в момент истинного пика','Скорость в момент истинного быстрого изменения']):
        for g,c,name in zip(models,colors,names):
            ax.plot(ls,[dx[(g,g,L,'gap_high25')][key] for L in ls],'-o',ms=3,color=c,label=name)
        ax.plot(ls,[dx[('unet_F','base',L,'gap_high25')][key] for L in ls],'k--',label='Сезонная интерполяция')
        ax.axhline(100,color='#999',lw=1,ls=':');ax.axhline(0,color='#999',lw=1)
        ax.set_xscale('log');ax.set_xticks(ls,ticks,rotation=40);ax.set_title(title);ax.set_ylabel('Восстановленная доля истинной величины, %')
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Восстановлена ли величина в правильный момент?\nВерхний квартиль внутренней активности · F · 100% — точная величина в сумме',fontsize=14)
    fig.savefig(out/'event_capture.png',dpi=180);fig.savefig(out/'event_capture.pdf');plt.close(fig)

def examples(root,data,out):
    """Choose by truth only; strongest-change timestamps at least seven days apart."""
    import matplotlib.pyplot as plt
    L=1440
    order=np.argsort(data[f'true_d5_max_{L}'])[::-1]
    basepath=root/'data/comp/dump_seadec_ARS_comp_val.npz'
    with np.load(basepath,allow_pickle=False) as z:
        m=int(z['margin']);full=z[f'L{L}_true'].astype(float)
        allt=full[:,m:m+L];alld=(allt[:,5:]-allt[:,:-5])/5
        event_offsets=np.argmax(abs(alld),axis=1)
        event_times=data[f'win0_{L}']+data[f'gap0_{L}']+event_offsets+2.5
        chosen=[];times=[]
        for i in order:
            if all(abs(event_times[i]-v)>=7*1440 for v in times):
                chosen.append(int(i));times.append(float(event_times[i]))
            if len(chosen)==3:break
        true=allt[chosen];base=z[f'L{L}_pred'][chosen,m:m+L].astype(float)
    preds={}
    for g in ['unet','moment']:
        stack=[]
        for s in SEEDS:
            with np.load(root/f'data/comp/seeds/dump_{g}_best_s{s}_ARS_comp_val.npz',allow_pickle=False) as z:
                stack.append(z[f'L{L}_pred'][chosen,m:m+L].astype(float))
        preds[g]=np.array(stack)
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    meta=[]
    for j,i in enumerate(chosen):
        absolute=int(data[f'win0_{L}'][i]+data[f'gap0_{L}'][i])
        date=datetime.datetime(2023,1,1)+datetime.timedelta(minutes=absolute)
        d5=(true[j,5:]-true[j,:-5])/5;di=int(np.argmax(abs(d5)))
        offset=float(np.median(true[j]));x=np.arange(L)/60
        ax=axes[j,0];ax.plot(x,true[j]-offset,color='#20252C',lw=1,label='Истина')
        ax.plot(x,base[j]-offset,'--',color='#999',lw=1.5,label='Сезонная')
        for g,c in [('unet','#008781'),('moment','#B14880')]:
            for si in range(5):ax.plot(x,preds[g][si,j]-offset,color=c,alpha=.3,lw=.7,label=g.upper()+' (5 сидов)' if si==0 else None)
        ax.set_ylabel('F − медиана истины, нТл');ax.set_xlabel('Часы от начала пропуска');ax.set_title(date.isoformat(' ')+' · сутки')
        ax.legend(fontsize=7)
        ax=axes[j,1];xd=(np.arange(L-5)+2.5)/60;lo=max(0,di-90);hi=min(L-5,di+91)
        ax.plot(xd[lo:hi],d5[lo:hi],color='#20252C',lw=1.2,label='Истина')
        bd=(base[j,5:]-base[j,:-5])/5;ax.plot(xd[lo:hi],bd[lo:hi],'--',color='#999',label='Сезонная')
        for g,c in [('unet','#008781'),('moment','#B14880')]:
            for si in range(5):
                pd=(preds[g][si,j,5:]-preds[g][si,j,:-5])/5
                ax.plot(xd[lo:hi],pd[lo:hi],color=c,alpha=.35,lw=.8)
        ax.axvline(xd[di],color='#777',ls=':',lw=1);ax.set_ylabel('Изменение за 5 минут, нТл/мин');ax.set_xlabel('Часы от начала пропуска')
        ax.set_title('±90 минут вокруг самого быстрого изменения')
        eventdate=datetime.datetime(2023,1,1)+datetime.timedelta(minutes=float(event_times[i]))
        meta.append(dict(window=i,gap_start_from_year_index=date.isoformat(),event_time_from_year_index=eventdate.isoformat(),true_max_rate=float(abs(d5[di])),block=int(data[f'block_{L}'][i])))
    fig.suptitle('Сильнейшие истинные 5-минутные изменения, разнесённые минимум на неделю\nОтбор по истине, без учёта ошибок моделей; линии показывают отдельные сиды',fontsize=13)
    fig.savefig(out/'examples.png',dpi=180);fig.savefig(out/'examples.pdf');plt.close(fig)
    (out/'examples.json').write_text(json.dumps(meta,indent=2))

def report(result,out):
    rows=[]
    for r in result['rows']:
        if r['subset'] not in ['all','gap_high25']:continue
        lo,hi=r['ci95'];gain='—' if r['gain_pct'] is None else f'{r["gain_pct"]:+.1f}%'
        star='*' if r.get('supported_win') else ''
        rows.append(f'<tr><td>{r["group"]}</td><td>{r["length"]}</td><td>{r["subset"]}</td><td>{r["metric"]}</td><td>{r["baseline"]:.3f}</td><td>{r["net"]:.3f}</td><td>{gain}{star}</td><td>[{lo:.3f}; {hi:.3f}]</td><td>{r["n"]}</td></tr>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Геомагнитные пропуски: пики и быстрые изменения</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#17232d}p,li{line-height:1.55;max-width:1100px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px}th,td{padding:6px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(4){text-align:left}thead{position:sticky;top:0;background:#edf3f6}input{padding:12px;width:350px}.note{padding:15px;background:#edf3f6}a{color:#007b75}</style>
<h1>Восстанавливают ли модели пики и быстрые изменения?</h1><p class="note">Арти, только валидация 2023. Все готовые архитектуры, входы F/XYZ, пять сидов. Оценивается модуль поля F. Это разведочная диагностика после подбора моделей, а не независимая проверка на тесте.</p>
<h2>Основной результат</h2><p><b>Улучшение MAE уровня поля не означает точного восстановления резких изменений.</b> На суточных пропусках в верхнем квартиле внутренней активности UNet/MOMENT снижают MAE уровня примерно на 32%/31%, ошибку величины сглаженного пика — на 15%/13%, но ошибку пятиминутных изменений — лишь на 1,0%/0,7%. В истинный момент самого резкого пятиминутного изменения восстанавливается около 1% его суммарной величины. Для минутных изменений ошибка у обоих лидеров выше сезонного бейзлайна.</p><p>Величина сглаженного отклонения в истинный момент пика восстанавливается примерно на 29% у обеих сетей против 18% у сезонного метода. Это улучшение, но существенная часть амплитуды остаётся потерянной. На трёхсуточных пропусках восстановленная доля сглаженного пика падает примерно до 10%, а резкого изменения — до 0,2%. Показатель capture означает долю величины, а не долю обнаруженных событий.</p><p>Надёжное улучшение времени максимальной скорости на суточных пропусках не обнаружено: средняя ошибка составляет около 7 часов у сетей и сезонной интерполяции. Анализ подтверждает преимущество в восстановлении общего хода, но не позволяет заявить, что модели достоверно восстанавливают резкие возмущения. Это не доказательство принципиальной невозможности: вывод ограничен данными, входами, моделями и протоколом.</p>
<p>Проверка устойчивости: медиана индивидуальных долей восстановленного резкого изменения составляет около 1,2% у UNet/MOMENT. После исключения 10% окон с максимальной истинной скоростью суммарная доля составляет 1,8%/2,4%. Поэтому низкое восстановление не объясняется только несколькими экстремальными окнами. Для CSDI F, наоборот, суммарная доля 72% вводит в заблуждение: пять окон дают примерно 99% её числителя, а медиана индивидуальных долей около 1%. В одном из этих окон предсказанные изменения многократно превышают истинные. Такой показатель нельзя трактовать как 72% точности или обнаружения событий; необходимы ошибки, знак, время и проверка выбросов вместе.</p>
<h2>Что измеряется</h2><ul><li>Пик: наибольшее по модулю отклонение от линии, построенной по медианам 60 видимых минут до и после пропуска. Для поиска пика используется 15-минутное сглаживание полностью внутри пропуска. Первые/последние 7 минут не оцениваются как центры пика. Это отклонение может содержать суточный ход и не является автоматически бурей.</li><li>Отдельно измеряются ошибка величины максимума независимо от времени, ошибка в момент истинного максимума и сдвиг времени максимума. Несколько близких по величине пиков делают последний показатель неоднозначным.</li><li>Быстрые изменения: (F[t+k]−F[t])/k для k=1,5,15 минут. Обе точки внутри пропуска: стыки с наблюдениями исключены. Для 5 минут дополнительно оцениваются максимальная скорость, её время, доля восстановленной величины в истинный момент и корреляция изменений после удаления среднего наклона каждого окна.</li><li>Доля восстановленной амплитуды — отношение сумм предсказанных величин в истинные моменты с учётом знака к сумме истинных модулей. Это не среднее нестабильных индивидуальных отношений. 100% в сумме не означает точного восстановления каждого события.</li><li>Совместное попадание пика: верный знак, амплитуда в пределах ±25%, время ±30 минут; дополнительно ±50%/60 минут. Для максимальной скорости — ±50% и ±30/60 минут. Это выбранные диагностические допуски, не общепринятые критерии геомагнитных событий.</li></ul>
<p>Высокая активность — верхние 25% средней абсолютной минутной разности истины внутри пропуска, отдельно для каждой длительности. Эти значения скрыты от модели и используются только для ретроспективного анализа. Отдельно сохранены нижние 25%, верхние 10% и верхний квартиль амплитуд пиков.</p>
<p>Ошибки усредняются по пяти отдельным моделям; ансамбль не строится. Парный бутстрэп: 2000 повторов, блоки по 14 дней и пересэмплирование сидов. * означает выигрыш по приближённым одновременным 95% интервалам для всех моделей, длительностей от часа, двух основных групп активности и семи основных ошибок. Остальные интервалы поточечные. Для UNet/MOMENT дополнительно проверены блоки по 28 дней. Перекрывающиеся окна не являются независимыми событиями; внешнего каталога бурь нет.</p>
<h2>Сравнение с сезонной интерполяцией</h2><img src="event_skill.png" alt="Ошибки уровня, пиков и быстрых изменений">
<h2>Величина в правильный момент</h2><img src="event_capture.png" alt="Восстановленная доля пиков и быстрых изменений">
<h2>Примеры сильнейших изменений</h2><img src="examples.png" alt="Примеры трёх сильных изменений"><p>Окна выбраны по величине истинного изменения, без просмотра ошибок моделей; моменты выбранных изменений разнесены минимум на семь суток. Это исключает повторное изображение одного пика через перекрывающиеся окна по разные стороны границы недели. Это иллюстрации трудных случаев, а не оценка средней точности. Даты восстановлены из минутного индекса года; они не удостоверяют наличие конкретной бури.</p>
<p><a href="analysis.json">Все показатели, определения и проверки чувствительности (JSON)</a> · <a href="sources.json">Источники</a> · <a href="examples.json">Выбранные окна</a></p>
<h2>Ошибки по моделям и длительностям</h2><input id="filter" placeholder="Фильтр: unet_F, d5_mae, peak_timing…"><table><thead><tr><th>Модель</th><th>Минут</th><th>Подгруппа</th><th>Метрика</th><th>Сезонная</th><th>Сеть</th><th>Снижение</th><th>95% ДИ разницы</th><th>N</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table><script>document.querySelector('#filter').oninput=e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--render-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data,meta=collect(a.root,a.out)
    if a.render_only:result=json.loads((a.out/'analysis.json').read_text())
    else:
        result=analyze(data,meta,2000)
        result['analysis_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (a.out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    plots(result,a.out);examples(a.root,data,a.out);report(result,a.out)
    with zipfile.ZipFile(a.out/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['analysis.json','sources.json','examples.json','report.html','event_skill.png','event_skill.pdf','event_capture.png','event_capture.pdf','examples.png','examples.pdf']:
            z.write(a.out/name,name)
        z.write(Path(__file__),'analyze_geomag_events.py')
    print('DONE '+str(a.out/'results.zip'),flush=True)

if __name__=='__main__':main()
