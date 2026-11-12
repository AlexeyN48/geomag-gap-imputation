"""Paired calendar-block and training-seed bootstrap on saved validation predictions."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def write(p,d):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    src=a.root/'cluster/analysis/event-followup-20261030';seeds=[1234,2345,3456,4567,5678];allrows=[]
    assert (a.root/'cluster/control/hold_station_training.json').exists()
    data={f.parent.name:json.loads(f.read_text())['rows'] for f in (src/'paired').glob('*/complete.json')}
    specifications=[]
    def event_values(group,L,metric,distance=None):
        ss=[];events=None;minutes=None
        for seed in seeds:
            rows=[r for r in data[f'{group}_s{seed}'] if r['length']==L and (distance is None or r['distance']==distance)]
            ids=sorted(set(r['event'] for r in rows));vals=np.array([np.mean([r[metric] for r in rows if r['event']==eid]) for eid in ids])
            mins=np.array([next(r['peak'] for r in rows if r['event']==eid) for eid in ids])
            if events is not None:assert events==ids and np.array_equal(minutes,mins)
            events=ids;minutes=mins;ss.append(vals)
        return np.array(ss),minutes
    for arch in ['unet','moment']:
        for L in [240,1440]:
            for metric in ['capture_pct','peak_rate_error','mae']:
                near,mins=event_values(arch+'_best',L,metric,5);far,mins2=event_values(arch+'_best',L,metric,L//2-5);assert np.array_equal(mins,mins2)
                specifications.append(dict(family='event_distance',model=arch,length=L,metric=metric,contrast='5min_from_edge minus near_center',diff=near-far,minutes=mins,den=None))
                treat,mins=event_values(arch+'_inc5',L,metric);control,mins2=event_values(arch+'_inc0',L,metric);assert np.array_equal(mins,mins2)
                specifications.append(dict(family='event_loss',model=arch,length=L,metric=metric,contrast='increment_weight5 minus MAE_control; all12placements averaged within event',diff=treat-control,minutes=mins,den=None))
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz') as d:
        for arch in ['unet','moment']:
            for L in [60,240,1440,4320]:
                metrics={k:[] for k in ['mae','d5_mae','capture_pct']};ids0=None
                for seed in seeds:
                    with np.load(src/'evaluation'/f'{arch}_inc0_s{seed}'/f'L{L}.npz') as control, np.load(src/'evaluation'/f'{arch}_inc5_s{seed}'/f'L{L}.npz') as treat:
                        ids=control['ids'];assert np.array_equal(ids,treat['ids'])
                        if ids0 is not None:assert np.array_equal(ids,ids0)
                        ids0=ids.copy()
                        for met in ['mae','d5_mae']:metrics[met].append(treat[met]-control[met])
                        metrics['capture_pct'].append(100*(treat['d5_capture']-control['d5_capture']))
                truth=d[f'L{L}_true'][ids0,int(d['margin']):int(d['margin'])+L].astype(float)
                strength=abs(truth[:,5:]-truth[:,:-5]).max(1)/5;minutes=d[f'L{L}_win0'][ids0]+d[f'L{L}_gap0'][ids0]
                for met,values in metrics.items():specifications.append(dict(family='validation_loss',model=arch,length=L,metric=met,contrast='increment_weight5 minus MAE_control',diff=np.array(values),minutes=minutes,den=strength if met=='capture_pct' else None))
    assert len(specifications)==48
    # Shared calendar/seed bootstrap draws across ALL contrasts preserve their dependence.
    # Repeated placements are averaged within an event before block resampling.
    for blockdays in [14,28]:
        nblocks=int(np.ceil(365/blockdays));reps=10000;rng=np.random.default_rng(20261101)
        weights=rng.multinomial(nblocks,np.ones(nblocks)/nblocks,size=reps)
        sw=rng.multinomial(5,np.ones(5)/5,size=reps)/5
        estimates=[];boot=[];temp=[]
        for spec in specifications:
            delta=spec['diff'];blocks=np.asarray(spec['minutes'])//(blockdays*1440);n=delta.shape[1]
            count=np.bincount(blocks,minlength=nblocks);num=np.array([np.bincount(blocks,weights=v,minlength=nblocks) for v in delta])
            denom=count if spec['den'] is None else np.bincount(blocks,weights=spec['den'],minlength=nblocks)
            est=float(delta.mean()) if spec['den'] is None else float(delta.mean()/spec['den'].mean())
            den=weights@denom;assert (den>0).all()
            draws=((weights@num.T)*sw).sum(1)/den
            estimates.append(est);boot.append(draws)
            temp.append(dict(family=spec['family'],model=spec['model'],length=spec['length'],metric=spec['metric'],contrast=spec['contrast'],
                units='percentage points' if spec['metric']=='capture_pct' else 'nT/min' if spec['metric'] in ['peak_rate_error','d5_mae'] else 'nT',
                n=n,calendar_blocks=int(np.count_nonzero(count)),block_days=blockdays,estimate=est,ci95=np.quantile(draws,[.025,.975]).tolist()))
        estimates=np.array(estimates);boot=np.array(boot);se=boot.std(1,ddof=1)
        critical=float(np.quantile(np.max(abs(boot-estimates[:,None])/np.maximum(se[:,None],1e-12),axis=0),.95))
        for r,s in zip(temp,se):r['simultaneous_ci95']=[r['estimate']-critical*float(s),r['estimate']+critical*float(s)]
        allrows.extend(temp)
    write(a.out/'paired_intervals.json',dict(complete=True,rows=allrows,bootstrap_replicates=10000,seed=20261101,
        method='Paired calendar-block bootstrap crossed with resampling of five matched training seeds; shared draws across48contrasts. Pointwise percentile95 and approximate simultaneous95 max-standardized-deviation intervals. 14-day primary,28-day sensitivity.',
        estimands={'events':'Equal distinct-episode weight; first average left/right or all12placements within an event',
                   'validation':'Equal selected-window weight; capture is ratio of averaged signed peak-rate difference to averaged true peak rate'},
        cautions=['All data are ARS2023 validation. Exploratory, no independent confirmation.',
                  'Five training seeds and one calendar year limit inference; CI is conditional on these event and window samples.',
                  'Changing gap placement also changes which other values are hidden; not an isolated causal intervention on visibility.',
                  'Simultaneous intervals address these48specified contrasts, not the entire history of research choices.']))
    print('Paired CIs complete:48contrasts,14/28-day blocks',flush=True)
if __name__=='__main__':main()
