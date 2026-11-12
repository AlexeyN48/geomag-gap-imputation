"""Isolated inference audit; never modifies original checkpoints or dumps."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np

def write(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.root/'scripts'))
    import torch
    import bench_common as C,bench_models as M
    torch.set_num_threads(8);assert torch.cuda.is_available()
    hold=a.root/'cluster/control/hold_station_training.json';assert hold.exists()
    F=C.load_year(2023,'ARS');XYZ=C.load_year_comp(2023,'ARS');L=1440
    with np.load(a.root/'data/comp/dump_seadec_ARS_comp_val.npz',allow_pickle=False) as d:
        starts=d['L1440_win0'].astype(int);gaps=d['L1440_gap0'].astype(int);m=int(d['margin'])
        true=d['L1440_true'][:,m:m+L].astype(float)
        rate=abs((true[:,5:]-true[:,:-5])/5).max(1)
    # Include the reported pathological case, truth-selected strong cases, and fixed random controls.
    ids=list(dict.fromkeys([52]+np.argsort(rate)[-3:].tolist()+np.random.default_rng(20261030).choice(len(starts),8,replace=False).tolist()))
    sampling_seeds=list(range(7001,7009));raw={};results=[];tests=[];input_stats=[]
    def rng(seed):torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    configs=[('F',s) for s in C.SEEDS]+[('XYZ',1234)]
    for mode,seed in configs:
        name=f'csdi_best'+('_xyz' if mode=='XYZ' else '')+f'_s{seed}'
        completed=a.out/(name+'_audit.json')
        if completed.exists():
            saved=json.loads(completed.read_text());results.extend(saved['results']);tests.extend(saved['tests']);input_stats.extend(saved['inputs']);continue
        net,ck=M.load(a.root/f'models/comp/seeds/{name}.pt','cuda');assert net.n_samples==1
        source=XYZ if mode=='XYZ' else F;inps=[];feats=[];targets=[];localinputs=[];localtests=[];localrows=[]
        for i in ids:
            st=int(starts[i]);gp=int(gaps[i]);target=source[st:st+C.W].copy();inp=target.copy();inp[gp:gp+L]=np.nan
            if mode=='XYZ':inp[~np.isfinite(F[st:st+C.W])]=np.nan
            ft=M.featurize(inp,st,ck['scale']);inps.append(inp);feats.append(ft);targets.append(target)
            values=ft[0][:,:M.POUT];mask=ft[1][:,:M.POUT].astype(bool)
            shifted=M.featurize(inp+np.float32(1000),st,ck['scale'])
            localinputs.append(dict(checkpoint=name,window=i,visible_min=float(np.nanmin(inp)),visible_max=float(np.nanmax(inp)),
                center=np.asarray(ft[2]).tolist(),scale=np.asarray(ck['scale']).tolist(),
                missing_minutes=int((~np.isfinite(inp).all(1)).sum()) if mode=='XYZ' else int((~np.isfinite(inp)).sum()),
                max_abs_visible_normalized=float(abs(values[mask]).max()),
                shift1000_feature_max_difference=float(abs(ft[0]-shifted[0]).max()),
                hidden_features_zero=bool((values[~mask]==0).all())))
        X=torch.from_numpy(np.stack([f[0] for f in feats])).cuda();mask=torch.from_numpy(np.stack([f[1] for f in feats])).cuda()
        with torch.no_grad():
            rng(7001);first=net(X,mask)
            rng(7001);repeat=net(X,mask)
            contaminated=X.clone();contaminated[:,:,:M.POUT]=torch.where(mask[:,:,:M.POUT]>0,X[:,:,:M.POUT],torch.full_like(X[:,:,:M.POUT],12345))
            rng(7001);hidden=net(contaminated,mask)
        localtests.append(dict(checkpoint=name,repeat_max_difference=float((first-repeat).abs().max()),
            hidden_value_contamination_max_difference=float((first-hidden).abs().max())))
        assert torch.equal(first,repeat),'Non-reproducible same-seed inference'
        assert torch.equal(first,hidden),'Masked values leaked into CSDI prediction'
        outputs=[]
        for s in sampling_seeds:
            with torch.no_grad():rng(s);y=net(X,mask).cpu().numpy().reshape(len(ids),C.W,M.NCH)
            predictions=[]
            for j,i in enumerate(ids):
                gp=int(gaps[i]);center=feats[j][2];scale=ck['scale']
                if mode=='XYZ':pr=y[j]*np.asarray(scale,np.float64)[None,:]+center;pr=np.where(np.isfinite(inps[j]),inps[j],pr);mag=np.sqrt((pr**2).sum(1))
                else:pr=y[j,:,0]*scale+center;mag=np.where(np.isfinite(inps[j]),inps[j],pr)
                pred=mag[gp:gp+L].astype(np.float32);assert np.isfinite(pred).all();predictions.append(pred)
            outputs.append(predictions)
        outputs=np.asarray(outputs);np.savez_compressed(a.out/(name+'_samples.npz'),predictions=outputs,window_ids=ids,sampling_seeds=sampling_seeds)
        for j,i in enumerate(ids):
            t=true[i];dr=(t[5:]-t[:-5])/5;peak=int(np.argmax(abs(dr)))
            samples=outputs[:,j].astype(float);dp=(samples[:,5:]-samples[:,:-5])/5
            med=np.median(samples,axis=0);md=(med[5:]-med[:-5])/5
            localrows.append(dict(checkpoint=name,window=i,training_seed=seed,sampling_seeds=sampling_seeds,
                raw_true_min=float(t.min()),raw_true_max=float(t.max()),true_max_rate=float(abs(dr).max()),
                sample_mae=abs(samples-t).mean(1).tolist(),sample_min=samples.min(1).tolist(),sample_max=samples.max(1).tolist(),
                sample_max_rate=abs(dp).max(1).tolist(),sample_signed_rate_at_true_peak=(np.sign(dr[peak])*dp[:,peak]).tolist(),
                median8_mae=float(abs(med-t).mean()),median8_max_rate=float(abs(md).max()),
                median8_signed_rate_at_true_peak=float(np.sign(dr[peak])*md[peak])))
        # Independent transcription of the installed conditional sampler, checked against original.
        if seed==1234 and mode=='F':
            with torch.no_grad():
                cond=mask[:,:,:M.POUT].permute(0,2,1);obs=X[:,:,:M.POUT].permute(0,2,1)*cond;side=net._side(X,cond)
                rng(7001);sample=torch.randn_like(obs);trace=[]
                for step in range(net.body.n_diffusion_steps-1,-1,-1):
                    inp=torch.cat([(cond*obs).unsqueeze(1),((1-cond)*sample).unsqueeze(1)],dim=1)
                    predicted=net.body.diff_model(inp,side,torch.tensor([step],device=X.device))
                    c1=1/net.body.alpha_hat[step]**.5;c2=(1-net.body.alpha_hat[step])/(1-net.body.alpha[step])**.5
                    sample=c1*(sample-c2*predicted)
                    if step>0:
                        noise=torch.randn_like(sample);sigma=((1-net.body.alpha[step-1])/(1-net.body.alpha[step])*net.body.beta[step])**.5;sample+=sigma*noise
                    if step in [49,40,30,20,10,0]:trace.append(dict(step=step,missing_sample_max_abs=(sample.abs()*(1-cond)).amax((1,2)).cpu().tolist()))
                assembled=(obs*cond+sample*(1-cond)).permute(0,2,1)
                delta=float((assembled-first).abs().max());assert delta<1e-5
            write(a.out/'csdi_sampler_trace.json',dict(window_ids=ids,manual_vs_installed_max_difference=delta,trace=trace))
        saved=dict(results=localrows,tests=localtests,inputs=localinputs);write(completed,saved)
        results.extend(localrows);tests.extend(localtests);input_stats.extend(localinputs)
        del net,X,mask,first,repeat,hidden;torch.cuda.empty_cache()
        print('CSDI audit complete '+name,flush=True)
    result=dict(complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),validation='ARS 2023 only',
        window_ids=ids,sampling_seeds=sampling_seeds,results=results,tests=tests,inputs=input_stats,
        original_protocol='One diffusion sample; original dumps and checkpoints unchanged',
        diagnostic_protocol='Eight fixed sampling seeds per checkpoint; five F training seeds and one XYZ control; median8 separately reported',
        limitations=['Targeted audit cases, not an unbiased estimate of benchmark performance',
            'Random streams differ from original full-dump batch sequence; reruns test recurrence rather than bitwise reproduction of stored outputs',
            'No retraining and no automatic replacement of primary results'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hold_present=hold.exists())
    assert hold.exists();write(a.out/'csdi_audit.json',result);print('CSDI AUDIT COMPLETE',flush=True)

if __name__=='__main__':main()
