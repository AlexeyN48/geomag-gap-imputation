"""Paired validation-only comparison of two training precision modes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import bench_metrics as B

p=argparse.ArgumentParser()
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--reference-training',type=Path,required=True)
p.add_argument('--candidate-training',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--components',action='store_true')
a=p.parse_args()
ref_cfg=json.loads(a.reference_training.read_text())
new_cfg=json.loads(a.candidate_training.read_text())
for key in ['arch','steps','batch','accum','lr','patch','window','code','seed','comp','kw','scale']:
    assert ref_cfg[key]==new_cfg[key], f'Training configurations differ: {key}'
rows=[]
component_rows=[]
def compare_arrays(true, pred_ref, pred_new, blocks):
    assert np.isfinite(pred_ref[np.isfinite(true)]).all(), 'Nonfinite reference predictions'
    assert np.isfinite(pred_new[np.isfinite(true)]).all(), 'Nonfinite candidate predictions'
    er=np.abs(pred_ref-true)
    en=np.abs(pred_new-true)
    result=dict(reference_mae=float(np.nanmean(er)),candidate_mae=float(np.nanmean(en)),
        reference_rmse=float(np.sqrt(np.nanmean(er**2))),candidate_rmse=float(np.sqrt(np.nanmean(en**2))),
        reference_p95=float(np.nanpercentile(er,95)),candidate_p95=float(np.nanpercentile(en,95)))
    for metric in ['MAE','RMSE']:
        diff,lo,hi,_=B.paired_bootstrap_ci(en,en**2,er,er**2,metric,blocks=blocks)
        result[metric.lower()+'_difference']=dict(mean=float(diff),ci95=[float(lo),float(hi)],significant=bool(lo>0 or hi<0))
    return result
with np.load(a.reference,allow_pickle=False) as ref, np.load(a.candidate,allow_pickle=False) as new:
    assert str(ref['setname'])==str(new['setname'])=='ARS_comp_val'
    for key in ['lengths','margin','seed','inference_seed']:
        assert np.array_equal(ref[key],new[key]),key
    margin=int(ref['margin'])
    for length in ref['lengths']:
        length=int(length)
        for key in ['true','win0','gap0','block']:
            assert np.array_equal(ref[f'L{length}_{key}'],new[f'L{length}_{key}'],equal_nan=True),(length,key)
        true=ref[f'L{length}_true'][:,margin:margin+length].astype(np.float64)
        pred_ref=ref[f'L{length}_pred'][:,margin:margin+length].astype(np.float64)
        pred_new=new[f'L{length}_pred'][:,margin:margin+length].astype(np.float64)
        blocks=ref[f'L{length}_block']
        row=dict(gap_minutes=length,windows=len(true),**compare_arrays(true,pred_ref,pred_new,blocks))
        if a.components:
            assert ref_cfg['comp'] and new_cfg['comp']
            assert np.array_equal(ref[f'L{length}_trueXYZ'],new[f'L{length}_trueXYZ'],equal_nan=True)
            for channel,label in enumerate('XYZ'):
                arrays=[source[f'L{length}_{key}'][:,margin:margin+length,channel].astype(np.float64)
                        for source,key in [(ref,'trueXYZ'),(ref,'predXYZ'),(new,'predXYZ')]]
                component_rows.append(dict(gap_minutes=length,channel=label,**compare_arrays(*arrays,blocks)))
        rows.append(row)
        print(json.dumps(row),flush=True)
ref_score=float(np.exp(np.mean(np.log([r['reference_mae'] for r in rows]))))
new_score=float(np.exp(np.mean(np.log([r['candidate_mae'] for r in rows]))))
result=dict(reference_precision=ref_cfg['amp'],candidate_precision=new_cfg['amp'],
    seed=ref_cfg['seed'],validation='ARS 2023 only; 1024 windows per gap length',
    reference_geomean_mae=ref_score,candidate_geomean_mae=new_score,
    relative_mae_change_pct=100*(new_score/ref_score-1),
    training_speedup=ref_cfg['seconds']/new_cfg['seconds'],
    reference_training_seconds=ref_cfg['seconds'],candidate_training_seconds=new_cfg['seconds'],
    reference_best_step=ref_cfg['best_step'],candidate_best_step=new_cfg['best_step'],
    bootstrap='Paired calendar-week bootstrap; 2000 resamples; does not estimate training-seed variability',
    limitation='Single-seed diagnostic; no automatic equivalence claim or production precision change',
    by_length=rows,components=component_rows)
a.out.parent.mkdir(parents=True,exist_ok=True)
tmp=a.out.with_suffix('.tmp.json')
tmp.write_text(json.dumps(result,indent=2))
tmp.replace(a.out)
print('GEOMEAN MAE',ref_score,new_score,'CHANGE %',result['relative_mae_change_pct'],flush=True)
print('TRAINING SPEEDUP',result['training_speedup'],flush=True)
