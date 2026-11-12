"""ARS validation ablation: pretrained, head-only and existing full MOMENT large."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'cluster/experiments/moment-modes-20261029'
SEEDS = [1234,2345,3456,4567,5678]
LRS = [1e-4,1e-3,1e-2]


def write_json(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def checkpoint(name):
    return DEST/(name+'.pt')


def dump(name):
    return DEST/f'dump_{name}_ARS_comp_val.npz'


def train_job(name, mode, comp, seed, lr, steps=12000):
    cmd = [sys.executable,'-u',str(ROOT/'scripts/bench_train.py'),'--arch','moment',
           '--code','ARS','--seed',str(seed),'--steps',str(steps),'--batch','8','--accum','2',
           '--lr',str(lr),'--patch','5','--out',str(checkpoint(name)),
           '--kw','size=large','mode='+mode]
    if comp:
        cmd.append('--comp')
    return dict(name=name,gpu=True,cmd=cmd,done=[str(checkpoint(name)),str(checkpoint(name).with_suffix('.json'))])


def eval_job(name, comp):
    cmd = [sys.executable,'-u',str(ROOT/'scripts/bench_run.py'),'--method',name,
           '--code','ARS','--split','val','--n','1024','--ckpt',str(checkpoint(name)),
           '--out-dir',str(DEST)]
    if comp:
        cmd.append('--save-comp')
    return dict(name='eval_'+name,gpu=True,cmd=cmd,done=[str(dump(name))])


def queue(stage, jobs):
    path = ROOT/f'cluster/jobs/moment_modes_{stage}.jsonl'
    path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    subprocess.run([sys.executable,'-u',str(ROOT/'cluster/run_queue.py'),str(path)],check=True)


def metrics(path, reference):
    rows=[]
    with np.load(path,allow_pickle=False) as d, np.load(reference,allow_pickle=False) as ref:
        assert str(d['setname'])==str(ref['setname'])=='ARS_comp_val'
        assert np.array_equal(d['lengths'],ref['lengths']) and int(d['margin'])==int(ref['margin'])
        m=int(d['margin'])
        for length in d['lengths']:
            length=int(length)
            for key in ['true','win0','gap0','block']:
                assert np.array_equal(d[f'L{length}_{key}'],ref[f'L{length}_{key}'],equal_nan=True),(path.name,length,key)
            true=d[f'L{length}_true'][:,m:m+length].astype(np.float64)
            pred=d[f'L{length}_pred'][:,m:m+length].astype(np.float64)
            assert np.isfinite(pred[np.isfinite(true)]).all()
            err=pred-true
            rows.append(dict(gap_minutes=length,mae=float(np.nanmean(abs(err))),rmse=float(np.sqrt(np.nanmean(err**2)))))
    return dict(geomean_mae=float(np.exp(np.mean(np.log([r['mae'] for r in rows])))),by_length=rows)


def report(selected, head_names):
    reference=ROOT/'data/comp/dump_pchip_ARS_comp_val.npz'
    rows=[]
    for comp in [False,True]:
        label='XYZ' if comp else 'F'
        name='moment_zero_'+label
        cfg=json.loads(checkpoint(name).with_suffix('.json').read_text())
        rows.append(dict(mode='zero-shot',input=label,seed=None,training_seconds=0,
                         preparation_seconds=cfg['seconds'],**metrics(dump(name),reference)))
        for seed in SEEDS:
            name=head_names[(comp,seed)]
            cfg=json.loads(checkpoint(name).with_suffix('.json').read_text())
            rows.append(dict(mode='head',input=label,seed=seed,training_seconds=cfg['seconds'],**metrics(dump(name),reference)))
            full_name='moment_best'+('_xyz' if comp else '')+f'_s{seed}'
            full_dump=ROOT/f'data/comp/seeds/dump_{full_name}_ARS_comp_val.npz'
            if full_dump.exists():
                full_cfg=json.loads((ROOT/f'models/comp/seeds/{full_name}.json').read_text())
                rows.append(dict(mode='full',input=label,seed=seed,training_seconds=full_cfg['seconds'],**metrics(full_dump,reference)))
    groups=[]
    for comp in ['F','XYZ']:
        for mode in ['zero-shot','head','full']:
            values=[r for r in rows if r['input']==comp and r['mode']==mode]
            if not values: continue
            scores=[r['geomean_mae'] for r in values]
            groups.append(dict(input=comp,mode=mode,n=len(scores),mean_geomean_mae=float(np.mean(scores)),
                               seed_sd=float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                               mean_training_hours=float(np.mean([r['training_seconds'] for r in values]))/3600))
    write_json(DEST/'comparison.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validation='ARS 2023 only; identical 1024 windows per gap length; all modes FP32',
        selected_head_lr=selected,groups=groups,rows=rows,
        full_complete=sum(r['mode']=='full' for r in rows)==10,
        limitations=['Validation comparison, not a held-out test claim',
                     'Large size fixed to existing full model; only head learning rate tuned',
                     'Head LR selected on F seed 1234; selected run reused in five-seed series',
                     'Zero-shot uses existing preprocessing and extended 8640-minute context; no weight updates',
                     'Seed SD is descriptive and is not a confidence interval']))
    print('Comparison updated:',json.dumps(groups),flush=True)
    return sum(r['mode']=='full' for r in rows)==10


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    DEST.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'models/comp/moment_best.json').read_text())
    assert cfg['kw']=={'size':'large'} and cfg['steps']==12000 and (cfg['batch'],cfg['accum'])==(8,2)
    assert cfg['patch']==5 and cfg['amp']=='float32'
    zero=[train_job('moment_zero_'+label,'full',comp,1234,1e-4,0) for comp,label in [(False,'F'),(True,'XYZ')]]
    grid=[train_job(f'moment_head_grid_{i}','head',False,1234,lr) for i,lr in enumerate(LRS)]
    sources=[ROOT/'scripts'/name for name in ['bench_train.py','bench_models.py','bench_common.py','bench_run.py']]
    recipe=dict(size='large',precision='float32',station='ARS',validation_year=2023,
                seeds=SEEDS,head_learning_rates=LRS,zero_jobs=zero,head_grid_jobs=grid,
                selection='Minimum geometric mean MAE over 12 lengths on ARS F 2023, 1024 windows/length; ties prefer smaller LR',
                head='Original head-only mode; pretrained head retained; encoder/embedding frozen; original dropout retained',
                checkpoint_selection='Same internal validation every 500 steps, 30 windows per length',
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                scope='ARS ablation only; isolated outputs; production finalists/test protocol unchanged')
    recipe_path=DEST/'recipe.json'
    if recipe_path.exists():
        assert json.loads(recipe_path.read_text())==recipe,'Recipe changed; inspect before resuming'
    else: write_json(recipe_path,recipe)
    for j in zero+grid:
        assert '--cfg' not in j['cmd'] and '--amp' not in j['cmd']
        assert all(Path(p).is_relative_to(DEST) for p in j['done'])
    if args.prepare_only:
        print('PASS: isolated FP32 plan, 2 zero-shot exports, 3 head LR candidates, 5 seeds per input',flush=True)
        return
    queue('zero_export',zero)
    queue('zero_val',[eval_job('moment_zero_'+label,comp) for comp,label in [(False,'F'),(True,'XYZ')]])
    reference=ROOT/'data/comp/dump_pchip_ARS_comp_val.npz'
    write_json(DEST/'zero_results.json',{label:metrics(dump('moment_zero_'+label),reference) for label in ['F','XYZ']})
    queue('head_grid',grid)
    queue('head_grid_val',[eval_job(j['name'],False) for j in grid])
    scores=[metrics(dump(j['name']),reference)['geomean_mae'] for j in grid]
    best=min(range(len(LRS)),key=lambda i:(scores[i],LRS[i]))
    selected=LRS[best]
    write_json(DEST/'head_selection.json',dict(lrs=LRS,scores=scores,selected_lr=selected,selected_name=grid[best]['name']))
    print('Selected head learning rate:',selected,flush=True)
    head_names={(False,1234):grid[best]['name']}
    jobs=[]
    for comp,label in [(False,'F'),(True,'XYZ')]:
        for seed in SEEDS:
            if (comp,seed) in head_names: continue
            name=f'moment_head_{label}_s{seed}'
            head_names[(comp,seed)]=name
            jobs.append(train_job(name,'head',comp,seed,selected))
    queue('head_seeds',jobs)
    queue('head_seedval',[eval_job(name,comp) for (comp,seed),name in head_names.items()])
    while not report(selected,head_names):
        print('New modes complete; waiting for existing full-model validation results',flush=True)
        time.sleep(600)
    print('MOMENT ZERO-SHOT / HEAD / FULL COMPARISON COMPLETE',flush=True)


if __name__=='__main__':
    main()
