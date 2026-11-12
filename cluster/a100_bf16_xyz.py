"""Isolated XYZ precision control; reuse the exact production training command."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    dest = ROOT / 'cluster/experiments/moment-bf16-xyz-20261028'
    dest.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in (ROOT/'cluster/jobs/seeds.jsonl').read_text().splitlines() if line.strip()]
    matches = [j for j in jobs if j['name'] == 'moment_best_xyz_s1234']
    assert len(matches) == 1
    reference_job = matches[0]
    original = reference_job['cmd']
    assert '--comp' in original and '--amp' not in original
    assert original[original.index('--seed')+1] == '1234'
    assert original[original.index('--arch')+1] == 'moment'
    cfg_path = Path(original[original.index('--cfg')+1])
    cfg = json.loads(cfg_path.read_text())
    assert cfg['kw'] == {'size': 'large'} and cfg['steps'] == 12000
    assert (cfg['batch'], cfg['accum']) == (8, 2)
    reference_pt = Path(original[original.index('--out')+1])
    candidate_pt = dest/'moment_large_xyz_bf16_s1234.pt'
    train = original.copy()
    train[train.index('--out')+1] = str(candidate_pt)
    train.append('--amp')
    # Exactly two changes: independent output and training autocast.
    restored = train[:-1].copy()
    restored[restored.index('--out')+1] = str(reference_pt)
    assert restored == original

    manifests = {}
    def manifest(stage, command, done):
        path = ROOT/f'cluster/jobs/bf16_xyz_{stage}.jsonl'
        job = dict(name=f'moment_xyz_precision_{stage}', gpu=True, cmd=command, done=[str(p) for p in done])
        path.write_text(json.dumps(job)+'\n')
        manifests[stage] = path

    manifest('train', train, [candidate_pt, candidate_pt.with_suffix('.json')])
    dumps = {}
    for label, pt in [('candidate', candidate_pt), ('reference', reference_pt)]:
        name = f'moment_xyz_precision_{label}'
        dumps[label] = dest/f'dump_{name}_ARS_comp_val.npz'
        command = [sys.executable, '-u', str(ROOT/'scripts/bench_run.py'), '--method', name,
                   '--code', 'ARS', '--split', 'val', '--n', '1024', '--ckpt', str(pt),
                   '--save-comp', '--out-dir', str(dest)]
        manifest(label+'_val', command, [dumps[label]])

    sources = [cfg_path, ROOT/'scripts/bench_train.py', ROOT/'scripts/bench_models.py',
               ROOT/'scripts/bench_common.py', ROOT/'scripts/bench_run.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    recipe = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  reference_job=reference_job, train=train, source_hashes=hashes,
                  difference='BF16 training autocast only; checkpoint selection and inference remain FP32',
                  evaluation='ARS 2023; 1024 identical windows per length; F and X/Y/Z metrics',
                  scope='Separate precision diagnostic; no test data; no automatic production switch')
    recipe_path = dest/'recipe.json'
    if recipe_path.exists():
        previous = json.loads(recipe_path.read_text())
        for key in ['reference_job', 'train', 'source_hashes']:
            assert previous[key] == recipe[key], f'Recipe changed: {key}'
    else:
        recipe_path.write_text(json.dumps(recipe, indent=2))
    if args.prepare_only:
        print('PASS: command matches production XYZ seed 1234 except AMP and separate output', flush=True)
        return

    def run_queue(stage):
        subprocess.run([sys.executable, '-u', str(ROOT/'cluster/run_queue.py'), str(manifests[stage])], check=True)

    print('XYZ BF16 control queued; waiting for a free GPU under shared locks', flush=True)
    run_queue('train')
    run_queue('candidate_val')
    print('BF16 XYZ training and validation complete; waiting for FP32 reference if necessary', flush=True)
    while not all(Path(p).exists() for p in reference_job['done']):
        time.sleep(60)
    ref_cfg = json.loads(reference_pt.with_suffix('.json').read_text())
    assert ref_cfg['amp'] == 'float32' and ref_cfg['comp'] and ref_cfg['seed'] == 1234
    run_queue('reference_val')
    subprocess.run([sys.executable, '-u', str(ROOT/'cluster/a100_compare_precision.py'),
                    '--reference', str(dumps['reference']), '--candidate', str(dumps['candidate']),
                    '--reference-training', str(reference_pt.with_suffix('.json')),
                    '--candidate-training', str(candidate_pt.with_suffix('.json')),
                    '--components', '--out', str(dest/'comparison.json')], check=True)
    print('XYZ BF16 CONTROL AND PAIRED COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
