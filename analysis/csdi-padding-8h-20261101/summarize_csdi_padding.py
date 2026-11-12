"""Paired calendar-block and training-seed intervals for padding experiment."""
import argparse,json,html,datetime
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.out/'protocol.json').read_text())
 rng=np.random.default_rng(20261102);R=5000;B=27
 sw=rng.multinomial(5,[.2]*5,size=R);bw=rng.multinomial(B,[1/B]*B,size=R)
 rows=[];cis=[];boot_columns=[]
 for stage,filename in [('primary','metrics.npz'),('secondary','median8_metrics.npz')]:
  for L in protocol['lengths']:
   paths=[a.out/stage/f's{s}_L{L}' for s in protocol['seeds']]
   if not all((d/'complete.json').exists() and (d/filename).exists() for d in paths):continue
   data=[np.load(d/filename) for d in paths]
   ids=data[0]['ids'];minutes=data[0]['minutes'];assert all(np.array_equal(z['ids'],ids) and np.array_equal(z['minutes'],minutes) for z in data)
   block=np.minimum((minutes//(14*1440)).astype(int),B-1);counts=np.bincount(block,minlength=B);denom=5*(bw@counts)
   for metric in ['mae','edge5_mae','first5_mae','last5_mae','d5_mae','max_rate_error','spike5x','d5_capture','d5_pred_max','d5_hit50_30']:
    if metric not in data[0]:continue
    values=np.stack([z[metric].mean(axis=1) for z in data]) # seed, arm, window
    assert np.isfinite(values).all()
    means=values.mean(axis=(0,2));rows.append(dict(stage=stage,length=L,metric=metric,windows=len(ids),means=means.tolist()))
    if stage!='primary' or metric not in ['mae','edge5_mae']:continue
    sums=np.stack([np.stack([values[:,:,block==b].sum(2) for b in range(B)],axis=-1)[s] for s in range(5)])
    # Same seed and block draws for every arm/length; windows remain paired.
    boots=np.einsum('rs,sab,rb->ra',sw,sums,bw)/denom[:,None]
    for control in [0,1]:
     rel=means[2]/means[control]-1;dist=boots[:,2]/boots[:,control]-1
     cis.append(dict(length=L,metric=metric,comparison='padding_aligned vs '+protocol['arms'][control],relative_change=float(rel),
      pointwise95=np.quantile(dist,[.025,.975]).tolist(),practical_limit=.01 if metric=='mae' else .02))
     boot_columns.append(dist)
   for z in data:z.close()
 if cis:
  samples=np.stack(boot_columns,axis=1);est=np.array([r['relative_change'] for r in cis]);sd=samples.std(0,ddof=1)
  critical=float(np.quantile(np.max((samples-est)/np.maximum(sd,1e-15),axis=1),.95))
  for i,row in enumerate(cis):
   row['simultaneous_upper95']=float(est[i]+critical*sd[i]);row['noninferiority_supported']=bool(row['simultaneous_upper95']<row['practical_limit'])
 result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_primary=len(list((a.out/'primary').glob('*/complete.json'))),
  completed_secondary=len(list((a.out/'secondary').glob('*/complete.json'))),descriptive=rows,paired_intervals=cis,
  caveats=['Exploratory ARS validation 2023 only.','Incomplete five-seed groups excluded from comparison.','Primary errors average two individual diffusion draws; secondary metrics use median of eight draws.','Reduced rate error or fewer spikes alone does not prove restoration of true disturbances.'])
 (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 body='<h1>CSDI: выравнивание без удаления наблюдений</h1><p>ARS, валидация 2023. Пять обученных моделей, три варианта входа. FP32.</p>'
 body+=f'<p>Готово: основная серия {result["completed_primary"]}/60; медиана 8 реализаций {result["completed_secondary"]}/60.</p>'
 body+='<p>Отрицательное изменение ошибки означает улучшение. Допуски неухудшения: MAE +1%, границы +2%. Вывод требует верхней одновременной границы ниже допуска.</p>'
 body+='<table><tr><th>Пропуск, мин</th><th>Метрика</th><th>Сравнение</th><th>Изменение</th><th>95% интервал</th><th>Одновременная верхняя граница</th><th>Неухудшение</th></tr>'
 for r in cis:
  body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['length'],r['metric'],r['comparison'],f'{100*r["relative_change"]:.2f}%',f'{100*r["pointwise95"][0]:.2f} … {100*r["pointwise95"][1]:.2f}%',f'{100*r["simultaneous_upper95"]:.2f}%',r['noninferiority_supported']])+'</tr>'
 body+='</table><h2>Ошибки и артефакты</h2><table><tr><th>Серия</th><th>Длина</th><th>Метрика</th><th>Исходный</th><th>Контроль padding</th><th>Выравнивание</th></tr>'
 for r in rows:body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['stage'],r['length'],r['metric']]+[f'{v:.6g}' for v in r['means']])+'</tr>'
 body+='</table><p>Уменьшение искусственных пиков само по себе не доказывает восстановление настоящих возмущений. Это исследовательская проверка на валидации, не слепой тест.</p>'
 (a.out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>CSDI padding</title><style>body{font:16px system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #ccd}th{background:#eef}</style>'+body,encoding='utf-8')
 print('Summary saved:',a.out/'report.html')
if __name__=='__main__':main()
