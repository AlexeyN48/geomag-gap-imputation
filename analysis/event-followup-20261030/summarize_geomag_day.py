"""Collect partial or completed campaign outputs without modifying experiments."""
import argparse,collections,datetime,html,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
manifest=[json.loads(x) for x in (out/'event_day_20261030.jsonl').read_text().splitlines()]
jobs=[dict(name=j['name'],complete=all(Path(f).exists() for f in j['done'])) for j in manifest]
paired=collections.defaultdict(list)
for f in (out/'paired').glob('*/complete.json'):
    d=json.loads(f.read_text());group=d['label'].rsplit('_s',1)[0] if '_s' in d['label'] else d['label']
    paired[group].append(d['rows'])
rows=[]
for group,seeds in paired.items():
    conditions=sorted({(r['length'],r['side'],r['distance']) for r in seeds[0]})
    for L,side,distance in conditions:
        sets=[[r for r in seed if (r['length'],r['side'],r['distance'])==(L,side,distance)] for seed in seeds]
        assert all([r['event'] for r in s]==[r['event'] for r in sets[0]] for s in sets)
        row=dict(group=group,length=L,side=side,distance=distance,seeds=len(sets),events=len(sets[0]),
            visible_fraction=float(np.mean([r['left_visible'] or r['right_visible'] for r in sets[0]])))
        for metric in ['mae','d5_mae','capture_pct','peak_rate_error','timing_error']:
            row[metric]=float(np.mean([[r[metric] for r in s] for s in sets]))
        rows.append(row)
evaluation=[]
for f in (out/'evaluation').glob('*/complete.json'):
    d=json.loads(f.read_text());evaluation.extend(dict(label=d['label'],**r) for r in d['rows'])
csdi=[]
for f in (out/'csdi').glob('*/complete.json'):
    d=json.loads(f.read_text());csdi.append(dict(mode=d['mode'],training_seed=d['training_seed'],rows=len(d['rows']),
        median_coverage90=float(np.median([r['coverage90'] for r in d['rows']])),
        median_crps=float(np.median([r['crps'] for r in d['rows']]))))
summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs,paired=rows,evaluation=evaluation,csdi=csdi,
    caveat='Descriptive exploratory summary; no statistical significance claims. Distinct events weighted equally within each condition; seed means. Test years and other stations remain untouched.')
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
def table(headers,data):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in data)+'</tbody></table>'
body='<h1>Проверки восстановления возмущений</h1><p>Арти, валидация 2023. Дополнительная исследовательская серия. Другие станции и тестовые годы остаются на паузе.</p>'
body+=f'<p>Завершено {sum(j["complete"] for j in jobs)} из {len(jobs)} заданий. Обновлено {summary["created_utc"]}.</p>'
body+='<h2>Парные положения пропуска</h2><p>Один и тот же пик и шестисуточное окно, меняется положение пропуска. Доля амплитуды — знаковая скорость пятиминутного изменения в момент истинного пика, не частота обнаружения событий. Разные положения также скрывают разные участки ряда.</p>'
body+=table(['Модель','Сидов','Пропуск, мин','Ближний край','До пика, мин','Эпизодов','Видимых, %','MAE, нТл','Амплитуда, %'],[[r['group'],r['seeds'],r['length'],r['side'],r['distance'],r['events'],round(r['visible_fraction']*100,1),round(r['mae'],3),round(r['capture_pct'],2)] for r in rows])
body+='<h2>Выполнение</h2>'+table(['Задание','Готово'],[[j['name'],'да' if j['complete'] else 'нет'] for j in jobs])
page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверки восстановления возмущений</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#182532}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #dde4eb;text-align:left}th{background:#eef4fa}p{line-height:1.6}h1{font-size:30px}</style>'+body+'</html>'
(out/'report.html').write_text(page,encoding='utf-8')
print('Summary saved; complete',sum(j['complete'] for j in jobs),'/',len(jobs))
