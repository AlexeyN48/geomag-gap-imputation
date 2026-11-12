"""Read existing validation predictions; never train models or open test data."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

ARCHS = 'dlinear nbeatsx nhits tide tsmixerx segrnn unet timesnet saits imputeformer csdi moment'.split()
SEEDS = [1234, 2345, 3456, 4567, 5678]
PRIMARY = ['all', 'low25', 'middle50', 'high25']
LABELS = {'all':'Все окна', 'low25':'Низкая: нижние 25%', 'middle50':'Средняя: средние 50%',
          'high25':'Высокая: верхние 25%', 'top10':'Верхние 10%', 'lower50':'Нижние 50%', 'upper50':'Верхние 50%'}
PRETTY = {'unet':'UNet','moment':'MOMENT','saits':'SAITS','timesnet':'TimesNet','csdi':'CSDI',
          'imputeformer':'ImputeFormer','tsmixerx':'TSMixerx','dlinear':'DLinear','nhits':'N-HiTS',
          'nbeatsx':'N-BEATSx','tide':'TiDE','segrnn':'SegRNN'}

def collect(root, out):
    cache = out/'window_errors.npz'
    manifest = out/'sources.json'
    if cache.exists() and manifest.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {k:z[k] for k in z.files}, json.loads(manifest.read_text())
    reference = root/'data/comp/dump_seadec_ARS_comp_val.npz'
    files=[]
    def record(p):
        st=p.stat()
        files.append(dict(path=str(p.relative_to(root)),size=st.st_size,mtime_ns=st.st_mtime_ns))
    record(reference)
    with np.load(reference,allow_pickle=False) as d:
        assert str(d['setname'])=='ARS_comp_val'
        lengths=[int(x) for x in d['lengths']]; margin=int(d['margin'])
        ref={f'L{L}_{k}':d[f'L{L}_{k}'] for L in lengths for k in ['true','win0','gap0','block','act']}
        data=dict(lengths=np.array(lengths),margin=np.array(margin),seeds=np.array(SEEDS))
        for L in lengths:
            t=ref[f'L{L}_true'][:,margin:margin+L].astype(float)
            p=d[f'L{L}_pred'][:,margin:margin+L].astype(float)
            assert np.isfinite(t).all() and np.isfinite(p).all()
            data[f'base_mae_{L}']=abs(p-t).mean(1)
            data[f'base_mse_{L}']=((p-t)**2).mean(1)
            data[f'act_{L}']=ref[f'L{L}_act'].astype(float)
            data[f'gap_act_{L}']=abs(np.diff(t,axis=1)).mean(1)
            for k in ['block','win0','gap0']:data[f'{k}_{L}']=ref[f'L{L}_{k}']
    groups=[]; incomplete=[]
    for a in ARCHS:
        for mode,suffix in [('F',''),('XYZ','_xyz')]:
            names=[f'{a}_best{suffix}_s{s}' for s in SEEDS]
            paths=[root/f'data/comp/seeds/dump_{name}_ARS_comp_val.npz' for name in names]
            if not all(p.exists() for p in paths):
                incomplete.append(dict(arch=a,input=mode,available=sum(p.exists() for p in paths)))
                continue
            group=f'{a}_{mode}'; groups.append(group)
            for L in lengths:
                data[f'{group}_mae_{L}']=[];data[f'{group}_mse_{L}']=[]
            for p in paths:
                record(p)
                with np.load(p,allow_pickle=False) as d:
                    assert str(d['setname'])=='ARS_comp_val'
                    assert list(d['lengths'])==lengths and int(d['margin'])==margin
                    for L in lengths:
                        for k in ['true','win0','gap0','block','act']:
                            key=f'L{L}_{k}'
                            assert np.array_equal(d[key],ref[key],equal_nan=True),(p.name,key)
                        t=ref[f'L{L}_true'][:,margin:margin+L].astype(float)
                        pred=d[f'L{L}_pred'][:,margin:margin+L].astype(float)
                        assert np.isfinite(pred).all()
                        data[f'{group}_mae_{L}'].append(abs(pred-t).mean(1))
                        data[f'{group}_mse_{L}'].append(((pred-t)**2).mean(1))
            for L in lengths:
                for k in ['mae','mse']:data[f'{group}_{k}_{L}']=np.array(data[f'{group}_{k}_{L}'])
        print('Collected '+a,flush=True)
    data['groups']=np.array(groups)
    meta=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),
              validation='ARS 2023; 1024 windows per length; complete five-seed groups only',
              sources=files,incomplete=incomplete,source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if '__file__' in globals() else 'stdin',
              station_training_hold=(root/'cluster/control/hold_station_training.json').exists())
    np.savez_compressed(cache,**data)
    manifest.write_text(json.dumps(meta,indent=2))
    return data,meta

def selections(act):
    assert np.isfinite(act).all()
    q25,q50,q75,q90=np.quantile(act,[.25,.5,.75,.9])
    return dict(all=np.ones(len(act),bool),low25=act<=q25,middle50=(act>q25)&(act<=q75),
                high25=act>q75,top10=act>q90,lower50=act<=q50,upper50=act>q50),[float(v) for v in [q25,q50,q75,q90]]

def bootstrap_weights(blocks,reps,span):
    # IDs in input are year*100 + integer week since Jan 1 (not ISO weeks).
    clusters=(blocks%100)//span
    levels,inv=np.unique(clusters,return_inverse=True)
    rng=np.random.default_rng(20261030+span)
    w=rng.multinomial(len(levels),np.ones(len(levels))/len(levels),size=reps)
    sw=rng.multinomial(5,np.ones(5)/5,size=reps)/5
    return inv,levels,w,sw

def compare(mae,mse,bmae,bmse,select,design,return_gain=False):
    inv,levels,w,sw=design
    n=len(levels)
    count=np.bincount(inv[select],minlength=n)
    denominator=w@count
    valid=denominator>0
    def summed(x):
        return np.stack([np.bincount(inv[select],weights=row[select],minlength=n) for row in np.atleast_2d(x)])
    boot_base=(w@summed(bmae)[0])[valid]/denominator[valid]
    boot_net=((w@summed(mae).T)[valid]/denominator[valid,None]*sw[valid]).sum(1)
    boot_br=np.sqrt((w@summed(bmse)[0])[valid]/denominator[valid])
    boot_nr=(np.sqrt((w@summed(mse).T)[valid]/denominator[valid,None])*sw[valid]).sum(1)
    base=float(bmae[select].mean());net=float(mae[:,select].mean())
    br=float(np.sqrt(bmse[select].mean()));nr=float(np.sqrt(mse[:,select].mean(1)).mean())
    diff=boot_net-boot_base; rdiff=boot_nr-boot_br
    result=dict(n=int(select.sum()),blocks=int((count>0).sum()),base_mae=base,net_mae=net,
                mae_gain_pct=100*(1-net/base),mae_diff=net-base,
                mae_diff_ci95=np.quantile(diff,[.025,.975]).tolist(),
                mae_gain_ci95=np.quantile(100*(1-boot_net/boot_base),[.025,.975]).tolist(),
                base_rmse=br,net_rmse=nr,rmse_gain_pct=100*(1-nr/br),
                rmse_diff_ci95=np.quantile(rdiff,[.025,.975]).tolist(),
                seed_mae_sd=float(np.std(mae[:,select].mean(1),ddof=1)),
                seeds_better=int((mae[:,select].mean(1)<base).sum()),
                window_win_pct=float(100*(mae[:,select]<bmae[select]).mean()))
    assert len(diff)==len(w),'Bootstrap subset absent in a resample'
    if return_gain:return result,diff,100*(1-boot_net/boot_base)
    return result,diff

def analyze(data,meta,reps):
    lengths=[int(x) for x in data['lengths']];groups=list(data['groups'])
    rows=[];sens=[];thresholds=[];primary_boot=[];primary_indices=[];contrasts=[]
    for L in lengths:
        sel,thr=selections(data[f'act_{L}']); gap_sel,gthr=selections(data[f'gap_act_{L}'])
        thresholds.append(dict(length=L,window_quantiles=thr,gap_quantiles=gthr,
                               n=len(data[f'act_{L}']),weeks=len(np.unique(data[f'block_{L}']))))
        design=bootstrap_weights(data[f'block_{L}'],reps,2)
        long_design=bootstrap_weights(data[f'block_{L}'],reps,4)
        for g in groups:
            args=[data[f'{g}_mae_{L}'],data[f'{g}_mse_{L}'],data[f'base_mae_{L}'],data[f'base_mse_{L}']]
            for sub in PRIMARY+['top10','lower50','upper50']:
                row,boot=compare(*args,sel[sub],design)
                row.update(group=g,length=L,subset=sub,activity='six_day_window')
                if sub in PRIMARY:primary_indices.append(len(rows));primary_boot.append(boot)
                rows.append(row)
            # Two predeclared sensitivity checks for main contenders, no selection by result.
            if g in ['unet_F','moment_F','saits_F','timesnet_F','unet_XYZ','saits_XYZ']:
                for sub in ['all','low25','high25']:
                    row,_=compare(*args,sel[sub],long_design)
                    row.update(group=g,length=L,subset=sub,activity='six_day_window',check='28_day_blocks')
                    sens.append(row)
                for sub in ['low25','high25']:
                    row,_=compare(*args,gap_sel[sub],design)
                    row.update(group=g,length=L,subset=sub,activity='hidden_gap',check='within_gap_activity')
                    sens.append(row)
                for activity,selection in [('six_day_window',sel),('hidden_gap',gap_sel)]:
                    low,_,lb=compare(*args,selection['low25'],design,return_gain=True)
                    high,_,hb=compare(*args,selection['high25'],design,return_gain=True)
                    contrasts.append(dict(group=g,length=L,activity=activity,
                        high_minus_low_gain_pp=high['mae_gain_pct']-low['mae_gain_pct'],
                        ci95=np.quantile(hb-lb,[.025,.975]).tolist()))
        print('Analyzed length '+str(L),flush=True)
    # Exploratory family-wide bootstrap max-standardized-deviation bands for MAE:
    # every complete architecture/input group x all 12 lengths x 4 primary strata.
    boots=np.array(primary_boot).T
    centers=np.array([rows[i]['mae_diff'] for i in primary_indices])
    se=boots.std(axis=0,ddof=1)
    assert (se>0).all()
    maximum=np.max(abs((boots-centers)/se),axis=1)
    critical=float(np.quantile(maximum,.95))
    for k,i in enumerate(primary_indices):
        lo,hi=centers[k]-critical*se[k],centers[k]+critical*se[k]
        rows[i]['mae_simultaneous_ci95']=[float(lo),float(hi)]
        rows[i]['family_supported_win']=bool(hi<0)
        rows[i]['family_supported_loss']=bool(lo>0)
    result=dict(meta=meta,bootstrap=dict(reps=reps,seed=20261030,blocks_days=14,
                training_seeds_resampled=True,estimator='Mean of five individual-model errors; not a prediction ensemble',
                family_comparisons=len(primary_indices),family_critical=critical,
                family_method='Approximate bootstrap max standardized deviation, all primary MAE comparisons',
                sensitivity_block_days=28),thresholds=thresholds,rows=rows,sensitivity=sens,activity_contrasts=contrasts,
                limitations=['Validation-only exploratory analysis; HPO was performed on this validation year',
                             'Local F variability is not Kp, Dst or a storm label',
                             'Activity uses unmasked truth for retrospective stratification, not model input or deployable routing',
                             'All inputs evaluated on magnitude F, not component-wise error',
                             'Five training seeds and one station-year limit generalization',
                             'Block bootstrap reduces overlap dependence but does not guarantee independence; 28-day sensitivity included',
                             'Simultaneous bands only cover primary MAE family; RMSE and sensitivity intervals are pointwise'])
    return result

def figures(data,result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    lengths=[int(x) for x in data['lengths']]
    ticks=['5м','15м','1ч','2ч','4ч','8ч','12ч','16ч40м','24ч','36ч','48ч','72ч']
    idx={(r['group'],r['length'],r['subset']):r for r in result['rows']}
    focus=['unet_F','moment_F','saits_F','timesnet_F','imputeformer_F','csdi_F']
    colors=['#007F86','#B34881','#E49B20','#5563AE','#647765','#999999']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    base=[idx[('unet_F',L,'all')]['base_mae'] for L in lengths]
    axes[0].plot(lengths,base,'k--',lw=2,label='Сезонная интерполяция')
    for g,c in zip(focus,colors):
        rr=[idx[(g,L,'all')] for L in lengths]
        label=PRETTY[g.rsplit('_',1)[0]]
        axes[0].plot(lengths,[r['net_mae'] for r in rr],'-o',ms=3,color=c,label=label)
        if g in focus[:4]:axes[1].plot(lengths[2:],[r['mae_gain_pct'] for r in rr][2:],'-o',ms=3,color=c,label=label)
    for ax in axes:
        ax.set_xscale('log');ax.set_xticks(lengths,ticks,rotation=50);ax.grid(alpha=.16)
        ax.set_xlabel('Длительность пропуска')
    axes[0].set_yscale('log');axes[0].set_ylabel('MAE модуля поля, нТл');axes[0].legend(fontsize=8)
    axes[1].axhline(0,color='black',lw=1)
    axes[1].set_xticks(lengths[2:],ticks[2:],rotation=50)
    axes[1].set_title('Выигрыш на пропусках от 1 часа',fontsize=11)
    axes[1].set_ylabel('Снижение MAE против сезонного метода, %');axes[1].legend(fontsize=8)
    fig.suptitle('Арти, валидация 2023 • средняя ошибка пяти запусков • вход F',fontsize=14)
    fig.savefig(out/'duration.png',dpi=180);fig.savefig(out/'duration.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for ax,g in zip(axes.flat,focus[:4]):
        matrix=np.array([[idx[(g,L,s)]['mae_gain_pct'] for L in lengths] for s in PRIMARY])
        im=ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=50),aspect='auto')
        for y,s in enumerate(PRIMARY):
            for x,L in enumerate(lengths):
                r=idx[(g,L,s)];v=matrix[y,x]
                mark='*' if r.get('family_supported_win') else ''
                ax.text(x,y,f'{v:.0f}{mark}',ha='center',va='center',fontsize=8,color='white' if v < -65 or v>32 else '#111111')
        ax.set_xticks(range(len(lengths)),ticks,rotation=50)
        ax.set_yticks(range(4),['Все','Нижние 25%','Средние 50%','Верхние 25%'])
        ax.set_title(PRETTY[g.rsplit('_',1)[0]]+' · вход F')
    fig.colorbar(im,ax=axes,shrink=.75,label='Снижение MAE, % (синий = лучше сезонного метода)')
    fig.suptitle('Где сети выигрывают у сезонной интерполяции\nАктивность: изменчивость F за шестисуточное окно',fontsize=14)
    fig.supxlabel('* — выигрыш с учётом одновременных интервалов по всей основной семье сравнений',fontsize=9)
    fig.savefig(out/'activity_heatmap.png',dpi=180);fig.savefig(out/'activity_heatmap.pdf');plt.close(fig)
    # All available architecture/input groups, no selection of a winning seed.
    groups=sorted(list(data['groups']),key=lambda g:np.mean(np.log([idx[(g,L,'all')]['net_mae'] for L in lengths])))
    mat=np.array([[idx[(g,L,'all')]['mae_gain_pct'] for L in lengths] for g in groups])
    fig,ax=plt.subplots(figsize=(11,10),layout='constrained')
    im=ax.imshow(mat,cmap='RdBu',norm=TwoSlopeNorm(vmin=-150,vcenter=0,vmax=50),aspect='auto')
    for y,g in enumerate(groups):
        for x,L in enumerate(lengths):
            v=mat[y,x];ax.text(x,y,f'{v:.0f}',ha='center',va='center',fontsize=8,color='white' if v < -95 or v>32 else '#111')
    ax.set_xticks(range(len(lengths)),ticks)
    ax.set_yticks(range(len(groups)),[PRETTY[g.rsplit('_',1)[0]]+' '+g.rsplit('_',1)[1] for g in groups])
    ax.set_title('Все готовые модели: снижение MAE против сезонной интерполяции, %\nАрти 2023 · все окна · среднее пяти запусков')
    fig.colorbar(im,ax=ax,shrink=.6)
    fig.savefig(out/'all_models.png',dpi=180);fig.savefig(out/'all_models.pdf');plt.close(fig)
    gapidx={(r['group'],r['length'],r['subset']):r for r in result['sensitivity'] if r['check']=='within_gap_activity'}
    fig,axes=plt.subplots(1,2,figsize=(12,5.1),layout='constrained')
    positions=[(1440,idx),(1440,gapidx),(4320,idx),(4320,gapidx)]
    for ax,g in zip(axes,['unet_F','moment_F']):
        for sub,shift,color,label in [('low25',-.19,'#168477','Низкая изменчивость (25%)'),('high25',.19,'#AF537B','Высокая изменчивость (25%)')]:
            rr=[source[(g,L,sub)] for L,source in positions]
            v=np.array([r['mae_gain_pct'] for r in rr]);ci=np.array([r['mae_gain_ci95'] for r in rr]).T
            ax.bar(np.arange(4)+shift,v,width=.36,color=color,label=label)
            ax.errorbar(np.arange(4)+shift,v,yerr=np.maximum(np.array([v-ci[0],ci[1]-v]),0),fmt='none',ecolor='#333',capsize=3,lw=1)
            for x,y in zip(np.arange(4)+shift,v):ax.text(x,y+1,f'{y:.0f}%',ha='center',fontsize=9)
        ax.set_xticks(range(4),['24 ч\nактивность окна','24 ч\nактивность пропуска','72 ч\nактивность окна','72 ч\nактивность пропуска'],fontsize=9)
        ax.set_ylim(0,75);ax.set_ylabel('Снижение MAE против сезонного метода, %');ax.set_title(PRETTY[g.rsplit('_',1)[0]]+' · вход F')
        ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Определение активности меняет вывод о зависимости выигрыша\nОкно — 6 суток; пропуск — только скрытый участок',fontsize=14)
    fig.supxlabel('Поточечные 95% интервалы: парный бутстрэп по 14-дневным блокам и пяти сидам',fontsize=9)
    fig.savefig(out/'activity_definition.png',dpi=180);fig.savefig(out/'activity_definition.pdf');plt.close(fig)

def report(result,out):
    import html
    rows=result['rows']
    table=['<table><thead><tr><th>Модель</th><th>Пропуск, мин</th><th>Активность</th><th>N</th><th>Сезонная MAE</th><th>Сеть MAE</th><th>Выигрыш MAE</th><th>95% ДИ разницы MAE</th><th>Общий ДИ</th><th>Выигрыш RMSE</th></tr></thead><tbody>']
    for r in rows:
        if r['subset'] not in PRIMARY:continue
        lo,hi=r['mae_diff_ci95']; flo,fhi=r['mae_simultaneous_ci95']
        g=r['group'];color='win' if r['family_supported_win'] else ('loss' if r['family_supported_loss'] else '')
        table.append(f'<tr class="{color}"><td>{html.escape(g)}</td><td>{r["length"]}</td><td>{LABELS[r["subset"]]}</td><td>{r["n"]}</td><td>{r["base_mae"]:.3f}</td><td>{r["net_mae"]:.3f}</td><td>{r["mae_gain_pct"]:+.1f}%</td><td>[{lo:.3f}; {hi:.3f}]</td><td>[{flo:.3f}; {fhi:.3f}]</td><td>{r["rmse_gain_pct"]:+.1f}%</td></tr>')
    table.append('</tbody></table>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Геомагнитные пропуски: анализ валидации</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#17232d}h1{font-size:30px}p,li{line-height:1.6;max-width:1050px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}thead{position:sticky;top:0;background:#eaf0f4}.win{background:#e7f4f0}.loss{background:#fff0ed}input{padding:12px;width:320px;margin:16px 0}.note{background:#edf3f6;padding:15px}a{color:#006977}</style>
<h1>Где нейросети выигрывают у сезонной интерполяции</h1>
<p class="note">Арти (ARS), только валидация 2023. 1024 одинаковых окна на каждую из 12 длительностей. Только завершённые группы по пяти сидам. Оценивается модуль F при входе F или XYZ. Это разведочный анализ данных, использованных для подбора моделей; он не заменяет независимый тест.</p>
<h2>Основные наблюдения</h2><ul><li>При входе F UNet и MOMENT проигрывают сезонной интерполяции на пропусках до часа. На двух часах переход ещё не произошёл; на четырёх часах средний выигрыш составляет около 9–10%, но общий интервал после поправки на множество сравнений включает ноль.</li><li>На пропусках от 8 часов оба лидера выигрывают в среднем по всем окнам с учётом одновременных интервалов. На 24–72 часах снижение MAE составляет около 31–37%. В верхнем квартиле активности шестисуточных окон уверенный выигрыш начинается с 12 часов.</li><li>Активность шестисуточного окна и активность внутри пропуска дают разные зависимости: на 24 часах выигрыш UNet равен 28% против 41% в нижнем/верхнем квартилях активности окна, но 52% против 32% при разбиении по изменчивости самого пропуска. Это разные группы окон; их нельзя подменять друг другом или автоматически считать метками бурь.</li><li>На 72 часах в высокоактивных шестисуточных окнах UNet снижает MAE на 31%, но RMSE лишь примерно на 14%. Крупные ошибки остаются важной частью результата.</li><li>Остальные сети также могут выигрывать на длинных пропусках. Например, SAITS F даёт около 35% снижения MAE на 24 часах, TimesNet F — 30%; слабый общий результат архитектуры не означает проигрыша на каждой длине.</li></ul>
<h2>Как читать результаты</h2><p>Выигрыш = 100 × (1 − MAE сети / MAE сезонной интерполяции). Положительное значение — преимущество сети. Усредняются ошибки пяти отдельных обучений; ансамбль предсказаний не строится. Ошибки измерены в нТл. Эталон — seadec с уже выбранными на валидации параметрами.</p>
<p>Активность — средняя абсолютная минутная разность F в исходном шестисуточном окне. Это локальная изменчивость, не Kp/Dst и не классификация бурь. Границы квартилей вычислены только по истинному ряду отдельно для каждой длительности, без использования ошибок моделей. Метки используют значения, скрытые от модели: они нужны для ретроспективного анализа, а не для выбора метода при реальном пропуске.</p>
<p>95% интервалы получены парным бутстрэпом с 2000 повторами: одновременно пересэмплируются пять сидов и двухнедельные временные блоки. Для основной семьи MAE построены приближённые одновременные интервалы по всем моделям, длительностям и четырём уровням активности. Зелёные строки подтверждают преимущество в этой семье. Для RMSE, верхних 10% и проверок чувствительности интервалы поточечные. Дополнительно проверены блоки по 28 дней и активность внутри самого скрытого пропуска; подробности в JSON. Зависимость соседних блоков и один год данных остаются ограничениями.</p>
<h2>Длительность пропуска</h2><img src="duration.png" alt="Ошибка по длительности">
<h2>Длительность и изменчивость поля</h2><img src="activity_heatmap.png" alt="Выигрыш по длительности и активности">
<h2>Почему определение активности существенно</h2><img src="activity_definition.png" alt="Сравнение двух определений активности"><p>На суточных пропусках UNet абсолютная MAE при разбиении по активности окна растёт с 2,56 до 10,36 нТл, хотя относительное преимущество перед сезонным методом растёт. Больший относительный выигрыш не означает меньшей ошибки или доказанного восстановления внезапных событий. Парные интервалы для разницы выигрыша между квартилями сохранены в activity_contrasts в JSON; эти дополнительные сравнения разведочные и без общей поправки.</p>
<h2>Все готовые архитектуры и варианты входа</h2><img src="all_models.png" alt="Все модели">
<p><a href="analysis.json">Полные показатели и проверки чувствительности (JSON)</a> · <a href="sources.json">Источники и время снимка</a> · <a href="window_errors.npz">Ошибки отдельных окон</a></p>
<h2>Таблица основной семьи сравнений</h2><input id="filter" placeholder="Фильтр: unet_F, moment_F, …"><div>'''+''.join(table)+'''</div><script>document.querySelector('#filter').oninput=e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reps',type=int,default=2000);p.add_argument('--render-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data,meta=collect(args.root,args.out)
    if args.render_only:
        result=json.loads((args.out/'analysis.json').read_text())
    else:
        result=analyze(data,meta,args.reps)
        (args.out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    figures(data,result,args.out);report(result,args.out)
    with zipfile.ZipFile(args.out/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['analysis.json','sources.json','window_errors.npz','report.html','duration.png','duration.pdf','activity_heatmap.png','activity_heatmap.pdf','all_models.png','all_models.pdf','activity_definition.png','activity_definition.pdf']:
            z.write(args.out/name,name)
        z.write(Path(__file__),'analyze_geomag_regimes.py')
    print('DONE '+str(args.out/'results.zip'),flush=True)

if __name__=='__main__':main()
