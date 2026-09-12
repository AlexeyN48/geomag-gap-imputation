r"""Семь метрик качества восстановления плюс бутстрэп-ДИ на P95/RMSE_P95.

Метрики MAE, RMSE, NSE, NMAE, MAPE, P95 считаются по точкам внутри дыры (пул
по всем окнам), RMSE_P95 — по-окнам:
  MAE       средняя абсолютная ошибка, нТл
  RMSE      корень из среднеквадратичной ошибки, нТл — штрафует крупные промахи
  NSE       1 − MSE/Var_w(true), где Var_w — ВНУТРИДЫРНАЯ дисперсия истины:
            каждое окно центрируется по среднему СВОЕЙ дыры, и только потом
            квадраты отклонений пулятся по всем окнам. 1 = идеал, 0 = не
            лучше горизонтальной линии на уровне дыры, <0 = хуже неё.
            NB: раньше знаменатель был дисперсией истины, слитой по всем
            окнам сразу, — то есть включал межоконный дрейф уровня поля
            (десятки нТл за год), который модель и не угадывает: уровень
            известен ей из контекста вокруг дыры. Такой NSE насыщался
            (≥0.95 у всех методов, 1.00 у LOCF на коротких дырах) и ничего
            не различал. На коротких дырах (≤240 мин) новый NSE у всех
            методов сильно отрицателен — внутри 5 минут истина меняется
            на доли нТл, и любая ошибка больше этого; содержательно
            сравнивать по нему стоит длины ≥720 мин.
  NMAE      MAE / sqrt(Var_w) — та же внутридырная нормировка, что у NSE, но
            через MAE: ошибка в долях изменчивости сигнала ВНУТРИ дыры (то,
            что модель угадывала), а не в долях годового разброса уровня
            (то, что ей известно из контекста). Раньше делилось на std по
            всем окнам сразу и не различало методы, как и старый NSE.
  MAPE      относительная ошибка по ЛОГАРИФМУ отношения |ln(pred/true)|×100% —
            не обычная |pred-true|/|true|: истина здесь — модуль поля, порядка
            5·10⁴ нТл и никогда не близка к нулю, но обычная процентная ошибка
            всё равно взрывается на редких точках, где предсказание уходит в
            отрицательные или близкие к нулю значения (модель не обязана знать,
            что F > 0). Логарифм отношения растёт только логарифмически, а не
            как 1/true, поэтому такие точки не рвут среднее по всей выборке.
            NB: поскольку истина всегда одного порядка (знаменатель почти
            константа), MAPE здесь по существу масштабированная MAE и НЕ несёт
            информации, независимой от неё, — метрика добавлена по запросу, а
            не потому что здесь есть подходящая для процентной ошибки шкала.
  P95       95-й перцентиль |ошибка| по всем точкам всех окон — хвост поточечно:
            редкая, но крупная ошибка внутри отдельных минут дыры
  RMSE_P95  95-й перцентиль RMSE, посчитанного ОТДЕЛЬНО для каждого окна —
            хвост по окнам целиком: насколько плохо выглядит худшее из окон,
            а не худшая отдельная точка (см. P95)

Плюс 95% доверительный интервал для MAE, RMSE, P95 и RMSE_P95 (…_CI_LO/HI) —
перцентильный БЛОЧНЫЙ бутстрэп по календарным НЕДЕЛЯМ (не по точкам и не по
окнам: точки внутри одной дыры зависимы, а окна в дампе перекрываются — на
длинных дырах каждая минута года лежит в ~8 дырах, и бутстрэп по окнам
считал бы один шторм восемью независимыми; см. docstring bootstrap_ci).
Показывает, насколько оценка держится на конкретном годе, а не только что
она такое. Блок берётся из поля L{L}_block дампа (bench_run.py); для старых
дампов без него позиции окон восстанавливаются по данным станции.

Запуск:  cd bench_arti/scripts
         python -u bench_metrics.py --pred dump_pchip_ARS_val.npz
         python -u bench_metrics.py --pred dump_saits_ARS_val.npz
         python -u bench_metrics.py --pred dump_saits_ARS_val.npz --no-ci  # без ДИ, быстрее
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

CI_M = 512      # размер ресэмпла (окон) для ЗАПАСНОГО бутстрэпа по окнам —
                # используется только если блоки недоступны (см. blocks_for)
CI_REPS = 200   # число бутстрэп-повторов на каждую (длина, subset) — компромисс
                # между устойчивостью границ ДИ и временем счёта по всем длинам
CI_SEED = 1234  # тот же SEED, что и в bench_common.py — воспроизводимость


def _resample_idx(n, blocks, rng, reps, m=CI_M):
    """reps наборов индексов окон для бутстрэпа.

    blocks — массив (n,) с номером блока (год·100 + неделя) каждого окна:
    на каждом повторе тянутся С ВОЗВРАЩЕНИЕМ целые блоки (столько, сколько
    их всего), и берутся ВСЕ окна вытянутых блоков. Окна одной недели
    уходят и приходят вместе, поэтому перекрытие дыр и общий шторм
    не раздувают эффективный объём выборки. blocks=None — запасной путь:
    m окон с возвращением, как раньше (ДИ на длинных дырах занижен)."""
    if blocks is None:
        for _ in range(reps):
            yield rng.integers(0, n, size=m)
        return
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    nb = len(groups)
    for _ in range(reps):
        pick = rng.integers(0, nb, size=nb)
        yield np.concatenate([groups[g] for g in pick])


def blocks_for(d, L, setname=None):
    """Номер блока (год·100 + календарная неделя начала дыры) для каждого
    окна длины L. Берётся из L{L}_block, если дамп сделан новым bench_run.py;
    иначе окна ищутся в рядах станции по первым K значениям истины (точное
    совпадение float32) и блок восстанавливается. Если найти не удалось —
    None, и бутстрэп идёт по окнам с предупреждением."""
    key = f"L{L}_block"
    if key in d:
        return d[key].astype(np.int64)
    import bench_common as C
    setname = setname or str(d["setname"])
    code, split = setname.split("_", 1)
    cache = blocks_for._cache.setdefault(setname, {})
    if "index" not in cache:
        from numpy.lib.stride_tricks import sliding_window_view
        K = 24
        index = {}
        for y, F in C.load_split(split, code=code).items():
            sw = sliding_window_view(F.astype(np.float32), K)
            for i in range(sw.shape[0]):
                index.setdefault(sw[i].tobytes(), (y, i))
        cache["index"], cache["K"] = index, K
    index, K = cache["index"], cache["K"]
    m = int(d["margin"])
    tr = d[f"L{L}_true"].astype(np.float32)
    out = np.full(tr.shape[0], -1, np.int64)
    for i, row in enumerate(tr):
        fin = np.flatnonzero(np.isfinite(row))
        if fin.size < K:
            continue
        j = int(fin[0])
        hit = index.get(row[j:j + K].tobytes())
        if hit is None:
            continue
        y, pos = hit
        out[i] = y * 100 + (pos - j + m) // C.BLOCK
    miss = int((out < 0).sum())
    if miss:
        print(f"  [блоки] L={L}: не найдено положение {miss} из {len(out)} окон — "
              f"бутстрэп по окнам (ДИ занижен); пересчитай дамп через bench_run.py")
        return None
    return out


blocks_for._cache = {}


def load(path):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    return np.load(p, allow_pickle=False)


def gap(d, L, key):
    m = int(d["margin"])
    return d[f"L{L}_{key}"][:, m:m + L].astype(np.float64)


def bootstrap_ci(err, se, blocks=None, m=CI_M, reps=CI_REPS, seed=CI_SEED):
    """95% перцентильный бутстрэп-ДИ для MAE, RMSE, P95(|ошибка|) и
    P95(RMSE по окнам) — НЕЗАВИСИМО для одного метода (не парный: см.
    paired_bootstrap_ci, если нужно убрать общий шум выборки при сравнении
    двух методов на одних и тех же окнах).

    Ресэмплинг БЛОЧНЫЙ, по календарным неделям (blocks — номер блока каждого
    окна, см. _resample_idx), а не по точкам и не по окнам. По точкам нельзя:
    точки одной дыры зависимы (один шторм даёт много плохих точек подряд).
    По окнам тоже нельзя: окна в дампе берутся случайно из одного года и
    перекрываются — при 1024 дырах по 4320 мин каждая минута года лежит в
    ~8 дырах, и бутстрэп по окнам считает один и тот же шторм восемью
    независимыми наблюдениями (проверено: на 4320 мин ДИ по окнам уже
    блочного в ~1.5–2 раза, а на ≤60 мин они совпадают — там дыры не
    пересекаются). Границы ДИ — 2.5-й и 97.5-й перцентиль по reps повторам.

    err — |pred-true| поточечно (n_окон, L); se — (pred-true)^2, той же формы.
    Возвращает (mae_lo, mae_hi, rmse_lo, rmse_hi,
                p95_lo, p95_hi, rmse_p95_lo, rmse_p95_hi)."""
    n = err.shape[0]
    rng = np.random.default_rng(seed)
    mae_boot = np.empty(reps)
    rmse_boot = np.empty(reps)
    p95_boot = np.empty(reps)
    rmse_p95_boot = np.empty(reps)
    for i, idx in enumerate(_resample_idx(n, blocks, rng, reps, m)):
        mae_boot[i] = np.nanmean(err[idx])
        rmse_boot[i] = np.sqrt(np.nanmean(se[idx]))
        p95_boot[i] = np.nanpercentile(err[idx], 95)
        win_rmse = np.sqrt(np.nanmean(se[idx], axis=1))
        rmse_p95_boot[i] = np.nanpercentile(win_rmse, 95)
    mae_lo, mae_hi = np.percentile(mae_boot, [2.5, 97.5])
    rmse_lo, rmse_hi = np.percentile(rmse_boot, [2.5, 97.5])
    p95_lo, p95_hi = np.percentile(p95_boot, [2.5, 97.5])
    r95_lo, r95_hi = np.percentile(rmse_p95_boot, [2.5, 97.5])
    return (float(mae_lo), float(mae_hi), float(rmse_lo), float(rmse_hi),
            float(p95_lo), float(p95_hi), float(r95_lo), float(r95_hi))


def paired_bootstrap_ci(err_a, se_a, err_b, se_b, key, blocks=None, m=CI_M,
                         reps=CI_REPS, seed=CI_SEED):
    """Парный 95% бутстрэп-ДИ на разницу (a − b) метрики key ('MAE', 'RMSE',
    'P95' или 'RMSE_P95') между двумя методами на ОДНИХ И ТЕХ ЖЕ окнах.

    В отличие от bootstrap_ci (независимый ДИ одного метода), здесь на
    каждом повторе ОБА метода ресэмплируются по ОДНОМУ И ТОМУ ЖЕ набору
    индексов окон — это убирает из разницы общий шум выборки (окно с
    сильным штормом одинаково утяжелит обоих, и в разности сокращается).
    Ресэмплинг блочный по неделям (blocks), по той же причине, что и в
    bootstrap_ci: окна перекрываются, и по-оконный ДИ на длинных дырах
    занижен — мелкие «значимые» разницы между сетями (0.03 нТл на 4320 мин)
    при блочном бутстрэпе значимость теряют.
    Независимые интервалы шире и почти всегда пересекаются даже там, где
    парная разница значима — они годятся только для грубой прикидки
    (непересечение надёжно подтверждает различие, а вот пересечение НЕ
    доказывает его отсутствие).

    err_a/se_a и err_b/se_b — |pred-true| и (pred-true)^2 поточечно
    (n_окон, L) для методов a и b на одной и той же длине L; число окон у
    a и b может отличаться (наборы урезаются до общего n).

    Возвращает (mean_diff, lo, hi, significant) для метрики key."""
    n = min(err_a.shape[0], err_b.shape[0])
    rng = np.random.default_rng(seed)
    if blocks is not None:
        blocks = blocks[:n]
    diffs = np.empty(reps)
    for i, idx in enumerate(_resample_idx(n, blocks, rng, reps, m)):
        if key == "MAE":
            diffs[i] = np.nanmean(err_a[idx]) - np.nanmean(err_b[idx])
        elif key == "RMSE":
            diffs[i] = (np.sqrt(np.nanmean(se_a[idx]))
                        - np.sqrt(np.nanmean(se_b[idx])))
        elif key == "P95":
            diffs[i] = (np.nanpercentile(err_a[idx], 95)
                        - np.nanpercentile(err_b[idx], 95))
        elif key == "RMSE_P95":
            wa = np.sqrt(np.nanmean(se_a[idx], axis=1))
            wb = np.sqrt(np.nanmean(se_b[idx], axis=1))
            diffs[i] = np.nanpercentile(wa, 95) - np.nanpercentile(wb, 95)
        else:
            raise ValueError(f"paired_bootstrap_ci: неизвестный key {key!r}, "
                              f"ожидается 'MAE', 'RMSE', 'P95' или 'RMSE_P95'")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    significant = bool((lo > 0) == (hi > 0))
    return float(diffs.mean()), float(lo), float(hi), significant


def mape_log(pred, true, eps=1e-6):
    """MAPE по логарифму отношения: 100 × mean(|ln(pred/true)|).

    Обычная |pred-true|/|true| взрывается, если знаменатель (здесь —
    предсказание МОДЕЛИ, не истина: F физически не бывает отрицательным или
    нулевым, но сеть об этом не знает и иногда выдаёт значение около нуля на
    редких плохих точках) оказывается близко к нулю. Логарифм отношения на
    тех же точках растёт как ln(1/eps), а не как 1/eps — единичные выбросы
    не разносят среднее по всей выборке. eps защищает сам логарифм от minus
    inf, если предсказание всё же ушло в ноль или в минус."""
    r = np.clip(pred, eps, None) / np.clip(true, eps, None)
    return float(np.nanmean(np.abs(np.log(r))) * 100.0)


def basic_metrics(t, pm, sel, ci=True, blocks=None):
    """MAE, RMSE, NSE, NMAE, MAPE, P95, RMSE_P95 — все на выбранном подмножестве окон.

    MAE/RMSE/NSE/NMAE/MAPE/P95 — пул по всем точкам дыры (не среднее
    по-окнам). NSE и NMAE нормируются на ВНУТРИДЫРНУЮ дисперсию: истина
    каждого окна центрируется по своему среднему до пула (см. докстринг
    модуля, почему общая дисперсия по всем окнам здесь не годится). MAPE
    делит на абсолютный уровень поля (~5.7e4 нТл) и потому вырождается
    (знаменатель почти константа, см. mape_log).

    RMSE_P95 — другое: RMSE считается ОТДЕЛЬНО для каждого окна (ось точек
    внутри окна), и только потом берётся перцентиль ПО ОКНАМ. Это отвечает
    на другой вопрос, чем P95: не «какая ошибка типична для худшей точки»,
    а «насколько плохо целиком выглядит худшее из окон» — одно окно с
    равномерно средними ошибками и окно с единственным острым выбросом дают
    разный RMSE_P95 даже при одинаковом P95 по точкам."""
    tt, pp = t[sel], pm[sel]
    err = np.abs(pp - tt)
    mae = float(np.nanmean(err))
    se = (pp - tt) ** 2
    mse = float(np.nanmean(se))
    rmse = float(np.sqrt(mse))
    # NSE и NMAE: дисперсия истины ВНУТРИ дыры — каждое окно центрируется по
    # среднему своей дыры (axis=1), иначе в знаменатель попадает межоконный
    # дрейф уровня поля, известный модели из контекста, и метрики насыщаются
    tc = tt - np.nanmean(tt, axis=1, keepdims=True)
    var_w = float(np.nanmean(tc ** 2))
    nse = (1.0 - mse / var_w) if var_w > 0 else np.nan
    nmae = (mae / np.sqrt(var_w)) if var_w > 0 else np.nan
    mape = mape_log(pp, tt)
    p95 = float(np.nanpercentile(err, 95))
    win_rmse = np.sqrt(np.nanmean(se, axis=1))
    rmse_p95 = float(np.nanpercentile(win_rmse, 95))
    out = {"MAE": mae, "RMSE": rmse, "NSE": nse, "NMAE": nmae, "MAPE": mape,
           "P95": p95, "RMSE_P95": rmse_p95}
    if ci:
        (mae_lo, mae_hi, rmse_lo, rmse_hi,
         p95_lo, p95_hi, r95_lo, r95_hi) = bootstrap_ci(
            err, se, blocks=None if blocks is None else blocks[sel])
        out["MAE_CI_LO"] = mae_lo
        out["MAE_CI_HI"] = mae_hi
        out["RMSE_CI_LO"] = rmse_lo
        out["RMSE_CI_HI"] = rmse_hi
        out["P95_CI_LO"] = p95_lo
        out["P95_CI_HI"] = p95_hi
        out["RMSE_P95_CI_LO"] = r95_lo
        out["RMSE_P95_CI_HI"] = r95_hi
    return out


def subsets(act):
    # порог активности берётся ВНУТРИ длины: иначе длины несравнимы между собой
    thr = np.median(act)
    hi = act > thr
    return {"all": np.ones(len(act), bool), "act": hi, "qui": ~hi}


# --------------------------------------------------------------- прогон
def run(args):
    d = load(args.pred)
    lengths = [int(x) for x in d["lengths"]]
    method = str(d["method"])
    setname = str(d["setname"])

    print(f"метод: {method}   набор: {setname}\n")

    rows = []
    for L in lengths:
        if f"L{L}_true" not in d:
            continue
        t = gap(d, L, "true")
        pm = gap(d, L, "pred")
        act = d[f"L{L}_act"]
        blk = None if args.no_ci else blocks_for(d, L, setname)

        for sub, sel in subsets(act).items():
            v = basic_metrics(t, pm, sel, ci=not args.no_ci, blocks=blk)
            for k, val in v.items():
                rows.append((k, L, sub, val))

    idx = {(k, L, s): v for k, L, s, v in rows}
    keys = ["MAE", "RMSE", "NSE", "NMAE", "MAPE", "P95", "RMSE_P95"]
    print(f"{'len':>6} " + " ".join(f"{h:>9}" for h in keys))
    print("-" * (7 + 10 * len(keys)))
    for L in lengths:
        cells = []
        for k in keys:
            val = idx.get((k, L, "all"), np.nan)
            cells.append("        -" if not np.isfinite(val) else f"{val:9.3f}")
        print(f"{L:>6} " + " ".join(cells))

    if not args.no_ci:
        print(f"\n95% ДИ (блочный бутстрэп по неделям, повторов={CI_REPS}), subset=all:")
        print(f"{'len':>6} {'MAE':>9} {'MAE 95%ДИ':>18} {'RMSE':>9} {'RMSE 95%ДИ':>18} "
              f"{'P95':>9} {'P95 95%ДИ':>18} {'RMSE_P95':>9} {'RMSE_P95 95%ДИ':>18}")
        for L in lengths:
            mae = idx.get(("MAE", L, "all"), np.nan)
            rmse = idx.get(("RMSE", L, "all"), np.nan)
            p95 = idx.get(("P95", L, "all"), np.nan)
            r95 = idx.get(("RMSE_P95", L, "all"), np.nan)
            mlo, mhi = idx.get(("MAE_CI_LO", L, "all"), np.nan), idx.get(("MAE_CI_HI", L, "all"), np.nan)
            rmlo, rmhi = idx.get(("RMSE_CI_LO", L, "all"), np.nan), idx.get(("RMSE_CI_HI", L, "all"), np.nan)
            lo, hi = idx.get(("P95_CI_LO", L, "all"), np.nan), idx.get(("P95_CI_HI", L, "all"), np.nan)
            rlo, rhi = idx.get(("RMSE_P95_CI_LO", L, "all"), np.nan), idx.get(("RMSE_P95_CI_HI", L, "all"), np.nan)
            print(f"{L:>6} {mae:>9.3f} {f'[{mlo:.2f}, {mhi:.2f}]':>18} "
                  f"{rmse:>9.3f} {f'[{rmlo:.2f}, {rmhi:.2f}]':>18} "
                  f"{p95:>9.3f} {f'[{lo:.2f}, {hi:.2f}]':>18} "
                  f"{r95:>9.3f} {f'[{rlo:.2f}, {rhi:.2f}]':>18}")

    tag = f"{method}_{setname}"
    out = args.out or os.path.join(DATA, f"metrics_{tag}.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("metric,gap_len,subset,value\n")
        for k, L, s, v in rows:
            f.write(f"{k},{L},{s},{v:.6g}\n")
    print(f"\nсохранено: {out}  ({len(rows)} строк)")

    if not args.no_fig:
        figure(idx, lengths, method, setname, tag)

    if args.base:
        run_paired(args, d, method, setname, lengths)
    if args.vs_station:
        run_cross_station(args, d, method, setname, lengths)


def run_cross_station(args, d, method, setname, lengths):
    """Парный бутстрэп-ДИ на P95 между ДВУМЯ СТАНЦИЯМИ для ОДНОГО И ТОГО ЖЕ
    метода (--vs-station), в отличие от run_paired (--base), который
    сравнивает два метода на ОДНОЙ станции.

    Проверка совпадения окон здесь принципиально другая: значения истины
    у разных станций НИКОГДА не совпадут (разное поле), поэтому сверяем
    ПОЗИЦИЮ дыры (gap0) — она обязана совпасть побитово, если дампы
    сделаны через bench_run.py --align-codes (см. gather_aligned в
    bench_common.py). Без --align-codes позиции почти наверняка разойдутся
    (проверено эмпирически: 93-94% совпадения на большинстве длин, 3% на
    4320 мин из-за расхождения ГСЧ по станциям) — тогда парность неверна,
    и скрипт отказывается считать её, а не молча даёт неверный ДИ."""
    b = load(args.vs_station)
    other_setname = str(b["setname"])
    code_a = setname.split("_")[0]
    code_b = other_setname.split("_")[0]
    print(f"\n--- межстанционное сравнение (P95): {setname} против "
          f"{other_setname}, метод {method} "
          f"(блочный бутстрэп по неделям, повторов={CI_REPS}) ---")
    print(f"{'len':>6} {'P95 diff':>9} {'95% ДИ':>18} {'значимо':>8}  окон  вердикт")

    prows = []
    for L in lengths:
        if f"L{L}_true" not in b:
            continue
        t, pm = gap(d, L, "true"), gap(d, L, "pred")
        tb, pb = gap(b, L, "true"), gap(b, L, "pred")
        g0, g0b = d[f"L{L}_gap0"], b[f"L{L}_gap0"]
        n = min(t.shape[0], tb.shape[0])
        if not np.array_equal(g0[:n], g0b[:n]):
            miss = float((g0[:n] != g0b[:n]).mean())
            print(f"{L:>6}  окна не выровнены по времени ({miss:.0%} "
                  f"расходятся) — пересчитать через bench_run.py "
                  f"--align-codes, пропуск")
            continue
        err_a, se_a = np.abs(pm[:n] - t[:n]), (pm[:n] - t[:n]) ** 2
        err_b, se_b = np.abs(pb[:n] - tb[:n]), (pb[:n] - tb[:n]) ** 2
        # окна выровнены по времени, поэтому блоки у обеих станций одни
        m_, lo, hi, sig = paired_bootstrap_ci(err_a, se_a, err_b, se_b, "P95",
                                              blocks=blocks_for(d, L, setname))
        # разница = a − b (P95 ошибки; меньше = точнее): значимо и <0 -> a
        # точнее, значимо и >0 -> b точнее, ДИ накрывает 0 -> не отличается
        if not sig:
            verdict = "не отличается"
        elif m_ < 0:
            verdict = f"{code_a} точнее"
        else:
            verdict = f"{code_b} точнее"
        prows.append(("P95", L, m_, lo, hi, sig, n, verdict))
        print(f"{L:>6} {m_:>+9.3f} {f'[{lo:+.2f}, {hi:+.2f}]':>18} "
              f"{'да' if sig else 'нет':>8}  {n}  {verdict}")

    pout = os.path.join(DATA, f"metrics_{method}_{setname}_vs_{other_setname}_P95.csv")
    with open(pout, "w", encoding="utf-8") as f:
        f.write("metric,gap_len,mean_diff,ci_lo,ci_hi,significant,n_windows,verdict\n")
        for key, L, m_, lo, hi, sig, n, verdict in prows:
            f.write(f"{key},{L},{m_:.6g},{lo:.6g},{hi:.6g},{sig},{n},{verdict}\n")
    print(f"\nсохранено: {pout}  ({len(prows)} строк)  "
          f"(разница = {setname} минус {other_setname}; "
          f"отрицательная -> {setname} лучше)")


def run_paired(args, d, method, setname, lengths):
    """Парное сравнение с --base: значим ли метод против конкретного
    другого метода на каждой длине (subset=all), а не только его
    независимый ДИ (см. paired_bootstrap_ci — почему это другой вопрос)."""
    b = load(args.base)
    base_method = str(b["method"])
    print(f"\n--- парное сравнение: {method} против {base_method} "
          f"(блочный бутстрэп по неделям, повторов={CI_REPS}) ---")
    print(f"{'len':>6} {'metric':>9} {'diff':>9} {'95% ДИ':>18} {'значимо':>8}")

    prows = []
    for L in lengths:
        if f"L{L}_true" not in b:
            continue
        t, pm = gap(d, L, "true"), gap(d, L, "pred")
        tb, pb = gap(b, L, "true"), gap(b, L, "pred")
        n = min(t.shape[0], tb.shape[0])
        if not np.allclose(t[:n], tb[:n], equal_nan=True):
            print(f"{L:>6}  окна {method} и {base_method} не совпадают — пропуск")
            continue
        err_a, se_a = np.abs(pm - t), (pm - t) ** 2
        err_b, se_b = np.abs(pb - tb), (pb - tb) ** 2
        blk = blocks_for(d, L, setname)
        for key in ("MAE", "RMSE", "P95", "RMSE_P95"):
            m_, lo, hi, sig = paired_bootstrap_ci(err_a, se_a, err_b, se_b, key,
                                                  blocks=blk)
            prows.append((key, L, m_, lo, hi, sig))
            print(f"{L:>6} {key:>9} {m_:>+9.3f} {f'[{lo:+.2f}, {hi:+.2f}]':>18} "
                  f"{'да' if sig else 'нет':>8}")

    pout = os.path.join(DATA, f"metrics_{method}_vs_{base_method}_{setname}.csv")
    with open(pout, "w", encoding="utf-8") as f:
        f.write("metric,gap_len,mean_diff,ci_lo,ci_hi,significant\n")
        for key, L, m_, lo, hi, sig in prows:
            f.write(f"{key},{L},{m_:.6g},{lo:.6g},{hi:.6g},{sig}\n")
    print(f"\nсохранено: {pout}  ({len(prows)} строк)  "
          f"(разница = {method} минус {base_method}; отрицательная -> {method} лучше)")


def figure(idx, lengths, method, setname, tag):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))

    ax = axes[0]
    xs = [L for L in lengths if ("MAE", L, "all") in idx]
    ax.plot(xs, [idx[("MAE", L, "all")] for L in xs], "-o", color="#1f77b4", label="MAE")
    ax.plot(xs, [idx[("RMSE", L, "all")] for L in xs], "-o", color="#d62728", label="RMSE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(lengths); ax.set_xticklabels(lengths)
    ax.set_xlabel("длина пропуска, мин"); ax.set_ylabel("ошибка, нТл")
    ax.set_title("MAE и RMSE"); ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)

    ax = axes[1]
    xs = [L for L in lengths if ("NSE", L, "all") in idx]
    l1 = ax.plot(xs, [idx[("NSE", L, "all")] for L in xs], "-o", color="#2ca02c", label="NSE")
    ax.axhline(0.0, color="k", lw=1, ls=":", alpha=0.4)
    ax.set_xscale("log"); ax.set_xticks(lengths); ax.set_xticklabels(lengths)
    ax.set_xlabel("длина пропуска, мин"); ax.set_ylabel("NSE  (1 = идеал, 0 = как уровень дыры)")
    ax.grid(alpha=0.25)

    # MAPE — на своей оси: у F порядок ~5·10⁴ нТл, поэтому MAPE численно
    # НА ПОРЯДКИ меньше NSE (см. докстринг mape_log) и на общей оси выглядел
    # бы плоской нулевой линией.
    axm = ax.twinx()
    xs2 = [L for L in lengths if ("MAPE", L, "all") in idx]
    l2 = axm.plot(xs2, [idx[("MAPE", L, "all")] for L in xs2], "-o", color="#9467bd", label="MAPE")
    axm.set_yscale("log")
    axm.set_ylabel("MAPE, % (по логарифму отношения)")
    ax.set_title("NSE и MAPE"); ax.legend(l1 + l2, [ln.get_label() for ln in l1 + l2], fontsize=8)

    fig.suptitle(f"{method} — {setname}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, f"fig_bench_metrics_{tag}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"сохранён {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="дамп оцениваемого метода")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--no-ci", action="store_true",
                     help="не считать бутстрэп-ДИ для P95/RMSE_P95 (быстрее)")
    ap.add_argument("--base", default=None,
                     help="дамп второго метода — парное сравнение P95/RMSE_P95 "
                          "с ДИ на РАЗНИЦУ (значимее независимых ДИ; см. "
                          "paired_bootstrap_ci), пишется отдельным файлом")
    ap.add_argument("--vs-station", default=None,
                     help="дамп того же метода на ДРУГОЙ станции (нужен "
                          "--pred и --vs-station из bench_run.py "
                          "--align-codes, иначе окна не выровнены) — парный "
                          "ДИ на разницу P95 между станциями")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
