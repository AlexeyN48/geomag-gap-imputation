r"""Семь метрик качества восстановления плюс бутстрэп-ДИ на P95/RMSE_P95.

Метрики MAE, RMSE, NSE, NMAE, P95 считаются по точкам внутри дыры (пул по
всем окнам), MASE и RMSE_P95 — по-окнам:
  MAE       средняя абсолютная ошибка, нТл
  RMSE      корень из среднеквадратичной ошибки, нТл — штрафует крупные промахи
  NSE       1 − MSE/Var(true): 1 = идеал, 0 = не лучше среднего, <0 = хуже среднего
  NMAE      MAE / std(true в дыре) — ошибка в долях реальной изменчивости сигнала
  MASE      MAE / MAE(LOCF на той же длине дыры L) — <1 лучше наивного, >1 хуже
  P95       95-й перцентиль |ошибка| по всем точкам всех окон — хвост поточечно:
            редкая, но крупная ошибка внутри отдельных минут дыры
  RMSE_P95  95-й перцентиль RMSE, посчитанного ОТДЕЛЬНО для каждого окна —
            хвост по окнам целиком: насколько плохо выглядит худшее из окон,
            а не худшая отдельная точка (см. P95)

Плюс 95% доверительный интервал для P95 и RMSE_P95 (P95_CI_LO/HI,
RMSE_P95_CI_LO/HI) — перцентильный бутстрэп по ОКНАМ (не по точкам: точки
внутри одной дыры зависимы, см. docstring bootstrap_ci). Показывает, насколько
сама оценка P95/RMSE_P95 держится на конкретной выборке из 1024 окон, а не
только что она такое.

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

CI_M = 512      # размер бутстрэп-ресэмпла (окон), берётся С ВОЗВРАЩЕНИЕМ —
                # меньше n=1024 специально: вдвое быстрее, и даёт представление
                # о разбросе оценки, а не только повторяет исходный размер
CI_REPS = 200   # число бутстрэп-повторов на каждую (длина, subset) — компромисс
                # между устойчивостью границ ДИ и временем счёта по всем длинам
CI_SEED = 1234  # тот же SEED, что и в bench_common.py — воспроизводимость


def load(path):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    return np.load(p, allow_pickle=False)


def gap(d, L, key):
    m = int(d["margin"])
    return d[f"L{L}_{key}"][:, m:m + L].astype(np.float64)


def win_mae(pred, true):
    return np.nanmean(np.abs(pred - true), axis=1)


def bootstrap_ci(err, se, m=CI_M, reps=CI_REPS, seed=CI_SEED):
    """95% перцентильный бутстрэп-ДИ для P95(|ошибка|) и P95(RMSE по окнам).

    Ресэмплинг ПО ОКНАМ (строкам err/se), не по отдельным точкам: точки
    внутри одной дыры зависимы (один шторм даёт много плохих точек подряд),
    и бутстрэп по точкам занизил бы интервал в разы. На каждом повторе
    берутся m окон С ВОЗВРАЩЕНИЕМ из имеющихся n (m может быть меньше,
    равно или больше n — это просто количество тянущихся окон), метрика
    пересчитывается заново; границы ДИ — 2.5-й и 97.5-й перцентиль по
    получившимся reps значениям.

    err — |pred-true| поточечно (n_окон, L); se — (pred-true)^2, той же формы.
    Возвращает (p95_lo, p95_hi, rmse_p95_lo, rmse_p95_hi)."""
    n = err.shape[0]
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, n, size=(reps, m))
    p95_boot = np.empty(reps)
    rmse_p95_boot = np.empty(reps)
    for i in range(reps):
        idx = idxs[i]
        p95_boot[i] = np.nanpercentile(err[idx], 95)
        win_rmse = np.sqrt(np.nanmean(se[idx], axis=1))
        rmse_p95_boot[i] = np.nanpercentile(win_rmse, 95)
    p95_lo, p95_hi = np.percentile(p95_boot, [2.5, 97.5])
    r95_lo, r95_hi = np.percentile(rmse_p95_boot, [2.5, 97.5])
    return float(p95_lo), float(p95_hi), float(r95_lo), float(r95_hi)


def paired_bootstrap_ci(err_a, se_a, err_b, se_b, key, m=CI_M, reps=CI_REPS,
                         seed=CI_SEED):
    """Парный 95% бутстрэп-ДИ на разницу (a − b) метрики key ('P95' или
    'RMSE_P95') между двумя методами на ОДНИХ И ТЕХ ЖЕ окнах.

    В отличие от bootstrap_ci (независимый ДИ одного метода), здесь на
    каждом повторе ОБА метода ресэмплируются по ОДНОМУ И ТОМУ ЖЕ набору
    индексов окон — это убирает из разницы общий шум выборки (окно с
    сильным штормом одинаково утяжелит обоих, и в разности сокращается).
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
    idxs = rng.integers(0, n, size=(reps, m))
    diffs = np.empty(reps)
    for i in range(reps):
        idx = idxs[i]
        if key == "P95":
            diffs[i] = (np.nanpercentile(err_a[idx], 95)
                        - np.nanpercentile(err_b[idx], 95))
        elif key == "RMSE_P95":
            wa = np.sqrt(np.nanmean(se_a[idx], axis=1))
            wb = np.sqrt(np.nanmean(se_b[idx], axis=1))
            diffs[i] = np.nanpercentile(wa, 95) - np.nanpercentile(wb, 95)
        else:
            raise ValueError(f"paired_bootstrap_ci: неизвестный key {key!r}, "
                              f"ожидается 'P95' или 'RMSE_P95'")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    significant = bool((lo > 0) == (hi > 0))
    return float(diffs.mean()), float(lo), float(hi), significant


def basic_metrics(t, pm, sel, ci=True):
    """MAE, RMSE, NSE, NMAE, P95, RMSE_P95 — все на выбранном подмножестве окон.

    MAE/RMSE/NSE/NMAE/P95 — пул по всем точкам дыры (не среднее по-окнам):
    NSE и NMAE сравниваются с честной изменчивостью истины на этом
    подмножестве, а не с абсолютным уровнем поля (~5.7e4 нТл), где
    относительные метрики вроде MAPE вырождаются (знаменатель почти константа).

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
    std_y = float(np.nanstd(tt))
    var_y = std_y ** 2
    nse = (1.0 - mse / var_y) if var_y > 0 else np.nan
    nmae = (mae / std_y) if std_y > 0 else np.nan
    p95 = float(np.nanpercentile(err, 95))
    win_rmse = np.sqrt(np.nanmean(se, axis=1))
    rmse_p95 = float(np.nanpercentile(win_rmse, 95))
    out = {"MAE": mae, "RMSE": rmse, "NSE": nse, "NMAE": nmae,
           "P95": p95, "RMSE_P95": rmse_p95}
    if ci:
        p95_lo, p95_hi, r95_lo, r95_hi = bootstrap_ci(err, se)
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


def locf_mae(locf, L, t):
    """MAE наивного LOCF на дыре ТОЙ ЖЕ длины L, что и оцениваемый метод —
    знаменатель MASE. Если сравнить с одношаговым (1 мин) наивным прогнозом,
    число растёт с L искусственно: одношаговый прогноз никогда не пытался
    предсказывать на часы/сутки вперёд, поэтому сравнение с ним на длинных
    дырах некорректно занижает знаменатель."""
    if locf is None or f"L{L}_true" not in locf:
        return None
    n = t.shape[0]
    tl = gap(locf, L, "true")
    if tl.shape[0] < n or not np.allclose(t, tl[:n], equal_nan=True):
        return None
    return win_mae(gap(locf, L, "pred")[:n], t)


# --------------------------------------------------------------- прогон
def run(args):
    d = load(args.pred)
    lengths = [int(x) for x in d["lengths"]]
    method = str(d["method"])
    setname = str(d["setname"])

    print(f"метод: {method}   набор: {setname}\n")

    locf_path = os.path.join(DATA, f"dump_locf_{setname}.npz")
    locf = load(locf_path) if os.path.exists(locf_path) else None

    rows = []
    for L in lengths:
        if f"L{L}_true" not in d:
            continue
        t = gap(d, L, "true")
        pm = gap(d, L, "pred")
        act = d[f"L{L}_act"]
        e_locf = locf_mae(locf, L, t)

        for sub, sel in subsets(act).items():
            v = basic_metrics(t, pm, sel, ci=not args.no_ci)
            if e_locf is not None:
                v["MASE"] = v["MAE"] / float(e_locf[sel].mean())
            for k, val in v.items():
                rows.append((k, L, sub, val))

    idx = {(k, L, s): v for k, L, s, v in rows}
    keys = ["MAE", "RMSE", "NSE", "NMAE", "MASE", "P95", "RMSE_P95"]
    print(f"{'len':>6} " + " ".join(f"{h:>9}" for h in keys))
    print("-" * (7 + 10 * len(keys)))
    for L in lengths:
        cells = []
        for k in keys:
            val = idx.get((k, L, "all"), np.nan)
            cells.append("        -" if not np.isfinite(val) else f"{val:9.3f}")
        print(f"{L:>6} " + " ".join(cells))

    if not args.no_ci:
        print(f"\n95% ДИ (бутстрэп по окнам, m={CI_M}, повторов={CI_REPS}), subset=all:")
        print(f"{'len':>6} {'P95':>9} {'P95 95%ДИ':>18} {'RMSE_P95':>9} {'RMSE_P95 95%ДИ':>18}")
        for L in lengths:
            p95 = idx.get(("P95", L, "all"), np.nan)
            r95 = idx.get(("RMSE_P95", L, "all"), np.nan)
            lo, hi = idx.get(("P95_CI_LO", L, "all"), np.nan), idx.get(("P95_CI_HI", L, "all"), np.nan)
            rlo, rhi = idx.get(("RMSE_P95_CI_LO", L, "all"), np.nan), idx.get(("RMSE_P95_CI_HI", L, "all"), np.nan)
            print(f"{L:>6} {p95:>9.3f} {f'[{lo:.2f}, {hi:.2f}]':>18} "
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


def run_paired(args, d, method, setname, lengths):
    """Парное сравнение с --base: значим ли метод против конкретного
    другого метода на каждой длине (subset=all), а не только его
    независимый ДИ (см. paired_bootstrap_ci — почему это другой вопрос)."""
    b = load(args.base)
    base_method = str(b["method"])
    print(f"\n--- парное сравнение: {method} против {base_method} "
          f"(бутстрэп по окнам, m={CI_M}, повторов={CI_REPS}) ---")
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
        for key in ("P95", "RMSE_P95"):
            m_, lo, hi, sig = paired_bootstrap_ci(err_a, se_a, err_b, se_b, key)
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
    ax.plot(xs, [idx[("NSE", L, "all")] for L in xs], "-o", color="#2ca02c", label="NSE")
    xs2 = [L for L in lengths if ("MASE", L, "all") in idx]
    # меньше 2 точек — одна точка не сравнима по масштабу с NSE и ломает ось,
    # обычно значит, что окна метода не совпали с LOCF-дампом почти нигде
    if len(xs2) >= 2:
        ax.plot(xs2, [idx[("MASE", L, "all")] for L in xs2], "-o", color="#9467bd", label="MASE")
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
    ax.axhline(0.0, color="k", lw=1, ls=":", alpha=0.4)
    ax.set_xscale("log"); ax.set_xticks(lengths); ax.set_xticklabels(lengths)
    ax.set_xlabel("длина пропуска, мин"); ax.set_ylabel("безразмерная")
    ax.set_title("NSE и MASE"); ax.grid(alpha=0.25); ax.legend(fontsize=8)

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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
