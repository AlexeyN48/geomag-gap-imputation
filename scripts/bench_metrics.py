r"""Пять метрик качества восстановления.

Метрики (все посчитаны по точкам внутри дыры, не по-окнам):
  MAE   средняя абсолютная ошибка, нТл
  RMSE  корень из среднеквадратичной ошибки, нТл — штрафует крупные промахи
  NSE   1 − MSE/Var(true): 1 = идеал, 0 = не лучше среднего, <0 = хуже среднего
  NMAE  MAE / std(true в дыре) — ошибка в долях реальной изменчивости сигнала
  MASE  MAE / MAE(LOCF на той же длине дыры L) — <1 лучше наивного, >1 хуже

Запуск:  cd bench_arti/scripts
         python -u bench_metrics.py --pred dump_pchip_ARS_val.npz
         python -u bench_metrics.py --pred dump_saits_ARS_val.npz
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def load(path):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    return np.load(p, allow_pickle=False)


def gap(d, L, key):
    m = int(d["margin"])
    return d[f"L{L}_{key}"][:, m:m + L].astype(np.float64)


def win_mae(pred, true):
    return np.nanmean(np.abs(pred - true), axis=1)


def basic_metrics(t, pm, sel):
    """MAE, RMSE, NSE, NMAE — пул по всем точкам дыры в выбранных окнах (не
    среднее по-окнам): NSE и NMAE сравниваются с честной изменчивостью истины
    на этом подмножестве, а не с абсолютным уровнем поля (~5.7e4 нТл), где
    относительные метрики вроде MAPE вырождаются (знаменатель почти константа)."""
    tt, pp = t[sel], pm[sel]
    mae = float(np.nanmean(np.abs(pp - tt)))
    mse = float(np.nanmean((pp - tt) ** 2))
    rmse = float(np.sqrt(mse))
    std_y = float(np.nanstd(tt))
    var_y = std_y ** 2
    nse = (1.0 - mse / var_y) if var_y > 0 else np.nan
    nmae = (mae / std_y) if std_y > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "NSE": nse, "NMAE": nmae}


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
            v = basic_metrics(t, pm, sel)
            if e_locf is not None:
                v["MASE"] = v["MAE"] / float(e_locf[sel].mean())
            for k, val in v.items():
                rows.append((k, L, sub, val))

    idx = {(k, L, s): v for k, L, s, v in rows}
    keys = ["MAE", "RMSE", "NSE", "NMAE", "MASE"]
    print(f"{'len':>6} " + " ".join(f"{h:>9}" for h in keys))
    print("-" * (7 + 10 * len(keys)))
    for L in lengths:
        cells = []
        for k in keys:
            val = idx.get((k, L, "all"), np.nan)
            cells.append("        -" if not np.isfinite(val) else f"{val:9.3f}")
        print(f"{L:>6} " + " ".join(cells))

    tag = f"{method}_{setname}"
    out = args.out or os.path.join(DATA, f"metrics_{tag}.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("metric,gap_len,subset,value\n")
        for k, L, s, v in rows:
            f.write(f"{k},{L},{s},{v:.6g}\n")
    print(f"\nсохранено: {out}  ({len(rows)} строк)")

    if not args.no_fig:
        figure(idx, lengths, method, setname, tag)


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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
