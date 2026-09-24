r"""Сравнение методов по режимам длины дыры: Score, ранги, Фридман + Немени (CD).

Режимы (по длине дыры, мин):        короткие ≤120 | средние 120..1000 | длинные >1000
                                  {5,15,60,120}   | {240,480,720,1000} | {1440,2160,2880,4320}
Длины, которых в дампе нет (480/2160 до дополнения), пропускаются — режим
считается по имеющимся.

Что считается для каждого метода на одном наборе (код станции + сплит):
  MAE_L      — MAE по точкам дыры на каждой длине (как в bench_metrics);
  MAE_режим  — ГЕОМЕТРИЧЕСКОЕ среднее MAE_L по длинам режима: длины внутри
               режима отличаются по ошибке в разы (0.3 нТл на 5 мин против
               2.5 на 120), арифметическое среднее считало бы только самую
               длинную; геометрическое даёт каждой длине равный вес;
  Score      — 1 − MAE/MAE_PCHIP: по длинам — из тех же дампов, по режимам —
               через геометрические средние (1 − GM_метод/GM_pchip). 1 = идеал,
               0 = как интерполяция по краям дыры, <0 = хуже неё.

Ранжирование (Demšar 2006). Единица сравнения — ячейка (неделя, длина):
для каждой календарной недели набора и каждой длины берётся MAE метода по
окнам этой недели, и методы получают места 1..k. Неделя, а не окно, потому
что окна перекрываются и внутри недели зависимы (см. bench_metrics, блочный
бутстрэп); N ячеек = недель × длин режима (52×4 на режим). Дальше:
  Фридман  — есть ли вообще различие между методами (H0: ранги случайны);
  Немени   — критическая разница CD = q_α·sqrt(k(k+1)/(6N)): методы, чьи
             средние ранги отличаются меньше CD, статистически неразличимы;
  CD-диаграмма — методы на оси среднего ранга (1 = лучший), жирная черта
             соединяет неразличимые.
Ячейки одного режима не независимы полностью (одни недели у всех длин), так
что p и CD скорее оптимистичны — это инструмент упорядочивания и обзора;
доказательство конкретной пары — парный бутстрэп в bench_metrics.

Запуск:  cd scripts
         python -u bench_regimes.py --split val
         python -u bench_regimes.py --split test --code KAK --methods unet_best saits_best pchip
Выход:   data/regimes_<код>_<сплит>.csv  (MAE и Score по длинам и режимам, ранги)
         figures/fig_cd_<код>_<сплит>.png  (CD-диаграммы: все длины + 3 режима)
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, chi2

import bench_common as C
from bench_metrics import load, gap

DATA = C.DATA
FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

REGIMES = [("≤120 мин", [5, 15, 60, 120]),
           ("120–1000 мин", [240, 480, 720, 1000]),
           (">1000 мин", [1440, 2160, 2880, 4320])]
DEFAULT_METHODS = ["unet_best", "saits_best", "crossformer_best", "timesnet_best",
                   "imputeformer_best", "csdi_best", "tsmixerx_best", "nhits_best",
                   "dlinear_best", "nbeatsx_best", "segrnn_best", "tide_best",
                   "pchip", "linear", "locf"]
PRETTY = {"unet": "U-Net", "saits": "SAITS", "crossformer": "CrossFormer", "timesnet": "TimesNet",
          "imputeformer": "ImputeFormer", "csdi": "CSDI", "tsmixerx": "TSMixer", "nhits": "N-HiTS",
          "dlinear": "DLinear", "nbeatsx": "N-BEATSx", "segrnn": "SegRNN", "tide": "TiDE",
          "pchip": "PCHIP", "linear": "Linear", "locf": "LOCF", "mean": "Mean", "daily": "Daily"}
# критические значения Немени, α = 0.05 (Demšar 2006, табл. 5), k = 2..20
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
       10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
       17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def pretty(m):
    return PRETTY.get(m.replace("_best", ""), m) + ("*" if m.endswith("_best") else "")


def load_all(methods, setname):
    """{метод: {L: (mae_по_окнам (n,), block (n,))}} + проверка одинаковых окон."""
    out, ref = {}, None
    for m in methods:
        d = load(os.path.join(DATA, f"dump_{m}_{setname}.npz"))
        per = {}
        for L in [int(x) for x in d["lengths"]]:
            t, p = gap(d, L, "true"), gap(d, L, "pred")
            if ref is None:
                ref = {}
            if L not in ref:
                ref[L] = t
            elif not np.allclose(ref[L], t, equal_nan=True):
                raise SystemExit(f"{m}: окна L={L} не совпадают с {methods[0]} — сравнение невозможно")
            per[L] = (np.nanmean(np.abs(p - t), axis=1), d[f"L{L}_block"].astype(np.int64))
        out[m] = per
    return out


def cell_table(data, methods, lengths):
    """Матрица (ячейка=(неделя,длина)) × методы со средним MAE по окнам недели."""
    rows = []
    for L in lengths:
        blocks = data[methods[0]][L][1]
        for b in np.unique(blocks):
            sel = blocks == b
            rows.append([float(np.nanmean(data[m][L][0][sel])) for m in methods])
    return np.array(rows)


def friedman_nemenyi(M):
    N, k = M.shape
    R = np.array([rankdata(r) for r in M])
    mean_rank = R.mean(0)
    stat, p = friedmanchisquare(*M.T) if k > 2 else (np.nan, np.nan)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    return mean_rank, stat, p, cd, N


def cd_panel(ax, names, ranks, cd, title, N):
    k = len(names)
    order = np.argsort(ranks)
    r, nm = ranks[order], [names[i] for i in order]
    ax.set_xlim(0.3, k + 0.7)
    ax.set_ylim(-0.75 * ((k + 1) // 2) - 1.2, 1.8)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color="k", lw=1)
    for x in range(1, k + 1):
        ax.plot([x, x], [0, 0.15], color="k", lw=1)
        ax.text(x, 0.3, str(x), ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (rr, n) in enumerate(zip(r, nm)):
        if i < half:
            y, xe, ha = -1.0 - 0.75 * i, 0.4, "right"
        else:
            y, xe, ha = -1.0 - 0.75 * (k - 1 - i), k + 0.6, "left"
        ax.plot([rr, rr], [0, y], color="k", lw=0.6)
        ax.plot([rr, xe], [y, y], color="k", lw=0.6)
        ax.text(xe - (0.05 if ha == "right" else -0.05), y + 0.08, f"{n}  {rr:.2f}",
                ha=ha, va="bottom", fontsize=8)
    cliques, i = [], 0
    while i < k:
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
        i += 1
    for n_, (a, b) in enumerate(cliques):
        ax.plot([r[a] - 0.05, r[b] + 0.05], [-0.4 - 0.2 * n_] * 2, color="k", lw=3.5,
                solid_capstyle="butt")
    ax.plot([1, 1 + cd], [1.2, 1.2], color="k", lw=1.5)
    for x in (1, 1 + cd):
        ax.plot([x, x], [1.1, 1.3], color="k")
    ax.text(1 + cd / 2, 1.35, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_title(f"{title}   (N = {N} ячеек «неделя × длина»)", fontsize=9, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=sorted(C.SPLIT))
    ap.add_argument("--code", default=C.CODE)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--base", default="pchip", help="базлайн для Score")
    a = ap.parse_args()
    setname = f"{a.code}_{a.split}"
    methods = list(a.methods)
    if a.base not in methods:
        methods.append(a.base)
    data = load_all(methods, setname)
    lengths = sorted(data[methods[0]])
    names = [pretty(m) for m in methods]

    # ---- MAE и Score по длинам и режимам
    mae = {m: {L: float(np.nanmean(data[m][L][0])) for L in lengths} for m in methods}
    groups = [("все длины", lengths)] + [(g, [L for L in Ls if L in lengths]) for g, Ls in REGIMES]
    gm = {m: {g: float(np.exp(np.mean([np.log(mae[m][L]) for L in Ls]))) for g, Ls in groups}
          for m in methods}
    score_L = {m: {L: 1 - mae[m][L] / mae[a.base][L] for L in lengths} for m in methods}
    score_g = {m: {g: 1 - gm[m][g] / gm[a.base][g] for g, _ in groups} for m in methods}

    order = sorted(methods, key=lambda m: gm[m]["все длины"])
    print(f"набор: {setname}   базлайн Score: {a.base}\n")
    print("MAE, нТл (геометрическое среднее по длинам режима)  |  Score = 1 − MAE/MAE_pchip")
    hdr = f"{'метод':16}" + "".join(f"{g:>14}" for g, _ in groups) + "  |" + "".join(f"{g:>14}" for g, _ in groups)
    print(hdr)
    print("-" * len(hdr))
    for m in order:
        print(f"{pretty(m):16}" + "".join(f"{gm[m][g]:14.3f}" for g, _ in groups) + "  |"
              + "".join(f"{score_g[m][g]:+14.3f}" for g, _ in groups))
    print(f"\nScore по длинам:\n{'метод':16}" + "".join(f"{L:>8}" for L in lengths))
    for m in order:
        print(f"{pretty(m):16}" + "".join(f"{score_L[m][L]:+8.3f}" for L in lengths))

    # ---- ранги
    results = []
    for g, Ls in groups:
        M = cell_table(data, methods, Ls)
        mr, stat, p, cd, N = friedman_nemenyi(M)
        results.append((g, mr, stat, p, cd, N))
        print(f"\n== {g}: Фридман χ²={stat:.1f}, p={p:.2g}; Немени CD={cd:.2f} (k={len(methods)}, N={N})")
        for i in np.argsort(mr):
            print(f"   {mr[i]:5.2f}  {names[i]}")

    # ---- CSV
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(DATA, f"regimes_{setname}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "kind", "group", "value"])
        for m in methods:
            for L in lengths:
                w.writerow([m, "MAE", L, f"{mae[m][L]:.6g}"])
                w.writerow([m, "Score", L, f"{score_L[m][L]:.6g}"])
            for g, _ in groups:
                w.writerow([m, "MAE_gm", g, f"{gm[m][g]:.6g}"])
                w.writerow([m, "Score_gm", g, f"{score_g[m][g]:.6g}"])
        for (g, mr, stat, p, cd, N) in results:
            for i, m in enumerate(methods):
                w.writerow([m, "mean_rank", g, f"{mr[i]:.4f}"])
            w.writerow(["", "friedman_chi2", g, f"{stat:.4g}"])
            w.writerow(["", "friedman_p", g, f"{p:.3g}"])
            w.writerow(["", "nemenyi_CD", g, f"{cd:.4f}"])
            w.writerow(["", "N_cells", g, N])
    print(f"\nсохранено: {out}")

    # ---- CD-диаграммы
    fig, axes = plt.subplots(len(results), 1, figsize=(9, 3.9 * len(results)))
    for ax, (g, mr, stat, p, cd, N) in zip(axes, results):
        cd_panel(ax, names, mr, cd, g, N)
    fig.suptitle(f"CD-диаграммы по MAE, {setname}" + chr(10) + "ранг 1 = лучший; жирная черта — методы, неразличимые по Немени (α = 0.05); * = конфигурация из грида", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fp = os.path.join(FIGS, f"fig_cd_{setname}.png")
    fig.savefig(fp, dpi=130)
    print(f"сохранён {fp}")


if __name__ == "__main__":
    main()
