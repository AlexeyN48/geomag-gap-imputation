r"""Точность каждой компоненты у компонентных моделей: X, Y, Z против PCHIP.

Разведочная проверка, сделанная после вскрытия тестов и в протокол не входящая
(PROTOCOL.md). Вопрос: компонентная модель восстанавливает весь вектор — но
насколько точно каждую его ось? Метрика бенчмарка считается только по модулю,
и в обычных дампах компоненты не хранятся.

Вход — полные прогоны с сохранением каналов (bench_run --save-comp на ВСЕХ
окнах, не только на окнах рисунков):
    python -u bench_run.py --method unet_best_xyz --code ARS --split test \
        --n 1024 --save-comp --out-dir ../data/comp/xyz_full
Такой дамп весит ~0.3 ГБ против ~0.05 ГБ обычного, поэтому лежит отдельно
(data/comp/xyz_full, в git не идёт). Модуль в нём побитово тот же, что в
основном дампе, — каналы только дописываются.

Базлайн — PCHIP по каждой компоненте отдельно, построенная по ПОЛНОМУ
шестисуточному окну ровно так, как в бенчмарке: вход — окно модуля с
искусственной дырой и настоящими пропусками, у компонент выброшены те же
минуты (как в bench_run.fill_comp). Строить PCHIP по вырезу окна с полями
180 мин нельзя: если у края дыры настоящий пропуск, опорных точек с одной
стороны не остаётся, и кубическая экстраполяция на трое суток улетает до
1e10 нТл. Правильность базлайна проверяется на модуле: та же процедура для F
должна воспроизвести dump_pchip бенчмарка на всех длинах (печатается
расхождение; ожидается округление float32, ~2e-3 нТл).

Score = 1 − MAE_сеть / MAE_PCHIP; MAE пулом по точкам дыры, по режиму —
геометрическое среднее по длинам; 95 % ДИ — бутстрэп по календарным неделям,
недели общие для сети и базлайна (парный).

Запуск:  cd scripts && IGF_SPLIT=comp python -u bench_components.py --split test
Выход:   data/comp/components_<код>_<сплит>.csv
"""
import os
import csv
import argparse
import numpy as np
from scipy.interpolate import PchipInterpolator

import bench_common as C
from bench_metrics import CI_REPS

FIN = ["unet", "saits", "timesnet", "imputeformer", "tsmixerx"]
REGIMES = [("≤120 мин", [5, 15, 60, 120]),
           ("120–1000 мин", [240, 480, 720, 1000]),
           (">1000 мин", [1440, 2160, 2880, 4320])]
CHANNELS = ["X", "Y", "Z", "F"]
M = C.MARGIN


def pchip_full(Z, L, years):
    """PCHIP по полному окну для всех окон длины L. Возвращает вырезы дыры:
    (n, L) для модуля и (n, L, 3) для компонент."""
    w0, g0, blk = Z[f"L{L}_win0"], Z[f"L{L}_gap0"], Z[f"L{L}_block"]
    n = len(w0)
    outF = np.full((n, L), np.nan)
    outC = np.full((n, L, 3), np.nan)
    x = np.arange(C.W)
    for i in range(n):
        F, XYZ = years[int(blk[i]) // 100]
        st, a = int(w0[i]), int(g0[i])
        f = F[st:st + C.W].astype(np.float64)
        f[a:a + L] = np.nan
        obs = np.isfinite(f)
        outF[i] = PchipInterpolator(x[obs], f[obs], extrapolate=True)(x[a:a + L])
        w = XYZ[st:st + C.W].astype(np.float64)
        w[~obs] = np.nan
        for k in range(3):
            ok = np.isfinite(w[:, k])
            outC[i, :, k] = PchipInterpolator(x[ok], w[ok, k], extrapolate=True)(x[a:a + L])
    return outF, outC


def week_table(err, blk, weeks):
    """(2, недели): сумма |ошибки| и число точек по каждой неделе."""
    idx = {w: j for j, w in enumerate(weeks)}
    out = np.zeros((2, len(weeks)))
    for e, b in zip(err, blk):
        ok = np.isfinite(e)
        out[0, idx[b]] += np.abs(e[ok]).sum()
        out[1, idx[b]] += ok.sum()
    return out


def regime_score(cells, nweeks, rng):
    """cells: [(сеть (2, недели), базлайн (2, недели))] по длинам режима.
    -> Score, ДИ, геом. средние MAE сети и базлайна."""
    def gm_ratio(wt):
        r = [np.log((a[0] @ wt) / (a[1] @ wt)) - np.log((b[0] @ wt) / (b[1] @ wt))
             for a, b in cells]
        return np.exp(np.mean(r))
    one = np.ones(nweeks)
    est = 1 - gm_ratio(one)
    bs = [1 - gm_ratio(np.bincount(rng.integers(0, nweeks, nweeks), minlength=nweeks))
          for _ in range(CI_REPS)]
    gm = lambda k: float(np.exp(np.mean([np.log(c[k][0].sum() / c[k][1].sum()) for c in cells])))
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), gm(0), gm(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", default="ARS")
    ap.add_argument("--split", required=True, choices=sorted(C.SPLIT))
    ap.add_argument("--dump-dir", default=os.path.join(C.OUT, "xyz_full"))
    a = ap.parse_args()

    setn = C.setname(a.code, a.split)
    dumps = {m: os.path.join(a.dump_dir, f"dump_{m}_best_xyz_{setn}.npz") for m in FIN}
    miss = [m for m, p in dumps.items() if not os.path.exists(p)]
    if miss:
        raise SystemExit(f"нет полных дампов с компонентами: {', '.join(miss)} — "
                         f"см. докстринг, bench_run --save-comp")
    Z = {m: np.load(p) for m, p in dumps.items()}
    P = np.load(os.path.join(C.OUT, f"dump_pchip_{setn}.npz"))
    years = {y: (C.load_year(y, code=a.code), C.load_year_comp(y, code=a.code))
             for y in C.SPLIT[a.split]}

    # базлайн и проверка на модуле
    base, worst = {}, 0.0
    Z0 = Z[FIN[0]]
    for L in C.LENGTHS:
        if not np.array_equal(P[f"L{L}_true"], Z0[f"L{L}_true"], equal_nan=True):
            raise SystemExit(f"L={L}: окна полного дампа не совпадают с бенчмарком")
        bf, bc = pchip_full(Z0, L, years)
        base[L] = bc
        worst = max(worst, float(np.nanmax(np.abs(bf - P[f"L{L}_pred"][:, M:M + L]))))
    print(f"проверка базлайна: PCHIP по полному окну против dump_pchip, все "
          f"{len(C.LENGTHS)} длин, макс. расхождение {worst:.1e} нТл")
    if worst > 0.01:
        raise SystemExit("базлайн не воспроизводит бенчмарк — счёт остановлен")

    weeks = {L: sorted(set(Z0[f"L{L}_block"].tolist())) for L in C.LENGTHS}
    rows = []
    for m in FIN:
        rng = np.random.default_rng(C.SEED)
        for g, Ls in REGIMES:
            for k, ch in enumerate(CHANNELS):
                cells, allw = [], sorted(set().union(*[weeks[L] for L in Ls]))
                for L in Ls:
                    blk = Z[m][f"L{L}_block"]
                    if ch == "F":
                        t = Z[m][f"L{L}_true"][:, M:M + L]
                        pn = Z[m][f"L{L}_pred"][:, M:M + L]
                        pb = P[f"L{L}_pred"][:, M:M + L]
                    else:
                        t = Z[m][f"L{L}_trueXYZ"][:, M:M + L, k]
                        pn = Z[m][f"L{L}_predXYZ"][:, M:M + L, k]
                        pb = base[L][:, :, k]
                    cells.append((week_table(pn - t, blk, allw), week_table(pb - t, blk, allw)))
                s, lo, hi, mn, mb = regime_score(cells, len(allw), rng)
                rows.append([m, g, ch, f"{s:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                             f"{mn:.3f}", f"{mb:.3f}"])
                print(f"{m:<13}{g:<14}{ch}  Score {s:+.3f} [{lo:+.3f}; {hi:+.3f}]  "
                      f"MAE {mn:.2f} / {mb:.2f} нТл")
    out = os.path.join(C.OUT, f"components_{setn}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arch", "regime", "channel", "score", "lo", "hi",
                    "mae_net", "mae_pchip"])
        w.writerows(rows)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
