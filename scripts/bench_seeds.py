r"""Разброс по сидам: сводка дампов моделей, обученных с разными зёрнами.

Модель с зерном s лежит в models/<сплит>/seeds/<имя>_s<s>.pt, её дампы — в
data/<сплит>/seeds/dump_<имя>_s<s>_<набор>.npz (bench_jobs.py, этапы seedval,
stationval и test). Здесь по каждой модели и набору считается, на сколько
результат зависит от зерна:

  MAE по длине    — пул по точкам внутри пропуска, как в bench_metrics;
  Score по длине  — 1 − MAE / MAE_PCHIP на тех же окнах (дамп PCHIP набора
                    берётся из data/<сплит>/, его готовит этап baselines);
  то же по режимам длины — через геометрическое среднее MAE по длинам режима.

По каждой величине — среднее, стандартное отклонение, минимум и максимум по
сидам и их число. Окна у всех сидов одни и те же (план окон от зерна модели не
зависит), поэтому разброс относится к обучению, а не к выборке окон.

Запуск:  cd scripts
         python -u bench_seeds.py                # -> data/<сплит>/seeds_summary.csv
"""
import os
import re
import csv
import argparse
from collections import defaultdict

import numpy as np

import bench_common as C
from bench_regimes import REGIMES

SEED_DIR = os.path.join(C.OUT, "seeds")
NAME = re.compile(r"^dump_(?P<model>.+)_s(?P<seed>\d+)_(?P<set>[A-Z]{3}_.+)\.npz$")


def mae_by_length(d):
    m = int(d["margin"])
    out = {}
    for L in d["lengths"]:
        L = int(L)
        t = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
        p = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
        out[L] = (float(np.nanmean(np.abs(p - t))), t)
    return out


def pchip_mae(setn, lengths_true):
    """MAE PCHIP на тех же окнах; None, если дампа нет или окна не совпали."""
    p = os.path.join(C.OUT, f"dump_pchip_{setn}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as d:
        m = int(d["margin"])
        out = {}
        for L, t in lengths_true.items():
            tb = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
            if tb.shape != t.shape or not np.allclose(tb, t, equal_nan=True):
                return None
            pb = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
            out[L] = float(np.nanmean(np.abs(pb - tb)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(C.OUT, "seeds_summary.csv"))
    a = ap.parse_args()
    if not os.path.isdir(SEED_DIR):
        raise SystemExit(f"нет {SEED_DIR} — сначала этапы seeds/stations")

    # (модель, набор) -> {величина: {сид: значение}}
    vals = defaultdict(lambda: defaultdict(dict))
    base_cache = {}
    for f in sorted(os.listdir(SEED_DIR)):
        g = NAME.match(f)
        if not g:
            continue
        model, seed, setn = g["model"], int(g["seed"]), g["set"]
        with np.load(os.path.join(SEED_DIR, f), allow_pickle=False) as d:
            by_L = mae_by_length(d)
        if setn not in base_cache:
            base_cache[setn] = pchip_mae(setn, {L: v[1] for L, v in by_L.items()})
        base = base_cache[setn]
        v = vals[(model, setn)]
        for L, (mae, _) in by_L.items():
            v[("MAE", str(L))][seed] = mae
            if base:
                v[("Score", str(L))][seed] = 1.0 - mae / base[L]
        for rname, Ls in [("все длины", sorted(by_L))] + REGIMES:
            if not all(L in by_L for L in Ls):
                continue
            g_m = float(np.exp(np.mean(np.log([by_L[L][0] for L in Ls]))))
            v[("MAE", rname)][seed] = g_m
            if base:
                g_b = float(np.exp(np.mean(np.log([base[L] for L in Ls]))))
                v[("Score", rname)][seed] = 1.0 - g_m / g_b

    if not vals:
        raise SystemExit(f"в {SEED_DIR} нет дампов вида dump_<модель>_s<сид>_<набор>.npz")
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "set", "metric", "gap", "mean", "std", "min", "max",
                    "n_seeds", "seeds"])
        for (model, setn), v in sorted(vals.items()):
            for (metric, gap), by_seed in v.items():
                x = np.array([by_seed[s] for s in sorted(by_seed)])
                w.writerow([model, setn, metric, gap, f"{x.mean():.5f}",
                            f"{x.std(ddof=1) if x.size > 1 else 0.0:.5f}",
                            f"{x.min():.5f}", f"{x.max():.5f}", x.size,
                            " ".join(map(str, sorted(by_seed)))])
    print(f"сохранено: {a.out}")

    # короткая сводка: Score на длинных пропусках
    print(f"\nScore на пропусках >1000 мин: среднее ± ст. откл. по сидам")
    for (model, setn), v in sorted(vals.items()):
        s = v.get(("Score", ">1000 мин"))
        if s:
            x = np.array(list(s.values()))
            sd = x.std(ddof=1) if x.size > 1 else 0.0
            print(f"  {model:<34}{setn:<22}{x.mean():+.3f} ± {sd:.3f}  (сидов {x.size})")


if __name__ == "__main__":
    main()
