r"""Выбор финалистов на валидации по пяти сидам.

Правило то же, что при отборе пятёрки в работе: средний ранг по ячейкам
«календарная неделя × длина пропуска» на валидации ARS, вход — модуль F.
Разница одна: ранги считаются отдельно для каждого сида (одни и те же окна,
разные обученные модели) и усредняются по сидам, так что место архитектуры
не держится на удачном зерне. Ранжируются только сети: интерполяция в
ячейках сдвигала бы средние ранги сетей неодинаково, а выбирается пятёрка
сетей.

Тестовые годы здесь не читаются. Финалисты дальше обучаются на остальных
станциях (этап stations), и только после этого считается тест.

Запуск:  cd scripts
         python -u bench_finalists.py            # -> data/<сплит>/seeds/finalists.json
"""
import os
import json
import argparse

import numpy as np
from scipy.stats import rankdata

import bench_common as C
import bench_grid as G
from bench_regimes import cell_table, PRETTY

SEED_DIR = os.path.join(C.OUT, "seeds")
OUT = os.path.join(SEED_DIR, "finalists.json")


def load(method, code, split):
    """{L: (mae по окнам, неделя)} из дампа сида; None, если дампа нет."""
    p = os.path.join(SEED_DIR, f"dump_{method}_{C.setname(code, split)}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as d:
        m = int(d["margin"])
        out = {}
        for L in [int(x) for x in d["lengths"]]:
            t = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
            pr = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
            out[L] = (np.nanmean(np.abs(pr - t), axis=1), d[f"L{L}_block"].astype(np.int64))
    return out


def ranks(archs, seeds, suffix):
    """Средний ранг каждой архитектуры: по ячейкам внутри сида, затем по сидам."""
    per_seed = {a: [] for a in archs}
    for s in seeds:
        names = [f"{a}_best{suffix}_s{s}" for a in archs]
        data = {n: load(n, C.CODE, "val") for n in names}
        miss = [n for n, v in data.items() if v is None]
        if miss:
            raise SystemExit(f"нет дампов на валидации: {', '.join(miss[:4])} — "
                             f"сначала этап seedval")
        lengths = sorted(data[names[0]])
        M = cell_table(data, names, lengths)
        R = np.array([rankdata(r) for r in M]).mean(0)
        for a, r in zip(archs, R):
            per_seed[a].append(float(r))
    return {a: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for a, v in per_seed.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5, help="сколько финалистов")
    ap.add_argument("--archs", nargs="+", default=list(G.GRID))
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    a = ap.parse_args()

    rf = ranks(a.archs, a.seeds, "")
    try:
        rx = ranks(a.archs, a.seeds, "_xyz")
    except SystemExit as e:
        print(f"компоненты: {e} — сводка только по модулю")
        rx = None
    order = sorted(a.archs, key=lambda x: rf[x][0])
    fin = order[:a.k]

    print(f"средний ранг на валидации ARS ({len(a.seeds)} сидов; 1 — лучший)")
    print(f"{'':16}{'модуль F':>18}" + (f"{'компоненты':>18}" if rx else ""))
    for i, x in enumerate(order):
        line = f"{PRETTY.get(x, x):16}{rf[x][0]:10.2f} ± {rf[x][1]:4.2f}"
        if rx:
            line += f"{rx[x][0]:10.2f} ± {rx[x][1]:4.2f}"
        print(line + ("   финалист" if i < a.k else ""))
    if rx:
        fx = sorted(a.archs, key=lambda x: rx[x][0])[:a.k]
        same = set(fx) == set(fin)
        print(f"\nпятёрка по компонентам {'совпадает' if same else 'НЕ совпадает'} "
              f"с пятёркой по модулю" + ("" if same else f": {', '.join(fx)}"))

    os.makedirs(SEED_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(rule="средний ранг по ячейкам неделя × длина на валидации ARS, "
                            "вход модуль, усреднение по сидам",
                       seeds=a.seeds, finalists=fin,
                       rank_F={x: rf[x] for x in order},
                       rank_XYZ={x: rx[x] for x in order} if rx else None),
                  f, ensure_ascii=False, indent=1)
    print(f"\nфиналисты: {', '.join(fin)} -> {OUT}")


if __name__ == "__main__":
    main()
