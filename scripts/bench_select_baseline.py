r"""Выбор параметров классического базлайна на валидации.

У сетей параметры подбирает грид (bench_grid.py), у сезонных базлайнов —
этот скрипт, по тому же правилу сводки: геометрическое среднее MAE по всем
12 длинам на 1024 окнах валидации. MAE на длине — пул по точкам внутри
пропуска, как в bench_metrics. Только валидация: тестовые годы к выбору не
допускаются.

Запуск:  cd scripts
         python -u bench_select_baseline.py --family seadec harmreg --run
         python -u bench_select_baseline.py --family harmreg        # только сводка

--run догоняет недостающие дампы через bench_run.py (CPU, ~1 мин на вариант).
Таблица -> data/<сплит>/baseline_selection.csv. Победителя семейства
вписывают в bench_run.METHODS под коротким именем (seadec, harmreg) — именно
оно идёт в тест.
"""
import os
import sys
import csv
import argparse
import subprocess
import numpy as np

import bench_common as C
import bench_run as R

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = "val"


def dump_path(method, code):
    return os.path.join(C.OUT, f"dump_{method}_{C.setname(code, SPLIT)}.npz")


def mae_by_length(path):
    with np.load(path, allow_pickle=False) as d:
        m = int(d["margin"])
        out = {}
        for L in d["lengths"]:
            L = int(L)
            t = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
            p = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
            out[L] = float(np.nanmean(np.abs(p - t)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", nargs="+", required=True,
                    help="префиксы вариантов в bench_run.METHODS, напр. seadec harmreg")
    ap.add_argument("--code", default=C.CODE)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--run", action="store_true", help="прогнать недостающие дампы")
    a = ap.parse_args()

    rows = []
    for fam in a.family:
        variants = sorted(m for m in R.METHODS if m.startswith(fam + "_"))
        if not variants:
            raise SystemExit(f"нет вариантов {fam}_* в bench_run.METHODS")
        for m in variants:
            p = dump_path(m, a.code)
            if not os.path.exists(p):
                if not a.run:
                    print(f"  нет дампа {os.path.basename(p)} — пропуск (нужен --run)")
                    continue
                print(f"  прогон {m}", flush=True)
                r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "bench_run.py"),
                                    "--method", m, "--split", SPLIT, "--code", a.code,
                                    "--n", str(a.n)],
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace")
                if r.returncode != 0:
                    raise SystemExit(f"{m} упал:\n{(r.stderr or r.stdout)[-1500:]}")
            by_L = mae_by_length(p)
            rows.append((fam, m, float(np.exp(np.mean(np.log(list(by_L.values()))))), by_L))

    if not rows:
        raise SystemExit("нечего сводить")
    lengths = sorted(rows[0][3])
    out = os.path.join(C.OUT, "baseline_selection.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "geomean_mae"] + [f"L{L}" for L in lengths] + ["chosen"])
        for fam in a.family:
            fr = sorted((r for r in rows if r[0] == fam), key=lambda r: r[2])
            if not fr:
                continue
            print(f"\n[{fam}]  геом. среднее MAE по {len(lengths)} длинам, нТл")
            for i, (_, m, g, by_L) in enumerate(fr):
                print(f"  {m:<26}{g:8.3f}" + ("   <- выбран" if i == 0 else ""))
                w.writerow([fam, m, f"{g:.5f}"] + [f"{by_L[L]:.5f}" for L in lengths]
                           + ["yes" if i == 0 else ""])
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
