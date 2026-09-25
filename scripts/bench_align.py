r"""Выравнивание обучающих данных станций переноса по станции обучения.

Зачем: в обратном эксперименте каждая станция обучается на себе, и объём
обучающих данных у них был разный. У ARS 8.39 года-эквивалента, у остальных
8.97-9.01 — на 7.4 % больше (почти вся разница это ARS 2016 с покрытием 49.6 %
против 100 % у прочих). Сравнение «своя модель против привезённой» при этом
частично мерило бы объём данных, а не станцию.

Что делает: в обучающих годах у станций переноса зануляются те минуты, где
пропуск у ARS. Данные ARS не трогаются, поэтому модели основного эксперимента
и грид остаются в силе; переобучения требуют только модели обратного
эксперимента. Остаточный разброс 0.4 % — это собственные дыры станций
(в основном CMO), убрать их можно было бы только пересечением по всем шести,
но тогда изменились бы и данные ARS.

Тестовые и валидационный годы НЕ трогаются: там окна и позиции дыр и так
совпадают побитово (bench_common.gather_aligned выбирает позицию из
пересечения кандидатов по всем станциям).

Маска переносится на все каналы: плохая минута плоха и в F, и в X, Y, Z.

Идемпотентно: повторный запуск ничего не меняет.

Запуск:  cd bench_arti/scripts
         python -u bench_align.py --dry-run
         python -u bench_align.py
"""
import os
import argparse
import shutil
import numpy as np

import bench_common as C

TRANSFER = ["WNG", "CMO", "KAK", "HUA", "HER"]
SUFFIX = "_before_align"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.CODE, help="станция-эталон (по ней равняем)")
    ap.add_argument("--codes", nargs="+", default=TRANSFER)
    ap.add_argument("--years", nargs="+", type=int, default=None,
                    help="по умолчанию — обучающие годы текущего сплита")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    years = a.years or C.SPLIT["train"]

    print(f"эталон: {a.base};  станции: {', '.join(a.codes)}")
    print(f"годы:   {', '.join(map(str, years))}  (сплит {C.SPLIT_NAME})\n")

    base_mask = {}
    for y in years:
        p = os.path.join(C.DATA, f"{a.base}_{y}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"нет эталонного {os.path.basename(p)}")
        with np.load(p, allow_pickle=False) as d:
            base_mask[y] = np.isfinite(d["F"])

    Y1 = 525600.0
    for code in a.codes:
        before = after = 0
        for y in years:
            p = os.path.join(C.DATA, f"{code}_{y}.npz")
            if not os.path.exists(p):
                print(f"  {code} {y}: нет файла, пропуск")
                continue
            with np.load(p, allow_pickle=False) as d:
                cur = {k: d[k] for k in d.files}
            m = base_mask[y]
            if cur["F"].size != m.size:
                raise SystemExit(f"{code} {y}: длина {cur['F'].size} != {m.size} у эталона")
            before += int(np.isfinite(cur["F"]).sum())
            kill = ~m
            if not a.dry_run and kill.any():
                bak = os.path.join(C.DATA, f"{code}_{y}{SUFFIX}.npz")
                if not os.path.exists(bak):
                    shutil.copy(p, bak)
                for k in ("F", "X", "Y", "Z"):
                    if k in cur:
                        cur[k] = cur[k].copy()
                        cur[k][kill] = np.nan
                np.savez(p, **cur)
            after += int((np.isfinite(cur["F"]) & m).sum())
        print(f"  {code}: было {before / Y1:.2f} года, стало {after / Y1:.2f} "
              f"({(after - before) / Y1:+.2f})")

    tot = 0
    for y in years:
        tot += int(base_mask[y].sum())
    print(f"\n  {a.base}: {tot / Y1:.2f} года (не изменялась)")
    if a.dry_run:
        print("\n--dry-run: файлы не тронуты")


if __name__ == "__main__":
    main()
