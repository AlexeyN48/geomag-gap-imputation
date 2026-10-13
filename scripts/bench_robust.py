r"""Проверка устойчивости выводов к способу взвешивания недель.

Зачем. MAE в отчёте считается пулом: все точки всех окон в одну кучу. Это
неявно взвешивает недели по числу окон в них, а окон в неделе от 1 до 40, и
больше их там, где данные полнее (связь с полнотой r ~ +0.5). То есть полные
недели влияют сильнее — никто этого не задавал, так вышло из отбраковки окон.

Альтернатива — усреднять сначала внутри недели, потом по неделям с равным
весом. Она убирает это смещение, но добавляет дисперсию: на длинных дырах
интервал шире в полтора раза, потому что неделя с одним окном входит наравне
с неделей с сорока.

Какой способ «правильный», не определено: оба защитимы. Поэтому основным
остаётся пул (он в отчёте с самого начала и у него меньше дисперсия), а этот
скрипт считает ОБА и сравнивает не числа, а ВЫВОДЫ: кто лидер в режиме, в
какую сторону разница в паре. Если выводы совпадают — способ взвешивания на
них не влияет, и это можно утверждать, а не надеяться. Если расходятся —
результат зависит от произвола в методе счёта, и выбирать «который красивее»
нельзя: так делается подгонка.

Запуск:  cd bench_arti/scripts
         python -u bench_robust.py --code ARS --split val
         IGF_SPLIT=comp python -u bench_robust.py --code ARS --split val --pairs
"""
import os
import re
import glob
import argparse
import numpy as np

import bench_common as C

REGIMES = [("<=120", [5, 15, 60, 120]),
           ("240-1000", [240, 480, 720, 1000]),
           (">1000", [1440, 2160, 2880, 4320])]


def per_window_err(path, L):
    """(ошибка каждого окна, номер недели) для длины L."""
    with np.load(path, allow_pickle=False) as d:
        if f"L{L}_true" not in d:
            return None, None
        m = int(d["margin"])
        t = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
        p = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
        return np.nanmean(np.abs(p - t), axis=1), d[f"L{L}_block"]


def mae_both(path, L):
    """MAE двумя способами: пул по окнам и равный вес недель."""
    e, b = per_window_err(path, L)
    if e is None:
        return None, None
    weeks = np.unique(b)
    return float(np.nanmean(e)), float(np.mean([np.nanmean(e[b == w]) for w in weeks]))


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def methods_of(setname):
    pat = os.path.join(C.OUT, f"dump_*_{setname}.npz")
    out = []
    for p in sorted(glob.glob(pat)):
        m = re.match(rf"dump_(.+)_{re.escape(setname)}\.npz$", os.path.basename(p))
        if m:
            out.append(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=C.CODE)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pairs", action="store_true",
                    help="дополнительно сверить вердикты пар «компоненты против "
                         "модуля» (<арх>_..._xyz против <арх>_...)")
    a = ap.parse_args()
    setname = C.setname(a.code, a.split)
    ms = methods_of(setname)
    if not ms:
        raise SystemExit(f"нет дампов для набора {setname} в {C.OUT}")
    print(f"набор {setname}, методов {len(ms)}\n")

    # MAE обоими способами
    P, Wk = {}, {}
    for m in ms:
        p = os.path.join(C.OUT, f"dump_{m}_{setname}.npz")
        P[m], Wk[m] = {}, {}
        for L in C.LENGTHS:
            a1, a2 = mae_both(p, L)
            if a1 is not None:
                P[m][L], Wk[m][L] = a1, a2

    rel = [abs(Wk[m][L] / P[m][L] - 1) for m in ms for L in P[m]]
    print(f"расхождение MAE между способами: медиана {np.median(rel) * 100:.1f} %, "
          f"худшее {max(rel) * 100:.1f} %\n")

    # вывод 1: кто лидер в режиме
    print("ЛИДЕР РЕЖИМА (по геометрическому среднему MAE)")
    print(f"{'режим':>10}{'пул по окнам':>22}{'равный вес недель':>24}{'':>8}")
    agree = total = 0
    for g, Ls in REGIMES:
        ok = [m for m in ms if all(L in P[m] for L in Ls)]
        if not ok:
            continue
        l1 = min(ok, key=lambda m: gm([P[m][L] for L in Ls]))
        l2 = min(ok, key=lambda m: gm([Wk[m][L] for L in Ls]))
        total += 1
        agree += l1 == l2
        print(f"{g:>10}{l1:>22}{l2:>24}{'совпал' if l1 == l2 else 'РАЗОШЁЛСЯ':>10}")
    print(f"\nлидеры совпали в {agree} режимах из {total}")

    # вывод 2: вердикты пар
    if a.pairs:
        pairs = [(m[:-4], m) for m in ms if m.endswith("_xyz") and m[:-4] in ms]
        if not pairs:
            print("\nпар «модуль / компоненты» в наборе нет")
            return
        print(f"\nВЕРДИКТ ПАРЫ «компоненты против модуля», по режимам")
        print(f"{'пара':>24}{'режим':>11}{'пул':>14}{'по неделям':>16}{'':>10}")
        ag = tt = 0
        for f, x in pairs:
            for g, Ls in REGIMES:
                if not all(L in P[f] and L in P[x] for L in Ls):
                    continue
                d1 = gm([P[x][L] for L in Ls]) - gm([P[f][L] for L in Ls])
                d2 = gm([Wk[x][L] for L in Ls]) - gm([Wk[f][L] for L in Ls])
                v1 = "компоненты" if d1 < 0 else "модуль"
                v2 = "компоненты" if d2 < 0 else "модуль"
                tt += 1
                ag += v1 == v2
                print(f"{f.split('_')[0]:>24}{g:>11}{v1:>14}{v2:>16}"
                      f"{'ок' if v1 == v2 else 'РАЗОШЁЛСЯ':>10}")
        print(f"\nвердикты совпали в {ag} случаях из {tt}")
        if ag < tt:
            print("ВНИМАНИЕ: вывод зависит от способа взвешивания — выбирать "
                  "более удобный нельзя, надо приводить оба")


if __name__ == "__main__":
    main()
