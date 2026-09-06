r"""Аудит подготовленных рядов (bench_prep.py) до любых прогонов.

Четыре проверки, каждая ловит свой класс дефекта:
  1. длина ровно по календарю года (иначе индекс "минута года" врёт);
  2. последние сутки года Y не дублируют первые сутки года Y+1;
  3. значения абсолютные, а не вариации (медиана порядка десятков тысяч нТл);
  4. покрытие и длиннейший непрерывный пропуск — что реально доступно окнам.

Запуск:  cd bench_arti/scripts && python -u bench_audit.py
"""
import os
import glob
import argparse
import numpy as np

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MIN_ABS_FIELD = 5000.0        # нТл: ниже этого медиана означает вариации


def minutes_in_year(y):
    leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
    return 527040 if leap else 525600


def longest_gap(fin):
    """Длиннейшая серия подряд идущих пропусков, мин."""
    if fin.all():
        return 0
    d = np.diff(np.concatenate(([1], fin.astype(np.int8), [1])))
    starts = np.flatnonzero(d == -1)
    ends = np.flatnonzero(d == 1)
    return int((ends - starts).max()) if starts.size else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="ARS")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA, f"{args.code}_*.npz")))
    if not files:
        raise SystemExit(f"нет файлов {args.code}_*.npz в {DATA}\n"
                         f"сначала: python -u bench_prep.py")

    print(f"каталог: {DATA}\n")
    print(f"{'год':>5}{'источник':>20}{'минут':>9}{'календарь':>11}{'валид':>8}"
          f"{'медиана F':>11}{'макс. дыра':>12}  замечание")
    print("-" * 96)

    bad_len, variations, series = [], [], {}
    for p in files:
        d = np.load(p, allow_pickle=False)
        y = int(d["year"])
        F = d["F"].astype(np.float64)
        fin = np.isfinite(F)
        exp = minutes_in_year(y)
        med = float(np.median(F[fin])) if fin.any() else np.nan
        series[y] = F

        note = ""
        if F.size != exp:
            note = f"ДЛИНА НЕ ПО КАЛЕНДАРЮ ({F.size} против {exp})"
            bad_len.append(y)
        if fin.any() and med < MIN_ABS_FIELD:
            note = (note + "; " if note else "") + "ВАРИАЦИИ, НЕ АБСОЛЮТ"
            variations.append(y)
        if not fin.any():
            note = (note + "; " if note else "") + "ПУСТО"

        mf = "        нет" if not np.isfinite(med) else f"{med:>11.0f}"
        print(f"{y:>5}{str(d['f_source']):>20}{F.size:>9}{exp:>11}"
              f"{fin.mean():>8.1%}{mf}{longest_gap(fin):>12}  {note}")

    # ---- стык годов: последние сутки Y против первых суток Y+1 ----
    print("\n" + "=" * 96)
    dups = []
    for y in sorted(series):
        if y + 1 not in series:
            continue
        tail, head = series[y][-1440:], series[y + 1][:1440]
        ok = np.isfinite(tail) & np.isfinite(head)
        if ok.sum() < 100:
            continue
        if np.abs(tail[ok] - head[ok]).max() < 1e-6:
            dups.append((y, int(ok.sum())))
    if dups:
        print("ДУБЛИРОВАНИЕ НА СТЫКЕ ГОДОВ (31 декабря совпало с 1 января):")
        for y, n in dups:
            print(f"  {y} -> {y + 1}: {n} минут совпадают побитово")
    else:
        print("Стыки годов чисты: 31 декабря нигде не дублирует 1 января.")

    print(f"\nИТОГ: лет {len(files)}; длина не по календарю {len(bad_len)}; "
          f"вариационных {len(variations)}; дублей на стыке {len(dups)}")
    if bad_len or variations or dups:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
