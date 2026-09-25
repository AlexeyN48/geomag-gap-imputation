r"""Препроцессинг INTERMAGNET XML -> минутные ряды F для бенчмарка.

Запуск:  cd bench_arti/scripts && python -u bench_prep.py
         python -u bench_prep.py --code ARS --years 2013 2014 2015
"""
import os
import re
import glob
import argparse
import numpy as np
import xml.etree.ElementTree as ET

BASE = os.environ.get("IGF_BASE") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(BASE, "dataset_intermagnet")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

TOL = 1.0             # нТл: допуск на расхождение уровней вектора и скаляра
MISSING = 99998.0     # значения >= этого в IAGA означают пропуск
DELTA_MAX = 1000.0    # |G| ниже этого => G это поправка dF, а не модуль
SPIKE_WIN = 11        # окно медианы для детекции засечек в reported-годах
SPIKE_THR = 100.0     # нТл: |F - медиана| выше => артефакт
GROSS_THR = 5000.0    # нТл: |F - медиана года| выше => физически невозможно, NaN
                      # (для ВСЕХ типов данных; ловит одиночные «островки» внутри
                      # пропусков, до которых скользящая медиана не дотягивается —
                      # пример: CMO 2023, день 300, F = 249 нТл между двумя NaN)

# Иглы — одиночные сбойные точки на спокойном фоне (скан всех станций, сентябрь
# 2026): точка отклоняется от медианы соседних ±10 мин больше NEEDLE_THR, соседи
# с фоном совпадают (< 3 нТл), фон спокойный (σ приращений < 1.5 нТл). Порог
# засечек 100 нТл их пропускал: у ARS 2013 147 игл по 6–72 нТл. Фильтр
# применяется ТОЛЬКО к обучающим годам (bench_common.SPLIT["train"]): тестовые
# и валидационный годы остаются как есть — в них по 0–1 игле, и их изменение
# сдвинуло бы план тестовых окон и обесценило все посчитанные дампы.
NEEDLE_THR = 5.0
# Участки, признанные сбоем вручную: хаотичные скачки ±150 нТл, которых нет ни
# на одной другой станции (WNG в эти часы спокоен, |dF| < 0.5 нТл/мин).
BAD_INTERVALS = {
    ("ARS", 2013): [("2013-04-02 05:45", "2013-04-02 10:30")],
    # ступени уровня ровно на +1265 нТл, держащиеся часами: 8 марта на WNG и
    # KAK совершенно спокойно (размах 64 и 50 нТл), так что это сбой записи,
    # а не геофизика. Январские всплески того же года, наоборот, оставлены —
    # там буря видна и на WNG (547 нТл), и на KAK (264 нТл)
    ("ARS", 2026): [("2026-03-07 15:22", "2026-03-09 06:45"),
                    ("2026-05-29 10:39", "2026-05-29 15:31")],
}


def train_years():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_common as C
    return set(C.SPLIT["train"])


def deneedle(F):
    """Одиночные иглы -> NaN. Возвращает (F, число игл)."""
    x = F.astype(np.float64)
    n = x.size
    d = np.abs(np.diff(x))
    bad = []
    for t in np.flatnonzero(d > NEEDLE_THR) + 1:
        if t < 11 or t > n - 12:
            continue
        w = np.r_[x[t - 10:t], x[t + 1:t + 11]]
        if not (np.isfinite(w).all() and np.isfinite(x[t])):
            continue
        base = np.median(w)
        if (np.std(np.diff(w)) < 1.5 and abs(x[t] - base) > NEEDLE_THR
                and abs(x[t - 1] - base) < 3 and abs(x[t + 1] - base) < 3):
            bad.append(t)
    if bad:
        F = F.copy()
        F[bad] = np.nan
    return F, len(bad)


def cut_bad(F, code, year):
    """Ручные сбойные участки -> NaN. Возвращает (F, число минут)."""
    import datetime as dt
    t0 = dt.datetime(year, 1, 1)
    k = 0
    for a, b in BAD_INTERVALS.get((code, year), []):
        i = int((dt.datetime.fromisoformat(a) - t0).total_seconds() // 60)
        j = int((dt.datetime.fromisoformat(b) - t0).total_seconds() // 60)
        F = F.copy()
        k += int(np.isfinite(F[i:j]).sum())
        F[i:j] = np.nan
    return F, k


def minutes_in_year(y):
    leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
    return 527040 if leap else 525600


def parse(path):
    """Читает XML потоком: метаданные + все каналы отсчётов."""
    comps, meta = {}, {}
    for _, el in ET.iterparse(path, events=("end",)):
        tag = el.tag
        if tag == "Sample":
            for ch in el:
                comps.setdefault(ch.tag, []).append(ch.text)
            el.clear()
        elif tag in ("ObservatoryCode", "ObservatoryName", "DataType",
                     "StartTime", "SensorLatitude", "SensorLongitude"):
            meta[tag] = el.text
    out = {}
    for k, v in comps.items():
        a = np.array([float(x) if x and x.strip() else np.nan for x in v])
        out[k] = np.where(a >= MISSING, np.nan, a)
    return out, meta


def vector_f(ch):
    """F по вектору и сами компоненты. Возвращает (F, X, Y, Z) или None."""
    if all(k in ch for k in "XYZ"):
        X, Y, Z = ch["X"], ch["Y"], ch["Z"]
    elif all(k in ch for k in "HDZ"):
        d = np.deg2rad(ch["D"])
        X, Y, Z = ch["H"] * np.cos(d), ch["H"] * np.sin(d), ch["Z"]
    else:
        return None
    return np.sqrt(X ** 2 + Y ** 2 + Z ** 2), X, Y, Z


def scalar_f(ch, f_vec):
    """Скалярный модуль. S/F — как есть; G — модуль либо поправка (см. шапку)."""
    for k in ("S", "F"):
        if k in ch:
            return ch[k], k
    if "G" in ch:
        g = ch["G"]
        med = np.nanmedian(np.abs(g))
        if np.isfinite(med) and med < DELTA_MAX:
            return (f_vec + g, "G(поправка)") if f_vec is not None else (None, None)
        return g, "G(модуль)"
    return None, None


def despike(F):
    """Засечки в reported-годах: отклонение от скользящей медианы > порога.
    Возвращает (F, n_spikes)."""
    fin = np.isfinite(F) if F is not None else None
    if F is None or F.size < SPIKE_WIN or not fin.any():
        return F, 0
    from scipy.ndimage import median_filter
    filled = np.where(fin, F, np.median(F[fin]))
    med = median_filter(filled, size=SPIKE_WIN, mode="nearest")
    bad = fin & (np.abs(F - med) > SPIKE_THR)
    out = F.copy()
    out[bad] = np.nan
    return out, int(bad.sum())


def to_grid(a, n):
    """Ровно n минут: длиннее — обрезать, короче — добить NaN."""
    if a is None:
        return None
    if a.size >= n:
        return a[:n]
    pad = np.full(n - a.size, np.nan)
    return np.concatenate([a, pad])


def process(path, args):
    """Чтение XML -> минутные ряды F, X, Y, Z + метаданные."""
    ch, meta = parse(path)
    code = meta.get("ObservatoryCode", "???")
    year = int(meta["StartTime"][:4])
    vec = vector_f(ch)
    f_vec, X, Y, Z = vec if vec else (None, None, None, None)
    f_sca, sca_name = scalar_f(ch, f_vec)
    dtype = meta.get("DataType", "")

    # --- выбор источника замером ---
    d_med = d_std = np.nan
    frac_big = np.nan
    if f_vec is not None and f_sca is not None:
        ok = np.isfinite(f_vec) & np.isfinite(f_sca)
        if ok.sum() > 1000:
            diff = f_vec[ok] - f_sca[ok]
            d_med, d_std = float(np.median(diff)), float(diff.std())
            frac_big = float((np.abs(diff) > 1.0).mean())
    # Пустой канал — не источник: у ARS_2026 тег S есть, а значений в нём нет,
    # и слепой выбор скаляра дал бы год из одних NaN.
    v_ok = f_vec is not None and np.isfinite(f_vec).any()
    s_ok = f_sca is not None and np.isfinite(f_sca).any()
    if v_ok and s_ok and np.isfinite(d_med):
        F, src = (f_vec, "vector") if abs(d_med) <= TOL else (f_sca, "scalar")
    elif v_ok and s_ok:
        # каналы есть, но не пересекаются по времени — сверить нечем
        F, src = f_vec, "vector(не сверен)"
    elif v_ok:
        F, src = f_vec, "vector(не сверен)"
    elif s_ok:
        F, src = f_sca, "scalar(не сверен)"
    else:
        F, src = (f_vec if f_vec is not None else f_sca), "ПУСТО"
    vector_ok = src.startswith("vector")

    spikes = 0
    if dtype == "reported":
        F, spikes = despike(F)
    if F is not None and np.isfinite(F).any():
        gross = np.isfinite(F) & (np.abs(F - np.nanmedian(F)) > GROSS_THR)
        if gross.any():
            F = F.copy(); F[gross] = np.nan
            spikes += int(gross.sum())
    needles = cut = 0
    if F is not None and year in train_years():
        F, needles = deneedle(F)
    if F is not None:
        F, cut = cut_bad(F, code, year)
    spikes += needles + cut

    n = minutes_in_year(year)
    # исходная длина запоминается ДО приведения к сетке: файл, залезающий в
    # следующий год, иначе молча обрежется и находка потеряется
    n_raw = 0 if F is None else int(F.size)
    F = to_grid(F, n).astype(np.float32)
    XYZ = [to_grid(a, n).astype(np.float32) if a is not None else np.full(n, np.nan, np.float32)
           for a in (X, Y, Z)]
    # чистка (иглы, грубые выбросы, сбойные интервалы) считалась по F; плохая
    # минута плоха во всех каналах, иначе компоненты остались бы с артефактами
    for a in XYZ:
        a[~np.isfinite(F)] = np.nan

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{code}_{year}.npz"),
             F=F, X=XYZ[0], Y=XYZ[1], Z=XYZ[2],
             code=code, year=np.int32(year), datatype=dtype,
             f_source=src, vector_ok=bool(vector_ok),
             scalar_elem=sca_name or "", diff_med=np.float32(d_med),
             lat=np.float32(float(meta.get("SensorLatitude", "nan"))),
             lon=np.float32(float(meta.get("SensorLongitude", "nan"))))

    order = [e for e in ("H", "D", "X", "Y", "Z", "S", "F", "G") if e in ch]
    fin = np.isfinite(F)
    return dict(code=code, year=year, dtype=dtype, elems="".join(order),
                src=src, sca=sca_name or "-", n=n, n_raw=n_raw,
                valid=float(fin.mean()),
                medF=float(np.median(F[fin])) if fin.any() else np.nan,
                d_med=d_med, d_std=d_std, frac_big=frac_big, spikes=spikes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="Arti", help="префикс имени файлов XML")
    ap.add_argument("--years", type=int, nargs="+", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SRC, f"{args.code}_*.xml")))
    if args.years:
        files = [f for f in files
                 if int(re.search(r"_(\d{4})", os.path.basename(f)).group(1)) in args.years]
    if not files:
        raise SystemExit(f"нет файлов {args.code}_*.xml в {SRC}")

    print(f"источник: {SRC}\nвыход:    {OUT}\nдопуск на расхождение: {TOL} нТл\n")
    print(f"{'год':>5}{'тип':>12}{'элем':>7}{'источник F':>18}{'валид':>8}"
          f"{'медиана F':>11}{'вект-скаляр':>13}{'СКО':>7}{'>1нТл':>7}{'засечки':>9}")
    print("-" * 108)
    rows = []
    for p in files:
        r = process(p, args)
        rows.append(r)
        dm = "            -" if not np.isfinite(r["d_med"]) else f"{r['d_med']:>13.3f}"
        ds = "      -" if not np.isfinite(r["d_std"]) else f"{r['d_std']:>7.2f}"
        fb = "      -" if not np.isfinite(r["frac_big"]) else f"{r['frac_big']:>7.1%}"
        mf = "        нет" if not np.isfinite(r["medF"]) else f"{r['medF']:>11.0f}"
        print(f"{r['year']:>5}{r['dtype']:>12}{r['elems']:>7}{r['src']:>18}"
              f"{r['valid']:>8.1%}{mf}{dm}{ds}{fb}{r['spikes']:>9}")

    sca = [r for r in rows if r["src"].startswith("scalar")]
    print(f"\nвсего лет: {len(rows)};  через скаляр: {len(sca)}"
          + (f" ({', '.join(str(r['year']) for r in sca)})" if sca else ""))
    trimmed = [r for r in rows if r["n_raw"] > r["n"]]
    if trimmed:
        print("ОБРЕЗАНЫ ПО КАЛЕНДАРЮ (файл залезал в следующий год — при разных "
              "сплитах это была бы утечка):")
        for r in trimmed:
            print(f"  {r['year']}: лишних {(r['n_raw'] - r['n']) / 1440:.1f} сут")
    empty = [r for r in rows if not np.isfinite(r["medF"])]
    if empty:
        print("НЕПРИГОДНЫ (ни одного конечного значения): "
              + ", ".join(str(r["year"]) for r in empty))
    # подозрительными считаем ТОЛЬКО годы, взятые по вектору: там расхождение
    # означает дефект, а у скалярных лет оно и есть причина выбора источника
    susp = [r for r in rows if r["src"].startswith("vector")
            and np.isfinite(r["frac_big"]) and r["frac_big"] > 0.005]
    if susp:
        print("ВНИМАНИЕ, приборы расходятся чаще 0.5% минут "
              "(смотреть отдельно, возможны засечки в одном из каналов):")
        for r in susp:
            print(f"  {r['year']}: {r['frac_big']:.1%} минут с |вектор-скаляр| > 1 нТл, "
                  f"СКО {r['d_std']:.1f} нТл")


if __name__ == "__main__":
    main()
