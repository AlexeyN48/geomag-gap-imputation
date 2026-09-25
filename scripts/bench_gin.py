r"""Докачка недостающих суток из ГИН ИНТЕРМАГНЕТ в готовый data/<код>_<год>.npz.

Зачем: часть скачанных XML неполна. У ARS 2024 не было 18.09-05.10 и 08.10-23.10
(32 суток, включая бурю 10-11 октября) — в ГИН эти сутки есть. Скрипт скачивает
IAGA-2002, разбирает элементы по заголовку (встречаются XYZF, HDZF, XYZG),
сверяет векторный модуль со скалярным и вклеивает ТОЛЬКО в те минуты, где у нас
дыра. Уже имеющиеся значения не трогаются, так что повторный запуск безвреден.

Порядок в пайплайне: bench_prep.py -> bench_gin.py -> всё остальное.

Запуск:  cd bench_arti/scripts
         python -u bench_gin.py --code ARS --year 2024 --from 2024-09-18 --days 18
         python -u bench_gin.py --code ARS --year 2024 --auto   # найти дыры самому
"""
import os
import sys
import argparse
import datetime as dt
import subprocess
import numpy as np

import bench_common as C

URL = ("https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&format=IAGA2002"
       "&testObsys=0&observatoryIagaCode={code}&samplesPerDay=minute"
       "&publicationState=Best+available&dataStartDate={start}&dataDuration={days}")
MISSING = 99998.0        # значения >= этого в IAGA означают пропуск
DELTA_MAX = 1000.0       # |G| ниже этого => G это поправка dF, а не модуль
TOL = 1.0                # нТл: допуск на расхождение вектора и скаляра


def fetch(code, start, days, timeout=300):
    r = subprocess.run(["curl", "-sk", "--max-time", str(timeout),
                        URL.format(code=code, start=start, days=days)],
                       capture_output=True, text=True)
    return r.stdout


def parse(txt, year):
    """IAGA-2002 -> (минуты от начала года, X, Y, Z, F). None, если пусто."""
    elems, rows = None, []
    for line in txt.splitlines():
        if line.startswith("DATE"):
            elems = [c[3:] for c in line.split()[3:7]]        # ARSX -> X
        elif len(line.split()) >= 7 and line[:2] == "20":
            rows.append(line.split())
    if not elems or not rows:
        return None
    a = np.array([[float(x) for x in r[3:7]] for r in rows], dtype=np.float64)
    a[a >= MISSING] = np.nan
    col = dict(zip(elems, a.T))
    if set("XYZ") <= set(col):
        X, Y, Z = col["X"], col["Y"], col["Z"]
    elif set("HDZ") <= set(col):
        rad = np.deg2rad(col["D"] / 60.0)                     # D в угловых минутах
        X, Y, Z = col["H"] * np.cos(rad), col["H"] * np.sin(rad), col["Z"]
    else:
        raise SystemExit(f"неожиданные элементы: {elems}")
    v = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    if "F" in col or "S" in col:
        F = col["F"] if "F" in col else col["S"]
    elif "G" in col:                                          # либо dF, либо модуль
        med = np.nanmedian(np.abs(col["G"]))
        F = v + col["G"] if med < DELTA_MAX else col["G"]
    else:
        F = np.full(len(a), np.nan)
    idx = np.array([(dt.date(*map(int, r[0].split("-"))) - dt.date(year, 1, 1)).days * 1440
                    + int(r[1][:2]) * 60 + int(r[1][3:5]) for r in rows])
    return idx, X, Y, Z, F, "".join(elems)


def blank(code, year):
    """Пустой год в формате bench_prep: всё NaN, метаданные с соседнего года."""
    n = 1440 * ((dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days)
    meta = {}
    for y in range(year - 1, year - 6, -1):
        q = os.path.join(C.DATA, f"{code}_{y}.npz")
        if os.path.exists(q):
            with np.load(q, allow_pickle=False) as d:
                meta = {k: d[k] for k in ("lat", "lon")}
            break
    out = {k: np.full(n, np.nan, np.float32) for k in ("F", "X", "Y", "Z")}
    out.update(code=np.array(code), year=np.int32(year),
               datatype=np.array("reported"), f_source=np.array("ГИН"),
               vector_ok=np.bool_(True), scalar_elem=np.array(""),
               diff_med=np.float32(np.nan),
               lat=meta.get("lat", np.float32(np.nan)),
               lon=meta.get("lon", np.float32(np.nan)))
    return out


def holes(F, min_len=1440):
    """Дыры длиннее min_len минут -> [(начало, конец)] в минутах от начала года."""
    m = (~np.isfinite(F)).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(st, en) if e - s >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=C.CODE)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--from", dest="start", help="дата начала, ГГГГ-ММ-ДД")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--auto", action="store_true",
                    help="найти дыры длиннее суток и докачать их все")
    ap.add_argument("--scan", action="store_true",
                    help="пройти год помесячно и закрыть все дыры, включая мелкие "
                         "(дороже по запросам, зато ловит одноминутные выпадения)")
    ap.add_argument("--min-gap", type=int, default=1440,
                    help="минимальная длина дыры для --auto, мин")
    ap.add_argument("--tol", type=float, default=TOL,
                    help="допуск |F-|v||, нТл. По умолчанию 1: столько у правильно "
                         "обработанного года. Для ОБУЧАЮЩЕГО года можно поднять — "
                         "постоянный сдвиг базиса снимается центрированием окна "
                         "(bench_common.comp_ok); для тестовых и валидационного нельзя")
    ap.add_argument("--create", action="store_true",
                    help="создать файл года с нуля, если его нет (весь год качается из ГИН)")
    ap.add_argument("--dry-run", action="store_true", help="только показать, не писать")
    a = ap.parse_args()

    p = os.path.join(C.DATA, f"{a.code}_{a.year}.npz")
    if not os.path.exists(p):
        if not a.create:
            raise SystemExit(f"нет {p} (добавьте --create, чтобы скачать год целиком)")
        print(f"файла нет, создаю пустой {a.code} {a.year} и качаю год целиком")
        cur = blank(a.code, a.year)
        a.scan = True
    else:
        with np.load(p, allow_pickle=False) as d:
            cur = {k: d[k] for k in d.files}
    n = cur["F"].size
    before = int(np.isfinite(cur["F"]).sum())

    if a.scan:
        jan1 = dt.date(a.year, 1, 1)
        jobs = []
        for m in range(1, 13):
            d0 = dt.date(a.year, m, 1)
            d1 = dt.date(a.year + 1, 1, 1) if m == 12 else dt.date(a.year, m + 1, 1)
            i, j = (d0 - jan1).days * 1440, min((d1 - jan1).days * 1440, n)
            if i < n and not np.isfinite(cur["F"][i:j]).all():
                jobs.append((d0.isoformat(), (d1 - d0).days))
        if not jobs:
            print("дыр нет"); return
    elif a.auto:
        jan1 = dt.date(a.year, 1, 1)
        jobs = [((jan1 + dt.timedelta(days=s // 1440)).isoformat(),
                 int(np.ceil((e - s) / 1440)) + 1)
                for s, e in holes(cur["F"], min_len=a.min_gap)]
        if not jobs:
            print(f"дыр длиннее {a.min_gap} мин нет"); return
    else:
        if not a.start:
            raise SystemExit("нужен --from или --auto")
        jobs = [(a.start, a.days)]

    got = 0
    for start, days in jobs:
        out = parse(fetch(a.code, start, days), a.year)
        if out is None:
            print(f"{start} +{days} сут: ответ пустой"); continue
        idx, X, Y, Z, F, elems = out
        idx = np.clip(idx, 0, n - 1)
        v = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        ok = np.isfinite(v) & np.isfinite(F)
        dif = float(np.median(np.abs(F[ok] - v[ok]))) if ok.any() else np.nan
        fin = np.isfinite(F) & np.isfinite(v)
        hole = ~np.isfinite(cur["F"][idx])
        sel = hole & fin
        print(f"{start} +{days:2} сут: элементы {elems}, с данными {int(fin.sum()):6}, "
              f"|F-|v|| {dif:7.3f} нТл, в наши дыры попадает {int(sel.sum()):6}")
        if not (dif <= a.tol):
            print("   пропуск: вектор не сходится со скаляром, компоненты негодны")
            continue
        if a.dry_run or not sel.any():
            continue
        put = idx[sel]
        for k, arr in (("F", F), ("X", X), ("Y", Y), ("Z", Z)):
            cur[k] = cur[k].copy()
            cur[k][put] = arr[sel].astype(np.float32)
        got += int(sel.sum())

    after = int(np.isfinite(cur["F"]).sum())
    print(f"\nпокрытие {a.code} {a.year}: было {before / n * 100:.1f} %, "
          f"стало {after / n * 100:.1f} %  (+{got} минут)")
    if got and not a.dry_run:
        np.savez(p, **cur)
        print(f"перезаписан {os.path.basename(p)}")


if __name__ == "__main__":
    main()
