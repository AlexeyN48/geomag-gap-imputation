r"""Сборка данных бенчмарка: декларативный рецепт + проверка целостности.

Зачем отдельный файл. Данные строятся не одной командой, а цепочкой шагов, и
часть из них — точечные правки конкретных станций и лет, найденные вручную
(докачка недостающих суток, вырезание сбойных интервалов, выравнивание станций
между собой). Пока они жили в истории команд, любая пересборка молча теряла их,
и расхождение между данными на диске и посчитанными результатами обнаруживалось
случайно, недели спустя.

Здесь рецепт объявлен данными (STAGES), а исполнитель — общий. Плюс lock-файл
data/DATA_MANIFEST.json с контрольной суммой каждого npz: в любой момент можно
спросить, соответствуют ли данные рецепту, ничего не меняя.

    python -u bench_data.py --check        сверить диск с манифестом (ничего не пишет)
    python -u bench_data.py --snapshot     записать манифест по готовым данным
    python -u bench_data.py                применить недостающие шаги и обновить манифест
    python -u bench_data.py --rebuild      то же, но начиная с пересборки npz из XML
    python -u bench_data.py --dry-run      показать план, ничего не делать

Шаг --rebuild разрушителен и поэтому включается явно: bench_prep пересобирает
npz с нуля, стирая результаты всех последующих шагов. Остальные шаги
идемпотентны, повторный запуск их безвреден.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
import datetime as dt

import numpy as np

import bench_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(C.DATA, "DATA_MANIFEST.json")

# Станции: код IAGA -> префикс имён XML в dataset_intermagnet
STATIONS = {"ARS": "Arti", "WNG": "WNG", "CMO": "CMO",
            "KAK": "KAK", "HUA": "HUA", "HER": "HER"}

# Годы, которые выравниваются между станциями. Список ЯВНЫЙ и намеренно не
# читает C.SPLIT: иначе при другом активном сплите выровнялся бы другой набор.
# 2026 сюда не входит — у ARS он обрывается 15 июля, у CMO есть только
# август-сентябрь, пересечение пустое, и равнение на ARS обнулило бы CMO.
ALIGN_YEARS = [2013, 2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# ---------------------------------------------------------------- рецепт
# Каждый шаг: имя, пояснение (зачем он вообще есть) и команда.
STAGES = [
    dict(
        name="prep",
        destructive=True,
        why="сборка npz из XML: выбор источника F по сверке вектора со скаляром, "
            "грубые выбросы, засечки в reported, иглы в годах CLEAN_YEARS, "
            "сбойные интервалы BAD_INTERVALS, маска на все каналы",
        cmds=[["bench_prep.py", "--code", pref] for pref in
              dict.fromkeys(STATIONS.values())],
    ),
    dict(
        name="gin-ars-2024",
        why="у ARS 2024 не хватало 34.8 суток, включая бурю 10-11 октября — "
            "в ГИН они есть (quasi-definitive, компоненты сходятся со скаляром)",
        cmds=[["bench_gin.py", "--code", "ARS", "--year", "2024", "--auto"]],
    ),
    dict(
        name="gin-cmo-2025",
        why="у CMO 2025 отсутствовали январь и февраль целиком; в ГИН они есть "
            "как Provisional. Допуск поднят, потому что базис там смещён на "
            "22 нТл — для ОБУЧАЮЩЕГО года это безвредно, сдвиг снимается "
            "центрированием окна (bench_common.comp_ok)",
        cmds=[["bench_gin.py", "--code", "CMO", "--year", "2025",
               "--scan", "--tol", "30"]],
    ),
    dict(
        name="gin-2026",
        why="2026 год не скачивался вовсе; собирается из ГИН помесячно",
        cmds=[["bench_gin.py", "--code", c, "--year", "2026", "--create", "--scan"]
              for c in ("WNG", "CMO", "KAK", "HUA", "HER")],
    ),
    dict(
        name="cut-cmo-2025-baseline-step",
        why="январь-февраль CMO 2025 пришли из Provisional (базис +4 нТл), "
            "остальной год quasi-definitive (-22 нТл) — на стыке ступень 26 нТл. "
            "Вырезается ровно одно окно, тогда ни одно обучающее окно не может "
            "захватить обе стороны ступени",
        call="cut_window_at",
        args=dict(code="CMO", year=2025, date="2025-03-01"),
    ),
    dict(
        name="align",
        why="в обратном эксперименте каждая станция учится на себе, и объём "
            "обучающих данных у них был разный (у ARS 8.39 года, у остальных "
            "8.97-9.01). Выравнивание под ARS убирает перекос с 7.4 % до 0.4 %",
        cmds=[["bench_align.py", "--years"] + [str(y) for y in ALIGN_YEARS]],
    ),
]


# ---------------------------------------------------------------- действия
def cut_window_at(code, year, date, dry=False):
    """Вырезать ровно одно окно (C.W минут) вокруг указанной даты во всех
    каналах. Идемпотентно: если там уже пусто, ничего не делает."""
    p = os.path.join(C.DATA, f"{code}_{year}.npz")
    if not os.path.exists(p):
        return f"нет {os.path.basename(p)}"
    b = int((dt.date.fromisoformat(date) - dt.date(year, 1, 1)).days) * 1440
    i, j = b - C.W // 2, b + C.W // 2
    with np.load(p, allow_pickle=False) as d:
        cur = {k: d[k] for k in d.files}
    n = int(np.isfinite(cur["F"][i:j]).sum())
    if n == 0:
        return "уже вырезано"
    if dry:
        return f"вырезать {n} минут"
    for k in ("F", "X", "Y", "Z"):
        if k in cur:
            cur[k] = cur[k].copy()
            cur[k][i:j] = np.nan
    np.savez(p, **cur)
    return f"вырезано {n} минут"


def run_cmd(cmd, dry=False):
    if dry:
        return "python -u " + " ".join(cmd)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("IGF_SPLIT", None)          # сборка данных не зависит от сплита
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, cmd[0])] + cmd[1:],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=HERE)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
        raise SystemExit(f"ОШИБКА на {' '.join(cmd)}:\n{tail}")
    return (r.stdout.strip().splitlines() or ["ok"])[-1]


# ---------------------------------------------------------------- манифест
def file_state(p):
    """Отпечаток файла: sha256 плюс то, что важно по смыслу — покрытие и
    сходимость вектора со скаляром. Хеша хватило бы для обнаружения правки,
    но не для понимания, что именно поехало."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with np.load(p, allow_pickle=False) as d:
        F = d["F"].astype(np.float64)
        ok = np.isfinite(F)
        st = dict(sha256=h.hexdigest()[:16], minutes=int(ok.sum()),
                  coverage=round(float(ok.mean()), 5))
        if all(k in d.files for k in "XYZ"):
            v = np.sqrt(sum(d[k].astype(np.float64) ** 2 for k in "XYZ"))
            m = ok & np.isfinite(v)
            st["dF_median"] = round(float(np.median(np.abs(F[m] - v[m]))), 3) if m.any() else None
    return st


def scan():
    out = {}
    for f in sorted(os.listdir(C.DATA)):
        if f.endswith(".npz") and "_" in f and f.split("_")[0] in STATIONS:
            out[f] = file_state(os.path.join(C.DATA, f))
    return out


def git_rev():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def write_manifest(files, stages):
    doc = dict(built=dt.datetime.now().isoformat(timespec="seconds"),
               git=git_rev(), window=int(C.W),
               clean_years=sorted(__import__("bench_prep").CLEAN_YEARS),
               align_years=ALIGN_YEARS, stages=stages, files=files)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


def check():
    """Сверить диск с манифестом. Возвращает число расхождений."""
    if not os.path.exists(MANIFEST):
        print("манифеста нет — сначала соберите данные: python -u bench_data.py")
        return -1
    old = json.load(open(MANIFEST, encoding="utf-8"))
    cur, ref = scan(), old["files"]
    add = sorted(set(cur) - set(ref))
    gone = sorted(set(ref) - set(cur))
    diff = [f for f in sorted(set(cur) & set(ref)) if cur[f]["sha256"] != ref[f]["sha256"]]
    print(f"манифест от {old['built']}"
          + (f", код {old['git']}" if old.get("git") else ""))
    print(f"файлов в манифесте {len(ref)}, на диске {len(cur)}")
    for f in diff:
        a, b = ref[f], cur[f]
        print(f"  ИЗМЕНЁН {f}: минут {a['minutes']} -> {b['minutes']}, "
              f"покрытие {a['coverage']:.3f} -> {b['coverage']:.3f}")
    for f in add:
        print(f"  НОВЫЙ   {f}: минут {cur[f]['minutes']}")
    for f in gone:
        print(f"  ПРОПАЛ  {f}")
    bad = len(diff) + len(add) + len(gone)
    print("данные соответствуют манифесту" if bad == 0
          else f"расхождений: {bad} — результаты, посчитанные раньше, могли устареть")
    return bad


# ---------------------------------------------------------------- сборка
def build(args):
    done = []
    for st in STAGES:
        if st.get("destructive") and not args.rebuild:
            print(f"[{st['name']}] пропуск (нужен --rebuild): {st['why']}")
            continue
        print(f"[{st['name']}] {st['why']}")
        t0 = time.time()
        if "call" in st:
            print("   ", globals()[st["call"]](dry=args.dry_run, **st["args"]))
        else:
            for cmd in st["cmds"]:
                print("   ", run_cmd(cmd, dry=args.dry_run))
        done.append(dict(name=st["name"], seconds=round(time.time() - t0, 1)))
    if args.dry_run:
        print("\n--dry-run: ничего не изменено")
        return
    write_manifest(scan(), done)
    print(f"\nманифест обновлён: {os.path.basename(MANIFEST)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="только сверить, ничего не менять")
    ap.add_argument("--rebuild", action="store_true",
                    help="начать с пересборки npz из XML (РАЗРУШИТЕЛЬНО)")
    ap.add_argument("--snapshot", action="store_true",
                    help="записать манифест по текущему состоянию диска, "
                         "не выполняя шагов (для уже собранных данных)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(1 if check() else 0)
    if a.snapshot:
        files = scan()
        write_manifest(files, [dict(name="snapshot",
                                    note="манифест снят с готовых данных, шаги не выполнялись")])
        print(f"снимок записан: {len(files)} файлов -> {os.path.basename(MANIFEST)}")
        return
    build(a)


if __name__ == "__main__":
    main()
