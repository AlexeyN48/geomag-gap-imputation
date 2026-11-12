r"""Парные интервалы для сравнения методов: сезонный seadec, PCHIP, сети, MOMENT.

Среднее ± стандартное отклонение по сидам (bench_seeds.py) показывает разброс
обучения, но не отвечает, различимы ли два метода: окна одной недели зависимы
(общий шторм), а сидов всего пять. Здесь для каждой станции (и ARS) и каждого
входа (F, XYZ) сравниваются пары методов на ОДНИХ И ТЕХ ЖЕ окнах:

  сеть ↔ seadec            каждая из сетей, обученных на месте (U-Net, SAITS,
                           TimesNet, ImputeFormer, MOMENT)
  MOMENT с Арти ↔ seadec   перенос без дообучения на станции
  seadec ↔ PCHIP           чем сезонный профиль лучше простой интерполяции
  MOMENT ↔ U-Net           две сильные сети друг с другом; MOMENT на станции ↔
                           его перенос с Арти

Величина — относительный выигрыш a над b:  1 − MAE_a / MAE_b  (больше нуля — a
точнее), где MAE — геометрическое среднее по длинам: по всем 12, по трём
режимам длины (≤120, 120–1000, >1000 мин) и, с --per-length, по каждой длине.

Как считаются интервалы (бутстрэп, 100000 повторов):
  • окна ресэмплируются БЛОКАМИ — календарными неделями (номер блока записан в
    дампе, bench_run.py); один набор недель берётся для всех длин и всех
    методов, поэтому пара остаётся парной, а шторм одной недели не разносится
    на «независимые» окна;
  • сиды ресэмплируются с возвращением внутри метода (у каждого свой набор),
    так что интервал включает и разброс обучения;
  • «значимо» = 95 %-й интервал выигрыша не содержит нуля (как в разделе 4
    протокола). Рядом даётся интервал с поправкой Бонферрони на число проверок
    запуска (уровень 5 %/m) и вердикт по нему: если вердикт утверждения от
    поправки меняется, отчёт обязан это сказать.

Чего интервал не покрывает: выбор архитектур и гиперпараметров на Арти, смену
года (один год на набор) и расхождение станций. Он говорит лишь о том, насколько
результат на этих окнах устойчив к выбору недель и сидов.

--subset act|qui берёт возмущённую (метка возмущённости выше медианы, L{L}_act
дампа) или спокойную половину окон — то же определение, что в У7.

Запуск (из корня; нужны дампы bench_jobs.py: baselines, seedval / stationval):
    python -u scripts/bench_station_ci.py                          # пять станций, валидация
    python -u scripts/bench_station_ci.py --stations ARS --per-length
    python -u scripts/bench_station_ci.py --split test
    -> data/<сплит>/station_ci_<набор>[_<станции>][_<подмножество>].csv
"""
import os
import csv
import zlib
import argparse

import numpy as np

import bench_common as C
from bench_regimes import REGIMES

SEED_DIR = os.path.join(C.OUT, "seeds")
STATIONS = ["WNG", "CMO", "KAK", "HUA", "HER"]
INPUTS = [("F", False), ("XYZ", True)]
NETS = ["unet", "saits", "timesnet", "imputeformer", "moment"]
BASE = "seadec"
PCHIP = "pchip"


def seed_name(arch, code, comp, seed):
    """Как bench_jobs.seed_name: <арх>[_<станция>]_best[_xyz]_s<сид>."""
    return (f"{arch}" + (f"_{code}" if code != C.CODE else "") + "_best"
            + ("_xyz" if comp else "") + f"_s{seed}")


def load_windows(path):
    """{длина: (MAE по каждому окну, номер блока, начало окна, метка возмущённости)}."""
    out = {}
    with np.load(path, allow_pickle=False) as d:
        m = int(d["margin"])
        for L in (int(x) for x in d["lengths"]):
            if f"L{L}_block" not in d:
                raise SystemExit(f"в {os.path.basename(path)} нет L{L}_block — "
                                 "дамп сделан старым bench_run.py, пересчитайте его")
            t = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
            p = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
            out[L] = (np.nanmean(np.abs(p - t), axis=1),
                      d[f"L{L}_block"].astype(np.int64),
                      d[f"L{L}_win0"].astype(np.int64),
                      d[f"L{L}_act"].astype(np.float64))
    return out


class Method:
    """Метод на одном наборе: ошибки окон по сидам (детерминированный — один «сид»)."""

    def __init__(self, label, paths, subset="all"):
        self.label = label
        wins = [load_windows(p) for p in paths]
        self.n_seeds = len(wins)
        self.lengths = sorted(wins[0])
        self.err, self.block, self.win0 = {}, {}, {}
        for L in self.lengths:
            act = wins[0][L][3]
            keep = {"all": np.ones(len(act), bool), "act": act > np.median(act),
                    "qui": act <= np.median(act)}[subset]
            for w in wins[1:]:
                if not np.array_equal(w[L][2], wins[0][L][2]):
                    raise SystemExit(f"{label}: окна сидов не совпадают на длине {L}")
            self.err[L] = np.stack([w[L][0][keep] for w in wins])
            self.block[L] = wins[0][L][1][keep]
            self.win0[L] = wins[0][L][2][keep]


def same_windows(a, b):
    return (a.lengths == b.lengths and
            all(np.array_equal(a.win0[L], b.win0[L]) for L in a.lengths))


def point_mae(m):
    """MAE по длинам на всех окнах, среднее по сидам."""
    return {L: float(m.err[L].mean()) for L in m.lengths}


def geomean(by_L, Ls):
    return float(np.exp(np.mean(np.log([by_L[L] for L in Ls]))))


def boot_mae(m, blocks, counts, rng):
    """(повторы × длины): MAE метода при ресэмпле недель и сидов.

    counts[r, b] — сколько раз неделя b вытянута в повторе r (общие для всех
    методов). Сиды — свои для метода: ws[r, s] — доля повтора r, где вытянут сид s."""
    reps = counts.shape[0]
    pick = rng.integers(0, m.n_seeds, size=(reps, m.n_seeds))
    ws = np.stack([(pick == s).mean(axis=1) for s in range(m.n_seeds)], axis=1)
    out = np.empty((reps, len(m.lengths)))
    for j, L in enumerate(m.lengths):
        idx = np.searchsorted(blocks, m.block[L])
        n_b = np.bincount(idx, minlength=len(blocks)).astype(np.float64)
        s_b = np.stack([np.bincount(idx, weights=m.err[L][s], minlength=len(blocks))
                        for s in range(m.n_seeds)], axis=1)          # (блоки × сиды)
        num = ((counts @ s_b) * ws).sum(axis=1)
        den = counts @ n_b
        out[:, j] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def collect(code, comp, split, seeds, base_dir, seed_dir, subset):
    """Методы станции code при входе comp: {метка: Method}; нет дампов — метода нет."""
    setn = C.setname(code, split)
    found, notes = {}, []

    def add(label, paths):
        miss = [p for p in paths if not os.path.exists(p)]
        if miss and len(miss) == len(paths):
            notes.append(f"{label}: нет дампов")
        elif miss:
            notes.append(f"{label}: нет {len(miss)} из {len(paths)} дампов — берутся имеющиеся")
        have = [p for p in paths if os.path.exists(p)]
        if have:
            found[label] = Method(label, have, subset)

    def seed_paths(arch, train_code):
        return [os.path.join(seed_dir, f"dump_{seed_name(arch, train_code, comp, s)}_{setn}.npz")
                for s in seeds]

    for b in (BASE, PCHIP):
        add(b, [os.path.join(base_dir, f"dump_{b}_{setn}.npz")])
    for arch in NETS:
        add(f"{arch}_local", seed_paths(arch, code))
    if code != C.CODE:                  # на ARS «перенос» и «на месте» — одно и то же
        add("moment_transfer", seed_paths("moment", C.CODE))
    return found, notes


def pairs(methods):
    """Пары (a, b): «a лучше b» — выигрыш a над b."""
    out = []
    for lab in methods:
        if (lab.endswith("_local") or lab == "moment_transfer") and BASE in methods:
            out.append((lab, BASE))
    if BASE in methods and PCHIP in methods:
        out.append((BASE, PCHIP))
    for a, b in [("moment_local", "unet_local"), ("moment_transfer", "unet_local"),
                 ("moment_local", "moment_transfer")]:
        if a in methods and b in methods:
            out.append((a, b))
    return out


def verdict(lo, hi):
    return "a лучше" if lo > 0 else "b лучше" if hi < 0 else "не различимы"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="val", help="val | test | test_hard")
    ap.add_argument("--reps", type=int, default=100000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--stations", nargs="+", default=STATIONS)
    ap.add_argument("--subset", default="all", choices=["all", "act", "qui"],
                    help="все окна | возмущённая половина | спокойная половина")
    ap.add_argument("--per-length", action="store_true", help="выигрыш и по каждой длине")
    ap.add_argument("--base-dir", default=C.OUT, help="где лежат дампы seadec и PCHIP")
    ap.add_argument("--seed-dir", default=SEED_DIR, help="где лежат дампы сетей по сидам")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows_in = []          # (станция, вход, a, b, Method a, Method b)
    for code in a.stations:
        for inp, comp in INPUTS:
            methods, notes = collect(code, comp, a.split, a.seeds, a.base_dir,
                                     a.seed_dir, a.subset)
            for n in notes:
                print(f"  {code} {inp}: {n}")
            for x, y in pairs(methods):
                if not same_windows(methods[x], methods[y]):
                    raise SystemExit(f"{code} {inp}: у {x} и {y} разные окна — сравнивать нельзя")
                rows_in.append((code, inp, x, y, methods[x], methods[y]))
    if not rows_in:
        raise SystemExit("нет ни одной пары дампов — сначала этапы baselines и stationval")

    def groups_of(m):
        gs = [("все длины", list(m.lengths))] + [(g, [L for L in Ls if L in m.lengths])
                                                  for g, Ls in REGIMES]
        if a.per_length:
            gs += [(str(L), [L]) for L in m.lengths]
        return [(g, Ls) for g, Ls in gs if Ls]

    m_fam = sum(len(groups_of(r[4])) for r in rows_in)
    print(f"сравнений: {len(rows_in)} пар, {m_fam} проверок; "
          f"поправка Бонферрони: уровень {a.alpha}/{m_fam}")
    q_pt = [100 * a.alpha / 2, 100 * (1 - a.alpha / 2)]
    q_fam = [100 * a.alpha / (2 * m_fam), 100 * (1 - a.alpha / (2 * m_fam))]
    if q_fam[0] * a.reps / 100 < 5:
        print(f"  ВНИМАНИЕ: при {a.reps} повторах поправленный хвост — менее 5 значений; "
              "увеличьте --reps или смотрите только поточечный интервал")

    tag = "_".join(a.stations) if a.stations != STATIONS else ""
    out = a.out or os.path.join(C.OUT, "station_ci_" + C.setname("", a.split)[1:]
                                + (f"_{tag}" if tag else "")
                                + (f"_{a.subset}" if a.subset != "all" else "") + ".csv")
    cache, counts_by_station, table = {}, {}, []
    for code, inp, x, y, mx, my in rows_in:
        if code not in counts_by_station:        # недели — общие для всех методов станции
            blocks = np.unique(np.concatenate([mx.block[L] for L in mx.lengths]))
            rng = np.random.default_rng(zlib.crc32(f"{code}|{a.split}|{a.subset}".encode()))
            draw = rng.integers(0, len(blocks), size=(a.reps, len(blocks)))
            counts = np.stack([np.bincount(row, minlength=len(blocks)) for row in draw]
                              ).astype(np.float64)
            counts_by_station[code] = (blocks, counts)
        blocks, counts = counts_by_station[code]

        def boot(m, lab):
            key = (code, inp, lab)
            if key not in cache:
                rng = np.random.default_rng(
                    zlib.crc32("|".join(key + (a.split, a.subset)).encode()))
                cache[key] = boot_mae(m, blocks, counts, rng)
            return cache[key]

        Bx, By = boot(mx, x), boot(my, y)
        px, py = point_mae(mx), point_mae(my)
        for gname, Ls in groups_of(mx):
            j = [mx.lengths.index(L) for L in Ls]
            gx = np.exp(np.log(Bx[:, j]).mean(axis=1))
            gy = np.exp(np.log(By[:, j]).mean(axis=1))
            gain = 100 * (1 - gx / gy)
            lo, hi = np.nanpercentile(gain, q_pt)
            flo, fhi = np.nanpercentile(gain, q_fam)
            ax, ay = geomean(px, Ls), geomean(py, Ls)
            table.append([a.split, a.subset, code, inp, x, y, gname, f"{ax:.4f}", f"{ay:.4f}",
                          f"{100 * (1 - ax / ay):.2f}", f"{lo:.2f}", f"{hi:.2f}",
                          verdict(lo, hi), f"{flo:.2f}", f"{fhi:.2f}", verdict(flo, fhi),
                          mx.n_seeds, my.n_seeds, len(blocks), m_fam])
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "subset", "station", "input", "a", "b", "group", "mae_a", "mae_b",
                    "gain_pct", "ci95_lo", "ci95_hi", "verdict", "ci_family_lo",
                    "ci_family_hi", "verdict_family", "seeds_a", "seeds_b", "weeks",
                    "family_m"])
        w.writerows(table)
    print(f"сохранено: {out}\n")

    print("Выигрыш a над b по всем длинам, %  [95 % интервал]  вердикт / по поправленному:")
    for r in table:
        if r[6] == "все длины":
            print(f"  {r[2]} {r[3]:<4}{r[4]:<17}vs {r[5]:<8}{float(r[9]):+6.1f}%  "
                  f"[{float(r[10]):+6.1f}; {float(r[11]):+6.1f}]  {r[12]} / {r[15]}")


if __name__ == "__main__":
    main()
