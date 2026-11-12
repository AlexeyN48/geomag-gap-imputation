r"""Вердикты утверждений У16–У24 (PROTOCOL.md, раздел 9) по таблицам bench_station_ci.py.

Код написан и зафиксирован ДО вскрытия тестовых лет. Он читает только готовые
таблицы интервалов и ничего не пересчитывает: правила вердиктов — те, что
записаны в протоколе, и меняться после вскрытия не должны (раздел 6 протокола).

Нужные таблицы на каждый набор <s> (val, test, test_hard):
    python -u scripts/bench_station_ci.py --split <s>
    python -u scripts/bench_station_ci.py --split <s> --stations ARS --per-length
    python -u scripts/bench_station_ci.py --split <s> --stations ARS --per-length --subset act
    python -u scripts/bench_seeds.py                      # seeds_summary.csv (для У24)

Запуск:
    python -u scripts/bench_verdict_cluster.py                      # валидация
    python -u scripts/bench_verdict_cluster.py --splits test test_hard

Исход утверждения на одном наборе: «подтверждено», «опровергнуто» или «не
доказано» (У22 — единственное с третьим исходом по существу). Итог по двум
тестовым годам — по правилу раздела 5: подтверждено в обоих — подтверждено;
опровергнуто в обоих — опровергнуто; подтверждено в одном и опровергнуто в
другом — «зависит от обстановки»; «не доказано» хотя бы в одном — «не доказано».
"""
import os
import re
import csv
import argparse

import bench_common as C

STATIONS = ["WNG", "CMO", "KAK", "HUA", "HER"]
NETS = ["unet", "saits", "timesnet", "imputeformer", "moment"]
INPUTS = ["F", "XYZ"]
SHORT, LONG, ALL = "≤120 мин", ">1000 мин", "все длины"
OK, BAD, UNPROVEN = "подтверждено", "опровергнуто", "не доказано"


def load(path):
    """{(подмножество, станция, вход, a, b, группа): строка}; None, если файла нет."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["subset"], r["station"], r["input"], r["a"], r["b"], r["group"]): r
                for r in csv.DictReader(f)}


def table_path(split, stations=None, subset="all"):
    return os.path.join(C.OUT, "station_ci_" + C.setname("", split)[1:]
                        + (f"_{stations}" if stations else "") + (f"_{subset}" if subset != "all" else "")
                        + ".csv")


def row(t, subset, st, inp, a, b, group):
    return t.get((subset, st, inp, a, b, group))


def check(items, want):
    """items — [(метка, строка или None)]; want(строка) -> bool. Возвращает (исход, нарушения, всего)."""
    bad = [lab for lab, r in items if r is None or not want(r)]
    return (OK if not bad else BAD), bad, len(items)


def U16(ars):
    items = []
    for inp in INPUTS:
        for n in NETS:
            for L, side in [(5, "b"), (15, "b"), (60, "b"), (1440, "a"), (2160, "a"),
                            (2880, "a"), (4320, "a")]:
                items.append(((n, inp, L, side),
                              row(ars, "all", "ARS", inp, f"{n}_local", "seadec", str(L))))
    bad = [lab for lab, r in items
           if r is None or r["verdict"] != ("b лучше" if lab[3] == "b" else "a лучше")]
    return (OK if not bad else BAD), bad, len(items)


def U17(st):
    return check([((n, c, i), row(st, "all", c, i, f"{n}_local", "seadec", LONG))
                  for n in NETS for c in STATIONS for i in INPUTS],
                 lambda r: r["verdict"] == "a лучше")


def U18(st):
    return check([((n, c, i), row(st, "all", c, i, f"{n}_local", "seadec", SHORT))
                  for n in NETS for c in STATIONS for i in INPUTS],
                 lambda r: r["verdict"] == "b лучше")


def U19(st):
    items = []
    for i in INPUTS:
        for c in STATIONS:
            if c in ("WNG", "HER"):
                continue                      # ±3 %: утверждением не является
            items.append(((c, i), row(st, "all", c, i, "unet_local", "seadec", ALL)))
    bad = []
    for (c, i), r in items:
        if r is None:
            bad.append((c, i)); continue
        if c == "CMO" and r["verdict"] != "a лучше":
            bad.append((c, i))
        if c in ("KAK", "HUA") and r["verdict"] == "a лучше":
            bad.append((c, i))
    return (OK if not bad else BAD), bad, len(items)


def U20(st):
    items, bad = [], []
    for c in STATIONS:
        r = row(st, "all", c, "F", "seadec", "pchip", LONG)
        items.append(c)
        if r is None:
            bad.append(c)
        elif c == "CMO" and not float(r["ci95_hi"]) < 15:
            bad.append(c)
        elif c != "CMO" and not float(r["ci95_lo"]) > 25:
            bad.append(c)
    return (OK if not bad else BAD), bad, len(items)


def U21(st):
    return check([(c, row(st, "all", c, "F", "moment_local", "moment_transfer", ALL))
                  for c in STATIONS], lambda r: r["verdict"] == "a лучше")


def U22(st):
    out, bad = [], []
    for c in STATIONS:
        r = row(st, "all", c, "F", "moment_local", "unet_local", ALL)
        if r is None:
            bad.append(c); out.append(BAD); continue
        lo, hi = float(r["ci95_lo"]), float(r["ci95_hi"])
        out.append(OK if (lo > -5 and hi < 5) else BAD if (lo > 5 or hi < -5) else UNPROVEN)
    if BAD in out:
        return BAD, [c for c, o in zip(STATIONS, out) if o == BAD], len(out)
    if UNPROVEN in out:
        return UNPROVEN, [c for c, o in zip(STATIONS, out) if o == UNPROVEN], len(out)
    return OK, [], len(out)


def U23(ars_act):
    return check([((n, L), row(ars_act, "act", "ARS", "F", f"{n}_local", "seadec", str(L)))
                  for n in NETS for L in (1440, 4320)], lambda r: r["verdict"] == "a лучше")


def U24(split):
    p = os.path.join(C.OUT, "seeds_summary.csv")
    if not os.path.exists(p):
        return None
    pat = re.compile(r"^unet(?:_(?P<code>[A-Z]{3}))?_best(?P<x>_xyz)?$")
    items, bad = {}, []
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            g = pat.match(r["model"])
            if not g or r["metric"] != "MAE" or r["gap"] != ALL or not r["set"].endswith("_" + split):
                continue
            code = g["code"] or C.CODE
            if r["set"] != C.setname(code, split):
                continue                      # только своя станция, не перенос
            key = (code, "XYZ" if g["x"] else "F")
            items[key] = 100 * float(r["std"]) / float(r["mean"])
    want = [(c, i) for c in [C.CODE] + STATIONS for i in INPUTS]
    for k in want:
        if k not in items or items[k] > 1.5:
            bad.append(k)
    return (OK if not bad else BAD), bad, len(want)


def combine(outs):
    """Итог по нескольким наборам (раздел 5 протокола)."""
    s = [o for o in outs if o is not None]
    if not s:
        return "нет данных"
    if any(o == UNPROVEN for o in s):
        return UNPROVEN
    if all(o == OK for o in s):
        return OK
    if all(o == BAD for o in s):
        return BAD
    return "зависит от обстановки"


CLAIMS = [
    ("У16", "ARS: seadec точнее сетей на 5–60 мин, сети точнее seadec от 1440 мин"),
    ("У17", "станции: сети на месте точнее seadec на длинных дырах, оба входа"),
    ("У18", "станции: seadec точнее сетей на коротких дырах, оба входа"),
    ("У19", "станции: по сводной MAE U-Net лучше seadec на CMO, не лучше на KAK и HUA"),
    ("У20", "станции: seadec над PCHIP на длинных дырах — мало на CMO (≤15 %), много на остальных (≥25 %)"),
    ("У21", "станции: MOMENT на месте точнее своего переноса с Арти (вход F)"),
    ("У22", "станции: MOMENT на месте и U-Net на месте различаются не более чем на 5 % (вход F)"),
    ("У23", "ARS: в возмущённой половине окон сети точнее seadec на 1440 и 4320 мин"),
    ("У24", "разброс по сидам U-Net не больше 1.5 % среднего на каждой станции и входе"),
]


def verdicts(split):
    st = load(table_path(split))
    ars = load(table_path(split, "ARS"))
    act = load(table_path(split, "ARS", "act"))
    res = {}
    for code, fn, t in [("У16", U16, ars), ("У17", U17, st), ("У18", U18, st), ("У19", U19, st),
                        ("У20", U20, st), ("У21", U21, st), ("У22", U22, st), ("У23", U23, act)]:
        res[code] = fn(t) if t is not None else None
    res["У24"] = U24(split)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=["val"])
    a = ap.parse_args()
    per = {s: verdicts(s) for s in a.splits}
    for code, text in CLAIMS:
        print(f"\n{code}. {text}")
        outs = []
        for s in a.splits:
            r = per[s][code]
            if r is None:
                print(f"   {s}: нет таблицы")
                outs.append(None)
                continue
            outcome, bad, n = r
            outs.append(outcome)
            extra = f" — нарушений {len(bad)} из {n}: {bad[:6]}" if bad else f" ({n} проверок)"
            print(f"   {s}: {outcome}{extra}")
        if len(a.splits) > 1:
            print(f"   ИТОГ: {combine(outs)}")


if __name__ == "__main__":
    main()
