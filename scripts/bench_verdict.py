r"""Вердикты по утверждениям протокола слепой проверки (PROTOCOL.md).

Скрипт написан ДО вскрытия тестовых лет и проверен на валидации: прогнанный
на 2023 годе, он должен воспроизвести «основания», записанные в протоколе.
Каждое место, где слова протокола допускали два прочтения, разрешено здесь
явно и вынесено в PROTOCOL.md, раздел «Операционные определения». Код между
вскрытием и вердиктом не правится (протокол, раздел 6).

Порядок по протоколу: дампы -> метрики -> рейтинги -> вердикты. Рейтинги
(bench_regimes) должны быть построены с метками Fbest/XYZbest для ARS и F/XYZ
для станций — так же, как на валидации.

Запуск:
  проверка на валидации:   python -u bench_verdict.py --split val
  тестовый год:            python -u bench_verdict.py --split test --test-ok
  итог по двум годам:      python -u bench_verdict.py --combine --test-ok

Предохранитель --test-ok нужен, чтобы тестовые числа не вывелись случайно,
пока протокол и этот код ещё правятся.
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np

import bench_common as C
import bench_doc as D

FIN = ["unet", "saits", "timesnet", "imputeformer", "tsmixerx"]
ALL = FIN + ["nhits", "dlinear", "nbeatsx", "segrnn", "tide"]
ST = ["WNG", "CMO", "KAK", "HUA", "HER"]
CODES = ["ARS"] + ST
SHORT = [5, 15, 60]                       # У2: PCHIP должен выигрывать
FROM480 = [480, 720, 1000, 1440, 2160, 2880, 4320]   # У2: все сети должны выигрывать
# 120 и 240 мин — переходная зона: порог зависит от архитектуры (на валидации
# U-Net выигрывает с 240, SAITS неразличим на 240, остальные трое хуже на 240)
MID = [240, 480, 720, 1000]               # средний режим (У3)
MID_XFER = [240, 480, 720]                # У12: «2-16 часов»
LONG = [1440, 2160, 2880, 4320]           # длинный режим
ALL_L = [5, 15, 60, 120] + MID + LONG
B = 2000
SEED = 20261008
ALPHA = 0.05
# доля регулярного суточного хода, 2023 год — зафиксирована в протоколе (У10)
REG_SHARE = {"HUA": 0.73, "HER": 0.65, "KAK": 0.59, "WNG": 0.57,
             "ARS": 0.48, "CMO": 0.28}
# разница крутизны поля с ARS без учёта знака, градусы (У13): | |I| - |I_ARS| |.
# Знак отброшен: модуль — длина вектора, и ему всё равно, вниз поле или вверх.
DELTA_I = {"CMO": 3.2, "WNG": 5.1, "HER": 9.1, "KAK": 23.7, "HUA": 72.3}
# наибольший по модулю направляющий косинус, 2023 (У15)
AXIS_DOM = {"ARS": 0.960, "WNG": 0.931, "CMO": 0.974, "KAK": 0.766,
            "HUA": 0.997, "HER": 0.904}
TAG = {"ARS": ("Fbest", "XYZbest")}
TAG.update({c: ("F", "XYZ") for c in ST})


def model(a, s, sfx=""):
    """Имя модели архитектуры a, обученной на станции s."""
    if s == "ARS":
        return f"{a}_best{sfx}"
    if a == "imputeformer":
        return f"imputeformer_{s}_comp{sfx}"
    return f"{a}_{s}_best{sfx}"


# ------------------------------------------------------------ данные окон
_cache = {}


def windows(method, code, split, L):
    """Поокновые суммы внутри дыры: |e|, e^2, отклонение истины от среднего
    своей дыры в квадрате, число точек; плюс блоки, метка возмущённости и
    старт окна (для проверки, что сравниваются одни и те же окна)."""
    key = (method, code, split, L)
    if key in _cache:
        return _cache[key]
    p = os.path.join(C.OUT, f"dump_{method}_{C.setname(code, split)}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=False)
    m = int(d["margin"])
    t = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
    pr = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
    e = pr - t
    ok = np.isfinite(e)
    dev = t - np.nanmean(t, axis=1, keepdims=True)
    out = dict(
        sabs=np.where(ok, np.abs(e), 0).sum(axis=1),
        ssq=np.where(ok, e * e, 0).sum(axis=1),
        sdev=np.where(ok, dev * dev, 0).sum(axis=1),
        n=ok.sum(axis=1).astype(np.float64),
        blk=d[f"L{L}_block"].astype(np.int64),
        act=d[f"L{L}_act"].astype(np.float64),
        win=d[f"L{L}_win0"].astype(np.int64) if f"L{L}_win0" in d.files else None)
    _cache[key] = out
    return out


def _mult(blk, rng):
    """Кратности окон для одного повтора блочного бутстрепа (недели целиком)."""
    ub, inv = np.unique(blk, return_inverse=True)
    pick = rng.integers(0, ub.size, ub.size)
    return np.bincount(pick, minlength=ub.size)[inv].astype(np.float64)


def _stat(w, x, stat):
    s = (w * x["n"]).sum()
    if stat == "mae":
        return (w * x["sabs"]).sum() / s
    if stat == "rmse":
        return np.sqrt((w * x["ssq"]).sum() / s)
    if stat == "nse":
        return 1 - (w * x["ssq"]).sum() / (w * x["sdev"]).sum()
    raise ValueError(stat)


def paired(ma, mb, code, split, L, stat="mae", subset=None, seed=SEED):
    """Отношение stat(A)/stat(B) на одних и тех же окнах и его 95 % интервал.
    Возвращает (отношение, низ, верх, исход): исход +1 — A значимо точнее
    (интервал целиком ниже 1), -1 — значимо хуже, 0 — неразличимы. Для
    значимости разность и отношение эквивалентны: в каждом повторе знак
    разности совпадает со знаком (отношение - 1)."""
    a, b = windows(ma, code, split, L), windows(mb, code, split, L)
    if a["win"] is not None and b["win"] is not None:
        assert np.array_equal(a["win"], b["win"]), f"разные окна: {ma} / {mb} {code} L={L}"
    sel = np.ones(a["n"].size, bool)
    if subset == "act":
        sel = a["act"] > np.median(a["act"])
    a = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    b = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    one = np.ones(a["n"].size)
    point = _stat(one, a, stat) / _stat(one, b, stat)
    rng = np.random.default_rng(seed + L)
    r = np.empty(B)
    for i in range(B):
        w = _mult(a["blk"], rng)
        r[i] = _stat(w, a, stat) / _stat(w, b, stat)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return point, lo, hi, (1 if hi < 1 else (-1 if lo > 1 else 0))


def arch_side(ma, mb, code, split, lengths):
    """Сводный исход по набору длин — правило У3: «+» если A значимо точнее
    хотя бы на одной длине и ни на одной не хуже значимо, «-» наоборот,
    иначе «0»; при значимых результатах в обе стороны — «±»."""
    res = [paired(ma, mb, code, split, L)[3] for L in lengths]
    b, w = res.count(1), res.count(-1)
    return "+" if b and not w else "-" if w and not b else "±" if b and w else "0"


def perm_p(x, y):
    """Ранговая корреляция и точные перестановочные p для обоих хвостов:
    p_pos = доля расстановок с ρ не меньше наблюдённого, p_neg — не больше."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(spearmanr(x, y).statistic)
    n = len(x)
    if n <= 8:
        perms = [spearmanr(x, np.array(p)).statistic for p in itertools.permutations(y)]
    else:
        rng = np.random.default_rng(SEED)
        perms = [spearmanr(x, rng.permutation(y)).statistic for _ in range(20000)]
    perms = np.array(perms)
    return r, float(np.mean(perms >= r - 1e-12)), float(np.mean(perms <= r + 1e-12))


OK, NO, ND = "выполнено", "НЕ выполнено", "не доказано"


def corr_outcome(r, p_pos, p_neg):
    """Три исхода для корреляций (протокол, «Операционные определения»):
    значимо в предсказанную сторону — выполнено; значимо в обратную —
    НЕ выполнено; незначимо — не доказано (точек не хватило, а не «связи нет»)."""
    if r > 0 and p_pos < ALPHA:
        return OK
    if r < 0 and p_neg < ALPHA:
        return NO
    return ND


def merge(outs):
    """Свод по частям утверждения: хоть одна часть опровергнута — опровергнуто;
    все выполнены — выполнено; иначе не доказано."""
    if NO in outs:
        return NO
    return OK if all(o == OK for o in outs) else ND


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def score_regime(net, code, split, lengths, base="pchip"):
    """Score по режиму: 1 - gm(MAE сети)/gm(MAE PCHIP) по длинам режима."""
    mn = [_stat(np.ones_like(windows(net, code, split, L)["n"]), windows(net, code, split, L), "mae")
          for L in lengths]
    mb = [_stat(np.ones_like(windows(base, code, split, L)["n"]), windows(base, code, split, L), "mae")
          for L in lengths]
    return 1 - gm(mn) / gm(mb)


# ------------------------------------------------------------ утверждения
def U1(split):
    out = {}
    for inp, tag in (("модуль", "Fbest"), ("компоненты", "XYZbest")):
        R = D.regimes("ARS", split, tag)
        rk = R["rank"]["все длины"]
        nets = sorted((m for m in rk if m not in ("pchip", "linear")), key=rk.get)
        top = {m.replace("_best_xyz", "").replace("_best", "") for m in nets[:5]}
        out[inp] = sorted(top)
    ok = all(set(v) == set(FIN) for v in out.values())
    return ok, f"пятёрка по модулю {out['модуль']}, по компонентам {out['компоненты']}"


def U2(split):
    bad = []
    for a in FIN:
        for L in SHORT:
            # протокол: опровергнуто, если сеть НЕ УСТУПАЕТ PCHIP значимо
            if paired("pchip", model(a, "ARS"), "ARS", split, L)[3] != 1:
                bad.append(f"{a} не уступает PCHIP значимо на {L}")
        for L in FROM480:
            if paired(model(a, "ARS"), "pchip", "ARS", split, L)[3] != 1:
                bad.append(f"{a} не значимо точнее PCHIP на {L}")
    # для полноты — сколько из «должно» выполнено строго
    strict_short = sum(paired("pchip", model(a, "ARS"), "ARS", split, L)[3] == 1
                       for a in FIN for L in SHORT)
    strict_long = sum(paired(model(a, "ARS"), "pchip", "ARS", split, L)[3] == 1
                      for a in FIN for L in FROM480)
    return not bad, (f"нарушений {len(bad)}{': ' + '; '.join(bad[:4]) if bad else ''}. "
                     f"PCHIP значимо точнее в {strict_short}/{len(FIN)*len(SHORT)} "
                     f"сочетаний ≤60 мин; сеть значимо точнее в "
                     f"{strict_long}/{len(FIN)*len(FROM480)} сочетаний ≥480 мин")


def U3(split):
    side = {a: arch_side(model(a, "ARS", "_xyz"), model(a, "ARS"), "ARS", split, MID) for a in FIN}
    worse = [a for a, s in side.items() if s in ("-", "±")]
    better = [a for a, s in side.items() if s == "+"]
    return (not worse and len(better) >= 3), f"по архитектурам {side}"


def U4(split):
    R, X = D.regimes("ARS", split, "Fbest"), D.regimes("ARS", split, "XYZbest")
    rk = R["rank"]["все длины"]
    res, outs = [], []
    for g in ("120–1000 мин", ">1000 мин"):
        x = [rk[f"{a}_best"] for a in ALL]
        y = [X["mae_gm"][f"{a}_best_xyz"][g] / R["mae_gm"][f"{a}_best"][g] - 1 for a in ALL]
        r, pp, pn = perm_p(x, y)
        outs.append(corr_outcome(r, pp, pn))
        res.append(f"{g}: ρ={r:+.2f}, p={pp:.3f}")
    return merge(outs), "; ".join(res)


def U5(split):
    bad, worst = [], 0.0
    for c in ST:
        for sfx in ("", "_xyz"):
            for a in FIN:
                for L in LONG:
                    pt, lo, hi, v = paired(model(a, c, sfx), "pchip", c, split, L)
                    worst = max(worst, hi)
                    if v != 1:
                        bad.append(f"{c}{sfx} {a} L={L}")
    return not bad, f"не значимо точнее: {len(bad)} из 200; худшая верхняя граница {worst:.3f}"


def U6(split):
    med, within = {}, {}
    for c in ST:                      # обучение на месте — пять станций, как в основании
        v = [score_regime(model(a, c), c, split, LONG) for a in FIN]
        med[c], within[c] = float(np.median(v)), max(v) - min(v)
    between = max(med.values()) - min(med.values())
    wmax = max(within, key=within.get)
    return (within[wmax] < between,
            f"медианы {', '.join(f'{c} {v:+.3f}' for c, v in med.items())}; "
            f"размах между станциями {between:.3f}, наибольший внутри {within[wmax]:.3f} ({wmax})")


def U7(split):
    bad = []
    for a in FIN:
        for L in LONG:
            if paired(model(a, "ARS"), "pchip", "ARS", split, L, subset="act")[3] != 1:
                bad.append(f"{a} L={L}")
    return not bad, f"в возмущённой половине не значимо точнее: {len(bad)} из {len(FIN)*len(LONG)}" + \
        (f" ({'; '.join(bad[:5])})" if bad else "")


def U8(split):
    one = lambda m: _stat(np.ones_like(windows(m, "ARS", split, 5)["n"]), windows(m, "ARS", split, 5), "mae")
    best = min(FIN, key=lambda a: one(model(a, "ARS")))
    ratio = one(model(best, "ARS")) / one("pchip")
    sig = paired("pchip", model(best, "ARS"), "ARS", split, 5)[3] == 1
    return (ratio >= 2 and sig), f"ошибка лучшей сети / PCHIP на 5 мин = {ratio:.2f}, PCHIP значимо точнее: {sig}"


def U9_year(split):
    """Часть утверждения, проверяемая внутри одного года: выигрыш по RMSE меньше,
    чем по MAE (медиана по пятёрке, длинный режим, ARS, модуль)."""
    def sc(stat):
        v = []
        for a in FIN:
            n = [_stat(np.ones_like(windows(model(a, "ARS"), "ARS", split, L)["n"]),
                       windows(model(a, "ARS"), "ARS", split, L), stat) for L in LONG]
            b = [_stat(np.ones_like(windows("pchip", "ARS", split, L)["n"]),
                       windows("pchip", "ARS", split, L), stat) for L in LONG]
            v.append(1 - gm(n) / gm(b))
        return float(np.median(v))
    s_mae, s_rmse = sc("mae"), sc("rmse")
    return (s_rmse < s_mae), f"Score по MAE {s_mae:+.3f}, по RMSE {s_rmse:+.3f}", (s_mae, s_rmse)


def nse_station(c, split, rng=None):
    """NSE станции: медиана по пятёрке среднего NSE по четырём длинным длинам.
    rng — для бутстрепа: тогда окна пересэмплируются по неделям. Набор недель
    выбирается ОДИН раз на длину и применяется ко всем пяти моделям: они
    оценены на одних и тех же окнах, и свой набор у каждой модели разорвал бы
    эту связь и раздул интервал."""
    ws = {}
    if rng is not None:
        for L in LONG:
            ws[L] = _mult(windows(model(FIN[0], c), c, split, L)["blk"], rng)
    v = []
    for a in FIN:
        per_L = []
        for L in LONG:
            x = windows(model(a, c), c, split, L)
            w = np.ones_like(x["n"]) if rng is None else ws[L]
            per_L.append(_stat(w, x, "nse"))
        v.append(np.mean(per_L))
    return float(np.median(v))


def U10(split):
    nse = {c: nse_station(c, split) for c in CODES}
    r, pp, pn = perm_p([REG_SHARE[c] for c in CODES], [nse[c] for c in CODES])
    return corr_outcome(r, pp, pn), (f"NSE {', '.join(f'{c} {v:+.2f}' for c, v in nse.items())}; "
                                     f"ρ={r:+.2f}, p={pp:.3f}"), nse


def U12(split):
    st = {}
    for c in ST:
        long_g = sum(arch_side(model(a, "ARS"), "pchip", c, split, LONG) == "+" for a in FIN)
        mid_g = sum(arch_side(model(a, "ARS"), "pchip", c, split, MID_XFER) == "+" for a in FIN)
        st[c] = (long_g, mid_g)
    long_ok = sum(v[0] >= 3 for v in st.values())
    mid_bad = sum(v[1] >= 3 for v in st.values())
    return (long_ok >= 3 and mid_bad < 3), (
        f"по станциям (моделей с выигрышем на длинных / на средних): "
        f"{', '.join(f'{c} {v[0]}/{v[1]}' for c, v in st.items())}")


def U13(split):
    gain, sig = {}, {}
    for c in ST:
        sides = [arch_side(model(a, c), model(a, "ARS"), c, split, ALL_L) for a in FIN]
        sig[c] = sum(s == "+" for s in sides) >= 3
        g = []
        for a in FIN:
            loc = [_stat(np.ones_like(windows(model(a, c), c, split, L)["n"]), windows(model(a, c), c, split, L), "mae") for L in ALL_L]
            tr = [_stat(np.ones_like(windows(model(a, "ARS"), c, split, L)["n"]), windows(model(a, "ARS"), c, split, L), "mae") for L in ALL_L]
            g.append(1 - gm(loc) / gm(tr))
        gain[c] = float(np.median(g))
    r, pp, pn = perm_p([DELTA_I[c] for c in ST], [gain[c] for c in ST])
    no_gain = sum(not v for v in sig.values())
    out = NO if no_gain >= 3 else corr_outcome(r, pp, pn)
    p = pp
    return out, (
        f"выигрыш обучения на месте {', '.join(f'{c} {gain[c]:+.3f}' for c in ST)}; "
        f"значим на {sum(sig.values())} из 5; связь с разницей крутизны: ρ={r:+.2f}, p={p:.3f}")


def U14(split):
    """Матрица 6x6, модуль. Score клетки — медиана по пятёрке; несимметричность
    пары — по каждой архитектуре разность Score двух направлений с независимым
    бутстрепом (цели разные, окна разные), пара несимметрична, если у
    большинства архитектур интервал разности не накрывает ноль."""
    def sc_boot(a, s, t, rng):
        x = [windows(model(a, s), t, split, L) for L in ALL_L]
        bpc = [windows("pchip", t, split, L) for L in ALL_L]
        if rng is None:
            ws = [np.ones_like(v["n"]) for v in x]
        else:
            ws = [_mult(v["blk"], rng) for v in x]
        return 1 - gm([_stat(w, v, "mae") for w, v in zip(ws, x)]) / \
            gm([_stat(w, v, "mae") for w, v in zip(ws, bpc)])
    cell = {(s, t): float(np.median([sc_boot(a, s, t, None) for a in FIN]))
            for s in CODES for t in CODES if s != t}
    asym = 0
    pairs = [(s, t) for s, t in itertools.combinations(CODES, 2)]
    nb = 300           # пары × архитектуры × длины: 300 повторов хватает для решения «0 внутри / снаружи»
    for s, t in pairs:
        votes = 0
        for a in FIN:
            # зерно из индексов, а не hash(): тот меняется от запуска к запуску
            rng = np.random.default_rng(SEED + 100 * CODES.index(s) + 10 * CODES.index(t) + FIN.index(a))
            d = [sc_boot(a, s, t, rng) - sc_boot(a, t, s, rng) for _ in range(nb)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            votes += (lo > 0 or hi < 0)
        asym += votes >= 3
    row = {s: np.mean([cell[(s, t)] for t in CODES if t != s]) for s in CODES}
    col = {t: np.mean([cell[(s, t)] for s in CODES if s != t]) for t in CODES}
    order_ok = (min(row, key=row.get) == "HUA" and max(col, key=col.get) == "HUA"
                and min(col, key=col.get) == "ARS")
    return (asym >= 8 and order_ok), (
        f"несимметричных пар {asym} из {len(pairs)}; источник хуже всех: "
        f"{min(row, key=row.get)}; цель лучше всех: {max(col, key=col.get)}; "
        f"цель хуже всех: {min(col, key=col.get)}")


def U15(split):
    codes = [c for c in CODES if not (c == "CMO" and split == "test")]   # исключение CMO 2021
    res, outs = [], []
    for g in ("120–1000 мин", ">1000 мин"):
        y = []
        for c in codes:
            tf, tx = TAG[c]
            R, X = D.regimes(c, split, tf), D.regimes(c, split, tx)
            y.append(np.mean([X["mae_gm"][model(a, c, "_xyz")][g] / R["mae_gm"][model(a, c)][g] - 1
                              for a in FIN]))
        r, pp, pn = perm_p([AXIS_DOM[c] for c in codes], y)
        outs.append(corr_outcome(r, pp, pn))
        res.append(f"{g}: ρ={r:+.2f}, p={pp:.3f} (n={len(codes)})")
    return merge(outs), "; ".join(res)


YEAR_CLAIMS = [("У1", U1), ("У2", U2), ("У3", U3), ("У4", U4), ("У5", U5),
               ("У6", U6), ("У7", U7), ("У8", U8), ("У15", U15)]
TRANSFER_CLAIMS = [("У12", U12), ("У13", U13), ("У14", U14)]


def run_year(split, only=None):
    rows = []
    claims = YEAR_CLAIMS + ([] if split == "val" else TRANSFER_CLAIMS)
    for name, fn in claims:
        if only and name not in only:
            continue
        try:
            ok, msg = fn(split)
            rows.append((name, ok if isinstance(ok, str) else (OK if ok else NO), msg))
        except FileNotFoundError as e:
            rows.append((name, "нет данных", os.path.basename(str(e))))
        print(f"  {rows[-1][0]:4} {rows[-1][1]:13} {rows[-1][2]}", flush=True)
    if not only or "У9" in only:
        ok, msg, _ = U9_year(split)
        rows.append(("У9a", "выполнено" if ok else "НЕ выполнено", msg))
        print(f"  У9a  {rows[-1][1]:13} {msg}", flush=True)
    if not only or "У10" in only:
        ok, msg, _ = U10(split)
        rows.append(("У10", ok, msg))
        print(f"  У10  {rows[-1][1]:13} {msg}", flush=True)
    return rows


def combine():
    """Межгодовые части (У9b, У11) и итог по разделу 5 протокола."""
    out = {}
    for split in ("test", "test_hard"):
        p = os.path.join(C.OUT, f"verdict_{split}.json")
        if not os.path.exists(p):
            raise SystemExit(f"нет {p} — сначала --split {split}")
        out[split] = {r[0]: r[1] for r in json.load(open(p, encoding="utf-8"))}
    # У9b: выигрыш по RMSE падает в бурный год сильнее, чем по MAE
    q, h = U9_year("test")[2], U9_year("test_hard")[2]
    u9b = (q[1] - h[1]) > (q[0] - h[0])
    print(f"  У9b  {'выполнено' if u9b else 'НЕ выполнено':13} падение Score от 2021 к 2024: "
          f"по MAE {q[0]-h[0]:+.3f}, по RMSE {q[1]-h[1]:+.3f}")
    # У11: падение NSE в бурю — независимый бутстреп двух лет
    drop, sig = {}, {}
    for c in CODES:
        rq, rh = np.random.default_rng(SEED + 1), np.random.default_rng(SEED + 2)
        d = [nse_station(c, "test", rq) - nse_station(c, "test_hard", rh) for _ in range(500)]
        drop[c] = nse_station(c, "test") - nse_station(c, "test_hard")
        sig[c] = np.percentile(d, 2.5) > 0
    mid = ["ARS", "WNG", "KAK"]
    u11 = (sum(sig[c] for c in mid) >= 2 and
           np.mean([drop[c] for c in ("HUA", "HER")]) < np.mean([drop[c] for c in mid]))
    print(f"  У11  {'выполнено' if u11 else 'НЕ выполнено':13} падение NSE: "
          f"{', '.join(f'{c} {drop[c]:+.2f}{'*' if sig[c] else ''}' for c in CODES)} (* значимо)")
    print("\nИТОГ ПО РАЗДЕЛУ 5 (подтверждено только если выполнено в обоих годах):")
    names = sorted(set(out["test"]) | set(out["test_hard"]), key=lambda s: (len(s), s))
    tally = {"подтверждено": 0, "зависит от обстановки": 0, "не доказано": 0,
             "опровергнуто": 0}
    for n in names:
        a, b = out["test"].get(n), out["test_hard"].get(n)
        # протокол, раздел 5: «не доказано» хотя бы в одном году -> не доказано
        v = ("не доказано" if ND in (a, b) else
             "подтверждено" if a == b == OK else
             "опровергнуто" if a == b == NO else "зависит от обстановки")
        tally[v] += 1
        print(f"  {n:4} 2021: {a:13} 2024: {b:13} -> {v}")
    print(f"  У9b  межгодовое: {'подтверждено' if u9b else 'опровергнуто'}")
    print(f"  У11  межгодовое: {'подтверждено' if u11 else 'опровергнуто'}")
    print(f"\nсчёт: {tally}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["val", "test", "test_hard"])
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--test-ok", action="store_true",
                    help="разрешить работу на тестовых годах")
    a = ap.parse_args()
    if (a.combine or a.split in ("test", "test_hard")) and not a.test_ok:
        raise SystemExit("тестовые годы: нужен явный --test-ok (см. докстринг)")
    if a.combine:
        combine()
        return
    if not a.split:
        raise SystemExit("нужен --split или --combine")
    print(f"вердикты, набор {a.split}:")
    rows = run_year(a.split, a.only)
    if not a.only:
        with open(os.path.join(C.OUT, f"verdict_{a.split}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
