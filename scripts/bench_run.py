r"""Прогон метода по окнам -> дамп предсказаний для bench_metrics.py.

методы без обучения (mean, locf, linear, pchip, daily)

Запуск:  cd bench_arti/scripts
         python -u bench_run.py --method pchip --split val
         python -u bench_run.py --method daily --split val --n 256
"""
import os
import argparse
import numpy as np
from scipy.interpolate import PchipInterpolator

import bench_common as C

DAY = 1440


# ------------------------------------------------------------ методы
def fill_mean(inp):
    """Среднее по наблюдаемым точкам; NaN вне дыры заменяются на среднее."""
    obs = np.isfinite(inp)
    out = inp.copy()
    out[~obs] = np.mean(inp[obs]) if obs.any() else 0.0
    return out


def fill_locf(inp):
    """Последнее наблюдение вперёд; начало окна — первым наблюдением назад."""
    obs = np.isfinite(inp)
    idx = np.where(obs, np.arange(inp.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = inp[idx]
    first = np.argmax(obs)
    out[:first] = inp[first] if obs.any() else 0.0
    return out


def fill_linear(inp):
    """Линейная интерполяция между наблюдаемыми точками; NaN вне дыры заменяются на среднее."""
    obs = np.isfinite(inp)
    x = np.arange(inp.size)
    if obs.sum() < 2:
        return fill_mean(inp)
    return np.interp(x, x[obs], inp[obs]).astype(np.float64)


def fill_pchip(inp):
    """PCHIP-интерполяция между наблюдаемыми точками; NaN вне дыры заменяются на среднее."""
    obs = np.isfinite(inp)
    x = np.arange(inp.size)
    if obs.sum() < 2:
        return fill_mean(inp)
    return PchipInterpolator(x[obs], inp[obs], extrapolate=True)(x)


def fill_daily(inp, ndays=3):
    """Суточный шаблон: значение берётся из тех же минут соседних суток
    (медиана по доступным), затем шаблон привязывается к краям дыры.

    Привязка обязательна: базовый уровень поля плывёт от суток к суткам, и без
    сдвига шаблон дал бы верную ФОРМУ на неверном УРОВНЕ. Соседние сутки берутся
    только внутри окна — как и у всех остальных методов."""
    n = inp.size
    obs = np.isfinite(inp)
    out = fill_pchip(inp)                      # запасной вариант и фон
    miss = np.flatnonzero(~obs)
    if miss.size == 0:
        return out

    idx = np.arange(n)
    cand = []
    for k in range(1, ndays + 1):
        for sh in (-k * DAY, k * DAY):
            j = idx + sh
            v = np.where((j >= 0) & (j < n), inp[np.clip(j, 0, n - 1)], np.nan)
            cand.append(v)
    st = np.stack(cand)
    cnt = np.isfinite(st).sum(0)
    tmpl = np.full(n, np.nan)
    ok = cnt > 0                    # без явной проверки nanmedian ругается на
    if ok.any():                    # столбцы, где соседних суток нет вовсе
        tmpl[ok] = np.nanmedian(st[:, ok], axis=0)
    if not ok[miss].any():
        return out

    # привязка по наблюдаемым точкам у краёв дыры
    edge = obs & ok
    if edge.sum() >= 10:
        near = np.zeros(n, bool)
        for s, e in _runs(~obs):
            near[max(0, s - C.CTX):s] = True
            near[e:min(n, e + C.CTX)] = True
        sel = edge & near
        if sel.sum() < 10:
            sel = edge
        tmpl = tmpl + np.median(inp[sel] - tmpl[sel])

    take = (~obs) & ok
    out[take] = tmpl[take]
    return out


def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


METHODS = {"mean": fill_mean, "locf": fill_locf, "linear": fill_linear,
           "pchip": fill_pchip, "daily": fill_daily}


# ------------------------------------------------------------ прогон
def resolve(method):
    """Классический метод по имени либо обученная модель по имени чекпойнта.
    Обе ветки возвращают функцию (окно, начало) -> заполненное окно, поэтому
    дальше прогон одинаков и дампы получаются одного формата."""
    if method in METHODS:
        fn = METHODS[method]
        return (lambda inp, start: fn(inp)), None
    import torch
    import bench_models as BM
    path = os.path.join(os.path.dirname(__file__), "..", "models", f"{method}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"нет ни метода, ни чекпойнта «{method}»")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = BM.load(path, dev)
    return (lambda inp, start: BM.fill(net, inp, start, ck["scale"], dev)), ck


def run(args):
    fn, ck = resolve(args.method)
    if ck is not None:
        print(f"чекпойнт: {args.method}.pt  шаг={ck['step']}  "
              f"val при обучении={ck['val']:.3f} нТл")
    years = C.load_split(args.split)
    acts = {y: C.activity(F) for y, F in years.items()}
    rng = np.random.default_rng(C.SEED + 7)
    lengths = args.lengths or C.LENGTHS
    setname = f"{C.CODE}_{args.split}"

    print(f"метод={args.method}  набор={setname} ({', '.join(map(str, sorted(years)))})"
          f"  n={args.n}  margin={C.MARGIN}")
    print(f"{'len':>6}{'окон':>6}{'MAE, нТл':>12}")
    print("-" * 24)

    out = {}
    for L in lengths:
        smps = C.gather(years, L, args.n, rng)
        if not smps:
            print(f"{L:>6}{0:>6}  нет окон")
            continue
        span = 2 * C.MARGIN + L
        tr = np.full((len(smps), span), np.nan, np.float32)
        pr = np.full((len(smps), span), np.nan, np.float32)
        act = np.zeros(len(smps), np.float32)
        g0 = np.zeros(len(smps), np.int32)
        errs = []
        for i, s in enumerate(smps):
            pred = fn(s["input"], s["start"])
            a = int(s["pos"][0])
            lo, hi = a - C.MARGIN, a + L + C.MARGIN
            clo, chi = max(0, lo), min(C.W, hi)
            d0 = clo - lo
            tr[i, d0:d0 + chi - clo] = s["target"][clo:chi]
            pr[i, d0:d0 + chi - clo] = pred[clo:chi]
            mk = s["mask_art"]
            errs.append(np.abs(pred[mk] - s["target"][mk]).mean())
            st = s["start"]
            act[i] = acts[s["year"]][st:st + C.W].mean()
            g0[i] = a
        out[f"L{L}_true"] = tr
        out[f"L{L}_pred"] = pr
        out[f"L{L}_act"] = act
        out[f"L{L}_gap0"] = g0
        print(f"{L:>6}{len(smps):>6}{np.mean(errs):>12.3f}")

    out["method"] = np.array(args.method)
    out["setname"] = np.array(setname)
    out["lengths"] = np.array(lengths, np.int32)
    out["margin"] = np.array(C.MARGIN, np.int32)
    out["seed"] = np.array(C.SEED, np.int32)
    path = args.out or os.path.join(C.DATA, f"dump_{args.method}_{setname}.npz")
    np.savez_compressed(path, **out)
    print(f"\nсохранено: {path}  ({os.path.getsize(path) / 1e6:.1f} МБ)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    help="классический метод (" + ", ".join(sorted(METHODS))
                         + ") либо имя чекпойнта из models/")
    ap.add_argument("--split", default="val", choices=sorted(C.SPLIT))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
