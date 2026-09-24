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
def resolve(method, ckpt=None):
    """Классический метод по имени либо обученная модель по имени чекпойнта.
    Обе ветки возвращают функцию (окно, начало) -> заполненное окно, поэтому
    дальше прогон одинаков и дампы получаются одного формата.
    ckpt — явный путь к .pt (bench_grid.py: ячейки лежат в models/grid/);
    method тогда только имя для дампа."""
    if method in METHODS:
        fn = METHODS[method]
        return (lambda inp, start: fn(inp)), None
    import torch
    import bench_models as BM
    path = ckpt or os.path.join(os.path.dirname(__file__), "..", "models", f"{method}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"нет ни метода, ни чекпойнта «{method}»")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = BM.load(path, dev)
    return (lambda inp, start: BM.fill(net, inp, start, ck["scale"], dev)), ck


def _build_length_fields(smps, fn, acts, L):
    """tr/pr/act/g0 для одной длины L из списка сэмплов — общая часть между
    обычным прогоном (один код станции) и выровненным (--align-codes)."""
    if not smps:
        return None, 0.0
    span = 2 * C.MARGIN + L
    n = len(smps)
    tr = np.full((n, span), np.nan, np.float32)
    pr = np.full((n, span), np.nan, np.float32)
    act = np.zeros(n, np.float32)
    g0 = np.zeros(n, np.int32)
    blk = np.zeros(n, np.int32)
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
        # блок для блочного бутстрэпа (bench_metrics): год и календарная
        # неделя, в которую попадает НАЧАЛО дыры. Окна в дампе перекрываются
        # (на длинных дырах каждая минута года лежит в ~8 дырах), поэтому
        # ресэмплировать по окнам нельзя — только неделями целиком.
        blk[i] = int(s["year"]) * 100 + (st + a) // C.BLOCK
    return dict(true=tr, pred=pr, act=act, gap0=g0, block=blk), float(np.mean(errs))


def _save_dump(args, code, lengths, fields_by_L):
    out = {}
    for L, fields in fields_by_L.items():
        out[f"L{L}_true"] = fields["true"]
        out[f"L{L}_pred"] = fields["pred"]
        out[f"L{L}_act"] = fields["act"]
        out[f"L{L}_gap0"] = fields["gap0"]
        out[f"L{L}_block"] = fields["block"]
    setname = f"{code}_{args.split}"
    out["method"] = np.array(args.method)
    out["setname"] = np.array(setname)
    out["lengths"] = np.array(lengths, np.int32)
    out["margin"] = np.array(C.MARGIN, np.int32)
    out["seed"] = np.array(C.SEED, np.int32)
    out_dir = getattr(args, "out_dir", None) or C.DATA
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"dump_{args.method}_{setname}.npz")
    np.savez_compressed(path, **out)
    print(f"сохранено: {path}  ({os.path.getsize(path) / 1e6:.1f} МБ)")


def window_plan(lengths, gather_fn):
    """Окна для запрошенных длин — по тому же пути ГСЧ, что у существующих
    дампов (см. C.LENGTHS_RNG_GROUPS): каждая группа длин идёт со своим
    свежим генератором в своём порядке, а длины группы, которые НЕ запрошены,
    всё равно прогоняются через gather (результат выбрасывается), чтобы
    генератор дошёл до запрошенной длины в том же состоянии. Так частичный
    прогон (--lengths 240) даёт те же окна, что полный, и любой новый дамп
    сопоставим с остальными побитово. gather_fn(L, rng) -> сэмплы."""
    want = set(lengths)
    known = {L for g in C.LENGTHS_RNG_GROUPS for L in g}
    unknown = sorted(want - known)
    if unknown:
        raise SystemExit(f"длины {unknown} не входят в C.LENGTHS_RNG_GROUPS — "
                         f"добавь их новой группой, не меняя существующие")
    out = {}
    for group in C.LENGTHS_RNG_GROUPS:
        if not (want & set(group)):
            continue
        rng = np.random.default_rng(C.SEED + 7)
        for L in group:
            smps = gather_fn(L, rng)
            if L in want:
                out[L] = smps
    return [(L, out[L]) for L in lengths]


def run(args):
    fn, ck = resolve(args.method, getattr(args, "ckpt", None))
    if ck is not None:
        print(f"чекпойнт: {args.method}.pt  шаг={ck['step']}  "
              f"val при обучении={ck['val']:.3f} нТл")
    lengths = args.lengths or C.LENGTHS

    if args.align_codes:
        # выровненный прогон: одни и те же (год, окно, позиция дыры) сразу
        # для нескольких станций — см. gather_aligned, почему это не то же
        # самое, что независимые прогоны по каждой станции отдельно.
        codes = [c.strip() for c in args.align_codes.split(",") if c.strip()]
        years_by_code = {c: C.load_split(args.split, code=c) for c in codes}
        acts_by_code = {c: {y: C.activity(F) for y, F in yy.items()}
                        for c, yy in years_by_code.items()}
        print(f"метод={args.method}  align_codes={codes}  сплит={args.split}  "
              f"n={args.n}  margin={C.MARGIN}")
        print(f"{'len':>6}{'код':>6}{'окон':>6}{'MAE, нТл':>12}")
        print("-" * 30)
        fields_by_code = {c: {} for c in codes}
        plan = window_plan(lengths, lambda L, rng: C.gather_aligned(years_by_code, L, args.n, rng))
        for L, smps_by_code in plan:
            for c in codes:
                fields, mae = _build_length_fields(smps_by_code[c], fn,
                                                    acts_by_code[c], L)
                if fields is None:
                    print(f"{L:>6}{c:>6}{0:>6}  нет окон")
                    continue
                fields_by_code[c][L] = fields
                print(f"{L:>6}{c:>6}{len(smps_by_code[c]):>6}{mae:>12.3f}")
        for c in codes:
            _save_dump(args, c, lengths, fields_by_code[c])
        return

    years = C.load_split(args.split, code=args.code)
    acts = {y: C.activity(F) for y, F in years.items()}
    setname = f"{args.code}_{args.split}"

    print(f"метод={args.method}  набор={setname} ({', '.join(map(str, sorted(years)))})"
          f"  n={args.n}  margin={C.MARGIN}")
    print(f"{'len':>6}{'окон':>6}{'MAE, нТл':>12}")
    print("-" * 24)

    fields_by_L = {}
    plan = window_plan(lengths, lambda L, rng: C.gather(years, L, args.n, rng))
    for L, smps in plan:
        fields, mae = _build_length_fields(smps, fn, acts, L)
        if fields is None:
            print(f"{L:>6}{0:>6}  нет окон")
            continue
        fields_by_L[L] = fields
        print(f"{L:>6}{len(smps):>6}{mae:>12.3f}")

    if args.out:
        # ручной путь вывода — сохраняем напрямую, без _save_dump (та всегда
        # строит имя по code/split)
        out = {}
        for L, fields in fields_by_L.items():
            out[f"L{L}_true"] = fields["true"]
            out[f"L{L}_pred"] = fields["pred"]
            out[f"L{L}_act"] = fields["act"]
            out[f"L{L}_gap0"] = fields["gap0"]
            out[f"L{L}_block"] = fields["block"]
        out["method"] = np.array(args.method)
        out["setname"] = np.array(setname)
        out["lengths"] = np.array(lengths, np.int32)
        out["margin"] = np.array(C.MARGIN, np.int32)
        out["seed"] = np.array(C.SEED, np.int32)
        np.savez_compressed(args.out, **out)
        print(f"\nсохранено: {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} МБ)")
    else:
        _save_dump(args, args.code, lengths, fields_by_L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    help="классический метод (" + ", ".join(sorted(METHODS))
                         + ") либо имя чекпойнта из models/")
    ap.add_argument("--split", default="val", choices=sorted(C.SPLIT))
    ap.add_argument("--code", default=C.CODE,
                     help="код станции (префикс файлов data/<код>_<год>.npz); "
                          "меняет и вход, и имя выходного дампа (dump_<метод>_<код>_<сплит>.npz)")
    ap.add_argument("--align-codes", default=None,
                     help="через запятую (напр. ARS,WNG,KAK) — выровненный "
                          "прогон: одни и те же (год, окно, позиция дыры) для "
                          "всех перечисленных станций разом, гарантируя "
                          "побитовое совпадение окон между ними (нужно для "
                          "парного бутстрепа/сравнения станций); пишет по "
                          "отдельному дампу на каждый код, --code/--out "
                          "игнорируются")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None,
                     help="явный путь к чекпойнту .pt вместо models/<method>.pt "
                          "(имя дампа по-прежнему строится из --method)")
    ap.add_argument("--out-dir", default=None,
                     help="каталог для дампов вместо data/ (имя файла стандартное; "
                          "работает и с --align-codes, где --out игнорируется) — "
                          "для частичных прогонов (--lengths), которые потом "
                          "сливаются в основной дамп")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
