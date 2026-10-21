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


def _level24(inp, min_count=360):
    """Скользящее суточное среднее по наблюдённым точкам (окно 1440 мин по центру).
    Где наблюдений в окне меньше min_count — NaN."""
    obs = np.isfinite(inp)
    v = np.where(obs, inp, 0.0)
    k = np.ones(DAY)
    s = np.convolve(v, k, mode="same")
    c = np.convolve(obs.astype(np.float64), k, mode="same")
    out = np.full(inp.size, np.nan)
    ok = c >= min_count
    out[ok] = s[ok] / c[ok]
    return out


def fill_dayint(inp, ndays=2, edge=15, smooth=1):
    """Интерполяция плюс типичный суточный ход — проверка того, сводится ли
    выигрыш сетей к знанию регулярной суточной вариации.

    1. Шаблон: из окна вычитается скользящее суточное среднее (уровень), и для
       каждой минуты берётся медиана таких отклонений в тех же минутах ±1..ndays
       соседних суток. Уровень снят, поэтому перепады уровня между сутками в
       шаблон не попадают. Шаблон сглаживается скользящим средним в smooth
       минут: медиана по нескольким суткам на минутном шаге шумная.
    2. Остаток «данные − шаблон» усредняется (медиана) по edge наблюдённым
       минутам у каждого края пропуска, и между краями проводится прямая — она
       несёт уровень и его дрейф через пропуск.
    3. Заполнение = шаблон + прямая: на краях совпадает с данными, внутри
       даёт суточный цикл. Где шаблона нет (нет соседних суток), — PCHIP.
    Как и все методы, видит только своё окно."""
    n = inp.size
    obs = np.isfinite(inp)
    out = fill_pchip(inp)
    if obs.all() or obs.sum() < 2:
        return out
    anom = inp - _level24(inp)
    idx = np.arange(n)
    cand = []
    for k in range(1, ndays + 1):
        for sh in (-k * DAY, k * DAY):
            j = idx + sh
            cand.append(np.where((j >= 0) & (j < n), anom[np.clip(j, 0, n - 1)], np.nan))
    st = np.stack(cand)
    ok = np.isfinite(st).any(0)
    tmpl = np.full(n, np.nan)
    if ok.any():
        tmpl[ok] = np.nanmedian(st[:, ok], axis=0)
    if smooth > 1:
        fin = np.isfinite(tmpl)
        k_ = np.ones(smooth)
        num = np.convolve(np.where(fin, tmpl, 0.0), k_, mode="same")
        den = np.convolve(fin.astype(np.float64), k_, mode="same")
        tmpl = np.where(fin & (den > 0), num / np.maximum(den, 1), np.nan)
    resid = inp - tmpl
    have = np.isfinite(resid)
    for s, e in _runs(~obs):
        left = np.flatnonzero(have[:s])[-edge:]
        right = s + np.flatnonzero(have[s:])
        right = right[right >= e][:edge]
        if left.size == 0 or right.size == 0:
            continue                            # пропуск у края окна — остаётся PCHIP
        r0, r1 = np.median(resid[left]), np.median(resid[right])
        x0, x1 = left.mean(), right.mean()
        g = np.arange(s, e)
        line = r0 + (r1 - r0) * (g - x0) / (x1 - x0)
        t = tmpl[g]
        fill = np.isfinite(t)
        out[g[fill]] = t[fill] + line[fill]
    return out


def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


METHODS = {"mean": fill_mean, "locf": fill_locf, "linear": fill_linear,
           "pchip": fill_pchip, "daily": fill_daily}
# сетка параметров «интерполяции плюс суточного хода» — выбор на валидации
# первая сетка: k 1–3, края 5/30, сглаживание 1/30/120; лучший вариант оказался на краю
# по k и по сглаживанию, поэтому сетка расширена (k до 5, сглаживание до 480 мин)
_GRID = [(k, w, sm) for k in (1, 2, 3) for w in (5, 30) for sm in (1, 30, 120)] +         [(k, 5, sm) for k in (3, 4, 5) for sm in (120, 240, 480) if (k, sm) != (3, 120)]
for _k, _w, _s in _GRID:
    METHODS[f"dayint_k{_k}_w{_w}_s{_s}"] = (
        lambda inp, k=_k, w=_w, sm=_s: fill_dayint(inp, k, w, sm))
# Выбор на валидации (ARS 2023, минимум геом. среднего MAE на пропусках 1440–4320 мин):
# k=5, края 5 мин, сглаживание 240 мин — 8.78 нТл против 11.64 у PCHIP; сетка на
# плато (k=4 ≈ k=5, сглаживание 120 ≈ 240, 480 хуже). Больше 5 суток окно не вмещает.
METHODS["dayint"] = lambda inp: fill_dayint(inp, 5, 5, 240)


# ------------------------------------------------------------ прогон
def resolve(method, ckpt=None, code=None, split=None, comp_sink=None):
    """Классический метод по имени либо обученная модель по имени чекпойнта.
    Обе ветки возвращают функцию (окно, начало) -> заполненное окно, поэтому
    дальше прогон одинаков и дампы получаются одного формата.
    ckpt — явный путь к .pt (bench_grid.py: ячейки лежат в models/grid/);
    method тогда только имя для дампа."""
    if method in METHODS:
        fn = METHODS[method]
        return (lambda inps, starts, years: [fn(a) for a in inps]), None
    import torch
    import bench_models as BM
    path = ckpt or os.path.join(C.MODELS, f"{method}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"нет ни метода, ни чекпойнта «{method}»")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = BM.load(path, dev)
    if int(ck.get("NCH", 1)) == 1:
        return (lambda inps, starts, years:
                BM.fill_many(net, inps, starts, ck["scale"], dev)), ck

    # модель обучена на компонентах: окно и дыра те же (их план построен по F),
    # но на вход идут X, Y, Z, а наружу — модуль восстановленного вектора.
    # Годится только там, где F и есть модуль вектора, то есть базисы
    # вариометра приложены верно; остальные годы отсеяны ещё в comp_ok.
    comp = {y: C.load_year_comp(y, code=code) for y in C.SPLIT[split]
            if C.comp_ok(y, code=code)}
    bad = [y for y in C.SPLIT[split] if y not in comp]
    if bad:
        raise SystemExit(f"в наборе {split} компоненты непригодны для годов "
                         f"{', '.join(map(str, bad))}: базисы вариометра не приложены "
                         "или устарели, модуль вектора расходится со скаляром")

    def fill_comp(inps, starts, years):
        ws = []
        for a, st, y in zip(inps, starts, years):
            w = comp[y][st:st + C.W].copy()
            w[~np.isfinite(a)] = np.nan          # те же дыры, что и в F
            ws.append(w)
        filled = BM.fill_many(net, ws, starts, ck["scale"], dev)
        # Наружу уходит модуль восстановленного вектора, и только он идёт в
        # метрику. Сами каналы по пути выбрасывались, из-за чего предсказанные
        # X, Y, Z нельзя было нарисовать. Теперь, если попросили сток, они
        # складываются в него — модуль от этого не меняется ни на бит, он
        # по-прежнему считается из тех же трёх каналов.
        if comp_sink is not None:
            comp_sink["pred"] = [np.asarray(o, np.float32) for o in filled]
            comp_sink["true"] = [np.asarray(comp[y][st:st + C.W], np.float32)
                                 for st, y in zip(starts, years)]
        return [np.sqrt((o ** 2).sum(axis=1)) for o in filled]

    return fill_comp, ck


def _build_length_fields(smps, fn, acts, L, comp_sink=None):
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
    w0 = np.zeros(n, np.int32)      # старт окна в годовом ряду
    blk = np.zeros(n, np.int32)
    errs = []
    preds = fn([s["input"] for s in smps], [s["start"] for s in smps],
               [s.get("year") for s in smps])
    for i, s in enumerate(smps):
        pred = preds[i]
        a = int(s["pos"][0])
        lo, hi = a - C.MARGIN, a + L + C.MARGIN
        clo, chi = max(0, lo), min(C.W, hi)
        d0 = clo - lo
        tr[i, d0:d0 + chi - clo] = s["target"][clo:chi]
        pr[i, d0:d0 + chi - clo] = pred[clo:chi]
        mk = s["mask_art"]
        errs.append(np.abs(pred[mk] - s["target"][mk]).mean())
        st = s["start"]
        # Метка возмущённости окна. acts — готовые значения по СТАРТАМ окна
        # (C.window_activity), а не поминутный ряд: усреднять поминутный
        # скользящий индекс по окну нельзя, его рамка вылезает за края окна
        # на 12 часов в каждую сторону и метка описывает соседние полсуток.
        act[i] = acts[s["year"]][st]
        g0[i] = a
        w0[i] = st
        # блок для блочного бутстрэпа (bench_metrics): год и календарная
        # неделя, в которую попадает НАЧАЛО дыры. Окна в дампе перекрываются
        # (на длинных дырах каждая минута года лежит в ~8 дырах), поэтому
        # ресэмплировать по окнам нельзя — только неделями целиком.
        blk[i] = int(s["year"]) * 100 + (st + a) // C.BLOCK
    fields = dict(true=tr, pred=pr, act=act, gap0=g0, block=blk, win0=w0)
    # Компонентные панели — тот же вырез окна, что у модуля, но по трём
    # каналам. Хранятся только когда просили: на полном бенчмарке это +218 МБ
    # на дамп ради трёх окон, которые нужны рисунку.
    if comp_sink and comp_sink.get("pred"):
        nch = comp_sink["pred"][0].shape[1]
        trc = np.full((n, span, nch), np.nan, np.float32)
        prc = np.full((n, span, nch), np.nan, np.float32)
        for i, s_ in enumerate(smps):
            a_ = int(s_["pos"][0])
            lo, hi = a_ - C.MARGIN, a_ + L + C.MARGIN
            clo, chi = max(0, lo), min(C.W, hi)
            d0 = clo - lo
            trc[i, d0:d0 + chi - clo] = comp_sink["true"][i][clo:chi]
            prc[i, d0:d0 + chi - clo] = comp_sink["pred"][i][clo:chi]
        fields["trueXYZ"] = trc
        fields["predXYZ"] = prc
    return fields, float(np.mean(errs))


def _fields_to_npz(fields_by_L):
    """Поля всех длин в плоский словарь для npz. Одна точка на оба пути
    сохранения: иначе новое поле (как компонентные панели) приходится
    добавлять дважды и в одном из путей о нём забывают."""
    out = {}
    for L, fields in fields_by_L.items():
        for key in ("true", "pred", "act", "gap0", "win0", "block",
                    "trueXYZ", "predXYZ"):
            if key in fields:
                out[f"L{L}_{key}"] = fields[key]
    return out


def _save_dump(args, code, lengths, fields_by_L):
    out = _fields_to_npz(fields_by_L)
    setname = C.setname(code, args.split)
    out["method"] = np.array(args.method)
    out["setname"] = np.array(setname)
    out["lengths"] = np.array(lengths, np.int32)
    out["margin"] = np.array(C.MARGIN, np.int32)
    out["seed"] = np.array(C.SEED, np.int32)
    out_dir = getattr(args, "out_dir", None) or C.OUT
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
    comp_sink = {} if getattr(args, "save_comp", False) else None
    fn, ck = resolve(args.method, getattr(args, "ckpt", None),
                     code=args.code, split=args.split, comp_sink=comp_sink)
    if comp_sink is not None and (ck is None or int(ck.get("NCH", 1)) == 1):
        raise SystemExit("--save-comp имеет смысл только для модели, обученной "
                         "на компонентах: у остальных методов трёх каналов нет")
    pick = None
    if getattr(args, "windows", None):
        pick = [int(x) for x in str(args.windows).replace(",", " ").split()]
        if args.align_codes:
            raise SystemExit("--windows и --align-codes вместе не поддержаны")
        print(f"окна выбраны вручную: {pick} (из первых {args.n} по плану)")
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
        acts_by_code = {c: {y: C.window_activity(F) for y, F in yy.items()}
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
                                                    acts_by_code[c], L,
                                                    comp_sink)
                if fields is None:
                    print(f"{L:>6}{c:>6}{0:>6}  нет окон")
                    continue
                fields_by_code[c][L] = fields
                print(f"{L:>6}{c:>6}{len(smps_by_code[c]):>6}{mae:>12.3f}")
        for c in codes:
            _save_dump(args, c, lengths, fields_by_code[c])
        return

    years = C.load_split(args.split, code=args.code)
    acts = {y: C.window_activity(F) for y, F in years.items()}
    setname = C.setname(args.code, args.split)

    print(f"метод={args.method}  набор={setname} ({', '.join(map(str, sorted(years)))})"
          f"  n={args.n}  margin={C.MARGIN}")
    print(f"{'len':>6}{'окон':>6}{'MAE, нТл':>12}")
    print("-" * 24)

    fields_by_L = {}
    plan = window_plan(lengths, lambda L, rng: C.gather(years, L, args.n, rng))
    for L, smps in plan:
        if pick is not None:
            # План окон детерминирован зерном, поэтому выбор по индексу даёт
            # ТЕ ЖЕ окна, что в полном прогоне, — номер окна в полном дампе и
            # здесь означает одно и то же.
            bad = [i for i in pick if i >= len(smps)]
            if bad:
                raise SystemExit(f"L={L}: окон всего {len(smps)}, нет номеров {bad}")
            smps = [smps[i] for i in pick]
        fields, mae = _build_length_fields(smps, fn, acts, L, comp_sink)
        if fields is None:
            print(f"{L:>6}{0:>6}  нет окон")
            continue
        fields_by_L[L] = fields
        print(f"{L:>6}{len(smps):>6}{mae:>12.3f}")

    if args.out:
        # ручной путь вывода — сохраняем напрямую, без _save_dump (та всегда
        # строит имя по code/split)
        out = _fields_to_npz(fields_by_L)
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
    ap.add_argument("--save-comp", action="store_true",
                     help="сохранить в дамп предсказанные и истинные X, Y, Z "
                          "(поля L<L>_predXYZ / L<L>_trueXYZ) помимо модуля. "
                          "На полном бенчмарке это +218 МБ на дамп, поэтому "
                          "рассчитано на точечный прогон вместе с --windows: "
                          "модуль и метрика от этого не меняются, панели нужны "
                          "только рисунку с примерами окон")
    ap.add_argument("--windows", default=None, metavar="НОМЕРА",
                     help="прогнать только окна с этими номерами (через "
                          "запятую), а не весь набор. Нумерация та же, что в "
                          "полном дампе с тем же --n: план окон детерминирован "
                          "зерном. Удобно, чтобы пересчитать одно выбранное "
                          "окно — например типичное для рисунка")
    ap.add_argument("--out-dir", default=None,
                     help="каталог для дампов вместо data/ (имя файла стандартное; "
                          "работает и с --align-codes, где --out игнорируется) — "
                          "для частичных прогонов (--lengths), которые потом "
                          "сливаются в основной дамп")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
