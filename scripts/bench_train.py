r"""Обучение моделей бенчмарка. Бюджет и правила ОДНИ на всех.

Лосс — masked MAE только на искусственных пропусках
Отбор чекпойнта — по MAE на фиксированном валидационном наборе окон.

Запуск:  cd bench_arti/scripts
         python -u bench_train.py --arch dlinear --steps 8000
         python -u bench_train.py --arch unet --steps 8000 --batch 8 --accum 2
"""
import os
import time
import json
import argparse
import numpy as np
import torch

import bench_common as C
import bench_models as M

MODELS = C.MODELS          # у неосновного сплита — своя подпапка


def make_batch(years, sampler, scale, rng, bs, long_only=False, comps=None):
    """Создаёт батч (X, mask, Y, A) для обучения. X — входные признаки, mask — маска реальных пропусков,
    Y — целевые значения, A — маска искусственных.
    comps — {год: (n, 3)}: план окон и дыр считается по F, как и в основном
    эксперименте, а в модель идут компоненты того же окна. Так сравнение с
    результатами по модулю остаётся парным: те же недели, те же минуты."""
    Xs, Ms, Ys, As = [], [], [], []
    arrs = [(y, F) for y, F in years.items() if F.size > C.W
            and (comps is None or y in comps)]
    while len(Xs) < bs:
        yy, F = arrs[rng.integers(len(arrs))]
        st = int(rng.integers(0, F.size - C.W))
        if long_only:
            # одна ДЛИННАЯ дыра на окно вместо смеси multi-gap
            L = min(sampler.sample(rng), C.W - 2 * C.CTX - 1)
            smp = C.make_sample(F, st, L, rng)
        else:
            smp = C.multi_gap_sample(F, st, rng, sampler)
        if smp is None or smp["mask_real"].mean() > C.MAX_REAL:
            continue
        if comps is None:
            inp, tgt, art = smp["input"], smp["target"], smp["mask_art"]
        else:
            tgt = comps[yy][st:st + C.W]
            inp = tgt.copy(); inp[smp["mask_art"]] = np.nan
            art = np.repeat(smp["mask_art"][:, None], tgt.shape[1], axis=1)
            # дыра ставится по маске F, и внутри неё F есть всегда — а вот
            # компонента может отсутствовать (у CMO таких минут 227 на 4.9 млн).
            # Без этой строки цель там занулилась бы, и лосс тянул бы модель
            # к медиане окна вместо истины
            art &= np.isfinite(tgt)
        X, msk, center = M.featurize(inp, st, scale)
        # NaN вне искусственной дыры (реальные пропуски) обязаны быть занулены:
        # маскирование умножением их не убирает, 0 * NaN = NaN, и лосс целиком
        # становится нефинитным. Внутри искусственной дыры истина есть всегда.
        y = np.nan_to_num(((tgt - center) / scale), nan=0.0).astype(np.float32)
        Xs.append(X); Ms.append(msk)
        Ys.append(y.reshape(M.WB, M.POUT))
        As.append(art.reshape(M.WB, M.POUT))
    return (torch.from_numpy(np.stack(Xs)), torch.from_numpy(np.stack(Ms)),
            torch.from_numpy(np.stack(Ys)),
            torch.from_numpy(np.stack(As).astype(np.float32)))

VAL_PER_L = 30          # окон на каждую длину бенчмарка в валидационном наборе


def build_valset(years, scale, n_per_L=VAL_PER_L, seed=C.SEED + 1, lengths=None,
                 comps=None):
    """Фиксированный валидационный набор: по n_per_L окон на КАЖДУЮ длину
    бенчмарка. Одни и те же окна для всех моделей и всех прогонов.

    Почему по длинам, а не по естественному распределению пропусков. Раньше
    длины брались из пула реальных пропусков станции (медиана 24 мин), а ошибка
    считалась пулом по точкам. Из-за этого вклад окна был пропорционален длине
    его дыры, и сигнал оказывался предельно концентрированным: 6 окон длиннее
    1000 мин давали 63 % всех точек, а ОДНО окно на 3525 мин — 17 %. Момент
    остановки фактически выбирали пять окон, причём какие именно — решало зерно.

    Теперь каждая длина представлена одинаково, а свёртка в validate() —
    геометрическое среднее по длинам, как в бенчмарке по режимам. Вклад одного
    окна падает с 17 % до 0.3 %, и вес каждого режима ровно треть.

    При обучении на компонентах в модель идут X, Y, Z, но ошибка чекпойнта
    считается по собранному из них модулю — тогда момент остановки выбирается
    по тому же числу, что и в опыте по модулю."""
    rng = np.random.default_rng(seed)
    lengths = list(lengths or C.LENGTHS)
    arrs = [(y, F) for y, F in years.items() if F.size > C.W
            and (comps is None or y in comps)]
    if not arrs:
        raise SystemExit("нет годов для валидационного набора")
    out = []
    for L in lengths:
        got, tries = 0, 0
        while got < n_per_L:
            tries += 1
            if tries > 20000 * n_per_L:
                raise SystemExit(f"не удалось набрать {n_per_L} окон на длину {L}")
            yy, F = arrs[rng.integers(len(arrs))]
            st = int(rng.integers(0, F.size - C.W))
            smp = C.make_sample(F, st, L, rng)
            if smp is None or smp["mask_real"].mean() > C.MAX_REAL:
                continue
            if comps is None:
                inp = smp["input"]
            else:
                inp = comps[yy][st:st + C.W].copy()
                inp[smp["mask_art"]] = np.nan
            X, msk, center = M.featurize(inp, st, scale)
            out.append((X, msk, smp["target"], smp["mask_art"], center, L))
            got += 1
    return out


def build_valset_legacy(years, scale, n=96, seed=C.SEED + 1, gmin=None, comps=None):
    """Валидационный набор ПРЕЖНЕГО вида — только для диагностики (--val-compare).

    Длины дыр берутся из пула реальных пропусков станции, а не по длинам
    бенчмарка, и ошибка считается пулом по точкам. Сохранено, чтобы в одном
    прогоне сравнить, какой чекпойнт выбрал бы прежний критерий и какой
    выбирает новый: траектория обучения при этом одна и та же, и разница
    относится к критерию, а не к шуму обучения."""
    rng = np.random.default_rng(seed)
    sampler = C.GapSampler(years, gmin=gmin or C.GAP_MIN)
    out = []
    arrs = [(y, F) for y, F in years.items() if F.size > C.W
            and (comps is None or y in comps)]
    while len(out) < n:
        yy, F = arrs[rng.integers(len(arrs))]
        st = int(rng.integers(0, F.size - C.W))
        L = min(max(gmin or C.GAP_MIN, sampler.sample(rng)), C.W - 2 * C.CTX - 1)
        smp = C.make_sample(F, st, L, rng)
        if smp is None or smp["mask_real"].mean() > C.MAX_REAL:
            continue
        if comps is None:
            inp = smp["input"]
        else:
            inp = comps[yy][st:st + C.W].copy()
            inp[smp["mask_art"]] = np.nan
        X, msk, center = M.featurize(inp, st, scale)
        out.append((X, msk, smp["target"], smp["mask_art"], center, L))
    return out


@torch.no_grad()
def validate(net, vals, scale, device, bs=16, pooled=False):
    """MAE на фиксированном наборе vals, только в искусственных пропусках.

    Свёртка — ГЕОМЕТРИЧЕСКОЕ среднее MAE по длинам, а не пул по всем точкам.
    Пул весил бы каждое окно пропорционально длине его дыры: точка в дыре на
    4320 мин и точка в дыре на 5 мин входили бы одинаково, а точек в первой в
    864 раза больше. Геометрическое среднее даёт каждой длине равный вес — так
    же, как считается MAE по режимам в отчёте."""
    net.eval()
    agg = {}
    tot = cnt = 0.0
    for i in range(0, len(vals), bs):
        chunk = vals[i:i + bs]
        xb = torch.from_numpy(np.stack([c[0] for c in chunk])).to(device)
        mb = torch.from_numpy(np.stack([c[1] for c in chunk])).to(device)
        out = net(xb, mb).cpu().numpy().reshape(len(chunk), -1)[:, :C.W * M.NCH]
        for j, (_, _, tgt, art, center, L) in enumerate(chunk):
            if M.NCH == 1:
                pred = out[j] * scale + center
            else:   # компоненты -> модуль: ошибка чекпойнта всегда в нТл по F
                pred = np.sqrt((((out[j].reshape(C.W, M.NCH)
                                  * np.asarray(scale).reshape(1, -1)) + center) ** 2).sum(1))
            e = float(np.abs(pred[art] - tgt[art]).sum())
            a = agg.setdefault(L, [0.0, 0.0])
            a[0] += e
            a[1] += float(art.sum())
            tot += e
            cnt += float(art.sum())
    net.train()
    if pooled:                       # прежний критерий: пул по всем точкам
        return tot / max(cnt, 1)
    per_L = [t / c for t, c in agg.values() if c > 0]
    if not per_L:
        return float("inf")
    return float(np.exp(np.mean(np.log(np.maximum(per_L, 1e-9)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(M.ARCH))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--long-only", action="store_true",
                    help="обучать только на длинных дырах (>=720 мин)")
    ap.add_argument("--patch", type=int, default=5,
                    help="размер патча, мин (по умолч. 5)")
    ap.add_argument("--kw", nargs="*", default=[],
                    help="параметры ёмкости, напр. base=48 depth=5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-per-length", type=int, default=0,
                    help="окон на каждую длину бенчмарка в валидационном "
                         "наборе (всего 12 x это число). 0 — взять значение "
                         f"архитектуры либо общее ({VAL_PER_L}). Меньше — "
                         "быстрее валидация, но момент остановки шумнее")
    ap.add_argument("--val-compare", action="store_true",
                    help="диагностика: вести ОБА критерия отбора чекпойнта в "
                         "одном прогоне — новый (по длинам бенчмарка, "
                         "геометрическое среднее) и прежний (естественные "
                         "длины, пул по точкам). Второй чекпойнт пишется рядом "
                         "с суффиксом _legacy")
    ap.add_argument("--amp", action="store_true",
                    help="обучать в смешанной точности bfloat16 (autocast). "
                         "На тяжёлых архитектурах даёт до 2.7x по времени. "
                         "GradScaler не нужен: у bfloat16 тот же диапазон "
                         "экспоненты, что у float32, переполнения градиентов не "
                         "возникает. ВАЖНО: модели, обученные в разной точности, "
                         "нельзя ставить в один рейтинг — это разный бюджет")
    ap.add_argument("--code", default=C.CODE,
                    help="станция обучения (train/val/scale — её; по умолчанию ARS)")
    ap.add_argument("--gap-pool-code", default=C.CODE,
                    help="станция, чьи реальные длины пропусков идут в пул GapSampler; "
                         "по умолчанию ARS для всех — у KAK/HUA/HER своих пропусков нет, "
                         "а распределение длин при обучении должно быть одинаковым")
    ap.add_argument("--comp", action="store_true",
                    help="обучать на компонентах X, Y, Z вместо модуля F; окна и дыры "
                         "те же (план строится по F), ошибка чекпойнта — по собранному "
                         "из компонент модулю; годы с непригодным вектором отбрасываются")
    ap.add_argument("--cfg", default=None,
                    help="json чекпойнта (models/<arch>_best.json): взять оттуда lr, kw, "
                         "steps, batch, accum, patch — всё, что не задано явно в командной строке")
    args = ap.parse_args()
    if args.cfg:
        # конфигурация-победитель грида: явные аргументы командной строки важнее json
        with open(args.cfg, encoding="utf-8") as f:
            cfgj = json.load(f)
        for k in ("lr", "steps", "batch", "accum", "patch"):
            if getattr(args, k) == ap.get_default(k) and k in cfgj:
                setattr(args, k, cfgj[k])
        if not args.kw and cfgj.get("kw"):
            args.kw = [f"{k}={v}" for k, v in cfgj["kw"].items()]
        if args.arch != cfgj.get("arch", args.arch):
            raise SystemExit(f"--arch {args.arch} не совпадает с arch в {args.cfg}")
    # разбор key=val в int/float/строку — идёт в конструктор модели И в чекпойнт
    kw = {}
    for item in args.kw:
        k, v = item.split("=", 1)
        try:
            kw[k] = int(v)
        except ValueError:
            try:
                kw[k] = float(v)
            except ValueError:
                kw[k] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = C.load_split("train", code=args.code)
    val = C.load_split("val", code=args.code)
    if args.comp:
        ctrain = C.load_split_comp("train", code=args.code)
        cval = C.load_split_comp("val", code=args.code)
        train = {y: F for y, F in train.items() if y in ctrain}
        val = {y: F for y, F in val.items() if y in cval}
        scale = C.scale_comp(ctrain)
        M.set_channels(3)
    else:
        ctrain = cval = None
        scale = C.compute_scale(train)
    GMIN = 720 if args.long_only else C.GAP_MIN
    pool_years = train if args.gap_pool_code == args.code else C.load_split("train", code=args.gap_pool_code)
    sampler = C.GapSampler(pool_years, gmin=GMIN)
    rng = np.random.default_rng(C.SEED)
    # Начальные веса тоже от фиксированного зерна. Без этого два запуска одной
    # конфигурации отличаются не только тем, что мы меняли, и сравнивать их
    # нельзя: разница точности смешивается с разницей инициализации.
    torch.manual_seed(C.SEED)
    torch.cuda.manual_seed_all(C.SEED)

    M.set_patch(args.patch)          # представление входа до построения
    net = M.build(args.arch, **kw).to(device)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    def lam(s):
        if s < args.warmup:
            return (s + 1) / max(1, args.warmup)
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    # набор строится по ДЛИНАМ бенчмарка, поэтому gmin/--long-only к нему не
    # относятся: длины заданы явно, а не берутся из пула реальных пропусков
    n_per_L = args.val_per_length or getattr(M.ARCH[args.arch], "VAL_PER_L",
                                              VAL_PER_L)
    vals = build_valset(val, scale, comps=cval, n_per_L=n_per_L)
    vals_leg = (build_valset_legacy(val, scale, comps=cval, gmin=GMIN)
                if args.val_compare else None)
    os.makedirs(MODELS, exist_ok=True)
    # имя: <арх>[_<станция>][_<сплит>][_xyz].pt — сплит и вход попадают в имя,
    # иначе прогон на другом сплите затёр бы модели основного эксперимента
    name = args.arch
    if args.code != C.CODE:
        name += f"_{args.code}"
    if C.SPLIT_NAME != "f":
        name += f"_{C.SPLIT_NAME}"
    if args.comp:
        name += "_xyz"
    out = args.out or os.path.join(MODELS, f"{name}.pt")

    print(f"устройство={device}  модель={args.arch}  параметров={npar/1e6:.3f} млн  "
          f"станция={args.code}  пул длин дыр={args.gap_pool_code} ({sampler.pool.size} реальных длин)")
    print(f"вход: {'компоненты X, Y, Z' if args.comp else 'модуль F'}; "
          f"годы обучения: {', '.join(map(str, sorted(train)))}")
    print(f"масштаб={np.array2string(np.atleast_1d(scale), precision=1)} нТл  шагов={args.steps}  батч={args.batch}x{args.accum}"
          f"  окон валидации={len(vals)}")
    print(f"{'шаг':>7}{'train':>10}{'val, нТл':>11}{'lr':>10}{'сек':>8}")
    print("-" * 46)

    best, best_step, t0, run, nonfin = np.inf, 0, time.time(), 0.0, 0
    best_leg = [np.inf, 0]
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            X, msk, Y, A = make_batch(train, sampler, scale, rng, args.batch,
                                      long_only=args.long_only, comps=ctrain)
            X, msk, Y, A = X.to(device), msk.to(device), Y.to(device), A.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.amp and device == "cuda"):
                if hasattr(net, "custom_loss"):
                    # у диффузии своя цель (предсказание шума); подменять её
                    # общим masked MAE нельзя, лосс берётся у самой модели
                    loss = net.custom_loss(X, msk, Y, A)
                else:
                    pred = net(X, msk)
                    loss = (torch.abs(pred - Y) * A).sum() / A.sum().clamp(min=1)
            if not torch.isfinite(loss):
                nonfin += 1          # нефинитный лосс в backward не пускаем:
                continue             # один такой шаг убил бы веса необратимо
            (loss / args.accum).backward()
            run += float(loss)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.every == 0:
            v = validate(net, vals, scale, device)
            if vals_leg is not None:
                vl = validate(net, vals_leg, scale, device, pooled=True)
                if vl < best_leg[0]:
                    best_leg = [vl, step]
                    M.save(out[:-3] + "_legacy.pt", net, args.arch, scale, kw,
                           step, vl)
            print(f"{step:>7}{run/(args.every*args.accum):>10.4f}{v:>11.3f}"
                  f"{sched.get_last_lr()[0]:>10.2e}{time.time()-t0:>8.0f}")
            run = 0.0
            if v < best:
                best, best_step = v, step
                M.save(out, net, args.arch, scale, kw, step, v)

    print(f"\nготово. лучший val MAE = {best:.3f} нТл, чекпойнт {out} (шаг {best_step})")
    if vals_leg is not None:
        print(f"прежний критерий выбрал бы шаг {best_leg[1]} (его val "
              f"{best_leg[0]:.3f} нТл пулом по точкам)")
    if nonfin:
        print(f"нефинитных шагов: {nonfin}")
    # вариант ёмкости нигде не хранился: у размера M список kw пуст (это
    # дефолты конструктора), у S и L в нём лежит knob сетки. По файлу модели
    # понять, какая ячейка грида выиграла, было нельзя — восстанавливаем.
    size = None
    try:
        import bench_grid as G
        if args.arch in G.GRID:
            for sz in G.SIZES:
                if kw == G.kw_for(args.arch, sz):
                    size = sz
                    break
    except Exception:
        pass
    with open(out.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(arch=args.arch, params=npar, steps=args.steps,
                       batch=args.batch, accum=args.accum, lr=args.lr,
                       best_val=best, scale=np.atleast_1d(scale).astype(float).tolist(),
                       channels=int(M.NCH), comp=bool(args.comp),
                       kw=kw, patch=args.patch, window=int(C.W), long_only=args.long_only,
                       code=args.code, gap_pool_code=args.gap_pool_code, cfg=args.cfg,
                       amp="bfloat16" if args.amp else "float32", seed=int(C.SEED),
                       size=size, best_step=int(best_step),
                       seconds=round(time.time() - t0)), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
