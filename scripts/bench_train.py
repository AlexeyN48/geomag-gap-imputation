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

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def make_batch(years, sampler, scale, rng, bs, long_only=False):
    """Создаёт батч (X, mask, Y, A) для обучения. X — входные признаки, mask — маска реальных пропусков,
    Y — целевые значения, A — маска искусственных."""
    Xs, Ms, Ys, As = [], [], [], []
    arrs = [F for F in years.values() if F.size > C.W]
    while len(Xs) < bs:
        F = arrs[rng.integers(len(arrs))]
        st = int(rng.integers(0, F.size - C.W))
        if long_only:
            # одна ДЛИННАЯ дыра на окно вместо смеси multi-gap
            L = min(sampler.sample(rng), C.W - 2 * C.CTX - 1)
            smp = C.make_sample(F, st, L, rng)
        else:
            smp = C.multi_gap_sample(F, st, rng, sampler)
        if smp is None or smp["mask_real"].mean() > C.MAX_REAL:
            continue
        X, msk, center = M.featurize(smp["input"], st, scale)
        # NaN вне искусственной дыры (реальные пропуски) обязаны быть занулены:
        # маскирование умножением их не убирает, 0 * NaN = NaN, и лосс целиком
        # становится нефинитным. Внутри искусственной дыры истина есть всегда.
        y = np.nan_to_num(((smp["target"] - center) / scale), nan=0.0).astype(np.float32)
        Xs.append(X); Ms.append(msk)
        Ys.append(y.reshape(M.WB, M.P))
        As.append(smp["mask_art"].reshape(M.WB, M.P))
    return (torch.from_numpy(np.stack(Xs)), torch.from_numpy(np.stack(Ms)),
            torch.from_numpy(np.stack(Ys)),
            torch.from_numpy(np.stack(As).astype(np.float32)))

def build_valset(years, scale, n=96, seed=C.SEED + 1, gmin=None):
    """Фиксированный валидационный набор: одиночные дыры, одни и те же окна для
    всех моделей и всех прогонов."""
    rng = np.random.default_rng(seed)
    sampler = C.GapSampler(years, gmin=gmin or C.GAP_MIN)
    out = []
    arrs = [F for F in years.values() if F.size > C.W]
    while len(out) < n:
        F = arrs[rng.integers(len(arrs))]
        st = int(rng.integers(0, F.size - C.W))
        L = min(max(gmin or C.GAP_MIN, sampler.sample(rng)), C.W - 2 * C.CTX - 1)
        smp = C.make_sample(F, st, L, rng)
        if smp is None or smp["mask_real"].mean() > C.MAX_REAL:
            continue
        X, msk, center = M.featurize(smp["input"], st, scale)
        out.append((X, msk, smp["target"], smp["mask_art"], center))
    return out


@torch.no_grad()
def validate(net, vals, scale, device, bs=16):
    """Получить предсказания модели на фиксированных vals и посчитать MAE только в тех местах, где были искусственно сделаны пропуски."""
    net.eval()
    tot = cnt = 0.0
    for i in range(0, len(vals), bs):
        chunk = vals[i:i + bs]
        xb = torch.from_numpy(np.stack([c[0] for c in chunk])).to(device)
        mb = torch.from_numpy(np.stack([c[1] for c in chunk])).to(device)
        out = net(xb, mb).cpu().numpy().reshape(len(chunk), -1)[:, :C.W]
        for j, (_, _, tgt, art, center) in enumerate(chunk):
            pred = out[j] * scale + center
            tot += np.abs(pred[art] - tgt[art]).sum()
            cnt += art.sum()
    net.train()
    return tot / max(cnt, 1)


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
    args = ap.parse_args()
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
    train = C.load_split("train")
    val = C.load_split("val")
    scale = C.compute_scale(train)
    GMIN = 720 if args.long_only else C.GAP_MIN
    sampler = C.GapSampler(train, gmin=GMIN)
    rng = np.random.default_rng(C.SEED)

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

    vals = build_valset(val, scale,
                        n=getattr(M.ARCH[args.arch], 'VAL_N', 96), gmin=GMIN)
    os.makedirs(MODELS, exist_ok=True)
    out = args.out or os.path.join(MODELS, f"{args.arch}.pt")

    print(f"устройство={device}  модель={args.arch}  параметров={npar/1e6:.3f} млн")
    print(f"масштаб={scale:.1f} нТл  шагов={args.steps}  батч={args.batch}x{args.accum}"
          f"  окон валидации={len(vals)}")
    print(f"{'шаг':>7}{'train':>10}{'val, нТл':>11}{'lr':>10}{'сек':>8}")
    print("-" * 46)

    best, t0, run, nonfin = np.inf, time.time(), 0.0, 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            X, msk, Y, A = make_batch(train, sampler, scale, rng, args.batch,
                                      long_only=args.long_only)
            X, msk, Y, A = X.to(device), msk.to(device), Y.to(device), A.to(device)
            if hasattr(net, "custom_loss"):
                # у диффузии своя цель (предсказание шума); подменять её общим
                # masked MAE нельзя, поэтому лосс берётся у самой модели
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
            print(f"{step:>7}{run/(args.every*args.accum):>10.4f}{v:>11.3f}"
                  f"{sched.get_last_lr()[0]:>10.2e}{time.time()-t0:>8.0f}")
            run = 0.0
            if v < best:
                best = v
                M.save(out, net, args.arch, scale, kw, step, v)

    print(f"\nготово. лучший val MAE = {best:.3f} нТл, чекпойнт {out}")
    if nonfin:
        print(f"нефинитных шагов: {nonfin}")
    with open(out.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(arch=args.arch, params=npar, steps=args.steps,
                       batch=args.batch, accum=args.accum, lr=args.lr,
                       best_val=best, scale=scale, kw=kw, patch=args.patch, window=int(C.W), long_only=args.long_only,
                       seconds=round(time.time() - t0)), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
