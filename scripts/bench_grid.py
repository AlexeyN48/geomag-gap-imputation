r"""Равный бюджет подбора для всех архитектур: ступенчатый грид lr × размер.

Зачем. В models/ у U-Net 13 вариантов, у SAITS 9, у N-HiTS/TiDE/TSMixer — по
одному: сравнение архитектур смешано с тем, сколько усилий в каждую вложено
(и «лучший из 13 попыток» выглядит лучше «единственной» уже по статистике).
Здесь каждая архитектура получает ОДИНАКОВЫЙ набор конфигураций и одно
правило выбора.

Протокол (см. GRID ниже):
  ступень 1 — lr ∈ {3e-4, 1e-3, 3e-3} при размере M (текущий дефолт);
  ступень 2 — размеры S и L при лучшем lr ступени 1 (по best_val);
  итого 5 прогонов на архитектуру (~50 ч GPU на 12 архитектур; полный 3×3
  стоил бы ~90 ч). Всё остальное фиксировано и одинаково: 12000 шагов,
  эффективный батч 16 (batch×accum как у базового прогона архитектуры —
  память), patch 5, окно 8640, seed 1234, masked-MAE, чекпойнт по val MAE.
«Размер» — один главный параметр ёмкости на архитектуру, S/L = вдвое
меньше/больше дефолта; у DLinear ёмкости нет (линейное отображение WB×WB),
его единственная структурная ручка — окно сглаживания тренда kernel.

Переиспользование. Если в models/ уже есть чекпойнт РОВНО такой же
конфигурации (тот же lr, kw, 12000 шагов, patch 5/дефолт), он копируется в
models/grid/ под именем ячейки, а не обучается заново (REUSE ниже, проверка
по json). Это не нарушает равенство бюджета: ячейка та же, сид тот же.

Результат ступеней — models/grid/{arch}__lr{lr}__{S|M|L}.pt/.json.

--select: выбор победителя на архитектуру НЕ по best_val, а по val-дампу.
best_val — ошибка на 96 окнах внутреннего val, которые (а) малы: разница
0.1 нТл на них — шум (TimesNet L 10.22 против M 10.32 по best_val, а на 1024
окнах L значимо хуже на всех длинах), и (б) почти все короткие (длины из
эмпирического пула, медиана 24 мин), тогда как архитектуры сравниваются по
10 длинам поровну. Поэтому:
  1. для каждой ячейки нужен дамп на val (1024 окна × 10 длин, те же окна,
     что у всех дампов; недостающие прогоняются здесь через bench_run.py
     --ckpt) и метрики с парным ДИ против БАЗОВОЙ ячейки (lr 1e-3, M);
  2. сводка ячейки — среднее log(MAE) по 10 длинам (геометрическое среднее:
     10% на 5-минутной дыре весят столько же, сколько 10% на трёхсуточной;
     арифметическое среднее целиком определялось бы длинными);
  3. ячейка допускается, только если по парному ДИ она НЕ проигрывает базе
     значимо ни на одной длине; победитель — допущенная ячейка с наименьшей
     сводкой, если она меньше базовой; иначе остаётся база.
Победитель -> models/{arch}_best.pt/.json, его val-дамп и метрики ->
data/dump_{arch}_best_ARS_val.npz и metrics_{arch}_best_*; таблица всех
ячеек -> models/grid_selection.csv. Чекпойнт ВНУТРИ каждого обучения
по-прежнему выбран по 96 окнам — это одинаково для всех архитектур и здесь
не трогается.

Запуск:  cd scripts
         python -u bench_grid.py --dry-run             # план и оценка времени
         python -u bench_grid.py --stage 1             # ступень 1 (дешёвые первыми)
         python -u bench_grid.py --stage 2             # ступень 2 (нужна готовая 1)
         python -u bench_grid.py --stage 1 --arch unet saits   # подмножество
         python -u bench_grid.py --select
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
GRID_DIR = os.path.join(MODELS, "grid")

STEPS = 12000
PATCH = 5
LRS = [3e-4, 1e-3, 3e-3]
SIZES = ("S", "M", "L")

# arch -> (главный параметр, {S, M, L: значение}, производные kw(value))
# порядок — по стоимости одного прогона (минуты на RTX 4060, по models/*.json)
GRID = {
    "dlinear":      ("kernel",     {"S": 13,  "M": 25,  "L": 49},  lambda v: {}),
    "nbeatsx":      ("mlp",        {"S": 128, "M": 256, "L": 512}, lambda v: {}),
    "nhits":        ("mlp",        {"S": 64,  "M": 128, "L": 256}, lambda v: {}),
    "tide":         ("hidden",     {"S": 128, "M": 256, "L": 512}, lambda v: {}),
    "tsmixerx":     ("ff_dim",     {"S": 32,  "M": 64,  "L": 128}, lambda v: {}),
    "segrnn":       ("d_model",    {"S": 64,  "M": 128, "L": 256}, lambda v: {}),
    "unet":         ("base",       {"S": 16,  "M": 32,  "L": 48},  lambda v: {}),
    "timesnet":     ("d_model",    {"S": 32,  "M": 64,  "L": 128}, lambda v: {"d_ffn": 2 * v}),
    "saits":        ("d_model",    {"S": 64,  "M": 128, "L": 256}, lambda v: {"d_ffn": 2 * v}),
    "imputeformer": ("d_model",    {"S": 32,  "M": 64,  "L": 128}, lambda v: {}),
    "csdi":         ("n_channels", {"S": 32,  "M": 64,  "L": 128}, lambda v: {}),
    "crossformer":  ("d_model",    {"S": 64,  "M": 128, "L": 256}, lambda v: {"d_ffn": v}),
}
COST_MIN = {"dlinear": 5, "nbeatsx": 5, "nhits": 6, "tide": 6, "tsmixerx": 6, "segrnn": 7,
            "unet": 8, "timesnet": 47, "saits": 73, "imputeformer": 82, "csdi": 151,
            "crossformer": 177}

# существующие чекпойнты, совпадающие с ячейкой грида (проверяется по json)
REUSE = {
    ("unet", 3e-4, "M"): "unet_lr3e4", ("unet", 3e-3, "M"): "unet_lr3e3",
    ("unet", 1e-3, "S"): "unet_b16",   ("unet", 1e-3, "L"): "unet_b48",
    ("segrnn", 1e-3, "S"): "segrnn_d64", ("segrnn", 1e-3, "L"): "segrnn_d256",
    ("saits", 1e-3, "L"): "saits_d256",
}
for _a in GRID:
    REUSE[(_a, 1e-3, "M")] = _a        # базовые прогоны = ячейка (1e-3, M)


def kw_for(arch, size):
    knob, vals, extra = GRID[arch]
    v = vals[size]
    if size == "M" and arch != "crossformer":
        # дефолт конструктора: kw пустой, как в базовых прогонах (для
        # crossformer M — не дефолт d_model=256, а вдвое меньше: 16 M параметров
        # при 256 делают L неподъёмным, поэтому его сетка сдвинута вниз)
        return {}
    return {knob: v, **extra(v)}


def cell_name(arch, lr, size):
    return f"{arch}__lr{lr:g}__{size}"


def base_batch(arch):
    """batch/accum базового прогона (память): эффективный батч 16 у всех."""
    p = os.path.join(MODELS, f"{arch}.json")
    if os.path.exists(p):
        j = json.load(open(p, encoding="utf-8"))
        return int(j.get("batch", 16)), int(j.get("accum", 1))
    return 16, 1


def json_matches(path, lr, kw):
    j = json.load(open(path, encoding="utf-8"))
    if int(j.get("steps", 0)) != STEPS or abs(float(j.get("lr", 0)) - lr) > 1e-12:
        return False
    if j.get("patch") not in (None, PATCH) or j.get("long_only"):
        return False
    jk = {k: v for k, v in (j.get("kw") or {}).items()}
    return jk == kw


def try_reuse(arch, lr, size, dst_pt):
    src = REUSE.get((arch, lr, size))
    if not src:
        return False
    sp, sj = os.path.join(MODELS, f"{src}.pt"), os.path.join(MODELS, f"{src}.json")
    if not (os.path.exists(sp) and os.path.exists(sj)):
        return False
    if not json_matches(sj, lr, kw_for(arch, size)):
        print(f"    {src}.json не совпадает с ячейкой — обучаем заново", flush=True)
        return False
    shutil.copyfile(sp, dst_pt)
    shutil.copyfile(sj, dst_pt.replace(".pt", ".json"))
    print(f"    переиспользован models/{src}.pt", flush=True)
    return True


def train(arch, lr, size, dry):
    name = cell_name(arch, lr, size)
    out = os.path.join(GRID_DIR, name + ".pt")
    if os.path.exists(out.replace(".pt", ".json")):
        print(f"  [{name}] готово, пропуск", flush=True)
        return
    print(f"  [{name}] ~{COST_MIN[arch]} мин", flush=True)
    if dry:
        return
    os.makedirs(GRID_DIR, exist_ok=True)
    if try_reuse(arch, lr, size, out):
        return
    batch, accum = base_batch(arch)
    kw = kw_for(arch, size)
    cmd = [sys.executable, "-u", os.path.join(HERE, "bench_train.py"), "--arch", arch,
           "--steps", str(STEPS), "--batch", str(batch), "--accum", str(accum),
           "--lr", f"{lr:g}", "--patch", str(PATCH), "--out", out]
    if kw:
        cmd += ["--kw"] + [f"{k}={v}" for k, v in kw.items()]
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(out.replace(".pt", ".log"), "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"обучение {name} упало, см. {out.replace('.pt', '.log')}")
    print(f"    ok, {(time.time() - t0) / 60:.0f} мин", flush=True)


def best_lr(arch):
    vals = {}
    for lr in LRS:
        p = os.path.join(GRID_DIR, cell_name(arch, lr, "M") + ".json")
        if os.path.exists(p):
            vals[lr] = float(json.load(open(p, encoding="utf-8"))["best_val"])
    if len(vals) < len(LRS):
        return None
    return min(vals, key=vals.get)


DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
SET = "ARS_val"


def _sh(args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-u"] + args, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-2500:], flush=True)
        raise SystemExit("упало: " + " ".join(args))


def _cells(arch):
    return sorted(f[:-5] for f in os.listdir(GRID_DIR)
                  if f.startswith(arch + "__") and f.endswith(".json"))


def _dump_name(arch, cell):
    """Имя метода в data/: базовая ячейка (lr 1e-3, M) — это сам arch, её дамп
    и метрики уже есть под именем arch; остальные — под именем ячейки."""
    return arch if cell == cell_name(arch, 1e-3, "M") else cell


def _copy_dump(src, dst, method):
    """Копия дампа под новым именем метода: поле method внутри дампа задаёт
    имена CSV в bench_metrics, поэтому его надо переписать."""
    import numpy as np
    with np.load(src, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["method"] = np.array(method)
    np.savez_compressed(dst, **d)


def _adopt_old_best(arch):
    """Дамп dump_{arch}_best_* от прежнего select (по best_val) отдать той
    ячейке, которой он принадлежит (по lr/kw в models/{arch}_best.json)."""
    bj = os.path.join(MODELS, f"{arch}_best.json")
    bd = os.path.join(DATA, f"dump_{arch}_best_{SET}.npz")
    if not (os.path.exists(bj) and os.path.exists(bd)):
        return
    j = json.load(open(bj, encoding="utf-8"))
    for cell in _cells(arch):
        cj = json.load(open(os.path.join(GRID_DIR, cell + ".json"), encoding="utf-8"))
        if cj.get("lr") == j.get("lr") and (cj.get("kw") or {}) == (j.get("kw") or {}):
            dst = os.path.join(DATA, f"dump_{_dump_name(arch, cell)}_{SET}.npz")
            if not os.path.exists(dst):
                _copy_dump(bd, dst, _dump_name(arch, cell))
                print(f"    дамп {cell} взят у прежнего {arch}_best", flush=True)
            return


def _cell_dump(arch, cell):
    """Путь к val-дампу ячейки; создаёт его, если нет. Переиспользованные
    чекпойнты (REUSE) уже имеют дампы под своими именами — они копируются."""
    dst = os.path.join(DATA, f"dump_{_dump_name(arch, cell)}_{SET}.npz")
    if os.path.exists(dst):
        return dst
    _, lr_s, size = cell.split("__")
    lr = float(lr_s[2:])
    src_name = REUSE.get((arch, lr, size))
    if src_name and os.path.exists(os.path.join(DATA, f"dump_{src_name}_{SET}.npz")) \
            and json_matches(os.path.join(MODELS, f"{src_name}.json"), lr, kw_for(arch, size)):
        _copy_dump(os.path.join(DATA, f"dump_{src_name}_{SET}.npz"), dst, cell)
        print(f"    дамп взят у {src_name}", flush=True)
        return dst
    t0 = time.time()
    _sh([os.path.join(HERE, "bench_run.py"), "--method", cell, "--split", "val", "--n", "1024",
         "--ckpt", os.path.join(GRID_DIR, cell + ".pt")])
    print(f"    прогон на val {time.time() - t0:.0f} с", flush=True)
    return dst


def _cell_metrics(arch, cell):
    """metrics_{ячейка}_ARS_val.csv и парный файл против базы (= arch);
    считает, если нет. У базовой ячейки метрики уже есть под именем arch."""
    name = _dump_name(arch, cell)
    mpath = os.path.join(DATA, f"metrics_{name}_{SET}.csv")
    if name == arch:
        return mpath, None
    ppath = os.path.join(DATA, f"metrics_{name}_vs_{arch}_{SET}.csv")
    if not (os.path.exists(mpath) and os.path.exists(ppath)):
        _sh([os.path.join(HERE, "bench_metrics.py"), "--pred", f"dump_{name}_{SET}.npz",
             "--base", f"dump_{arch}_{SET}.npz", "--no-fig"])
    return mpath, ppath


def _logmean_mae(mpath):
    import csv
    import numpy as np
    vals = [float(r["value"]) for r in csv.DictReader(open(mpath, encoding="utf-8"))
            if r["metric"] == "MAE" and r["subset"] == "all"]
    return float(np.mean(np.log(vals))), len(vals)


def _worse_somewhere(ppath):
    """Есть ли длина, где ячейка значимо ХУЖЕ базы по MAE (разница > 0)."""
    import csv
    bad = []
    for r in csv.DictReader(open(ppath, encoding="utf-8")):
        if r["metric"] == "MAE" and r["significant"] == "True" and float(r["mean_diff"]) > 0:
            bad.append(int(r["gap_len"]))
    return bad


def select():
    import csv
    rows, chosen = [], {}
    for arch in GRID:
        base_cell = cell_name(arch, 1e-3, "M")
        cells = _cells(arch)
        if base_cell not in cells:
            print(f"{arch}: нет базовой ячейки {base_cell} — пропуск", flush=True)
            continue
        print(f"\n[{arch}]", flush=True)
        _adopt_old_best(arch)
        info = {}
        for cell in cells:
            print(f"  {cell}", flush=True)
            _cell_dump(arch, cell)
            mpath, ppath = _cell_metrics(arch, cell)
            lm, n = _logmean_mae(mpath)
            bad = _worse_somewhere(ppath) if ppath else []
            bv = float(json.load(open(os.path.join(GRID_DIR, cell + ".json"), encoding="utf-8"))["best_val"])
            info[cell] = (lm, bad, bv, n)
        base_lm = info[base_cell][0]
        ok = {c: v for c, v in info.items() if not v[1]}
        win = min(ok, key=lambda c: ok[c][0])
        if info[win][0] >= base_lm:
            win = base_cell
        chosen[arch] = win
        for cell in cells:
            lm, bad, bv, n = info[cell]
            rows.append([arch, cell, f"{bv:.4f}", f"{lm:.5f}", f"{100 * (1 - __import__('math').exp(lm - base_lm)):+.1f}",
                         ";".join(map(str, bad)) if bad else "", "yes" if cell == win else ""])
        print(f"  -> {win}   (лог-средний MAE {info[win][0]:.4f}, база {base_lm:.4f})", flush=True)
        # победитель -> models/{arch}_best, его дамп и метрики -> *_best
        for ext in (".pt", ".json"):
            shutil.copyfile(os.path.join(GRID_DIR, win + ext), os.path.join(MODELS, f"{arch}_best{ext}"))
        for f in os.listdir(DATA):
            if f.startswith(f"dump_{arch}_best_") or f.startswith(f"metrics_{arch}_best_"):
                os.remove(os.path.join(DATA, f))
        _copy_dump(os.path.join(DATA, f"dump_{_dump_name(arch, win)}_{SET}.npz"),
                   os.path.join(DATA, f"dump_{arch}_best_{SET}.npz"), f"{arch}_best")
        _sh([os.path.join(HERE, "bench_metrics.py"), "--pred", f"dump_{arch}_best_{SET}.npz",
             "--base", f"dump_{arch}_{SET}.npz"])
    with open(os.path.join(MODELS, "grid_selection.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "cell", "best_val_96", "logmean_mae_1024", "gain_vs_base_pct",
                    "worse_than_base_at", "chosen"])
        w.writerows(rows)
    print(f"\nсохранено: {os.path.join(MODELS, 'grid_selection.csv')}")
    print("победители:", chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None)
    ap.add_argument("--arch", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    if a.select:
        select()
        return
    archs = a.arch or list(GRID)
    stages = [a.stage] if a.stage else [1, 2]
    total = 0
    for st in stages:
        print(f"\n=== ступень {st} ===", flush=True)
        for arch in archs:
            if st == 1:
                for lr in LRS:
                    name = cell_name(arch, lr, "M")
                    if not os.path.exists(os.path.join(GRID_DIR, name + ".json")) \
                            and (arch, lr, "M") not in REUSE:
                        total += COST_MIN[arch]
                    train(arch, lr, "M", a.dry_run)
            else:
                lr = best_lr(arch)
                if lr is None:
                    print(f"  {arch}: ступень 1 не завершена — пропуск", flush=True)
                    continue
                print(f"  {arch}: лучший lr ступени 1 = {lr:g}", flush=True)
                for size in ("S", "L"):
                    if not os.path.exists(os.path.join(GRID_DIR, cell_name(arch, lr, size) + ".json")) \
                            and (arch, lr, size) not in REUSE:
                        total += COST_MIN[arch]
                    train(arch, lr, size, a.dry_run)
    if a.dry_run:
        print(f"\nоценка обучения (без переиспользованных): ~{total / 60:.1f} ч")


if __name__ == "__main__":
    main()
