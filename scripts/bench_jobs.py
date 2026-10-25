r"""Задания для кластера: список команд по этапам -> cluster/jobs/<этап>.jsonl.

Каждое задание — одна команда (обучение, прогон, сводка) с признаком
«нужна ли видеокарта» и файлом-результатом: если он уже есть, очередь
задание пропускает, поэтому прерванный этап просто запускается заново.
Исполняет задания cluster/run_queue.py — по одному на видеокарту.

Порядок этапов повторяет логику работы: все выборы — только на валидации,
тестовые годы вскрываются последним этапом, когда выбирать уже нечего.
Следующий этап генерируется, когда закончен предыдущий: grid2 берёт лучшую
скорость из grid1, seeds — конфигурации из select, stations — финалистов из
finalists.

  baselines   CPU  интерполяция и сезонные базлайны на валидации всех станций
                   (отсчёт Score для сетей)
  grid1       GPU  подбор CSDI и MOMENT, ступень 1: три скорости, размер M
  grid2       GPU  ступень 2: размеры S и L при лучшей скорости
  griddumps   GPU  ячейки грида на валидации (1024 окна)
  select      CPU  победитель ячеек -> models/<сплит>/<арх>_best.json
  seeds       GPU  все сети на ARS, оба входа, 5 сидов
  seedval     GPU  их прогоны на валидации ARS
  finalists   CPU  пятёрка по среднему рангу на валидации, усреднённому по сидам
  stations    GPU  финалисты, обученные на каждой из 5 станций, оба входа, 5 сидов
  stationval  GPU  их прогоны на валидации своей станции
  test        GPU  тестовые годы: базлайны и все модели на ARS, перенос
                   финалистов ARS на станции, модели станций на всех станциях
  summary     CPU  сводка по сидам -> data/<сплит>/seeds_summary.csv

Запуск (из корня проекта):
    python -u scripts/bench_jobs.py baselines
    python -u cluster/run_queue.py cluster/jobs/baselines.jsonl --gpus 0,1,2,3,4,5,6,7
"""
import os
import sys
import json
import argparse

import bench_common as C
import bench_grid as G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
JOBS = os.path.join(ROOT, "cluster", "jobs")
PY = [sys.executable, "-u"]

CODES = ["ARS", "WNG", "CMO", "KAK", "HUA", "HER"]
OTHER = [c for c in CODES if c != C.CODE]
TESTS = ["test", "test_hard"]
BASELINES = ["pchip", "linear", "dayint", "seadec", "harmreg"]
NEW = ["csdi", "moment"]
ALL_ARCHS = list(G.GRID)
N = 1024
SEED_MODELS = os.path.join(C.MODELS, "seeds")
SEED_DUMPS = os.path.join(C.OUT, "seeds")
FINALISTS_JSON = os.path.join(SEED_DUMPS, "finalists.json")


def script(name):
    return os.path.join(HERE, name)


def job(name, cmd, done, gpu):
    return dict(name=name, gpu=bool(gpu), cmd=[str(x) for x in cmd],
                done=[done] if isinstance(done, str) else list(done))


def dump_path(method, code, split, out_dir=None):
    return os.path.join(out_dir or C.OUT, f"dump_{method}_{C.setname(code, split)}.npz")


def run_cmd(method, code, split, ckpt=None, out_dir=None):
    cmd = PY + [script("bench_run.py"), "--method", method, "--code", code,
                "--split", split, "--n", N]
    if ckpt:
        cmd += ["--ckpt", ckpt]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def seed_name(arch, code, comp, seed):
    """<арх>[_<станция>]_best[_xyz]_s<сид> — как у прежних моделей, плюс сид."""
    return (f"{arch}" + (f"_{code}" if code != C.CODE else "") + "_best"
            + ("_xyz" if comp else "") + f"_s{seed}")


def best_cfg(arch):
    return os.path.join(C.MODELS, f"{arch}_best.json")


def ready_archs(archs):
    ok = [a for a in archs if os.path.exists(best_cfg(a))]
    for a in archs:
        if a not in ok:
            print(f"  {a}: нет {os.path.relpath(best_cfg(a), ROOT)} — пропуск "
                  f"(для CSDI и MOMENT сначала этапы grid1..select)")
    return ok


def finalists(a):
    if a.finalists:
        return a.finalists
    if not os.path.exists(FINALISTS_JSON):
        raise SystemExit(f"нет {os.path.relpath(FINALISTS_JSON, ROOT)} — сначала этап finalists")
    with open(FINALISTS_JSON, encoding="utf-8") as f:
        fin = json.load(f)["finalists"]
    print(f"  финалисты: {', '.join(fin)}")
    return fin


def models(archs, codes, seeds):
    """(имя, путь .pt, станция обучения) по всем сочетаниям и обоим входам."""
    for arch in ready_archs(archs):
        for code in codes:
            for comp in (False, True):
                for s in seeds:
                    name = seed_name(arch, code, comp, s)
                    yield arch, code, comp, s, name, os.path.join(SEED_MODELS, name + ".pt")


def train_jobs(archs, codes, seeds):
    out = []
    for arch, code, comp, s, name, pt in models(archs, codes, seeds):
        cmd = PY + [script("bench_train.py"), "--arch", arch, "--code", code,
                    "--cfg", best_cfg(arch), "--seed", s, "--out", pt]
        if comp:
            cmd.append("--comp")
        out.append(job(name, cmd, pt[:-3] + ".json", True))
    return out


def dump_jobs(archs, codes, seeds, targets):
    """targets(станция обучения) -> [(станция проверки, набор)]."""
    out = []
    for arch, code, comp, s, name, pt in models(archs, codes, seeds):
        for tc, split in targets(code):
            out.append(job(f"{name}@{C.setname(tc, split)}",
                           run_cmd(name, tc, split, ckpt=pt, out_dir=SEED_DUMPS),
                           dump_path(name, tc, split, SEED_DUMPS), True))
    return out


# ------------------------------------------------------------------ этапы
def st_baselines(a):
    return [job(f"{m}_{C.setname(c, 'val')}", run_cmd(m, c, "val"), dump_path(m, c, "val"), False)
            for m in BASELINES for c in CODES]


def st_grid1(a):
    out = []
    for arch in a.archs or NEW:
        for lr in G.lrs_for(arch):
            cmd, pt = G.train_cmd(arch, lr, "M")
            out.append(job(G.cell_name(arch, lr, "M"), cmd, pt[:-3] + ".json", True))
    return out


def st_grid2(a):
    out = []
    for arch in a.archs or NEW:
        lr = G.best_lr(arch)
        if lr is None:
            raise SystemExit(f"{arch}: ступень 1 не завершена — сначала grid1")
        print(f"  {arch}: лучшая скорость ступени 1 = {lr:g}")
        for size in ("S", "L"):
            cmd, pt = G.train_cmd(arch, lr, size)
            out.append(job(G.cell_name(arch, lr, size), cmd, pt[:-3] + ".json", True))
    return out


def st_griddumps(a):
    out = []
    for arch in a.archs or NEW:
        cells = sorted(f[:-5] for f in os.listdir(G.GRID_DIR)
                       if f.startswith(arch + "__") and f.endswith(".json"))
        for cell in cells:
            pt = os.path.join(G.GRID_DIR, cell + ".pt")
            out.append(job(f"dump_{cell}", run_cmd(cell, C.CODE, "val", ckpt=pt),
                           dump_path(cell, C.CODE, "val"), True))
    return out


def st_select(a):
    archs = a.archs or NEW
    return [job("select_" + "_".join(archs),
                PY + [script("bench_grid.py"), "--select", "--arch"] + archs,
                [os.path.join(C.MODELS, f"{x}_best.json") for x in archs], False)]


def st_seeds(a):
    return train_jobs(a.archs or ALL_ARCHS, [C.CODE], a.seeds)


def st_seedval(a):
    return dump_jobs(a.archs or ALL_ARCHS, [C.CODE], a.seeds, lambda tc: [(C.CODE, "val")])


def st_finalists(a):
    return [job("finalists", PY + [script("bench_finalists.py"), "--archs"]
                + ready_archs(a.archs or ALL_ARCHS) + ["--seeds"] + a.seeds,
                FINALISTS_JSON, False)]


def st_stations(a):
    return train_jobs(finalists(a), OTHER, a.seeds)


def st_stationval(a):
    return dump_jobs(finalists(a), OTHER, a.seeds, lambda tc: [(tc, "val")])


def st_test(a):
    fin = finalists(a)
    out = [job(f"{m}_{C.setname(c, s)}", run_cmd(m, c, s), dump_path(m, c, s), False)
           for m in BASELINES for c in CODES for s in TESTS]
    # все сети ARS на тестах ARS
    out += dump_jobs(a.archs or ALL_ARCHS, [C.CODE], a.seeds,
                     lambda tc: [(C.CODE, s) for s in TESTS])
    # перенос финалистов ARS на остальные станции
    out += dump_jobs(fin, [C.CODE], a.seeds, lambda tc: [(c, s) for c in OTHER for s in TESTS])
    # модели станций: своя станция и перенос на все остальные, включая ARS
    out += dump_jobs(fin, OTHER, a.seeds, lambda tc: [(c, s) for c in CODES for s in TESTS])
    return out


def st_summary(a):
    return [job("seeds_summary", PY + [script("bench_seeds.py")],
                os.path.join(C.OUT, "seeds_summary.csv"), False)]


STAGES = {"baselines": st_baselines, "grid1": st_grid1, "grid2": st_grid2,
          "griddumps": st_griddumps, "select": st_select, "seeds": st_seeds,
          "seedval": st_seedval, "finalists": st_finalists, "stations": st_stations,
          "stationval": st_stationval, "test": st_test, "summary": st_summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--archs", nargs="+", default=None,
                    help="архитектуры (по умолчанию: для grid* — CSDI и MOMENT, "
                         "иначе — все с готовой конфигурацией)")
    ap.add_argument("--finalists", nargs="+", default=None,
                    help="финалисты вручную вместо выбранных этапом finalists")
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    a = ap.parse_args()

    jobs = STAGES[a.stage](a)
    os.makedirs(JOBS, exist_ok=True)
    os.makedirs(SEED_MODELS, exist_ok=True)
    os.makedirs(SEED_DUMPS, exist_ok=True)
    path = os.path.join(JOBS, f"{a.stage}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    done = sum(all(os.path.exists(d) for d in j["done"]) for j in jobs)
    gpu = sum(j["gpu"] for j in jobs)
    print(f"{a.stage}: заданий {len(jobs)} (на GPU {gpu}, на CPU {len(jobs) - gpu}), "
          f"уже готово {done} -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
