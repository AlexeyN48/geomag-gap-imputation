r"""Очередь заданий на одном узле с несколькими видеокартами.

Читает cluster/jobs/<этап>.jsonl (готовит scripts/bench_jobs.py) и выполняет
задания параллельно: задания на GPU — по одному на видеокарту
(CUDA_VISIBLE_DEVICES), задания на CPU — в --cpu-workers потоков. Задание,
у которого уже есть файл-результат, пропускается, поэтому после сбоя или
остановки этап запускается той же командой заново и продолжает с места.
Упавшее задание повторяется один раз; лог каждого — cluster/logs/<этап>/.

    python -u cluster/run_queue.py cluster/jobs/seeds.jsonl --gpus 0,1,2,3,4,5,6,7
    python -u cluster/run_queue.py cluster/jobs/baselines.jsonl --cpu-workers 32
    python -u cluster/run_queue.py cluster/jobs/seeds.jsonl --dry-run

Под SLURM: выделить узел целиком (8 GPU) и запустить эту же команду внутри
задания — см. cluster/slurm_stage.sh.
"""
import os
import sys
import json
import time
import queue
import argparse
import threading
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_print_lock = threading.Lock()


def say(msg):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def is_done(j):
    return all(os.path.exists(p) for p in j["done"])


def run_one(j, env, log_dir, retries):
    log = os.path.join(log_dir, j["name"].replace("/", "_").replace("@", "__") + ".log")
    for attempt in range(1 + retries):
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(j["cmd"], stdout=f, stderr=subprocess.STDOUT,
                               env=env, cwd=ROOT)
        mins = (time.time() - t0) / 60
        if r.returncode == 0 and is_done(j):
            return True, mins, log
        if attempt < retries:
            say(f"  сбой {j['name']} (код {r.returncode}) — повтор")
            time.sleep(15)          # дать драйверу освободить контекст
    return False, mins, log


def worker(q, gpu, base_env, log_dir, retries, stats):
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    tag = "cpu" if gpu is None else f"gpu{gpu}"
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        say(f"[{tag}] старт {j['name']}")
        ok, mins, log = run_one(j, env, log_dir, retries)
        with _print_lock:
            stats["ok" if ok else "fail"].append(j["name"])
        say(f"[{tag}] {'готово' if ok else 'ОШИБКА'} {j['name']}  {mins:.1f} мин"
            + ("" if ok else f"  лог: {os.path.relpath(log, ROOT)}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", help="cluster/jobs/<этап>.jsonl")
    ap.add_argument("--gpus", default=None,
                    help="номера видеокарт через запятую; по умолчанию все из "
                         "CUDA_VISIBLE_DEVICES либо 0..7")
    ap.add_argument("--cpu-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with open(a.jobs, encoding="utf-8") as f:
        jobs = [json.loads(x) for x in f if x.strip()]
    todo = [j for j in jobs if not is_done(j)]
    stage = os.path.splitext(os.path.basename(a.jobs))[0]
    print(f"{stage}: заданий {len(jobs)}, уже готово {len(jobs) - len(todo)}, "
          f"осталось {len(todo)} (GPU {sum(j['gpu'] for j in todo)}, "
          f"CPU {sum(not j['gpu'] for j in todo)})", flush=True)
    if a.dry_run:
        for j in todo[:10]:
            print("  ", " ".join(j["cmd"]))
        return
    if not todo:
        return

    if a.gpus is not None:
        gpus = [g for g in a.gpus.split(",") if g.strip() != ""]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        gpus = [str(i) for i in range(8)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.setdefault("IGF_SPLIT", "comp")
    log_dir = os.path.join(ROOT, "cluster", "logs", stage)
    os.makedirs(log_dir, exist_ok=True)

    qg, qc = queue.Queue(), queue.Queue()
    for j in todo:
        (qg if j["gpu"] else qc).put(j)
    stats = {"ok": [], "fail": []}
    threads = []
    if not qg.empty():
        if not gpus:
            raise SystemExit("есть задания на GPU, а видеокарт не задано (--gpus)")
        threads += [threading.Thread(target=worker, args=(qg, g, env, log_dir, a.retries, stats))
                    for g in gpus]
    if not qc.empty():
        threads += [threading.Thread(target=worker, args=(qc, None, env, log_dir, a.retries, stats))
                    for _ in range(a.cpu_workers)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\n{stage}: готово {len(stats['ok'])}, ошибок {len(stats['fail'])}, "
          f"{(time.time() - t0) / 3600:.1f} ч", flush=True)
    if stats["fail"]:
        print("упали:", ", ".join(stats["fail"][:20]), flush=True)
        sys.exit(1)
    print(f"ЭТАП {stage} ЗАВЕРШЁН", flush=True)


if __name__ == "__main__":
    main()
