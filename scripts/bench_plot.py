r"""Сводные рисунки бенчмарка: все методы на одном графике, по панели на метрику.

Читает готовые дампы и таблицы метрик.

Запуск:  cd bench_arti/scripts && python -u bench_plot.py --split val
"""
import os
import re
import csv
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_common as C

FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

STYLE = {                       # цвет и подпись закреплены за методом во всех
    "mean":   ("#8c8c8c", "Среднее"),        # рисунках, иначе их не сопоставить
    "locf":   ("#8c564b", "LOCF"),
    "linear": ("#e8820c", "Линейная"),
    "pchip":  ("#d62728", "PCHIP"),
    "daily":  ("#1f77b4", "Суточный шаблон"),
    "dlinear": ("#2ca02c", "DLinear"),
    "unet":    ("#9467bd", "U-Net"),
    "segrnn":  ("#17becf", "SegRNN"),
    "timesnet": ("#bcbd22", "TimesNet"),
    "saits":    ("#e377c2", "SAITS"),
    "imputeformer": ("#7f7f7f", "ImputeFormer"),
    "crossformer":  ("#8B0000", "Crossformer"),
    "csdi":         ("#00868B", "CSDI"),
}
ORDER = ["mean", "locf", "linear", "pchip", "daily",
         "dlinear", "unet", "segrnn", "timesnet", "saits",
         "imputeformer", "crossformer", "csdi"]

# панель: (ключ метрики, заголовок, подпись оси, опорная линия, лог. шкала)
PANELS = [
    ("MAE",  "MAE — средняя абсолютная ошибка", "MAE, нТл", None, True),
    ("RMSE", "RMSE — среднеквадратичная ошибка", "RMSE, нТл", None, True),
    ("NSE",  "NSE — эффективность Нэша-Сатклиффа", "NSE  (1 = идеал, 0 = как среднее)", 0.0, False),
    ("NMAE", "NMAE — ошибка / изменчивость сигнала", "NMAE = MAE / std(true в дыре)", None, False),
    ("MASE", "MASE — ошибка / наивный LOCF", "MASE = MAE / MAE(LOCF), той же длины L", 1.0, False),
]


CORE_METHODS = {"mean", "locf", "linear", "pchip", "daily", "dlinear", "unet",
                 "segrnn", "timesnet", "saits", "imputeformer", "crossformer", "csdi"}


def read_metrics(split):
    """{метод: {метрика: {длина: значение}}} из всех таблиц набора. Только
    основные 13 методов — варианты бюджета/ёмкости (unet_b16, saits_l4, ...)
    считались на 128 окнах вместо 1024 и не сравнимы построчно с остальными
    на графике; их место в отдельном разделе отчёта, не на общем рисунке."""
    out = {}
    pat = os.path.join(C.DATA, f"metrics_*_{C.CODE}_{split}*.csv")
    for path in sorted(glob.glob(pat)):
        name = os.path.basename(path)[len("metrics_"):-len(".csv")]
        method = name.split(f"_{C.CODE}_{split}")[0]
        if method not in CORE_METHODS:
            continue
        d = out.setdefault(method, {})
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["subset"] != "all":
                    continue
                d.setdefault(r["metric"], {})[int(r["gap_len"])] = float(r["value"])
    return out


def read_mae(split):
    """Абсолютная MAE из дампов — она есть даже без таблиц метрик."""
    out = {}
    for path in sorted(glob.glob(os.path.join(C.DATA, f"dump_*_{C.CODE}_{split}.npz"))):
        d = np.load(path, allow_pickle=False)
        m = str(d["method"])
        if m not in CORE_METHODS:
            continue   # варианты бюджета/ёмкости — в отдельном разделе отчёта
        margin = int(d["margin"])
        cur = {}
        for L in [int(x) for x in d["lengths"]]:
            if f"L{L}_true" not in d:
                continue
            t = d[f"L{L}_true"][:, margin:margin + L].astype(np.float64)
            p = d[f"L{L}_pred"][:, margin:margin + L].astype(np.float64)
            cur[L] = float(np.nanmean(np.abs(p - t)))
        out[m] = cur
    return out


def write_summary(met, methods, lengths, split):
    """Табличная версия того же сравнения: CSV (все методы x длины x метрики).
    Таблица как таковая (для чтения точных чисел) — в отчёте .docx, настоящей
    таблицей Word, а не картинкой; здесь только сырые данные."""
    csv_path = os.path.join(C.DATA, f"metrics_summary_{split}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,gap_len,MAE,RMSE,NSE,NMAE,MASE\n")
        for m in methods:
            for L in lengths:
                vals = [met.get(m, {}).get(k, {}).get(L) for k in
                        ("MAE", "RMSE", "NSE", "NMAE", "MASE")]
                if all(v is None for v in vals):
                    continue
                cells = ["" if v is None else f"{v:.4g}" for v in vals]
                f.write(f"{m},{L}," + ",".join(cells) + "\n")
    print(f"сохранён {csv_path}")
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=sorted(C.SPLIT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    met = read_metrics(args.split)
    mae = read_mae(args.split)
    if not mae:
        raise SystemExit(f"нет дампов для набора {args.split} — сначала bench_run.py")
    for m, v in mae.items():
        met.setdefault(m, {})["MAE"] = v

    methods = [m for m in ORDER if m in met] + \
              [m for m in sorted(met) if m not in ORDER]
    lengths = sorted({L for m in methods for k in met[m] for L in met[m][k]})

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 8.5))
    for ax, (key, title, ylab, ref, logy) in zip(axes.ravel(), PANELS):
        drawn = 0
        for m in methods:
            series = met.get(m, {}).get(key)
            if not series:
                continue
            xs = sorted(series)
            ys = [series[L] for L in xs]
            col, lab = STYLE.get(m, ("#333333", m))
            ax.plot(xs, ys, "-o", ms=4, color=col, label=lab)
            drawn += 1
        if ref is not None:
            ax.axhline(ref, color="k", lw=1, ls="--", alpha=0.7)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(lengths)
        ax.set_xticklabels(lengths, fontsize=8)
        ax.set_xlabel("длина пропуска, мин", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
        if drawn == 0:
            ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                    transform=ax.transAxes, color="#999999")
        elif key == "MAE":
            ax.legend(fontsize=8)

    for ax in axes.ravel()[len(PANELS):]:
        ax.axis('off')
    fig.suptitle(f"Сравнение методов восстановления пропусков — "
                 f"{C.CODE}, набор {args.split}", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.975))
    os.makedirs(FIGS, exist_ok=True)
    out = args.out or os.path.join(FIGS, f"fig_bench_all_metrics_{args.split}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"сохранён {out}")
    miss = [k for k, *_ in PANELS
            if not any(met.get(m, {}).get(k) for m in methods)]
    if miss:
        print("панели без данных: " + ", ".join(miss)
              + "  (посчитать bench_metrics.py)")

    write_summary(met, methods, lengths, args.split)


if __name__ == "__main__":
    main()
