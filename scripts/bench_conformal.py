r"""Доверительный интервал для заполненного пропуска: нормированное сплит-
конформное предсказание (normalized split conformal prediction) с проверкой
покрытия по группам.

ЗАЧЕМ. Модель выдаёт одно число на каждую минуту пропуска. Практику нужно
«± сколько нТл». Интервал строится из ошибок модели на данных, где истина
известна (искусственные пропуски), и проверяется на других данных.

ИДЕЯ (Papadopoulos et al.; см. DeWolf et al. arXiv:2309.08313 ур. 20 и
Nolte et al. arXiv:2402.14080 ур. 5–6).
  1. Трудность окна u — насколько беспокойно поле ВОКРУГ пропуска. Считается
     ТОЛЬКО по контексту (margin минут до и после дыры), потому что на
     настоящем пропуске внутри него данных нет. Считать u по всему окну —
     значит подсматривать ответ, и проверка станет недействительной.
  2. Нормированная ошибка (мера несоответствия):
         s = |y − ŷ| / (u + β)
     Деление на u убирает влияние обстановки: ошибка 5 нТл в спокойном окне
     (u=2) и 40 нТл в буре (u=15) дают почти одинаковые s. β не даёт делить
     на почти ноль.
  3. Калибровка: q̂ = квантиль уровня (1−α)(1 + 1/n) от s на КАЛИБРОВОЧНОМ
     наборе (n = число окон). Поправка 1+1/n — на конечность выборки.
  4. Интервал для нового окна:  ŷ ± q̂ · (u + β).
     Ширина своя у каждого окна: узкая в спокойное время, широкая в бурю.

ЧТО ЭТО ГАРАНТИРУЕТ. Только МАРГИНАЛЬНОЕ покрытие: 1−α в среднем по всему
набору. УСЛОВНОЕ (внутри спокойных / бурных окон отдельно) не гарантировано
ничем — его надо мерить. Поэтому скрипт печатает покрытие по группам: если в
бурной трети оно заметно ниже 1−α, значит u плохо описывает трудность.

ТРИ НАБОРА ДАННЫХ РАЗНЫЕ:
    обучение моделей  train (2013–2023)
    калибровка q̂      val   (2025)          <- --calib-split
    проверка покрытия test, test_hard        <- --test-splits
Калибровать и проверять на одном наборе нельзя: покрытие 1−α получится
автоматически, по определению квантиля, и ничего не будет значить.

ДВА ВИДА ПОЛОСЫ (--band):
    pointwise    — q̂ своё для каждой минуты дыры; гарантия «в этой минуте
                   истина внутри у 95 % окон».
    simultaneous — одно q̂ на окно по s = max по минутам; гарантия «ВСЯ
                   кривая целиком внутри полосы у 95 % окон» (строже).
Минуты внутри одного окна сильно зависимы, поэтому единица обмениваемости
здесь — ОКНО, а не минута: квантиль всегда берётся по окнам.

Запуск (ничего не пересчитывает сам — читает готовые дампы bench_run.py):
    cd scripts
    python -u bench_conformal.py --length 1440
    python -u bench_conformal.py --length 2880 --band simultaneous --fig
Нужны дампы метода на калибровочном наборе И на проверочных:
    dump_<метод>_<код>_val.npz, dump_<метод>_<код>_test.npz, ..._test_hard.npz
"""
import os
import csv
import argparse
import numpy as np

import bench_common as C

DATA = C.DATA
FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
DEFAULT_METHODS = ["unet_best", "saits_best", "crossformer_best",
                   "timesnet_best", "imputeformer_best", "pchip"]
DEFAULT_CODES = ["ARS", "WNG", "CMO", "KAK", "HUA", "HER"]
PRETTY = {"unet": "U-Net", "saits": "SAITS", "crossformer": "CrossFormer",
          "timesnet": "TimesNet", "imputeformer": "ImputeFormer", "csdi": "CSDI",
          "pchip": "PCHIP", "linear": "Linear", "locf": "LOCF"}


def name(m):
    return PRETTY.get(m.replace("_best", ""), m)


def dump_path(method, code, split):
    return os.path.join(DATA, f"dump_{method}_{code}_{split}.npz")


def load(method, code, split, L):
    """(истина в дыре, прогноз в дыре, трудность u по контексту) или None.

    Трудность считается по margin минутам ДО и ПОСЛЕ дыры — только по тому,
    что известно и на настоящем пропуске. Мера: среднее |ΔF| между соседними
    минутами контекста (тот же смысл, что у C.activity, но без окна
    сглаживания, которого на 180 минутах не построить)."""
    p = dump_path(method, code, split)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=False)
    if f"L{L}_true" not in d:
        return None
    m = int(d["margin"])
    full = d[f"L{L}_true"].astype(np.float64)
    true = full[:, m:m + L]
    pred = d[f"L{L}_pred"].astype(np.float64)[:, m:m + L]
    left, right = full[:, :m], full[:, m + L:]
    u = np.empty(full.shape[0])
    for i in range(full.shape[0]):
        dd = np.concatenate([np.abs(np.diff(left[i])), np.abs(np.diff(right[i]))])
        u[i] = np.nanmean(dd) if np.isfinite(dd).any() else np.nan
    return true, pred, u


def beta_auto(u):
    """β по умолчанию: 10 % медианной трудности калибровочного набора —
    порядок величины, ниже которого различия в u уже не осмысленны."""
    med = np.nanmedian(u)
    return 0.1 * med if np.isfinite(med) and med > 0 else 1e-3


def calib_level(n, alpha):
    """Уровень квантиля с поправкой на конечную выборку: (1−α)(1+1/n).
    Если он больше 1, калибровочных окон слишком мало для такого α —
    конечной гарантии нет (интервал формально бесконечен)."""
    return (1.0 - alpha) * (1.0 + 1.0 / n)


def fit(true, pred, u, beta, alpha, band):
    """q̂ по калибровочному набору: скаляр (simultaneous) или вектор по
    позициям внутри дыры (pointwise). Квантиль всегда по ОКНАМ."""
    s = np.abs(true - pred) / (u[:, None] + beta)          # (окна, минуты)
    ok = np.isfinite(u)
    s = s[ok]
    n = s.shape[0]
    lvl = calib_level(n, alpha)
    if lvl > 1.0:
        return None, n, lvl
    if band == "simultaneous":
        per_win = np.nanmax(s, axis=1)                      # худшая минута окна
        q = float(np.nanquantile(per_win, lvl))
    else:
        q = np.nanquantile(s, lvl, axis=0)                  # своё q на минуту
    return q, n, lvl


def apply_band(pred, u, beta, q):
    """Полуширина интервала для каждого окна и минуты: q·(u+β)."""
    half = np.asarray(q) * (u[:, None] + beta)
    return pred - half, pred + half


def coverage(true, lo, hi):
    """Две доли попаданий: по точкам и по окнам целиком."""
    inside = (true >= lo) & (true <= hi)
    fin = np.isfinite(true) & np.isfinite(lo)
    point = float(inside[fin].mean()) if fin.any() else np.nan
    whole = np.array([bool(inside[i][fin[i]].all()) if fin[i].any() else False
                      for i in range(true.shape[0])])
    return point, float(whole.mean()), inside, fin


def groups_by_u(u, n_groups=3):
    """Деление окон на равные по числу группы трудности: спокойные, средние,
    бурные. Границы — квантили u самого проверочного набора (это только
    разрез для отчёта, на построение интервала не влияет)."""
    lab = ["спокойные", "средние", "бурные"][:n_groups]
    edges = np.nanquantile(u, np.linspace(0, 1, n_groups + 1)[1:-1])
    idx = np.digitize(u, edges)
    return lab, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--codes", nargs="+", default=DEFAULT_CODES)
    ap.add_argument("--length", type=int, default=1440, help="длина пропуска, мин")
    ap.add_argument("--calib-split", default="val",
                    help="набор для калибровки q̂ (по умолчанию val = 2025)")
    ap.add_argument("--test-splits", nargs="+", default=["test", "test_hard"])
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 => интервал 95 %%")
    ap.add_argument("--band", default="pointwise", choices=["pointwise", "simultaneous"])
    ap.add_argument("--beta", type=float, default=None,
                    help="сглаживание в знаменателе; по умолчанию 0.1 медианы u")
    ap.add_argument("--flat", action="store_true",
                    help="СРАВНЕНИЕ: полоса постоянной ширины (u=1 для всех) — "
                         "показывает, что даёт нормировка")
    ap.add_argument("--fig", action="store_true", help="рисунок покрытия по группам")
    ap.add_argument("--out", default=None, help="куда класть CSV")
    a = ap.parse_args()
    L = a.length
    if a.calib_split in a.test_splits:
        raise SystemExit(f"калибровка и проверка на одном наборе ({a.calib_split}) — "
                         "покрытие получится {1-α} по определению квантиля и ничего не покажет")

    print(f"длина пропуска {L} мин;  калибровка: {a.calib_split} "
          f"({', '.join(map(str, C.SPLIT[a.calib_split]))});  проверка: "
          + ", ".join(f"{s} ({C.SPLIT[s][0]})" for s in a.test_splits))
    print(f"полоса: {a.band};  цель покрытия {100 * (1 - a.alpha):.0f} %"
          + (";  БЕЗ нормировки (--flat)" if a.flat else ""))

    rows = []
    for method in a.methods:
        for code in a.codes:
            cal = load(method, code, a.calib_split, L)
            if cal is None:
                print(f"  [нет] {name(method):13} {code}: нет дампа "
                      f"{os.path.basename(dump_path(method, code, a.calib_split))}")
                continue
            ct, cp, cu = cal
            if a.flat:
                cu = np.ones_like(cu)
            beta = a.beta if a.beta is not None else beta_auto(cu)
            q, n_cal, lvl = fit(ct, cp, cu, beta, a.alpha, a.band)
            if q is None:
                print(f"  [мало] {name(method):13} {code}: {n_cal} окон мало для α={a.alpha}"
                      f" (нужен квантиль уровня {lvl:.4f} > 1)")
                continue
            q_show = float(np.mean(q)) if np.ndim(q) else float(q)
            print(f"\n{name(method):13} {code}:  q̂={q_show:.2f}  β={beta:.3f}  "
                  f"окон калибровки {n_cal}  уровень {lvl:.4f}")

            for split in a.test_splits:
                tst = load(method, code, split, L)
                if tst is None:
                    print(f"    {split}: нет дампа")
                    continue
                tt, tp, tu = tst
                if a.flat:
                    tu = np.ones_like(tu)
                lo, hi = apply_band(tp, tu, beta, q)
                point, whole, inside, fin = coverage(tt, lo, hi)
                width = float(np.nanmean(hi - lo))
                lab, gi = groups_by_u(tu)
                parts = []
                for g, nm in enumerate(lab):
                    sel = gi == g
                    if not sel.any():
                        continue
                    f2 = fin[sel]
                    cov_g = float(inside[sel][f2].mean()) if f2.any() else np.nan
                    w_g = float(np.nanmean((hi - lo)[sel]))
                    parts.append(f"{nm} {100 * cov_g:.1f}% (ширина {w_g:.1f})")
                    rows.append(dict(method=method, code=code, split=split, length=L,
                                     group=nm, coverage=round(cov_g, 4),
                                     width_nT=round(w_g, 2), n_windows=int(sel.sum()),
                                     band=a.band, alpha=a.alpha, calib=a.calib_split,
                                     q=round(q_show, 4), beta=round(beta, 4)))
                rows.append(dict(method=method, code=code, split=split, length=L,
                                 group="все", coverage=round(point, 4),
                                 width_nT=round(width, 2), n_windows=int(tt.shape[0]),
                                 band=a.band, alpha=a.alpha, calib=a.calib_split,
                                 q=round(q_show, 4), beta=round(beta, 4)))
                print(f"    {split:9} покрытие по точкам {100 * point:.1f} %, "
                      f"окон целиком {100 * whole:.1f} %, средняя ширина {width:.1f} нТл")
                print(f"      по группам: " + ";  ".join(parts))

    if not rows:
        raise SystemExit("нечего сохранять: не нашлось ни одной пары калибровка+проверка")
    out = a.out or os.path.join(DATA, f"conformal_{a.band}_L{L}_{a.calib_split}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nсохранено: {out}")
    print("ЧИТАТЬ ТАК: строка «все» — маргинальное покрытие (должно быть ≈ "
          f"{100 * (1 - a.alpha):.0f} %, оно и так почти всегда сходится). Смотреть надо на "
          "группы: если в бурных заметно ниже цели, а в спокойных заметно выше — "
          "трудность u описывает обстановку плохо, и интервалу нельзя доверять "
          "поокновно. Сравнить с --flat: там разрыв между группами должен быть больше.")

    if a.fig:
        make_fig(rows, L, a)


def make_fig(rows, L, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = ["спокойные", "средние", "бурные"]
    methods = sorted({r["method"] for r in rows}, key=lambda m: DEFAULT_METHODS.index(m)
                     if m in DEFAULT_METHODS else 99)
    splits = [s for s in a.test_splits if any(r["split"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4.2), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for m in methods:
            ys = []
            for g in labs:
                v = [r["coverage"] for r in rows
                     if r["method"] == m and r["split"] == split and r["group"] == g]
                ys.append(100 * np.mean(v) if v else np.nan)
            ax.plot(labs, ys, marker="o", lw=1.4, label=name(m))
        ax.axhline(100 * (1 - a.alpha), color="#555", ls="--", lw=1,
                   label=f"цель {100 * (1 - a.alpha):.0f} %")
        ax.set_title(f"{split} ({C.SPLIT[split][0]}), пропуск {L} мин")
        ax.set_ylabel("покрытие, %")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Доля попаданий истины в интервал по группам трудности "
                 f"(калибровка на {a.calib_split}, полоса {a.band})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIGS, f"fig_conformal_{a.band}_L{L}.png")
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"рисунок: {out}")


if __name__ == "__main__":
    main()
