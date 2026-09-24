r"""Примеры заполнения пропусков: как именно сети закрывают дыру.

Один рисунок: 3 панели — режимы длины, по одной представительной длине на
режим. В каждой — типичное окно: из 1024 тестовых окон этой длины берётся
то, что стоит посередине по средней (по сетям) MAE внутри дыры — половина
окон легче него, половина тяжелее. Среднее по сетям, а не по одной модели,
чтобы выбор окна никого не выделял.

Все методы на одном окне: дампы одного набора побитово одинаковы по окнам.
Рисуются только точки внутри дыры (снаружи вход = истина); истина — по всему
сохранённому фрагменту (дыра ± margin минут контекста).

Запуск:  cd scripts && python -u bench_examples.py --code ARS --split test
Выход:   figures/gap_examples/fig_examples_<код>_<сплит>.png
         render(..., style="report") — вариант для отчёта: 10 × 2.95 дюйма (ширина
         альбомной страницы A4), без подзаголовка и прямых подписей, режим — в
         заголовке панели; bench_report.py вставляет его в масштабе 1:1.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_common as C
from bench_regimes import PRETTY

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "gap_examples"))
ARCHS = ["unet", "saits", "crossformer", "timesnet", "imputeformer"]
NETS = [f"{a}_best" for a in ARCHS]       # переопределяется в main() по --train-code
# фиксированный порядок цветов (палитра проверена валидатором dataviz: all-pairs,
# light; пары с ΔE в полосе 6–8 закрыты прямыми подписями и разным штрихом)
COLORS = {"unet": "#2a78d6", "saits": "#eda100", "crossformer": "#1baf7a",
          "timesnet": "#4a3aa7", "imputeformer": "#e87ba4"}
STYLES = {"unet": "-", "saits": "-", "crossformer": "--", "timesnet": "-", "imputeformer": "--"}


def arch_of(n):
    return n.rsplit("_", 1)[0]
TRUTH, PCHIP_C, CTX_C = "#1a1a1a", "#8a8a8a", "#ececec"
REP_LEN = [(("≤120 мин"), 60), (("120–1000 мин"), 720), ((">1000 мин"), 2880)]


def load(method, code, split):
    return np.load(os.path.join(C.DATA, f"dump_{method}_{code}_{split}.npz"), allow_pickle=False)


def pick_windows(dumps, L):
    """Индексы медианного и P95-окна по средней по сетям MAE внутри дыры."""
    m = int(dumps[NETS[0]]["margin"])
    per = []
    for n in NETS:
        d = dumps[n]
        t = d[f"L{L}_true"][:, m:m + L].astype(np.float64)
        p = d[f"L{L}_pred"][:, m:m + L].astype(np.float64)
        per.append(np.nanmean(np.abs(p - t), axis=1))
    mean_mae = np.exp(np.mean(np.log(np.stack(per) + 1e-9), axis=0))
    order = np.argsort(mean_mae)
    med = order[len(order) // 2]
    return med, mean_mae


def when(dump, L, idx):
    """Когда было окно. Точной даты в дампе нет: L{L}_gap0 — это смещение дыры
    ВНУТРИ окна (0..W), а начало самого окна в году не сохраняется. Восстановимо
    только то, что лежит в L{L}_block = год*100 + номер календарной недели."""
    blk = int(dump[f"L{L}_block"][idx])
    return f"{blk // 100}, неделя {blk % 100 + 1}"


STYLE = {
    # direct=False везде: цветные подписи моделей прямо на поле мешают читать
    # кривые, названия остаются только в легенде
    "screen": dict(fig=(16, 4.8), lw=1.5, lw_true=1.2, title=9, tick=8, lab=9, leg=7, direct=False),
    "report": dict(fig=(10.0, 3.1), lw=1.1, lw_true=0.9, title=7.5, tick=6.5, lab=7, leg=5.8, direct=False),
    # сетка «станции × режимы» на одну книжную страницу (render_grid): без
    # легенды в панелях (общая сверху) и без заголовков панелей (подписи строк/столбцов)
    "grid": dict(lw=0.8, lw_true=0.65, title=7, tick=5.5, lab=6.5, leg=0, direct=False,
                 legend=False, notitle=True),
}


def draw_panel(ax, dumps, L, idx, code, split, st=STYLE["screen"], regime=None):
    m = int(dumps[NETS[0]]["margin"])
    ref = dumps["pchip"]
    truth = ref[f"L{L}_true"][idx].astype(np.float64)
    n_all = truth.size
    x = (np.arange(n_all) - m) / 60.0            # часы от начала дыры
    base = np.nanmean(truth[:m].tolist() + truth[m + L:].tolist())   # уровень контекста
    y0 = truth - base
    ax.axvspan(x[0], 0, color=CTX_C, lw=0, zorder=0)
    ax.axvspan(L / 60.0, x[-1], color=CTX_C, lw=0, zorder=0)
    ax.plot(x, y0, color=TRUTH, lw=st["lw_true"], zorder=3, label="истина")
    gap = slice(m, m + L)
    xg = x[gap]
    # PCHIP
    pp = ref[f"L{L}_pred"][idx].astype(np.float64)[gap] - base
    mae_p = np.nanmean(np.abs(pp - y0[gap]))
    ax.plot(xg, pp, color=PCHIP_C, lw=st["lw"], ls=(0, (2, 2)), zorder=2, label=f"PCHIP  {mae_p:.1f}")
    ends = []
    for n in NETS:
        pr = dumps[n][f"L{L}_pred"][idx].astype(np.float64)[gap] - base
        mae = np.nanmean(np.abs(pr - y0[gap]))
        ax.plot(xg, pr, color=COLORS[arch_of(n)], lw=st["lw"], ls=STYLES[arch_of(n)], zorder=4,
                label=f"{PRETTY[arch_of(n)]}  {mae:.1f}")
        ends.append((float(pr[-1]), PRETTY[arch_of(n)], COLORS[arch_of(n)]))
    # прямые подписи в правой полосе контекста: разведены по вертикали и
    # зажаты внутри осей (иначе вылезают на заголовок соседней панели)
    ax.set_xlim(x[0], x[-1])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi)
    span = hi - lo
    step, pad = span * 0.075, span * 0.04
    ends.sort()
    ys = [min(max(e[0], lo + pad), hi - pad) for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + step)
    over = ys[-1] - (hi - pad)
    if over > 0:
        ys = [v - over for v in ys]
    for (yv, nm, col), yy in (zip(ends, ys) if st["direct"] else []):
        ax.annotate(nm, xy=(xg[-1], yv), xytext=(xg[-1] + (x[-1] - xg[-1]) * 0.12, yy),
                    fontsize=7.5, color=col, va="center", ha="left", annotation_clip=True,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.7), zorder=6)
    head = f"режим {regime}\n" if regime else ""
    if not st.get("notitle"):
        ax.set_title(f"{head}пропуск {L} мин · {when(ref, L, idx)}", fontsize=st["title"], loc="left")
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.tick_params(labelsize=st["tick"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if st.get("legend", True):
        ax.legend(fontsize=st["leg"], loc="best", frameon=True, framealpha=0.9, edgecolor="none",
                  title="MAE в дыре, нТл", title_fontsize=st["leg"], ncol=1, handlelength=1.8,
                  borderpad=0.3, labelspacing=0.25)


def render(code, split, train_code="ARS", out=None, style="screen", lengths=None):
    """Рисунок «типичное окно» для одной станции; возвращает путь к png."""
    global NETS
    NETS = [f"{x}_best" if train_code == "ARS" else f"{x}_{train_code}" for x in ARCHS]
    st = STYLE[style]
    lengths = lengths or [L for _, L in REP_LEN]
    dumps = {n: load(n, code, split) for n in NETS + ["pchip"]}
    fig, axes = plt.subplots(1, 3, figsize=st["fig"])
    for j, ((g, _), L) in enumerate(zip(REP_LEN, lengths)):
        med, mm = pick_windows(dumps, L)
        draw_panel(axes[j], dumps, L, med, code, split, st, regime=g if style == "report" else None)
        axes[j].set_xlabel("часы от начала пропуска", fontsize=st["lab"])
        if style == "screen":
            axes[j].text(0.5, 1.16, f"режим {g}", transform=axes[j].transAxes, ha="center",
                         fontsize=11, fontweight="bold")
    axes[0].set_ylabel("F − уровень контекста, нТл", fontsize=st["lab"])
    year = C.SPLIT[split][0]
    if style == "screen":
        fig.suptitle(f"{code}, {year}: типичное окно · модели обучены на {train_code}", fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout(pad=0.4, w_pad=0.8)
    if out is None:
        suffix = "" if train_code == "ARS" else f"_train{train_code}"
        out = os.path.join(OUT, f"fig_examples_{code}_{split}{suffix}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=250 if style == "report" else 200)
    plt.close(fig)
    return out


def render_grid(rows, split, out, lengths=None, width=6.7, row_h=1.3):
    """Один рисунок на книжную страницу: строка — станция (code, train_code, подпись),
    столбец — режим длины, в каждой панели типичное окно этой станции. Общая легенда
    сверху; MAE по окнам не подписываются (они в таблицах отчёта)."""
    global NETS
    from matplotlib.lines import Line2D
    st = STYLE["grid"]
    lengths = lengths or [L for _, L in REP_LEN]
    n = len(rows)
    H = 0.45 + row_h * n
    fig, axes = plt.subplots(n, 3, figsize=(width, H), squeeze=False)
    for i, (code, tcode, label) in enumerate(rows):
        NETS = [f"{x}_best" if tcode == "ARS" else f"{x}_{tcode}" for x in ARCHS]
        dumps = {m: load(m, code, split) for m in NETS + ["pchip"]}
        for j, L in enumerate(lengths):
            ax = axes[i, j]
            med, _ = pick_windows(dumps, L)
            draw_panel(ax, dumps, L, med, code, split, st)
            if i == 0:
                ax.set_title(f"режим {REP_LEN[j][0]}\nпропуск {L} мин", fontsize=st["title"])
            if i == n - 1:
                ax.set_xlabel("часы от начала пропуска", fontsize=st["lab"])
            if j == 0:
                ax.set_ylabel(f"{label}, нТл", fontsize=st["lab"])
    handles = [Line2D([], [], color=TRUTH, lw=1.0, label="истина"),
               Line2D([], [], color=PCHIP_C, lw=1.0, ls=(0, (2, 2)), label="PCHIP")] + \
              [Line2D([], [], color=COLORS[a], lw=1.2, ls=STYLES[a], label=PRETTY[a]) for a in ARCHS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / H), pad=0.3, h_pad=0.4, w_pad=0.6)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="ARS")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lengths", type=int, nargs=3, default=[L for _, L in REP_LEN],
                    help="по одной длине на режим (короткий, средний, длинный)")
    ap.add_argument("--train-code", default="ARS",
                    help="станция обучения моделей: ARS -> <арх>_best, иначе <арх>_<код>")
    ap.add_argument("--style", default="screen", choices=sorted(STYLE))
    a = ap.parse_args()
    print("сохранён", render(a.code, a.split, a.train_code, style=a.style, lengths=a.lengths))


if __name__ == "__main__":
    main()
