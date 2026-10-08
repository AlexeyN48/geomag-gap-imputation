r"""Типичное окно по компонентам: четыре панели X, Y, Z и модуль.

Зачем отдельно от bench_examples.py. Тот рисунок отвечает на вопрос «как
выглядит заполнение на разных длинах» и показывает только модуль. Этот
отвечает на другой: что происходит с ТРЕМЯ компонентами, когда сеть
восстанавливает вектор, и видно ли глазом разницу между подачей модуля и
подачей компонент. Поэтому панели здесь — по осям, а не по режимам длины.

Окно выбирается тем же правилом, что в bench_examples: медиана по средней
(по финалистам) ошибке внутри дыры на входе-МОДУЛЕ. Правило намеренно одно и
то же, и считается оно по модульным дампам, а не по компонентным: тогда оба
рисунка показывают ОДНО окно и их можно сличать. Медиана, а не «показательное»
окно, — чтобы выбор нельзя было подогнать; среднее по пятёрке, а не по одной
модели, — чтобы выбор никого не выделял.

Предсказанные компоненты в дампах не лежат: fill_comp сворачивает три канала
в модуль, и только модуль идёт в метрику. Хранить каналы по всему бенчмарку
значит +218 МБ на дамп ради трёх окон, поэтому здесь делается точечный прогон
ровно выбранного окна (bench_run --windows N --save-comp). Окно при этом то
же самое: план окон детерминирован зерном, так что номер окна в полном дампе
и в точечном прогоне означает одно и то же.

Запуск:  cd scripts && python -u bench_examples_comp.py --code ARS --split val
Выход:   figures/gap_examples/fig_examples_comp_<код>_<сплит>_L<длина>.png
"""
import os
import sys
import argparse
import datetime as dt
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_common as C
import bench_examples as E
from bench_regimes import PRETTY

OUT = E.OUT
CHANNELS = [("X", "север"), ("Y", "восток"), ("Z", "вниз")]

# Пятёрка финалистов. В bench_examples палитра на четыре архитектуры (TSMixer
# попал в пятёрку только после подбора гиперпараметров), поэтому цвет и штрих
# для него задаются здесь, а остальные берутся оттуда без изменений — чтобы
# одна и та же модель на обоих рисунках была одного цвета.
FINALISTS = ["unet", "saits", "timesnet", "imputeformer", "tsmixerx"]
COLORS = dict(E.COLORS, tsmixerx="#1b9e77")
STYLES = dict(E.STYLES, tsmixerx="-.")


def when(dump, L, idx):
    """Дата начала дыры. Раньше восстанавливалось только «год, неделя N»:
    старт окна в дампы не писался. Теперь есть L{L}_win0, и дата считается
    точно — год плюс смещение в минутах."""
    key = f"L{L}_win0"
    blk = int(dump[f"L{L}_block"][idx])
    year = blk // 100
    if key not in dump.files:
        return f"{year}, неделя {blk % 100 + 1}"
    mins = int(dump[key][idx]) + int(dump[f"L{L}_gap0"][idx])
    t = dt.datetime(year, 1, 1) + dt.timedelta(minutes=mins)
    return t.strftime("%d.%m.%Y %H:%M")


def targeted_run(method, code, split, L, idx, n, workdir, refresh=False):
    """Точечный прогон одного окна с сохранением компонент -> путь к дампу.

    Результат КЛАДЁТСЯ НА ДИСК и переиспользуется: имя несёт метод, набор,
    длину и номер окна, так что файл однозначно описывает, что внутри.
    Раньше прогон шёл во временный каталог и стирался — тогда любая
    перекомпоновка рисунка (другой состав моделей, другой стиль, панель на
    модель) требовала заново гонять сети. На тестовых годах это вдобавок
    означало бы лишний прогон по тестовым данным."""
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir,
                       f"ex_{method}_{C.setname(code, split)}_L{L}_w{idx}.npz")
    if os.path.exists(out) and not refresh:
        return out, True
    cmd = [sys.executable, "-u", "bench_run.py", "--method", method,
           "--code", code, "--split", split, "--n", str(n),
           "--lengths", str(L), "--windows", str(idx),
           "--save-comp", "--out", out]
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0 or not os.path.exists(out):
        raise SystemExit(f"точечный прогон {method} не вышел:\n"
                         f"{(r.stderr or r.stdout or '')[-1500:]}")
    return out, False


def ctx_minutes(L, margin, want=None):
    """Сколько контекста показывать с каждой стороны. По умолчанию — доля от
    длины дыры: на часовой дыре три часа контекста по краям сжимают саму дыру
    в узкую полоску, а рисунок про дыру, а не про то, что вокруг. Больше
    сохранённого margin всё равно нет."""
    if want is not None:
        return int(min(max(want, 0), margin))
    return int(min(margin, max(30, L // 6)))


def _level(y, m, L):
    """Уровень контекста: среднее по наблюдённым краям, без самой дыры.
    Вычитается, чтобы панели разных компонент были сопоставимы по масштабу —
    у X, Y и Z уровни различаются на десятки тысяч нанотесл."""
    edges = np.concatenate([y[:m], y[m + L:]])
    v = np.nanmean(edges)
    return 0.0 if not np.isfinite(v) else float(v)


def draw(dumps_f, comp_paths, L, idx, code, split, archs, st, ctx=None):
    """Четыре панели сверху вниз: X, Y, Z, модуль."""
    cp = {a: np.load(p, allow_pickle=False) for a, p in comp_paths.items()}
    ref = dumps_f["pchip"]
    m = int(ref["margin"])
    span = ref[f"L{L}_true"][idx].size
    x = (np.arange(span) - m) / 60.0              # часы от начала дыры
    gap = slice(m, m + L)
    xg = x[gap]
    # окно показа: дыра плюс немного контекста по краям. Уровень по-прежнему
    # считается по ВСЕМУ сохранённому контексту, обрезка только визуальная.
    c = ctx_minutes(L, m, ctx)
    xlim = (-c / 60.0, (L + c) / 60.0)

    fig, axes = plt.subplots(4, 1, figsize=st["fig"], sharex=True)
    first = cp[archs[0]]
    for row, (ch, side) in enumerate(CHANNELS):
        ax = axes[row]
        truth = first[f"L{L}_trueXYZ"][0][:, row].astype(np.float64)
        base = _level(truth, m, L)
        ax.axvspan(x[0], 0, color=E.CTX_C, lw=0, zorder=0)
        ax.axvspan(L / 60.0, x[-1], color=E.CTX_C, lw=0, zorder=0)
        ax.plot(x, truth - base, color=E.TRUTH, lw=st["lw_true"], zorder=3,
                label="истина")
        for a in archs:
            pr = cp[a][f"L{L}_predXYZ"][0][:, row].astype(np.float64)[gap] - base
            mae = np.nanmean(np.abs(pr - (truth[gap] - base)))
            ax.plot(xg, pr, color=COLORS[a], lw=st["lw"], ls=STYLES[a],
                    zorder=4, label=f"{PRETTY[a]}  {mae:.1f}")
        ax.set_ylabel(f"{ch} ({side}), нТл", fontsize=st["lab"])
        ax.set_title(f"компонента {ch}", fontsize=st["title"], loc="left")

    # модуль: истина, интерполяция и оба входа у каждой сети — ровно тот
    # вопрос, ради которого рисунок и нужен
    ax = axes[3]
    truth = ref[f"L{L}_true"][idx].astype(np.float64)
    base = _level(truth, m, L)
    ax.axvspan(x[0], 0, color=E.CTX_C, lw=0, zorder=0)
    ax.axvspan(L / 60.0, x[-1], color=E.CTX_C, lw=0, zorder=0)
    ax.plot(x, truth - base, color=E.TRUTH, lw=st["lw_true"], zorder=3,
            label="истина")
    pp = ref[f"L{L}_pred"][idx].astype(np.float64)[gap] - base
    ax.plot(xg, pp, color=E.PCHIP_C, lw=st["lw"], ls=(0, (2, 2)), zorder=2,
            label=f"PCHIP  {np.nanmean(np.abs(pp - (truth[gap] - base))):.1f}")
    ty = truth[gap] - base
    for a in archs:
        pf = dumps_f[f"{a}_best"][f"L{L}_pred"][idx].astype(np.float64)[gap] - base
        ax.plot(xg, pf, color=COLORS[a], lw=st["lw"], ls=STYLES[a], zorder=4,
                label=f"{PRETTY[a]}  {np.nanmean(np.abs(pf - ty)):.1f}")
    # Компонентный вход показан только у первой сети: десять кривых на одной
    # панели не читаются, а вопрос «меняет ли вход картину» виден и на одной.
    a0 = archs[0]
    pc = cp[a0][f"L{L}_predXYZ"][0].astype(np.float64)
    pxyz = np.sqrt((pc ** 2).sum(axis=1))[gap] - base
    ax.plot(xg, pxyz, color=COLORS[a0], lw=st["lw"] * 1.3, ls=(0, (4, 1.5)),
            zorder=6, label=f"{PRETTY[a0]}, вход компоненты  "
                            f"{np.nanmean(np.abs(pxyz - ty)):.1f}")
    ax.set_ylabel("модуль F, нТл", fontsize=st["lab"])
    ax.set_title(f"модуль F — вход модуль у всех, штрихом {PRETTY[a0]} "
                 f"на входе из компонент",
                 fontsize=st["title"], loc="left")
    ax.set_xlabel("часы от начала пропуска (серое — контекст, он виден методу)",
                  fontsize=st["lab"])
    vis = (x >= xlim[0]) & (x <= xlim[1])
    for ax in axes:
        ys = []
        for ln in ax.lines:
            xd, yd = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
            k = (xd >= xlim[0]) & (xd <= xlim[1]) & np.isfinite(yd)
            if k.any():
                ys.append((np.nanmin(yd[k]), np.nanmax(yd[k])))
        if ys:
            lo = min(v[0] for v in ys); hi = max(v[1] for v in ys)
            pad = (hi - lo) * 0.08 or 1.0
            # сверху места больше: там стоит легенда
            ax.set_ylim(lo - pad, hi + pad * 3.2)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.tick_params(labelsize=st["tick"])
        ax.legend(fontsize=st["leg"], loc="upper left", ncol=2,
                  framealpha=0.85, borderpad=0.3)
    fig.suptitle(f"Типичное окно: пропуск {L} мин, {code}, {when(ref, L, idx)}. "
                 f"Числа в легенде — MAE внутри дыры, нТл",
                 fontsize=st["title"] + 1, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", default="ARS")
    ap.add_argument("--split", default="val", choices=sorted(C.SPLIT))
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[L for _, L in E.REP_LEN],
                    help="по одной представительной длине на режим")
    ap.add_argument("--archs", nargs="+", default=FINALISTS,
                    help="какие сети рисовать; по умолчанию пятёрка финалистов")
    ap.add_argument("--ctx", type=int, default=None, metavar="МИН",
                    help="сколько минут контекста показывать по краям дыры. "
                         "По умолчанию доля от длины дыры (не меньше 30 мин и "
                         "не больше сохранённого margin): рисунок про дыру, а "
                         "не про то, что вокруг неё")
    ap.add_argument("--n", type=int, default=1024,
                    help="размер набора, по которому строился полный дамп — "
                         "от него зависит нумерация окон")
    ap.add_argument("--style", default="screen", choices=["screen", "report"])
    ap.add_argument("--dump-dir", default=os.path.join(C.OUT, "examples"),
                    help="куда класть точечные прогоны с компонентами. Они "
                         "сохраняются и переиспользуются: перекомпоновать "
                         "рисунок можно без GPU и без повторного прогона")
    ap.add_argument("--refresh", action="store_true",
                    help="пересчитать точечные прогоны, даже если файлы есть")
    a = ap.parse_args()

    st = dict(E.STYLE[a.style])
    st["fig"] = (10.0, 9.0) if a.style == "report" else (14.0, 11.0)
    # выбор окна — по модульным дампам финалистов, см. докстринг
    E.NETS = [f"{x}_best" for x in FINALISTS]
    need = E.NETS + ["pchip"]
    dumps_f = {}
    for n in need:
        p = os.path.join(C.OUT, f"dump_{n}_{C.setname(a.code, a.split)}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"нет дампа {os.path.basename(p)} — рисунок "
                             f"строится по готовому бенчмарку")
        dumps_f[n] = np.load(p, allow_pickle=False)

    os.makedirs(OUT, exist_ok=True)
    for L in a.lengths:
        idx, maes = E.pick_windows(dumps_f, L)
        print(f"L={L}: выбрано окно {idx} из {maes.size} — медиана по средней "
              f"ошибке финалистов ({maes[idx]:.2f} нТл), "
              f"{when(dumps_f['pchip'], L, idx)}")
        paths = {}
        for arch in a.archs:
            paths[arch], reused = targeted_run(f"{arch}_best_xyz", a.code,
                                               a.split, L, idx, a.n,
                                               a.dump_dir, a.refresh)
            print(f"  компоненты {'взяты с диска' if reused else 'пересчитаны'}:"
                  f" {arch}  ->  {os.path.basename(paths[arch])}")
        fig = draw(dumps_f, paths, L, idx, a.code, a.split, a.archs, st,
                   ctx=a.ctx)
        out = os.path.join(OUT, f"fig_examples_comp_{a.code}_{a.split}_L{L}.png")
        fig.savefig(out, dpi=170, facecolor="white")
        plt.close(fig)
        print(f"  сохранено: {out}  ({os.path.getsize(out)/1e3:.0f} КБ)")


if __name__ == "__main__":
    main()
