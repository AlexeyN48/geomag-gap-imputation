r"""Слой данных бенчмарка: сплит, нарезка окон, искусственные пропуски.

Общий для всех методов — только так окна совпадают и сравнение честно.
Импортируется bench_run.py и обучающими скриптами; сам не запускается.
"""
import os
import numpy as np

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CODE = "ARS"

SEED = 1234
W = int(os.environ.get("IGF_W", 8640))   # окно, мин (6 сут по умолч.);
                       # длиннее => больше контекста вокруг дыры
CTX = 120             # минимум контекста с каждой стороны дыры
MARGIN = 180          # поля вокруг дыры, уходящие в дамп (нужны M5 и M7)
LENGTHS = [5, 15, 60, 240, 720, 1440, 2880, 4320]
GAP_MIN, GAP_MAX = 5, 4320
MAX_REAL = 0.15       # доля реальных пропусков в окне, выше которой окно не берём

SPLIT = {
    "train": [2013, 2014, 2015, 2016, 2018, 2020, 2021, 2022, 2023],
    "val": [2025],
    "test": [2019],            # слепой
    "test_hard": [2024],       # предельно трудный, не слепой
}


def load_year(year, code=CODE):
    p = os.path.join(DATA, f"{code}_{year}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"нет {p} — сначала: python -u bench_prep.py")
    return np.load(p, allow_pickle=False)["F"].astype(np.float32)


def load_split(name, code=CODE):
    return {y: load_year(y, code=code) for y in SPLIT[name]}


def activity(F, smooth=1440):
    """Мера возмущённости: сглаженный модуль минутной разности. Считается по
    самим данным — внешние индексы в бенчмарк не подаются никому."""
    d = np.abs(np.diff(F, prepend=F[:1]))
    d = np.nan_to_num(d, nan=0.0)
    k = np.ones(smooth, dtype=np.float64) / smooth
    return np.convolve(d, k, mode="same")


def candidate_starts(real, gap_len):
    """Позиции искусственной дыры внутри окна: внутри дыры не должно быть
    реальных пропусков (иначе нет истины), по краям — не меньше CTX минут."""
    csum = np.concatenate(([0], np.cumsum(real)))
    nan_in_block = csum[gap_len:] - csum[:-gap_len]
    cand = np.flatnonzero(nan_in_block == 0)
    lo, hi = CTX, real.size - gap_len - CTX
    return cand[(cand >= lo) & (cand <= hi)]


def make_sample(F, start, gap_len, rng, year=None):
    win = F[start:start + W].astype(np.float32).copy()
    real = ~np.isfinite(win)
    cand = candidate_starts(real, gap_len)
    if cand.size == 0:
        return None
    s = int(rng.choice(cand))
    art = np.zeros(W, dtype=bool)
    art[s:s + gap_len] = True
    inp = win.copy()
    inp[art] = np.nan
    return dict(input=inp, target=win, mask_art=art, mask_real=real,
                pos=(s, gap_len), start=start, year=year)


def gather(years, gap_len, n, rng, max_real=MAX_REAL):
    """n окон с искусственной дырой длины gap_len, равномерно по годам набора."""
    arrs = [(y, F) for y, F in years.items() if F.size > W]
    out, tries = [], n * 300
    while len(out) < n and tries > 0:
        tries -= 1
        y, F = arrs[rng.integers(len(arrs))]
        start = int(rng.integers(0, F.size - W))
        smp = make_sample(F, start, gap_len, rng, year=y)
        if smp is not None and smp["mask_real"].mean() < max_real:
            out.append(smp)
    return out


def real_gap_lengths(years):
    """Длины реальных непрерывных пропусков — эмпирический пул для сэмплера."""
    lens = []
    for F in years.values():
        bad = ~np.isfinite(F)
        if not bad.any():
            continue
        d = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        lens.extend((ends - starts).tolist())
    lens = np.array([L for L in lens if GAP_MIN <= L <= GAP_MAX], dtype=int)
    return lens


MG_COVER = (0.10, 0.30)      # доля окна под искусственными дырами при обучении


def multi_gap_sample(F, start, rng, sampler):
    """Окно с несколькими искусственными дырами — плотная супервизия.

    При одной дыре под обучающий сигнал попадает ~3% окна, и любая модель
    недоучивается. Первая дыра берётся из сэмплера длин (чтобы распределение
    длин сохранялось), остальные добиваются логравномерно до целевого покрытия;
    между дырами оставляется зазор CTX."""
    win = F[start:start + W].astype(np.float32).copy()
    real = ~np.isfinite(win)
    blocked = real.copy()
    art = np.zeros(W, dtype=bool)
    target = int(rng.uniform(*MG_COVER) * W)
    lens = [min(max(1, sampler.sample(rng)), W - 2 * CTX - 1)]
    for _ in range(24):
        rem = target - sum(lens)
        if rem <= 0:
            break
        g = int(round(np.exp(rng.uniform(0.0, np.log(max(2, rem))))))
        lens.append(max(1, min(g, rem)))
    for g in sorted(lens, reverse=True):
        cand = candidate_starts(blocked, g)
        if cand.size == 0:
            continue
        s = int(rng.choice(cand))
        art[s:s + g] = True
        blocked[max(0, s - CTX):s + g + CTX] = True
    if not art.any():
        return None
    inp = win.copy()
    inp[art] = np.nan
    return dict(input=inp, target=win, mask_art=art, mask_real=real,
                pos=(0, lens[0]), start=start, year=None)


def compute_scale(years, n=1500, seed=SEED):
    """Робастный масштаб: медиана размаха 5-95% внутри окна ПОСЛЕ центрирования.
    Уровень поля плывёт по годам на сотни нТл, поэтому масштаб берётся от
    внутриоконной изменчивости, а не от абсолютных значений."""
    rng = np.random.default_rng(seed)
    arrs = [F for F in years.values() if F.size > W]
    vals = []
    for _ in range(n):
        F = arrs[rng.integers(len(arrs))]
        s = int(rng.integers(0, F.size - W))
        w = F[s:s + W]
        w = w[np.isfinite(w)]
        if w.size < W // 2:
            continue
        vals.append(np.percentile(w, 95) - np.percentile(w, 5))
    return float(np.median(vals)) if vals else 1.0


class GapSampler:
    """Длины дыр для ОБУЧЕНИЯ: гибрид эмпирического пула и логравномерного
    хвоста. Чисто эмпирический пул вырожден — реальных длин в диапазоне единицы,
    и длинный режим модель бы вообще не увидела."""

    def __init__(self, years, p_emp=0.7, gmin=GAP_MIN):
        self.gmin = gmin
        pool = real_gap_lengths(years)
        self.pool = pool[pool >= gmin] if pool.size else pool
        # при высоком gmin реальных длинных дыр почти нет => розыгрыш идёт
        # логравномерно из [gmin, GAP_MAX], т.е. чисто длинный режим
        self.p_emp = p_emp if self.pool.size else 0.0

    def sample(self, rng):
        if self.pool.size and rng.random() < self.p_emp:
            return int(rng.choice(self.pool))
        lo, hi = np.log(self.gmin), np.log(GAP_MAX)
        return int(round(np.exp(rng.uniform(lo, hi))))
