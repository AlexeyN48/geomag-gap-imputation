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
TOL_COMP = 3.0        # нТл: допуск |F - |XYZ|| для валидации и тестов. 1-2 нТл —
                      # настоящая приборная разность скаляра и флюксгейта
                      # (CMO 2019-2021), а не дефект; дефект начинается с десятков
LENGTHS = [5, 15, 60, 120, 240, 480, 720, 1000, 1440, 2160, 2880, 4320]
                      # 120 и 1000 добавлены позже: они лежат на переломах
                      # «интерполяция → сеть» (60..240) и между 720 и 1440,
                      # чтобы граница режимов была измерена, а не назначена;
                      # 480 и 2160 — чтобы в каждом из трёх режимов длины
                      # (≤120 | 120..1000 | >1000) было по четыре точки:
                      # {5,15,60,120} | {240,480,720,1000} | {1440,2160,2880,4320}
# Окна для каждой длины тянутся ОДНИМ генератором подряд, поэтому набор окон
# длины L зависит от того, какие длины шли до неё. Существующие дампы (94 шт.)
# сделаны так: исходные 8 длин — одним генератором в этом порядке, а 120 и
# 1000 — ОТДЕЛЬНЫМ прогоном (свежий генератор, порядок 120, 1000) и слиты в
# те же файлы. bench_run.py воспроизводит именно этот путь (см. там
# window_plan): иначе новый дамп получил бы другие окна для всех L ≥ 120,
# и парные сравнения/SS с существующими дампами стали бы невозможны.
LENGTHS_RNG_GROUPS = [[5, 15, 60, 240, 720, 1440, 2880, 4320], [120, 1000], [480, 2160]]
# NB: дампы для 480 и 2160 ещё НЕ прогнаны (сентябрь 2026): bench_run.py
# сгенерирует их как третью группу; существующие дампы дополняются через
# scratch-скрипт extend_dumps.py (NEW_L = [480, 2160]).
GAP_MIN, GAP_MAX = 5, 4320
MAX_REAL = 0.15       # доля реальных пропусков в окне, выше которой окно не берём
BLOCK = 7 * 1440      # блок для блочного бутстрэпа в bench_metrics: календарная
                      # неделя; окна внутри одной недели считаются зависимыми

# Наборы лет. Тестовые годы СЛЕПЫЕ в обоих сплитах: не участвуют ни в
# обучении, ни в выборе чекпойнта, ни в выборе конфигураций (грид — по val).
SPLITS = {
    # основной эксперимент, восстановление модуля F
    "f": {
        "train": [2013, 2014, 2015, 2016, 2018, 2020, 2021, 2022, 2023],
        "val": [2025],
        "test": [2019],          # спокойный, возмущённость 0.187
        "test_hard": [2024],     # активный,  возмущённость 0.275
    },
    # эксперимент на компонентах X, Y, Z. 2013, 2017, 2019 и второе полугодие
    # 2016 выпали: базисы вариометра не приложены либо неверны, векторный
    # модуль расходится со скаляром на 80-56000 нТл (см. bench_prep). Зато
    # тестовые годы дают вдвое больший контраст по возмущённости, чем в "f"
    # (0.152 против 0.297 вместо 0.187 против 0.275), а валидационный год
    # лежит между ними, а не выше обоих.
    "comp": {
        # 2013, 2017 и 2019 исключены даже из обучения. Центрирование окна
        # снимает ПОСТОЯННЫЙ сдвиг базиса целиком, но проверка показала, что
        # он там не постоянный: подгонка одного вектора b по всему году
        # оставляет 5-6 нТл невязки внутри 6-суточного окна против 0.2-0.4 нТл
        # в исправном году с независимым скаляром (CMO 2019-2021). Это медленно
        # плывущая ошибка в метках, сравнимая с суточным ходом и большая, чем
        # ошибка модели; она била бы по компонентной стороне и только по ней
        # 2026 не берём: он есть лишь до середины года, у станций покрытие
        # от 15 до 73 % и общих минут с ARS почти нет — выровнять нельзя,
        # а брать его только для ARS значило бы обучать станции на разном
        "train": [2014, 2015, 2016, 2018, 2020, 2022, 2025],
        "val": [2023],           # середина,   возмущённость 0.213
        "test": [2021],          # спокойный,  возмущённость 0.152
        "test_hard": [2024],     # бурный,     возмущённость 0.275
    },
}
SPLIT_NAME = os.environ.get("IGF_SPLIT", "f")
if SPLIT_NAME not in SPLITS:
    raise SystemExit(f"IGF_SPLIT={SPLIT_NAME}: есть только {', '.join(SPLITS)}")
SPLIT = SPLITS[SPLIT_NAME]


def load_year(year, code=CODE):
    p = os.path.join(DATA, f"{code}_{year}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"нет {p} — сначала: python -u bench_prep.py")
    return np.load(p, allow_pickle=False)["F"].astype(np.float32)


def load_split(name, code=CODE):
    return {y: load_year(y, code=code) for y in SPLIT[name]}


# Базисы вариометра. Если они не приложены (ARS 2019) или неверны (ARS 2013,
# CMO 2025), векторный модуль расходится со скаляром — от десятков нТл до
# 56000. Но это СДВИГ УРОВНЯ, а не порча вариаций: суточный размах H в ARS
# 2019 равен 40.3 нТл при 38.7 в 2018 и 36.7 в 2020, а дрейф уровня внутри
# 6-суточного окна там 2.72 нТл — меньше, чем в любом благополучном году.
#
# featurize центрирует каждое окно по покомпонентной медиане, поэтому
# постоянный сдвиг из входа и цели уходит ПОЛНОСТЬЮ (проверено: X, маска и
# цель совпадают побитово). Значит для ОБУЧЕНИЯ такие годы годны.
#
# А вот на валидации и тестах из компонент собирается обратно модуль и
# сравнивается с истиной — там сдвиг войдёт в ответ целиком, и год негоден.
def comp_ok(year, code=CODE, tol=TOL_COMP):
    p = os.path.join(DATA, f"{code}_{year}.npz")
    if not os.path.exists(p):
        return False
    with np.load(p, allow_pickle=False) as d:
        F = d["F"].astype(np.float64)
        v = np.sqrt(sum(d[k].astype(np.float64) ** 2 for k in "XYZ"))
    ok = np.isfinite(F) & np.isfinite(v)
    return bool(ok.any() and np.nanmedian(np.abs(F[ok] - v[ok])) <= tol)


def load_year_comp(year, code=CODE):
    """(n, 3) — X, Y, Z одного года."""
    p = os.path.join(DATA, f"{code}_{year}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"нет {p} — сначала: python -u bench_prep.py")
    with np.load(p, allow_pickle=False) as d:
        return np.stack([d[k] for k in "XYZ"], axis=1).astype(np.float32)


def load_split_comp(name, code=CODE, tol=TOL_COMP):
    """Компоненты по годам набора. На обучении сдвиг базиса допустим (его
    снимает центрирование), на валидации и тестах — нет: там из компонент
    собирается модуль и сравнивается с истиной."""
    strict = name != "train"
    out, skip = {}, []
    for y in SPLIT[name]:
        if strict and not comp_ok(y, code, tol):
            skip.append(y)
            continue
        out[y] = load_year_comp(y, code)
    if skip:
        print(f"компоненты: пропущены годы {', '.join(map(str, skip))} ({name}) — "
              f"модуль вектора расходится со скаляром больше {tol} нТл, а в этой роли "
              "сдвиг базиса вошёл бы в ответ")
    if not out:
        raise SystemExit(f"в наборе {name} нет годов с пригодными компонентами")
    return out


def scale_comp(years, n=1500, seed=SEED):
    """Масштаб по каждой компоненте отдельно: у X, Y, Z разный размах."""
    rng = np.random.default_rng(seed)
    arrs = [A for A in years.values() if A.shape[0] > W]
    acc = [[] for _ in range(3)]
    for _ in range(n):
        A = arrs[rng.integers(len(arrs))]
        st = int(rng.integers(0, A.shape[0] - W))
        w = A[st:st + W]
        for c in range(3):
            v = w[:, c][np.isfinite(w[:, c])]
            if v.size:
                acc[c].append(np.median(np.abs(v - np.median(v))) * 1.4826)
    return np.array([max(float(np.median(a)), 1e-3) if a else 1.0 for a in acc],
                    dtype=np.float32)


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


def gather_aligned(codes_years, gap_len, n, rng, max_real=MAX_REAL):
    """То же самое, что gather(), но СРАЗУ для нескольких станций, с
    гарантией побитового совпадения (год, начало окна, позиция дыры) между
    ними — а не совпадения "как получится".

    Обычный gather() расходится между станциями: отбраковка (dырой
    попавшей на реальный пропуск, или mask_real выше порога) зависит от
    паттерна пропусков КОНКРЕТНОЙ станции, и как только одна станция
    отбраковывает кандидата, а другая — нет, последовательность ГСЧ
    сдвигается, и все дальнейшие окна расходятся (проверено эмпирически:
    93-94% совпадения на большинстве длин, 3% на 4320 мин). Здесь позиция
    дыры выбирается один раз из ПЕРЕСЕЧЕНИЯ допустимых кандидатов по всем
    станциям сразу, и окно принимается только если проходит порог
    max_real НА КАЖДОЙ станции — гарантия совпадения по построению, а не
    по случаю.

    codes_years: {код станции: {год: F}}; набор годов должен совпадать
    у всех станций (иначе ValueError — иное сравнение бессмысленно).
    Возвращает {код: [sample, ...]}, списки одной длины n; year/start/pos
    одинаковы у всех кодов, input/target/mask_real — свои для каждой
    станции (свои значения поля)."""
    codes = list(codes_years)
    years_sets = {c: set(codes_years[c]) for c in codes}
    for c in codes[1:]:
        if years_sets[c] != years_sets[codes[0]]:
            raise ValueError(f"разные годы у станций {codes[0]} и {c}: "
                             f"{sorted(years_sets[codes[0]])} vs {sorted(years_sets[c])}")
    arrs = {c: {y: F for y, F in codes_years[c].items() if F.size > W}
            for c in codes}
    years_ok = [y for y in years_sets[codes[0]] if all(y in arrs[c] for c in codes)]
    out = {c: [] for c in codes}
    if not years_ok:
        return out
    tries = n * 300
    while len(out[codes[0]]) < n and tries > 0:
        tries -= 1
        y = years_ok[rng.integers(len(years_ok))]
        size = min(arrs[c][y].size for c in codes)
        start = int(rng.integers(0, size - W))
        reals, cand_common = {}, None
        for c in codes:
            win = arrs[c][y][start:start + W]
            real = ~np.isfinite(win)
            reals[c] = real
            cc = set(candidate_starts(real, gap_len).tolist())
            cand_common = cc if cand_common is None else (cand_common & cc)
        if not cand_common:
            continue
        s = int(rng.choice(sorted(cand_common)))
        samples, ok = {}, True
        for c in codes:
            real = reals[c]
            if real.mean() >= max_real:
                ok = False
                break
            win = arrs[c][y][start:start + W].astype(np.float32).copy()
            art = np.zeros(W, dtype=bool)
            art[s:s + gap_len] = True
            inp = win.copy()
            inp[art] = np.nan
            samples[c] = dict(input=inp, target=win, mask_art=art,
                              mask_real=real, pos=(s, gap_len),
                              start=start, year=y)
        if not ok:
            continue
        for c in codes:
            out[c].append(samples[c])
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
