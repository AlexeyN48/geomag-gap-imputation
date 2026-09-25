r"""Обучаемые модели бенчмарка и общее представление входа.

Представление одно на всех: окно 8640 мин режется на участки по P=5 минут
(1728 токенов)

признаки участка = P значений + sin/cos суток + sin/cos года(NF=9), маска той же формы. 

Выход модели — P значений на участок, то есть та же минутная сетка. 

Модели: dlinear (линейная), unet (свёрточная), segrnn (рекуррентная).
"""
import numpy as np
import torch
import torch.nn as nn

import bench_common as C

P = 5                                  # минут в участке
NCH = 1                                # каналов на минуту: 1 = F, 3 = X, Y, Z
WB = C.W // P                          # 1728 токенов
NF = P * NCH + 4                       # значения участка + 4 гармоники времени
POUT = P * NCH                         # ширина выхода на токен
DAY, YEAR = 1440.0, 525960.0


def _resize():
    global WB, NF, POUT
    WB = C.W // P; NF = P * NCH + 4; POUT = P * NCH


def set_patch(p):
    """Переключить размер патча: пересчитывает число токенов и ширину входа.
    Вызывать ДО построения модели и до featurize — все ссылаются на эти
    глобалы во время исполнения."""
    global P
    P = int(p); _resize()


def set_channels(n):
    """1 — работаем с модулем F, 3 — с компонентами X, Y, Z. Как и set_patch,
    вызывать ДО построения модели: ширина входа и выхода меняется."""
    global NCH
    NCH = int(n); _resize()


# ------------------------------------------------------------ представление
def time_feats(start, n=None):
    """Гармоники суток и года в серединах участков. n читается во время вызова:
    значение по умолчанию связалось бы с WB на момент импорта и не увидело бы
    set_patch (был такой баг)."""
    if n is None:
        n = WB
    t = start + np.arange(n) * P + P / 2.0
    return np.stack([np.sin(2 * np.pi * t / DAY), np.cos(2 * np.pi * t / DAY),
                     np.sin(2 * np.pi * t / YEAR), np.cos(2 * np.pi * t / YEAR)],
                    axis=1).astype(np.float32)


def featurize(inp, start, scale):
    """Окно значений -> (X, mask, center). Центрируем по медиане ВИДИМЫХ точек:
    базовый уровень поля плывёт по годам на сотни нТл, без центрирования модель
    учила бы год, а не форму. Вход — (W,) для F либо (W, NCH) для компонент;
    center и scale тогда покомпонентные: у X, Y, Z и уровни, и размах разные."""
    a = inp if inp.ndim == 2 else inp[:, None]
    sc = np.broadcast_to(np.asarray(scale, np.float32).reshape(-1), (a.shape[1],))
    obs = np.isfinite(a)
    center = np.array([float(np.median(a[obs[:, c], c])) if obs[:, c].any() else 0.0
                       for c in range(a.shape[1])], np.float32)
    v = np.where(obs, (a - center) / sc, 0.0).astype(np.float32)
    X = np.concatenate([v.reshape(WB, P * a.shape[1]), time_feats(start)], axis=1)
    M = np.concatenate([obs.reshape(WB, P * a.shape[1]).astype(np.float32),
                        np.ones((WB, 4), np.float32)], axis=1)
    return X, M, (center if inp.ndim == 2 else float(center[0]))


# ------------------------------------------------------------ модели
class DLinear(nn.Module):
    """Разложение на тренд и остаток + линейное отображение по времени."""

    def __init__(self, kernel=25, **kw):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(WB, WB)
        self.season = nn.Linear(WB, WB)
        self.proj = nn.Linear(NF, POUT)

    def forward(self, X, M):
        x = torch.cat([X, M[:, :, :P]], dim=2)[:, :, :NF]      # форма [B,WB,NF]
        xt = x.transpose(1, 2)                                  # [B,NF,WB]
        pad = self.kernel // 2
        trend = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(xt, (pad, pad), mode="replicate"),
            self.kernel, stride=1)
        season = xt - trend
        out = self.trend(trend) + self.season(season)           # [B,NF,WB]
        return self.proj(out.transpose(1, 2))                   # [B,WB,P]


class UNet1D(nn.Module):
    """Свёрточный энкодер-декодер по оси участков."""

    def __init__(self, base=32, depth=4, **kw):
        super().__init__()
        self.depth = depth
        ch = [NF * 2] + [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([self._blk(ch[i], ch[i + 1]) for i in range(depth)])
        self.mid = self._blk(ch[depth], ch[depth + 1])
        self.up = nn.ModuleList([nn.ConvTranspose1d(ch[depth + 1 - i], ch[depth - i],
                                                    2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([self._blk(ch[depth - i] * 2, ch[depth - i])
                                  for i in range(depth)])
        self.out = nn.Conv1d(ch[1], POUT, 1)

    @staticmethod
    def _blk(a, b):
        return nn.Sequential(nn.Conv1d(a, b, 5, padding=2), nn.GroupNorm(8, b),
                             nn.GELU(),
                             nn.Conv1d(b, b, 5, padding=2), nn.GroupNorm(8, b),
                             nn.GELU())

    def forward(self, X, M):
        h = torch.cat([X, M], dim=2).transpose(1, 2)            # [B,2NF,WB]
        skips = []
        for e in self.enc:
            h = e(h)
            skips.append(h)
            h = torch.nn.functional.max_pool1d(h, 2)
        h = self.mid(h)
        for i, (u, d) in enumerate(zip(self.up, self.dec)):
            h = u(h)
            h = d(torch.cat([h, skips[-1 - i]], dim=1))
        return self.out(h).transpose(1, 2)                      # [B,WB,P]


class SegRNN(nn.Module):
    """Сегментная рекуррентная сеть."""

    def __init__(self, seg_len=48, d_model=128, **kw):
        super().__init__()
        from pypots.nn.modules.segrnn import BackboneSegRNN
        self.body = BackboneSegRNN(WB, NF, n_pred_steps=WB,
                                   seg_len=seg_len, d_model=d_model, dropout=0.1)
        self.proj = nn.Linear(NF, POUT)

    def forward(self, X, M):
        x = torch.where(M[:, :, :NF] > 0, X, torch.zeros_like(X))
        out = self.body(x)
        if isinstance(out, tuple):
            out = out[0]
        return self.proj(out)                                   # [B,WB,P]


class Saits(nn.Module):
    """Самовнимание, спроектированное под импутацию.

    Глубина 2 выбрана замером: при 4 слоях внимание на 1728
    токенах требует 11.7 ГиБ против 8 доступных, карта уходит в подкачку и шаг
    дорожает с 0.13 до 2.34 с — сравнение при равном бюджете стало бы
    невозможным."""

    def __init__(self, n_layers=2, d_model=128, n_heads=4, d_ffn=256, **kw):
        super().__init__()
        from pypots.nn.modules.saits import BackboneSAITS
        self.body = BackboneSAITS(WB, NF, n_layers, d_model, n_heads,
                                  d_model // n_heads, d_model // n_heads,
                                  d_ffn, 0.1, 0.1)
        self.proj = nn.Linear(NF, POUT)

    def forward(self, X, M):
        out = self.body(X, M[:, :, :NF])
        y = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else out
        if isinstance(y, (tuple, list)):
            y = y[0]
        return self.proj(y)                                     # [B,WB,P]


class TimesNet(nn.Module):
    """Явная работа с периодом: ряд сворачивается в двумерную форму по найденным
    периодам, поэтому суточный ход обрабатывается напрямую, а не как дальний лаг.
    Прямая проверка главной физической гипотезы задачи."""

    def __init__(self, n_layers=2, d_model=64, d_ffn=128, top_k=3, n_kernels=4, **kw):
        super().__init__()
        from pypots.nn.modules.timesnet import BackboneTimesNet
        self.emb = nn.Linear(NF * 2, d_model)
        self.body = BackboneTimesNet(n_layers, WB, 0, top_k, d_model, d_ffn, n_kernels)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, POUT)

    def forward(self, X, M):
        h = self.emb(torch.cat([X, M], dim=2))
        h = self.body(h)
        if isinstance(h, (tuple, list)):
            h = h[0]
        return self.proj(self.norm(h))                          # [B,WB,P]




class ImputeFormer(nn.Module):
    """Проецированное внимание: длинная ось сжимается в узкое место из dim_proj
    векторов, поэтому стоимость линейна по числу токенов, а не квадратична."""

    def __init__(self, n_layers=3, d_model=64, n_heads=4, dim_proj=32,
                 d_ffn=128, node_dim=16, **kw):
        super().__init__()
        from pypots.nn.modules.imputeformer import (ProjectedAttentionLayer,
                                                    EmbeddedAttentionLayer)
        self.emb = nn.Linear(2, d_model)                 # (значение, маска) -> d
        self.temporal = nn.ModuleList([
            ProjectedAttentionLayer(NF, dim_proj, d_model, n_heads, d_ffn, 0.1)
            for _ in range(n_layers)])
        self.spatial = nn.ModuleList([
            EmbeddedAttentionLayer(d_model, node_dim, d_ffn, 0.1)
            for _ in range(n_layers)])
        self.node_emb = nn.Parameter(torch.randn(WB, NF, node_dim) * 0.02)  # слой ждёт
                                                                     # эмбеддинг на каждый токен
        self.head = nn.Linear(NF * d_model, POUT)

    def forward(self, X, M):
        h = torch.stack([X, M[:, :, :NF]], dim=-1)       # [B,WB,NF,2]
        h = self.emb(h).permute(0, 2, 1, 3)              # [B,NF,WB,d]
        for tl, sl in zip(self.temporal, self.spatial):
            h = tl(h)                                    # внимание по времени
            h = sl(h, self.node_emb, dim=1)              # внимание по признакам
        h = h.permute(0, 2, 1, 3).flatten(2)             # [B,WB,NF*d]
        return self.head(h)                              # [B,WB,P]


class Crossformer(nn.Module):
    """Двухстадийное внимание: по времени И по латентным измерениям.

    Ряд режется на сегменты по seg_len, поэтому внимание идёт по 72 сегментам,
    а не по 1728 отсчётам — без сегментации конфигурация требует 13.3 ГиБ при
    8 доступных (замерено)."""

    def __init__(self, n_layers=3, d_model=256, n_heads=4, d_ffn=256,
                 factor=10, seg_len=24, win_size=2, **kw):
        super().__init__()
        from math import ceil
        from einops import rearrange
        from pypots.nn.modules.crossformer import CrossformerEncoder, ScaleBlock
        from pypots.nn.modules.patchtst import PatchEmbedding
        from pypots.nn.modules.saits import SaitsEmbedding
        self.rearrange = rearrange
        self.d_model = d_model
        pad = ceil(WB / seg_len) * seg_len
        seg_num = pad // seg_len
        self.emb = SaitsEmbedding(NF * 2, d_model, with_pos=False)
        self.patch = PatchEmbedding(d_model, seg_len, seg_len, pad - WB, 0)
        self.pos = nn.Parameter(torch.randn(1, d_model, seg_num, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.enc = CrossformerEncoder([
            ScaleBlock(1 if l == 0 else win_size, d_model, n_heads, d_ffn, 1, 0.1,
                       seg_num if l == 0 else ceil(seg_num / win_size ** l), factor)
            for l in range(n_layers)])
        self.out_seg = ceil(seg_num / (win_size ** (n_layers - 1)))
        self.head = nn.Sequential(nn.Flatten(start_dim=-2),
                                  nn.Linear(self.out_seg * d_model, WB))
        self.proj = nn.Linear(d_model, POUT)

    def forward(self, X, M):
        h = self.emb(X, M[:, :, :NF])
        h = self.patch(h.permute(0, 2, 1))
        h = h[0] if isinstance(h, tuple) else h
        h = self.rearrange(h, "(b d) s dm -> b d s dm", d=self.d_model)
        h = self.norm(h + self.pos)
        out, _ = self.enc(h)
        h = out[-1] if isinstance(out, list) else out    # [B,d_model,seg,dm]
        h = self.head(h.permute(0, 1, 3, 2))             # [B,d_model,WB]
        return self.proj(h.permute(0, 2, 1))             # [B,WB,P]



class CSDI(nn.Module):
    """Диффузионная модель: восстановление сэмплированием из распределения.

    Единственное семейство в наборе, дающее интервалы неопределённости
    напрямую, а не conformal-надстройкой. Плата двойная — обучение дороже и
    ВЫВОД дороже: 0.9 с на окно при одном сэмпле (замерено), поэтому
    валидационный набор здесь уменьшен.

    Работает по P значащим каналам; гармоники времени уходят в side_info, куда
    их и кладёт исходная реализация."""

    VAL_N = 24                       # окон на валидации: вывод дорог

    def __init__(self, n_layers=2, n_heads=4, n_channels=64, d_feat_emb=16,
                 n_diff_steps=50, n_samples=1, **kw):
        super().__init__()
        from pypots.nn.modules.csdi import BackboneCSDI
        self.K, self.L = P, WB
        self.n_samples = n_samples
        self.d_time = 4                                   # sin/cos суток и года
        self.feat_emb = nn.Parameter(torch.randn(P, d_feat_emb) * 0.02)
        self.body = BackboneCSDI(n_layers, n_heads, n_channels, P,
                                 self.d_time, d_feat_emb, n_channels,
                                 False, n_diff_steps, "quad", 1e-4, 0.5)

    def _side(self, X, cond):
        """side_info [B, d_time + d_feat + 1, K, L]: время, признак, маска."""
        B = X.shape[0]
        t = X[:, :, P:P + 4].permute(0, 2, 1)                    # [B,4,L]
        t = t.unsqueeze(2).expand(-1, -1, self.K, -1)            # [B,4,K,L]
        f = self.feat_emb.t()[None, :, :, None].expand(B, -1, -1, self.L)
        return torch.cat([t, f, cond.unsqueeze(1)], dim=1)

    def custom_loss(self, X, M, Y, A):
        """Обучение идёт родным лоссом диффузии, а не общим masked MAE: у неё
        цель — предсказать шум, и подменять её нельзя."""
        obs = Y.permute(0, 2, 1)                                 # [B,K,L]
        cond = M[:, :, :P].permute(0, 2, 1)
        ind = A.permute(0, 2, 1)
        return self.body.calc_loss(obs, cond, ind, self._side(X, cond))

    @torch.no_grad()
    def forward(self, X, M):
        cond = M[:, :, :P].permute(0, 2, 1)
        obs = X[:, :, :P].permute(0, 2, 1) * cond
        out = self.body(obs, cond, self._side(X, cond), self.n_samples)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 4:                                       # [B,S,K,L]
            out = out.median(dim=1).values
        # backbone отдаёт СЫРЫЕ сэмплы; ответ собирается как наблюдения там, где
        # они есть, и сэмпл в пропусках — так же, как в исходной реализации
        out = obs * cond + out * (1.0 - cond)
        return out.permute(0, 2, 1)                              # [B,WB,P]

def _stack_forward(blocks, y, mask):
    """Общий цикл backcast/forecast NHITS и N-BEATS (взят из forward() обеих
    архитектур в neuralforecast дословно): реконструкция уточняется по
    остатку блок за блоком, остаток обнуляется маской там, где входа не было.
    y, mask — [Bc, WB] (Bc = B*NF: канал/патч-признак развёрнут в батч, веса
    блока общие на всех каналах — тот же приём, что у DLinear.trend/season)."""
    resid = y.flip(dims=(-1,))
    mask_f = mask.flip(dims=(-1,))
    forecast = y[:, -1:, None].repeat(1, y.shape[1], 1)
    zero = torch.zeros(y.shape[0], 0, device=y.device)
    for blk in blocks:
        backcast, block_fc = blk(resid, zero, zero, zero)
        resid = (resid - backcast) * mask_f
        forecast = forecast + block_fc
    return forecast[..., 0]


class NHITS(nn.Module):
    """Иерархическая интерполяция (блоки NHITSBlock из neuralforecast): несколько
    MLP-блоков смотрят на один и тот же ряд с разным шагом пулинга (весь
    участок сразу -> редко, почти без пулинга -> часто) и достраивают
    недостающее кусочно-линейной интерполяцией по немногим узлам. Третий,
    независимый способ проверить многомасштабность — рядом со свёрткой
    (U-Net) и явным периодом (TimesNet).

    Канал (P значений участка + 4 гармоники времени) идёт как отдельная
    "серия" с общими весами блока, как во всех остальных обёртках здесь."""

    def __init__(self, n_pool=(8, 4, 1), mlp=128, dropout=0.0, **kw):
        super().__init__()
        from neuralforecast.models.nhits import NHITSBlock, _IdentityBasis
        blocks = []
        for k in n_pool:
            basis = _IdentityBasis(backcast_size=WB, forecast_size=WB,
                                   interpolation_mode="linear", out_features=1)
            n_theta = WB + max(WB // k, 1)
            blocks.append(NHITSBlock(
                input_size=WB, h=WB, n_theta=n_theta, mlp_units=[[mlp, mlp]],
                basis=basis, futr_input_size=0, hist_input_size=0,
                stat_input_size=0, n_pool_kernel_size=k, pooling_mode="MaxPool1d",
                dropout_prob=dropout, activation="ReLU"))
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Linear(NF, POUT)

    def forward(self, X, M):
        x = torch.cat([X, M[:, :, :P]], dim=2)[:, :, :NF]      # форма [B,WB,NF]
        B = x.shape[0]
        y = x.transpose(1, 2).reshape(B * NF, WB)               # канал -> батч
        m = M[:, :, :NF].transpose(1, 2).reshape(B * NF, WB)
        out = _stack_forward(self.blocks, y, m).reshape(B, NF, WB).transpose(1, 2)
        return self.proj(out)                                    # [B,WB,P]


class NBEATSx(nn.Module):
    """Интерпретируемый N-BEATS: тренд (низкая степень полинома) + сезонность
    (несколько гармоник суточной и годовой частоты) — явный компактный базис,
    независимый от свёртки (U-Net) и от пулинга (NHITS) способ проверить
    главную гипотезу задачи про суточный ход, да ещё и с разложением,
    которое можно прочитать напрямую (вклад тренда отдельно от сезонности).

    Базис свой, а не готовый TrendBasis/SeasonalityBasis из neuralforecast:
    там число гармоник в SeasonalityBasis растёт вместе с forecast_size и не
    зависит от параметра harmonics — при типичном для библиотеки горизонте
    в десятки шагов это компактно, а на нашем участке в WB=1728 токенов
    посчитанный по их формуле базис выходит порядка 1728 гармоник, то есть
    вырождается почти в тождественную интерполяцию и ничего не даёт против
    NHITS выше. Здесь число гармоник задаётся явно и не зависит от WB."""

    def __init__(self, degree=3, n_harm_day=4, n_harm_year=2, mlp=256,
                dropout=0.0, **kw):
        super().__init__()
        t = torch.arange(WB, dtype=torch.float32) / WB               # [0,1)
        basis = [t ** i for i in range(degree + 1)]                   # тренд
        day_k = WB * P / DAY                                          # циклов суток в участке
        year_k = WB * P / YEAR
        for k in range(1, n_harm_day + 1):
            basis += [torch.sin(2 * np.pi * k * day_k * t),
                     torch.cos(2 * np.pi * k * day_k * t)]
        for k in range(1, n_harm_year + 1):
            basis += [torch.sin(2 * np.pi * k * year_k * t),
                     torch.cos(2 * np.pi * k * year_k * t)]
        self.register_buffer("basis", torch.stack(basis))             # [Q,WB]
        Q = self.basis.shape[0]
        self.mlp = nn.Sequential(nn.Linear(WB, mlp), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(mlp, mlp), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(mlp, Q))
        self.proj = nn.Linear(NF, POUT)

    def forward(self, X, M):
        x = torch.cat([X, M[:, :, :P]], dim=2)[:, :, :NF]
        B = x.shape[0]
        y = x.transpose(1, 2).reshape(B * NF, WB)                      # канал -> батч
        theta = self.mlp(y)                                             # [B*NF,Q]
        rec = theta @ self.basis                                        # [B*NF,WB]
        out = rec.reshape(B, NF, WB).transpose(1, 2)
        return self.proj(out)                                           # [B,WB,P]


class TSMixerx(nn.Module):
    """Смешивание по времени и по признакам MLP-слоями (MixingLayer из
    neuralforecast), без внимания и без свёртки: прямое заострение вывода
    про DLinear — если нелинейность не нужна, у нелинейного аналога той же
    структуры («смешивание по времени») не должно быть преимущества при
    равном бюджете. MixingLayer уже устроен над формой [B, время, признаки]
    — ровно наше представление, без адаптации."""

    def __init__(self, n_layers=4, ff_dim=64, dropout=0.1, **kw):
        super().__init__()
        from neuralforecast.models.tsmixerx import MixingLayer
        C = NF * 2                                              # X и M вместе
        self.layers = nn.ModuleList([
            MixingLayer(in_features=C, out_features=C, h=WB,
                       dropout=dropout, ff_dim=ff_dim)
            for _ in range(n_layers)])
        self.proj = nn.Linear(C, POUT)

    def forward(self, X, M):
        h = torch.cat([X, M], dim=2)                             # [B,WB,2NF]
        for layer in self.layers:
            h = layer(h)
        return self.proj(h)                                       # [B,WB,P]


class TiDE(nn.Module):
    """Плотный энкодер-декодер (MLPResidual-блоки из neuralforecast): всё окно
    сжимается целиком в один вектор, решение разворачивается MLP-декодером
    обратно на WB токенов, а гармоники времени подмешиваются в решение
    ОТДЕЛЬНО на каждом токене через temporal-декодер. Единственная модель
    набора, построенная вокруг ковариат, а не вокруг формы сигнала —
    прямая проверка, даёт ли им что-то отдельный путь для sin/cos, которые
    у всех остальных моделей идут наравне со значениями участка.

    Глобальный skip оригинального TiDE — Linear по всей минутной сетке
    (WB*P на WB*P, ~75М параметров на нашем окне) — заменён на потоковый,
    как self.proj у всех остальных моделей здесь: иначе одна эта связь
    вынесла бы модель на порядок за пределы бюджета параметров остальных."""

    def __init__(self, hidden=256, temporal_width=8, n_enc=2, n_dec=2,
                temporal_decoder_dim=64, dropout=0.1, **kw):
        super().__init__()
        from neuralforecast.models.tide import MLPResidual
        self.temporal_width = temporal_width
        enc_in = NF * WB * 2                                     # X и M плоско по всему окну
        self.encoder = nn.Sequential(*[
            MLPResidual(enc_in if i == 0 else hidden, hidden, hidden,
                       dropout, layernorm=True)
            for i in range(n_enc)])
        dec_out = temporal_width * WB
        self.decoder = nn.Sequential(*[
            MLPResidual(hidden, hidden,
                       hidden if i < n_dec - 1 else dec_out,
                       dropout, layernorm=True)
            for i in range(n_dec)])
        self.cov_proj = MLPResidual(4, hidden, temporal_width, dropout, layernorm=True)
        self.temporal = MLPResidual(temporal_width * 2, temporal_decoder_dim, POUT,
                                    dropout, layernorm=True)
        self.skip = nn.Linear(NF, POUT)

    def forward(self, X, M):
        B = X.shape[0]
        flat = torch.cat([X, M], dim=2).reshape(B, -1)            # [B,NF*WB*2]
        h = self.decoder(self.encoder(flat)).reshape(B, WB, self.temporal_width)
        cov = self.cov_proj(X[:, :, P:P + 4])                      # [B,WB,tw]
        out = self.temporal(torch.cat([h, cov], dim=2))            # [B,WB,P]
        return out + self.skip(X)                                   # [B,WB,P]


ARCH = {"dlinear": DLinear, "unet": UNet1D, "segrnn": SegRNN,
        "saits": Saits, "timesnet": TimesNet,
        "imputeformer": ImputeFormer, "crossformer": Crossformer,
        "csdi": CSDI,
        "nhits": NHITS, "nbeatsx": NBEATSx,
        "tsmixerx": TSMixerx, "tide": TiDE}


def build(name, **kw):
    if name not in ARCH:
        raise SystemExit(f"нет модели {name}; есть: {', '.join(sorted(ARCH))}")
    return ARCH[name](**kw)


def save(path, net, name, scale, kw, step, val):
    # при NCH > 1 масштаб покомпонентный, скаляром его не записать
    scale = float(scale) if np.ndim(scale) == 0 else np.asarray(scale, np.float32)
    torch.save(dict(state=net.state_dict(), arch=name, scale=scale,
                    kw=kw, step=int(step), val=float(val), P=P, NCH=NCH, W=int(C.W)), path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if "W" in ck:
        C.W = int(ck["W"])          # окно оценки = окно обучения
    set_channels(int(ck.get("NCH", 1)))   # каналы оценки = каналы обучения
    set_patch(int(ck.get("P", 5)))        # патч оценки = патч обучения
    net = build(ck["arch"], **ck.get("kw", {}))
    net.load_state_dict(ck["state"])
    net.to(device).eval()
    return net, ck


@torch.no_grad()
def fill(net, inp, start, scale, device="cpu"):
    """Восстановление одного окна: только пропущенные точки заменяются, видимые
    остаются как есть — иначе метод «портил» бы наблюдения."""
    X, M, center = featurize(inp, start, scale)
    xb = torch.from_numpy(X[None]).to(device)
    mb = torch.from_numpy(M[None]).to(device)
    out = net(xb, mb)[0].cpu().numpy().reshape(-1)[:C.W * NCH]
    if inp.ndim == 2:
        pred = out.reshape(C.W, NCH) * np.asarray(scale, np.float64).reshape(1, -1) + center
    else:
        pred = out.reshape(C.W) * scale + center
    return np.where(np.isfinite(inp), inp, pred).astype(np.float64)
