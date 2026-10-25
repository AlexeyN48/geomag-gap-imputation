r"""Скачать предобученные веса MOMENT в models/pretrained/ заранее.

Нужно, если на вычислительных узлах нет интернета: bench_models.Moment сначала
ищет веса в models/pretrained/MOMENT-1-<размер>/, и только если их там нет,
идёт в Hugging Face. Запускать на узле с интернетом:

    python -u scripts/get_weights.py                  # small, base, large (~2 ГБ)
    python -u scripts/get_weights.py --sizes base
"""
import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = {"small": "AutonLab/MOMENT-1-small", "base": "AutonLab/MOMENT-1-base",
        "large": "AutonLab/MOMENT-1-large"}
DST = os.path.abspath(os.path.join(HERE, "..", "models", "pretrained"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="+", default=list(REPO), choices=list(REPO))
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    for s in a.sizes:
        out = os.path.join(DST, REPO[s].split("/")[1])
        for fn in ("config.json", "model.safetensors"):
            p = hf_hub_download(REPO[s], fn, local_dir=out)
            print(f"{REPO[s]}/{fn} -> {p}  ({os.path.getsize(p) / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
