r"""Архив данных бенчмарка: упаковать у автора, скачать и сверить у того, кто
запускает.

Зачем. Годовые ряды data/<код>_<год>.npz в git не хранятся, а пересобрать их
из INTERMAGNET побитово нельзя: недостающие сутки докачиваются из ГИН в
режиме «лучшее доступное», и со временем там предварительные данные
заменяются окончательными. Поэтому данные передаются архивом ровно в том
состоянии, на котором посчитаны результаты, а data/DATA_MANIFEST.json (он в
git) подтверждает, что скачалось именно оно.

    python -u data_archive.py pack                          автор: собрать архив
    python -u data_archive.py fetch --url <ссылка>          скачать, распаковать, сверить
    python -u data_archive.py fetch --file geomag_data.zip  то же из уже скачанного файла
    python -u data_archive.py check                         только сверка с манифестом

Ссылка — прямая на файл либо публичная ссылка Яндекс.Диска (disk.yandex.ru/d/…
или yadi.sk/d/…): для неё прямой адрес берётся через открытый API Диска. Для
Google Drive нужен пакет gdown (pip install gdown). Если на вычислительных
узлах нет интернета, скачайте архив на узле входа и передайте его через
--file.
"""
import os
import sys
import json
import shutil
import zipfile
import hashlib
import argparse
import urllib.parse
import urllib.request

# сверка с манифестом идёт через bench_data, а тот через bench_common требует
# IGF_SPLIT; от сплита данные не зависят, поэтому значение подставляется
os.environ.setdefault("IGF_SPLIT", "comp")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
MANIFEST = os.path.join(DATA, "DATA_MANIFEST.json")
DEFAULT_ARCHIVE = os.path.join(DATA, "geomag_data.zip")


def manifest_files():
    with open(MANIFEST, encoding="utf-8") as f:
        return sorted(json.load(f)["files"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check():
    import bench_data
    return bench_data.check()


def pack(out):
    """Ровно файлы манифеста — ни резервных копий шагов сборки, ни дампов."""
    names = manifest_files()
    missing = [n for n in names if not os.path.exists(os.path.join(DATA, n))]
    if missing:
        raise SystemExit(f"нет файлов из манифеста: {', '.join(missing[:5])}")
    if check():
        raise SystemExit("данные на диске расходятся с манифестом — архив не собран")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in names:
            z.write(os.path.join(DATA, n), n)
        z.write(MANIFEST, os.path.basename(MANIFEST))
    print(f"\nархив: {out}  ({os.path.getsize(out) / 1e6:.0f} МБ, {len(names)} файлов)")
    print(f"sha256: {sha256(out)}")
    print("Выложите архив в облако и передайте ссылку; дальше — "
          "python -u data_archive.py fetch --url <ссылка>")


def direct_url(url):
    """Публичная ссылка Яндекс.Диска -> прямой адрес файла через API Диска."""
    host = urllib.parse.urlparse(url).netloc
    if host.endswith("disk.yandex.ru") or host.endswith("yadi.sk") \
            or host.endswith("disk.yandex.com"):
        api = ("https://cloud-api.yandex.net/v1/disk/public/resources/download?"
               + urllib.parse.urlencode({"public_key": url}))
        with urllib.request.urlopen(api, timeout=60) as r:
            return json.load(r)["href"]
    return url


def download(url, out):
    host = urllib.parse.urlparse(url).netloc
    if "drive.google.com" in host:
        try:
            import gdown
        except ImportError:
            raise SystemExit("для ссылки Google Drive нужен gdown: pip install gdown")
        gdown.download(url, out, quiet=False, fuzzy=True)
        return
    href = direct_url(url)
    tmp = out + ".part"
    with urllib.request.urlopen(href, timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: r.read(1 << 22), b""):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} МБ", end="", flush=True)
    print()
    os.replace(tmp, out)


def unpack(archive):
    """Распаковка только годовых файлов из манифеста. Манифест из архива не
    подменяет тот, что в git: сверяемся с репозиторием, а не с архивом."""
    want = set(manifest_files())
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        lack = sorted(want - names)
        if lack:
            raise SystemExit(f"в архиве нет {len(lack)} файлов манифеста, "
                             f"например {lack[0]}")
        for n in sorted(want):
            with z.open(n) as src, open(os.path.join(DATA, n), "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"распаковано {len(want)} файлов в {DATA}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="собрать архив из файлов манифеста")
    p.add_argument("--out", default=DEFAULT_ARCHIVE)
    f = sub.add_parser("fetch", help="скачать (или взять файл), распаковать, сверить")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="ссылка на архив")
    g.add_argument("--file", help="уже скачанный архив")
    f.add_argument("--keep", action="store_true", help="не удалять скачанный архив")
    sub.add_parser("check", help="только сверить data/ с манифестом")
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    if a.cmd == "pack":
        pack(a.out)
    elif a.cmd == "check":
        raise SystemExit(1 if check() else 0)
    else:
        archive = a.file
        if a.url:
            archive = DEFAULT_ARCHIVE
            print(f"скачивание {a.url}")
            download(a.url, archive)
        print(f"архив {archive}: sha256 {sha256(archive)}")
        unpack(archive)
        if a.url and not a.keep:
            os.remove(archive)
        bad = check()
        if bad:
            raise SystemExit("данные НЕ совпадают с манифестом — не запускайте "
                             "обучение, сообщите автору")
        print("готово: данные совпадают с манифестом побитово")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
