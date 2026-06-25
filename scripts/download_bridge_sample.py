"""
下载 Bridge V2 TFDS 数据集的少量 shard, 用于开发 dataloader.

数据源: UC Berkeley RAIL Lab
    https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/

整体结构:
    bridge_dataset/
      1.0.0/
        dataset_info.json
        features.json
        bridge_dataset-train.tfrecord-00000-of-01024
        bridge_dataset-train.tfrecord-00001-of-01024
        ...
        bridge_dataset-val.tfrecord-00000-of-00128
        ...

用法:
    # 下 1 个 train shard
    uv run scripts/download_bridge_sample.py --output-dir $HOME/datasets --num-shards 1

    # 下 2 个 train + 1 个 val
    uv run scripts/download_bridge_sample.py --output-dir $HOME/datasets --num-shards 2 --num-val-shards 1

特性:
    - 自动列出远端目录, 不写死 shard 总数
    - 自动下载所有元数据 (json / 非 tfrecord 文件)
    - 支持断点续传 (HTTP Range)
    - 已下完的文件自动跳过
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

BASE_URL = (
    "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"
)

# 匹配 TFDS 分片命名: xxx.tfrecord-00000-of-01024
TFRECORD_RE = re.compile(r"\.tfrecord-(\d{5})-of-(\d{5})$")


def list_remote_files(url: str, timeout: int = 30) -> list[str]:
    """抓取目录的 HTML index, 用 regex 解析出所有 href, 返回文件名列表."""
    print(f"[list] {url}")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"]+)"', r.text)
    # 过滤掉 ../ ? sort param / 子目录等
    files = []
    for h in hrefs:
        if h.startswith("?") or h.startswith("/") or h.startswith(".."):
            continue
        if h.endswith("/"):
            continue
        files.append(h)
    return files


def http_size(url: str, timeout: int = 30) -> int:
    """HEAD 请求拿 Content-Length. 拿不到就返回 0."""
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """带断点续传的下载."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"

    existing = dest.stat().st_size if dest.exists() else 0
    total_remote = http_size(url)

    if existing > 0 and total_remote > 0 and existing >= total_remote:
        print(f"[skip] {dest.name}  ({existing:,} B 已完整)")
        return

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"[resume] {dest.name}  from byte {existing:,}")

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        # Range 请求时 Content-Length 是剩余字节数
        remain = int(r.headers.get("Content-Length", 0))
        total = total_remote if total_remote else (existing + remain)
        bar = tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        )
        with open(dest, mode) as f:
            for buf in r.iter_content(chunk_size=chunk):
                if not buf:
                    continue
                f.write(buf)
                bar.update(len(buf))
        bar.close()


def select_shards(files: list[str], split: str, n: int) -> list[str]:
    """从所有 tfrecord 文件里挑出指定 split 的前 n 个 shard."""
    pat = re.compile(rf"-{split}\.tfrecord-\d{{5}}-of-\d{{5}}$")
    matched = sorted(f for f in files if pat.search(f))
    return matched[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="数据根目录, 比如 $HOME/datasets, 实际文件会落在 <output-dir>/bridge_dataset/1.0.0/",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="train 分片数")
    parser.add_argument("--num-val-shards", type=int, default=0, help="val 分片数")
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help="远端 base URL (默认 RAIL Bridge V2)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser() / "bridge_dataset" / "1.0.0"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dest] {out_dir}")

    # 1) 列远端文件
    try:
        files = list_remote_files(args.base_url)
    except Exception as e:
        print(f"[error] failed to list directory: {e}", file=sys.stderr)
        sys.exit(1)

    tfrecords = [f for f in files if TFRECORD_RE.search(f)]
    others = [f for f in files if not TFRECORD_RE.search(f)]

    print(f"[info] 共 {len(files)} 个文件: {len(tfrecords)} tfrecord + {len(others)} 元数据")

    # 2) 下载所有非 tfrecord 元数据 (一般几 MB)
    print("\n=== 下载元数据 ===")
    for name in others:
        download_file(urljoin(args.base_url, name), out_dir / name)

    # 3) 下载 train shards
    train_shards = select_shards(tfrecords, "train", args.num_shards)
    print(f"\n=== 下载 train shards ({len(train_shards)}/{args.num_shards}) ===")
    for name in train_shards:
        download_file(urljoin(args.base_url, name), out_dir / name)

    # 4) 下载 val shards
    if args.num_val_shards > 0:
        val_shards = select_shards(tfrecords, "val", args.num_val_shards)
        print(f"\n=== 下载 val shards ({len(val_shards)}/{args.num_val_shards}) ===")
        for name in val_shards:
            download_file(urljoin(args.base_url, name), out_dir / name)

    # 5) 汇总
    print("\n=== 完成 ===")
    total_bytes = sum(p.stat().st_size for p in out_dir.glob("*") if p.is_file())
    print(f"[done] {out_dir}  总计 {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
