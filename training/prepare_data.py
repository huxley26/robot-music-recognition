# -*- coding: utf-8 -*-
"""按源录音生成多窗口音频切片及训练索引。运行时用 --root 指定数据目录。"""
import os
import argparse
import math
import glob
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ======== 路径配置 ========
ROOT = "data"
SONGS_DIR = os.path.join(ROOT, "songs")
NOISE_DIR = os.path.join(ROOT, "noises")
OUT_ROOT  = os.path.join(ROOT, "slices_multi")

# 5 首歌的子目录名（作为类别名使用）
SONG_CLASSES = ["renshi", "dream", "aoteman", "amigo", "wuren"]
NOISE_CLASS = "noises"

# ======== 切片方案定义 ========
# name: (segment_sec, hop_sec, mode)
# mode = "sliding" -> 滑窗； "firstN" -> 只取起始 N 秒
SCHEMES = {
    "seg1s":       {"segment_sec": 1.0,  "hop_sec": 1.0,  "mode": "sliding"},
    "seg3s":       {"segment_sec": 3.0,  "hop_sec": 3.0,  "mode": "sliding"},
    "first10s":    {"segment_sec": 10.0, "hop_sec": None, "mode": "firstN"},
    "seg5s_s2s":   {"segment_sec": 5.0,  "hop_sec": 2.0,  "mode": "sliding"},
    "seg50s_s5s":  {"segment_sec": 50.0, "hop_sec": 5.0,  "mode": "sliding"},
}

# ======== 工具函数 ========
def list_wavs():
    """返回 (filepath, class_name) 列表"""
    items = []
    # 五首歌（各自子目录）
    for cls in SONG_CLASSES:
        cls_dir = os.path.join(SONGS_DIR, cls)
        for fp in glob.glob(os.path.join(cls_dir, "*.wav")):
            items.append((fp, cls))
    # noises
    for fp in glob.glob(os.path.join(NOISE_DIR, "*.wav")):
        items.append((fp, NOISE_CLASS))
    return items

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def seconds_to_samples(sec, sr):
    return int(round(sec * sr))

def write_clip(y, sr, out_path):
    ensure_dir(os.path.dirname(out_path))
    sf.write(out_path, y, sr)

def slice_sliding(y, sr, seg_sec, hop_sec):
    seg_n = seconds_to_samples(seg_sec, sr)
    hop_n = seconds_to_samples(hop_sec, sr)
    total = y.shape[0]
    if total < seg_n:
        return []

    starts = list(range(0, total - seg_n + 1, hop_n))
    clips = []
    for st in starts:
        ed = st + seg_n
        clips.append((st, ed))
    return clips

def slice_firstN(y, sr, n_sec):
    seg_n = seconds_to_samples(n_sec, sr)
    if y.shape[0] < seg_n:
        return None
    return (0, seg_n)

def safe_basename(path):
    base = os.path.splitext(os.path.basename(path))[0]
    # 去掉可能影响文件名的奇怪字符
    for ch in [' ', ':', '/', '\\', '?', '*', '"', '<', '>', '|']:
        base = base.replace(ch, '_')
    return base

# ======== 主流程 ========
def main():
    global ROOT, SONGS_DIR, NOISE_DIR, OUT_ROOT
    parser = argparse.ArgumentParser(description="从五类歌曲和噪声录音生成切片")
    parser.add_argument("--root", default="data", help="含songs和noises的目录")
    args = parser.parse_args()
    ROOT = os.path.abspath(args.root)
    SONGS_DIR = os.path.join(ROOT, "songs")
    NOISE_DIR = os.path.join(ROOT, "noises")
    OUT_ROOT = os.path.join(ROOT, "slices_multi")
    files = list_wavs()
    if not files:
        print("未发现任何 .wav 文件，请检查目录结构。")
        return

    print(f"将处理 {len(files)} 个音频文件。输出根目录：{OUT_ROOT}\n")

    index_meta = []  # 记录每个输出切片的元信息（可选）

    for scheme_name, cfg in SCHEMES.items():
        print(f"==== 方案：{scheme_name} ====")
        for fp, cls in tqdm(files, ncols=80):
            try:
                y, sr = sf.read(fp, always_2d=False)  # 保持原采样率/声道
                # 保持 (samples, channels) 形状一致性
                if y.ndim == 1:
                    y = y.reshape(-1, 1)

                base = safe_basename(fp)
                out_dir = os.path.join(OUT_ROOT, scheme_name, cls)
                ensure_dir(out_dir)

                if cfg["mode"] == "firstN":
                    pair = slice_firstN(y, sr, cfg["segment_sec"])
                    if pair is None:
                        # 太短，跳过
                        continue
                    st, ed = pair
                    y_clip = y[st:ed, :]
                    out_name = f"{base}_0.00-{cfg['segment_sec']:.2f}s.wav"
                    out_path = os.path.join(out_dir, out_name)
                    write_clip(y_clip, sr, out_path)
                    index_meta.append({
                        "scheme": scheme_name,
                        "class": cls,
                        "src": fp,
                        "start_sec": 0.0,
                        "end_sec": cfg["segment_sec"],
                        "out": out_path
                    })
                else:
                    seg_sec = cfg["segment_sec"]
                    hop_sec = cfg["hop_sec"]
                    pairs = slice_sliding(y, sr, seg_sec, hop_sec)
                    for st, ed in pairs:
                        y_clip = y[st:ed, :]
                        st_s = st / sr
                        ed_s = ed / sr
                        out_name = f"{base}_{st_s:.2f}-{ed_s:.2f}s.wav"
                        out_path = os.path.join(out_dir, out_name)
                        write_clip(y_clip, sr, out_path)
                        index_meta.append({
                            "scheme": scheme_name,
                            "class": cls,
                            "src": fp,
                            "start_sec": round(st_s, 3),
                            "end_sec": round(ed_s, 3),
                            "out": out_path
                        })

            except Exception as e:
                print(f"[跳过] {fp} 出错：{e}")

    # 可选：保存一份索引（方便你核对）
    try:
        ensure_dir(OUT_ROOT)
        with open(os.path.join(OUT_ROOT, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index_meta, f, ensure_ascii=False, indent=2)
        print(f"\n完成！索引已写入：{os.path.join(OUT_ROOT, 'index.json')}")
    except Exception as e:
        print(f"写入索引失败：{e}")

if __name__ == "__main__":
    main()
