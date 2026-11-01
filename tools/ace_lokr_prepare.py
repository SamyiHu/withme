#!/usr/bin/env python3
"""为 ACE-Step LoKr 小试准备 6 段 60 秒「她的音色 + 伴奏」训练片段。

输入是 RVC 技术语料的产物：
- singing_corpus/converted/<track>_her.wav : RVC 换成她音色的干声
- singing_corpus/work/htdemucs_ft/<track>/no_vocals.wav : 原曲伴奏

每首自动挑「人声高音最集中」的 60 秒窗口，混成 48k 立体声，
同时生成 ACE-Step dataset JSON（歌词取网易云 LRC，caption 手写）。
"""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.songsearch import get_lyric  # noqa: E402

RVC = r"C:/path/to/AI/rvc"
OUT = r"C:/path/to/AI/ACE-Step-1.5/datasets/xiaoyi_6x60"
CLIP_SEC = 60.0

TRACKS = [
    dict(slug="reason", track="alin_reason", title="给我一个理由忘记", artist="A-Lin",
         lyric_id=205342,
         caption="Mandopop emotional ballad, female lead vocal with breathy chest voice "
                 "and soaring high notes, piano and strings, slow 4/4"),
    dict(slug="lightyear", track="gem_lightyear", title="光年之外", artist="G.E.M.",
         lyric_id=449818741,
         caption="Mandopop cinematic pop ballad, powerful female vocal with high belt "
                 "and head voice, piano, strings and electronic drums"),
    dict(slug="atleast", track="sandy_atleast", title="至少还有你", artist="林忆莲",
         lyric_id=256621,
         caption="Mandopop classic ballad, warm female vocal with clear high notes, "
                 "acoustic guitar and strings, gentle 4/4"),
    dict(slug="ordinary_friend", track="src_bili_BV11D4y1t7ff", title="普通朋友", artist="陶喆",
         lyric_id=150623,
         caption="Mandopop R&B ballad, female cover vocal with smooth runs and high "
                 "head voice, clean guitar and soft drums"),
    dict(slug="love_is_simple", track="src_bili_BV1Rbt36FEDb", title="爱很简单", artist="陶喆",
         lyric_id=150654,
         caption="Mandopop R&B love song, breathy female cover vocal with emotional "
                 "high notes, piano and live band"),
    dict(slug="melody", track="src_bili_BV1VG4y1o7Fr", title="Melody", artist="陶喆",
         lyric_id=150520,
         caption="Mandopop R&B ballad, female cover vocal with emotional high register "
                 "and melismatic runs, piano and strings"),
]


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return (os.environ.get("FFMPEG_EXE")
                or r"C:/path/to/Python/Lib/"
                   r"site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")


def clean_lrc(text: str) -> str:
    """LRC -> 纯歌词（去掉时间戳、作词/作曲等元信息）。"""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # [mm:ss.xx] / [mm:ss]
        if s.startswith("[") and "]" in s:
            s = s.split("]", 1)[1].strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in ("作词", "作曲", "编曲", "制作人", "混音", "母带")):
            continue
        out.append(s)
    return "\n".join(out)


def pick_window(vocal_path: str) -> tuple:
    """用 YIN 找 60 秒内高音最集中的窗口；返回 (start, p95_hz)。"""
    import librosa

    y, sr = librosa.load(vocal_path, sr=16000, mono=True)
    dur = len(y) / float(sr)
    if dur <= CLIP_SEC + 2:
        return 0.0, 0.0
    f0 = librosa.yin(y, fmin=65, fmax=1200, sr=sr,
                     frame_length=1024, hop_length=160)
    times = librosa.times_like(f0, sr=sr, hop_length=160)
    best = (0.0, -1.0)
    # 5 秒步进，避免刚好切在字中间；优先高音 p95，同时要求足够多 voiced 帧
    for start in np.arange(0.0, max(0.0, dur - CLIP_SEC), 5.0):
        m = (times >= start) & (times < start + CLIP_SEC)
        vals = f0[m]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 20:
            continue
        p95 = float(np.percentile(vals, 95))
        voiced = float(np.mean(vals > 0))
        score = p95 * min(1.0, voiced / 0.35)
        if score > best[1]:
            best = (float(start), score)
    return best[0], best[1]


def mix_clip(vocal_path: str, inst_path: str, out_path: str, start: float):
    """裁同一段时间的人声+伴奏，混成 48k 立体声。"""
    ff = ffmpeg()
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "%.3f" % start, "-t", "%.3f" % CLIP_SEC, "-i", vocal_path,
        "-ss", "%.3f" % start, "-t", "%.3f" % CLIP_SEC, "-i", inst_path,
        "-filter_complex",
        "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "volume=1.0[v];"
        "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "volume=0.82[i];"
        "[v][i]amix=inputs=2:duration=shortest:normalize=0[out]",
        "-map", "[out]", "-ar", "48000", "-ac", "2",
        "-c:a", "pcm_s16le", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 混音失败：" + (r.stderr or "")[-400:])


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = []
    for item in TRACKS:
        vocal = os.path.join(RVC, "singing_corpus", "converted",
                             item["track"] + "_her.wav")
        inst = os.path.join(RVC, "singing_corpus", "work", "htdemucs_ft",
                            item["track"], "no_vocals.wav")
        if not (os.path.exists(vocal) and os.path.exists(inst)):
            print("SKIP missing", item["slug"], vocal, inst)
            continue
        start, score = pick_window(vocal)
        out_wav = os.path.join(OUT, item["slug"] + ".wav")
        print(f"[{item['slug']}] start={start:.1f}s score={score:.1f} -> {out_wav}")
        mix_clip(vocal, inst, out_wav, start)

        lyrics = clean_lrc(get_lyric(item["lyric_id"]))
        if not lyrics:
            lyrics = "[Verse]\n" + item["title"]
        with open(os.path.join(OUT, item["slug"] + ".lyrics.txt"), "w",
                  encoding="utf-8") as f:
            f.write(lyrics)
        with open(os.path.join(OUT, item["slug"] + ".caption.txt"), "w",
                  encoding="utf-8") as f:
            f.write(item["caption"])
        samples.append({
            "id": item["slug"],
            "audio_path": out_wav,
            "filename": os.path.basename(out_wav),
            "caption": item["caption"],
            "genre": "Mandopop R&B ballad",
            "lyrics": lyrics,
            "bpm": None,
            "keyscale": "",
            "timesignature": "4",
            "duration": CLIP_SEC,
            "language": "zh",
            "is_instrumental": False,
            "labeled": True,
        })
    dataset = {
        "metadata": {
            "name": "xiaoyi_lokr_6x60",
            "all_instrumental": False,
            "genre_ratio": 0,
            "tag_position": "prepend",
            "num_samples": len(samples),
        },
        "samples": samples,
    }
    path = os.path.join(OUT, "dataset.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    print("WROTE", path, "samples", len(samples))


if __name__ == "__main__":
    main()
