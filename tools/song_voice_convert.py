"""唱歌换音色管线（在 seedvc 虚拟环境里运行，核心仓库不依赖它）。

用法（由 src/hands.py 以 seedvc venv 的 python 调起）：
  python tools/song_voice_convert.py \
      --input song.mp3 --reference ref.mp3 --output out.mp3 \
      --workdir <tmp> --ffmpeg <ffmpeg.exe> \
      --seedvc-dir C:/path/to/AI/seedvc/code \
      --demucs-repo C:/path/to/AI/seedvc/models \
      [--steps 30] [--jobs 4] [--timeout 600]

三步：
  1) demucs htdemucs_ft --two-stems=vocals 分离人声/伴奏（CPU）
  2) Seed-VC inference.py 零样本换音色（44.1kHz f0 模型，CPU，fp16 关）
  3) 换完的人声 + 原伴奏混音，ffmpeg 编码 mp3 320k

设计约束：任何一步失败都以非零退出并打印原因，调用方回落原歌声；
不在本脚本里吞错误假装成功。
"""
import argparse
import glob
import os
import subprocess
import sys
import time


def run(cmd, cwd=None, env=None, timeout=600, label=""):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                       text=True, timeout=timeout)
    dt = round(time.time() - t0, 1)
    if r.returncode != 0:
        tail = ((r.stderr or "") + (r.stdout or ""))[-800:]
        print(f"[convert] {label or cmd[1]} FAILED rc={r.returncode} ({dt}s)\n{tail}",
              flush=True)
        raise SystemExit(2)
    print(f"[convert] {label or cmd[1]} ok ({dt}s)", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--seedvc-dir", required=True)   # 含 inference.py
    ap.add_argument("--demucs-repo", required=True)  # 含 htdemucs_ft.yaml + .th
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--vocal-gain", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    total0 = time.time()

    # 1) 人声分离
    run([sys.executable, "-m", "demucs", "-n", "htdemucs_ft",
         "--repo", args.demucs_repo, "--two-stems=vocals",
         "-j", str(args.jobs), "-o", args.workdir, args.input],
        timeout=args.timeout, label="demucs")
    vocals = os.path.join(args.workdir, "htdemucs_ft", stem, "vocals.wav")
    no_vocals = os.path.join(args.workdir, "htdemucs_ft", stem, "no_vocals.wav")
    for p in (vocals, no_vocals):
        if not os.path.exists(p):
            print(f"[convert] demucs 输出缺失：{p}", flush=True)
            raise SystemExit(2)

    # 2) Seed-VC 零样本换音色（离线：缓存已由 curl 预灌）
    vc_out = os.path.join(args.workdir, "vc")
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env.pop("HF_ENDPOINT", None)
    run([sys.executable, "inference.py",
         "--source", os.path.abspath(vocals),
         "--target", os.path.abspath(args.reference),
         "--output", os.path.abspath(vc_out),
         "--f0-condition", "True", "--auto-f0-adjust", "True",
         "--diffusion-steps", str(args.steps),
         "--inference-cfg-rate", "0.7", "--length-adjust", "1.0",
         "--fp16", "False"],
        cwd=args.seedvc_dir, env=env, timeout=args.timeout, label="seed-vc")
    wavs = sorted(glob.glob(os.path.join(vc_out, "vc_*.wav")))
    if not wavs:
        print("[convert] seed-vc 没有产出 wav", flush=True)
        raise SystemExit(2)
    converted = wavs[-1]

    # 3) 混音（换色人声 + 原伴奏），ffmpeg 出 mp3 320k
    import numpy as np   # venv 里随 librosa 装好
    import soundfile as sf
    v, sr1 = sf.read(converted, always_2d=True)
    i, sr2 = sf.read(no_vocals, always_2d=True)
    if sr1 != sr2:
        print(f"[convert] 采样率不一致 {sr1} vs {sr2}", flush=True)
        raise SystemExit(2)
    n = min(len(v), len(i))
    mix = i[:n] + v[:n] * args.vocal_gain
    peak = float(np.max(np.abs(mix))) if n else 0.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)
    mix_wav = os.path.join(args.workdir, "mix.wav")
    sf.write(mix_wav, mix, sr1)
    run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-i", mix_wav, "-codec:a", "libmp3lame", "-b:a", "320k",
         os.path.abspath(args.output)],
        timeout=120, label="encode")
    print(f"[convert] DONE {args.output} total={round(time.time()-total0,1)}s",
          flush=True)


if __name__ == "__main__":
    main()
