#!/usr/bin/env python3
"""用 acestep.cpp 服务生成 LoKr / 基线的对比样本，并算高音/相似度指标。

前置：ace-server 已带 --adapters 启动，适配器名为 xiaoyi_lokr_v1.safetensors。
"""
import email
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVC = r"C:/path/to/AI/rvc"
if RVC not in sys.path:
    sys.path.insert(0, RVC)

BASE = "http://127.0.0.1:8085"
OUT = r"C:/path/to/AI/ACE-Step-1.5/outputs/lokr_tests"
ADAPTER = "xiaoyi_lokr_v1.safetensors"

CAPTION = ("Mandopop power ballad, female soprano lead vocal with an extremely high "
           "head voice and sustained high notes, dramatic piano and strings, "
           "slow 4/4, emotional climax")
LYRICS = "[Verse]\n啊——\n[Chorus]\n我爱你 我爱你\n啊——"


def post(path, obj, timeout=180):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def wait(job_id):
    for _ in range(240):
        body, _ = get("/job?id=%s" % job_id)
        state = json.loads(body.decode("utf-8")).get("status")
        if state == "done":
            return
        if state in ("failed", "cancelled"):
            raise RuntimeError("job %s %s" % (job_id, state))
        time.sleep(1)
    raise RuntimeError("job timeout")


def extract_audio(body, ctype):
    if str(ctype).lower().startswith("audio/"):
        return body
    msg = email.message_from_bytes(
        b"Content-Type: " + ctype.encode("latin-1", "replace") +
        b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    for part in msg.walk():
        if part.get_content_maintype() == "audio":
            data = part.get_payload(decode=True)
            if data:
                return data
    raise RuntimeError("no audio part")


def synth(lm, out_path, adapter=None, scale=1.0):
    req = dict(lm)
    req["synth_model"] = "acestep-v15-turbo-Q8_0.gguf"
    req["inference_steps"] = 8
    req["guidance_scale"] = 1.0
    if adapter:
        req["adapter"] = adapter
        req["adapter_scale"] = scale
    else:
        req.pop("adapter", None)
        req.pop("adapter_scale", None)
    job = post("/synth", req)
    wait(job["id"])
    body, ctype = get("/job?id=%s&result=1" % job["id"], 300)
    audio = extract_audio(body, ctype)
    with open(out_path, "wb") as f:
        f.write(audio)
    return out_path


def metrics(path):
    import numpy as np
    from crack_metric import f0, hnr, speaker_sim
    a = f0(path)
    a = a[np.isfinite(a)]
    a = a[a > 0]
    return {
        "p50_hz": round(float(np.median(a)), 1) if a.size else None,
        "p95_hz": round(float(np.percentile(a, 95)), 1) if a.size else None,
        "max_hz": round(float(a.max()), 1) if a.size else None,
        "hnr": round(hnr(path), 2),
        "speaker_sim": speaker_sim(path),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    req = {
        "caption": CAPTION, "lyrics": LYRICS, "duration": 20,
        "vocal_language": "zh", "lm_mode": "generate",
        "lm_model": "acestep-5Hz-lm-1.7B-Q8_0.gguf",
        "synth_model": "acestep-v15-turbo-Q8_0.gguf",
    }
    job = post("/lm", req)
    wait(job["id"])
    body, _ = get("/job?id=%s&result=1" % job["id"])
    lm = json.loads(body.decode("utf-8"))
    lm = lm[0] if isinstance(lm, list) and lm else lm
    print("LM:", {k: lm.get(k) for k in ("caption", "bpm", "keyscale", "duration")})
    results = {}
    for name, adapter, scale in (
            ("base", None, 1.0),
            ("lokr_1.0", ADAPTER, 1.0),
            ("lokr_1.5", ADAPTER, 1.5)):
        path = os.path.join(OUT, name + ".mp3")
        print("synth", name)
        synth(lm, path, adapter=adapter, scale=scale)
        results[name] = {"path": path, **metrics(path)}
        print(name, results[name])
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
