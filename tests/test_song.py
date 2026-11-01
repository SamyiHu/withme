"""唱歌链路：歌词整理 / ACE-Step 请求 / 手部异步任务 / 配置校验。

全部离线：provider=mock 或 monkeypatch，不连本地 ace-server。
"""
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import songgen
from src.config import ConfigError, SongGenConfig, config_from_dict
from src.hands import Hands
from src.storage import init_db


def test_sanitize_lyrics_strips_tags_and_waves():
    out = songgen.sanitize_lyrics("[Verse]\n夜色～落在窗台\n\n你把心事藏起来")
    assert out == "夜色 落在窗台\n你把心事藏起来", repr(out)
    assert songgen.sanitize_lyrics("") == songgen.DEFAULT_LYRICS
    assert songgen.sanitize_lyrics("   ") == songgen.DEFAULT_LYRICS


def test_sanitize_lyrics_splits_long_paragraph():
    out = songgen.sanitize_lyrics("夜色落在窗台，你把心事藏起来，我轻轻唱给你听")
    assert out.count("\n") >= 2, repr(out)
    assert "夜色落在窗台" in out


def test_mock_provider_generates_wav():
    tmp = make_tmpdir()
    try:
        cfg = SongGenConfig(provider="mock", out_dir=tmp)
        info = songgen.generate_song("a\nb", "测试", cfg)
        assert os.path.exists(info["path"]) and info["path"].endswith(".wav")
        assert info["duration_ms"] == 1000
        assert info["provider"] == "mock"
    finally:
        rmtree(tmp)


def test_acestep_cpp_payload_and_multipart():
    tmp = make_tmpdir()
    try:
        calls = []
        lm_payload = [{
            "caption": "enriched folk", "lyrics": "一\n二", "duration": 55,
            "audio_codes": "codes", "bpm": 80, "keyscale": "C Major",
        }]
        audio_bytes = b"ID3FAKEMP3"
        boundary = "ace-boundary"
        multipart = (
            f"--{boundary}\r\nContent-Type: audio/mpeg\r\n\r\n".encode()
            + audio_bytes + f"\r\n--{boundary}--\r\n".encode())

        def fake_post(base, path, obj, timeout):
            calls.append((path, obj))
            return {"id": "job-%d" % len(calls)}

        def fake_wait(base, job_id, cfg, logger=None, label="song"):
            return None

        def fake_get(base, path, timeout):
            if "&result=1" in path:
                # job 序号：奇数=第 N 轮的 /lm 结果，偶数=该轮 /synth 的音频
                n = int(path.split("job-")[1].split("&")[0])
                if n % 2 == 1:
                    return json.dumps(lm_payload).encode(), "application/json"
                return multipart, f"multipart/mixed; boundary={boundary}"
            return b'{"status":"done"}', "application/json"

        old_post, old_wait, old_get, old_dur = (
            songgen._post_json, songgen._wait_job, songgen._get, songgen.audio_duration_ms)
        songgen._post_json, songgen._wait_job, songgen._get = fake_post, fake_wait, fake_get
        songgen.audio_duration_ms = lambda p: 54240
        try:
            # 时长：对话里"1分钟"覆盖默认；不再被 max_send_seconds 在生成侧砍掉
            cfg = SongGenConfig(provider="acestep_cpp", out_dir=tmp,
                                duration_seconds=120, max_duration_seconds=300,
                                max_send_seconds=55)
            info = songgen.generate_song("我自己写的词", "温柔民谣", cfg,
                                         voice="低沉男声", duration="1分钟")
            songgen.generate_song("词", "民谣", cfg, duration="10分钟")   # 超上限夹住
            songgen.generate_song("词", "民谣", cfg)                      # 没给用默认
        finally:
            songgen._post_json, songgen._wait_job, songgen._get, songgen.audio_duration_ms = (
                old_post, old_wait, old_get, old_dur)

        lm_payloads = [obj for path, obj in calls if path == "/lm"]
        assert [p["duration"] for p in lm_payloads] == [60, 300, 120], \
            [p["duration"] for p in lm_payloads]
        payload = lm_payloads[0]
        assert payload["lyrics"] == "我自己写的词"
        assert payload["vocal_language"] == "zh"
        assert "低沉男声" in payload["caption"], payload["caption"]
        assert os.path.exists(info["path"])
        with open(info["path"], "rb") as f:
            assert f.read() == audio_bytes
        assert info["duration_ms"] == 54240
        assert info["lyrics"] == "一\n二"
        assert info["voice"] == "低沉男声"
    finally:
        rmtree(tmp)


def test_parse_duration():
    assert songgen.parse_duration("30") == 30.0
    assert songgen.parse_duration("45秒") == 45.0
    assert songgen.parse_duration("1分钟") == 60.0
    assert songgen.parse_duration("1.5分钟") == 90.0
    assert songgen.parse_duration(62.5) == 62.5
    assert songgen.parse_duration("随便") is None            # 解析不了 -> default
    assert songgen.parse_duration(None, default=55.0) == 55.0
    assert songgen.parse_duration("999", hi=300.0) == 300.0   # 超上限夹住
    assert songgen.parse_duration("3", lo=10.0) == 10.0       # 低于下限抬住


def test_sanitize_keeps_user_lyrics_many_lines():
    text = "\n".join("第%d行测试歌词" % i for i in range(1, 9))   # 8 行用户歌词
    out = songgen.sanitize_lyrics(text)
    assert out.count("\n") == 7, out
    assert "第8行测试歌词" in out


def test_hands_song_sing_runs_in_background():
    tmp = make_tmpdir()
    try:
        conn = init_db(":memory:", threaded=True)
        cfg = SimpleNamespace(
            hands=SimpleNamespace(enabled=True, allow_shell=False),
            song_gen=SongGenConfig(provider="mock", out_dir=tmp))
        hands = Hands(cfg, conn, "test-role")
        got = []

        def fake_generate(lyrics, style, cfg, out_dir=None, voice="",
                          duration=None, logger=None):
            return {"path": os.path.join(tmp, "x.mp3"), "duration_ms": 1000,
                    "lyrics": lyrics, "style": style, "voice": voice,
                    "duration": duration, "provider": "mock"}

        old = songgen.generate_song
        songgen.generate_song = fake_generate
        hands.song_sender = got.append
        try:
            res = hands.execute("song.sing", {"lyrics": "一\n二", "style": "测试",
                                              "voice": "男声", "duration": "45秒"})
            assert res["ok"] and res.get("singing"), res
            assert res.get("duration") == 45.0, res
            deadline = time.time() + 5
            while time.time() < deadline and not got:
                time.sleep(0.02)
        finally:
            songgen.generate_song = old
        assert got and got[0]["duration_ms"] == 1000, got
        assert got[0]["voice"] == "男声", got
        assert got[0]["duration"] == 45.0, got
    finally:
        rmtree(tmp)


def test_hands_song_sing_asks_style_when_missing():
    tmp = make_tmpdir()
    try:
        conn = init_db(":memory:", threaded=True)
        cfg = SimpleNamespace(
            hands=SimpleNamespace(enabled=True, allow_shell=False),
            song_gen=SongGenConfig(provider="mock", out_dir=tmp, style=""))
        hands = Hands(cfg, conn, "test-role")
        res = hands.execute("song.sing", {"lyrics": "一\n二"})
        assert res["ok"] is False and res.get("need_style"), res
        assert "风格" in res["summary"], res
    finally:
        rmtree(tmp)


def test_manifest_contains_song_sing():
    cfg = SimpleNamespace(
        hands=SimpleNamespace(enabled=True, allow_shell=False),
        song_gen=SongGenConfig(provider="mock"))
    hands = Hands(cfg, None, "test-role")
    assert "song.sing" in [m["name"] for m in hands.manifest()]


def test_bridge_song_bubble_uses_silk():
    from src import silk_audio
    from src.wechat_bridge import WeChatBridge

    tmp = make_tmpdir()
    try:
        mp3 = os.path.join(tmp, "song.mp3")
        with open(mp3, "wb") as f:
            f.write(b"ID3FAKE")
        sent = []

        class Channel:
            voice_prefers = "silk"

            def send_voice_silk(self, path, peer, duration_ms=0):
                sent.append((path, peer, duration_ms))
                return 0

            def send_file(self, path, peer):
                sent.append(("file", path, peer))
                return 0

        fake = SimpleNamespace(channel=Channel(), media_dir=tmp,
                               cfg=SimpleNamespace(song_gen=SimpleNamespace(
                                   max_send_seconds=55)),
                               log=lambda m: None)
        old = silk_audio.mp3_to_silk
        silk_audio.mp3_to_silk = lambda data, path, work_dir=None: {
            "path": os.path.join(tmp, "x.silk"), "duration_ms": 55000, "bytes": 123}
        try:
            ok = WeChatBridge._send_song_bubble(fake, "peer", mp3, 55000)
        finally:
            silk_audio.mp3_to_silk = old
        assert ok is True
        assert sent and sent[0][1] == "peer" and sent[0][2] == 55000, sent
    finally:
        rmtree(tmp)


def test_bridge_send_mode_both_and_file():
    from src import silk_audio
    from src.wechat_bridge import WeChatBridge

    tmp = make_tmpdir()
    try:
        mp3 = os.path.join(tmp, "song.mp3")
        with open(mp3, "wb") as f:
            f.write(b"ID3FAKE")

        class Channel:
            voice_prefers = "silk"

            def __init__(self):
                self.calls = []

            def send_voice_silk(self, path, peer, duration_ms=0):
                self.calls.append(("bubble", peer, duration_ms))
                return 0

            def send_file(self, path, peer):
                self.calls.append(("file", path, peer))
                return 0

        def make_bridge(mode, max_send=55):
            ch = Channel()
            cfg = SimpleNamespace(song_gen=SimpleNamespace(send_mode=mode,
                                                           max_send_seconds=max_send))
            bridge = SimpleNamespace(cfg=cfg, channel=ch, media_dir=tmp,
                                     log=lambda m: None)
            bridge._send_song_bubble = lambda *a, **k: WeChatBridge._send_song_bubble(
                bridge, *a, **k)
            bridge._send_song_file = lambda *a, **k: WeChatBridge._send_song_file(
                bridge, *a, **k)
            bridge._send_song_split_bubbles = lambda *a, **k: (
                ch.calls.append(("split",)), True)[1]
            bridge._send_text = lambda *a, **k: None
            return bridge, ch

        old = silk_audio.mp3_to_silk
        silk_audio.mp3_to_silk = lambda data, path, work_dir=None: {
            "path": os.path.join(tmp, "x.silk"), "duration_ms": 55000, "bytes": 123}
        try:
            bridge, ch = make_bridge("both")
            WeChatBridge._send_song_result(bridge, "peer", {"path": mp3,
                                                             "duration_ms": 55000})
            assert [c[0] for c in ch.calls] == ["bubble", "file"], ch.calls
            bridge, ch = make_bridge("file")
            WeChatBridge._send_song_result(bridge, "peer", {"path": mp3,
                                                             "duration_ms": 55000})
            assert [c[0] for c in ch.calls] == ["file"], ch.calls
            # 超过气泡上限（默认 55s）：SILK 通道切多段语音；文件通道仍发文件
            bridge, ch = make_bridge("both")
            WeChatBridge._send_song_result(bridge, "peer", {"path": mp3,
                                                             "duration_ms": 90000})
            assert [c[0] for c in ch.calls] == ["split"], ch.calls
            bridge, ch = make_bridge("both")
            ch.voice_prefers = "mp3"
            WeChatBridge._send_song_result(bridge, "peer", {"path": mp3,
                                                             "duration_ms": 90000})
            assert [c[0] for c in ch.calls] == ["file"], ch.calls
        finally:
            silk_audio.mp3_to_silk = old
    finally:
        rmtree(tmp)


def test_config_song_gen_defaults_and_validation():
    cfg = config_from_dict({"song_gen": {"provider": "acestep_cpp",
                                         "duration_seconds": 55}})
    assert cfg.song_gen.duration_seconds == 55
    assert cfg.song_gen.max_duration_seconds == 300
    assert cfg.song_gen.max_lyrics_chars == 600
    assert cfg.song_gen.vocal_language == "zh"
    assert cfg.song_gen.mp3_bitrate == 320
    assert cfg.song_gen.send_mode == "both"
    assert cfg.song_gen.style == "", "不该有写死的默认曲风"
    assert cfg.song_gen.voice == "女声"
    try:
        config_from_dict({"song_gen": {"provider": "nope"}})
    except ConfigError as e:
        assert "song_gen.provider" in str(e)
    else:
        raise AssertionError("非法 provider 应该报错")


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_song"))


def test_duet_titles_are_not_used_for_solo_acappella():
    from src import songsearch
    assert songsearch._is_duet("周杰伦『珊瑚海』双人女声翻唱")
    assert songsearch._is_duet("珊瑚海 男女对唱")
    assert songsearch._is_duet("珊瑚海 女生和声版")
    assert not songsearch._is_duet("珊瑚海 女声部分")
    assert not songsearch._is_duet("珊瑚海 女声独唱")


def test_bad_source_markers_block_ai_and_short_clips():
    from src import songsearch
    assert songsearch._is_bad_source("流沙 / 天天 (Live)")
    assert songsearch._is_bad_source("你要的全拿走 - SUNO改编")
    assert songsearch._is_bad_source("珊瑚海 现场版")
    assert not songsearch._is_bad_source("珊瑚海 女声部分")
