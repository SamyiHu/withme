"""wxhook 通道适配器验收：内容解码 / 表与房间映射 / 拉新消息 / 语音气泡发送参数 /
语音 BLOB 获取。全部用假件直测，不需要真实 hook 与微信。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, repo_config_dir, rmtree, run_tests

from src.config import ConfigError, config_from_dict, load_config
from src.wxhook import (Msg, WxHookChannel, WxHookError, decode_content, table_for_room)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER = "wxid_alice"
SELF = "wxid_bot"


def _cfg(**wxhook_over):
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    for k, v in wxhook_over.items():
        setattr(cfg.wechat.wxhook, k, v)
    return cfg


class FakeHook:
    """替身：拦截 _call（发送/状态），_query（查库）由 cases 预置返回。"""

    def __init__(self):
        self.calls = []          # [(path, body)]
        self.queries = {}        # (db, 关键子串) -> rows；命中先注册先得
        self.default_rows = []

    def __call__(self, path, body=None, timeout=30, method="POST"):
        self.calls.append((path, dict(body or {}), method))
        if path == "/QueryDB/status":
            return {"ret": 0, "IsLogin": 1}
        if path == "/GetSelfProfile":
            return {"ret": 0, "wxid": SELF}
        return {"ret": 0}

    def query(self, db, sql, timeout=30):
        self.calls.append(("SQL:" + db, sql))
        if "COALESCE(MAX" in sql:
            # MAX 水位查询：从对应表的行预设里算最大 local_id
            for (d, key), rows in self.queries.items():
                if d == db and key in sql and rows and "local_id" in rows[0]:
                    return [{"m": max(int(r.get("local_id") or 0) for r in rows)}]
            return [{"m": 0}]
        for (d, key), rows in self.queries.items():
            if d == db and key in sql:
                return rows
        return self.default_rows


def _channel(tmp, fake=None, **wxhook_over):
    fake = fake or FakeHook()
    cfg = _cfg(**wxhook_over).wechat
    ch = WxHookChannel(cfg, os.path.join(tmp, "media"), log=lambda m: None)
    ch._call = fake
    ch._query = lambda db, sql, timeout=30: fake.query(db, sql, timeout)
    ch._silk_to_wav = _fake_silk_to_wav
    return ch, fake


def _fake_silk_to_wav(silk, wav_path):
    assert silk.startswith(b"\x02#!SILK_V3")
    with open(wav_path, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 40)
    return wav_path


def _hex(b: bytes) -> str:
    return b.hex()


# ---------------- 内容解码 ----------------

def test_decode_content_plain_and_zstd():
    assert decode_content(_hex("你好呀".encode("utf-8"))) == "你好呀"
    assert decode_content("") == ""
    assert decode_content(None) == ""
    # 非 hex（奇数长度/含非 hex 字符）原样返回
    assert decode_content("plain-text") == "plain-text"
    # zstd：注入假解压器
    zstd_frame = b"\x28\xb5\x2f\xfd" + b"PAYLOAD"
    out = decode_content(_hex(zstd_frame), zstd_decompress=lambda b: "解压了".encode())
    assert out == "解压了"
    # zstd 且没装解压器 -> 明确报错
    try:
        decode_content(_hex(zstd_frame), zstd_decompress=None)
        raise AssertionError("应报 zstandard 缺失")
    except WxHookError as e:
        assert "zstandard" in str(e)


# ---------------- 表名/配置 ----------------

def test_table_for_room_is_md5():
    import hashlib
    assert table_for_room("wxid_alice") == "Msg_" + hashlib.md5(b"wxid_alice").hexdigest()


def test_config_channel_and_wxhook_section():
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    assert cfg.wechat.channel in ("wcf", "wxhook")
    assert cfg.wechat.wxhook.base_url == "http://127.0.0.1:30001"
    assert cfg.wechat.wxhook.notify_port == 18998
    bad = {"wechat": {"channel": "telegram"}}
    try:
        config_from_dict(bad)
        raise AssertionError("应拒绝非法 channel")
    except ConfigError as e:
        assert "channel" in str(e)
    ok = config_from_dict({"wechat": {"channel": "wxhook",
                                      "wxhook": {"notify_port": 19000, "data_root": "D:/w"}}})
    assert ok.wechat.channel == "wxhook" and ok.wechat.wxhook.notify_port == 19000


# ---------------- 基础面/发送面 ----------------

def test_login_and_self_wxid():
    tmp = make_tmpdir()
    try:
        ch, fake = _channel(tmp)
        assert ch.is_login() and ch.get_self_wxid() == SELF
        assert ch.voice_prefers == "silk"
    finally:
        rmtree(tmp)


def test_send_voice_silk_uses_bytes_mode():
    tmp = make_tmpdir()
    try:
        ch, fake = _channel(tmp)
        silk = os.path.join(tmp, "media", "r.silk")
        open(silk, "wb").write(b"#!SILK_V3xx")
        code = ch.send_voice_silk(silk, OWNER, duration_ms=3300)
        assert code == 0
        path, body, _m = fake.calls[-1]
        assert path == "/SendVoiceMsg"
        assert body == {"wxidorgid": OWNER, "path": silk,
                        "str0Mode": "bytes", "durationMs": 3300}
        # 文本/图片参数
        ch.send_text("hi", OWNER)
        ch.send_image("D:/a.png", OWNER)
        assert fake.calls[-2][1] == {"wxidorgid": OWNER, "msg": "hi"}
        assert fake.calls[-1][1] == {"wxidorgid": OWNER, "path": "D:/a.png"}
        # 非图片文件：明确不支持
        assert ch.send_file("D:/a.mp3", OWNER) != 0
    finally:
        rmtree(tmp)


def test_error_ret_raises():
    tmp = make_tmpdir()
    try:
        ch, fake = _channel(tmp)
        # 打桩 HTTP 层：ret!=0 应转成 WxHookError（保留原始码与信息）
        import io
        import src.wxhook as wxmod

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            return FakeResp(b'{"ret":-1001,"retmsg":"voice disabled"}')
        ch._call = WxHookChannel._call.__get__(ch)   # 恢复真实 _call（_channel 里换成了假件）
        orig_urlopen = wxmod.urllib.request.urlopen
        wxmod.urllib.request.urlopen = fake_urlopen
        try:
            ch.send_text("x", OWNER)
            raise AssertionError("ret!=0 应抛 WxHookError")
        except WxHookError as e:
            assert "-1001" in str(e) and "voice disabled" in str(e)
        finally:
            wxmod.urllib.request.urlopen = orig_urlopen
    finally:
        rmtree(tmp)


# ---------------- 收消息 ----------------

def _mock_tables(fake, rooms=(OWNER,), rows_by_room=None):
    tables = [table_for_room(r) for r in rooms]
    fake.queries[("message_0.db", "sqlite_master")] = [
        {"name": t} for t in tables]
    fake.queries[("message_0.db", "SessionTable")] = [
        {"username": r} for r in rooms]
    fake.queries[("message_0.db", "Name2Id")] = [
        {"rid": 1, "user_name": SELF}, {"rid": 2, "user_name": OWNER}]
    for room, rows in (rows_by_room or {}).items():
        fake.queries[("message_0.db", table_for_room(room))] = rows


def _row(lid, ltype, sender_rid, content=b"hi", source=b""):
    return {"local_id": lid, "local_type": ltype, "create_time": 1790000000,
            "real_sender_id": sender_rid,
            "message_content": _hex(content) if content else "",
            "source": _hex(source) if source else ""}


def test_poll_table_watermark_and_normalize():
    tmp = make_tmpdir()
    try:
        ch, fake = _channel(tmp)
        room_rows = [_row(1, 1, 2, "在吗".encode()), _row(2, 34, 2, "<voicemsg/>".encode())]
        _mock_tables(fake, rows_by_room={OWNER: room_rows})
        # 启动水位：已存在的表只记水位，不回放历史
        ch.bootstrap_watermarks()
        ch.poll_table(table_for_room(OWNER))
        assert ch._q.empty()
        # 新行（含自己发的）-> 规范化入队
        fake.queries[("message_0.db", table_for_room(OWNER))] = [
            _row(2, 34, 2, "<voicemsg/>".encode()),          # 主人语音（重复行，水位内）
            _row(3, 1, 2, "早上好".encode()),                 # 主人文本
            _row(4, 1, 1, "我自己发的".encode()),             # self
        ]
        ch.poll_table(table_for_room(OWNER))
        msgs = [ch._q.get_nowait() for _ in range(2)]
        assert [m.id for m in msgs] == ["3", "4"]
        assert msgs[0].content == "早上好" and msgs[0].sender == OWNER
        assert not msgs[0].from_self()
        assert msgs[1].from_self() and msgs[1].sender == SELF
        assert ch._last_local_id[table_for_room(OWNER)] == 4
    finally:
        rmtree(tmp)


def test_group_tables_gated_by_config():
    """群表拉不拉由 wechat.groups_enabled 决定：关=兜底全表轮询跳过群；
    开=照常拉（回应被@）。"""
    tmp = make_tmpdir()
    try:
        for groups_on, expect_n in ((False, 0), (True, 1)):
            ch, fake = _channel(tmp)
            ch.cfg.groups_enabled = groups_on
            group = "123@chatroom"
            src = "<msgsource><atuserlist><![CDATA[wxid_bot]]></atuserlist></msgsource>"
            _mock_tables(fake, rooms=(OWNER, group),
                         rows_by_room={group: [_row(1, 1, 2, "wxid_a: @阿满 群里".encode(),
                                                    source=src.encode())]})
            ch.bootstrap_watermarks()
            fake.queries[("message_0.db", table_for_room(group))] = [
                _row(1, 1, 2, "占位".encode()),                       # 水位内
                _row(2, 1, 2, "wxid_a: @阿满 群里".encode(), source=src.encode())]
            ch.poll_tables()       # 兜底全表轮询
            msgs = []
            while not ch._q.empty():
                msgs.append(ch._q.get_nowait())
            assert len(msgs) == expect_n, f"groups_on={groups_on}"
            if groups_on:
                assert msgs[0].from_group() and msgs[0].at_me
    finally:
        rmtree(tmp)


# ---------------- 语音 BLOB ----------------

def test_fetch_voice_from_voiceinfo():
    tmp = make_tmpdir()
    try:
        ch, fake = _channel(tmp)
        blob = b"\x02#!SILK_V3" + b"\x01\x02\x03" * 30
        fake.queries[("media_0.db", "Name2Id")] = [{"rid": 7}]
        fake.queries[("media_0.db", "VoiceInfo")] = [{"voice_data": _hex(blob)}]
        msg = Msg(34, "55", OWNER, OWNER, "<voicemsg/>")
        out = ch.fetch_voice(msg)
        assert out and out["mime"] == "audio/wav"
        assert os.path.exists(out["path"])
        # 查询条件带 chat_name_id + local_id
        sql = [c for c in fake.calls if c[0] == "SQL:media_0.db"][-1][1]
        assert "chat_name_id = 7" in sql and "local_id = 55" in sql
        # 库里没有 -> None（桥温和降级）
        fake.queries[("media_0.db", "VoiceInfo")] = []
        assert ch.fetch_voice(msg) is None
    finally:
        rmtree(tmp)


# ---------------- 通知竞争：content 迟到 -> 回读重试 ----------------

def _fake_urlopen(payload: bytes):
    import io

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda req, timeout=None: FakeResp(payload)


def test_emoji_rereads_content_when_cdnurl_missing():
    """实况（2026-10-27 09:28）：msg.xml 是 source 列的 <msgsource> 元数据（非空），
    把真载荷 msg.content 整个遮住 -> 表情/图片全灭。必须按标记挑载荷；
    message_content 迟到时回读消息行兜底。"""
    import src.wxhook as wxmod
    tmp = make_tmpdir()
    orig_sleep = wxmod.time.sleep
    orig_urlopen = wxmod.urllib.request.urlopen
    wxmod.time.sleep = lambda s: None
    wxmod.urllib.request.urlopen = _fake_urlopen(b"J" * 3000)   # 非 GIF -> 静态 jpg
    try:
        ch, fake = _channel(tmp)
        logs = []
        ch.log = logs.append
        # 场景1（当天现场）：content 有真 XML，xml 列是 msgsource 垃圾 -> 直接命中
        xml = '<msg><emoji cdnurl="http://cdn/e.jpg&amp;x=1"/></msg>'
        junk = '<msgsource><signature>N0_V1_x</signature></msgsource>'
        msg = Msg(47, "9", OWNER, OWNER, xml, junk)
        paths = ch.fetch_emoji(msg)
        assert paths and os.path.getsize(paths[0]) == 3000, \
            "msgsource 不得遮蔽真载荷"
        # 场景2：content 也迟到（空）-> 回读消息行恢复
        fake.queries[("message_0.db", table_for_room(OWNER))] = [
            {"message_content": _hex(xml.encode())}]
        paths = ch.fetch_emoji(Msg(47, "10", OWNER, OWNER, "", junk))
        assert paths, "回读后必须恢复"
        sqls = [c for c in fake.calls if c[0] == "SQL:message_0.db"]
        assert any("SELECT message_content" in c[1] for c in sqls), "必须回读消息行"
        # 反面：回读也一直拿不到 -> 有日志（不再静默 None）
        fake.queries[("message_0.db", table_for_room(OWNER))] = [{"message_content": ""}]
        assert ch.fetch_emoji(Msg(47, "11", OWNER, OWNER, "", junk)) is None
        assert any("cdnurl" in m for m in logs)
    finally:
        wxmod.time.sleep = orig_sleep
        wxmod.urllib.request.urlopen = orig_urlopen
        rmtree(tmp)


def test_image_refreshes_xml_across_retries():
    """图片解密同理：xml 列是 msgsource 时必须用 content；首轮两者都缺 key
    时按轮回读。"""
    if sys.platform != "win32":
        print("  [skip] 仅 Windows：刷新链路依赖 ctypes.windll")
        return
    import src.wxhook as wxmod
    from src import wximg
    tmp = make_tmpdir()
    acct = os.path.join(tmp, "acct")
    os.makedirs(os.path.join(acct, "msg", "attach"), exist_ok=True)
    orig_sleep = wxmod.time.sleep
    wxmod.time.sleep = lambda s: None
    seen = []

    def fake_decode(dst, xml, attach, log=None, **kwargs):
        seen.append(xml)
        if 'aeskey="' in xml:
            with open(dst, "wb") as f:
                f.write(b"J" * 500)
            return "jpg"
        return None

    orig_decode = wximg.decode_img_xml
    wximg.decode_img_xml = fake_decode
    try:
        ch, fake = _channel(tmp)
        fake.queries[("message_0.db", "database_list")] = [
            {"file": os.path.join(acct, "db_storage", "message", "message_0.db")}]
        xml = '<msg><img aeskey="abcd1234" cdnthumburl="x"/></msg>'
        junk = '<msgsource><signature>N0_V1_x</signature></msgsource>'
        # 场景1：content 有真 XML + xml 列 msgsource -> 首轮即中
        out = ch.fetch_image(Msg(3, "12", OWNER, OWNER, xml, junk))
        assert out and os.path.getsize(out) == 500
        assert 'aeskey="' in seen[0], "msgsource 不得遮蔽真载荷"
        # 场景2：content 迟到 -> 回读后命中
        seen.clear()
        fake.queries[("message_0.db", table_for_room(OWNER))] = [
            {"message_content": _hex(xml.encode())}]
        out = ch.fetch_image(Msg(3, "13", OWNER, OWNER, "", junk))
        assert out and any('aeskey="' in x for x in seen), "后续轮次应回读到完整 XML"
    finally:
        wximg.decode_img_xml = orig_decode
        wxmod.time.sleep = orig_sleep
        rmtree(tmp)


# ---------------- 群聊拉取与 @ 检测 ----------------

def test_poll_group_table_and_at_me():
    """群表：groups_enabled=true 时照常拉取；source 列的 atuserlist 命中自己
    -> at_me=True（@ 的硬信号，不靠昵称猜）。"""
    tmp = make_tmpdir()
    try:
        group = "45729355366@chatroom"
        src_hit = "<msgsource><atuserlist><![CDATA[wxid_bot]]></atuserlist></msgsource>"
        src_miss = "<msgsource><atuserlist>someone_else</atuserlist></msgsource>"
        old_rows = [_row(6, 1, 2, "旧消息".encode())]
        new_rows = [
            _row(7, 1, 2, "wxid_friend: @阿满 你好".encode(), source=src_hit.encode()),
            _row(8, 1, 2, "wxid_friend: 闲聊".encode(), source=src_miss.encode()),
            _row(9, 10000, 5, "邀请你加入了群聊".encode()),
        ]
        for groups_on, expect_n in ((True, 3), (False, 0)):
            ch, fake = _channel(tmp)
            ch.cfg.groups_enabled = groups_on
            _mock_tables(fake, rooms=(OWNER, group), rows_by_room={group: old_rows})
            ch.bootstrap_watermarks()   # 水位=6：历史不回放
            fake.queries[("message_0.db", table_for_room(group))] = old_rows + new_rows
            ch.poll_table(table_for_room(group))
            msgs = []
            while not ch._q.empty():
                msgs.append(ch._q.get_nowait())
            assert len(msgs) == expect_n, f"groups_on={groups_on} 应拉 {expect_n} 条"
            if groups_on:
                assert msgs[0].from_group() and msgs[0].at_me
                assert msgs[0].sender == OWNER   # real_sender_id=2 -> Name2Id 映射
                assert msgs[1].from_group() and not msgs[1].at_me
                assert not msgs[2].at_me
    finally:
        rmtree(tmp)


def test_nickname_of_from_contact_db():
    tmp = make_tmpdir()
    try:
        ch, fake = _channel(tmp)
        fake.queries[("contact.db", "FROM contact")] = [
            {"nick_name": "阿澄", "remark": ""}]
        assert ch.nickname_of(OWNER) == "阿澄"
        assert ch.nickname_of(OWNER) == "阿澄"   # 命中缓存（不再查库）
        n_calls = len([c for c in fake.calls if c[0] == "SQL:contact.db"])
        fake.queries[("contact.db", "FROM contact")] = []
        assert ch.nickname_of("wxid_nope") == ""     # 查不到给空串，静默
        assert len([c for c in fake.calls if c[0] == "SQL:contact.db"]) == n_calls + 1
    finally:
        rmtree(tmp)


# ---------------- 通知接收端（进程内 HTTP） ----------------

def test_notify_server_triggers_poll():
    tmp = make_tmpdir()
    try:
        ch, fake = _channel(tmp, notify_port=18999)
        _mock_tables(fake, rows_by_room={OWNER: []})
        ch.enable_receiving()
        try:
            import urllib.request
            body = json.dumps({"t": "msg", "table": table_for_room(OWNER)}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:18999/recv", data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200
            # 通知线程异步拉表：等它跑完（首见记水位）
            for _ in range(50):
                if table_for_room(OWNER) in ch._last_local_id:
                    break
                time.sleep(0.1)
            assert table_for_room(OWNER) in ch._last_local_id
        finally:
            ch.disable_receiving()
    finally:
        rmtree(tmp)


# ---------------- SILK 编解码链 ----------------

def test_silk_codec_roundtrip():
    import os
    import struct
    from src.silk_audio import SILK_MAGIC, SilkError, _pcm_to_silk, _pcm_to_wav16, _silk_ok, silk_to_wav16
    tmp2 = make_tmpdir()
    try:
        # WAV 包装：canonical 44 字节头（16k/mono/PCM16）
        pcm = struct.pack("<h", 0) * 16000
        wav = os.path.join(tmp2, "a.wav")
        _pcm_to_wav16(pcm, wav)
        head = open(wav, "rb").read(44)
        assert head[:4] == b"RIFF" and head[8:12] == b"WAVE"
        assert struct.unpack_from("<I", head, 24)[0] == 16000
        assert struct.unpack_from("<I", head, 40)[0] == len(pcm)
        # 格式判定
        assert _silk_ok(b"\x02" + SILK_MAGIC + b"xx") and _silk_ok(SILK_MAGIC + b"xx")
        assert not _silk_ok(b"not silk at all")
        # 真实编解码回环（环境里任一提供者可用时；无提供者则跳过保持 CI 可跑）
        pcm24 = b"\x00\x01" * 2400  # 0.1s @24kHz
        pp = os.path.join(tmp2, "t.pcm")
        open(pp, "wb").write(pcm24)
        try:
            silk = _pcm_to_silk(pp)
        except SilkError:
            return
        assert _silk_ok(silk)
        wav2 = os.path.join(tmp2, "r.wav")
        out = silk_to_wav16(silk, wav2)
        size = os.path.getsize(out)
        assert size > 44 and 0.05 < (size - 44) / 32000.0 < 0.2
    finally:
        rmtree(tmp2)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_wxhook"))
