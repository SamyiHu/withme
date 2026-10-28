"""微信桥验收：路由/主人绑定/水位线去重/语音镜像/图片链路/链接解析/配置校验。
全部用假件（FakeWcf/FakeAPI）直测 WeChatBridge，不需要真实微信与服务端。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, repo_config_dir, rmtree, run_tests

from src.config import ConfigError, config_from_dict, load_config
from src.asr_client import ASRClient, ASRError, multipart_body
from src.main import Agent
from src.wechat_bridge import (AgentAPI, BridgeState, WeChatBridge,
                               MSG_APPMSG, MSG_IMAGE, MSG_TEXT, MSG_VOICE)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(**wechat_over):
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    cfg.wechat.multi_user = False   # 本文件验单用户路径；多用户见 test_wechat_multi
    for k, v in wechat_over.items():
        setattr(cfg.wechat, k, v)
    return cfg


class FakeMsg:
    def __init__(self, type=MSG_TEXT, content="", sender="wxid_owner",
                 roomid="", id="1000", xml="", extra=""):
        self.type = type
        self.content = content
        self.sender = sender
        self.roomid = roomid
        self.id = id
        self.xml = xml
        self.extra = extra

    def from_self(self):
        return False

    def from_group(self):
        return bool(self.roomid and self.roomid.startswith("xxx@chatroom"))


class FakeWcf:
    """通道接口假件（与 WcfChannel 同构）。"""

    voice_prefers = "mp3"

    def __init__(self, media_dir):
        self.media = media_dir
        self.texts = []      # [(text, receiver)]
        self.images = []     # [(path, receiver)]
        self.files = []      # [(path, receiver)]
        self.silks = []      # [(path, receiver, duration_ms)]
        self.audio_fail = False
        self.image_fail = False

    def bind_media_dir(self, media_dir):
        self.media = media_dir
        return self

    def is_login(self):
        return True

    def get_self_wxid(self):
        return "wxid_bot"

    def enable_receiving_msg(self):
        return True

    def disable_receiving_msg(self):
        return True

    def send_text(self, msg, receiver, aters=""):
        self.texts.append((msg, receiver))
        return 0

    def send_image(self, path, receiver):
        self.images.append((path, receiver))
        return 0

    def send_file(self, path, receiver):
        self.files.append((path, receiver))
        return 0

    def send_voice_silk(self, path, receiver, duration_ms=0):
        self.silks.append((path, receiver, duration_ms))
        return 0

    def _write_media(self, name, data=b"x"):
        path = os.path.join(self.media, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def get_audio_msg(self, msg_id, directory, timeout=3):
        if self.audio_fail:
            return ""
        return self._write_media("voice_%s.mp3" % msg_id, b"FAKE_MP3")

    def download_image(self, msg_id, extra, directory, timeout=30):
        if self.image_fail:
            return ""
        return self._write_media("img_%s.jpg" % msg_id, bytes([0xFF, 0xD8, 0xFF]) + b"FAKE_JPG")

    # ---- 通道媒体接口（桥只调这两个） ----
    def fetch_voice(self, msg):
        mp3 = self.get_audio_msg(int(msg.id), self.media, timeout=5)
        return {"path": mp3, "mime": "audio/mpeg"} if mp3 else None

    def fetch_image(self, msg):
        path = self.download_image(int(msg.id), getattr(msg, "extra", ""), self.media, 30)
        return path or None


class FakeASR:
    def __init__(self, text="今天好累啊"):
        self.text = text
        self.calls = []

    def transcribe(self, audio, filename="audio.mp3", mime="audio/mpeg"):
        self.calls.append(audio)
        return self.text


class FakeAPI:
    """chat 结果与 pending 队列都可预置。"""

    def __init__(self):
        self.chats = []           # [{"text":…, "images":…}]
        self.chat_result = {"reply": "嗯哼，我在", "addon": None,
                            "meta": {"message_id": 1, "hand": None}}
        self.items = []           # /pending 返回的 agent 消息
        self.latest_id = 0
        self.tts_audio = b"MP3BYTES"
        self.tts_error = False

    def chat(self, text, images=None):
        self.chats.append({"text": text, "images": images})
        return self.chat_result

    def pending(self, after_id):
        return [it for it in self.items if it["id"] > after_id]

    def latest_message_id(self):
        return self.latest_id

    def tts(self, text):
        if self.tts_error:
            from src.wechat_bridge import BridgeError
            raise BridgeError("/tts HTTP 503")
        return self.tts_audio

    def health(self):
        return {"ok": True, "role_id": "xiaoman"}


def _bridge(tmp, cfg=None, api=None, state=None, wcf=None, asr=None):
    media = os.path.join(tmp, "media")
    os.makedirs(media, exist_ok=True)
    return WeChatBridge(cfg or _cfg(), wcf or FakeWcf(media), api or FakeAPI(),
                        state or BridgeState(os.path.join(tmp, "state.json")),
                        asr=asr, media_dir=media, log=lambda m: None), media


# ---------------- 配置 ----------------

def test_config_sections_and_validation():
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    assert cfg.wechat.voice_reply == "mirror"
    assert cfg.wechat.poll_seconds == 2.0
    assert cfg.asr.model == "glm-asr-2512"
    assert cfg.image_gen.model == "cogview-3-flash" and cfg.image_gen.enabled
    bad = {"wechat": {"voice_reply": "sometimes"}}
    try:
        config_from_dict(bad)
        raise AssertionError("应拒绝非法 voice_reply")
    except ConfigError as e:
        assert "voice_reply" in str(e)
    bad2 = {"wechat": {"poll_seconds": 0}}
    try:
        config_from_dict(bad2)
        raise AssertionError("应拒绝非法 poll_seconds")
    except ConfigError:
        pass


# ---------------- 路由与主人绑定 ----------------

def test_owner_binding_first_private_sender():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp)
        b.handle_message(FakeMsg(content="在吗", sender="wxid_alice"))
        assert b.state.owner_wxid == "wxid_alice"
        assert len(b.api.chats) == 1 and b.api.chats[0]["text"] == "在吗"
        # 第二个人私聊：已绑定主人，直接忽略
        b.handle_message(FakeMsg(content="你是谁", sender="wxid_bob"))
        assert len(b.api.chats) == 1
    finally:
        rmtree(tmp)


def test_group_and_self_ignored():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp)
        b.state.owner_wxid = "wxid_owner"

        class GroupMsg(FakeMsg):
            def from_group(self):
                return True

        b.handle_message(GroupMsg(content="群里@你", roomid="123@chatroom"))
        assert b.api.chats == []
    finally:
        rmtree(tmp)


def test_hand_images_sent_inline():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp)
        b.state.owner_wxid = "wxid_owner"
        b.api.chat_result = {"reply": "画好了", "addon": None,
                             "meta": {"message_id": 9,
                                      "hand": {"ok": True, "images": ["D:/a.png"]}}}
        b.handle_message(FakeMsg(content="画只猫"))
        assert b.channel.images == [("D:/a.png", "wxid_owner")]
    finally:
        rmtree(tmp)


# ---------------- 语音链路 ----------------

def test_voice_inbound_and_mirror_reply():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp, asr=FakeASR("今天好累啊"))
        b.state.owner_wxid = "wxid_owner"
        b.handle_message(FakeMsg(type=MSG_VOICE, id="77"))
        assert len(b.api.chats) == 1 and b.api.chats[0]["text"] == "[语音] 今天好累啊"
        # 镜像：下一条回复（非主动）应以语音文件卡片发出
        b.api.items = [{"id": 10, "content": "辛苦了，好好歇会儿",
                        "meta": {"prompt_tokens": 1}}]
        b.deliver_pending()
        assert len(b.channel.files) == 1 and b.channel.texts == []
        # 语音意愿已消费：再一条主动消息走文字
        b.api.items = [{"id": 11, "content": "晚上记得吃饭",
                        "meta": {"proactive": {"intent": "express_care"}}}]
        b.deliver_pending()
        assert b.channel.texts == [("晚上记得吃饭", "wxid_owner")]
        assert len(b.channel.files) == 1
    finally:
        rmtree(tmp)


def test_voice_download_fail_notifies():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp, asr=FakeASR())
        b.state.owner_wxid = "wxid_owner"
        b.channel.audio_fail = True
        b.handle_message(FakeMsg(type=MSG_VOICE, id="78"))
        assert b.api.chats == []
        assert any("没能收到" in t[0] for t in b.channel.texts)
    finally:
        rmtree(tmp)


def test_voice_too_long_falls_back_to_text():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp, asr=FakeASR())
        b.state.owner_wxid = "wxid_owner"
        b._voice_pending = 1
        long_reply = "很长" * 300
        b.api.items = [{"id": 12, "content": long_reply, "meta": {}}]
        b.deliver_pending()
        assert b.channel.files == []
        assert b.channel.texts and b.channel.texts[0][0] == long_reply
        assert b._voice_pending == 0  # 意愿已消费，不泄漏
    finally:
        rmtree(tmp)


def test_voice_reply_always_policy():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp, cfg=_cfg(voice_reply="always"))
        b.state.owner_wxid = "wxid_owner"
        b.api.items = [{"id": 13, "content": "早呀", "meta": {}}]
        b.deliver_pending()
        assert len(b.channel.files) == 1
        # 主动消息在 always 策略下也走语音
        b.api.items = [{"id": 14, "content": "晚安",
                        "meta": {"proactive": {"intent": "express_care"}}}]
        b.deliver_pending()
        assert len(b.channel.files) == 2
    finally:
        rmtree(tmp)


def test_tts_unavailable_falls_back():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp, cfg=_cfg(voice_reply="always"))
        b.state.owner_wxid = "wxid_owner"

        def _tts_down(text):
            from src.wechat_bridge import BridgeError
            raise BridgeError("tts down")
        b._tts_bytes = _tts_down   # TTS 全链路不可用（不打网络）
        b.api.tts_error = True
        b.api.items = [{"id": 15, "content": "短回复", "meta": {}}]
        b.deliver_pending()
        assert b.channel.files == [] and b.channel.texts == [("短回复", "wxid_owner")]
    finally:
        rmtree(tmp)


# ---------------- 图片链路 ----------------

def test_image_inbound_sends_b64():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp)
        b.state.owner_wxid = "wxid_owner"
        b.handle_message(FakeMsg(type=MSG_IMAGE, id="88", extra="C:/x.dat"))
        assert len(b.api.chats) == 1
        sent = b.api.chats[0]
        assert sent["text"] == "" and sent["images"]
        assert sent["images"][0]["b64"] and sent["images"][0]["mime"] == "image/jpeg"
        # 下载失败 -> 温和提示，不进 /chat
        b.channel.image_fail = True
        b.handle_message(FakeMsg(type=MSG_IMAGE, id="89"))
        assert len(b.api.chats) == 1
        assert any("没能打开" in t[0] for t in b.channel.texts)
    finally:
        rmtree(tmp)


# ---------------- 链接解析 ----------------

def test_link_message_parsed():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp)
        b.state.owner_wxid = "wxid_owner"
        xml = ("<appmsg><title>某某新闻</title><url>"
               "<![CDATA[https://example.com/a?b=1]]></url></appmsg>")
        b.handle_message(FakeMsg(type=MSG_APPMSG, content=xml, id="91"))
        assert b.api.chats and b.api.chats[0]["text"] == "[链接] 某某新闻 https://example.com/a?b=1"
        # 无 url 的 appmsg（文件/小程序）跳过
        b.handle_message(FakeMsg(type=MSG_APPMSG, content="<appmsg><title>文件</title></appmsg>",
                                 id="92"))
        assert len(b.api.chats) == 1
    finally:
        rmtree(tmp)


# ---------------- 水位线去重与持久化 ----------------

def test_watermark_dedup_and_bootstrap():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp)
        b.state.owner_wxid = "wxid_owner"
        b.api.items = [
            {"id": 20, "content": "第一条", "meta": {"proactive": {}}},
            {"id": 21, "content": "第二条", "meta": {}}],
        # 库内已有更晚的消息：全新启动水位线=当前最大 id，历史不重播
        b.api.latest_id = 25
        b.bootstrap_watermark()
        assert b.state.watermark == 25
        assert b.deliver_pending() == 0 and b.channel.texts == []
        # 持久化值存在时：沿用并夹到当前最大 id
        b.state.watermark = 99
        b.api.latest_id = 30
        b.bootstrap_watermark()
        assert b.state.watermark == 30
        # 正常投递：一次拉取只发一次
        b.api.items = [{"id": 31, "content": "主动消息", "meta": {"proactive": {}}}]
        assert b.deliver_pending() == 1
        assert b.deliver_pending() == 0
        assert b.state.watermark == 31
        assert json.load(open(b.state.path))["watermark"] == 31
    finally:
        rmtree(tmp)


def test_chat_failure_rate_limited_note():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp)
        b.state.owner_wxid = "wxid_owner"

        class DeadAPI(FakeAPI):
            def chat(self, text, images=None):
                raise OSError("refused")

        b.api = DeadAPI()
        b.handle_message(FakeMsg(content="在吗"))
        assert any("后台掉线" in t[0] for t in b.channel.texts)
        # 10 分钟限频：连续失败不再重复提示
        n_before = len(b.channel.texts)
        b.handle_message(FakeMsg(content="还在吗"))
        assert len(b.channel.texts) == n_before
    finally:
        rmtree(tmp)


# ---------------- ASR / multipart ----------------

def test_multipart_body_structure():
    boundary, body = multipart_body([("model", "glm-asr-2512")], "file",
                                    "a.mp3", "audio/mpeg", b"AUDIO")
    text = body.decode("utf-8", "replace")
    assert 'name="model"' in text and "glm-asr-2512" in text
    assert 'filename="a.mp3"' in text and b"AUDIO" in body
    assert body.endswith(f"\r\n--{boundary}--\r\n".encode())


def test_asr_client_requires_key():
    tmp = make_tmpdir()
    try:
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("BIGMODEL_API_KEY", None)
        cfg = _cfg()
        client = ASRClient(cfg.asr, cfg.llm)
        assert client.base_url.endswith("/api/paas/v4")
        try:
            client.transcribe(b"x")
            raise AssertionError("缺 key 应报错")
        except ASRError as e:
            assert cfg.asr.api_key_env in str(e)  # 跟配置走（现为 BIGMODEL_API_KEY）
    finally:
        rmtree(tmp)


# ---------------- Agent 图片入站（核心链路） ----------------

def test_agent_images_enrichment_mock():
    tmp = make_tmpdir()
    try:
        agent = Agent(config_dir=repo_config_dir(),
                      db_path=os.path.join(tmp, "agent.db"))
        result = agent.on_user_message("看看这只猫",
                                       images=[{"bytes": b"JPEGDATA", "mime": "image/jpeg"}])
        assert result["reply"]
        row = agent.conn.execute(
            "SELECT content, meta FROM messages WHERE sender='user'"
            " ORDER BY id DESC LIMIT 1").fetchone()
        assert "看看这只猫" in row["content"] and "[图片1]" in row["content"]
        assert json.loads(row["meta"])["images"][0]["desc"]
    finally:
        agent.conn.close()
        rmtree(tmp)


def test_agent_images_only_message():
    tmp = make_tmpdir()
    try:
        agent = Agent(config_dir=repo_config_dir(),
                      db_path=os.path.join(tmp, "agent.db"))
        agent.on_user_message("", images=[{"bytes": b"IMG", "mime": "image/png"}])
        row = agent.conn.execute(
            "SELECT content FROM messages WHERE sender='user'"
            " ORDER BY id DESC LIMIT 1").fetchone()
        assert row["content"].startswith("[图片1]")
    finally:
        agent.conn.close()
        rmtree(tmp)


# ---------------- 手部 image.draw ----------------

def test_image_draw_disabled_and_enabled():
    tmp = make_tmpdir()
    try:
        cfg = _cfg()
        from src.hands import Hands
        from src.storage import init_db
        conn = init_db(os.path.join(tmp, "a.db"))
        hands = Hands(cfg, conn, "xiaoman")
        assert any(m["name"] == "image.draw" for m in hands.manifest())
        # 关闭开关 -> 明确失败（不打真实网络）
        cfg.image_gen.enabled = False
        r2 = hands.execute("image.draw", {"prompt": "一只猫"})
        assert not r2["ok"] and "未开启" in r2["summary"]
        # 打桩验证成功路径（含缺 prompt 的失败分支）
        import src.imagegen as ig

        def fake_gen(prompt, c, llm, out_dir):
            return {"url": "https://x/y.jpg", "path": os.path.join(out_dir, "y.jpg")}

        orig = ig.generate_image
        ig.generate_image = fake_gen
        try:
            cfg.image_gen.enabled = True
            r3 = hands.execute("image.draw", {"prompt": "一只猫"})
            assert r3["ok"] and r3["image"].endswith("y.jpg")
        finally:
            ig.generate_image = orig
        r4 = hands.execute("image.draw", {})
        assert not r4["ok"] and "缺少" in r4["summary"]
    finally:
        rmtree(tmp)


# ---------------- HTTP 层 /chat images ----------------

def test_server_chat_images_roundtrip():
    import base64
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from src.server import make_server

    tmp = make_tmpdir()
    httpd = None
    try:
        agent = Agent(config_dir=repo_config_dir(), db_path=os.path.join(tmp, "agent.db"),
                      threaded=True)
        httpd = make_server(agent, "127.0.0.1", 0)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/chat"

        def post(payload):
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))

        r = post({"text": "看图", "images": [
            {"b64": base64.b64encode(b"IMG").decode(), "mime": "image/png"},
            {"b64": "!!!not-base64!!!"}]})  # 第二张坏数据应被跳过
        assert r["reply"]
        row = agent.conn.execute(
            "SELECT content FROM messages WHERE sender='user'"
            " ORDER BY id DESC LIMIT 1").fetchone()
        assert "[图片1]" in row["content"]
        # 无文本无图片 -> 400
        try:
            post({"text": ""})
            raise AssertionError("空消息应 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        if httpd:
            httpd.shutdown()
        agent.conn.close()
        rmtree(tmp)


# ---------------- 语音气泡（wxhook 通道链路，编码打桩） ----------------

def test_voice_bubble_silk_channel():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp, cfg=_cfg(voice_reply="always"))
        b.state.owner_wxid = "wxid_owner"
        b.channel.voice_prefers = "silk"
        b._tts_bytes = lambda text: b"MP3BYTES"   # 不打网络 TTS
        import src.silk_audio as sa

        def fake_mp3_to_silk(mp3, silk_path, work_dir=None):
            assert mp3 == b"MP3BYTES"
            with open(silk_path, "wb") as f:
                f.write(b"#!SILK_V3....")
            return {"path": silk_path, "duration_ms": 3400, "bytes": 12}

        orig = sa.mp3_to_silk
        sa.mp3_to_silk = fake_mp3_to_silk
        try:
            b.api.items = [{"id": 40, "content": "晚安，好梦", "meta": {}}]
            b.deliver_pending()
        finally:
            sa.mp3_to_silk = orig
        assert len(b.channel.silks) == 1
        silk_path, peer, dur = b.channel.silks[0]
        assert peer == "wxid_owner" and dur == 3400 and silk_path.endswith(".silk")
        assert b.channel.texts == [] and b.channel.files == []
    finally:
        rmtree(tmp)


def test_voice_bubble_encode_fail_falls_back():
    tmp = make_tmpdir()
    try:
        b, _ = _bridge(tmp, cfg=_cfg(voice_reply="always"))
        b.state.owner_wxid = "wxid_owner"
        b.channel.voice_prefers = "silk"

        def _tts_down(text):
            from src.wechat_bridge import BridgeError
            raise BridgeError("tts down")
        b._tts_bytes = _tts_down
        import src.silk_audio as sa

        def boom(mp3, silk_path, work_dir=None):
            raise sa.SilkError("pilk 没装")

        orig = sa.mp3_to_silk
        sa.mp3_to_silk = boom
        try:
            b.api.items = [{"id": 41, "content": "短回复", "meta": {}}]
            b.deliver_pending()
        finally:
            sa.mp3_to_silk = orig
        assert b.channel.silks == []
        assert b.channel.texts == [("短回复", "wxid_owner")]
        assert b._voice_pending == 0
    finally:
        rmtree(tmp)




def test_quote_reply_parsed():
    """引用回复（appmsg type=57）此前整条隐形——必须还原成
    '[引用 X：「…」] 新回复'，被引的是表情包/图片时按类型概括。"""
    class QMsg(FakeMsg):
        def __init__(self, xml):
            super().__init__(type=49, content=xml, sender="wxid_friend")

    xml_text = ('<msg><appmsg><title>这句话用语音发给我</title><type>57</type><url />'
                '<refermsg><content>生椰那味儿香是香，但太甜了</content>'
                '<displayname>阿满</displayname></refermsg></appmsg></msg>')
    out = WeChatBridge._parse_quote(QMsg(xml_text))
    assert out.startswith('[引用 阿满：「生椰') and '语音发给我' in out
    xml_emoji = ('<msg><appmsg><title>这啥</title><type>57</type><url />'
                 '<refermsg><content>&lt;msg&gt;&lt;emoji md5="a5"/&gt;&lt;/msg&gt;</content>'
                 '<displayname>阿澄</displayname></refermsg></appmsg></msg>')
    out2 = WeChatBridge._parse_quote(QMsg(xml_emoji))
    assert '表情包' in out2 and '这啥' in out2
    xml_link = ('<msg><appmsg><title>分享</title><type>5</type>'
                '<url>https://example.com/a</url></appmsg></msg>')
    assert WeChatBridge._parse_quote(QMsg(xml_link)) is None

if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_wechat_bridge"))
