"""多用户模式验收：每个私聊用户一份独立小满（独立库）、主人沿用默认库、
白名单与上限、语音镜像即时投递。真实 Agent（mock 渲染）+ 假通道。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, repo_config_dir, rmtree, run_tests

from src.config import config_from_dict
from src.main import Agent
from src.storage import init_db, default_db_path
from src.wechat_bridge import BridgeState, UserPool, WeChatBridge
from test_wechat_bridge import FakeASR, FakeAPI, FakeMsg, FakeWcf, MSG_VOICE

OWNER = "wxid_owner"
FRIEND = "wxid_friend"


def _mcfg(**over):
    cfg = config_from_dict({"wechat": {"multi_user": True, "max_users": 2}})
    for k, v in over.items():
        setattr(cfg.wechat, k, v)
    return cfg


def _mbridge(tmp, cfg=None, asr=None):
    media = os.path.join(tmp, "media")
    os.makedirs(media, exist_ok=True)
    b = WeChatBridge(cfg or _mcfg(), FakeWcf(media), FakeAPI(),
                     BridgeState(os.path.join(tmp, "state.json")),
                     asr=asr or FakeASR("语音内容"), media_dir=media, log=lambda m: None)
    b.async_dispatch = False   # 测试同步直调，断言无需等待
    b._started_at = time.time() - 3600   # 当成跑了很久：跳过启动静默期（warmup）
    return b


def test_pool_isolation_and_owner_continuity():
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")  # 别碰生产库
    try:
        pool = UserPool(_mcfg(), lambda m: None, lambda: OWNER, base_dir=tmp)
        owner_agent = pool.get(OWNER)
        friend_agent = pool.get(FRIEND)
        assert owner_agent is not friend_agent
        # 主人 -> 默认库（指向 tmp）；朋友 -> 独立库目录
        assert pool._db_path_for(OWNER) is None
        friend_db = pool._db_path_for(FRIEND)
        assert "wechat_users" in friend_db and FRIEND in friend_db
        # 消息各入各库
        owner_agent.on_user_message("主人的话")
        friend_agent.on_user_message("朋友的话")
        o = owner_agent.conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE sender='user'").fetchone()["c"]
        f = friend_agent.conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE sender='user'").fetchone()["c"]
        assert o == 1 and f == 1
        # 上限：max_users=2，第三人被拒
        assert pool.get("wxid_third") is None
        owner_agent.conn.close()
        friend_agent.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_multi_routing_and_replies():
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp)
        b.state.owner_wxid = OWNER          # 已有主人：沿用默认库
        b.handle_message(FakeMsg(content="你好呀", sender=OWNER))
        b.handle_message(FakeMsg(content="早上好", sender=FRIEND))
        # 两个人都得到了回复，且发给了对的人
        texts = b.channel.texts
        assert texts and all(t[1] in (OWNER, FRIEND) for t in texts)
        owners = [t for t in texts if t[1] == OWNER]
        friends = [t for t in texts if t[1] == FRIEND]
        assert owners and friends
        # 记忆/消息隔离：主人库只进了主人的话，朋友库只进了朋友的话
        assert b.pool.agents[OWNER].conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE sender='user'").fetchone()["c"] == 1
        assert b.pool.agents[FRIEND].conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE sender='user'"
            " AND content LIKE '%早上好%'").fetchone()["c"] == 1
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_multi_whitelist():
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        cfg = _mcfg()
        cfg.wechat.allowed_wxids = [OWNER]
        b = _mbridge(tmp, cfg)
        b.state.owner_wxid = OWNER
        b.handle_message(FakeMsg(content="在吗", sender=FRIEND))
        assert b.channel.texts == [] and b.pool.agents.get(FRIEND) is None
        b.handle_message(FakeMsg(content="在吗", sender=OWNER))
        assert any(t[1] == OWNER for t in b.channel.texts)
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_multi_voice_mirror_inline():
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(voice_reply="mirror"))
        b.state.owner_wxid = OWNER
        b.channel.voice_prefers = "silk"
        import src.silk_audio as sa

        def fake_mp3_to_silk(mp3, silk_path, work_dir=None):
            open(silk_path, "wb").write(b"#!SILK_V3")
            return {"path": silk_path, "duration_ms": 2600, "bytes": 10}
        orig = sa.mp3_to_silk
        sa.mp3_to_silk = fake_mp3_to_silk
        b._tts_bytes = lambda text: b"MP3BYTES"   # 不打网络 TTS
        try:
            b.handle_message(FakeMsg(type=MSG_VOICE, id="701", sender=OWNER))
        finally:
            sa.mp3_to_silk = orig
        # 语音入站 -> 回复以语音气泡即时发出（主动附加层允许走文字）
        assert len(b.channel.silks) == 1 and b.channel.silks[0][1] == OWNER
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)




# ---------------- 拟人发送（拆条+延迟） ----------------

def test_split_bubbles():
    from src.wechat_bridge import split_bubbles
    assert split_bubbles("好的") == ["好的"]
    assert split_bubbles("") == []
    long = "今天真的累死了，开了一整天会什么都没定下来。跟你说个事，我们组跟了俩月的那个项目今天正式黄了，甲方拍板不做了。哎，瞬间就不想干了都。晚上得吃点好的安慰一下自己，你也早点休息。"
    assert len(long) > 60
    out = split_bubbles(long)
    assert 2 <= len(out) <= 3 and "".join(out) == long, out
    assert all(len(b) > 5 for b in out)
    # 已是两短句 -> 不强拆
    two = "行吧。那先这样咯。"
    assert len(split_bubbles(two, max_bubbles=3)) <= 2
    # 真实病例（2026-10-29）：43 字双行文本 ≤48，旧快速路径原样单发——
    # 微信呈现为"一条消息中间空一行"。含换行必须拆成两条气泡。
    two_line = ("行行行，收到收到，这话题翻篇翻篇\n\n"
                "那聊点别的？你中午吃了没，拉瓦萨券最后搞到没")
    out2 = split_bubbles(two_line)
    assert len(out2) == 2, out2
    assert out2[0] == "行行行，收到收到，这话题翻篇翻篇", out2
    assert all("\n" not in b for b in out2), out2


def test_humanized_reply_delay_and_split():
    import src.wechat_bridge as wb
    old_db2 = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(make_tmpdir(), "o.db")
    tmp = make_tmpdir()
    try:
        b = _mbridge(tmp)
        b.state.owner_wxid = OWNER
        sleeps = []
        b.stop.wait = lambda t: sleeps.append(round(t, 1))  # 记录延迟而非真睡
        b._send_reply("今天真的累死了，开了一整天会什么都没定下来。跟你说个事，我们组跟了俩月的那个项目今天正式黄了，甲方拍板不做了。哎，瞬间就不想干了都。", OWNER)
        assert len(b.channel.texts) >= 2, "长回复应拆成多条"
        assert sleeps and all(d >= 1.0 for d in sleeps), sleeps
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db2:
            os.environ["AGENT_DB_PATH"] = old_db2
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


# ---------------- 连发合并（batch） ----------------

def test_burst_batching_combines_messages():
    import queue as _q
    old_db3 = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(make_tmpdir(), "o.db")
    tmp = make_tmpdir()
    try:
        b = _mbridge(tmp, _mcfg())
        b.state.owner_wxid = OWNER
        b.async_dispatch = True
        calls = []
        b._multi_chat = lambda text, images, was_voice, peer, **_kw: calls.append((text, peer))
        b.cfg.wechat.batch_quiet_seconds = 0.3   # 测试窗口缩短
        b._multi_message(FakeMsg(content="我跟你讲", sender=OWNER), OWNER)
        b._multi_message(FakeMsg(content="今天那个事", sender=OWNER), OWNER)
        b._multi_message(FakeMsg(content="真的绝了", sender=OWNER), OWNER)
        for _ in range(40):
            if calls:
                break
            import time as _t; _t.sleep(0.2)
        assert len(calls) == 1, f"应合并为一次处理，实际 {len(calls)}"
        text, peer = calls[0]
        NL = chr(10)
        assert "我跟你讲" in text and "今天那个事" in text and "真的绝了" in text
        assert NL in text, "多条应以换行拼接"
        assert peer == OWNER
    finally:
        if old_db3:
            os.environ["AGENT_DB_PATH"] = old_db3
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)




# ---------------- 表情包识别 ----------------

def test_emoji_placeholder_when_unavailable():
    old_db4 = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(make_tmpdir(), "o.db")
    tmp = make_tmpdir()
    try:
        b = _mbridge(tmp)
        b.state.owner_wxid = OWNER
        calls = []
        b._multi_chat = lambda text, images, was_voice, peer, **_kw: calls.append((text, images))
        b._process_batch([FakeMsg(type=47, content='<msg><emoji md5="abc"/></msg>',
                                  sender=OWNER)], OWNER)   # FakeWcf 无 fetch_emoji
        assert calls and calls[0][0].find("表情包") >= 0, calls
    finally:
        if old_db4:
            os.environ["AGENT_DB_PATH"] = old_db4
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_emoji_image_recognized(tmp_mode="real"):
    import base64 as _b64
    old_db5 = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(make_tmpdir(), "o.db")
    tmp = make_tmpdir()
    try:
        b = _mbridge(tmp)
        b.state.owner_wxid = OWNER
        gif = os.path.join(tmp, "fake_emoji.jpg")
        open(gif, "wb").write(bytes([0xFF, 0xD8, 0xFF]) + b"JPEGDATA" * 100)
        def fake_fetch_emoji(msg):
            return [gif, gif, gif]   # 多帧抽样：列表
        b.channel.fetch_emoji = fake_fetch_emoji
        calls = []
        b._multi_chat = lambda text, images, was_voice, peer, **_kw: calls.append((text, images))
        b._process_batch([FakeMsg(type=47, content="<msg></msg>", sender=OWNER)], OWNER)
        assert calls, "应触发一次处理"
        text, images = calls[0]
        expect = _b64.b64encode(
            bytes([0xFF, 0xD8, 0xFF]) + b"JPEGDATA" * 100).decode()
        # 一个动图表情包 = 一条图片消息：首帧主图 + 其余帧采样组（不再是 3 张独立图）
        assert images and len(images) == 1, images
        im = images[0]
        assert im["b64"] == expect and im.get("animated") is True
        assert len(im.get("frames") or []) == 2 and all(f == expect for f in im["frames"])
        assert text.find("[表情包]") < 0, "拿到图就不该用占位符"
    finally:
        if old_db5:
            os.environ["AGENT_DB_PATH"] = old_db5
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_local_agent_accepts_b64_images():
    """多用户桥直调本地 Agent 时 images 是 REST 线格式（b64）。
    实况（2026-10-27 09:41）：桥收齐 3 帧，Agent 视觉层却全读成（空图片）——
    bytes/b64 两种格式必须都认。"""
    import base64
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "agent.db")
    try:
        seen_bytes = []

        class VisionClient:
            name = "fake"

            def __init__(self):
                self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                                   "estimated": True}

            def render(self, prompt, seed):
                return "哦这图啊"

            def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
                return ""

            def render_image(self, text, image_bytes, mime="image/jpeg",
                             seed=0, max_tokens=500):
                seen_bytes.append((bytes(image_bytes), mime))
                return "一只橘色丑萌玩偶表情包"

        agent = Agent(config_dir=repo_config_dir(),
                      db_path=os.path.join(tmp, "agent.db"),
                      llm_client=VisionClient())
        png = bytes([0x89, 0x50, 0x4E, 0x47]) + b"fakepng" * 40
        out = agent.on_user_message("看看这个", now=1000.0, images=[
            {"b64": base64.b64encode(png).decode(), "mime": "image/png"}])
        assert out["reply"]
        assert seen_bytes and seen_bytes[0][0] == png, "b64 必须解出原始字节"
        assert seen_bytes[0][1] == "image/png"
        row = agent.conn.execute(
            "SELECT content FROM messages WHERE role_id=? AND sender='user' "
            "ORDER BY id DESC LIMIT 1", ("xiaoman",)).fetchone()
        assert "丑萌玩偶" in row["content"], "视觉描述必须进入消息文本"
        assert "空图片" not in row["content"]
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_animated_emoji_single_vision_call():
    """动图表情包 = 一次视觉调用：采样帧走同一条消息的 frames part，
    指令言明"同一张动图"——模型不再把首/中/尾帧当成三张独立图片。"""
    import base64
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "agent.db")
    try:
        calls = []

        class VisionClient:
            name = "fake"

            def __init__(self):
                self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                                   "estimated": True}

            def render(self, prompt, seed):
                return "这表情包绝了"

            def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
                return ""

            def render_image(self, text, image_bytes, mime="image/jpeg",
                             seed=0, max_tokens=500, frames=None):
                calls.append((text, bytes(image_bytes), list(frames or [])))
                return "一只猫拍手鼓掌的动图"

        agent = Agent(config_dir=repo_config_dir(),
                      db_path=os.path.join(tmp, "agent.db"),
                      llm_client=VisionClient())
        f1 = bytes([0xFF, 0xD8, 0xFF]) + b"frame1" * 30
        f2 = b"frame2" * 30
        f3 = b"frame3" * 30
        out = agent.on_user_message("哈哈这个", now=1000.0, images=[{
            "b64": base64.b64encode(f1).decode(), "mime": "image/jpeg",
            "animated": True,
            "frames": [base64.b64encode(f2).decode(),
                       base64.b64encode(f3).decode()]}])
        assert out["reply"]
        assert len(calls) == 1, "一个动图表情包只该有一次视觉调用"
        text, main, frames = calls[0]
        assert main == f1
        assert "动图" in text and "不是多张图" in text, text
        assert [b for b, _ in frames] == [f2, f3], "采样帧必须原样到达视觉端"
        row = agent.conn.execute(
            "SELECT content FROM messages WHERE role_id=? AND sender='user' "
            "ORDER BY id DESC LIMIT 1", ("xiaoman",)).fetchone()
        assert "拍手鼓掌" in row["content"]
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_skip_marker_never_leaks():
    """[skip] 是内部控制标记。真实泄露病例（2026-10-29，AI技术冲了群）：
    模型写 "[skip] 一碗确实悬…"（标记+正文），旧逻辑只认完全相等，标记原样
    发到微信。修复：前缀标记剥掉——只剩标记=跳过；带正文=发剥掉后的正文。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "agent.db")
    try:
        class SkipClient:
            name = "fake"
            last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}

            def __init__(self, text):
                self.text = text

            def render(self, prompt, seed):
                return self.text

            def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
                return ""

        agent = Agent(config_dir=repo_config_dir(),
                      db_path=os.path.join(tmp, "agent.db"),
                      llm_client=SkipClient("[skip] 一碗确实悬，天冷了饭量骗人的"))
        out = agent.on_user_message("今天吃了几碗", now=1000.0)
        assert out["reply"] == "一碗确实悬，天冷了饭量骗人的", out["reply"]
        assert "[skip]" not in out["reply"]
        # 只有标记 = 跳过：不发、不落库
        agent2 = Agent(config_dir=repo_config_dir(),
                       db_path=os.path.join(tmp, "agent2.db"),
                       llm_client=SkipClient("[skip]"))
        out2 = agent2.on_user_message("群里在聊别人的事", now=1000.0)
        assert out2["reply"] == "" and out2["meta"].get("skipped") is True
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_chime_skips_message_aimed_at_others():
    """群里@别人的定向对话，AI 不插嘴——对话对象判断（meta.at_names）在旁听
    入库时就已算好，插嘴门槛直接复用。真实病例（2026-10-29）：主人@marblue
    "到点走不"被插嘴接成"走啊，说好05就05"，看起来像没@也乱回。"""
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=OWNER, xml=""):
                super().__init__(type=1, content=content, sender=sender, roomid=room)
                self.at_me = False
                self.xml = xml

            def from_group(self):
                return True

        # 主人 @ 别人：旁听入库（带 at_names），插嘴必须跳过
        xml = "<atuserlist><![CDATA[wxid_friend]]></atuserlist>"
        b.handle_message(GroupMsg("@marblue 到点走不", sender=OWNER, xml=xml))
        agent = b.pool.agents[room]
        calls = []
        agent.chime_in = lambda *a, **kw: calls.append(1) or ""
        assert b.chime_check(room, now=time.time() + 5) is True
        assert not calls, "定向@别人的消息不该触发插嘴判断"
        # @别人但明显是给大家看的（广播信号）-> 交给模型判断
        b.handle_message(GroupMsg("@marblue 大家今晚聚餐啊", sender=OWNER, xml=xml))
        assert b.chime_check(room, now=time.time() + 9) is True
        assert calls, "@别人的广播型消息应进入插嘴判断"
        calls.clear()
        # 对照：群友普通消息（无@）照常进入插嘴判断
        b.handle_message(GroupMsg("你们几点走", sender="wxid_friend"))
        assert b.chime_check(room, now=time.time() + 13) is True
        assert calls, "普通群聊消息应进入插嘴判断"
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_bare_mention_stripped_not_used_as_content():
    """只@不说话（"@小满"）：剥成空串+提示应一声，不能把昵称当正文让模型
    拿旧话题凑答案（2026-10-29 实测：@ 一下就回了上一个话题的股市分析）。"""
    from src.wechat_bridge import WeChatBridge
    sep = chr(0x2005)
    assert WeChatBridge._strip_group_noise("@小满") == ""
    assert WeChatBridge._strip_group_noise("@小满" + sep) == ""
    assert WeChatBridge._strip_group_noise("@小满" + sep + "在吗") == "在吗"
    assert WeChatBridge._strip_group_noise("@小满 在吗") == "在吗"


def test_bare_at_with_wxid_prefix_gets_ack():
    """裸@带 wxid 前缀（真实形态 "wxid_xxx:
@小满 "）：不能静默丢弃。
    真实病例（2026-10-29 家人群）：连发两条 @小满 没回，像机器人没启动——
    旧实现跳前缀检查 _raw.startswith("@") 永远不成立，提示没注入、消息被丢。"""
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_self_aliases=["小满", "机器人"]))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=OWNER):
                super().__init__(type=1, content=content, sender=sender, roomid=room)
                self.at_me = True

            def from_group(self):
                return True

        b.handle_message(GroupMsg("wxid_owner:\n@小满" + chr(0x2005)))
        texts = [t for t in b.channel.texts if t[1] == room]
        assert texts, "裸@必须回一声，不能静默丢弃"
        row = b.pool.agents[room].conn.execute(
            "SELECT content FROM messages WHERE sender='user' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["content"] == "@小满", "裸@保留原样入库，不伪造用户发言"
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_mention_routes_and_labels():
    """群聊：被@（at_me 硬信号）才回；文本去掉 wxid 前缀与@称呼、标注说话人；
    回复发到群；群绝不绑定为「主人」。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_self_aliases=["小满", "机器人"]))
        b.state.owner_wxid = OWNER
        sep5 = chr(0x2005)

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=FRIEND, at_me=True, **kw):
                super().__init__(content=content, sender=sender,
                                 roomid="457@chatroom", **kw)
                self.at_me = at_me

            def from_group(self):
                return True

        # 群友被@：路由到群 Agent，回复发到群
        b.handle_message(GroupMsg(content=f"wxid_friend: @阿满{sep5}今天有啥热点"))
        room = "457@chatroom"
        assert room in b.pool.agents, "被@必须为群建独立小满"
        texts = [t for t in b.channel.texts if t[1] == room]
        assert texts, "回复必须发到群"
        row = b.pool.agents[room].conn.execute(
            "SELECT content, meta FROM messages WHERE sender='user' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["content"] == "今天有啥热点", "正文只留内容，角色不进正文"
        assert json.loads(row["meta"] or "{}").get("speaker_role") == "member"
        assert "wxid_friend:" not in row["content"] and "@阿满" not in row["content"]
        # 没被@的群友闲聊：不建 Agent、不回复
        n_agents = len(b.pool.agents)
        b.handle_message(GroupMsg(content="wxid_friend: 有人吗", at_me=False))
        assert len(b.pool.agents) == n_agents
        assert len([t for t in b.channel.texts if t[1] == room]) == len(texts)
        # 主人叫名字 = 对话对象是我：回
        b.handle_message(GroupMsg(content="wxid_owner: 小满 看看", sender=OWNER, at_me=False))
        assert len([t for t in b.channel.texts if t[1] == room]) > len(texts)
        owner_row = b.pool.agents[room].conn.execute(
            "SELECT content, meta FROM messages WHERE sender='user' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert owner_row["content"] == "小满 看看", "主人正文也不带角色前缀"
        assert json.loads(owner_row["meta"] or "{}").get("speaker_role") == "owner"
        # 主人跟群友聊天、没叫 AI（无@/名字/你/问号）-> 旁听不接
        # （2026-10-29 实测：主人和 marblue/AW 聊工作，AI 每条都插一句）
        n_before = len([t for t in b.channel.texts if t[1] == room])
        b.handle_message(GroupMsg(content="wxid_owner: 一个品质两个研发",
                                  sender=OWNER, at_me=False))
        assert len([t for t in b.channel.texts if t[1] == room]) == n_before,             "主人跟别人聊天不该接"
        # 指令句式（无"你"无问号）也算对我说："用mcp" 必须回
        # （2026-10-29 实测：主人回"用mcp"，被当旁听不吭声了）
        n_before2 = len([t for t in b.channel.texts if t[1] == room])
        b.handle_message(GroupMsg(content="wxid_owner: 用mcp", sender=OWNER, at_me=False))
        assert len([t for t in b.channel.texts if t[1] == room]) > n_before2,             "指令句式必须命中回复路径"
        # 接话词（纯延续指令）也必须走回复："继续吧""还有呢"——
        # 靠插嘴兜底会因冷却漏回（2026-10-29 AI技术群实测）
        for cont in ("继续吧", "还有呢", "分点一口气说完吧", "都一口气讲完"):
            n0 = len([t for t in b.channel.texts if t[1] == room])
            b.handle_message(GroupMsg(content="wxid_owner: " + cont,
                                      sender=OWNER, at_me=False))
            assert len([t for t in b.channel.texts if t[1] == room]) > n0, cont
        # 系统消息（10000 邀请通知）不回
        b.handle_message(GroupMsg(type=10000, content='"X"邀请你加入了群聊'))
        # 群绝不抢主人绑定
        b2 = _mbridge(make_tmpdir(), _mcfg(groups_enabled=True))
        b2.handle_message(GroupMsg(content="wxid_friend: @阿满 hi"))
        assert b2.state.owner_wxid == "", "群不得绑定为唯一主人"
        for a in list(b.pool.agents.values()) + list(b2.pool.agents.values()):
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_voice_mirror_sends_voice():
    """群语音：mirror 策略全场景生效——群里收到语音也回语音（2026-10-27
    按反馈放开，旧「群内一律文字」的验收已废止）。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True))   # voice_reply 默认 mirror
        b.state.owner_wxid = OWNER

        class GroupVoice(FakeMsg):
            def __init__(self):
                super().__init__(type=MSG_VOICE, content="", sender=OWNER,
                                 roomid="457@chatroom", id="501")

            def from_group(self):
                return True

        GroupVoice.at_me = True
        b.handle_message(GroupVoice())
        room = "457@chatroom"
        assert room in b.pool.agents
        assert [f for f in b.channel.files if f[1] == room], "群语音 mirror 应回语音"
        assert not b.channel.texts, "语音链路成功时不再补文字"
        assert not b.channel.silks, "FakeWcf 走 mp3 文件卡片，不是 SILK"
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_proactive_mentions_last_speaker():
    """群主动（引擎参数把关频率）：发送时 @ 最近说话人，免得像自言自语；
    私聊不 @。"""
    tmp = make_tmpdir()
    try:
        b = _mbridge(tmp)
        room = "457@chatroom"

        class FakeAgent:
            def __init__(self):
                self.evaluated = 0

            def evaluate_proactive(self, observe=False):
                self.evaluated += 1
                return {"sent": {"text": "群里冒个泡", "level": 2}}

            def maybe_reflect(self):
                pass

        ga = FakeAgent()
        b.pool.agents[room] = ga
        b._group_last[room] = (FRIEND, "AW")
        b._tick_agent(room, ga)
        assert ("@AW 群里冒个泡", room) in b.channel.texts, "群主动必须 @ 最近说话人"
        # 没记录到说话人（比如群里还没人说过话）：原文发，不加 @
        b._group_last.pop(room)
        b._tick_agent(room, ga)
        assert ("群里冒个泡", room) in b.channel.texts
        # 私聊主动：不加 @
        pa = FakeAgent()
        b.pool.agents[OWNER] = pa
        b._tick_agent(OWNER, pa)
        assert ("群里冒个泡", OWNER) in b.channel.texts
        assert not any(t[0].startswith("@") and t[1] == OWNER for t in b.channel.texts)
        assert ga.evaluated == 2 and pa.evaluated == 1
    finally:
        rmtree(tmp)


def test_enabled_chats_scope():
    """启用范围（wechat.enabled_chats）：非空=只处理列出的会话。
    2026-10-29 需求：只启用鱼群，其他群/私聊静默。"""
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        room = "50107328065@chatroom"
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_self_aliases=["小满", "机器人"],
                                enabled_chats=[room]))
        b.state.owner_wxid = OWNER

        class GroupMsg(FakeMsg):
            def __init__(self, roomid):
                super().__init__(type=1, content="wxid_friend: 闲聊一句",
                                 sender="wxid_friend", roomid=roomid)
                self.at_me = False

            def from_group(self):
                return True

        # 范围外的群：不建 Agent、不处理
        b.handle_message(GroupMsg("999999@chatroom"))
        assert "999999@chatroom" not in b.pool.agents
        # 范围内的群：正常建 Agent 并旁听入库
        b.handle_message(GroupMsg(room))
        assert room in b.pool.agents
        n = b.pool.agents[room].conn.execute(
            "SELECT COUNT(*) FROM messages").fetchone()[0]
        assert n >= 1
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_overhear_without_reply():
    """旁听：群友没@我 -> 不回复，但话要记进群 Agent 的消息表（带 overheard
    标记）——被@时接得上前文、主动引擎有素材。只旁听文字。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=FRIEND, mtype=1):
                super().__init__(type=mtype, content=content, sender=sender, roomid=room)

            def from_group(self):
                return True

        GroupMsg.at_me = False
        b.handle_message(GroupMsg(content="wxid_friend: 今天天气不错"))
        assert not [t for t in b.channel.texts if t[1] == room], "旁听不得回复"
        assert room in b.pool.agents, "旁听也要有群 Agent 接收"
        rows = b.pool.agents[room].conn.execute(
            "SELECT content, meta FROM messages WHERE sender='user'").fetchall()
        assert any("今天天气不错" in r["content"] and "overheard" in (r["meta"] or "")
                   for r in rows), "旁听内容要入库且带标记"
        # 长消息不腰斩（曾 text[:200] 把长消息截断入库）：常规长度全量、
        # 防刷屏上限之外显式带截断标记——模型不能把半句当整句
        b.handle_message(GroupMsg(content="wxid_friend: " + "今天" * 150))    # 300 字
        b.handle_message(GroupMsg(content="wxid_friend: " + "啊" * 4500))     # 超上限
        rows2 = b.pool.agents[room].conn.execute(
            "SELECT content FROM messages WHERE sender='user' ORDER BY id").fetchall()
        ok_stored = [r["content"] for r in rows2 if "今天今天" in r["content"]]
        assert ok_stored and len(ok_stored[0]) >= 300, "300 字消息必须全量入库"
        over_stored = [r["content"] for r in rows2 if "啊啊" in r["content"]]
        assert over_stored and "超长已截断" in over_stored[0], "超长必须显式带截断标记"
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_owner_at_other_is_not_a_command():
    """主人 @ 别人 = 在跟别人说话：不命中、不执行，addressee=群聊、@名字入库。

    2026-10-28 实测：主人说「你是大肥鱼，你不要说话」(@大肥鱼)，AI 当成给自己的
    指令，自己闭嘴了——身份认知混乱的根源是"主人的每句话都算对我说"。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        import json as _json
        import sqlite3
        import threading
        b = _mbridge(tmp, _mcfg(groups_enabled=True))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class FakeAgent:
            role_id = "g"
            _lock = threading.RLock()

            def __init__(self):
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL,"
                    " speaker TEXT, speaker_role TEXT)")

        fa = FakeAgent()
        b.pool.agents[room] = fa
        hits = []
        b._multi_message = lambda msg, peer: hits.append(
            (getattr(msg, "_addressee", ""), list(getattr(msg, "_at_names", []))))

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=OWNER, xml=""):
                super().__init__(content=content, sender=sender, roomid=room)
                self.at_me = False
                self.xml = xml

            def from_group(self):
                return True

        other = "wxid_someone"
        b.handle_message(GroupMsg("@大肥鱼 你别说话了",
                                  xml="<atuserlist><![CDATA[%s]]></atuserlist>" % other))
        assert not hits, "主人@别人不该命中回复路径"
        rows = fa.conn.execute(
            "SELECT content, meta, speaker_role FROM messages WHERE sender='user'").fetchall()
        assert rows, "旁听也要入库"
        assert rows[-1]["speaker_role"] == "owner", "主人的话不能被记成群友"
        meta = _json.loads(rows[-1]["meta"])
        assert meta["addressee"] == "群聊", meta
        assert meta.get("at_names"), "要记住 @ 了谁"

        # @ 我自己：命中，addressee=你
        b.handle_message(GroupMsg("@阿满 你说",
                                  xml="<atuserlist>wxid_bot</atuserlist>"))
        assert hits and hits[0][0] == "你", hits
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_sticker_marker_sends_image():
    """回复里的 [sticker:情绪词] 要发成真表情图，正文里不留协议标记。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        from test_stickers import FakeStore
        b = _mbridge(tmp, _mcfg())
        b.state.owner_wxid = OWNER
        store = FakeStore(os.path.join(tmp, "stickers"))

        class Agent:
            role_id = "g"
            stickers = store

            def on_user_message(self, text, **kw):
                return {"reply": "笑死\n[sticker:摸鱼]", "addon": None, "meta": {}}

        b.pool.agents[OWNER] = Agent()
        b._multi_chat("哈哈", None, False, OWNER)
        assert b.channel.images, "表情图没发出去"
        assert b.channel.images[0][0].endswith(".png"), "static 模式发缩略图"
        assert b.channel.texts and all("[sticker" not in t[0] for t in b.channel.texts)
        assert any("笑死" in t[0] for t in b.channel.texts)
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_chime_gates_and_decision():
    """群插嘴门控（新）：有一条没人接的话 + 静默 4s 就判一次；
    主人消息不攒门；冷却只拦"自发插话"；话题冷掉不翻旧账。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_self_aliases=["小满", "机器人"],
                                group_chime_cooldown_seconds=60,
                                group_chime_daily_max=12))
        b._send_reply = lambda text, peer, **kw: b.channel.texts.append((text, peer))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class ChimeAgent:
            def __init__(self):
                self.decisions = 0
                self.last_chime_meta = {"to_me": False, "worth": 0.0}
                import sqlite3
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL, speaker TEXT, speaker_role TEXT)")
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','user','[AW] 今天服务器又炸了','{}',1.0)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "哈哈这梗我接一下"

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=FRIEND, at_me=False):
                super().__init__(content=content, sender=sender, roomid=room)
                self.at_me = at_me

            def from_group(self):
                return True

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        # 主人消息（带名字=对我说话，走回复路径）不攒插嘴门
        b.handle_message(GroupMsg("小满在吗", sender=OWNER, at_me=False))
        assert not b.chime_check(room, now=time.time() + 10)
        assert ca.decisions == 0
        # 单条群友消息 + 静默窗（主人那条刚占过 20s 静默）：判一次（不再要求攒两条）
        b.handle_message(GroupMsg("wxid_friend: 闲聊一句"))
        now = time.time()
        assert b.chime_check(room, now=now + 25)
        assert ca.decisions == 1
        assert any(t[1] == room and "接一下" in t[0] for t in b.channel.texts)
        # 冷却期（60s）内再聊：照样判断，但"自发插话"不发
        b.handle_message(GroupMsg("wxid_friend: 又聊一句"))
        n_texts = len(b.channel.texts)
        assert b.chime_check(room, now=now + 45)
        assert ca.decisions == 2
        assert len(b.channel.texts) == n_texts, "冷却期只判不发"
        # 冷却过了 + 话题还热：判完就发
        b.handle_message(GroupMsg("wxid_friend: 再聊一句"))
        assert b.chime_check(room, now=now + 95)
        assert ca.decisions == 3
        assert len(b.channel.texts) > n_texts
        # 话题冷掉（>240s）：不翻旧账
        b.handle_message(GroupMsg("wxid_friend: 隔了很久才回"))
        assert not b.chime_check(room, now=now + 400)
        assert ca.decisions == 3
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_chime_judges_mention_and_waits_after_reply():
    """①"提到我"不再挡住判断；②刚发过话 20s 内不判；③点名不受冷却限制。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_chime_cooldown_seconds=120,
                                group_chime_after_reply_seconds=20))
        b._send_reply = lambda text, peer, **kw: b.channel.texts.append((text, peer))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class ChimeAgent:
            def __init__(self):
                self.decisions = 0
                self.last_chime_meta = {"to_me": True, "worth": 9.0}
                import sqlite3
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL, speaker TEXT, speaker_role TEXT)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "在呢，什么事"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        now = time.time()
        # 刚发过话（10s 前）：先不判
        b._group_watch[room] = {"msgs": 2, "oh": 2, "mention": 1, "about_me": 1,
                                "last_ts": now, "last_chime": 0.0,
                                "day": 0, "chimes": 0}
        b._last_send[room] = now - 10
        assert not b.chime_check(room, now=now + 8)
        assert ca.decisions == 0
        # 过了静默期：就算在冷却中、就算只是"提到我"，也要判（交给模型）
        b._last_send[room] = now - 30
        b._group_watch[room]["last_chime"] = now - 30
        assert b.chime_check(room, now=now + 10), "提到我也要能进判断"
        assert ca.decisions == 1
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_chime_model_to_me_respects_cooldown():
    """模型判的"在跟我说话"（to_me）也受冷却约束——只有硬点名（directed）豁免。
    2026-10-29 22:40 实测：旁听对话被模型判 to_me，1 分钟内连发 3 条插嘴。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_chime_cooldown_seconds=120))
        b._send_reply = lambda text, peer, **kw: b.channel.texts.append((text, peer))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class ChimeAgent:
            def __init__(self):
                import sqlite3
                self.decisions = 0
                self.last_chime_meta = {"to_me": True, "worth": 9.0}
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL,"
                    " speaker TEXT, speaker_role TEXT)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "在呢，什么事"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        now = time.time()
        b._group_watch[room] = {"msgs": 2, "oh": 2, "mention": 0, "about_me": 0,
                                "last_ts": now, "last_chime": now - 30,
                                "day": time.localtime(now).tm_yday, "chimes": 0}
        b._last_send[room] = now - 70
        assert b.chime_check(room, now=now + 10)
        assert ca.decisions == 1
        assert not [t for t in b.channel.texts if t[1] == room],             "to_me 但非硬点名：冷却中不应发送"
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_chime_not_blocked_by_own_db_row():
    """chime 自己落库后不能被自己的"防双发"守卫拦住——main 在返回文本前
    已把 chime 记进库，旧守卫按"最后一条是 agent"判断会把每条 chime 误杀
    （2026-10-29 23:38 实测：一句都发不出去）。只有"最新用户消息之后已有
    别的 agent 回复"才该跳过。"""
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_chime_cooldown_seconds=1))
        b._send_reply = lambda text, peer, **kw: b.channel.texts.append((text, peer))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class ChimeAgent:
            def __init__(self):
                import sqlite3
                self.last_chime_meta = {"to_me": False, "worth": 7.0}
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL,"
                    " speaker TEXT, speaker_role TEXT)")
                self.conn.execute("INSERT INTO messages(role_id,sender,content,meta,created_at)"
                                  " VALUES('g','user','最新一句','{}',1.0)")
                self.role_id = "g"
                self.logged = False

            def chime_in(self, dialogue, now=None, **kwargs):
                # 模拟 main：chime 返回前先落库
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','agent','接一句试试','{}',2.0)")
                self.logged = True
                return "接一句试试"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        now = time.time()
        b._group_watch[room] = {"msgs": 2, "oh": 2, "mention": 0, "about_me": 0,
                                "last_ts": now, "last_chime": 0.0,
                                "day": time.localtime(now).tm_yday, "chimes": 0}
        b._last_send[room] = now - 120
        assert b.chime_check(room, now=now + 10)
        assert ca.logged and any(t[1] == room for t in b.channel.texts),             "chime 自己落库不应拦自己"
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_chime_skips_conversation_after_at_others():
    """近几条里有人 @ 别人（非广播型、3 分钟内）：紧随其后的接话也不插嘴。
    真实病例（2026-10-29 23:50 鱼群）：@深情的大肥鱼 "还在吗" 之后的
    "坠机了吗" 被 chime 当成问自己，回了"在呢，信号好好的"。"""
    tmp = make_tmpdir()
    old_db = os.environ.pop("AGENT_DB_PATH", None)
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_chime_cooldown_seconds=1))
        b._send_reply = lambda text, peer, **kw: b.channel.texts.append((text, peer))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class ChimeAgent:
            def __init__(self):
                import sqlite3
                self.decisions = 0
                self.last_chime_meta = {"to_me": True, "worth": 9.0}
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL,"
                    " speaker TEXT, speaker_role TEXT)")
                now = time.time()
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','user','wxid_friend: @深情的大肥鱼 还在吗',?,?)",
                    ('{"at_names":["深情的大肥鱼"]}', now - 40))
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','user','坠机了吗','{}',?)", (now - 20,))
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "在呢，信号好好的"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        now = time.time()
        b._group_watch[room] = {"msgs": 2, "oh": 2, "mention": 0, "about_me": 0,
                                "last_ts": now - 10, "last_chime": 0.0,
                                "day": time.localtime(now).tm_yday, "chimes": 0}
        b._last_send[room] = now - 120
        assert b.chime_check(room, now=now)
        assert ca.decisions == 0, "在@别人的对话里不该插嘴"
        assert not [t for t in b.channel.texts if t[1] == room]
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_name_mention_goes_to_reply_path_once():
    """叫名字（没@）= 软点名：走正常回复路径（攒批 + 静默窗），只答一次。

    2026-10-28 实测事故：点名被当成"插嘴硬信号"，插嘴对每条消息各接一次
    （群里每句都以 阿满 开头），几分钟刷出十几条——所以点名改走回复路径，
    插嘴不再重复接。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        from test_wechat_bridge import FakeWcf as _FW

        class NamedWcf(_FW):
            def nickname_of(self, wxid):
                return "阿满" if wxid == "wxid_bot" else ""

        cfg = _mcfg(groups_enabled=True, group_self_aliases=["小满", "机器人"])
        media = os.path.join(tmp, "media")
        os.makedirs(media, exist_ok=True)
        b = WeChatBridge(cfg, NamedWcf(media), FakeAPI(),
                         BridgeState(os.path.join(tmp, "s.json")), media_dir=media,
                         log=lambda m: None)
        b.async_dispatch = False
        b.state.owner_wxid = OWNER
        room = "457@chatroom"
        routed = []
        b._multi_message = lambda msg, peer: routed.append((msg.content, peer))

        class GroupMsg(FakeMsg):
            def __init__(self, content):
                super().__init__(content=content, sender=FRIEND, roomid=room)
                self.at_me = False

            def from_group(self):
                return True

        b.handle_message(GroupMsg("wxid_friend: 阿满 在吗"))
        assert len(routed) == 1, "点名必须走回复路径（一次）"
        assert not b.chime_check(room, now=time.time() + 8), "插嘴不许再重复接一次"

        # 通用别名「机器人」同样算点名
        routed.clear()
        b.handle_message(GroupMsg("wxid_friend: 机器人都不理我，还聊啥"))
        assert len(routed) == 1, "叫机器人也要接（走回复路径）"
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_chime_not_vetoed_by_about_me():
    """群里"提到我"不再是硬否决：交给模型判断接不接。

    2026-10-28 实测：鱼群里每句话都带"阿满"，旧逻辑 about_me 直接 return False，
    插嘴永远进不去 → 表现为"只回主人"。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                 group_chime_cooldown_seconds=120,
                                 group_chime_after_reply_seconds=20))
        b._send_reply = lambda text, peer, **kw: b.channel.texts.append((text, peer))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class ChimeAgent:
            def __init__(self):
                self.decisions = 0
                import sqlite3
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL, speaker TEXT, speaker_role TEXT)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "接一句"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        now = time.time()
        b._group_watch[room] = {"msgs": 2, "oh": 2, "mention": 0, "about_me": 1,
                                "last_ts": now, "last_chime": 0.0,
                                "day": 0, "chimes": 0}
        b._last_send[room] = now - 100
        assert b.chime_check(room, now=now + 8), "提到我也要能进插嘴决策"
        assert ca.decisions == 1
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_chime_cooldown_only_blocks_self_initiated():
    """冷却只拦"自发插话"：判断照常做，冷却期内不发；冷却过了才发。

    门的顺序很重要：先让模型判"是不是在跟我说话"，再决定要不要受冷却约束。
    旧顺序（先冷却、后判断）会让"别人明明在跟我说话"也被一起静音。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_chime_cooldown_seconds=300,
                                group_chime_daily_max=12))
        b._send_reply = lambda text, peer, **kw: b.channel.texts.append((text, peer))
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class ChimeAgent:
            def __init__(self):
                self.decisions = 0
                self.last_chime_meta = {"to_me": False, "worth": 7.0}
                import sqlite3
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row
                self.conn.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, role_id TEXT,"
                    " sender TEXT, content TEXT, meta TEXT, created_at REAL, speaker TEXT, speaker_role TEXT)")
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','user','[AW] 今天服务器又炸了','{}',1.0)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "接一句"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        # 上次插嘴在 50s，现在 200s（冷却未过）：照判，但不发
        b._group_watch[room] = {"msgs": 2, "oh": 2, "mention": 0,
                                "last_ts": 100.0, "last_chime": 50.0,
                                "day": 0, "chimes": 0}
        assert b.chime_check(room, now=200), "冷却也要先判断（判断不等于发）"
        assert ca.decisions == 1
        assert not b.channel.texts, "自发插话在冷却期不许发"
        # 冷却过了（410-50>300）且话题还热：判完就发
        b._group_watch[room]["oh"] = 2
        b._group_watch[room]["last_ts"] = 400.0
        assert b.chime_check(room, now=410)
        assert ca.decisions == 2
        assert any(t[1] == room and "接一句" in t[0] for t in b.channel.texts)
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_retry_look_again_refetches_last_media():
    """'再看看'：没有新图时自动重取该会话最近媒体（落盘竞争的一线生机）。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        from test_wechat_bridge import FakeWcf as _FW

        class RetryWcf(_FW):
            refetched = 0

            def last_media_msg(self, room):
                class M:
                    type = 3
                    id = "10"
                    content = "<msg><img aeskey='k'/></msg>"
                    roomid = room
                return M()

            def fetch_image(self, msg):
                RetryWcf.refetched += 1
                p = os.path.join(self.media, "refetch.jpg")
                open(p, "wb").write(b"J" * 300)
                return p

        cfg = _mcfg()
        media = os.path.join(tmp, "media")
        os.makedirs(media, exist_ok=True)
        b = WeChatBridge(cfg, RetryWcf(media), FakeAPI(),
                         BridgeState(os.path.join(tmp, "s.json")), media_dir=media,
                         log=lambda m: None)
        b.async_dispatch = False
        b.state.owner_wxid = OWNER
        b.handle_message(FakeMsg(content="你再看看呢", sender=OWNER))
        assert RetryWcf.refetched == 1, "喊'再看看'必须重取最近媒体"
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)



def test_at_me_lazy_reread_rescues_race():
    """@ 竞争：轮询时 source 未写（at_me=False），处理时回读 source 重判——
    家族群实测群友 @阿满 三连全被当旁听（冷暴力）。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        from test_wechat_bridge import FakeWcf as _FW

        class RaceWcf(_FW):
            def _reread_col(self, msg, col="message_content"):
                return ("<msgsource><atuserlist><![CDATA[wxid_bot]]>"
                        "</atuserlist></msgsource>") if col == "source" else ""

            @staticmethod
            def _at_me(source, self_wxid):
                import re as _re
                m = _re.search(r"atuserlist>(?:<!\[CDATA\[)?([^<]+?)", source)
                return bool(m and self_wxid in m.group(1))

        cfg = _mcfg(groups_enabled=True)
        media = os.path.join(tmp, "media")
        os.makedirs(media, exist_ok=True)
        b = WeChatBridge(cfg, RaceWcf(media), FakeAPI(),
                         BridgeState(os.path.join(tmp, "s.json")), media_dir=media,
                         log=lambda m: None)
        b.async_dispatch = False
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class GroupMsg(FakeMsg):
            def __init__(self, content):
                super().__init__(content=content, sender=FRIEND, roomid=room)

            def from_group(self):
                return True

        m = GroupMsg("wxid_friend: @阿满 认真看")
        m.at_me = False   # 轮询瞬间 source 未写 -> at_me 判 False
        b.handle_message(m)
        assert room in b.pool.agents, "回读重判后 @ 必须命中（而不是旁听）"
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)

if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_wechat_multi"))
