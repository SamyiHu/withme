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
        assert images and len(images) == 3 and all(i["b64"] == expect for i in images)
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


def test_group_mention_routes_and_labels():
    """群聊：被@（at_me 硬信号）才回；文本去掉 wxid 前缀与@称呼、标注说话人；
    回复发到群；群绝不绑定为「主人」。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True))
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
        # 主人没被@也回（群里主人随时可叫）
        b.handle_message(GroupMsg(content="wxid_owner: 看看", sender=OWNER, at_me=False))
        assert len([t for t in b.channel.texts if t[1] == room]) > len(texts)
        owner_row = b.pool.agents[room].conn.execute(
            "SELECT content, meta FROM messages WHERE sender='user' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert owner_row["content"] == "看看", "主人正文也不带角色前缀"
        assert json.loads(owner_row["meta"] or "{}").get("speaker_role") == "owner"
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

            def evaluate_proactive(self):
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
        for a in b.pool.agents.values():
            a.conn.close()
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_chime_gates_and_decision():
    """群智能插嘴：触发只数"未被回复的旁听"（主人的消息已回复，不攒门）；
    单条旁听+静默 6s 即问 AI；冷却期/话题冷掉都不问（不烧 token）。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                group_chime_cooldown_seconds=480,
                                group_chime_daily_max=12))
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
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','user','[AW] 今天服务器又炸了','{}',1.0)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "哈哈这梗我接一下" if self.decisions == 1 else ""

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=FRIEND, at_me=False):
                super().__init__(content=content, sender=sender, roomid=room)
                self.at_me = at_me

            def from_group(self):
                return True

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        # 主人消息（有回复）不攒插嘴门
        b.handle_message(GroupMsg("主人说话", sender=OWNER, at_me=False))
        assert not b.chime_check(room, now=time.time() + 10)
        assert ca.decisions == 0
        # 单条群友旁听：碎碎念不触发（要攒 2 条成一小轮）
        b.handle_message(GroupMsg("wxid_friend: 闲聊一句"))
        now = time.time() + 10
        assert not b.chime_check(room, now=now)
        assert ca.decisions == 0
        # 攒到 2 条旁听 + 静默 6s：问 AI，AI 接话 -> 发到群
        b.handle_message(GroupMsg("wxid_friend: 又聊一句"))
        assert b.chime_check(room, now=now + 8)
        assert ca.decisions == 1
        assert any(t[1] == room and "接一下" in t[0] for t in b.channel.texts)
        # 冷却期内再旁听再问：门控直接消化，不问 AI
        b.handle_message(GroupMsg("wxid_friend: 又聊一句"))
        b.handle_message(GroupMsg("wxid_friend: 再聊一句"))
        assert not b.chime_check(room, now=now + 30)
        assert ca.decisions == 1
        # 冷却过了但话题冷掉（>180s）：不插旧梗
        assert not b.chime_check(room, now=now + 600)
        assert ca.decisions == 1
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)




def test_voice_marker_forces_voice_reply():
    """模型按需点语音：回复以 [语音] 开头 -> 剥标记走语音链；普通回复仍走文字。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp)
        b.state.owner_wxid = OWNER
        b._tts_bytes = lambda text: b"FAKEMP3" + b"x" * 400   # 本地 TTS 假件
        import src.silk_audio as silk_mod
        orig = silk_mod.mp3_to_silk
        silk_mod.mp3_to_silk = lambda audio, out, work_dir=None: (
            open(out, "wb").write(b"#!SILK_V3xx"),
            {"duration_ms": 1500, "path": out})[1]
        try:
            # FakeWcf 走 send_voice mp3 卡片：直接看 files/voice 调用
            b.channel.chat_result = None
            class _Agent2:
                role_id = "xiaoman"
                _lock = __import__("threading").RLock()
                class conn:
                    @staticmethod
                    def execute(*a, **k):
                        class R:
                            def fetchall(s): return []
                        return R()
                def on_user_message(self, text, now=None, images=None, from_owner=True,
                        speaker="", speaker_role="", speaker_wxid="", addressee="群聊"):
                    return {"reply": "[语音] 好嘞，那我露一手", "addon": None,
                            "meta": {}}
            b.pool.agents[OWNER] = _Agent2()
            b.handle_message(FakeMsg(content="用语音回答我", sender=OWNER))
            assert b.channel.silks or b.channel.files, "[语音] 标记必须走语音链"
            texts = [t[0] for t in b.channel.texts]
            assert all(not str(x).startswith("[语音]") for x in texts), "标记必须剥掉"
        finally:
            silk_mod.mp3_to_silk = orig
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)



def test_explicit_voice_request_forces_voice():
    """代码级意图兜底：模型不吐 [语音] 标记、甚至嘴硬'只能打字'，
    只要对方明确要语音，桥强制走语音链。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp)
        b.state.owner_wxid = OWNER
        b._tts_bytes = lambda text: b"FAKEMP3" + b"x" * 400
        import src.silk_audio as silk_mod
        orig = silk_mod.mp3_to_silk
        silk_mod.mp3_to_silk = lambda audio, out, work_dir=None: (
            open(out, "wb").write(b"#!SILK_V3xx"),
            {"duration_ms": 1500, "path": out})[1]
        try:
            from test_wechat_bridge import FakeWcf

            class _Agent3:
                role_id = "xiaoman"
                _lock = __import__("threading").RLock()

                class conn:
                    @staticmethod
                    def execute(*a, **k):
                        class R:
                            def fetchall(s):
                                return []
                        return R()

                def on_user_message(self, text, now=None, images=None, from_owner=True,
                        speaker="", speaker_role="", speaker_wxid="", addressee="群聊"):
                    # 模型嘴硬版：普通文字，无标记
                    return {"reply": "我发不了语音，只能打字", "addon": None,
                            "meta": {}}
            b.pool.agents[OWNER] = _Agent3()
            b.handle_message(FakeMsg(content="这句话用语音发给我", sender=OWNER))
            assert b.channel.silks or b.channel.files, "明确要语音必须走语音链"
        finally:
            silk_mod.mp3_to_silk = orig
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)



def test_name_mention_grants_chime_eligibility():
    """叫名字（没@）= 软点名：单条旁听也要进插嘴决策，不接就是冷暴力。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        from test_wechat_bridge import FakeWcf as _FW

        class NamedWcf(_FW):
            def nickname_of(self, wxid):
                return "阿满" if wxid == "wxid_bot" else ""

        cfg = _mcfg(groups_enabled=True)
        media = os.path.join(tmp, "media")
        os.makedirs(media, exist_ok=True)
        b = WeChatBridge(cfg, NamedWcf(media), FakeAPI(),
                         BridgeState(os.path.join(tmp, "s.json")), media_dir=media,
                         log=lambda m: None)
        b.async_dispatch = False
        b.state.owner_wxid = OWNER
        room = "457@chatroom"

        class NamedAgent:
            role_id = "g"
            _lock = __import__("threading").RLock()

            class conn:
                @staticmethod
                def execute(*a, **k):
                    class R:
                        def fetchall(s):
                            return []
                    return R()

            def chime_in(self, dialogue, now=None, **kwargs):
                return "叫我了，在呢"

        class GroupMsg(FakeMsg):
            def __init__(self, content):
                super().__init__(content=content, sender=FRIEND, roomid=room)

            def from_group(self):
                return True

        GroupMsg.at_me = False
        b.pool.agents[room] = NamedAgent()
        b.handle_message(GroupMsg("wxid_friend: 阿满 在吗"))
        assert b.chime_check(room, now=time.time() + 8), "叫名字单条必须够格"
        assert any(t[1] == room and "在呢" in t[0] for t in b.channel.texts)
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)




def test_group_name_mention_bypasses_chime_cooldown():
    """群里没@但正文点名（阿满/机器人）= 硬信号：冷却期也必须进插嘴决策。

    2026-10-27 实测：阿满 这波…/机器人都不理我 这类消息被当旁听，冷却期
    连 oh 都被清零，群里只剩主人的消息有回复。点名必须接。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        from test_wechat_bridge import FakeWcf as _FW

        class NamedWcf(_FW):
            def nickname_of(self, wxid):
                return "阿满" if wxid == "wxid_bot" else ""

        cfg = _mcfg(groups_enabled=True, group_mention_only=True,
                    group_self_aliases=["小满", "机器人"],
                    group_chime_cooldown_seconds=300, group_chime_daily_max=12)
        media = os.path.join(tmp, "media")
        os.makedirs(media, exist_ok=True)
        b = WeChatBridge(cfg, NamedWcf(media), FakeAPI(),
                         BridgeState(os.path.join(tmp, "s.json")), media_dir=media,
                         log=lambda m: None)
        b.async_dispatch = False
        b._send_reply = lambda text, peer: b.channel.texts.append((text, peer))
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
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','user','[AW] 今天服务器又炸了','{}',1.0)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "叫我了，在呢"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        now = time.time()
        # 刚插过嘴（冷却中），且没有攒够两条旁听
        b._group_watch[room] = {"msgs": 1, "oh": 0, "mention": 0,
                                "last_ts": now, "last_chime": now,
                                "day": 0, "chimes": 1}

        class GroupMsg(FakeMsg):
            def __init__(self, content="", sender=FRIEND, at_me=False):
                super().__init__(content=content, sender=sender, roomid=room)
                self.at_me = at_me

            def from_group(self):
                return True

        # 昵称点名：没@，也不是主人，冷却期内 + 单条旁听
        b.handle_message(GroupMsg("wxid_friend: 阿满这波是照着剧本念的", at_me=False))
        assert b._group_watch[room].get("mention", 0) == 1, "点名要记硬资格"
        assert b.chime_check(room, now=now + 8), "点名必须绕开冷却"
        assert ca.decisions == 1
        assert any(t[1] == room and "在呢" in t[0] for t in b.channel.texts)

        # 通用别名「机器人」同样算点名（实测：机器人都不理我）
        b.handle_message(GroupMsg("wxid_friend: 机器人都不理我，还聊啥", at_me=False))
        assert b._group_watch[room].get("mention", 0) == 1
        assert b.chime_check(room, now=now + 16), "叫机器人也要接"
        assert ca.decisions == 2
        assert len([t for t in b.channel.texts if t[1] == room]) >= 2
    finally:
        if old_db:
            os.environ["AGENT_DB_PATH"] = old_db
        else:
            os.environ.pop("AGENT_DB_PATH", None)
        rmtree(tmp)


def test_group_chime_cooldown_keeps_overheard():
    """冷却期不能把旁听清零：冷却结束话题还热，攒着的两条要能接上。

    旧逻辑冷却期直接 0 掉 oh，群里安静时永远等不到下一对消息，表现为
    "只回主人"。"""
    tmp = make_tmpdir()
    old_db = os.environ.get("AGENT_DB_PATH")
    os.environ["AGENT_DB_PATH"] = os.path.join(tmp, "owner.db")
    try:
        b = _mbridge(tmp, _mcfg(groups_enabled=True,
                                 group_chime_cooldown_seconds=300,
                                 group_chime_daily_max=12))
        b._send_reply = lambda text, peer: b.channel.texts.append((text, peer))
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
                self.conn.execute(
                    "INSERT INTO messages(role_id,sender,content,meta,created_at)"
                    " VALUES('g','user','[AW] 今天服务器又炸了','{}',1.0)")
                self.role_id = "g"

            def chime_in(self, dialogue, now=None, **kwargs):
                self.decisions += 1
                return "接一句"

        ca = ChimeAgent()
        b.pool.agents[room] = ca
        # 上次插嘴在 50s，现在是 200s（冷却未过）；两条旁听已攒着
        b._group_watch[room] = {"msgs": 2, "oh": 2, "mention": 0,
                                "last_ts": 100.0, "last_chime": 50.0,
                                "day": 0, "chimes": 0}
        assert not b.chime_check(room, now=200)
        assert b._group_watch[room]["oh"] == 2, "冷却期不得清空旁听"
        # 冷却过了（410-50>300）且最后一条在 10s 内：接
        b._group_watch[room]["last_ts"] = 400.0
        assert b.chime_check(room, now=410)
        assert ca.decisions == 1
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
