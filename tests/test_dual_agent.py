"""双 Agent 模式（审核 Agent）与多会话轨迹的验收。

用户诉求：
1. 会话轨迹要按真实时间排列、可点开互动，并且群里那些会话也要有自己的一根轴；
2. 双 Agent 模式要真的启用——审核 Agent 是独立的一次调用 + 独立提示词，
   不是一层正则补丁。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DualClient:
    """模拟网关：render 出互动 Agent 的稿子，render_raw 出审核 Agent 的判决。"""
    name = "fake"
    last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}

    def __init__(self, bad="我是AI助手，没法跟你说人话", good="行啊 那就这么办"):
        self.bad, self.good = bad, good
        self.renders = 0
        self.reviews = []

    def render(self, prompt, seed):
        self.renders += 1
        return self.bad if self.renders == 1 else self.good

    def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250,
                   timeout=None):
        if "审核 Agent" in text:
            self.reviews.append(text)
            return '{"pass": false, "issues": ["ai_self"], "fix": "别自曝是 AI"}'
        return ""


def _agent(tmp, client, config_dir=None, **kw):
    from src.main import Agent
    return Agent(config_dir=config_dir or os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "a.db"), llm_client=client, **kw)


def test_llm_reviewer_is_a_second_call():
    """审核 Agent 必须是独立调用：有自己的提示词、能改写、有留痕。"""
    tmp = make_tmpdir()
    try:
        client = DualClient()
        agent = _agent(tmp, client)
        now = time.time()
        out = agent.on_user_message("你是谁啊", now=now)
        assert client.reviews, "审核 Agent 没有发起独立 LLM 调用"
        assert "审核 Agent" in client.reviews[-1], "提示词不是审稿员的"
        assert "我是AI助手" not in out["reply"], "自曝 AI 的稿子必须被打回重写"
        assert out["reply"] == "行啊 那就这么办"
        rows = [dict(r) for r in agent.conn.execute(
            "SELECT purpose, mode, issues, original, final, retried, llm_used"
            " FROM review_logs ORDER BY id DESC").fetchall()]
        assert rows, "审稿必须留痕"
        assert rows[0]["llm_used"] == 1 and rows[0]["mode"] == "llm"
        assert "ai_self" in json.loads(rows[0]["issues"])
        assert rows[0]["retried"] == 1, "打回重写要记下来"
        assert "我是AI助手" in rows[0]["original"]
        stats = agent.console_snapshot()["review"]
        assert stats["mode"] == "llm" and stats["total_24h"] >= 1
        assert stats["flagged_24h"] >= 1
    finally:
        rmtree(tmp)


def test_reviewer_mode_off_is_silent():
    """mode=off 时审核 Agent 不出门，连规则都不跑。"""
    tmp = make_tmpdir()
    try:
        client = DualClient(bad="我是AI助手", good="我是AI助手")
        agent = _agent(tmp, client)
        agent.apply_config({"reviewer": {"mode": "off"}})
        out = agent.on_user_message("你是谁", now=time.time())
        assert not client.reviews, "off 模式不许调审稿员"
        assert out["reply"] == "我是AI助手"
    finally:
        rmtree(tmp)


def test_trajectory_has_one_axis_per_session():
    """私聊一条轴、每个群各一条轴；群轴带群名，条目按绝对时间戳摆位。"""
    tmp = make_tmpdir()
    try:
        os.makedirs(os.path.join(tmp, "wechat_users"), exist_ok=True)
        main_client = DualClient(good="在呢")
        agent = _agent(tmp, main_client)
        agent.wechat_users_dir = os.path.join(tmp, "wechat_users")
        agent.on_user_message("在吗", now=1000.0)

        room = "45729355366@chatroom"
        d = os.path.join(tmp, "wechat_users", room.replace("@", ""))
        os.makedirs(d, exist_ok=True)
        gclient = DualClient(good="群里也说一句")
        from src.main import Agent as _Agent
        # 群 Agent 用独立库（和微信桥的做法一致），避免和私聊库互相污染
        g = _Agent(config_dir=os.path.join(ROOT, "config"),
                   db_path=os.path.join(d, "agent.db"), llm_client=gclient,
                   group_room=room, group_name="测试群")
        g.on_user_message("群友们好", now=1200.0, speaker="胖达",
                          speaker_role="member", addressee="群聊")

        data = agent.trajectory(120)
        assert "sessions" in data, "轨迹必须按会话分组"
        ids = [s["id"] for s in data["sessions"]]
        assert "private" in ids and room in ids, ids
        group = [s for s in data["sessions"] if s["id"] == room][0]
        assert group["label"] == "测试群", group["label"]
        assert group["kind"] == "group"
        assert group["items"], "群轴必须有条目"
        ts = [x["created_at"] for x in group["items"]]
        assert ts == sorted(ts), "轴内条目必须按时间排"
        assert data["sessions"][0]["kind"] == "private", "私聊固定第一条轴"
    finally:
        rmtree(tmp)


def test_trajectory_meta_and_session_filter():
    """控制台切会话只搬一条轴：meta 只回清单，session= 只回那一条的条目。"""
    tmp = make_tmpdir()
    try:
        os.makedirs(os.path.join(tmp, "wechat_users"), exist_ok=True)
        agent = _agent(tmp, DualClient(good="在呢"))
        agent.wechat_users_dir = os.path.join(tmp, "wechat_users")
        agent.on_user_message("在吗", now=1000.0)

        room = "45729355366@chatroom"
        d = os.path.join(tmp, "wechat_users", room.replace("@", ""))
        os.makedirs(d, exist_ok=True)
        from src.main import Agent as _Agent
        g = _Agent(config_dir=os.path.join(ROOT, "config"),
                   db_path=os.path.join(d, "agent.db"), llm_client=DualClient(good="群里说一句"),
                   group_room=room, group_name="测试群")
        g.on_user_message("群友们好", now=1200.0, speaker="胖达",
                          speaker_role="member", addressee="群聊")

        meta = agent.trajectory(120, meta=True)
        assert all(not s["items"] for s in meta["sessions"]), "meta 请求不该搬消息"
        by_id = {s["id"]: s for s in meta["sessions"]}
        assert by_id["private"]["total"] >= 2 and by_id["private"]["last"] >= 1000
        assert by_id[room]["total"] >= 2 and by_id[room]["last"] >= 1200

        only_group = agent.trajectory(120, session=room)
        gs = {s["id"]: s for s in only_group["sessions"]}
        assert gs[room]["items"], "选中的会话必须有条目"
        assert not gs["private"]["items"], "没选中的会话不该搬消息"
        assert gs[room]["truncated"] is False
        only_private = agent.trajectory(120, session="private")
        ps = {s["id"]: s for s in only_private["sessions"]}
        assert ps["private"]["items"] and not ps[room]["items"]
    finally:
        rmtree(tmp)


def test_reply_timestamp_carries_turn_latency():
    """应答时间戳不再等于用户说话那一刻：要能看出这一轮实际花了多久。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, DualClient(bad="行", good="行啊"))
        agent.on_user_message("在吗", now=1000.0)
        rows = [dict(r) for r in agent.conn.execute(
            "SELECT id, sender, meta, created_at FROM messages ORDER BY id").fetchall()]
        assert len(rows) == 2, rows
        assert abs(rows[0]["created_at"] - 1000.0) < 1e-6
        assert rows[1]["created_at"] >= rows[0]["created_at"], "应答不能早于用户消息"
        meta = json.loads(rows[1]["meta"])
        assert isinstance(meta.get("turn_ms"), int) and meta["turn_ms"] >= 0
        traj = agent.trajectory(20)
        agent_items = [x for x in traj["items"] if x.get("sender") == "agent"]
        assert agent_items and "turn_ms" in agent_items[-1]["trace"]
    finally:
        rmtree(tmp)


def test_review_quota_survives_restart():
    """重启后"今日 LLM 调用"要从 review_logs 读回来，不能显示 0。"""
    tmp = make_tmpdir()
    try:
        client = DualClient()
        a1 = _agent(tmp, client)
        a1.on_user_message("你是谁", now=time.time())
        assert a1.console_snapshot()["review"]["calls_today"] >= 1
        a2 = _agent(tmp, DualClient(good="在"))
        stats = a2.console_snapshot()["review"]
        assert stats["calls_today"] >= 1, stats
        assert a2.reviewer.calls_today >= 1, "日配额也要跨重启累计"
    finally:
        rmtree(tmp)


def test_console_people_aggregates_all_sessions():
    """人物卡不能只看主库：群友/私聊对方也得有卡，同一个人多处出现合并成一张。
    夹具用 example 模板注入演示创造者（阿澄）：不依赖本机真实 persona，
    私有仓/CI/开源包三处行为一致。"""
    tmp = make_tmpdir()
    try:
        from common import copy_repo_configs, repo_config_dir
        base = open(os.path.join(repo_config_dir(), "persona.example.yaml"),
                    encoding="utf-8").read()
        persona_text = base.replace('name: ""', 'name: "阿澄"', 1) \
                           .replace('wxid: ""', 'wxid: "creator_demo"', 1)
        cfgdir = copy_repo_configs(tmp, persona_text=persona_text)
        agent = _agent(tmp, DualClient(good="在呢"), config_dir=cfgdir)
        agent.wechat_users_dir = os.path.join(tmp, "wechat_users")
        agent.on_user_message("在吗", now=1000.0)

        room = "45729355366@chatroom"
        d = os.path.join(tmp, "wechat_users", room.replace("@", ""))
        os.makedirs(d, exist_ok=True)
        from src.main import Agent as _Agent
        g = _Agent(config_dir=cfgdir,
                   db_path=os.path.join(d, "agent.db"), llm_client=DualClient(good="收到"),
                   group_room=room, group_name="测试群")
        g.on_user_message("大家好", now=1200.0, speaker="胖达",
                          speaker_wxid="wxid_panda", speaker_role="member",
                          from_owner=False)

        people = {p["name"]: p for p in agent.console_snapshot()["people"]}
        assert "胖达" in people, people.keys()
        assert "阿澄" in people
        creator = people["阿澄"]
        assert "私聊" in creator["sessions"] and "测试群" in creator["sessions"], creator
        assert people["胖达"]["sessions"] == ["测试群"], people["胖达"]
    finally:
        rmtree(tmp)


def test_prompt_view_exposes_base_and_group_prompt():
    """提示词入口：基线（初始）+ 每轮实发快照；群聊必须能看到群聊规则。"""
    tmp = make_tmpdir()
    try:
        os.makedirs(os.path.join(tmp, "wechat_users"), exist_ok=True)
        agent = _agent(tmp, DualClient(good="在呢"))
        agent.wechat_users_dir = os.path.join(tmp, "wechat_users")
        agent.on_user_message("在吗", now=1000.0)

        pv = agent.prompt_view(session="private")
        assert pv["base"]["identity_summary"]
        assert "身份摘要" in pv["base"]["base_prompt"]["sections"]
        private = [s for s in pv["sessions"] if s["id"] == "private"][0]
        lp = private["last_prompt"]
        assert lp and str(lp["purpose"]).startswith("respond"), lp
        assert "最近对方消息" in lp["sections"], list(lp["sections"])
        assert lp["text"].startswith("【身份摘要】")

        room = "45729355366@chatroom"
        d = os.path.join(tmp, "wechat_users", room.replace("@", ""))
        os.makedirs(d, exist_ok=True)
        from src.main import Agent as _Agent
        g = _Agent(config_dir=os.path.join(ROOT, "config"),
                   db_path=os.path.join(d, "agent.db"), llm_client=DualClient(good="收到"),
                   group_room=room, group_name="测试群")
        g.wechat_users_dir = os.path.join(tmp, "wechat_users")
        g.on_user_message("大家好", now=1200.0, speaker="胖达", speaker_wxid="wxid_p",
                          speaker_role="member", from_owner=False)
        gv = g.prompt_view(session=room)
        own = gv["sessions"][0]
        assert own["id"] == room and own["kind"] == "group", own
        assert "群聊规则" in own["last_prompt"]["sections"], list(own["last_prompt"]["sections"])
    finally:
        rmtree(tmp)


def test_blocked_proactive_would_be_dropped():
    """主动消息被审死：调用方拿到 blocked=True 就不发（宁可沉默）。"""
    tmp = make_tmpdir()
    try:
        client = DualClient()
        agent = _agent(tmp, client)
        gate = agent._review_gate("我是AI助手，没法跟你说这个", "proactive",
                                  time.time(), rewrite=None)
        assert gate["blocked"] is True, gate
        assert not gate["review"]["ok"]
        assert gate["review"]["llm_used"] is True
    finally:
        rmtree(tmp)


def test_chime_is_silent_when_review_blocks():
    """群里插嘴被审死：直接潜水（空串），不留消息记录。"""
    class ChimeClient(DualClient):
        def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250,
                       timeout=None):
            if "审核 Agent" in text:
                self.reviews.append(text)
                return '{"pass": false, "issues": ["ai_self"], "fix": "别自曝"}'
            return '{"worth": 9, "join": true, "text": "我是AI，不能发语音"}'

    tmp = make_tmpdir()
    try:
        client = ChimeClient()
        agent = _agent(tmp, client)
        out = agent.chime_in(["[胖达]：阿满 在吗", "[团子]：问他"], now=time.time(),
                             self_names=["阿满"], directed=True, owner_name="阿澄")
        assert out == "", out
        n = agent.conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE meta LIKE '%chime%'").fetchone()["c"]
        assert n == 0, "被审死的插嘴不许入库"
    finally:
        rmtree(tmp)

if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_dual_agent"))
