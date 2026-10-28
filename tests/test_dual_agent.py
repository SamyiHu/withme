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


def _agent(tmp, client, **kw):
    from src.main import Agent
    return Agent(config_dir=os.path.join(ROOT, "config"),
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
