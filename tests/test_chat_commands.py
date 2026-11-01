"""聊天命令验收：解析/权限/校验/持久化/审计回滚/桥级覆盖恢复。

全部离线：Agent 用 mock 渲染，桥用 Duck 类型假件，不碰真实微信与 API。
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, repo_config_dir, rmtree, run_tests

from src import chat_commands as cc
from src.config import load_config
from src.main import Agent
from src.storage import get_setting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeState:
    def __init__(self):
        self.overrides = {}
        self.saves = 0

    def save(self):
        self.saves += 1


class FakeBridge:
    def __init__(self, cfg, state=None, names=None):
        self.cfg = cfg
        self.state = state or FakeState()
        self.names = dict(names or {})   # room id -> 群名
        self.pool = None

    def resolve_chat(self, token):
        t = str(token or "").strip()
        for wid, name in self.names.items():
            if t == name or (t and t in name):
                return wid
        return None

    def chat_label(self, peer):
        return self.names.get(str(peer), str(peer))


def _agent(tmp) -> Agent:
    return Agent(config_dir=repo_config_dir(),
                 db_path=os.path.join(tmp, "agent.db"))


def _ctx(agent, bridge, peer="wxid_owner", group=False, owner=True):
    return cc.Ctx(agent, bridge, peer, is_group=group, is_owner=owner)


def test_parse_commands_and_normalization():
    assert cc.parse("/help") == ("help", [])
    assert cc.parse("／set th2 0.3") == ("set", ["th2", "0.3"])   # 全角斜杠
    assert cc.parse("@小满" + chr(0x2005) + "/off chime") == ("off", ["chime"])
    assert cc.parse("/帮助") == ("help", [])
    assert cc.parse("/set th2 0.3") == ("set", ["th2", "0.3"])
    # 不是命令：普通文本、分数、未知命令、纯@
    assert cc.parse("你好") is None
    assert cc.parse("1/2") is None
    assert cc.parse("/frobnicate") is None
    assert cc.parse("@小满") is None


def test_set_persist_audit_and_undo():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        bridge = FakeBridge(agent.config)
        ctx = _ctx(agent, bridge)
        old = agent.config.gate.thresholds["L2"]
        r = cc.handle("/set th2 0.3", ctx)
        assert "→ 0.3" in r and "✓" in r
        assert abs(agent.config.gate.thresholds["L2"] - 0.3) < 1e-9
        # 持久化：同库新建 Agent 仍是 0.3
        agent2 = _agent(tmp)
        assert abs(agent2.config.gate.thresholds["L2"] - 0.3) < 1e-9
        # 审计 + 回滚
        assert "th2" in cc.handle("/audit", ctx)
        r2 = cc.handle("/undo", ctx)
        assert "th2" in r2 and "0.3" in r2
        assert abs(agent.config.gate.thresholds["L2"] - old) < 1e-9
    finally:
        rmtree(tmp)


def test_invalid_value_keeps_config():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        ctx = _ctx(agent, FakeBridge(agent.config))
        old = agent.config.gate.thresholds["L2"]
        r = cc.handle("/set th2 9", ctx)
        assert "没改成" in r and "0–1" in r
        assert agent.config.gate.thresholds["L2"] == old
        r = cc.handle("/set nope 1", ctx)
        assert "没有 nope" in r
        r = cc.handle("/set chime maybe", ctx)
        assert "on / off" in r
        # 非法命令不产生审计
        assert "还没有改过" in cc.handle("/audit", ctx)
        assert "没有可撤销" in cc.handle("/undo", ctx)
    finally:
        rmtree(tmp)


def test_switches_levels_and_status():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        ctx = _ctx(agent, FakeBridge(agent.config))
        assert "关 ✓" in cc.handle("/off perception", ctx)
        assert agent.config.perception.enabled is False
        assert "感知 关" in cc.handle("/status", ctx)
        assert "开 ✓" in cc.handle("/on perception", ctx)
        assert agent.config.perception.enabled is True
        cc.handle("/off l2", ctx)
        assert (get_setting(agent.conn, "proactive_disabled_levels") or {})["L2"] is True
        assert "L2✗" in cc.handle("/status", ctx)
        # 非开关项走 /on 会被拒绝
        assert "不是开关" in cc.handle("/on th2", ctx)
    finally:
        rmtree(tmp)


def test_quiet_and_roast():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        ctx = _ctx(agent, FakeBridge(agent.config))
        assert "✓" in cc.handle("/quiet 23:00-08:00", ctx)
        assert agent.persona.quiet_hours == ["23:00-08:00"]
        assert "没改成" in cc.handle("/quiet 25:00-08:00", ctx)
        assert agent.persona.quiet_hours == ["23:00-08:00"]
        cc.handle("/quiet off", ctx)
        assert agent.persona.quiet_hours == []
        cc.handle("/set roast on", ctx)
        assert agent.persona.style.get("roast_level") == "on"
        # roast 落库并能回读
        assert _agent(tmp).persona.style.get("roast_level") == "on"
        assert "只能 off / mild / on" in cc.handle("/set roast 狠", ctx)
    finally:
        rmtree(tmp)


def test_model_private_only_and_group_denial():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        bridge = FakeBridge(agent.config)
        group_ctx = _ctx(agent, bridge, peer="xxx@chatroom", group=True)
        assert "私聊" in cc.handle("/set model test-model", group_ctx)
        assert agent.config.llm.model != "test-model"
        r = cc.handle("/set model test-model", _ctx(agent, bridge))
        assert "→ test-model" in r
        assert agent.config.llm.model == "test-model"
        assert "test-model" in cc.handle("/model", _ctx(agent, bridge))
    finally:
        rmtree(tmp)


def test_bridge_params_persist_and_restore():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        bridge = FakeBridge(agent.config)
        ctx = _ctx(agent, bridge)
        assert "关 ✓" in cc.handle("/off chime", ctx)
        assert agent.config.wechat.group_chime is False
        assert bridge.state.overrides["group_chime"] is False
        assert bridge.state.saves >= 1
        cc.handle("/set voice always", ctx)
        assert bridge.state.overrides["voice_reply"] == "always"
        # 重启：新 cfg 读回覆盖
        cfg2 = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
        applied = cc.apply_bridge_overrides(cfg2, bridge.state.overrides)
        assert "chime" in applied and "voice" in applied
        assert cfg2.wechat.group_chime is False
        assert cfg2.wechat.voice_reply == "always"
        # 非法覆盖值不会炸启动
        assert cc.apply_bridge_overrides(cfg2, {"voice_reply": "bogus"}) == []
    finally:
        rmtree(tmp)


def test_chat_scope_toggle():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        bridge = FakeBridge(agent.config)
        agent.config.wechat.enabled_chats = ["other@chatroom"]
        agent.config.wechat.disabled_chats = []
        ctx = _ctx(agent, bridge, peer="xxx@chatroom", group=True)
        assert "已停用" in cc.handle("/chat", ctx)
        assert "开 ✓" in cc.handle("/chat on", ctx)
        assert "xxx@chatroom" in agent.config.wechat.enabled_chats
        assert "xxx@chatroom" not in agent.config.wechat.disabled_chats
        assert "关 ✓" in cc.handle("/chat off", ctx)
        assert "xxx@chatroom" in agent.config.wechat.disabled_chats
        assert cc._chat_enabled(ctx) is False
        # 白名单只剩自己时关掉：不能因为列表被删空变成"其余全部启用"
        agent.config.wechat.enabled_chats = ["xxx@chatroom"]
        cc.handle("/chat off", ctx)
        assert cc._chat_enabled(ctx) is False
        assert agent.config.wechat.enabled_chats == ["xxx@chatroom"]
    finally:
        rmtree(tmp)


def test_chat_target_by_name_audit_and_undo():
    """在别的会话里按群名开关（/chat AI技术冲了 on），审计记被改的会话，/undo 能回滚它。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        target = "45729355366@chatroom"
        bridge = FakeBridge(agent.config, names={target: "AI技术冲了"})
        agent.config.wechat.enabled_chats = ["50107328065@chatroom"]
        agent.config.wechat.disabled_chats = []
        ctx = _ctx(agent, bridge, peer="50107328065@chatroom", group=True)
        r = cc.handle("/chat AI技术冲了 on", ctx)
        assert "AI技术冲了 开 ✓" in r
        assert target in agent.config.wechat.enabled_chats
        row = agent.conn.execute(
            "SELECT room, name FROM config_audit ORDER BY id DESC LIMIT 1").fetchone()
        assert row["room"] == target and row["name"] == "chat"
        # undo 作用到目标会话，而不是发命令的这个群
        r2 = cc.handle("/undo", ctx)
        assert "AI技术冲了" in r2
        assert cc._chat_enabled(cc._sub_ctx(ctx, target)) is False
        # 名字找不到时给提示，不乱改
        r3 = cc.handle("/chat 不存在的群 on", ctx)
        assert "没找到" in r3
    finally:
        rmtree(tmp)


def test_chat_list_known_chats():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        bridge = FakeBridge(agent.config, names={
            "xxx@chatroom": "鱼🐟",
            "45729355366@chatroom": "AI技术冲了",
        })
        bridge.pool = types.SimpleNamespace(agents={
            "xxx@chatroom": types.SimpleNamespace(group_name="鱼🐟"),
            "45729355366@chatroom": types.SimpleNamespace(group_name="AI技术冲了"),
        })
        agent.config.wechat.enabled_chats = ["xxx@chatroom"]
        ctx = _ctx(agent, bridge, peer="xxx@chatroom", group=True)
        r = cc.handle("/chat list", ctx)
        assert "本会话 开" in r and "AI技术冲了 关" in r
    finally:
        rmtree(tmp)


def test_non_owner_commands_ignored():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        ctx = _ctx(agent, FakeBridge(agent.config), owner=False)
        old = agent.config.gate.thresholds["L2"]
        assert cc.handle("/set th2 0.3", ctx) == ""
        assert agent.config.gate.thresholds["L2"] == old
    finally:
        rmtree(tmp)


def test_command_reply_has_no_secret_fields():
    """命令回执里不出现 key/token 这类字段（今后加 /key 也别回显）。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        ctx = _ctx(agent, FakeBridge(agent.config))
        for cmd in ("/help", "/status", "/audit"):
            reply = cc.handle(cmd, ctx)
            low = reply.lower()
            assert "api_key" not in low and "token" not in low


    finally:
        rmtree(tmp)


def test_single_user_mode_agent_params_refused():
    """单用户模式没有本地 Agent：桥级能改，会话级要明确拒绝。"""
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    bridge = FakeBridge(cfg)
    ctx = cc.Ctx(None, bridge, "wxid_owner", is_group=False, is_owner=True)
    assert "单用户" in cc.handle("/set th2 0.3", ctx)
    assert "关 ✓" in cc.handle("/off chime", ctx)
    assert cfg.wechat.group_chime is False


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_chat_commands"))
