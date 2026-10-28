"""设置面板验收：运行时热改 + 持久化 + HTTP 路由。
全部离线（llm 切 mock），不依赖真实 API。"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.llm_client import MockLLMClient
from src.main import Agent
from src.server import make_server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _at(hour: int, minute: int = 0) -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, minute, 0, 0, 0, -1))


def _agent(tmp) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"))


def test_style_override_hot_and_persisted():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        view = agent.apply_settings({"persona_style": {
            "catchphrases": ["哦嚯"], "tone": "测试语气",
            "chat_rules": ["第一条测试规则"]}})
        assert view["persona_style"]["catchphrases"] == ["哦嚯"]
        # 立即生效：Prompt 带新口头禅
        row = agent.engine.new_row(0)
        prompt = agent.builder.build(
            intent="respond", source=None, state_vector=row["state_vector"],
            relation_state=row["relation_state"], memories=[], open_intents=[],
            permission_level="L2", last_active_effect="x")
        assert "哦嚯" in str(prompt["表达风格"]) and "第一条测试规则" in str(prompt["表达风格"])
        # 持久化：同库新建 Agent 仍生效
        agent2 = Agent(config_dir=os.path.join(ROOT, "config"),
                       db_path=os.path.join(tmp, "agent.db"))
        assert agent2.persona.style["catchphrases"] == ["哦嚯"]
    finally:
        rmtree(tmp)


def test_quiet_hours_and_proactive_overrides():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        view = agent.apply_settings({"quiet_hours": ["12:00-13:30"]})
        assert view["quiet_hours"] == ["12:00-13:30"]
        assert agent.persona.quiet_hours == ["12:00-13:30"]
        # 闸门按新时段拦截 L2
        from src.motivation import Candidate
        cand = Candidate("remind_open_intent", 2, 0.8, {"type": "intent", "id": 1}, {})
        d = agent.gate.decide(cand, 0.9, agent.engine.new_row(_at(12, 30)), _at(12, 30),
                              source_exists=lambda s: True)
        assert not d.allowed and d.reasons["quiet_hours"]["blocked"]
        # 总开关与分级开关
        view = agent.apply_settings({"proactive": {"enabled": False}})
        assert view["proactive"]["enabled"] is False
        view = agent.apply_settings({"proactive": {"levels": {"L2": False}}})
        assert view["proactive"]["levels"]["L2"] is False
        view = agent.apply_settings({"proactive": {"enabled": True, "levels": {"L2": True}}})
        assert view["proactive"]["levels"]["L2"] is True
    finally:
        rmtree(tmp)


def test_llm_and_perception_overrides_offline():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        view = agent.apply_settings({"llm": {"provider": "mock"}})
        assert view["llm"]["provider"] == "mock"
        assert isinstance(agent.client, MockLLMClient)
        assert agent.on_user_message("测试一句话就好", now=1000.0)["reply"]
        view = agent.apply_settings({"perception": {"enabled": False}})
        assert view["perception"]["enabled"] is False
        assert agent._perceive("anything", agent.engine.new_row(0), 123) is None
        # 非法值被拒
        try:
            agent.apply_settings({"llm": {"provider": "xxx"}})
            raise AssertionError("非法 provider 应被拒绝")
        except ValueError:
            pass
        try:
            agent.apply_settings({"quiet_hours": ["9点-18点"]})
            raise AssertionError("非法时段应被拒绝")
        except ValueError:
            pass
    finally:
        rmtree(tmp)


def test_settings_http_roundtrip():
    tmp = tempfile.mkdtemp()
    agent = Agent(config_dir=os.path.join(ROOT, "config"),
                  db_path=os.path.join(tmp, "a.db"), threaded=True)
    httpd = make_server(agent, "127.0.0.1", 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def req(method, path, body=None):
        r = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json"}, method=method)
        with urllib.request.urlopen(r, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        view = req("GET", "/settings")
        assert view["proactive"]["enabled"] in (True, False)
        assert "persona_style" in view and "cache" in view
        # PUT 热改（离线安全项）
        view2 = req("PUT", "/settings", {"persona_style": {"catchphrases": ["嘿嘿"]},
                                         "proactive": {"enabled": False}})
        assert view2["persona_style"]["catchphrases"] == ["嘿嘿"]
        assert view2["proactive"]["enabled"] is False
        # 非法请求 400
        try:
            req("PUT", "/settings", {"quiet_hours": ["bad"]})
            raise AssertionError("应 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_settings"))
