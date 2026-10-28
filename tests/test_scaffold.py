"""P0 验收：配置加载与校验、SQLite 初始化、mock LLM 无 key 可渲染。"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import copy_repo_configs, make_tmpdir, repo_config_dir, rmtree, run_tests

from src.config import (AppConfig, ConfigError, config_from_dict, load_config,
                        load_persona, parse_quiet_hours, persona_from_dict)
from src.llm_client import MockLLMClient, create_client, render_prompt_text
from src.storage import SCHEMA_VERSION, get_setting, init_db, set_setting
from src import yamlite


def _sample_prompt():
    return {
        "身份摘要": "我是小满，陪伴型 AI。",
        "核心不变量": ["不用客服腔"],
        "表达风格": {"口头禅": ["嗯哼", "诶"], "语气": "轻松", "禁忌": ["亲"]},
        "当前状态": {"数值": {"energy": 0.6}, "描述": "状态平稳", "基调": "平和"},
        "关系状态": "亲近度中等",
        "相关记忆": [{"id": 1, "content": "用户想学冲浪", "salience": 0.8, "type": "intent"}],
        "未完成意图": [{"id": 1, "content": "想学冲浪"}],
        "主动权限": "L2",
        "最近主动效果": "上次主动得到了正面回应",
        "最近对方消息": "最近在忙一个项目",
        "本次任务": {"意图": "remind_open_intent", "来源": "intent:1",
                     "指令": "只渲染以下意图，不要自行决定主动"},
        "输出格式": "自然语言回复；若为 L0 则输出空或微反应",
    }


def test_yamlite_parses_repo_configs():
    cfg = yamlite.loads(open(os.path.join(repo_config_dir(), "config.example.yaml"), encoding="utf-8").read())
    assert cfg["chaos"]["seed_tau"] == 300
    assert cfg["motivation"]["weights"]["social"] == 0.35
    assert cfg["gate"]["thresholds"]["L2"] == 0.35
    assert isinstance(cfg["gate"]["scene_whitelist"], list)
    persona = yamlite.loads(open(os.path.join(repo_config_dir(), "persona.example.yaml"), encoding="utf-8").read())
    assert persona["role_id"] == "xiaoman"
    assert isinstance(persona["core_invariants"], list) and len(persona["core_invariants"]) >= 3
    assert isinstance(persona["attractor"], list) and persona["attractor"][0]["state_name"] == "energy"
    assert persona["baseline_state"]["energy"] == 0.6
    assert persona["quiet_hours"] == ["23:00-08:00"]


def test_config_load_and_validate():
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.provider == "mock"
    assert cfg.chaos.epsilon_base == 0.05
    assert cfg.convergence.alpha == 0.15
    assert cfg.motivation.weights["social"] == 0.35
    persona = load_persona(os.path.join(repo_config_dir(), "persona.example.yaml"))
    assert persona.role_id == "xiaoman"
    assert len(persona.core_invariants) == 5
    assert persona.style.get("chat_rules"), "聊天习惯（真人感规则）不能为空"
    assert persona.active_permissions["L3"] is False
    assert {e.state_name for e in persona.attractor} == {
        "energy", "social_need", "curiosity", "care", "expression", "arousal", "valence"}


def test_config_rejects_bad_persona():
    base = open(os.path.join(repo_config_dir(), "persona.example.yaml"), encoding="utf-8").read()
    # 身份摘要超 200 字
    import re as _re
    long_identity = _re.sub(r"identity_summary: .*", "identity_summary: " + "长" * 201,
                            base, count=1)
    try:
        persona_from_dict(yamlite.loads(long_identity))
        raise AssertionError("超长 identity_summary 应校验失败")
    except ConfigError as e:
        assert "identity_summary" in str(e)
    # attractor 未知维度
    bad = base.replace("state_name: energy", "state_name: hp")
    try:
        persona_from_dict(yamlite.loads(bad))
        raise AssertionError("未知维度应校验失败")
    except ConfigError as e:
        assert "hp" in str(e)
    # 未知配置项
    try:
        config_from_dict({"llm": {"providerx": 1}})
        raise AssertionError("未知配置项应校验失败")
    except ConfigError as e:
        assert "llm.providerx" in str(e)


def test_quiet_hours_span():
    assert parse_quiet_hours(["23:00-08:00"], (23, 30)) is True
    assert parse_quiet_hours(["23:00-08:00"], (3, 0)) is True
    assert parse_quiet_hours(["23:00-08:00"], (12, 0)) is False
    assert parse_quiet_hours(["12:00-13:30"], (13, 0)) is True
    assert parse_quiet_hours([], (2, 0)) is False


def test_db_init_and_settings():
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "agent.db"))
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"settings", "states", "memories", "active_logs", "messages"} <= tables
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        set_setting(conn, "proactive_enabled", True)
        assert get_setting(conn, "proactive_enabled") is True
        # 重复 init 幂等
        conn2 = init_db(os.path.join(tmp, "agent.db"))
        conn2.execute("SELECT 1")
        conn.close()
        conn2.close()
    finally:
        rmtree(tmp)


def test_mock_llm_offline_render():
    tmp = make_tmpdir()
    try:
        cfg = load_config(os.path.join(copy_repo_configs(tmp), "config.yaml"))
        persona = load_persona(os.path.join(tmp, "config", "persona.yaml"))
        client = create_client(cfg.llm, persona.style)
        assert isinstance(client, MockLLMClient)
        prompt = _sample_prompt()
        a = client.render(prompt, seed=42)
        b = client.render(prompt, seed=42)
        assert a == b and a, "同 seed 的 mock 渲染必须确定且非空"
        c = client.render(prompt, seed=43)
        # 43 可能落在不同模板/口头禅上，仅要求确定性
        assert c == client.render(prompt, seed=43)
        # 意图渲染：必须引用给定的记忆/意图来源，不凭空捏造
        share_prompt = dict(prompt)
        share_prompt["本次任务"] = {"意图": "share_related_memory", "来源": "memory:1",
                                    "指令": "只渲染该意图"}
        text = client.render(share_prompt, seed=7)
        assert "[M1]" in text and "冲浪" in text
        silence_prompt = dict(prompt)
        silence_prompt["本次任务"] = {"意图": "silence", "来源": None, "指令": "保持沉默"}
        assert client.render(silence_prompt, seed=7) == ""
        # 低精力时的语气标记
        low_energy = dict(prompt)
        low_energy["当前状态"] = {"数值": {"energy": 0.2}, "描述": "低电量", "基调": "疲惫"}
        low = client.render(low_energy, seed=7)
        assert low.startswith("（")
        # system 文本渲染
        sys_text = render_prompt_text(prompt)
        assert "【身份摘要】" in sys_text and "【本次任务】" in sys_text
    finally:
        rmtree(tmp)


def test_openai_client_message_modes():
    from src.config import LLMConfig
    from src.llm_client import OpenAICompatibleClient
    prompt = _sample_prompt()
    two = OpenAICompatibleClient(LLMConfig()).build_messages(prompt)
    assert [m["role"] for m in two] == ["system", "user"]
    assert "【身份摘要】" in two[0]["content"]
    one = OpenAICompatibleClient(LLMConfig(prompt_in_user=True)).build_messages(prompt)
    assert len(one) == 1 and one[0]["role"] == "user"
    assert "【身份摘要】" in one[0]["content"] and "直接输出回复内容" in one[0]["content"]
    # 配置文件可声明该模式
    cfg = config_from_dict({"llm": {"provider": "openai_compatible", "prompt_in_user": True}})
    assert cfg.llm.prompt_in_user is True


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_scaffold"))
