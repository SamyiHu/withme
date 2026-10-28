"""拟人情绪回路验收：共鸣/冲撞反馈 + 离散心情盘 + 立场存档。

用户设计：情绪多维修量化随时间回归（引擎已有），缺的是拿对方回复
对比自己的内心想法——同频获得共鸣、顶撞获得火气，反馈进下一轮。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, repo_config_dir, rmtree, run_tests

from src.config import load_config, load_persona
from src.main import Agent
from src.prompt_builder import state_to_natural_language
from src.state_engine import StateEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE = "xiaoman"


class MoodClient:
    """render_raw 可编程：perception 返回指定情绪的 JSON。"""

    name = "fake"

    def __init__(self, sentiment=0.5, emotion="开心", plan_mood="乐"):
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5,
                           "estimated": True}
        self.prompts = []
        self._sentiment = sentiment
        self._emotion = emotion
        self._plan_mood = plan_mood

    def render(self, prompt, seed):
        self.prompts.append(prompt)
        return "嗯，在呢"

    def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
        if "分析用户发给陪伴型 AI" in text:
            return json.dumps({
                "sentiment": self._sentiment, "emotion": self._emotion,
                "intent_type": "分享", "topics": ["测试"],
                "reply_plan": {"what": "接话", "how": "短", "mood": self._plan_mood}})
        return ""


def _agent(tmp, client) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"), llm_client=client)


def test_resonate_and_clash_impulses():
    tmp = make_tmpdir()
    try:
        cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
        persona = load_persona(os.path.join(repo_config_dir(), "persona.example.yaml"))
        eng = StateEngine(cfg, persona)
        row = {"state_vector": dict(persona.baseline_state)}
        v0 = row["state_vector"]["valence"]
        row = eng.apply_event_impulse(row, "resonate")
        assert row["state_vector"]["valence"] > v0, "共鸣要涨心情"
        row2 = eng.apply_event_impulse({"state_vector": dict(persona.baseline_state)}, "clash")
        assert row2["state_vector"]["valence"] < persona.baseline_state["valence"]
        assert row2["state_vector"]["arousal"] > persona.baseline_state["arousal"], \
            "冲撞要来火气（唤醒度上）"
        row3 = eng.apply_event_impulse({"state_vector": dict(persona.baseline_state)},
                                       "clash_streak")
        assert row3["state_vector"]["valence"] < row2["state_vector"]["valence"], \
            "连撞更冲"
    finally:
        rmtree(tmp)


def test_mood_feedback_loop_end_to_end():
    tmp = make_tmpdir()
    try:
        # 第 1 轮：小满带正面立场回复（兴奋），立场要存档
        c1 = MoodClient(sentiment=0.6, emotion="兴奋", plan_mood="兴奋")
        agent = _agent(tmp, c1)
        agent.on_user_message("跟你说个特好玩的事", now=1000.0)
        meta = json.loads(agent.conn.execute(
            "SELECT meta FROM messages WHERE sender='agent' ORDER BY id DESC LIMIT 1"
        ).fetchone()["meta"])
        assert meta.get("my_stance", {}).get("mood") == "兴奋", "立场要随回复存档"
        # 第 2 轮：对方泼冷水（负面强情绪）-> 冲撞，心情事件进 prompt
        c1._sentiment, c1._emotion = -0.8, "烦躁"
        agent.on_user_message("行了别说了，无聊", now=1100.0)
        respond = [p for p in c1.prompts
                   if p.get("本次任务", {}).get("意图") == "respond"][-1]
        ev = respond.get("心情事件") or {}
        assert ev.get("type") == "冲撞", f"应判冲撞，得到 {ev}"
        assert "当前状态" in respond
        # 第 3 轮：对方同频热聊 -> 共鸣
        c1._sentiment, c1._emotion = 0.7, "开心"
        agent.on_user_message("哈哈对吧，我也觉得好笑", now=1200.0)
        respond = [p for p in c1.prompts
                   if p.get("本次任务", {}).get("意图") == "respond"][-1]
        assert (respond.get("心情事件") or {}).get("type") == "共鸣"
    finally:
        rmtree(tmp)


def test_discrete_mood_palette():
    base = {"energy": 0.6, "social_need": 0.5, "curiosity": 0.7, "care": 0.7,
            "expression": 0.55, "arousal": 0.4, "valence": 0.35}
    angry = dict(base, valence=-0.5, arousal=0.7)
    assert "火气" in state_to_natural_language(angry)["基调"]
    low = dict(base, valence=-0.5, arousal=0.3)
    assert "憋屈" in state_to_natural_language(low)["基调"]
    hyped = dict(base, valence=0.6, arousal=0.7)
    assert "兴奋" in state_to_natural_language(hyped)["基调"]
    nice = dict(base, valence=0.6, arousal=0.3)
    assert "愉悦" in state_to_natural_language(nice)["基调"]
    assert state_to_natural_language(base)["基调"] == "平静"


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_mood_feedback"))
