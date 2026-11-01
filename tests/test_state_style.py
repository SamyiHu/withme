"""状态驱动语言（梯队1-1）验收：约束推导、机检、打回重写、机械兜底、
中常态零影响。全部离线。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import state_style as ss
from src.llm_client import MockLLMClient
from src.main import Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sv(**kw):
    base = {"energy": 0.6, "social_need": 0.5, "curiosity": 0.7, "care": 0.7,
            "expression": 0.55, "arousal": 0.4, "valence": 0.35}
    base.update(kw)
    return base


class ScriptedClient(MockLLMClient):
    """按脚本顺序返回文本，模拟违规->修正/屡教不改。"""
    name = "scripted"

    def __init__(self, scripts):
        super().__init__()
        self.scripts = list(scripts)
        self.renders = []

    def render(self, prompt, seed):
        text = self.scripts.pop(0) if self.scripts else "嗯。"
        self.renders.append({"constraints": prompt.get("本轮语气约束"),
                             "correction": prompt.get("修正指令"), "text": text})
        return text


def test_derive_and_check():
    # 低精力：限长+禁叹号
    c1 = ss.derive(sv(energy=0.2))
    assert "LEN<=40" in c1 and "EXCL<=0" in c1
    # 情绪低落：禁叹号+禁笑声
    c2 = ss.derive(sv(valence=-0.4),
                                                    {"energy": 0.6, "valence": 0.35, "arousal": 0.4})
    assert "EXCL<=0" in c2 and "LAUGH<=0" in c2 and not any(x.startswith("LEN") for x in c2)
    # 亢奋：放开
    c3 = ss.derive(sv(valence=0.8, arousal=0.8))
    assert "EXCL<=4" in c3 and "LAUGH<=2" in c3
    # 中常态：零约束
    assert ss.derive(sv()) == []
    # 机检
    assert ss.check("短句。", c1) == []
    long_text = "今天天气真好！！哈哈这个太搞笑了，" + "讲真我觉得非常不错，" * 6
    v = ss.check(long_text, c1)
    assert any("长度" in x for x in v) and any("感叹号" in x for x in v)
    assert ss.check("哈哈！！", c2) != []
    # 人读化含"机器会逐条检查"
    assert "机器会逐条检查" in ss.to_prompt(["LEN<=60", "EXCL<=0"])


def test_retry_path():
    tmp = make_tmpdir()
    try:
        client = ScriptedClient(["这句超长，" * 20 + "！！结束", "短的了。"])
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "agent.db"), llm_client=client)
        # 人为把精力压到极低
        row = agent.engine.new_row(1000.0)
        row["state_vector"]["energy"] = 0.2
        agent.save_row(row, 1000.0)
        r = agent.on_user_message("讲讲今天呗", now=1000.0)
        assert r["meta"]["style"]["retried"] is True
        assert r["meta"]["style"]["passed"] is True
        assert len("".join(r["reply"].split())) <= 40, r["reply"]
        # 第二次渲染带修正指令
        assert client.renders[1]["correction"] and "违反" in client.renders[1]["correction"]
        # 第一次渲染带了约束节
        assert client.renders[0]["constraints"]
    finally:
        rmtree(tmp)


def test_mechanical_fallback():
    tmp = make_tmpdir()
    try:
        long_text = "这是一句很长的话，说了很多很多很多的内容，真的很长。" * 6 + "！！"
        client = ScriptedClient([long_text, long_text])  # 重写也不改
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "agent.db"), llm_client=client)
        row = agent.engine.new_row(1000.0)
        row["state_vector"]["energy"] = 0.2
        agent.save_row(row, 1000.0)
        r = agent.on_user_message("说说", now=1000.0)
        meta = r["meta"]["style"]
        assert meta["retried"] is True and meta["mechanical"] is True
        assert len("".join(r["reply"].split())) <= 40, r["reply"]
        assert "！" not in r["reply"] or r["reply"].count("！") == 0
    finally:
        rmtree(tmp)


def test_normal_state_zero_overhead():
    tmp = make_tmpdir()
    try:
        client = ScriptedClient(["正常的一句回复。"])
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "agent.db"), llm_client=client)
        r = agent.on_user_message("随便聊聊", now=1000.0)  # 基线状态：无约束
        meta = r["meta"]["style"]
        assert meta["constraints"] == [] and meta["retried"] is False
        assert meta["passed"] is True
        assert client.renders[0]["constraints"] is None
        assert len(client.renders) == 1, "中常态不得触发第二次渲染"
    finally:
        rmtree(tmp)


def test_toolong_backstop_cuts_at_boundary_not_midword():
    """超长兜底必须切在边界上，不能无声腰斩（真实病例 2026-10-29）：
    "行哦，那我用下功夫瞅瞅\n\n先说个粗略的…" 旧逻辑 18 字硬 slice——唯一逗号
    在 pos2 够不到 pos>=8 门槛、换行不被当边界，"…瞅瞅\n\n先说个粗略" 出门：
    词中间断掉、空行留在正中，对方读起来就是消息被截断。"""
    out = ss.toolong_backstop("行哦，那我用下功夫瞅瞅\n\n先说个粗略的思路明天给你")
    assert out == "行哦，那我用下功夫瞅瞅", out      # 切在换行=保留完整第一行
    assert "\n" not in out
    # 无任何边界的长串：显式省略号，绝不无声切断在词中间
    assert ss.toolong_backstop("哈" * 20).endswith("…")
    # 不超限原样返回；串内唯一边界（逗号）正常可用
    assert ss.toolong_backstop("短的不动") == "短的不动"
    assert ss.toolong_backstop("今天天气不错，我们出去玩要不要带伞出门呢") == "今天天气不错"


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_state_style"))
