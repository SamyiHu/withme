"""P2 验收：每轮动态重建 Prompt——不含聊天历史、记忆 top_k 受控、
身份摘要 <=200 字、记忆 token 占比 <=40%、同输入同输出。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.config import load_config, load_persona
from src.prompt_builder import (MEMORY_TOKEN_SHARE_LIMIT, PromptBuilder,
                                estimate_tokens, state_to_natural_language,
                                strip_media_markers, strip_role_prefix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _builder(tmp):
    cfg = load_config(os.path.join(ROOT, "config", "config.example.yaml"))
    persona = load_persona(os.path.join(ROOT, "config", "persona.example.yaml"))
    return PromptBuilder(cfg, persona), persona


def _mem(id_, content, sal=0.5):
    return {"id": id_, "type": "event", "content": content, "salience": sal,
            "tags": [], "status": "open"}


def _base_kwargs(persona):
    return dict(
        intent="remind_open_intent", source={"type": "intent", "id": 7},
        state_vector=dict(persona.baseline_state),
        relation_state={"affinity": 0.6, "trust": 0.5, "familiarity": 0.4,
                        "boundaries": [], "recent_active_effect": ""},
        memories=[_mem(1, "用户想学冲浪", 0.8)],
        open_intents=[{"id": 7, "content": "想学冲浪", "type": "intent", "salience": 0.8,
                       "tags": [], "status": "open"}],
        permission_level="L2", last_active_effect="上次主动收到正面回复",
    )


def test_prompt_structure_and_constraints():
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        prompt = builder.build(**_base_kwargs(persona))
        required = ["身份摘要", "核心不变量", "表达风格", "当前状态", "关系状态",
                    "相关记忆", "未完成意图", "主动权限", "最近主动效果", "本次任务", "输出格式"]
        for key in required:
            assert key in prompt, key
        # 结构上就不存在聊天历史键
        assert not any("历史" in k or "chat" in k.lower() for k in prompt)
        assert len(prompt["身份摘要"]) <= 200
        assert len(prompt["相关记忆"]) <= 5
        assert prompt["相关记忆"][0]["id"] == 1
        assert prompt["本次任务"]["来源"] == "intent:7"
        assert "只渲染" in prompt["本次任务"]["指令"]
        # LLM 不参与决策：任务指令里只出现被选中的意图
        assert prompt["本次任务"]["意图"] == "remind_open_intent"
    finally:
        rmtree(tmp)


def test_prompt_deterministic_and_state_sensitive():
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kwargs = _base_kwargs(persona)
        a = builder.build(**kwargs)
        b = builder.build(**kwargs)
        assert a == b, "同输入必须重建出相同 Prompt"
        kwargs2 = dict(kwargs)
        kwargs2["state_vector"] = dict(persona.baseline_state, energy=0.1, valence=-0.5)
        c = builder.build(**kwargs2)
        assert c["当前状态"]["描述"] != a["当前状态"]["描述"]
        assert "憋屈" in c["当前状态"]["基调"]      # 离散心情盘：低落+低唤醒
        nl = state_to_natural_language({"energy": 0.9, "social_need": 0.8, "curiosity": 0.8,
                                        "care": 0.8, "expression": 0.8, "arousal": 0.8,
                                        "valence": 0.8})
        assert "精力充沛" in nl["描述"] and "兴奋" in nl["基调"]
        angry = state_to_natural_language({"energy": 0.6, "social_need": 0.5, "curiosity": 0.6,
                                           "care": 0.6, "expression": 0.5, "arousal": 0.7,
                                           "valence": -0.6})
        assert "火气" in angry["基调"]              # 负效价+高唤醒=被气到
    finally:
        rmtree(tmp)


def test_dialogue_speaker_labels():
    """最近对话必须显示"我/创造者/群友/你"，不能把所有发言都写成"你"。"""
    rows = [
        {"sender": "agent", "content": "我说的话", "meta": {}},
        {"sender": "user", "content": "创造者说的话",
         "meta": {"speaker": "阿澄", "speaker_role": "owner"}},
        {"sender": "user", "content": "群友说的话",
         "meta": {"speaker": "胖达", "speaker_role": "member",
                  "addressee": "你"}},
        {"sender": "user", "content": "私聊里的话", "meta": {}},
    ]
    lines = [PromptBuilder._dialogue_line(r, group_context=True) for r in rows]
    assert lines[0].startswith("我：")
    assert lines[1].startswith("[创造者 阿澄]")
    assert lines[2].startswith("[群友 胖达 → 你]")
    assert lines[3] == "你：私聊里的话"
    private = PromptBuilder._dialogue_line(
        {"sender": "user", "content": "私聊里的话",
         "meta": {"speaker": "你", "speaker_role": "owner"}}, group_context=False)
    assert private == "你：私聊里的话", "私聊不能被标成[创造者 你]"
    assert strip_role_prefix("[主人 阿澄] 收到\n[群友 胖达] 在吗") == "收到\n在吗"
    assert strip_role_prefix("正常回复") == "正常回复"
    assert strip_media_markers("奶龙吧你说的，接好～[图片]") == "奶龙吧你说的，接好～"
    assert strip_media_markers("[图片1]") == ""


def test_proactive_prompt_carries_dialogue_and_rules():
    """主动问用户不能只给一条最近消息；必须带最近对话 + 主动发起规则。"""
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kwargs = _base_kwargs(persona)
        kwargs.update(intent="ask_light_question", source={"type": "intent", "id": 7},
                      recent_user_message="最近在忙项目",
                      recent_dialogue=[{"sender": "user", "content": "最近在忙项目"},
                                       {"sender": "agent", "content": "忙完记得吃饭"}])
        prompt = builder.build(**kwargs)
        assert "主动发起" in prompt
        assert "具体" in " ".join(prompt["主动发起"])
        assert "最近对话" in prompt and len(prompt["最近对话"]) == 2
        assert "只问一句" in " ".join(prompt["主动发起"])
    finally:
        rmtree(tmp)


def test_group_prompt_has_creator_bond():
    """群聊 Prompt：创造者是亲近的信任锚，不是主人/上级；保持友善、理智、不站队。"""
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kwargs = _base_kwargs(persona)
        kwargs.update(intent="respond", source=None,
                      recent_user_message="群里有人拉我站队",
                      group_context=True, group_self_names=["阿满", "小满"],
                      group_owner_name="阿澄")
        prompt = builder.build(**kwargs)
        rules = "\n".join(prompt["群聊规则"])
        assert "不是你的主人/上级" in rules
        assert "稳定的信任和依赖" in rules
        assert "信任不等于盲从" in rules
        assert "不替任何一方攻击另一方" in rules
        assert "不要因为谁声音大" in rules
        assert "不评价亲友" not in rules, "不能再拿边界当服从借口"
        assert prompt["群聊身份"]["创造者"] == "阿澄"
        assert "创造者关系" in prompt
        assert "信任" in prompt["创造者关系"]["关系"]
        plain = builder.build(**_base_kwargs(persona))
        assert "群聊规则" not in plain
    finally:
        rmtree(tmp)


def test_memory_token_share_capped():
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kwargs = _base_kwargs(persona)
        # 塞入超长记忆，逼迫裁剪生效
        kwargs["memories"] = [
            _mem(1, "用户想学冲浪，等夏天去海边", 0.9),
            _mem(2, "超长记忆" * 300, 0.8),
            _mem(3, "又一条超长记忆" * 260, 0.7),
            _mem(4, "还有一条", 0.6),
        ]   # 能力清单加入后 prompt 底座变大，样本同步加量保持越过 40% 阈值
        prompt = builder.build(**kwargs)
        mem_tokens = sum(estimate_tokens(str(m)) for m in prompt["相关记忆"])
        total = sum(estimate_tokens(str(v)) for v in prompt.values())
        assert mem_tokens <= MEMORY_TOKEN_SHARE_LIMIT * total + 1, \
            f"记忆占比 {mem_tokens / total:.2f} 超过 {MEMORY_TOKEN_SHARE_LIMIT}"
        assert len(prompt["相关记忆"]) < len(kwargs["memories"]), "应裁掉低分记忆"
        assert prompt["相关记忆"][0]["id"] == 1, "高分记忆保留"
        assert estimate_tokens("") == 0 and estimate_tokens("abcd") == 1
    finally:
        rmtree(tmp)


def test_respond_intent_includes_recent_message():
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kwargs = _base_kwargs(persona)
        kwargs.update(intent="respond", source=None, recent_user_message="最近在忙项目")
        prompt = builder.build(**kwargs)
        assert prompt["最近对方消息"] == "最近在忙项目"
        assert prompt["本次任务"]["来源"] is None
    finally:
        rmtree(tmp)


def test_prompt_cache_friendly_ordering():
    """DeepSeek 前缀缓存优化：分节按变化频率升序，静态人设必须在最前，
    每轮必变的当前状态/任务沉底——钉住顺序防止回退。"""
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        prompt = builder.build(**_base_kwargs(persona))
        keys = list(prompt.keys())
        pos = {k: keys.index(k) for k in (
            "身份摘要", "表达风格", "输出格式", "未完成意图", "最近对话",
            "关系状态", "相关记忆", "当前状态", "本次任务")}
        assert pos["身份摘要"] < pos["表达风格"] < pos["输出格式"], "静态段必须在前"
        assert pos["输出格式"] < pos["未完成意图"] < pos["最近对话"], "少变段居前"
        # 对话块是追加式稳定的，必须排在每轮全变的记忆/关系/状态之前
        assert pos["最近对话"] < pos["关系状态"] < pos["相关记忆"] < pos["当前状态"]
        assert pos["当前状态"] < pos["本次任务"], "易变段必须沉底"
        # 当前时间每轮都变，必须排在一切可复用段之后（末位），否则打穿前缀缓存
        assert keys[-1] == "当前时间", keys[-2:]
        assert pos["本次任务"] < keys.index("当前时间")
    finally:
        rmtree(tmp)


def test_prompt_carries_current_time():
    """当前时间进 prompt：模型对'现在'唯一的可靠来源；now 缺省兜底系统时钟。"""
    import time as _time
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        fixed = 1790000000.0   # 2026-09-22（UTC）附近，按本地时区展开断言
        kwargs = _base_kwargs(persona)
        kwargs["memories"] = [dict(_mem(1, "用户想学冲浪", 0.8), created_at=fixed)]
        prompt = builder.build(now=fixed, **kwargs)
        ct = prompt["当前时间"]
        assert ct.startswith(_time.strftime("%Y-%m-%d", _time.localtime(fixed))), ct
        assert " 周" in ct and "（" in ct, ct
        # 记忆相对标签在同一 now 下工作
        assert prompt["相关记忆"][0].get("时间") == "今天"
        # 不传 now 也不炸（兜底系统时钟）
        prompt2 = builder.build(**_base_kwargs(persona))
        assert "当前时间" in prompt2
    finally:
        rmtree(tmp)


def test_relative_age_tolerates_mem0_iso_strings():
    """mem0/qdrant 的 created_at 是 ISO 字符串（naive UTC），SQLite 是 float——
    两种来源都要能算；垃圾输入返回空标签而不是 TypeError 炸掉整轮回复。
    （回归：09:11 群聊处理失败 float-str，就是这条路径。）"""
    from src.prompt_builder import relative_age
    import calendar, time as _time
    now = calendar.timegm(_time.strptime("2026-10-28 09:00:00", "%Y-%m-%d %H:%M:%S"))
    assert relative_age("2026-10-28T08:30:00.123456", now) == "今天"
    assert relative_age("2026-10-26T09:00:00Z", now) == "2天前"
    assert relative_age("2026-10-27T09:00:00", now) == "昨天"
    assert relative_age(now, now) == "今天"                      # float 原样
    assert relative_age(str(now), now) == "今天"                 # 数字字符串
    assert relative_age("不是时间", now) == ""                    # 垃圾输入不炸
    assert relative_age(None, now) == ""


def test_parse_usage_cache_fields():
    from src.llm_client import parse_usage
    u = parse_usage({"prompt_tokens": 100, "completion_tokens": 5,
                     "prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40})
    assert u["cache_hit_tokens"] == 60 and u["cache_miss_tokens"] == 40
    u2 = parse_usage({"prompt_tokens": 10, "cache_read_input_tokens": 8})
    assert u2["cache_hit_tokens"] == 8 and u2["cache_miss_tokens"] is None
    assert parse_usage(None)["prompt_tokens"] == 0


def test_reply_plan_enters_prompt():
    """回应思路（腹稿）进 prompt；没给就不出现，不影响既有结构。"""
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kw = _base_kwargs(persona)
        p1 = builder.build(**kw)
        assert "回应思路" not in p1
        p2 = builder.build(**kw, reply_plan={
            "what": "接显卡涨价的话头", "how": "先惊讶再问，两短句",
            "mood": "惊讶", "对方此刻情绪": "兴奋"})
        assert p2["回应思路"]["mood"] == "惊讶"
        assert p2["回应思路"]["对方此刻情绪"] == "兴奋"
    finally:
        rmtree(tmp)


def test_old_dialogue_lines_carry_time_tags():
    """最近对话带"多久之前说的"：防止把几小时前自己的话当成刚说的接着聊。

    2026-10-28 实测：18:54 主动说了句"先去吃口热乎的"，22:19 用户说别的，
    它又把"饭还是要吃的"接上去——因为窗口里那句话没有时间标记。"""
    import time
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        now = time.time()
        old = {"sender": "agent", "content": "先去吃口热乎的",
               "created_at": now - 3.5 * 3600, "meta": {}}
        fresh = {"sender": "agent", "content": "饭还是要吃的",
                 "created_at": now - 10, "meta": {}}
        user = {"sender": "user", "content": "刚刚AI魔了", "created_at": now - 5,
                "speaker": "阿澄", "speaker_role": "owner", "meta": {}}
        kw = _base_kwargs(persona)
        p = builder.build(**kw, now=now, recent_dialogue=[user, old, fresh],
                          group_context=True)
        lines = p["最近对话"]
        assert lines[1].startswith("我（3.5小时前）："), lines
        assert lines[2] == "我：饭还是要吃的", lines
        assert "时间标记说明" in p, "有时间标记时要解释它"
        # 全是新消息时不加标记、也不加说明
        p2 = builder.build(**kw, now=now, recent_dialogue=[user, fresh],
                           group_context=True)
        assert all("前）" not in x for x in p2["最近对话"])
        assert "时间标记说明" not in p2
    finally:
        rmtree(tmp)

def test_group_prompt_tells_creator_he_is_the_favourite():
    """创造者问"你对谁好感最高"时要直说是他，别用"法官/贿赂"段子挡回去。

    2026-10-28 实测：群里主人问好感排名，AI 答"这题是坑吧…我排谁都要挨喷"。"""
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kw = _base_kwargs(persona)
        p = builder.build(**kw, group_context=True, speaker_name="阿澄",
                          speaker_role="owner", speaker_addressee="你")
        rel = p.get("创造者关系") or {}
        assert "好感" in (rel.get("被问到好感") or ""), rel
        rules = " ".join(p.get("群聊规则") or [])
        assert "好感最高" in rules, "群聊规则要写清楚这条"
    finally:
        rmtree(tmp)

def test_needs_detail_injects_length_instruction():
    """长度决策（感知层判定）落地到 Prompt：needs_detail=True 才注入"该详细"
    指令；默认（闲聊）不注入，保持短说人设。"""
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kwargs = _base_kwargs(persona)
        plain = builder.build(**kwargs)
        assert "长度指令" not in plain
        detailed = builder.build(**kwargs, needs_detail=True)
        assert "长度指令" in detailed
        assert "别为短而短" in detailed["长度指令"]
        assert "说透" in detailed["长度指令"]
    finally:
        rmtree(tmp)


def test_roast_level_controls_playful_teasing():
    """玩梗互损尺度（persona.style.roast_level）：off/缺省=不配合损人；
    mild/on=注入"朋友间玩梗式互损可以"的表达风格与群规则。"""
    tmp = make_tmpdir()
    try:
        builder, persona = _builder(tmp)
        kwargs = _base_kwargs(persona)
        out = builder.build(**kwargs, group_context=True,
                            group_self_names=["小满"], group_owner_name="阿澄")
        assert "玩梗尺度" not in out["表达风格"]
        assert not any("玩梗式互损" in r for r in out["群聊规则"])
        persona.style["roast_level"] = "mild"
        out2 = builder.build(**kwargs, group_context=True,
                             group_self_names=["小满"], group_owner_name="阿澄")
        assert "玩梗尺度" in out2["表达风格"]
        assert any("玩梗式互损" in r for r in out2["群聊规则"])
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_prompt_rebuild"))
