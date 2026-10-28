"""候选意图目录：level、来源要求与渲染提示（motivation 与 prompt_builder 共用）。

原则 E：除 silence/micro 外，每个主动意图都必须有来源
（open_intent / memory / event），否则闸门直接拒绝。
"""

INTENTS = {
    "silence": {
        "level": 0, "needs_source": False, "source_types": [],
        "hint": "保持沉默，不输出任何内容",
    },
    "micro_reaction": {
        "level": 1, "needs_source": False, "source_types": [],
        "hint": "一句极简微反应，不超过 10 个字（语气词或短短几个字），不打断对方",
    },
    "ask_light_question": {
        "level": 2, "needs_source": True, "source_types": ["memory", "intent"],
        "hint": "围绕来源和最近对话，问一个具体、接得上话的问题；不要泛泛问'在干嘛/最近怎么样'，只问一句",
    },
    "share_related_memory": {
        "level": 2, "needs_source": True, "source_types": ["memory"],
        "hint": "先接上最近对话，再自然提起这条记忆，句末附上 [M{记忆ID}] 标记，不得引用列表之外的记忆",
    },
    "remind_open_intent": {
        "level": 2, "needs_source": True, "source_types": ["intent"],
        "hint": "先接上最近对话，再自然地提醒这个未完成意图，一句话，不连续追问",
    },
    "express_care": {
        "level": 2, "needs_source": True, "source_types": ["intent", "memory"],
        "hint": "基于来源简短表达关心，不给压力",
    },
    "shift_topic": {
        "level": 3, "needs_source": True, "source_types": ["memory"],
        "hint": "换个话题并顺带说明为什么想到它",
    },
    "propose_small_activity": {
        "level": 3, "needs_source": True, "source_types": ["intent"],
        "hint": "围绕未完成意图提议一个当下就能做的小活动",
    },
    # 到点提醒（手部）：用户自己约的闹钟，由调度在到点时触发
    "reminder_due": {
        "level": 1, "needs_source": True, "source_types": ["reminder"],
        "hint": "把到点的提醒转达给用户：一句话说清提醒内容，自然、不播音腔",
    },
    # 应答路径（非主动）：用户消息到达时始终渲染
    "respond": {
        "level": 0, "needs_source": False, "source_types": [],
        "hint": "回应最近一条用户消息，顺着「最近对话」自然接话，像熟人在微信上闲聊。"
                "对用户正在经历的事保持真好奇：消息含糊或情绪突然时，先问一句怎么了/在乐什么；"
                "话题可以递进、发散、联想、岔开再绕回；不要连环追问细节，"
                "不要'是A还是B'的选择题式提问；情绪回应用大白话不说金句；"
                "该多说时就多说几句描述和联想，随口一聊就一两句接住；"
                "结构要有变化，不要每条都以口头禅开头或都以提问结尾",
    },
}


def level_of(intent_name: str) -> int:
    return INTENTS[intent_name]["level"]


def source_ref(source_type: str, source_id: int) -> str:
    return f"{source_type}:{source_id}"
