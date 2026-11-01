"""离线沙盒：在一个临时库里跑一条"用户消息 -> 回复"，不接触任何真实通道。

安全边界（硬性）：
  - 不 import wechat_bridge、不连微信 hook、不发任何消息；
  - 用 tempfile 建库，绝不碰 data/agent.db 或 data/wechat_users/；
  - 关掉主动/反思/手部外部动作，只走"收到消息 -> 生成回复"这一条路径。
"""
import os
import tempfile

os.environ.setdefault("AGENT_CONFIG_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config"))


class Sandbox:
    def __init__(self, config_dir: str, llm_client=None):
        if os.environ.get("AGENT_DB_PATH"):
            # 防止误把线上库指进来
            raise RuntimeError("检测到 AGENT_DB_PATH，拒绝在线上库上跑评测")
        from src.main import Agent
        self.tmp = tempfile.TemporaryDirectory(prefix="reply_lab_",
                                               ignore_cleanup_errors=True)
        self.agent = Agent(config_dir=config_dir,
                           db_path=os.path.join(self.tmp.name, "eval.db"),
                           llm_client=llm_client)
        # 关掉会"自己开口/自己成长"的部分，评测只看回复质量
        self.agent.config.reflection.enabled = False
        self.agent.config.perception.enabled = False
        # 评测要把随机性压住：温度固定 0.3，否则同一批用例两次跑能差 4~7 个字，
        # 分不清是提示词改了还是运气（2026-10-29 实测两次同 seed 差 7.7 字）
        self.agent.config.llm.temperature_surface = 0.3
        self.agent.config.stickers.enabled = False
        try:
            self.agent.set_proactive(False)
        except Exception:  # noqa: BLE001
            pass

    def reply(self, text: str, now: float = None) -> str:
        out = self.agent.on_user_message(text, now=now, from_owner=True,
                                         speaker="对方", speaker_role="owner",
                                         speaker_wxid="wxid_eval", addressee="你")
        reply = (out or {}).get("reply") or ""
        import re as _re
        return _re.sub(r"\[sticker[:：][^\]]{0,24}\]", "", reply).strip()

    def close(self):
        try:
            self.agent.conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.tmp.cleanup()
        except Exception:  # noqa: BLE001  Windows 上偶发文件占用，忽略
            pass
