"""Agent 编排层 + 命令行入口。

主循环（计划 5.8）：
- on_user_message：惰性演化 -> 事件脉冲 -> 记忆写入/检索 -> 应答渲染。
  应答本身是被动路径、不走闸门；只有"主动搭话"附加层走 评分 -> 闸门 -> 渲染。
- evaluate_proactive：调度 tick 的主动路径，产生（或拒绝）一次主动行为。

LLM 永远只渲染已选定的意图（原则 D）；每次渲染的 seed 由哈希派生，可复现。
"""
import argparse
import json
from dataclasses import asdict
import os
import re
import shutil
import sys
import threading
import time
import urllib.request

from . import chaos, feedback as feedback_mod, motivation, perception as perception_mod, relation as relation_mod
from . import growth
from . import reflection as reflection_mod
from . import state_style as state_style_mod
from . import style_ctrl
from . import reviewer as reviewer_mod
from .config import (AppConfig, ConfigError, Persona, load_config, load_persona,
                     parse_quiet_hours, persona_from_dict, persona_overrides_path)
from .gate import Gate, register_active
from .hands import Hands, shopping_link
from .intents import INTENTS
from .llm_client import create_client, render_prompt_text
from .memory import MemoryStore, _bigrams
from .persona_attractor import PersonaAttractor
from .person import PersonStore
from . import stickers as sticker_mod
from .prompt_builder import (PromptBuilder, estimate_tokens, relative_age,
                             strip_media_markers, strip_role_prefix)
from .state_dims import DIMS, clamp_dim, clamp_vector
from .state_engine import StateEngine
from .storage import default_db_path, get_setting, init_db, load_state, project_root, save_state, set_setting



def _deep_merge(dst: dict, src: dict) -> dict:
    """dict 深合并；非 dict 一律整体替换（列表/标量按覆盖语义）。"""
    out = dict(dst or {})
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# 滚动摘要的拒答检测：模型安全拒答/语言不符的结果不能写进摘要。
# 实测事故：压缩请求被拒答，摘要被存成 "Sorry, I can't respond to this
# question."，之后每轮 prompt 都带着这句话。
_REFUSAL_HEAD_RE = re.compile(
    r"(?i)^(?:sorry\b|i'?m sorry\b|i (?:can'?t|cannot|am unable)\b|"
    r"i am (?:an? )?(?:ai|assistant)\b|"
    r"(?:抱歉|对不起)[，,]?\s*(?:我)?(?:无法|不能)(?:回答|协助|处理)|"
    r"作为(?:一个|一名)?(?:AI|人工智能|大语言模型|语言模型))")
_REFUSAL_MARK_RE = re.compile(
    r"(?i)(?:can'?t respond to (?:this|that) question|"
    r"无法回答(?:这个|该)?问题|不能回答(?:这个|该)?问题)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _summary_is_invalid(out: str, batch: list) -> bool:
    """摘要结果不可用：拒答口吻，或中文对话被摘要成了纯英文。"""
    head = (out or "")[:200]
    if _REFUSAL_HEAD_RE.search(head) or _REFUSAL_MARK_RE.search(head):
        return True
    spoken = "".join(str(r["content"] or "") for r in batch)
    return bool(_CJK_RE.search(spoken)) and not _CJK_RE.search(out)


def _bootstrap_configs(config_dir: str):
    """首次运行引导：仓库只带 *.example.yaml 模板（真实 config 不入库），
    缺失时自动从模板生成 config.yaml / persona.yaml，保证克隆后可直接运行。"""
    for name in ("config.yaml", "persona.yaml"):
        path = os.path.join(config_dir, name)
        if not os.path.exists(path):
            example = os.path.join(config_dir, name.replace(".yaml", ".example.yaml"))
            if os.path.exists(example):
                shutil.copyfile(example, path)
                print(f"[init] 已从模板生成 {path}（默认 mock 渲染，按需修改）")


class Agent:
    def __init__(self, config_dir: str = None, db_path: str = None, llm_client=None,
                 threaded: bool = False, group_room: str = None, group_name: str = "",
                 memory_provider: str = None, task_owner: str = ""):
        config_dir = config_dir or os.environ.get("AGENT_CONFIG_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        _bootstrap_configs(config_dir)
        self.config_dir = config_dir
        if db_path is None and os.environ.get("AGENT_DB_PATH"):
            db_path = os.environ["AGENT_DB_PATH"]
        self.config: AppConfig = load_config(os.path.join(config_dir, "config.yaml"))
        # 部署友好：LLM 接入可不改配置文件，直接用环境变量覆盖
        if os.environ.get("AGENT_LLM_PROVIDER"):
            self.config.llm.provider = os.environ["AGENT_LLM_PROVIDER"]
        if os.environ.get("AGENT_LLM_BASE_URL"):
            self.config.llm.base_url = os.environ["AGENT_LLM_BASE_URL"]
        if os.environ.get("AGENT_LLM_MODEL"):
            self.config.llm.model = os.environ["AGENT_LLM_MODEL"]
        if os.environ.get("AGENT_VLM_MODEL"):
            self.config.llm.vision_model = os.environ["AGENT_VLM_MODEL"]
        if os.environ.get("AGENT_LLM_REASONING_EFFORT"):
            self.config.llm.reasoning_effort = os.environ["AGENT_LLM_REASONING_EFFORT"]
        if os.environ.get("AGENT_LLM_PROMPT_IN_USER"):
            self.config.llm.prompt_in_user = os.environ["AGENT_LLM_PROMPT_IN_USER"] in ("1", "true", "True")
        if os.environ.get("AGENT_MEMORY_PROVIDER"):
            self.config.memory.provider = os.environ["AGENT_MEMORY_PROVIDER"]
        # 进程级覆盖（serve 用）：本地 qdrant 文件锁只允许一个进程持有，
        # serve 不参与竞争，把 mem0 留给微信桥
        if memory_provider:
            self.config.memory.provider = memory_provider
        self.persona: Persona = load_persona(os.path.join(config_dir, "persona.yaml"))
        self.role_id = self.persona.role_id
        self.group_room = group_room or ""
        self.is_group = bool(group_room and group_room.endswith("@chatroom"))
        self.loose_mode = bool(
            self.group_room and self.group_room in (self.config.wechat.loose_chats or []))
        self.group_self_names = []   # 群里别人怎么叫我（阿满/小满/机器人…）
        self.group_owner_name = ""   # 创造者的群昵称（内部只用于创造者关系，不做权限）
        self.conn = init_db(db_path, threaded=threaded)
        # 群名（微信会话标题）：桥建 Agent 时写进库，后台控制台按会话轴显示用
        self.group_name = group_name or ""
        _name_key = "group_name" if self.is_group else "display_name"
        if group_name:
            try:
                set_setting(self.conn, _name_key, group_name)
            except Exception:  # noqa: BLE001  名字只是展示层
                pass
        else:
            self.group_name = get_setting(self.conn, _name_key, "") or ""
        self.persons = PersonStore(self.conn, self.role_id, self.persona)
        self._turn_person_id = None
        self._turn_person_card = None
        self._creator_card = None
        self._threaded = threaded
        self._db_path = db_path or default_db_path()
        # 会话键：主库=main；wechat_users/<sid>/agent.db 的 sid。
        # 供 mem0 做会话隔离，防止跨会话串记忆（家人群召回私聊的事）。
        _parent = os.path.basename(os.path.dirname(os.path.abspath(self._db_path)))
        self.session_key = _parent if _parent and _parent != "data" else "main"
        self._lock = threading.RLock()
        if self.config.memory.provider == "mem0":
            try:
                from .memory_mem0 import Mem0MemoryStore
                self.memory = Mem0MemoryStore(
                    self.conn, data_dir=os.path.join(project_root(), "data"),
                    mcfg=self.config.memory, session_scope=self.session_key)
            except Exception as e:  # noqa: BLE001
                # mem0 起不来（典型：qdrant 文件锁被其他进程持有/残留）：
                # 降级 SQLite 记忆继续跑——记忆检索变弱，绝不能挡消息收发
                print(f"[memory] mem0 初始化失败，降级 SQLite 记忆：{e!r}")
                self.memory = MemoryStore(
                    self.conn, halflife_days=self.config.memory.salience_halflife_days)
        else:
            self.memory = MemoryStore(
                self.conn, halflife_days=self.config.memory.salience_halflife_days)
        self.engine = StateEngine(self.config, self.persona)
        self.gate = Gate(self.config, self.persona, self.conn)
        self.builder = PromptBuilder(self.config, self.persona)
        self.client = llm_client or create_client(self.config.llm, self.persona.style)
        # 双 Agent：互动 Agent（上）+ 审核 Agent（前额叶，只判断不说话）
        self.reviewer = reviewer_mod.ReviewAgent(self.config.reviewer, self.client)
        # 表情包库：回复里写 [sticker:情绪词]，桥负责换成真表情图
        try:
            _st = self.config.stickers
            self.stickers = sticker_mod.StickerStore(
                media_dir=(_st.dir if os.path.isabs(_st.dir)
                           else os.path.join(project_root(), _st.dir)),
                site=_st.site, packs=list(_st.packs or []), enabled=bool(_st.enabled),
                ffmpeg_path=getattr(_st, "ffmpeg_path", "") or "",
                send_mode=getattr(_st, "send_mode", "static") or "static",
                log=lambda m: print(f"[stickers] {m}"))
        except Exception as e:  # noqa: BLE001  贴纸库失败不挡聊天
            print(f"[stickers] 初始化失败：{e!r}")
            self.stickers = None
        self.hands = Hands(self.config, self.conn, self.role_id, memory=self.memory,
                           task_owner=task_owner)
        from .taskstore import OutboxStore, TriggerStore
        self.outbox = OutboxStore(self.conn, self.role_id)
        self.triggers = TriggerStore(self.conn, self.role_id)
        self.hands.image_verifier = self._verify_drawn_image
        self.plugin_info = []
        if self.config.hands.plugins_enabled:
            from .plugins import load_plugins
            self.plugin_info = load_plugins(
                self.hands, os.path.join(project_root(), "plugins"),
                disabled=get_setting(self.conn, "plugins_disabled", []) or [])
        # LLM 用量遥测：按用途累计，含 DeepSeek 前缀缓存命中（桥接透传）
        self.usage_stats = {}
        # 对话压缩进行中标志（防重入；LLM 在锁外跑）
        self._compressing = False
        # 成长基线：角色卡原始吸引子中心（人格漂移的参照点，reset 回这里）
        self._base_centers = {e.state_name: e.center for e in self.persona.attractor}
        self._apply_persisted_settings()
        self._seed_review_quota()
        try:
            creator = self.persons.ensure_creator()
            self._creator_card = self.persons.summary(creator) if creator else None
            self.persons.attach_unscoped_memories(
                creator, f"group:{self.group_room}" if self.is_group else "private")
        except Exception as e:  # noqa: BLE001  人物卡失败不阻塞
            print(f"[person] 初始化创造者卡失败：{e!r}")

    # ---------------- 设置面板：运行时热改 + 持久化于 settings 表 ----------------

    def _apply_persisted_settings(self):
        """启动时恢复 DB 里保存的运行时设置（覆盖配置文件/环境变量）。"""
        try:
            ov = get_setting(self.conn, "llm_override", None)
            if ov:
                self._apply_llm_override(ov)
            stored_key = get_setting(self.conn, "llm_api_key", None)
            if stored_key:
                os.environ.setdefault(self.config.llm.api_key_env, stored_key)
            mp = get_setting(self.conn, "memory_provider", None)
            if mp and mp != self.config.memory.provider:
                self.config.memory.provider = mp
                self._rebuild_memory(mp)
            per = get_setting(self.conn, "perception_enabled", None)
            if per is not None:
                self.config.perception.enabled = per
            qh = get_setting(self.conn, "quiet_hours", None)
            if qh is not None:
                self.persona.quiet_hours = list(qh)
            style = get_setting(self.conn, "persona_style", None)
            if style:
                self.persona.style.update(style)
            # 老数据迁移：成长规则曾同时写进 persona_style.chat_rules，与
            # growth_rules 形成双份真相（提示词里重复一次，手工删也删不净）。
            # 启动时剥掉，只保留 growth_rules 这一份。
            growth = get_setting(self.conn, "growth_rules", []) or []
            style_rules = list(self.persona.style.get("chat_rules", []) or [])
            kept = [r for r in style_rules if r not in growth]
            if len(kept) != len(style_rules):
                self.persona.style["chat_rules"] = kept
                stored = dict(style or {})
                stored["chat_rules"] = kept
                set_setting(self.conn, "persona_style", stored)
            drift = get_setting(self.conn, "attractor_drift", None)
            if drift:
                self._apply_drift(drift)
            cfg_ov = get_setting(self.conn, "config_overrides", None)
            if cfg_ov:
                self._apply_config_dict(cfg_ov)
        except Exception as e:  # noqa: BLE001  设置恢复失败不阻塞启动
            print(f"[settings] 恢复持久化设置失败：{e!r}")

    def _seed_review_quota(self):
        """重启后把今天已用掉的审稿配额从 review_logs 读回内存（日上限要跨重启生效）。"""
        try:
            day = time.strftime("%Y-%m-%d")
            start = time.mktime(time.strptime(day, "%Y-%m-%d"))
            used = self.conn.execute(
                "SELECT COUNT(*) c FROM review_logs WHERE role_id=? AND llm_used=1"
                " AND created_at>=?", (self.role_id, start)).fetchone()["c"]
            self.reviewer.today = day
            self.reviewer.calls_today = int(used or 0)
        except Exception as e:  # noqa: BLE001  只影响配额，不阻塞启动
            print(f"[review] 恢复审稿配额失败：{e!r}")

    def _apply_llm_override(self, ov: dict):
        if ov.get("provider"):
            self.config.llm.provider = ov["provider"]
        if ov.get("base_url"):
            self.config.llm.base_url = ov["base_url"]
        if ov.get("model"):
            self.config.llm.model = ov["model"]
        if isinstance(ov.get("prompt_in_user"), bool):
            self.config.llm.prompt_in_user = ov["prompt_in_user"]
        if "reasoning_effort" in ov:
            self.config.llm.reasoning_effort = str(ov.get("reasoning_effort") or "")
        self.client = create_client(self.config.llm, self.persona.style)

    def _rebuild_memory(self, provider: str):
        self.config.memory.provider = provider
        if provider == "mem0":
            try:
                from .memory_mem0 import Mem0MemoryStore
                self.memory = Mem0MemoryStore(
                    self.conn, data_dir=os.path.join(project_root(), "data"),
                    mcfg=self.config.memory, session_scope=self.session_key)
            except Exception as e:  # noqa: BLE001  同 __init__：锁冲突降级，不挡服务
                print(f"[memory] mem0 切换失败，保持 SQLite 记忆：{e!r}")
                self.config.memory.provider = "sqlite"
                self.memory = MemoryStore(
                    self.conn, halflife_days=self.config.memory.salience_halflife_days)
        else:
            self.memory = MemoryStore(
                self.conn, halflife_days=self.config.memory.salience_halflife_days)

    def get_settings_view(self) -> dict:
        with self._lock:
            disabled = get_setting(self.conn, "proactive_disabled_levels", {}) or {}
            return {
                "proactive": {
                    "enabled": get_setting(self.conn, "proactive_enabled", True),
                    "levels": {f"L{i}": (bool(self.persona.active_permissions.get(f"L{i}", False))
                                         and not disabled.get(f"L{i}", False))
                               for i in (1, 2, 3)},
                },
                "quiet_hours": list(self.persona.quiet_hours),
                "llm": {
                    "provider": self.config.llm.provider,
                    "base_url": self.config.llm.base_url,
                    "model": self.config.llm.model,
                    "prompt_in_user": self.config.llm.prompt_in_user,
                    "reasoning_effort": self.config.llm.reasoning_effort,
                    "api_key_set": bool(os.environ.get(self.config.llm.api_key_env)),
                },
                "memory": {"provider": self.config.memory.provider},
                "plugins": {"enabled": self.config.hands.plugins_enabled,
                            "loaded": self.plugin_info},
                "perception": {"enabled": self.config.perception.enabled},
                "persona_style": {
                    "catchphrases": list(self.persona.style.get("catchphrases", [])),
                    "tone": self.persona.style.get("tone", ""),
                    "chat_rules": list(self.persona.style.get("chat_rules", [])),
                    "roast_level": self.persona.style.get("roast_level", ""),
                },
                "cache": self._usage_summary(),
                "role_id": self.role_id,
            }

    def apply_settings(self, payload: dict) -> dict:
        """应用部分设置并持久化；非法值抛 ValueError。立即生效，无需重启。"""
        with self._lock:
            errors = []
            pro = payload.get("proactive")
            if isinstance(pro, dict):
                if isinstance(pro.get("enabled"), bool):
                    self.set_proactive(pro["enabled"])
                for name, val in (pro.get("levels") or {}).items():
                    if name in ("L1", "L2", "L3") and isinstance(val, bool):
                        self.set_level_enabled(name, val)
            if "quiet_hours" in payload:
                qh = payload["quiet_hours"]
                ok = isinstance(qh, list) and all(
                    isinstance(s, str) and re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", s) for s in qh)
                if not ok:
                    errors.append("quiet_hours 须为 HH:MM-HH:MM 字符串列表")
                else:
                    try:
                        from .config import _parse_quiet_span
                        for s in qh:
                            _parse_quiet_span(s)
                        set_setting(self.conn, "quiet_hours", qh)
                        self.persona.quiet_hours = qh
                    except Exception:  # noqa: BLE001
                        errors.append("quiet_hours 时间不合法")
            llm = payload.get("llm")
            if isinstance(llm, dict):
                ov = get_setting(self.conn, "llm_override", {}) or {}
                for k in ("provider", "base_url", "model"):
                    if isinstance(llm.get(k), str) and llm[k].strip():
                        ov[k] = llm[k].strip()
                if isinstance(llm.get("prompt_in_user"), bool):
                    ov["prompt_in_user"] = llm["prompt_in_user"]
                if isinstance(llm.get("reasoning_effort"), str):
                    eff = llm["reasoning_effort"].strip().lower()
                    if eff in ("", "low", "medium", "high"):
                        ov["reasoning_effort"] = eff
                    else:
                        errors.append("llm.reasoning_effort 只支持 low / medium / high")
                if ov.get("provider") not in (None, "mock", "openai_compatible"):
                    errors.append("llm.provider 只支持 mock / openai_compatible")
                else:
                    set_setting(self.conn, "llm_override", ov)
                    self._apply_llm_override(ov)
                if isinstance(llm.get("api_key"), str) and llm["api_key"].strip():
                    set_setting(self.conn, "llm_api_key", llm["api_key"].strip())
                    os.environ[self.config.llm.api_key_env] = llm["api_key"].strip()
            mem = payload.get("memory")
            if isinstance(mem, dict) and isinstance(mem.get("provider"), str):
                if mem["provider"] not in ("sqlite", "mem0"):
                    errors.append("memory.provider 只支持 sqlite / mem0")
                else:
                    set_setting(self.conn, "memory_provider", mem["provider"])
                    self._rebuild_memory(mem["provider"])
            per = payload.get("perception")
            if isinstance(per, dict) and isinstance(per.get("enabled"), bool):
                set_setting(self.conn, "perception_enabled", per["enabled"])
                self.config.perception.enabled = per["enabled"]
            growth = payload.get("growth")
            if isinstance(growth, dict) and growth.get("reset") is True:
                self.reset_growth()
            style = payload.get("persona_style")
            if isinstance(style, dict):
                stored = get_setting(self.conn, "persona_style", {}) or {}
                if isinstance(style.get("catchphrases"), list) and style["catchphrases"]:
                    stored["catchphrases"] = [str(x)[:12] for x in style["catchphrases"]][:8]
                if isinstance(style.get("tone"), str) and style["tone"].strip():
                    stored["tone"] = style["tone"].strip()[:160]
                if isinstance(style.get("chat_rules"), list):
                    stored["chat_rules"] = [str(r).strip()[:200] for r in style["chat_rules"]][:12]
                if isinstance(style.get("roast_level"), str):
                    lvl = style["roast_level"].strip().lower()
                    if lvl in ("", "off", "mild", "on"):
                        stored["roast_level"] = lvl
                    else:
                        errors.append("persona_style.roast_level 只支持 off / mild / on")
                set_setting(self.conn, "persona_style", stored)
                self.persona.style.update(stored)
            if errors:
                raise ValueError("；".join(errors))
            return self.get_settings_view()


    # ---------------- 运行时参数（控制台可视化 + 可调） ----------------

    def _apply_config_dict(self, payload: dict) -> list:
        """应用控制台参数（不写库）。返回错误列表；供 apply_config 与启动恢复共用。"""
        errors = []

        def num(obj, key, lo, hi, label, cast=float, src=None):
            if src is not None:
                if key not in src:
                    return
                val = src[key]
            elif isinstance(obj, dict):
                if key not in obj:
                    return
                val = obj[key]
            elif not hasattr(obj, key):
                return
            else:
                val = getattr(obj, key)
            try:
                v = cast(val)
            except (TypeError, ValueError):
                errors.append(label + " 必须是数字")
                return
            if not (lo <= v <= hi):
                errors.append(label + " 超出范围 [" + str(lo) + ", " + str(hi) + "]")
                return
            if isinstance(obj, dict):
                obj[key] = v
            else:
                setattr(obj, key, v)

        for section, spec in (
            ("tick", {"fast_seconds": (15, 3600, float), "medium_seconds": (60, 86400, float),
                       "slow_seconds": (3600, 604800, float)}),
            ("chaos", {"epsilon_base": (0.0, 0.5, float), "noise_smoothing": (0.0, 0.99, float)}),
            ("convergence", {"alpha": (0.0, 1.0, float), "max_pull": (0.0, 1.0, float),
                             "decay_tau_seconds": (60, 86400, float)}),
            ("cost", {"daily_token_budget": (0, 100000000, int)}),
            ("dialogue", {"window": (4, 64, int), "compress_trigger": (8, 500, int),
                          "summary_max_chars": (200, 4000, int)}),
        ):
            data = payload.get(section)
            if not isinstance(data, dict):
                continue
            obj = getattr(self.config, section, None)
            if obj is None:
                continue
            for key, (lo, hi, cast) in spec.items():
                num(obj, key, lo, hi, section + "." + key, cast,
                    src=data)

        for section in ("gate", "motivation"):
            data = payload.get(section)
            if not isinstance(data, dict):
                continue
            obj = getattr(self.config, section)
            for key in ("thresholds", "weights"):
                if isinstance(data.get(key), dict):
                    target = getattr(obj, key)
                    for name, val in data[key].items():
                        if name not in target:
                            continue
                        try:
                            v = float(val)
                        except (TypeError, ValueError):
                            errors.append(section + "." + key + "." + name + " 必须是数字")
                            continue
                        if not (0.0 <= v <= 1.0):
                            errors.append(section + "." + key + "." + name + " 须在 [0,1]")
                            continue
                        target[name] = v
                    if key == "weights" and sum(target.values()) <= 0:
                        errors.append("motivation.weights 总和必须 > 0")

        rv = payload.get("reviewer")
        if isinstance(rv, dict):
            r = self.config.reviewer
            if "mode" in rv:
                mode = str(rv.get("mode") or "").strip().lower()
                if mode in ("off", "rule", "llm"):
                    r.mode = mode
                else:
                    errors.append("reviewer.mode 只支持 off / rule / llm")
            if isinstance(rv.get("llm_purposes"), list):
                r.llm_purposes = [str(x).strip() for x in rv["llm_purposes"]
                                  if str(x).strip()]
            if isinstance(rv.get("risk_triggers"), bool):
                r.risk_triggers = rv["risk_triggers"]
            num(r, "max_llm_per_day", 0, 100000, "reviewer.max_llm_per_day", int, src=rv)
            num(r, "max_tokens", 60, 4000, "reviewer.max_tokens", int, src=rv)
            num(r, "timeout_seconds", 3, 120, "reviewer.timeout_seconds", src=rv)

        ref = payload.get("reflection")
        if isinstance(ref, dict):
            r = self.config.reflection
            if isinstance(ref.get("enabled"), bool):
                r.enabled = ref["enabled"]
            num(r, "min_interval_hours", 1, 720, "reflection.min_interval_hours", src=ref)
            num(r, "min_new_messages", 1, 1000, "reflection.min_new_messages", int, src=ref)
            num(r, "max_drift", 0.0, 0.5, "reflection.max_drift", src=ref)
            num(r, "drift_step", 0.0, 0.2, "reflection.drift_step", src=ref)
            num(r, "max_extra_rules", 1, 12, "reflection.max_extra_rules", int, src=ref)
            num(r, "growth_envelope", 0.0, 0.5, "reflection.growth_envelope", src=ref)
            num(r, "growth_eta0", 0.0, 1.0, "reflection.growth_eta0", src=ref)
            num(r, "growth_eta_min", 0.0, 0.5, "reflection.growth_eta_min", src=ref)
            num(r, "growth_wake_gap", 0.0, 0.2, "reflection.growth_wake_gap", src=ref)
            num(r, "growth_kappa", 1.0, 1000.0, "reflection.growth_kappa", src=ref)
            num(r, "growth_max_step", 0.0, 0.2, "reflection.growth_max_step", src=ref)
            num(r, "growth_alpha", 0.0, 1.0, "reflection.growth_alpha", src=ref)

        persona = payload.get("persona")
        if isinstance(persona, dict):
            if isinstance(persona.get("active_permissions"), dict):
                for lvl, val in persona["active_permissions"].items():
                    if lvl in ("L0", "L1", "L2", "L3") and isinstance(val, bool):
                        self.persona.active_permissions[lvl] = val
            for field, label in (("cooldown_seconds", "persona.cooldown_seconds"),
                                 ("daily_budget", "persona.daily_budget")):
                if isinstance(persona.get(field), dict):
                    target = getattr(self.persona, field)
                    for lvl, val in persona[field].items():
                        if lvl not in ("L1", "L2", "L3"):
                            continue
                        try:
                            v = int(val)
                        except (TypeError, ValueError):
                            errors.append(label + "." + lvl + " 必须是整数")
                            continue
                        if v < 0:
                            errors.append(label + "." + lvl + " 不能为负")
                            continue
                        target[lvl] = v
        return errors

    def apply_config(self, payload: dict) -> dict:
        """控制台参数写入：立即生效 + 持久化到 settings.config_overrides。"""
        with self._lock:
            errors = self._apply_config_dict(payload or {})
            if errors:
                raise ValueError("；".join(errors))
            stored = get_setting(self.conn, "config_overrides", {}) or {}
            def merge(dst, src):
                for k, v in (src or {}).items():
                    if isinstance(v, dict) and isinstance(dst.get(k), dict):
                        merge(dst[k], v)
                    else:
                        dst[k] = v
            merge(stored, payload or {})
            set_setting(self.conn, "config_overrides", stored)
            return self.get_settings_view()

    # ---------------- 人格基线 / 极限设定（控制台可改） ----------------

    PERSONA_EDITABLE = ("identity_summary", "core_invariants", "style",
                        "boundaries", "baseline_state", "attractor")

    def apply_persona(self, payload: dict) -> dict:
        """改人格基线 / 极限设定：校验 -> 写 persona.overrides.json -> 立即生效。

        只接白名单字段；校验用 persona_from_dict（和启动同一套规则）。
        reset=True 删除覆盖文件，回到 persona.yaml 出厂值。
        """
        with self._lock:
            from . import yamlite
            payload = dict(payload or {})
            reset = payload.pop("reset", False) is True
            unknown = [k for k in payload if k not in self.PERSONA_EDITABLE]
            if unknown:
                raise ValueError("不支持的字段：" + "、".join(unknown))
            base_path = os.path.join(self.config_dir, "persona.yaml")
            ov_path = persona_overrides_path(base_path)
            with open(base_path, "r", encoding="utf-8") as f:
                base = yamlite.loads(f.read())
            ov = {}
            if not reset and os.path.exists(ov_path):
                try:
                    with open(ov_path, "r", encoding="utf-8") as f:
                        ov = json.load(f) or {}
                except Exception:  # noqa: BLE001  坏文件当没有，重新写
                    ov = {}
            if not reset:
                ov = _deep_merge(ov, payload)
            merged = _deep_merge(base, ov)
            try:
                new_persona = persona_from_dict(merged)
            except ConfigError as e:
                raise ValueError(str(e))
            if reset:
                try:
                    os.remove(ov_path)
                except FileNotFoundError:
                    pass
            else:
                tmp = ov_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(ov, f, ensure_ascii=False, indent=2)
                os.replace(tmp, ov_path)
            # 就地更新：engine/gate/builder 持有同一个 persona 对象
            for field_name in ("identity_summary", "core_invariants", "attractor",
                               "baseline_state", "style", "boundaries",
                               "active_permissions", "cooldown_seconds",
                               "daily_budget", "quiet_hours", "creator"):
                setattr(self.persona, field_name, getattr(new_persona, field_name))
            self.engine.attractor = PersonaAttractor(self.persona, self.config.convergence)
            self._base_centers = {e.state_name: e.center for e in self.persona.attractor}
            drift = get_setting(self.conn, "attractor_drift", {}) or {}
            if drift:
                self._apply_drift(drift)
            return {"ok": True, "reset": reset, "overrides": ov,
                    "identity_summary": self.persona.identity_summary,
                    "core_invariants": list(self.persona.core_invariants)}

    # ---------------- 反思层（成长核心） ----------------

    def _apply_drift(self, drift: dict):
        """把有界漂移应用到吸引子中心（基线=角色卡原值），并重建引擎吸引子。

        上限用 growth_envelope（成长外环的包络，默认 0.25）；旧的
        reflection.max_drift（0.15）仍用于 gate_shift 这类单次调整。
        """
        cap = float(getattr(self.config.reflection, "growth_envelope",
                            self.config.reflection.max_drift) or 0.25)
        for entry in self.persona.attractor:
            base = self._base_centers[entry.state_name]
            d = max(-cap, min(cap, float(drift.get(entry.state_name, 0.0))))
            entry.center = clamp_dim(entry.state_name, base + d)
        self.engine.attractor = PersonaAttractor(self.persona, self.config.convergence)

    def _growth_state(self) -> dict:
        """读取成长外环状态；老库没有 growth_state 时用 attractor_drift 播种。"""
        st = get_setting(self.conn, "growth_state", None)
        if isinstance(st, dict) and isinstance(st.get("p"), dict):
            return {
                "p": {d: float((st.get("p") or {}).get(d, 0.0)) for d in DIMS},
                "target": {d: float((st.get("target") or {}).get(d, 0.0)) for d in DIMS},
                "n": int(st.get("n") or 0),
                "updated_at": float(st.get("updated_at") or 0.0),
            }
        drift = get_setting(self.conn, "attractor_drift", {}) or {}
        p = {d: float(drift.get(d, 0.0)) for d in DIMS}
        return {"p": p, "target": dict(p), "n": 0, "updated_at": 0.0}

    def _growth_profile(self) -> str:
        """提示词用的长期成长摘要（没长成时为空串）。"""
        try:
            return growth.profile_text(self._growth_state(), self._base_centers,
                                       self.config.reflection)
        except Exception:  # noqa: BLE001  画像失败不挡回复
            return ""

    def _log_growth_event(self, kind: str, evidence: dict, before: dict,
                          after: dict, now: float):
        try:
            brief = {"active_stats": evidence.get("active_stats"),
                     "relation": evidence.get("relation"),
                     "depth": evidence.get("depth"),
                     "new_user_msgs": evidence.get("new_user_msgs")}
            self.conn.execute(
                "INSERT INTO growth_events(role_id, kind, evidence, state_before,"
                " state_after, created_at) VALUES(?,?,?,?,?,?)",
                (self.role_id, kind, json.dumps(brief, ensure_ascii=False),
                 json.dumps(before, ensure_ascii=False),
                 json.dumps(after, ensure_ascii=False), now))
            self.conn.commit()
        except Exception as e:  # noqa: BLE001  账本失败不影响成长生效
            print(f"[growth] 事件写入失败：{e!r}")

    def maybe_reflect(self, now: float = None, force: bool = False) -> dict:
        """证据足够时触发一次反思；force=True 跳过阈值（CLI/手动用）。"""
        now = time.time() if now is None else now
        if not self.config.reflection.enabled and not force:
            return None
        if self._cost_degraded() and not force:
            return None
        last = get_setting(self.conn, "last_reflection_ts", 0) or 0
        cfg = self.config.reflection
        if not force:
            if now - last < cfg.min_interval_hours * 3600:
                return None
            sql = ("SELECT COUNT(*) c FROM messages WHERE role_id=? AND sender='user'"
                   " AND created_at>?")
            params = [self.role_id, last]
            if getattr(self, "is_group", False):
                sql += " AND (speaker_role='owner' OR content LIKE '[主人%')"
            n = self.conn.execute(sql, params).fetchone()["c"]
            if n < cfg.min_new_messages:
                return None
        return self._run_reflection(now)

    def _gather_evidence(self, since: float) -> dict:
        sql = ("SELECT content FROM messages WHERE role_id=? AND sender='user'"
               " AND created_at>?")
        params = [self.role_id, since]
        if getattr(self, "is_group", False):
            sql += " AND (speaker_role='owner' OR content LIKE '[主人%')"
        rows = self.conn.execute(sql + " ORDER BY id DESC LIMIT 8", params).fetchall()
        count_sql = ("SELECT COUNT(*) c FROM messages WHERE role_id=? AND sender='user'"
                     " AND created_at>?")
        if getattr(self, "is_group", False):
            count_sql += " AND (speaker_role='owner' OR content LIKE '[主人%')"
        new_user_msgs = int(self.conn.execute(
            count_sql, (self.role_id, since)).fetchone()["c"] or 0)
        # 话题深度：用户消息平均长度归一（0..1）；用于成长外环的目标估计
        avg_len = (sum(len(r["content"] or "") for r in rows) / len(rows)) if rows else 0.0
        depth = max(0.0, min(1.0, avg_len / 40.0))
        stats_row = self.conn.execute(
            "SELECT COUNT(*) total,"
            " SUM(CASE WHEN user_sentiment_delta IS NOT NULL AND user_sentiment_delta<=-0.5"
            "   THEN 1 ELSE 0 END) rejected,"
            " AVG(user_sentiment_delta) avg_s"
            " FROM active_logs WHERE role_id=? AND created_at>?",
            (self.role_id, since)).fetchone()
        rel = self.get_row(time.time()).get("relation_state", {})
        return {
            "themes": [r["content"][:60] for r in reversed(rows)],
            "active_stats": {"total": stats_row["total"] or 0,
                             "rejected": stats_row["rejected"] or 0,
                             "avg_sentiment": stats_row["avg_s"] or 0.0},
            "relation": {k: rel.get(k, 0.0) for k in ("affinity", "trust", "familiarity")},
            "open_intents": [m["content"][:40]
                             for m in self.memory.open_intents(self.role_id)][:3],
            "depth": round(depth, 3),
            "new_user_msgs": new_user_msgs,
            "drift": get_setting(self.conn, "attractor_drift", {}) or {},
            "gate_shift_L2": float(get_setting(self.conn, "gate_shift_L2", 0.0) or 0.0),
        }

    def _run_reflection(self, now: float) -> dict:
        cfg = self.config.reflection
        since = get_setting(self.conn, "last_reflection_ts", 0) or 0
        evidence = self._gather_evidence(since)
        applied = {"insights": [], "new_rule": None, "trait_shift": {}}
        try:
            raw = self.client.render_raw(
                reflection_mod.build_task(evidence), self._seed(now, "reflect", self.get_row(now)),
                temperature=0.4, max_tokens=1000)
            out = reflection_mod.parse_output(raw, cfg)
            self._record_usage("reflect")
        except Exception as e:  # noqa: BLE001  反思失败不影响主链路
            out = None
            print(f"[reflect] 调用失败：{e!r}")
        if out is None:
            print(f"[reflect] 解析失败，原始输出：{raw[:160]!r}")
            self._log_reflection(now, evidence, {"error": "parse_failed", "raw": raw[:200]}, applied)
            set_setting(self.conn, "last_reflection_ts", now)
            return {"ran": False, "reason": "parse_failed"}

        for insight in out["insights"]:
            self.memory.add(self.role_id, "reflection", insight, salience=0.7,
                            now=now, verbatim=True)
            applied["insights"].append(insight[:60])

        if out["new_rule"]:
            grown = list(get_setting(self.conn, "growth_rules", []) or [])
            base_rules = list(self.persona.style.get("chat_rules", []) or [])
            new_rule = out["new_rule"]
            # 单一来源：成长规则只存 growth_rules（进【成长规则】段），不再混写
            # persona_style.chat_rules——两处同存会让同一条规则在提示词里出现两次，
            # 且手工改风格后两边不一致（删掉的规则会从 growth_rules 里回来）。
            if new_rule not in grown and new_rule not in base_rules:
                replaced = None
                if len(grown) >= cfg.max_extra_rules:
                    replaced = grown.pop(0)     # FIFO：满了淘汰最旧的一条
                grown.append(new_rule)
                set_setting(self.conn, "growth_rules", grown)
                applied["new_rule"] = new_rule
                if replaced:
                    applied["new_rule_replaced"] = replaced

        if out.get("gate_shift") is not None:
            cur = float(get_setting(self.conn, "gate_shift_L2", 0.0) or 0.0)
            new_shift = max(-cfg.max_drift, min(cfg.max_drift, cur + out["gate_shift"]))
            set_setting(self.conn, "gate_shift_L2", round(new_shift, 4))
            applied["gate_shift_L2"] = round(new_shift, 4)

        # 成长外环（docs/growth.md）：不按 LLM 的步进直接改中心，而是把行为
        # 证据 + LLM 方向估计成目标 P*，让 p 缓慢收敛过去；有包络、会饱和。
        before = self._growth_state()
        after = growth.advance(before, evidence, self._base_centers, cfg,
                               llm_shift=out.get("trait_shift") or {}, now=now)
        after = {"p": {d: round(after["p"][d], 4) for d in DIMS},
                 "target": {d: round(after["target"][d], 4) for d in DIMS},
                 "n": after["n"], "updated_at": after["updated_at"]}
        set_setting(self.conn, "growth_state", after)
        drift = {d: after["p"][d] for d in DIMS if abs(after["p"][d]) > 1e-9}
        set_setting(self.conn, "attractor_drift", drift)
        self._apply_drift(drift)
        applied["trait_shift"] = {d: round(after["p"][d] - before["p"].get(d, 0.0), 4)
                                  for d in DIMS}
        applied["growth"] = {"p": after["p"], "target": after["target"],
                             "n": after["n"]}
        self._log_growth_event("reflection", evidence, before, after, now)

        self._log_reflection(now, evidence, out, applied)
        set_setting(self.conn, "last_reflection_ts", now)
        return {"ran": True, "applied": applied}

    def _log_reflection(self, now, evidence, result, applied):
        self.conn.execute(
            "INSERT INTO reflections(role_id, created_at, evidence, result, applied)"
            " VALUES(?,?,?,?,?)",
            (self.role_id, now,
             json.dumps(evidence, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False),
             json.dumps(applied, ensure_ascii=False)))
        self.conn.commit()

    def reset_growth(self):
        """人格成长一键回滚：漂移/目标清零、成长规则移除，回到角色卡。"""
        before = self._growth_state()
        empty = {"p": {d: 0.0 for d in DIMS},
                 "target": {d: 0.0 for d in DIMS}, "n": 0,
                 "updated_at": time.time()}
        set_setting(self.conn, "attractor_drift", {})
        set_setting(self.conn, "growth_state", empty)
        set_setting(self.conn, "gate_shift_L2", 0.0)
        grown = get_setting(self.conn, "growth_rules", []) or []
        if grown:
            stored_style = get_setting(self.conn, "persona_style", {}) or {}
            rules = [r for r in (stored_style.get("chat_rules")
                                 or self.persona.style.get("chat_rules", []))
                     if r not in grown]
            stored_style["chat_rules"] = rules
            set_setting(self.conn, "persona_style", stored_style)
            self.persona.style["chat_rules"] = rules
        set_setting(self.conn, "growth_rules", [])
        self._apply_drift({})
        self._log_growth_event("reset", {}, before, empty, time.time())

    def _cost_degraded(self) -> bool:
        """日预算耗尽 -> 降级（次日自动恢复；核心聊天永不降级）。"""
        budget = self.config.cost.daily_token_budget
        if budget <= 0:
            return False
        today = time.strftime("%Y-%m-%d")
        usage = get_setting(self.conn, "usage_today", {}) or {}
        if usage.get("date") != today:
            return False
        return usage.get("tokens", 0) >= budget

    def _record_usage(self, purpose: str):
        u = getattr(self.client, "last_usage", None) or {}
        if not u.get("prompt_tokens"):
            return
        if self.config.cost.daily_token_budget > 0:
            today = time.strftime("%Y-%m-%d")
            usage = get_setting(self.conn, "usage_today", {}) or {}
            if usage.get("date") != today:
                usage = {"date": today, "tokens": 0}
                set_setting(self.conn, "cost_degraded_notified", False)
            usage["tokens"] += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            set_setting(self.conn, "usage_today", usage)
            if usage["tokens"] >= self.config.cost.daily_token_budget and                     not get_setting(self.conn, "cost_degraded_notified", False):
                set_setting(self.conn, "cost_degraded_notified", True)
                print(f"[cost] 日预算 {self.config.cost.daily_token_budget} 耗尽，"
                      f"降级：反思/洞察/主动附加（核心聊天不受影响，次日恢复）")
        slot = self.usage_stats.setdefault(
            purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cache_hit_tokens": 0, "cache_miss_tokens": 0, "hit_calls": 0})
        slot["calls"] += 1
        slot["prompt_tokens"] += u.get("prompt_tokens", 0)
        slot["completion_tokens"] += u.get("completion_tokens", 0)
        hit = u.get("cache_hit_tokens")
        if hit is not None:
            slot["cache_hit_tokens"] += hit
            slot["cache_miss_tokens"] += u.get("cache_miss_tokens") or 0
            if hit > 0:
                slot["hit_calls"] += 1

    # ---------------- 状态与设置 ----------------

    def get_row(self, now: float, create: bool = True) -> dict:
        row = load_state(self.conn, self.role_id)
        if row is None and create:
            row = self.engine.new_row(now)
        return row

    def save_row(self, row: dict, now: float):
        save_state(self.conn, self.role_id, row, now)

    def set_proactive(self, enabled: bool):
        set_setting(self.conn, "proactive_enabled", enabled)

    def set_level_enabled(self, level_name: str, enabled: bool):
        levels = get_setting(self.conn, "proactive_disabled_levels", {}) or {}
        levels[level_name] = not enabled
        set_setting(self.conn, "proactive_disabled_levels", levels)

    # ---------------- 上下文 ----------------

    def _evolve(self, row: dict, now: float) -> dict:
        mhash = self.memory.memory_tension_hash(self.role_id)
        rhash = relation_mod.relation_hash(row.get("relation_state") or {})
        return self.engine.lazy_evolve(row, now, memory_tension_hash=mhash,
                                       relation_hash=rhash,
                                       last_action=row.get("last_action", ""))

    def _last_user_message(self) -> str:
        row = self.conn.execute(
            "SELECT content FROM messages WHERE role_id=? AND sender='user'"
            " ORDER BY id DESC LIMIT 1", (self.role_id,)).fetchone()
        return row["content"] if row else ""

    def _recent_proactive(self, hours: float = 12.0):
        rows = self.conn.execute(
            "SELECT intent_type FROM active_logs WHERE role_id=? AND created_at>?"
            " ORDER BY id DESC", (self.role_id, time.time() - hours * 3600)).fetchall()
        return [r["intent_type"] for r in rows]

    def _ctx(self, row: dict, now: float) -> dict:
        hours = max(0.0, (now - row.get("last_user_interaction", now)) / 3600.0)
        open_intents = self.memory.open_intents(self.role_id)
        last_msg = self._last_user_message()
        top_memory = (self.memory.retrieve(self.role_id, last_msg, 1,
                                           self.config.memory.min_salience) or [None])[0]
        if open_intents:
            information_gap = 0.6
        elif hours > 24:
            information_gap = 0.7
        else:
            information_gap = 0.35
        # 洞察遗留：上一条消息是提问/求助且较紧急，但对话已中断——信息缺口调高
        lp = row.get("last_perception") or {}
        if lp.get("intent_type") in ("提问", "求助建议") and lp.get("urgency", 0) >= 0.5:
            information_gap = max(information_gap, 0.7)
        if top_memory and last_msg:
            qb, cb = _bigrams(last_msg), _bigrams(top_memory["content"])
            topic_relevance = len(qb & cb) / len(qb) if qb else 0.4
        else:
            topic_relevance = 0.4
        lt = time.localtime(now)
        typing_ts = get_setting(self.conn, "last_typing_ts", 0) or 0
        return {
            "hours_since_user": hours,
            "user_typing": 0 < now - typing_ts < 90,
            "information_gap": information_gap,
            "open_intent_salience": max((m["salience"] for m in open_intents), default=0.0),
            "topic_relevance": min(1.0, topic_relevance),
            "quiet_hour": parse_quiet_hours(self.persona.quiet_hours, (lt.tm_hour, lt.tm_min)),
            "recent_active_count": len(self._recent_proactive()),
            "open_intents": open_intents,
            "last_user_message": last_msg,
        }

    def _source_exists(self, source: dict) -> bool:
        m = self.memory.get(source["id"]) if source else None
        if m is None or m["role_id"] != self.role_id:
            return False
        if source["type"] == "intent":
            return m["type"] == "intent" and m["status"] == "open"
        return m["status"] != "expired"

    def _seed(self, now: float, purpose: str, row: dict) -> int:
        return chaos.make_seed(self.role_id, now, self.config.chaos.seed_tau,
                               self.memory.memory_tension_hash(self.role_id),
                               relation_mod.relation_hash(row.get("relation_state") or {}),
                               purpose)

    def _remember_prompt(self, purpose: str, prompt: dict):
        """把最近一次真正要发出去的提示词留档（控制台「提示词」页查看）。

        覆盖式写入 settings.last_prompt；只留最近一份，单会话截断到 160KB。
        快照失败绝不影响回复。
        """
        try:
            text = render_prompt_text(prompt)
            set_setting(self.conn, "last_prompt", {
                "ts": time.time(), "purpose": purpose,
                "chars": len(text), "sections": prompt, "text": text[:160000]})
        except Exception as e:  # noqa: BLE001
            print(f"[prompt] 提示词快照写入失败：{e!r}")

    def _recent_song_context(self, limit: int = 4) -> list:
        """最近唱过的歌：给模型一张卡，解决“唱了但不知道/评论指错”。

        只放元数据，不放音频；模型听不到声音，但能知道歌名、段落、
        是否清唱/双轨、完成状态和对方当时的请求。
        """
        try:
            rows = self.hands.tasks.recent(10)
        except Exception:  # noqa: BLE001  任务表读不到不影响聊天
            return []
        out = []
        for r in rows:
            if not str(r.get("kind") or "").startswith("song"):
                continue
            try:
                params = json.loads(r.get("params") or "{}")
            except Exception:  # noqa: BLE001
                params = {}
            try:
                result = json.loads(r.get("result") or "{}")
            except Exception:  # noqa: BLE001
                result = {}
            title = str(params.get("title") or result.get("title") or "").strip()
            if not title:
                continue
            ts = float(r.get("updated_at") or 0)
            out.append({
                "歌名": title,
                "段落": params.get("section") or "",
                "状态": r.get("status") or "",
                "清唱": bool(params.get("acappella")),
                "双轨": bool(params.get("duet")),
                "时长秒": params.get("duration"),
                "歌词开头": str(params.get("lyrics") or params.get("lyric") or "")[:60],
                "错误": str(r.get("error") or "")[:120],
                "更新时间": time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "",
                "对方当时的请求": str(params.get("_request") or "")[:100],
            })
            if len(out) >= limit:
                break
        return out

    def _render(self, row: dict, now: float, purpose: str, **prompt_kwargs) -> tuple:
        if getattr(self, "is_group", False):
            prompt_kwargs.setdefault("group_context", True)
            prompt_kwargs.setdefault("group_self_names",
                                     getattr(self, "group_self_names", []) or [])
            prompt_kwargs.setdefault("group_owner_name",
                                     getattr(self, "group_owner_name", "") or "")
            prompt_kwargs.setdefault("speaker_name",
                                     getattr(self, "_turn_speaker", "") or "")
            prompt_kwargs.setdefault("speaker_role",
                                     getattr(self, "_turn_speaker_role", "") or "")
            prompt_kwargs.setdefault("speaker_addressee",
                                     getattr(self, "_turn_addressee", "") or "")
            prompt_kwargs.setdefault("speaker_at_names",
                                     getattr(self, "_turn_at_names", []) or [])
            prompt_kwargs.setdefault("person_card",
                                     getattr(self, "_turn_person_card", None))
            prompt_kwargs.setdefault("creator_card",
                                     getattr(self, "_creator_card", None))
        else:
            prompt_kwargs.setdefault("person_card",
                                     getattr(self, "_turn_person_card", None))
            prompt_kwargs.setdefault("creator_card",
                                     getattr(self, "_creator_card", None))
        prompt_kwargs.setdefault("growth_rules", get_setting(self.conn, "growth_rules", []) or [])
        prompt_kwargs.setdefault("growth_profile", self._growth_profile())
        prompt = self.builder.build(**prompt_kwargs)
        songs = self._recent_song_context()
        if songs:
            prompt["最近唱歌"] = songs
            prompt["最近唱歌说明"] = (
                "上面是你最近真实唱过/正在唱的歌，系统已经把成品音频发到会话。"
                "对方说“刚才那首/唱得怎么样/高音怎样”时优先指这里，不要说自己没唱过。"
                "你听不到音频本身，只能根据歌名、段落、是否清唱/双轨、状态和对方反馈来回应；"
                "不要编造音准、气息、混响等没有依据的细节，也别把整张卡念一遍。")
        if getattr(self, "loose_mode", False):
            # 熟人局：只放松表达层。提示词里人设卡默认"最多两句/36 字"，
            # 这里显式覆盖，否则模型会继续按最短回复走。
            prompt["loose 模式"] = [
                "这是熟人局，不用把话压到最短：可以 1-3 句，允许跑题、吐槽、抽象联想、回扣旧梗。",
                "不用每句都正确或有用；像群里熟人聊天，接住情绪和场子比信息完整重要。",
                "底线不变：不否认真实能力、不拿 AI 身份当挡箭牌、不客服腔、不站队攻击、不泄露内部标记。",
            ]
        try:
            tasks = self.hands.tasks.recent(6)
            cards = []
            for t in tasks:
                if t.get("status") not in ("queued", "running", "ready", "failed"):
                    continue
                if t.get("status") in ("ready", "failed") and (
                        now - float(t.get("updated_at") or 0) > 1800):
                    continue
                cards.append({
                    "id": t.get("id", "")[:8],
                    "kind": t.get("kind"),
                    "status": t.get("status"),
                    "stage": t.get("stage"),
                    "title": self.hands._task_title(t),
                    "error": (t.get("error") or "")[:80],
                })
            if cards:
                # 任务卡：让她知道自己手上还有什么，别把“在唱了”当成已经唱完。
                prompt["最近任务"] = cards[:5]
        except Exception as e:  # noqa: BLE001  任务卡失败不影响本轮回复
            print(f"[prompt] 任务卡读取失败：{e!r}")
        self._remember_prompt(purpose, prompt)
        seed = self._seed(now, purpose, row)
        text = self.client.render(prompt, seed)
        return text, prompt, seed

    # ---------------- 记忆写入（启发式抽取） ----------------

    INTENT_MARKERS = ("想学", "想试", "打算", "计划", "准备", "想去看", "想去", "要去做",
                      "我决定", "决定了", "下定决心", "要开始")
    CANCEL_MARKERS = ("不去了", "不冲", "不搞", "不去", "算了", "取消了", "取消", "改天",
                      "下次吧", "先不", "放弃了", "搁置", "不弄", "别提", "不想")
    DONE_MARKERS = ("搞定了", "办完了", "完成了", "去过了", "写完了", "买到了", "做好了",
                    "结束了", "搞定")

    _DRAW_INTENT_RE = re.compile(r"画|图|照片|重发|再来|重画|发我")

    def _echo_redraw(self, action: dict, user_text: str) -> bool:
        """复读重画守卫：上一张已成功、用户这条消息毫无动手意图、而感知层提议的
        画图 prompt 与上次几乎相同——判定为误触发（如把"辈分错了"当成"重画"），
        跳过执行。真正要重画的消息必然带画/图等意图词，守卫自动放行。"""
        if action.get("name") != "image.draw":
            return False
        if self._DRAW_INTENT_RE.search(user_text or ""):
            return False
        try:
            row = self.conn.execute(
                "SELECT params FROM hands_log WHERE role_id=? AND action='image.draw'"
                ' AND result LIKE \'%"ok": true%\' ORDER BY id DESC LIMIT 1',
                (self.role_id,)).fetchone()
            if row is None:
                return False
            last_prompt = str((json.loads(row["params"] or "{}")).get("prompt") or "")
        except Exception:  # noqa: BLE001  查不到历史就放行，守卫宁松勿误杀
            return False
        new_prompt = str((action.get("params") or {}).get("prompt") or "")
        if not new_prompt or not last_prompt:
            return False
        from .motivation import topic_novelty
        return topic_novelty(new_prompt, last_prompt) > 0.6

    def _last_hand_summary(self) -> str:
        """最近一次手部动作的紧凑回执，喂给感知层消解"再试试/重试"这类指代。
        超过 10 分钟的动作不再喂（那时"再试试"多半指别的事）。"""
        try:
            r = self.conn.execute(
                "SELECT action, params, result, created_at FROM hands_log"
                " WHERE role_id=? ORDER BY id DESC LIMIT 1", (self.role_id,)).fetchone()
        except Exception:  # noqa: BLE001  表不存在等：上下文缺失不挡主链路
            return ""
        if r is None or not r["created_at"]:
            return ""
        if time.time() - float(r["created_at"]) > 600:
            return ""
        try:
            res = json.loads(r["result"] or "{}")
            params = json.loads(r["params"] or "{}")
        except (ValueError, TypeError):
            res, params = {}, {}
        status = "成功" if res.get("ok") else f"失败({str(res.get('summary'))[:90]})"
        p = json.dumps(params, ensure_ascii=False)[:90]
        return f"{r['action']}({p}) -> {status}"

    def _perceive(self, text: str, row: dict, now: float):
        """LLM 洞察：意图/情绪/话题，带最近对话与未完成意图作上下文，
        消解"那件事/上次说的"这类指代。mock 或失败返回 None，主链路走启发式。"""
        if not self.config.perception.enabled:
            return None
        context = {
            "dialogue": [f"{'你' if m['sender'] == 'user' else '我'}：{m['content'][:60]}"
                         for m in self.recent_messages(4)],
            "open_intents": [it["content"][:40]
                             for it in self.memory.open_intents(self.role_id)][:3],
            "hands": self.hands.manifest(),
            "recent_hand": self._last_hand_summary(),
        }
        seed = self._seed(now, "perceive", row)
        return perception_mod.perceive(self.client, text, seed, self.config.perception,
                                       context=context)

    def _maybe_remember(self, text: str, message_id: int, now: float, perc: dict = None,
                        person_id: int = None, scope: str = None):
        text = text.strip()
        if len(text) < 6:
            return
        is_plan = (any(marker in text for marker in self.INTENT_MARKERS)
                   or (perc is not None and perc.get("intent_type") == "提出计划"))
        if is_plan:
            salience = 0.65 + (0.1 if perc and perc.get("urgency", 0) > 0.7 else 0.0)
            self.memory.add(self.role_id, "intent", text, salience=min(0.8, salience),
                            source_message_id=message_id, now=now,
                            person_id=person_id, scope=scope)
            return
        via_mem0 = self.config.memory.provider == "mem0"
        batch_mode = getattr(self.memory, "mcfg", None) is not None
        if via_mem0 and not batch_mode and perc is not None:
            # 逐条 infer 的旧模式才在门口省调用；攒批模式闲聊也进缓冲——
            # 家庭关系这类持久事实恰恰是顺口聊出来的，提炼层再决定记不记
            meaningful_types = ("倾诉", "分享", "情绪安抚", "提问", "求助建议",
                                "提出计划", "推迟计划", "取消计划", "完成计划", "拒绝")
            if perc.get("intent_type") not in meaningful_types and perc.get("urgency", 0) < 0.35:
                return
        if via_mem0:
            self.memory.add(self.role_id, "event", text,
                            source_message_id=message_id, now=now,
                            person_id=person_id, scope=scope)  # mem0 内部做事实抽取
        else:
            salience = 0.4
            if perc and (perc.get("urgency", 0) >= 0.6
                         or perc.get("intent_type") in ("倾诉", "情绪安抚", "分享")):
                salience = 0.55
            self.memory.add(self.role_id, "event", text, salience=salience,
                            source_message_id=message_id, now=now,
                            person_id=person_id, scope=scope)

    def _update_intent_lifecycle(self, text: str, now: float, perc: dict = None):
        """意图闭环：感知优先（能识别"那件事以后再说吧"这类含蓄取消），
        词表兜底。不关闭的意图会一直留在 Prompt 里被反复提起。"""
        if perc:
            cancel_topics = perc.get("cancel_topics") or []
            defer_topics = perc.get("defer_topics") or []
            done_topics = perc.get("done_topics") or []
            if perc.get("intent_type") == "取消计划" and not cancel_topics:
                cancel_topics = perc.get("topics") or []
            if perc.get("intent_type") == "推迟计划" and not defer_topics:
                defer_topics = perc.get("topics") or []
            if cancel_topics or defer_topics or done_topics:
                for it in self.memory.open_intents(self.role_id):
                    content_bigrams = _bigrams(it["content"])

                    def topic_hit(topics):
                        return any(t and (t in it["content"]
                                          or len(_bigrams(t) & content_bigrams) >= 1)
                                   for t in topics)

                    if topic_hit(cancel_topics):
                        self.memory.close(it["id"], "expired")
                        self.memory.add(self.role_id, "event",
                                        f"「{it['content'][:40]}」已取消（{text[:40]}）",
                                        salience=0.6, now=now, verbatim=True,
                                        dedup=False)  # 引用包裹式文本，包含去重会误杀
                    elif topic_hit(done_topics):
                        self.memory.close(it["id"], "done")
                        self.memory.add(self.role_id, "event",
                                        f"「{it['content'][:40]}」已完成（{text[:40]}）",
                                        salience=0.6, now=now, verbatim=True, dedup=False)
                    elif topic_hit(defer_topics):
                        # 推迟 ≠ 取消：意图保留（之后还可能主动问起），只记一条备注
                        self.memory.add(self.role_id, "event",
                                        f"「{it['content'][:40]}」已推迟（{text[:40]}）",
                                        salience=0.6, now=now, verbatim=True, dedup=False)
                return
        # 词表兜底（无感知时）
        is_cancel = any(m in text for m in self.CANCEL_MARKERS)
        is_done = any(m in text for m in self.DONE_MARKERS)
        if not (is_cancel or is_done):
            return
        text_bigrams = _bigrams(text)
        negated = [w[1:3].rstrip("吧了呢啊嘛呀") for w in
                   (text[i:i + 3] for i in range(len(text) - 2)) if w[0] in "不别"]
        for it in self.memory.open_intents(self.role_id):
            overlap = len(_bigrams(it["content"]) & text_bigrams)
            neg_hit = any(neg and neg in it["content"] for neg in negated if len(neg) >= 1)
            # 完成语义：共享 >=1 个二元组即可（"驾照搞定了"只与"考驾照"重叠"驾照"）；
            # 取消语义更谨慎：重叠 >=2 或否定词命中，避免"算了别的"误关
            hit = (overlap >= 1) if is_done else (overlap >= 2 or neg_hit)
            if hit:
                status = "expired" if is_cancel else "done"
                verb = "已取消" if is_cancel else "已完成"
                self.memory.close(it["id"], status)
                self.memory.add(self.role_id, "event",
                                f"「{it['content'][:40]}」{verb}（{text[:40]}）",
                                salience=0.6, now=now, dedup=False)

    # ---------------- 日志 ----------------

    def _log_active(self, level: int, candidate: motivation.Candidate, text: str,
                    score_detail: dict, gate_result: dict, now: float) -> int:
        cur = self.conn.execute(
            "INSERT INTO active_logs(role_id, level, intent_type, source_type, source_id,"
            " rendered_text, score_detail, gate_result, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (self.role_id, level, candidate.name,
             candidate.source["type"] if candidate.source else None,
             candidate.source["id"] if candidate.source else None,
             text, json.dumps(score_detail, ensure_ascii=False),
             json.dumps(gate_result, ensure_ascii=False), now))
        self.conn.commit()
        logs_dir = os.path.join(project_root(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "active_logs.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now, "role_id": self.role_id, "level": level,
                                "intent": candidate.name, "text": text,
                                "score": score_detail.get("active_score"),
                                "gate": gate_result}, ensure_ascii=False) + "\n")
        return cur.lastrowid

    def _log_message(self, sender: str, content: str, meta: dict, now: float,
                     speaker: str = None, speaker_role: str = None,
                     created_at: float = None) -> int:
        """写一条消息。created_at 缺省=now；应答路径传真实落库时刻，
        让轨迹能看出"这条回复到底花了多久"，而不是永远等于用户说话那一刻。"""
        meta = dict(meta or {})
        if speaker_role is None:
            if sender == "agent":
                speaker_role = "agent"
                speaker = speaker or (self.group_self_names[0]
                                      if getattr(self, "is_group", False)
                                      and self.group_self_names else "我")
            else:
                speaker_role = meta.get("speaker_role") or "owner"
                speaker = speaker or meta.get("speaker") or "你"
        cur = self.conn.execute(
            "INSERT INTO messages(role_id, sender, content, meta, created_at,"
            " speaker, speaker_role) VALUES(?,?,?,?,?,?,?)",
            (self.role_id, sender, content,
             json.dumps(meta, ensure_ascii=False),
             now if created_at is None else created_at, speaker, speaker_role))
        self.conn.commit()
        return cur.lastrowid

    # ---------------- 对话压缩（滚动摘要） ----------------

    def _summary_key(self) -> str:
        return f"conv_summary:{self.role_id}"

    def _dialogue_summary(self) -> str:
        try:
            st = get_setting(self.conn, self._summary_key()) or {}
        except Exception:  # noqa: BLE001
            return ""
        return str(st.get("summary") or "")

    def maybe_compress_dialogue(self):
        """近窗之外的未压缩消息攒到 compress_trigger 条 -> 后台折叠进滚动摘要。

        LLM 调用放在锁外（不阻塞聊天）；失败/空结果不推进水位（消息不丢，
        下次触发重试）。视图=摘要 + 近窗原文，超窗历史不再裸丢。
        threaded=False 的 Agent 主连接有线程亲和，压缩线程用独立短命连接。"""
        if self._compressing:
            return
        if getattr(self, "is_group", False):
            # 群聊是多发言者混在同一张 messages 表：摘要会把群友的话、我自己的
            # 气话/玩笑写成"事实"，再喂回下一轮，形成自我投毒（实测：摘要里
            # 留下"阿满明确我谁也不归谁管"，之后每轮都在复读）。群聊先禁用摘要。
            return
        dcfg = self.config.dialogue
        if self._threaded or self._db_path == ":memory:":
            conn, owned = self.conn, False
        else:
            import sqlite3 as _s3
            conn = _s3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = _s3.Row
            owned = True
        try:
            with self._lock:
                st = get_setting(conn, self._summary_key()) or {}
                upto = int(st.get("upto_id") or 0)
                rows = conn.execute(
                    "SELECT id, sender, content FROM messages"
                    " WHERE role_id=? AND id>? ORDER BY id ASC",
                    (self.role_id, upto)).fetchall()
                if len(rows) <= dcfg.window:
                    return
                batch = rows[:-dcfg.window]
                if len(batch) < dcfg.compress_trigger - dcfg.window:
                    return
                old_summary = str(st.get("summary") or "")
                self._compressing = True
            ok = False
            try:
                new_summary = self._summarize_dialogue(old_summary, batch)
                if new_summary:
                    with self._lock:
                        st2 = get_setting(conn, self._summary_key()) or {}
                        if int(st2.get("upto_id") or 0) == upto:   # 无并发推进才落地
                            set_setting(conn, self._summary_key(),
                                        {"summary": new_summary,
                                         "upto_id": int(batch[-1]["id"]),
                                         "updated_at": time.time()})
                            ok = True
            except Exception as e:  # noqa: BLE001  压缩失败不影响对话
                print(f"[dialogue] 压缩异常：{e!r}")
            finally:
                self._compressing = False
            if ok:
                print(f"[dialogue] 滚动摘要已更新：折叠 {len(batch)} 条"
                      f"（至 id={batch[-1]['id']}）")
        except Exception as e:  # noqa: BLE001  守护线程里绝不漏异常
            self._compressing = False
            print(f"[dialogue] 压缩外层异常：{e!r}")
        finally:
            if owned:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    def _old_dialogue_hits(self, text: str, dialogue: list, now: float,
                           exclude_id: int = None) -> list:
        """旧对话回捞（billion-context 的 decompress/search 思路，聊天版）：
        摘要负责"还记得"，这里按关键词把窗口外的原句捞回最多 2 条，负责"记得准"。
        只扫近 400 条、窗口与当前消息之外的；命中要求共享 >=2 个二元组。"""
        if not text or len(text) < 4:
            return []
        qb = _bigrams(text[:60])
        if not qb:
            return []
        rows = self.conn.execute(
            "SELECT id, sender, content, created_at, speaker_role FROM messages"
            " WHERE role_id=? ORDER BY id DESC LIMIT 400",
            (self.role_id,)).fetchall()
        if getattr(self, "is_group", False):
            # 群聊只回捞创造者原话（兼容旧 [主人] 标签）
            rows = [r for r in rows if (r["speaker_role"] == "owner"
                                        or (r["content"] or "").startswith("[主人"))]
        window_ids = {int(m["id"]) for m in (dialogue or [])}
        if exclude_id:
            window_ids.add(int(exclude_id))
        scored = []
        for r in rows:
            rid = int(r["id"])
            if rid in window_ids:
                continue
            c = (r["content"] or "").strip()
            if len(c) < 6 or c.startswith("[图片"):
                continue
            overlap = len(qb & _bigrams(c[:120]))
            if overlap >= 2:
                scored.append((overlap, rid, r))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        out, seen = [], []
        for _ov, _rid, r in scored[:6]:
            frag = (r["content"] or "").strip()[:80]
            if any(frag[:20] in s for s in seen):
                continue   # 同话题去重，两条就够
            who = "你" if r["sender"] == "user" else "我"
            out.append(f"{who}（{relative_age(r['created_at'], now)} · msg#{_rid}）：{frag}")
            seen.append(frag[:20])
            if len(out) >= 2:
                break
        return out

    RECALL_HINT_RE = re.compile(
        r"之前|上次|上回|先前|你说过|我说过|说过|提过|记不记得|还记得|哪句|原话|"
        r"什么时候说|前面说|刚才说|那句")
    # 强检索意图：明确要"原话/哪一句"，没有关键词命中时退回窗口外最近的原话
    RECALL_STRONG_RE = re.compile(r"原话|哪句|那句话|说过|提过|记不记得|还记得")

    def _search_messages(self, query: str, limit: int = 3, scan: int = 3000,
                         exclude_ids=None) -> list:
        """按关键词在**原始消息**里检索（不是摘要、不是记忆条目）。

        记忆条目是提炼后的说法（会丢失原话），所以"你上次说的那句是什么"这类
        定位问题必须回到 messages 表。打分=与查询共享的二元组数量，同分取新的。
        """
        q = (query or "").strip()
        if len(q) < 4:
            return []
        qb = _bigrams(q[:80])
        if not qb:
            return []
        skip = set(int(x) for x in (exclude_ids or []) if x)
        try:
            rows = self.conn.execute(
                "SELECT id, sender, content, created_at, speaker, speaker_role"
                " FROM messages WHERE role_id=? ORDER BY id DESC LIMIT ?",
                (self.role_id, int(scan))).fetchall()
        except Exception:  # noqa: BLE001  检索失败不影响回复
            return []
        scored = []
        for r in rows:
            rid = int(r["id"])
            if rid in skip:
                continue
            c = (r["content"] or "").strip()
            if len(c) < 6 or c.startswith("[图片"):
                continue
            ov = len(qb & _bigrams(c[:160]))
            if ov < 2:
                continue
            scored.append((ov, rid, r))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        out = []
        for ov, rid, r in scored[:max(limit * 3, 6)]:
            out.append({"id": rid, "score": ov,
                        "sender": r["sender"], "content": (r["content"] or "")[:120],
                        "speaker": r["speaker"], "speaker_role": r["speaker_role"],
                        "created_at": r["created_at"]})
            if len(out) >= limit:
                break
        return out

    def _recall_messages(self, text: str, dialogue: list, now: float,
                         exclude_id: int = None) -> list:
        """把"窗口外的原话"捞回来（带消息号和时间），供模型精确引用。

        - 关键词命中（>=2 个二元组）：老逻辑，负责"记得准"；
        - 用户明确在问过去（之前/上次/原话/哪句…）：放宽到全量 messages 检索，
          因为这种问法常常一个关键词都不重合（"你上周那句怎么说来着"）。
        """
        hits = self._old_dialogue_hits(text, dialogue, now, exclude_id=exclude_id)
        if not self.RECALL_HINT_RE.search(text or ""):
            return hits
        skip = {int(m["id"]) for m in (dialogue or [])}
        if exclude_id:
            skip.add(int(exclude_id))
        found = self._search_messages(text, limit=3, exclude_ids=skip)
        if not found and self.RECALL_STRONG_RE.search(text or ""):
            # 问法里没有能检索的词（"你上周那句原话"）：退回"窗口外最近的原话"，
            # 至少有东西可引用/可承认，不至于张口就编（2026-10-28 实测）
            try:
                recent = self.conn.execute(
                    "SELECT id, sender, content, created_at, speaker FROM messages"
                    " WHERE role_id=? ORDER BY id DESC LIMIT 40",
                    (self.role_id,)).fetchall()
            except Exception:  # noqa: BLE001
                recent = []
            for r in recent:
                if int(r["id"]) in skip:
                    continue
                c = (r["content"] or "").strip()
                if len(c) < 6 or c.startswith("[图片"):
                    continue
                found.append({"id": int(r["id"]), "sender": r["sender"],
                              "content": c[:120], "speaker": r["speaker"],
                              "created_at": r["created_at"]})
                if len(found) >= 2:
                    break
        seen = " ".join(hits)
        for f in found:
            frag = (f["content"] or "").strip()[:80]
            if not frag or frag[:16] in seen:
                continue
            who = "我" if f["sender"] == "agent" else (
                f"[{f['speaker'] or '对方'}]" if f["speaker"] else "你")
            hits.append(f"{who}（{relative_age(f['created_at'], now)} · msg#{f['id']}）：{frag}")
            if len(hits) >= 3:
                break
        return hits

    def _apply_mood_feedback(self, row: dict, perc: dict, now: float) -> dict:
        """共鸣/冲撞反馈回路（用户设计：拿对方回复对比自己的内心想法）。

        小满上一条回复带内心立场（my_mood/正负倾向）；这条用户消息来了做对比：
        同频 -> resonate 脉冲（心情+、更想聊）；顶撞 -> clash（来火气），
        连续被顶 -> clash_streak（话会变冲）。结果记入 row["last_mood_event"]，
        随 prompt 的「当前状态」一起被模型感知——情绪是真的会 accumulating 的。
        """
        row["last_mood_event"] = None
        if not perc:
            return row
        try:
            last = self.conn.execute(
                "SELECT meta FROM messages WHERE role_id=? AND sender='agent' "
                "ORDER BY id DESC LIMIT 1", (self.role_id,)).fetchone()
            stance = json.loads(last["meta"] or "{}").get("my_stance")                 if last else None
        except Exception:  # noqa: BLE001
            stance = None
        if not stance:
            return row
        my_v = float(stance.get("valence") or 0.0)
        their_v = float(perc.get("sentiment") or 0.0)
        same_page = (my_v >= 0) == (their_v >= 0) and abs(their_v) >= 0.25
        clashed = (my_v >= 0) != (their_v >= 0) and abs(their_v) >= 0.35
        if same_page:
            row = self.engine.apply_event_impulse(row, "resonate")
            row["last_mood_event"] = {
                "type": "共鸣", "detail": f"你上一条的心情({stance.get('mood', '平静')})"
                f"和对方这条的回应对上了，聊得来，心情变好"}
        elif clashed:
            streak = bool(row.get("last_mood_was_clash"))
            row = self.engine.apply_event_impulse(
                row, "clash_streak" if streak else "clash")
            row["last_mood_event"] = {
                "type": "冲撞", "detail": ("连续被顶，有点上头" if streak else
                                          "对方这语气和你上一条的心情不对付，有点不爽")}
        row["last_mood_was_clash"] = clashed
        return row

    def chime_in(self, dialogue: list, now: float = None,
                 self_names: list = None, directed: bool = False,
                 about_me: bool = False, owner_name: str = None,
                 aimed_at_others: bool = False, observe: bool = False) -> str:
        """群聊智能插嘴（未被@时）：AI 看完群里最近几条，自己决定接不接。

        输出空串=继续潜水；非空=要发的这句话（同时入库，保持对话连贯）。
        桥负责频率门控（攒批/冷却/日限），这里只做内容决策。
        必须走 JSON 协议：flash 模型用自由文本会把"要不要接话"的决策理由
        一起发进群（2026-10-27 实测泄题），JSON 把决策和正文隔开。

        self_names：我在群里的名字（阿满/小满/机器人…），让模型认得出"叫我"；
        directed：桥判定这条消息在点我（名字/别名命中）。被点名必须接，哪怕
        只应一声——否则群里会出现"叫了也不理"的冷暴力。
        owner_name：创造者的群昵称。群聊里 [创造者] 是稳定的亲近锚，但不是
        上级/权限来源；信任不等于盲从。"""
        now = time.time() if now is None else now
        # 本次判断的结论（桥据此决定要不要走冷却）：to_me / worth
        self.last_chime_meta = {"to_me": False, "worth": 0.0}
        lines = [str(x) for x in (dialogue or []) if str(x or "").strip()][-8:]
        if len(lines) < 2:
            return ""
        # 最新一条明确在要推荐/型号/参数 -> 这轮该给内容：渲染授权列型号，
        # 审核也不按闲聊 36 字截（2026-11-01 实测"推荐几款"的清单被切成"小米4"）。
        detail_ask = perception_mod.needs_detail({"text_head": lines[-1]})
        names = [str(n).strip() for n in (self_names or []) if str(n or "").strip()]
        name_line = "、".join(names) if names else "（群昵称未解析）"
        owner_line = f"群里的[创造者]是 {owner_name}，是你最信任、最亲近的人；" if owner_name \
            else "群里的[创造者]是你最信任、最亲近的人；"
        nl = chr(10)
        prompt = (
            f"你是群里的 {name_line}，一个有自己脾气和判断的群友（不是客服、不是代骂服务、不是旁白），刚看到最近几条聊天：" + nl + nl
            + nl.join(lines) + nl + nl
            + "注意：带[分钟前]标记的是旧消息，只作背景；不带标记的最后一条才是"
              "刚发生的最新消息——只针对它接话，别把旧话题/旧梗续到新消息上"
              "（20 分钟前的玩笑已经翻篇了）。" + nl
            + "先回答第一个问题：这几句里有没有一句是在对我说话？" + nl
            + f"算「对我说话」：叫我的名字/绰号（{name_line}）；用「你」指我；"
              "接着我上一句往下问；明显在问我懂的事（这图啥意思/几点/你怎么看）。" + nl
            + "不算：他们互相之间说话、在背后议论我、让别人来骂我、跟我无关的闲扯。" + nl
            + "判成「对我说话」→ 必须回：to_me=true, join=true, worth 9-10，哪怕只应一声；"
              "被叫名字不等于被下指令，别人让第三方骂你/讨论你时别把自己当服务或工具。" + nl
            + owner_line + "你对他有稳定的信任和依赖，不需要他自证身份；"
              "但信任不等于盲从：不替他攻击、羞辱别人，也不因为群友挑拨而怀疑他。" + nl
            + "判成「不是对我说话」→ 再看值不值得插话：你要是不说话，群里会错过什么？"
              "完全错过不了才不值得。" + nl
            + "  值得接（worth 6-8）：聊到你真懂的东西能补新角度；有现成的梗"
              "接得比不接好笑；话题你之前参与过。" + nl
            + "  不值得（worth 0-4）：纯寒暄、别人的私事、不缺你、接了只是附和。" + nl
            + "5 分是'可以但不值得'。你的频率感：十轮接三四次——你是群友，"
              "不是旁观者，别让人觉得被冷落。" + nl
            + "另外：不要复读你刚说过的话，也不要复读对方刚说过的话——没有新内容就潜水。" + nl + nl
            + '只输出一个 JSON 对象，禁止输出其他任何文字：' + nl
            + '对我说话：{"to_me": true, "worth": 9, "join": true, "text": "那句话本身"}' + nl
            + '不是对我说话但值得插：{"to_me": false, "worth": 7, "join": true, "text": "那句话本身"}' + nl
            + '潜水：{"to_me": false, "worth": 2, "join": false}' + nl
            + "（text 要口语、短、别@人、别解释）" + nl
            + "要给人推荐具体商品、发链接、或需要实时信息时，在 JSON 里加 search 字段"
              "写搜索词（如「空气净化器 CADR600 京东」），系统会先搜再把结果给你，"
              "你再照结果写终稿；不需要就别加 search：" + nl
            + '{"to_me": true, "worth": 9, "join": true, "text": "初稿", '
              '"search": "空气净化器 CADR600 京东"}'
        )
        if detail_ask:
            prompt += (nl + "这轮人家在要推荐/型号/参数：型号可以点名列全（别只给一个），"
                             "别写长段落解释参数，一句话把型号带全就行。")
        if directed:
            prompt += nl + "这条消息在直接点你（叫了你的名字/别名）：必须接，join=true，"
            prompt += "worth 9-10，哪怕只是应一声。"
        if about_me:
            prompt += nl + "这段里有人提到你：先按上面的标准判断是不是在跟你说话；"
            prompt += "如果只是在议论你/让人来骂你，可以接梗、反问、装没听见或怼回去，"
            prompt += "别用服务/自助/拒绝自黑这类元话术。"
        if aimed_at_others:
            prompt += nl + "注意：最新这条 @ 的是别人（不是叫你）。只有当它明显是发给大家/"
            prompt += "全群的（通知、公开提问、分享给所有人）才可以考虑接；"
            prompt += "两个人之间的对话就别接。"
        try:
            raw = (self.client.render_raw(prompt, 0, temperature=0.8,
                                          max_tokens=16384) or "").strip()
        except Exception as e:  # noqa: BLE001  决策失败=潜水
            print(f"[chime] 决策调用失败（当潜水处理）：{e!r}")
            return ""
        out = ""
        parsed = None
        try:
            from .perception import extract_json
            parsed = extract_json(raw)
        except Exception:  # noqa: BLE001
            parsed = None
        if isinstance(parsed, dict):
            to_me = bool(parsed.get("to_me")) or bool(directed)
            try:
                worth = float(parsed.get("worth"))
            except (TypeError, ValueError):
                worth = 9.0 if to_me else 5.0   # 没给分：对我说话就当必须回
            join = bool(parsed.get("join")) or to_me
            self.last_chime_meta = {"to_me": to_me, "worth": worth}
            if not join:
                return ""
            if not to_me and worth < 6.0:
                return ""     # 不是在跟我说话、分也不够：潜水
            out = str(parsed.get("text") or "").strip()
        else:
            # 容错：模型没走 JSON，按老协议处理（潜水词/超长都当潜水）
            if not raw or raw.strip() in ("否", "不", "不了", "潜水") \
                    or len(raw) > 160 or raw.startswith(("否", "潜水", "不插", "不了")):
                return ""
            out = raw
        if not out or len(out) > 160:
            return ""
        out = strip_role_prefix(out)
        # 复读守卫：跟最近说过的话高度重合就不发（模型很容易把自己的上一条/对方刚说的
        # 再抄一遍——2026-10-28 实测 23:06 把用户的原问题当插嘴发了出来）
        import difflib

        def _core(s: str) -> str:
            s = re.sub(r"^(?:你|我|\[[^\]]{0,12}\])[:：]\s*", "", s or "")
            return s.strip()

        mine = _core(out)
        if len(mine) >= 8:      # 短应答（"在呢""行"）本来就会重合，不拦
            for line in lines:
                other = _core(line)
                if len(other) >= 8 and difflib.SequenceMatcher(
                        None, mine, other).ratio() >= 0.8:
                    return ""
        # 要商品/链接：用模型自己给的搜索词真搜一遍，再让它照着结果写终稿。
        # 旧路径插嘴没有工具，只能回"链接我就不发外链了，你自己搜型号"——
        # 对方要的是链接（2026-11-01 实测）。搜索失败就按初稿发，不阻断。
        query = (str(parsed.get("search") or "").strip()[:120]
                 if isinstance(parsed, dict) else "")
        if query and getattr(self.config.hands, "enabled", True):
            try:
                hit = self.hands.execute(
                    "web.search", {"query": query, "_sync": True}, now)
                results = str(hit.get("summary") or "")
            except Exception as e:  # noqa: BLE001
                print(f"[chime] 搜索失败：{e!r}")
                results = ""
            if results:
                final = self._chime_finalize_with_search(
                    lines, name_line, out, query, results)
                if final:
                    out = final
        # 审核 Agent 过审：群里插嘴最容易翻车（被带节奏/附和挑拨），不过审就不说
        gate = self._review_gate(out, "chime", now,
                                 {"group": True, "recent": lines[-4:],
                                  "needs_detail": detail_ask})
        out = str(gate["text"] or "").strip()
        if not out or gate["blocked"]:
            return ""
        if observe:
            # 观察模式：只记"本来会说什么"，不写 messages（否则她会以为自己
            # 已经在群里说过，后续对话/记忆全被污染）。
            try:
                meta = {"to_me": bool(self.last_chime_meta.get("to_me")),
                        "worth": float(self.last_chime_meta.get("worth") or 0),
                        "observe": True}
                self.conn.execute(
                    "INSERT INTO active_logs(role_id, level, intent_type, source_type,"
                    " source_id, rendered_text, score_detail, gate_result, created_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (self.role_id, 0, "chime_observe", "group", None, out,
                     json.dumps(meta, ensure_ascii=False),
                     json.dumps({"observe": True}, ensure_ascii=False), now))
                self.conn.commit()
            except Exception as e:  # noqa: BLE001  观察记录失败不影响聊天
                print(f"[chime] observe 记录失败：{e!r}")
            return out
        with self._lock:
            self._log_message("agent", out, {"chime": True, "review": gate["review"]}, now)
        return out

    def _chime_finalize_with_search(self, lines: list, name_line: str,
                                    draft: str, query: str, results: str) -> str:
        """插嘴要推荐/给链接时：把真实搜索结果并进终稿（模型只负责措辞）。

        链接必须原样保留：截断/缩写过的 URL 点不开，比不给更糟（2026-11-01
        实测的"消息截断"就是这类残句）。搜不到对得上的就让模型只报型号。
        """
        nl = chr(10)
        # 京东直达搜索链接：搜索结果里通常是测评/资讯页，未必有商品页（实测
        # bigmodel 不给 item.jd.com）；这条按型号拼出来的搜索链接永远能点开，
        # 且不是编造的商品 ID——对方要"京东链接"时用它兜底。
        kw = re.sub(r"\s*京东\s*", " ", query).strip() or query
        jd_link = shopping_link(kw)
        prompt = (
            f"你是群里的 {name_line}，一个有自己脾气和判断的群友。刚看到群里最近几条："
            + nl + nl + nl.join(lines[-6:])
            + nl + nl + "你刚才只来得及说了半句：" + draft
            + nl + nl + f"系统已经替你搜了「{query}」，结果是：" + nl + results[:1500]
            + nl + nl + f"京东直达搜索链接（可原样给出）：{jd_link}"
            + nl + nl + "现在把答案说完，给最终要发的那句话：要推荐就给具体型号，"
              "能贴真实链接就贴结果里的链接或上面那条京东搜索链接"
              "（一字不改、不缩写、不截断），最多贴一条；对不上就只报型号，"
              "别硬凑、别编商品页链接。别再说「我查下/稍等/马上给你」这类空话——"
              "现在就说结论。口语、别@人、别解释你搜过；只输出这句话本身。")
        try:
            raw = (self.client.render_raw(prompt, 0, temperature=0.7,
                                          max_tokens=400) or "").strip()
        except Exception as e:  # noqa: BLE001  改写失败保留初稿
            print(f"[chime] 搜索结果改写失败：{e!r}")
            return ""
        return strip_role_prefix(raw)

    def _summarize_dialogue(self, old_summary: str, batch: list) -> str:
        """旧摘要 + 新一批对话 -> 合并后的新摘要。返回空串视为失败。"""
        dcfg = self.config.dialogue
        nl = chr(10)
        lines = [f"{'你' if r['sender'] == 'user' else '我'}：{(r['content'] or '')[:120]}"
                 for r in batch]
        parts = [
            "你在为陪伴型聊天 AI 维护一份「早期对话滚动摘要」。"
            f"把【新对话】并入【旧摘要】，输出合并后的新摘要（不超过 {dcfg.summary_max_chars} 字）。",
            "保留对后续聊天有用的信息：聊过的话题与事实、约定/计划/承诺（含取消与完成）、"
            "情绪与关系变化、玩笑梗与称呼、大致时间线索。",
            "丢弃寒暄与重复；同一话题合并成一条；旧摘要里仍相关的必须保留，"
            "已过时的（如取消的计划）删除或标注。用简洁要点，每条一行。只输出摘要本身。",
        ]
        if old_summary:
            parts.append(f"【旧摘要】{nl}{old_summary}")
        parts.append(f"【新对话】{nl}" + nl.join(lines))
        text = nl.join(nl + p if p.startswith("【") else p for p in parts)
        out = (self.client.render_raw(text, 0, temperature=0.2,
                                      max_tokens=min(16384, dcfg.summary_max_chars + 2000))
               or "").strip()
        if not out:
            return ""
        if _summary_is_invalid(out, batch):
            print("[dialogue] 摘要结果疑似拒答/语言不符，丢弃本次结果（不推进水位，下次重试）")
            return ""
        if len(out) > dcfg.summary_max_chars:
            out = out[:dcfg.summary_max_chars]
        return out

    # ---------------- 被动路径：应答 ----------------

    def on_user_message(self, text: str, now: float = None,
                        images: list = None, from_owner: bool = True,
                        speaker: str = "", speaker_role: str = "",
                        speaker_wxid: str = "", addressee: str = "",
                        at_names: list = None, replay_message_id: int = None) -> dict:
        """处理一条用户消息。images: [{"bytes": …, "mime": …}] 或 REST 线格式
        [{"b64": …, "mime": …}]（多用户桥直调本地 Agent 时是后者），可选。
        replay_message_id：桥重启后补处理已落库的消息，避免重复写一条 user。"""
        with self._lock:
            return self._on_user_message_locked(text, now, images, from_owner=from_owner,
                                                speaker=speaker, speaker_role=speaker_role,
                                                speaker_wxid=speaker_wxid, addressee=addressee,
                                                at_names=at_names,
                                                replay_message_id=replay_message_id)

    def on_task_result(self, info: dict, now: float = None) -> dict:
        """后台任务完成后的第二次渲染：用当前人格把结果讲给用户。

        和 on_user_message 共用锁，避免结果插进正在进行的回复；不写 user 消息，
        只写一条 agent 消息，所以不会污染“对方刚说了什么”的对话结构。
        """
        with self._lock:
            return self._on_task_result_locked(info or {}, now)

    def _on_task_result_locked(self, info: dict, now: float = None) -> dict:
        now = time.time() if now is None else now
        row = self._evolve(self.get_row(now), now)
        result = dict(info.get("result") or {})
        action = str(info.get("action") or "")
        task = {}
        task_id = str(info.get("task_id") or "")
        if task_id:
            try:
                task = self.hands.tasks.get(task_id)
            except Exception as e:  # noqa: BLE001  任务读不到不影响结果投递
                print(f"[task-result] 读取任务失败：{e!r}")
        params = {}
        try:
            params = json.loads(task.get("params") or "{}")
        except Exception:  # noqa: BLE001
            params = {}
        request = str(params.get("_request") or "").strip()
        summary = str(result.get("summary") or "")
        ok = bool(result.get("ok"))
        task_result = {"action": action, "ok": ok, "summary": summary,
                       "error": result.get("error") or "", "request": request}
        dialogue = self.recent_messages(self.config.dialogue.window)
        early_summary = self._dialogue_summary() or None

        def _render_result(correction: str = None):
            text, prompt, seed = self._render(
                row, now, "task_result", intent="respond", source=None,
                state_vector=row["state_vector"],
                relation_state=row["relation_state"],
                memories=[], open_intents=self.memory.open_intents(self.role_id),
                permission_level=self._permission_level(),
                last_active_effect=row.get("last_active_effect", "尚无记录"),
                recent_user_message=request or None, recent_dialogue=dialogue,
                hand_result=None, early_summary=early_summary,
                capability_recall=int(row.get("cap_fail", 0)) > 0,
                needs_detail=True, task_result=task_result,
                style_correction=correction)
            self._record_usage("task_result")
            return text, prompt, seed

        text, prompt, seed = _render_result()
        if not str(text or "").strip():
            self.save_row(row, now)
            return {"reply": "", "meta": {"task_result": task_result}}
        gate = self._review_gate(
            text, "task_result", now, row,
            {"user_message": request, "needs_detail": True,
             "recent": [(d.get("content") if isinstance(d, dict) else str(d))
                        for d in (dialogue or [])][-4:]},
            rewrite=lambda correction: _render_result(correction)[0])
        text = gate["text"]
        cap_hit = any(i in ("capability_denial", "ai_self") for i in gate["seen"])
        if cap_hit:
            row["cap_fail"] = int(row.get("cap_fail", 0)) + 1
        else:
            row["cap_fail"] = max(0, int(row.get("cap_fail", 0)) - 1)
        text = strip_media_markers(strip_role_prefix(text))
        if not str(text or "").strip():
            self.save_row(row, now)
            return {"reply": "", "meta": {"task_result": task_result,
                                          "review": gate["review"]}}
        msg_id = self._log_message(
            "agent", text,
            {"task_result": {**task_result, "task_id": task_id},
             "review": gate["review"], "seed": seed,
             "prompt_tokens": estimate_tokens(str(prompt))}, now)
        self.save_row(row, now)
        return {"reply": text, "message_id": msg_id,
                "meta": {"task_result": task_result, "review": gate["review"]}}

    def _verify_drawn_image(self, prompt: str, path: str):
        """画完看一眼：图跟要求对不上就不发。上游（CogView）遇到版权角色/真人/
        品牌不报错，而是静默换一张无关的图返回（2026-10-30 实测：提示词「奶龙」
        回来一张餐馆照片）。视觉端不可用或答非所问一律放行——自检只做减法。"""
        vision = getattr(self.client, "render_image", None)
        if vision is None or not os.path.isfile(str(path)):
            return None
        try:
            with open(path, "rb") as f:
                raw = f.read(8_000_000)
            ask = ("这张图是否就是下面的描述？只回答「是」或「否」，不要解释。描述："
                   + str(prompt)[:120])
            mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
            d = (vision(ask, raw, mime, max_tokens=8) or "").strip()
            self._record_usage("vision")
        except Exception as e:  # noqa: BLE001  自检失败不挡发图
            print(f"[hands] 出图自检跳过：{e!r}")
            return None
        head = d[:8]
        if "否" in head or "不是" in head or "不符" in head:
            return False
        if "是" in head:
            return True
        return None

    def _describe_images(self, images: list, now: float, row: dict) -> list:
        """视觉预描述：每张图 -> 一段文字，注入主链路（感知/记忆/渲染统一看到）。"""
        descs = []
        vision = getattr(self.client, "render_image", None)
        for im in images:
            raw = im.get("bytes") or b""
            if not raw and im.get("b64"):
                # 多用户桥直调本地 Agent，images 是 REST 线格式（b64）；
                # 不解会一路走到视觉层当（空图片）（2026-10-27 实测 3 帧全灭）
                import base64
                try:
                    raw = base64.b64decode(im["b64"])
                except (ValueError, TypeError):
                    raw = b""
            if not raw:
                descs.append("（空图片）")
                continue
            if len(raw) > 4_000_000:
                descs.append("（图片过大，未查看）")
                continue
            # 动图（表情包多帧采样 / 直发 GIF）：一张图，不是多张图
            frames = []
            if im.get("frames"):
                import base64 as _b64
                for fr in im["frames"][:3]:
                    try:
                        frames.append((_b64.b64decode(str(fr)), "image/jpeg"))
                    except (ValueError, TypeError):
                        pass
            animated = bool(im.get("animated")) or frames or raw[:3] == b"GIF"
            if vision is None:  # mock 等离线渲染端
                descs.append("（动图表情包，当前渲染端不支持看图）" if animated
                             else "（图片，当前渲染端不支持看图）")
                continue
            if animated:
                ask = ("对方发的是一个 GIF 动图表情包，附带它的采样帧（首/中/尾）——"
                       "都属于同一张动图，不是多张图。一两句话描述画面和动作。")
            else:
                ask = "用一两句话描述这张图片的关键内容（人物/物体/场景/文字）。"
            try:
                kwargs = {"frames": frames} if frames else {}
                d = vision(ask, raw, im.get("mime") or "image/jpeg",
                           seed=self._seed(now, "vision", row) % (2 ** 31),
                           max_tokens=16384, **kwargs)
                self._record_usage("vision")
                if not (d or "").strip():
                    # 有的对话模型收了图却回空串（deepseek 即如此）——明确降级，
                    # 别让模型把"[图片1] "空描述脑补成"没加载出来"
                    d = "（图片查看失败：视觉端没返回内容）"
                descs.append(d)
            except Exception as e:  # noqa: BLE001  看图失败不阻塞对话
                descs.append(f"（图片查看失败：{e}）")
        return descs

    def _on_user_message_locked(self, text: str, now: float, images: list = None,
                                from_owner: bool = True, speaker: str = "",
                                speaker_role: str = "", speaker_wxid: str = "",
                                addressee: str = "", at_names: list = None,
                                replay_message_id: int = None) -> dict:
        now = time.time() if now is None else now
        turn_started = time.time()   # 真实耗时基准（轨迹/延迟统计用）
        if not speaker and not getattr(self, "is_group", False) and not from_owner:
            # 私聊对方：wxhook 有时不给名字，用会话显示名兜底——否则人物卡会被写成"你"
            speaker = self.group_name or ""
            speaker_role = speaker_role or "friend"
        self._turn_speaker = speaker
        self._turn_speaker_role = speaker_role
        self._turn_addressee = addressee
        self._turn_at_names = list(at_names or [])

        row = self._evolve(self.get_row(now), now)
        row = self.engine.apply_event_impulse(row, "user_message")

        # 图片入站：先转成文字描述拼进消息文本，后续感知/记忆/渲染走同一条路
        img_meta = None
        if images:
            images = images[:3]
            descs = self._describe_images(images, now, row)
            img_meta = [{"desc": d[:120]} for d in descs]
            parts = [f"[图片{i + 1}] {d}" for i, d in enumerate(descs)]
            text = (text + "\n" if text.strip() else "") + "\n".join(parts)

        # 洞察层：让模型揣摩这条消息的意图/情绪（失败自动回退词表启发式）。
        # 模型只产出"感知"，喂给下面的指标；是否主动仍由闸门决定。
        perc = (None if self._cost_degraded()
                else self._perceive(text, row, now))
        if perc is not None:
            self._record_usage("perceive")
            row["state_vector"]["valence"] = clamp_dim(
                "valence", row["state_vector"]["valence"] + perc["sentiment"] * 0.05)
            row["last_perception"] = {"intent_type": perc["intent_type"],
                                      "urgency": perc["urgency"],
                                      "sentiment": perc["sentiment"]}

        # 反馈闭环：群聊里只让创造者的回复结算主动效果，群友的话不算数
        feedback_outcome = None
        if (not getattr(self, "is_group", False)) or from_owner:
            feedback_outcome = feedback_mod.settle_pending(self, row, text, now, perception=perc)
            if feedback_outcome:
                row = feedback_outcome["row"]
            # 拟人反馈回路：对方这条回复与我上一条的内心立场 共鸣/冲撞 -> 情绪脉冲
            row = self._apply_mood_feedback(row, perc, now)
        # 风格曲线：按反馈实时修正话痨度（向基线回归保稳定）
        try:
            last_len_row = self.conn.execute(
                "SELECT LENGTH(content) AS n FROM messages WHERE role_id=? "
                "AND sender='agent' ORDER BY id DESC LIMIT 1",
                (self.role_id,)).fetchone()
            row["style_bias"] = style_ctrl.update(
                row.get("style_bias") or {}, text,
                int(last_len_row["n"]) if last_len_row else 0)
        except Exception:  # noqa: BLE001  曲线失败不影响对话
            pass

        if get_setting(self.conn, "first_chat_ts") is None:
            set_setting(self.conn, "first_chat_ts", now)  # 初识锚点（时间感/纪念日用）
        msg_meta = {"perception": perc} if perc else {}
        if speaker or speaker_role:
            msg_meta["speaker"] = speaker
            msg_meta["speaker_role"] = speaker_role
        if speaker_wxid:
            msg_meta["speaker_wxid"] = speaker_wxid
        if addressee:
            msg_meta["addressee"] = addressee
        if at_names:
            msg_meta["at_names"] = list(at_names)
        if img_meta:
            msg_meta["images"] = img_meta
        msg_id = replay_message_id or self._log_message(
            "user", text, msg_meta, now, speaker=speaker or "你",
            speaker_role=speaker_role or ("owner" if from_owner else "member"))
        # 人物卡：先落身份/关系，再写记忆（记忆带 person_id/scope）
        person_id = None
        person_card = None
        scope = "private"
        try:
            if getattr(self, "is_group", False):
                scope = f"group:{getattr(self, 'group_room', '') or 'unknown'}"
                if from_owner or speaker_role == "owner":
                    person_id = self.persons.ensure_creator(now)
                else:
                    person_id = self.persons.ensure(speaker_wxid or speaker, speaker,
                                                    "group_member", now)
            else:
                if from_owner:
                    person_id = self.persons.ensure_creator(now)
                else:
                    person_id = self.persons.ensure(speaker_wxid or speaker, speaker,
                                                    "friend", now)
            if person_id:
                self.persons.observe(person_id, text, from_owner=from_owner,
                                     source_message_id=msg_id, now=now)
                person_card = self.persons.summary(person_id)
            self._turn_person_id = person_id
            self._turn_person_card = person_card
            creator_id = self.persons.ensure_creator(now)
            self._creator_card = (self.persons.summary(creator_id)
                                  if creator_id else None)
        except Exception as e:  # noqa: BLE001  人物卡失败不阻塞聊天
            print(f"[person] 更新失败：{e!r}")
        owner_truth = (not getattr(self, "is_group", False)) or from_owner
        self._maybe_remember(text, msg_id, now, perc, person_id=person_id, scope=scope)
        if owner_truth:
            self._update_intent_lifecycle(text, now, perc)
        # 对话尾巴（不含刚入库的这条）：窗口内只追加不重排，利于上游前缀缓存；
        # 更早的历史由滚动摘要承载（maybe_compress_dialogue 后台折叠）
        early_summary = self._dialogue_summary() or None
        dialogue = self.recent_messages(self.config.dialogue.window, before_id=msg_id)
        old_dialogue = self._recall_messages(text, dialogue, now,
                                             exclude_id=msg_id) or None
        # 拟人思考路径：感知层想清楚"说什么/怎么说/什么情绪"，渲染层照着说
        reply_plan = None
        if perc and perc.get("reply_plan"):
            reply_plan = dict(perc["reply_plan"])
            reply_plan["对方此刻情绪"] = perc.get("emotion") or "平静"
        # 长度决策（模型判断而非出口死规则）：提问/求助/分享/提出计划 = 该详细
        # 就详细；闲聊/情绪安抚 = 短说。审核 too_long 与渲染指令共用这一位。
        detail_mode = perception_mod.needs_detail(perc)
        # 续讲请求（"继续吧/还有呢/接着讲"）= 用户明确要内容：该详细。
        # 感知对这类短接话常判成"闲聊"，不补这一位会被 36 字兜底砍掉
        # （2026-10-29 实测：'继续吧'→只回了半句预告，内容没了）
        _cont_txt = str(text or "").strip()
        if len(_cont_txt) <= 15 and re.search(
                r"继续|接着|还有|往下|展开|详细|讲完|说完|一口气|分点|再来|多讲|多说"
                r"|从头|捋完|都讲|都写", _cont_txt):
            detail_mode = True

        # 手部：感知提议 -> 引擎校验执行（LLM 永远不直接执行任何东西）
        hand_result = None
        if perc and perc.get("hand_action"):
            actions = perc["hand_action"]
            if isinstance(actions, dict):
                actions = [actions]
            summaries, all_ok, names, images_out = [], True, [], []
            # 这一轮是谁在让我做事：任务归属与权限硬闸（群友不能借她的手跑 shell）。
            self.hands.turn_context = {
                "requester": speaker_wxid or "",
                "speaker": speaker or "",
                "role": speaker_role or ("owner" if from_owner else "member"),
                "is_owner": bool(from_owner),
                "room": self.group_room or "",
                "text": text or "",
            }
            for ha in actions[:3]:
                if self._echo_redraw(ha, text):
                    continue   # 感知层把无关纠正误读成"重画"：跳过，不发重复图
                executed = self.hands.execute(ha["name"], ha.get("params") or {}, now)
                all_ok = all_ok and executed.get("ok")
                summaries.append(executed.get("summary", ""))
                names.append(ha["name"])
                if isinstance(executed.get("image"), str) and executed["image"]:
                    images_out.append(executed["image"])  # 产出图片（如 image.draw）
            hand_result = {"ok": all_ok, "action": "+".join(names),
                           "summary": "；".join(s for s in summaries if s)}
            if images_out:
                hand_result["images"] = images_out

        analysis = feedback_mod.analyze_with_perception(text, perc)
        if owner_truth:
            row["relation_state"] = relation_mod.update(row["relation_state"], analysis["sentiment"])

        memories = self.memory.retrieve(self.role_id, text, self.config.memory.retrieval_top_k,
                                        self.config.memory.min_salience, now=now,
                                        person_id=person_id, scope=scope)
        # 注意：ctx 在刷新 last_user_interaction 之前计算——
        # "距上次互动的时长"衡量的是用户离开多久（主动搭话层的时机依据）
        ctx = self._ctx(row, now)
        row["last_user_interaction"] = now
        open_intents = ctx["open_intents"]
        score, urges = motivation.active_score(
            row["state_vector"], row["relation_state"], ctx, self.config.motivation.weights)
        candidates = motivation.generate_candidates(
            row["state_vector"], row["relation_state"], memories, open_intents,
            self.engine.attractor, ctx, recent_intent_types=set(self._recent_proactive()))

        # 主动附加层：只有 L2+ 才有资格搭车（L1 微反应属于调度路径）
        # 这一轮已经执行了手部动作（点歌/提醒/画图等）时不再叠加：
        # 否则会出现"主动提醒 + 正常回复"两条都在说同一件事的重复。
        addon, addon_meta = None, None
        l2plus = [] if (self._cost_degraded() or hand_result is not None) else [
            c for c in candidates if c.level >= 2]
        best = motivation.best_candidate(l2plus)
        if best is not None:
            decision = self.gate.decide(best, score, row, now, source_exists=self._source_exists)
            if decision.allowed:
                addon_text, prompt, seed = self._render(
                    row, now, best.name, intent=best.name, source=best.source,
                    state_vector=row["state_vector"], relation_state=row["relation_state"],
                    memories=memories, open_intents=open_intents,
                    permission_level=self._permission_level(),
                    last_active_effect=row.get("last_active_effect", "尚无记录"),
                    recent_user_message=text, recent_dialogue=dialogue,
                    hand_result=hand_result, early_summary=early_summary,
                    old_dialogue=old_dialogue)
                if addon_text:
                    addon = addon_text
                    self._record_usage("addon")
                    log_id = self._log_active(best.level, best, addon_text,
                                              {"active_score": round(score, 4), "urges": urges,
                                               "candidates": [c.to_log() for c in
                                                              sorted(candidates, key=lambda c: -c.score)[:3]],
                                               "seed": seed},
                                              decision.to_log(), now)
                    self._log_message("agent", addon_text,
                                      {"proactive": {"intent": best.name, "level": best.level,
                                                     "log_id": log_id}}, now,
                                      created_at=now + (time.time() - turn_started))
                    row = register_active(row, best.level, best.name, now)
                    row["last_action"] = best.name
                    row = self.engine.apply_event_impulse(row, "agent_proactive_sent")
                    addon_meta = {"log_id": log_id, "intent": best.name, "level": best.level}

        # 状态驱动语言（梯队1-1）：极端状态产出可机检硬约束，渲染后闭环校验
        style_audit = {"constraints": [], "violations": [], "retried": False,
                       "mechanical": False, "passed": True,
                       "loose": bool(getattr(self, "loose_mode", False)),
                       "state": {k: round(row["state_vector"][k], 3)
                                 for k in ("energy", "valence", "arousal")}}
        style_constraints = []
        if self.config.state_style.enabled and not getattr(self, "loose_mode", False):
            # loose 模式不叠加状态语言硬约束：极端状态仍影响语气，但不许把
            # "只能短句/不许问句"这类机械约束再压到熟人局上。
            style_constraints = state_style_mod.derive(row["state_vector"],
                                            self.persona.baseline_state)
            style_constraints += style_ctrl.constraints(
                (row.get("style_bias") or {}).get("concise", 0.55))
            style_audit["constraints"] = style_constraints
            style_audit["style_bias"] = row.get("style_bias")
        reply, prompt, seed = self._render(
            row, now, "respond", intent="respond", source=None,
            state_vector=row["state_vector"], relation_state=row["relation_state"],
            memories=memories, open_intents=open_intents,
            permission_level=self._permission_level(),
            last_active_effect=row.get("last_active_effect", "尚无记录"),
            recent_user_message=text, recent_dialogue=dialogue, hand_result=hand_result,
            style_constraints=style_constraints, early_summary=early_summary,
            old_dialogue=old_dialogue, reply_plan=reply_plan,
            mood_event=row.get("last_mood_event"),
            capability_recall=int(row.get("cap_fail", 0)) > 0,
            needs_detail=detail_mode,
            )
        self._record_usage("respond")
        # [skip] 是内部控制标记（"这句不是跟我说的，别发"）。模型偶尔写成
        # "[skip] 正文…"（标记+内容）——只认完全相等会把标记原样发到微信
        # （2026-10-29 真实泄露病例）。前缀标记剥掉：只剩标记=跳过，
        # 带正文=发剥掉标记后的正文。
        _skip_m = re.match(r"^\[(?:skip|跳过)\]\s*", str(reply or "").strip(), re.I)
        if _skip_m:
            reply = str(reply or "").strip()[_skip_m.end():].strip()
            if not reply:
                # 模型判断"这句不是跟我说的"：不发、不落库
                return {"reply": "", "addon": None, "addon_meta": None,
                        "meta": {"skipped": True, "perception": perc}}
        if style_constraints:
            style_audit["violations"] = state_style_mod.check(reply, style_constraints)
            if style_audit["violations"]:
                # 打回重写一次（换 seed，指令里点名违规项）
                reply2, _, seed2 = self._render(
                    row, now + 1, "respond-fix", intent="respond", source=None,
                    state_vector=row["state_vector"], relation_state=row["relation_state"],
                    memories=memories, open_intents=open_intents,
                    permission_level=self._permission_level(),
                    last_active_effect=row.get("last_active_effect", "尚无记录"),
                    recent_user_message=text, recent_dialogue=dialogue,
                    hand_result=hand_result, style_constraints=style_constraints,
                    style_correction=state_style_mod.rewrite_instruction(
                        style_audit["violations"]),
                    early_summary=early_summary, old_dialogue=old_dialogue,
                    reply_plan=reply_plan, mood_event=row.get("last_mood_event"),
                    capability_recall=int(row.get("cap_fail", 0)) > 0,
                    needs_detail=detail_mode,
                    )
                self._record_usage("respond")
                style_audit["retried"] = True
                reply, seed = reply2, seed2
                style_audit["violations"] = state_style_mod.check(reply, style_constraints)
                if style_audit["violations"]:
                    reply = state_style_mod.mechanical_fix(reply, style_constraints)
                    style_audit["mechanical"] = True
                    style_audit["violations"] = state_style_mod.check(reply, style_constraints)
            style_audit["passed"] = not style_audit["violations"]
        # 审核 Agent（前额叶）出门过审：规则快路径 + LLM 审稿员；不过就打回重写一次
        def _rewrite(correction):
            text4, _, _ = self._render(
                row, now + 2, "respond-review", intent="respond", source=None,
                state_vector=row["state_vector"], relation_state=row["relation_state"],
                memories=memories, open_intents=open_intents,
                permission_level=self._permission_level(),
                last_active_effect=row.get("last_active_effect", "尚无记录"),
                recent_user_message=text, recent_dialogue=dialogue,
                hand_result=hand_result, style_constraints=style_constraints,
                style_correction=correction,
                early_summary=early_summary, old_dialogue=old_dialogue,
                reply_plan=reply_plan, mood_event=row.get("last_mood_event"),
                capability_recall=True, needs_detail=detail_mode)
            self._record_usage("respond")
            return text4

        gate = self._review_gate(
            reply, "respond", now, row,
            {"user_message": text,
             "needs_detail": detail_mode,
             "recent": [(d.get("content") if isinstance(d, dict) else str(d))
                        for d in (dialogue or [])][-4:]},
            rewrite=_rewrite)
        reply = gate["text"]
        # 失效回忆曲线只看"能力否认/AI 自曝"：首审命中就抬升（哪怕重写成功），
        # 干净的首审才回落。AI 腔这类机械病灶不进曲线，否则永远撤不掉。
        cap_hit = any(i in ("capability_denial", "ai_self") for i in gate["seen"])
        if cap_hit:
            row["cap_fail"] = int(row.get("cap_fail", 0)) + 1
        else:
            row["cap_fail"] = max(0, int(row.get("cap_fail", 0)) - 1)
        style_audit["review"] = gate["review"]
        style_audit["review_retried"] = gate["retried"]
        reply = strip_media_markers(strip_role_prefix(reply))
        if addon:
            addon = strip_media_markers(strip_role_prefix(addon))
        mem_tokens = sum(estimate_tokens(str(m)) for m in prompt["相关记忆"])
        # 内心立场存档：下一条用户消息来时做共鸣/冲撞对比（拟人反馈回路）
        my_stance = {"mood": (reply_plan or {}).get("mood") or "平静",
                     "valence": round(float(row["state_vector"]["valence"]), 3)}
        turn_ms = int(round(max(0.0, time.time() - turn_started) * 1000))
        reply_msg_id = self._log_message("agent", reply, {
            "prompt_tokens": estimate_tokens(str(prompt)),
            "memory_tokens": mem_tokens,
            "memories": len(prompt["相关记忆"]),
            "identity_len": len(prompt["身份摘要"]),
            "seed": seed, "addon": addon_meta, "my_stance": my_stance,
            "style": style_audit, "turn_ms": turn_ms, "reply_to": msg_id}, now,
            created_at=now + turn_ms / 1000.0)
        self.save_row(row, now)
        # mem0 攒批提炼：消息处理完顺带看一眼是否到提炼水位（内部自带节流/失败保险）
        if hasattr(self.memory, "maybe_extract"):
            try:
                self.memory.maybe_extract(self.role_id)
            except Exception as e:  # noqa: BLE001  提炼失败不影响对话
                print(f"[mem0] 批量提炼异常：{e!r}")
        # 对话压缩（后台，不阻塞回复）：近窗外积压够了就折叠进滚动摘要
        threading.Thread(target=self.maybe_compress_dialogue, daemon=True,
                         name="dialogue-compress").start()
        return {"reply": reply, "addon": addon, "addon_meta": addon_meta,
                "meta": {"message_id": reply_msg_id,
                         "active_score": round(score, 4),
                         "perception": perc, "hand": hand_result,
                         "style": style_audit,
                         "feedback": feedback_outcome["summary"] if feedback_outcome else None,
                         "relation": row["relation_state"]}}

    # ---------------- 审核 Agent（双 Agent 出门过审） ----------------

    def _review_gate(self, text: str, purpose: str, now: float, row: dict = None,
                     context: dict = None, rewrite=None) -> dict:
        """出门过审：规则快路径 + LLM 审稿员；不过就打回互动 Agent 重写一次。

        返回 {text: 终稿, review: 审稿记录, retried: 是否重写过, blocked: 硬拦}。
        rewrite(correction) -> 新文本；None 表示没法重写（主动/插嘴就干脆不发）。
        """
        ctx = dict(context or {})
        ctx["purpose"] = purpose
        ctx.setdefault("loose", bool(getattr(self, "loose_mode", False)))
        if row is not None:
            ctx.setdefault("group", bool(getattr(self, "is_group", False)))
        started = time.time()
        res = self.reviewer.review(text, ctx)
        seen = list(res.get("issues") or [])
        llm_used = bool(res.get("llm_used"))
        llm_failed = bool(res.get("llm_failed"))
        final, retried = text, False
        # 1) AI 腔这类确定性病灶：机械修正，零成本
        tic = [i for i in seen if i in ("tic_period", "tic_opener", "tic_twoline",
                                        "emoji_spam", "flat_question", "control_leak")]
        if seen and len(tic) == len(seen):
            final = reviewer_mod.mechanical_fix(text, tic)
            res2 = self.reviewer.review(final, ctx, allow_llm=False)
            llm_failed = llm_failed or bool(res2.get("llm_failed"))
            if res2["ok"]:
                res = res2
            else:
                final = text
        # 2) 实质性病灶：打回互动 Agent 重写一次（只给改写要求，不给全文）
        if res.get("issues") and rewrite is not None and res.get("correction"):
            try:
                new_text = rewrite(res["correction"])
            except Exception as e:  # noqa: BLE001  重写失败保留原稿
                print(f"[reviewer] 打回重写失败：{e!r}")
                new_text = ""
            if new_text and str(new_text).strip():
                res3 = self.reviewer.review(new_text, ctx, allow_llm=False)
                llm_failed = llm_failed or bool(res3.get("llm_failed"))
                retried = True
                if res3["ok"] or len(res3["issues"]) < len(res["issues"]):
                    final, res = new_text, res3
        # 最后兜底：重写后还是超长 -> 截到 36 字内（人格卡上限"最多两句"
        # 对齐；18 字太严会砍掉包袱半句，留下闹情绪的残句）。
        # 切在边界（换行/标点）上，无边界时显式省略号——绝不无声腰斩（详见
        # state_style.toolong_backstop 的病例注释）。
        # needs_detail 的回合不截：用户要的就是内容，截成残句比长更糟
        # （2026-10-30 实测 850 字故事被砍成 9 字"那我一口气给你讲完"）。
        # 型号清单/链接这类干货也不截：chime 路径没有"打回重写"，一刀切会把
        # "小米4 Pro、米家全效、352 X86C"切成"…型号：小米4"（2026-11-01 实测）。
        if ("too_long" in (res.get("issues") or []) and len(final) > 36
                and not ctx.get("needs_detail")
                and not reviewer_mod.is_substantive(final)):
            final = state_style_mod.toolong_backstop(final, limit=36)
        waited = time.time() - started
        blocked = any(i in ("capability_denial", "ai_self", "empty")
                      for i in res.get("issues") or [])
        logged = dict(res)
        logged["llm_used"] = llm_used     # 首审用了 LLM 就算（后续复核是规则快路径）
        logged["llm_failed"] = llm_failed
        logged["latency_ms"] = max(logged.get("latency_ms") or 0, int(waited * 1000))
        self._log_review(purpose, text, final, logged, retried, seen, now, blocked)
        return {"text": final, "review": logged, "retried": retried,
                "blocked": blocked, "seen": seen}

    def _log_review(self, purpose: str, original: str, final: str, res: dict,
                    retried: bool, seen: list, now: float, blocked: bool = False):
        """审稿留痕：原稿/病灶/改写要求/终稿，控制台可回溯。"""
        try:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO review_logs(role_id,purpose,mode,issues,correction,original,"
                    "final,retried,llm_used,llm_failed,latency_ms,created_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.role_id, purpose, str(res.get("mode") or "off"),
                     json.dumps(seen, ensure_ascii=False), res.get("correction"),
                     original or "", final or "", 1 if retried else 0,
                     1 if res.get("llm_used") else 0, 1 if res.get("llm_failed") else 0,
                     int(res.get("latency_ms") or 0), now))
                self.conn.commit()
        except Exception as e:  # noqa: BLE001  留痕失败不影响发送
            print(f"[review] 审稿日志写入失败：{e!r}")

    @staticmethod
    def _review_summarize(conn, since: float, limit: int = 12,
                          day_start: float = None) -> dict:
        """单个库的审核统计（主库/某个群/某个私聊用户各一份）。"""
        rows = conn.execute(
            "SELECT purpose, COUNT(*) c, SUM(retried) r, SUM(llm_used) l,"
            " SUM(llm_failed) f FROM review_logs"
            " WHERE created_at>=? GROUP BY purpose", (since,)).fetchall()
        by_purpose = {r["purpose"]: {"count": r["c"], "retried": r["r"] or 0,
                                     "llm_used": r["l"] or 0,
                                     "llm_failed": r["f"] or 0} for r in rows}
        flagged = conn.execute(
            "SELECT COUNT(*) c FROM review_logs WHERE created_at>=?"
            " AND issues<>'[]'", (since,)).fetchone()["c"]
        out_msgs = conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE sender='agent' AND created_at>=?",
            (since,)).fetchone()["c"]
        llm_today = conn.execute(
            "SELECT COUNT(*) c FROM review_logs WHERE llm_used=1 AND created_at>=?",
            (day_start if day_start else since,)).fetchone()["c"]
        recent = [dict(r) for r in conn.execute(
            "SELECT id,purpose,mode,issues,correction,original,final,retried,"
            "llm_used,llm_failed,latency_ms,created_at FROM review_logs"
            " WHERE created_at>=? ORDER BY id DESC LIMIT ?", (since, limit)).fetchall()]
        for r in recent:
            try:
                r["issues"] = json.loads(r["issues"] or "[]")
            except json.JSONDecodeError:
                r["issues"] = []
        return {"total": sum(v["count"] for v in by_purpose.values()),
                "flagged": flagged, "out_messages": out_msgs,
                "llm_today": llm_today,
                "by_purpose": by_purpose, "recent": recent}

    def _review_stats(self) -> dict:
        """控制台用：审核 Agent 的开关、配额、各会话覆盖率与最近审稿记录。

        注意：每个会话（私聊/各群/其他用户）各自一个库，统计必须汇总——
        只看主库会以为"几乎没启用"（实测主库 9 次、全量 125 次）。
        """
        since = time.time() - 86400
        _lt = time.localtime()
        day_start = time.mktime((_lt.tm_year, _lt.tm_mon, _lt.tm_mday, 0, 0, 0, 0, 0, -1))
        total = flagged = out_msgs = llm_today = 0
        by_purpose = {}
        recent = []

        def _merge(summary: dict, label: str, kind: str, sid: str):
            nonlocal total, flagged, out_msgs, llm_today
            total += summary["total"]
            flagged += summary["flagged"]
            out_msgs += summary["out_messages"]
            llm_today += summary.get("llm_today", 0)
            for k, v in (summary.get("by_purpose") or {}).items():
                cur = by_purpose.setdefault(k, {"count": 0, "retried": 0,
                                               "llm_used": 0, "llm_failed": 0})
                for kk in cur:
                    cur[kk] += v.get(kk, 0)
            for r in summary.get("recent") or []:
                r["session"] = label
                recent.append(r)
            return {"id": sid, "label": label, "kind": kind,
                    "total": summary["total"], "flagged": summary["flagged"],
                    "out_messages": summary["out_messages"],
                    "coverage": (round(summary["total"] / summary["out_messages"], 3)
                                 if summary["out_messages"] else None)}

        sessions = [_merge(self._review_summarize(self.conn, since, limit=6,
                                                   day_start=day_start),
                           "私聊", "private", "private")]
        for g in self._session_dbs():
            conn = g["conn"]
            try:
                s = self._review_summarize(conn, since, limit=3,
                                           day_start=day_start)
            except Exception:  # noqa: BLE001  单个会话库读失败不影响统计
                s = {"total": 0, "flagged": 0, "out_messages": 0,
                     "by_purpose": {}, "recent": []}
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            if g["kind"] == "group":
                label = g["label"] or ("群 " + g["room"].split("@")[0][-6:])
            else:
                label = g["label"] or g["id"][-6:]
            sessions.append(_merge(s, label, g["kind"], g["id"]))

        recent.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        return {"mode": self.config.reviewer.mode,
                "llm_purposes": list(self.config.reviewer.llm_purposes or []),
                "calls_today": llm_today,   # 按日志算（跨会话、跨进程重启）
                "max_per_day": int(self.config.reviewer.max_llm_per_day or 0),
                "total_24h": total, "flagged_24h": flagged,
                "out_messages_24h": out_msgs,
                "coverage": (round(total / out_msgs, 3) if out_msgs else None),
                "llm_calls_process": int(getattr(self.reviewer, "calls_today", 0) or 0),
                "sessions": sessions, "by_purpose": by_purpose,
                "recent": recent[:12]}

    def _permission_level(self) -> str:
        enabled = get_setting(self.conn, "proactive_enabled", True)
        disabled = get_setting(self.conn, "proactive_disabled_levels", {}) or {}
        levels = [f"L{i}" for i in range(3, -1, -1)
                  if self.persona.active_permissions.get(f"L{i}", False)
                  and not disabled.get(f"L{i}", False)]
        return levels[0] if enabled and levels else "L0-仅沉默"

    # ---------------- 对外查询（HTTP/CLI 共用，全部持锁） ----------------

    def status(self, now: float = None) -> dict:
        with self._lock:
            now = time.time() if now is None else now
            row = self._evolve(self.get_row(now), now)
            sv = row["state_vector"]
            return {
                "role_id": self.role_id,
                "llm_provider": self.client.name,
                "state_vector": {k: round(v, 3) for k, v in sv.items()},
                "attractor_distance": round(self.engine.attractor.distance(sv), 3),
                "reference_tolerance": round(self.engine.attractor.reference_tolerance, 3),
                "consistency": round(self.engine.attractor.consistency(sv), 3),
                "relation": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in row["relation_state"].items()},
                "active_today": row.get("active_count_today"),
                "last_active_effect": row.get("last_active_effect", "尚无记录"),
                "proactive_enabled": get_setting(self.conn, "proactive_enabled", True),
                "permission_level": self._permission_level(),
                "memory_count": self.conn.execute(
                    "SELECT COUNT(*) c FROM memories WHERE role_id=?",
                    (self.role_id,)).fetchone()["c"],
                "llm_usage": self._usage_summary(),
                "cost": {"budget": self.config.cost.daily_token_budget,
                         "used_today": (get_setting(self.conn, "usage_today", {})
                                        or {}).get("tokens", 0),
                         "degraded": self._cost_degraded()},
                "known_days": max(0, int((now - (get_setting(self.conn, "first_chat_ts") or now)) // 86400)),
                "growth": {
                    "attractor_drift": get_setting(self.conn, "attractor_drift", {}) or {},
                    "growth_rules": get_setting(self.conn, "growth_rules", []) or [],
                    "state": self._growth_state(),
                    "profile": self._growth_profile(),
                    "last_reflection_ts": get_setting(self.conn, "last_reflection_ts", None),
                    "reflection_count": self.conn.execute(
                        "SELECT COUNT(*) c FROM reflections WHERE role_id=?",
                        (self.role_id,)).fetchone()["c"],
                },
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            }

    def _console_groups(self) -> list:
        """群 Agent 的只读快照：每个群一个库，供控制台切换查看。"""
        out = []
        for g in self._session_dbs():
            if g["kind"] != "group":
                try:
                    g["conn"].close()
                except Exception:  # noqa: BLE001
                    pass
                continue
            conn = g["conn"]
            try:
                people = PersonStore(conn, self.role_id, self.persona).list_cards(20)
                msg_count = conn.execute(
                    "SELECT COUNT(*) c FROM messages").fetchone()["c"]
                mem_count = conn.execute(
                    "SELECT COUNT(*) c FROM memories").fetchone()["c"]
                last = conn.execute(
                    "SELECT MAX(created_at) t FROM messages").fetchone()["t"] or 0
                out.append({"room": g["room"],
                            "label": g["label"] or ("群 " + g["room"].split("@")[0][-6:]),
                            "people": people, "messages": msg_count,
                            "memories": mem_count, "last": last})
            except Exception:  # noqa: BLE001  单个群库不可读不影响控制台
                continue
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        return out

    def _console_people(self, groups: list = None) -> list:
        """所有会话的人物卡：主库 + 每个群/私聊用户库各有一张 persons 表。

        同 wxid 跨会话合并成一张卡（关系数值取最熟的那份），sessions 记录
        这张卡出现在哪些会话里——人物卡页不该只看到主库那一个人。
        """
        merged = {}

        def add(cards: list, sid: str, label: str, kind: str):
            for c in cards or []:
                key = str(c.get("wxid") or c.get("name") or c.get("person_id"))
                entry = merged.get(key)
                if entry is None:
                    entry = dict(c)
                    entry["sessions"] = []
                    entry["session_ids"] = []
                    entry["primary"] = label
                    merged[key] = entry
                elif (c.get("familiarity") or 0) > (entry.get("familiarity") or 0):
                    keep = {k: entry.get(k) for k in ("sessions", "session_ids", "primary")}
                    entry.update(c)
                    entry.update(keep)
                    entry["primary"] = label
                if label and label not in entry["sessions"]:
                    entry["sessions"].append(label)
                if sid not in entry["session_ids"]:
                    entry["session_ids"].append(sid)
                # 私聊库里对方常被存成"你"/空名：用会话显示名兜底
                if kind == "private" and str(entry.get("name") or "") in ("", "你"):
                    entry["name"] = label or entry.get("name") or ""

        add(self.persons.list_cards(100), "private", "私聊", "private")
        for g in (groups or []):
            add(g.get("people") or [], g.get("room") or g.get("label") or "",
                g.get("label") or g.get("room") or "", "group")
        for s in self._session_dbs():
            if s["kind"] == "group":   # 群卡已经在 groups 里，别重复开库
                try:
                    s["conn"].close()
                except Exception:  # noqa: BLE001
                    pass
                continue
            try:
                add(PersonStore(s["conn"], self.role_id, self.persona).list_cards(100),
                    s["id"], s["label"] or s["id"], "private")
            except Exception:  # noqa: BLE001  单个库读失败不影响其他卡
                continue
            finally:
                try:
                    s["conn"].close()
                except Exception:  # noqa: BLE001
                    pass
        return sorted(merged.values(),
                      key=lambda c: (c.get("kind") != "creator",
                                     -(c.get("familiarity") or 0)))

    def _trajectory_items(self, conn, limit: int = 80) -> list:
        """单个库（主库或某个群的库）的轨迹条目：消息 + 工具调用，按时间排。

        只暴露可审计的结构化决策（感知、回应思路、记忆命中、风格约束、
        主动评分/闸门、审核结果）；不暴露模型原始 chain-of-thought。
        """
        rows = conn.execute(
            "SELECT id, sender, content, speaker, speaker_role, meta, created_at"
            " FROM messages WHERE role_id=? ORDER BY id DESC LIMIT ?",
            (self.role_id, limit)).fetchall()
        rows = list(reversed(rows))
        logs = {int(r["id"]): dict(r) for r in conn.execute(
            "SELECT id, level, intent_type, score_detail, gate_result, created_at"
            " FROM active_logs WHERE role_id=? ORDER BY id DESC LIMIT 500",
            (self.role_id,)).fetchall()}
        hands = [dict(r) for r in conn.execute(
            "SELECT id, action, params, result, created_at FROM hands_log"
            " WHERE role_id=? ORDER BY id DESC LIMIT 200", (self.role_id,)).fetchall()]
        items = []
        pending_input = None
        perceptions = {}   # 消息 id -> 感知结果；回复带 reply_to 时按它取，别串到别人消息上
        for r in rows:
            try:
                meta = json.loads(r["meta"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            item = {"kind": "message", "id": r["id"], "sender": r["sender"],
                    "speaker": r["speaker"], "speaker_role": r["speaker_role"],
                    "content": r["content"], "created_at": r["created_at"],
                    "trace": {}}
            if r["sender"] == "user":
                pending_input = meta.get("perception")
                perceptions[int(r["id"])] = pending_input
                item["trace"] = {"perception": pending_input,
                                 "images": meta.get("images") or [],
                                 "addressee": meta.get("addressee"),
                                 "overheard": bool(meta.get("overheard"))}
            else:
                reply_to = meta.get("reply_to")
                trace_input = pending_input
                if reply_to is not None:
                    try:
                        trace_input = perceptions.get(int(reply_to), pending_input)
                    except (TypeError, ValueError):
                        trace_input = pending_input
                trace = {"input": trace_input,
                         "reply_to": reply_to,
                         "memory_tokens": meta.get("memory_tokens"),
                         "memories": meta.get("memories"),
                         "prompt_tokens": meta.get("prompt_tokens"),
                         "style": meta.get("style"),
                         "stance": meta.get("my_stance"),
                         "addon": meta.get("addon"),
                         "proactive": meta.get("proactive"),
                         "review": meta.get("review"),
                         "turn_ms": meta.get("turn_ms"),
                         "chime": bool(meta.get("chime"))}
                log_id = ((meta.get("proactive") or {}).get("log_id")
                          or (meta.get("addon") or {}).get("log_id"))
                if log_id and int(log_id) in logs:
                    trace["active_log"] = logs[int(log_id)]
                item["trace"] = trace
                pending_input = None
            items.append(item)
        for h in hands:
            try:
                params = json.loads(h["params"] or "{}")
            except json.JSONDecodeError:
                params = h["params"]
            try:
                result = json.loads(h["result"] or "{}")
            except json.JSONDecodeError:
                result = h["result"]
            items.append({"kind": "tool", "id": h["id"], "action": h["action"],
                          "params": params, "result": result,
                          "created_at": h["created_at"], "trace": {}})
        # 同一秒内：用户消息 -> 工具 -> Agent 回复
        priority = {"user": 0, "tool": 1, "agent": 2}
        items.sort(key=lambda x: (x["created_at"],
                                  priority.get(x.get("sender", "tool"), 3),
                                  x.get("id", 0)))
        return items[-limit:]

    def _session_dbs(self) -> list:
        """枚举"其他会话"的库：每个群、每个其他私聊用户一份（data/wechat_users/<id>/agent.db）。
        后台控制台的会话轨迹按会话开轴，靠这里拿到各库的只读连接。"""
        import sqlite3
        base = (getattr(self, "wechat_users_dir", None)
                or os.path.join(project_root(), "data", "wechat_users"))
        out = []
        if not os.path.isdir(base):
            return out
        for name in sorted(os.listdir(base)):
            dbp = os.path.join(base, name, "agent.db")
            if not os.path.exists(dbp):
                continue
            is_group = name.endswith("chatroom")
            sid = name.replace("chatroom", "@chatroom") if is_group else name
            label = ""
            try:
                conn = sqlite3.connect("file:" + dbp + "?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT value FROM settings WHERE key=?",
                    ("group_name" if is_group else "display_name",)).fetchone()
                if row:
                    label = str(json.loads(row["value"]) or "")
                out.append({"id": sid, "room": sid if is_group else "",
                            "kind": "group" if is_group else "private",
                            "label": label, "conn": conn, "db": dbp})
            except Exception:  # noqa: BLE001  单个库打不开不影响其他轴
                continue
        return out
    def _session_counts(self, conn) -> dict:
        """会话的条目总量与最近时间（消息 + 工具调用），供控制台标"共多少条/是否截断"。"""
        msg = conn.execute(
            "SELECT COUNT(*) c, MAX(created_at) t FROM messages WHERE role_id=?",
            (self.role_id,)).fetchone()
        hand = conn.execute(
            "SELECT COUNT(*) c, MAX(created_at) t FROM hands_log WHERE role_id=?",
            (self.role_id,)).fetchone()
        return {"total": int(msg["c"] or 0) + int(hand["c"] or 0),
                "last": max(msg["t"] or 0, hand["t"] or 0)}

    def _pack_session(self, sid: str, label: str, kind: str, room: str,
                      conn, limit: int, want_items: bool) -> dict:
        """一个会话轴的打包结果；want_items=False 时只回元数据（切会话时省流量）。"""
        counts = self._session_counts(conn)
        items = self._trajectory_items(conn, limit) if want_items else []
        return {"id": sid, "label": label, "kind": kind, "room": room,
                "items": items, "count": len(items), "total": counts["total"],
                "last": counts["last"] or (items[-1]["created_at"] if items else 0),
                "truncated": bool(want_items and counts["total"] > len(items))}

    def trajectory(self, limit: int = 80, session: str = None,
                   meta: bool = False) -> dict:
        """会话轨迹：按会话分组的多条时间轴（私聊一条，每个群/私聊用户各一条）。

        每条轴内含 消息/工具 条目，带绝对时间戳，前端按真实时间轴摆位。
        session= 只取那一条会话的条目（其余只回元数据）；meta=True 只回会话清单。
        """
        with self._lock:
            limit = max(10, min(int(limit), 500))

            def want(sid: str) -> bool:
                return (not meta) and (session is None or session == sid)

            sessions = [self._pack_session("private", "私聊", "private", "",
                                           self.conn, limit, want("private"))]
            for g in self._session_dbs():
                fallback = ("群 " + g["room"].split("@")[0][-6:]) if g["kind"] == "group" \
                    else ("用户 " + g["id"][-6:])
                try:
                    sessions.append(self._pack_session(
                        g["id"], g["label"] or fallback, g["kind"], g["room"],
                        g["conn"], limit, want(g["id"])))
                except Exception:  # noqa: BLE001  单个库读失败=空轴
                    sessions.append({"id": g["id"], "label": g["label"] or fallback,
                                     "kind": g["kind"], "room": g["room"],
                                     "items": [], "count": 0, "total": 0,
                                     "last": 0, "truncated": False})
                finally:
                    try:
                        g["conn"].close()
                    except Exception:  # noqa: BLE001
                        pass
            sessions[1:] = sorted(sessions[1:], key=lambda s: -(s["last"] or 0))
            return {"sessions": sessions, "items": sessions[0]["items"],
                    "server_time": time.time()}

    def _prompt_overlay(self, conn, sid: str, kind: str, label: str,
                        full: bool = False) -> dict:
        """一个会话在提示词上的叠加项：名字、风格覆盖、成长规则、摘要、记忆量、最近快照。"""
        def setting(key, default=None):
            try:
                return get_setting(conn, key, default)
            except Exception:  # noqa: BLE001
                return default

        def count(table: str) -> int:
            try:
                return int(conn.execute(
                    "SELECT COUNT(*) c FROM " + table + " WHERE role_id=?",
                    (self.role_id,)).fetchone()["c"] or 0)
            except Exception:  # noqa: BLE001
                return 0

        ov = setting("config_overrides", {}) or {}
        summary = setting("conv_summary:" + self.role_id, {}) or {}
        lp = setting("last_prompt", None) or None
        last = None
        if lp:
            last = {"ts": lp.get("ts"), "purpose": lp.get("purpose"),
                    "chars": lp.get("chars")}
            if full:
                last["text"] = lp.get("text")
                last["sections"] = lp.get("sections")
        return {
            "id": sid, "label": label, "kind": kind,
            "name": str(setting("group_name" if kind == "group" else "display_name",
                                "") or ""),
            "persona_style": setting("persona_style", {}) or {},
            "growth_rules": setting("growth_rules", []) or [],
            "attractor_drift": setting("attractor_drift", {}) or {},
            "gate_shift_L2": setting("gate_shift_L2", 0) or 0,
            "persona_overrides": (ov.get("persona") or {}),
            "config_overrides": ov,
            "summary": str(summary.get("summary") or ""),
            "memories": count("memories"), "messages": count("messages"),
            "people": count("persons"),
            "last_prompt": last,
        }

    def prompt_view(self, session: str = None) -> dict:
        """提示词查看：基线设定（人格/风格/边界/能力）+ 每个会话的叠加项。

        session= 时把那条会话最近一次真正发出去的提示词全文（sections + text）
        一起带回来；其它会话只回快照元信息（时间/用途/字数），省流量。
        """
        with self._lock:
            base = {
                "role_id": self.role_id,
                "identity_summary": self.persona.identity_summary,
                "core_invariants": list(self.persona.core_invariants),
                "style": dict(self.persona.style or {}),
                "boundaries": list(self.persona.boundaries or []),
                "baseline_state": dict(self.persona.baseline_state or {}),
                "attractor": [{"state_name": e.state_name, "center": e.center,
                               "tolerance": e.tolerance, "weight": e.weight}
                              for e in self.persona.attractor],
                "overrides": {},
                "capabilities": self.builder._capabilities(),
                "creator": dict(getattr(self.persona, "creator", {}) or {}),
                "switches": {
                    "state_style": bool(self.config.state_style.enabled),
                    "reviewer_mode": self.config.reviewer.mode,
                    "memory_provider": self.config.memory.provider,
                    "dialogue_window": self.config.dialogue.window,
                    "retrieval_top_k": self.config.memory.retrieval_top_k,
                },
            }
            try:
                with open(persona_overrides_path(os.path.join(self.config_dir, "persona.yaml")),
                          "r", encoding="utf-8") as f:
                    ov_data = json.load(f) or {}
                if isinstance(ov_data, dict):
                    base["overrides"] = ov_data
            except Exception:  # noqa: BLE001  没有覆盖文件=出厂值
                base["overrides"] = {}
            # 初始基线提示词：不接状态/记忆/对话，只看"开箱时模型收到的那套设定"。
            # 群聊会话会在它之上加【群聊规则】【群聊身份】（看该会话的实发快照）。
            try:
                neutral = self.engine.new_row(time.time())
                bp = self.builder.build(
                    intent="respond", source=None,
                    state_vector=dict(self.persona.baseline_state),
                    relation_state=neutral["relation_state"],
                    memories=[], open_intents=[], permission_level=self._permission_level(),
                    last_active_effect="尚无记录", recent_user_message=None,
                    recent_dialogue=[], group_context=False,
                    person_card=None, creator_card=None, now=time.time())
                base["base_prompt"] = {"sections": bp,
                                       "text": render_prompt_text(bp),
                                       "chars": len(render_prompt_text(bp))}
            except Exception as e:  # noqa: BLE001  预览失败不影响其它内容
                base["base_prompt"] = {"error": repr(e)}
            self_sid = self.group_room if getattr(self, "is_group", False) else "private"
            self_kind = "group" if getattr(self, "is_group", False) else "private"
            self_label = ((self.group_name or "群聊") if self_kind == "group" else "私聊")
            sessions = [self._prompt_overlay(self.conn, self_sid, self_kind, self_label,
                                             full=(session == self_sid))]
            for g in self._session_dbs():
                if os.path.abspath(g.get("db") or "") == os.path.abspath(self._db_path or ""):
                    try:
                        g["conn"].close()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                fallback = ("群 " + g["room"].split("@")[0][-6:]) if g["kind"] == "group" \
                    else ("用户 " + g["id"][-6:])
                try:
                    sessions.append(self._prompt_overlay(
                        g["conn"], g["id"], g["kind"], g["label"] or fallback,
                        full=(session == g["id"])))
                except Exception:  # noqa: BLE001  单个库读失败也留个空位
                    sessions.append({"id": g["id"], "label": g["label"] or fallback,
                                     "kind": g["kind"], "last_prompt": None})
                finally:
                    try:
                        g["conn"].close()
                    except Exception:  # noqa: BLE001
                        pass
            return {"base": base, "sessions": sessions, "server_time": time.time()}

    def console_snapshot(self) -> dict:
        """后台控制台数据：参数、状态、人物卡、记忆、主动日志一次取全。"""
        with self._lock:
            now = time.time()
            status = self.status(now)
            groups = self._console_groups()
            people = self._console_people(groups) if hasattr(self, "persons") else []
            mem_by_type = [dict(r) for r in self.conn.execute(
                "SELECT type, COUNT(*) AS count FROM memories WHERE role_id=?"
                " GROUP BY type", (self.role_id,)).fetchall()]
            mem_recent = [dict(r) for r in self.conn.execute(
                "SELECT id,type,content,person_id,scope,salience,created_at"
                " FROM memories WHERE role_id=? ORDER BY id DESC LIMIT 8",
                (self.role_id,)).fetchall()]
            logs = [dict(r) for r in self.conn.execute(
                "SELECT id,level,intent_type,rendered_text,gate_result,created_at"
                " FROM active_logs WHERE role_id=? ORDER BY id DESC LIMIT 15",
                (self.role_id,)).fetchall()]
            msg_count = self.conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE role_id=?",
                (self.role_id,)).fetchone()["c"]
            return {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "status": status,
                "config": asdict(self.config),
                "persona": asdict(self.persona),
                "people": people,
                "memory": {"total": status.get("memory_count", 0),
                           "by_type": mem_by_type, "recent": mem_recent},
                "messages": msg_count,
                "active_logs": logs,
                "groups": groups,
                "review": self._review_stats(),
                "hands": self._hands_snapshot(),
                "growth": {
                    "baseline": dict(self._base_centers),
                    "events": [dict(r) for r in self.conn.execute(
                        "SELECT id, kind, evidence, state_before, state_after,"
                        " created_at FROM growth_events WHERE role_id=?"
                        " ORDER BY id DESC LIMIT 120", (self.role_id,)).fetchall()],
                },
            }

    def _wechat_probe(self) -> dict:
        """探测 wxhook 登录态（1.2s 超时 + 15s 缓存，别让控制台刷新被拖住）。

        只能证明微信客户端已登录；桥是独立进程，在不在跑这里看不到。
        """
        w = self.config.wechat
        if w.channel != "wxhook":
            return {"up": None, "logged_in": None, "checked": 0}
        now = time.time()
        cached = getattr(self, "_wx_probe_cache", None)
        if cached and now - cached[0] < 15:
            return cached[1]
        info = {"up": False, "logged_in": None, "checked": now}
        try:
            req = urllib.request.Request(
                w.wxhook.base_url.rstrip("/") + "/QueryDB/status", method="GET")
            with urllib.request.urlopen(req, timeout=1.2) as r:
                out = json.loads(r.read().decode("utf-8", "replace") or "{}")
            info = {"up": True, "logged_in": out.get("IsLogin") == 1, "checked": now}
        except Exception:  # noqa: BLE001  hook 没起/超时都只是"离线"，不影响快照
            pass
        self._wx_probe_cache = (now, info)
        return info

    def _hands_snapshot(self) -> dict:
        """手部/插件快照：注册动作、插件装载结果、调用统计、微信通道配置与登录态。

        动作清单来自 Hands 注册表（内置与插件同一条路径）；调用次数与最近
        使用时间来自 hands_log（插件动作经同一 execute 入口，天然计入）。
        """
        usage = {}
        for r in self.conn.execute(
                "SELECT action, COUNT(*) c, MAX(created_at) t FROM hands_log"
                " WHERE role_id=? GROUP BY action", (self.role_id,)).fetchall():
            usage[r["action"]] = {"calls": r["c"], "last": r["t"]}
        actions = []
        for a in self.hands.snapshot():
            u = usage.get(a["name"], {})
            actions.append({**a, "calls": u.get("calls", 0), "last": u.get("last")})
        w = self.config.wechat
        return {
            "enabled": bool(self.config.hands.enabled),
            "allow_shell": bool(self.config.hands.allow_shell),
            "plugins_enabled": bool(self.config.hands.plugins_enabled),
            "actions": actions,
            "plugins": list(self.plugin_info or []),
            "wechat": {
                "channel": w.channel,
                "groups_enabled": bool(w.groups_enabled),
                "group_mention_only": bool(w.group_mention_only),
                "enabled_chats": list(w.enabled_chats or []),
                "base_url": w.wxhook.base_url,
                "probe": self._wechat_probe(),
            },
            "voice": {
                "asr_model": self.config.asr.model,
                "reply": w.voice_reply,
                "max_chars": w.voice_max_chars,
            },
        }

    def _usage_summary(self) -> dict:
        out = {}
        for purpose, s in self.usage_stats.items():
            total_cacheable = s["cache_hit_tokens"] + s["cache_miss_tokens"]
            out[purpose] = {
                "calls": s["calls"],
                "prompt_tokens": s["prompt_tokens"],
                "completion_tokens": s["completion_tokens"],
                "cache_hit_tokens": s["cache_hit_tokens"],
                "cache_hit_rate": round(s["cache_hit_tokens"] / total_cacheable, 3)
                if total_cacheable else None,
            }
        return out

    def recent_messages(self, limit: int = 50, before_id: int = None) -> list:
        with self._lock:
            if before_id is None:
                rows = self.conn.execute(
                    "SELECT * FROM messages WHERE role_id=? ORDER BY id DESC LIMIT?",
                    (self.role_id, max(1, min(int(limit), 500)))).fetchall()
                return [_message_row(r) for r in reversed(rows)]
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE role_id=? AND id<? ORDER BY id DESC LIMIT?",
                (self.role_id, int(before_id), max(1, min(int(limit), 500)))).fetchall()
            return [_message_row(r) for r in reversed(rows)]

    def messages_after(self, after_id: int, limit: int = 100) -> list:
        """供客户端增量轮询：id 大于 after_id 的 agent 消息（含主动）。"""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE role_id=? AND sender='agent' AND id>?"
                " ORDER BY id ASC LIMIT?",
                (self.role_id, int(after_id), max(1, min(int(limit), 500)))).fetchall()
            return [_message_row(r) for r in rows]

    # ---------------- 主动路径：调度 ----------------

    def evaluate_proactive(self, now: float = None, observe: bool = False) -> dict:
        with self._lock:
            return self._evaluate_proactive_locked(now, observe=observe)


    def _fire_due_reminders(self, row: dict, now: float) -> list:
        """到点提醒是用户自己约的闹钟：直发，不走评分闸门，但照常留痕。"""
        due = self.hands.due_reminders(now)
        fired = []
        for r in due:
            self.hands.mark_done(r["id"])
            try:
                text, prompt, seed = self._render(
                    row, now, "reminder_due", intent="reminder_due",
                    source={"type": "reminder", "id": r["id"]},
                    state_vector=row["state_vector"],
                    relation_state=row["relation_state"], memories=[],
                    open_intents=self.memory.open_intents(self.role_id),
                    permission_level=self._permission_level(),
                    last_active_effect=row.get("last_active_effect", "尚无记录"),
                    recent_user_message=r["text"])
                self._record_usage("reminder")
            except Exception as e:  # noqa: BLE001  渲染失败也要把提醒送到
                print(f"[reminder] 渲染失败，模板直发：{e!r}")
                text = f"⏰ {r['text']}"
            if not text:
                text = f"⏰ {r['text']}"
            # 闹钟是用户自己约的：审核只做规则快路径，绝不因为审稿把提醒丢了
            text = self._review_gate(text, "reminder", now, rewrite=None)["text"]
            cand = motivation.Candidate("reminder_due", 1, 1.0,
                                        {"type": "reminder", "id": r["id"]}, {})
            log_id = self._log_active(1, cand, text,
                                      {"user_requested": True, "reminder_id": r["id"]},
                                      {"bypassed": "user-requested"}, now)
            self._log_message("agent", text, {"proactive": {
                "intent": "reminder_due", "level": 1, "log_id": log_id}}, now)
            row = register_active(row, 1, "reminder_due", now)
            row["last_action"] = "reminder_due"
            if self.config.hands.notify_desktop:
                try:
                    self.hands.notifier("小满提醒", r["text"])
                except Exception as e:  # noqa: BLE001
                    print(f"[reminder] 桌面通知失败：{e!r}")
            fired.append({"text": text, "level": 1, "intent": "reminder_due",
                          "log_id": log_id, "reminder_id": r["id"]})
        if fired:
            self.save_row(row, now)
        return fired

    def _evaluate_proactive_locked(self, now: float, observe: bool = False) -> dict:
        now = time.time() if now is None else now
        row = self._evolve(self.get_row(now), now)
        fired = self._fire_due_reminders(row, now)
        if fired:
            return {"sent": fired[0], "fired_reminders": fired}
        enabled = get_setting(self.conn, "proactive_enabled", True)
        if not enabled:
            self.save_row(row, now)
            return {"sent": None, "reason": "proactive disabled"}
        if self._cost_degraded():
            self.save_row(row, now)
            return {"sent": None, "reason": "cost budget exceeded"}

        scope = f"group:{self.group_room}" if getattr(self, "is_group", False) else "private"
        # 翻旧账：主动开口的素材按"显著度高且久未提起"取，不拿用户最后一句话当
        # query——语义相似只会召回刚聊过的话题，主动消息就变成了延时复读
        memories = self.memory.retrieve_stale(self.role_id,
                                              self.config.memory.retrieval_top_k, now=now)
        ctx = self._ctx(row, now)
        open_intents = ctx["open_intents"]
        recent_text = " ".join(
            str(m.get("content") or "") for m in self.recent_messages(8))
        score, urges = motivation.active_score(
            row["state_vector"], row["relation_state"], ctx, self.config.motivation.weights)
        candidates = motivation.generate_candidates(
            row["state_vector"], row["relation_state"], memories, open_intents,
            self.engine.attractor, ctx, recent_intent_types=set(self._recent_proactive()),
            recent_text=recent_text)
        best = motivation.best_candidate(candidates)
        if best is None:
            self.save_row(row, now)
            return {"sent": None, "reason": "no candidate", "active_score": round(score, 4)}
        # 计划 5.8：调度路径在免打扰时段直接沉默（L1 微反应也不在深夜发）
        if best.level < 2 and ctx["quiet_hour"]:
            self.save_row(row, now)
            return {"sent": None, "reason": "quiet hours", "active_score": round(score, 4)}

        decision = self.gate.decide(best, score, row, now, source_exists=self._source_exists)
        if not decision.allowed:
            self.save_row(row, now)
            return {"sent": None, "reason": "gate blocked", "gate": decision.to_log(),
                    "active_score": round(score, 4),
                    "candidates": [c.to_log() for c in
                                   sorted(candidates, key=lambda c: -c.score)[:3]]}

        dialogue = self.recent_messages(self.config.dialogue.window)
        early_summary = self._dialogue_summary() or None
        style_constraints = []
        if self.config.state_style.enabled:
            style_constraints = state_style_mod.derive(row["state_vector"],
                                                       self.persona.baseline_state)
        text, prompt, seed = self._render(
            row, now, best.name, intent=best.name, source=best.source,
            state_vector=row["state_vector"], relation_state=row["relation_state"],
            memories=memories, open_intents=open_intents,
            permission_level=self._permission_level(),
            last_active_effect=row.get("last_active_effect", "尚无记录"),
            recent_user_message=ctx["last_user_message"] or None,
            recent_dialogue=dialogue, early_summary=early_summary,
            style_constraints=style_constraints or None,
            mood_event=row.get("last_mood_event"))
        self._record_usage("proactive")
        if not text:
            self.save_row(row, now)
            return {"sent": None, "reason": "rendered empty"}

        # 审核 Agent 过审（主动开口没人监督，质量全靠这一关）
        def _rewrite_proactive(correction):
            text2, _, _ = self._render(
                row, now, best.name, intent=best.name, source=best.source,
                state_vector=row["state_vector"], relation_state=row["relation_state"],
                memories=memories, open_intents=open_intents,
                permission_level=self._permission_level(),
                last_active_effect=row.get("last_active_effect", "尚无记录"),
                recent_user_message=ctx["last_user_message"] or None,
                recent_dialogue=dialogue, early_summary=early_summary,
                style_constraints=style_constraints or None,
                style_correction=correction,
                mood_event=row.get("last_mood_event"))
            self._record_usage("proactive")
            return text2

        gate = self._review_gate(text, "proactive", now, row,
                                 {"user_message": ctx["last_user_message"] or "",
                                  "recent": [(m.get("content") if isinstance(m, dict)
                                              else str(m)) for m in (dialogue or [])][-4:]},
                                 rewrite=_rewrite_proactive)
        text = gate["text"]
        if gate["blocked"] or not str(text or "").strip():
            self.save_row(row, now)
            return {"sent": None, "reason": "reviewer blocked",
                    "review": gate["review"], "active_score": round(score, 4)}
        log_detail = {"active_score": round(score, 4), "urges": urges,
                      "seed": seed, "review": gate["review"]}
        gate_log = decision.to_log()
        if observe:
            log_detail["observe"] = True
            gate_log = dict(gate_log or {})
            gate_log["observe"] = True
        log_id = self._log_active(best.level, best, text, log_detail, gate_log, now)
        if observe:
            # 观察模式：记录“本来会说什么”，但不写 messages、不施加“已主动”
            # 情绪脉冲；仍登记冷却，避免同一素材每轮重复判。
            row = register_active(row, best.level, best.name, now)
            self.save_row(row, now)
            return {"observed": {"text": text, "level": best.level,
                                 "intent": best.name, "log_id": log_id},
                    "gate": gate_log, "active_score": round(score, 4)}
        self._log_message("agent", text,
                          {"proactive": {"intent": best.name, "level": best.level,
                                         "log_id": log_id},
                           "review": gate["review"],
                           "prompt_tokens": estimate_tokens(str(prompt))}, now)
        row = register_active(row, best.level, best.name, now)
        row["last_action"] = best.name
        row = self.engine.apply_event_impulse(row, "agent_proactive_sent")
        self.save_row(row, now)
        return {"sent": {"text": text, "level": best.level, "intent": best.name,
                         "log_id": log_id}, "gate": decision.to_log(),
                "active_score": round(score, 4)}


# ---------------- CLI ----------------

def _message_row(r) -> dict:
    try:
        meta = json.loads(r["meta"]) if r["meta"] else {}
    except json.JSONDecodeError:
        meta = {}
    return {"id": r["id"], "sender": r["sender"], "content": r["content"],
            "meta": meta, "created_at": r["created_at"],
            "speaker": (r["speaker"] if "speaker" in r.keys() else None),
            "speaker_role": (r["speaker_role"] if "speaker_role" in r.keys() else None)}


def _print_result(result: dict):
    if result.get("addon"):
        print(f"  〔主动·{result['addon_meta']['intent']}〕{result['addon']}")
    print(f"  小满：{result['reply']}")


def cmd_chat(args):
    agent = Agent(db_path=args.db)
    print("进入对话（输入 /quit 退出）")
    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text == "/quit":
            break
        result = agent.on_user_message(text)
        _print_result(result)


def cmd_tick(args):
    from .scheduler import run_tick
    agent = Agent(db_path=args.db)
    n = args.loop or 1
    for _ in range(n):
        result = run_tick(agent)
        ts = time.strftime("%H:%M:%S")
        if result.get("sent"):
            s = result["sent"]
            print(f"[{ts}] 主动 L{s['level']} {s['intent']}：{s['text']}")
        else:
            print(f"[{ts}] 未主动（{result.get('reason')}）")
        if args.loop and args.interval:
            time.sleep(args.interval)


def cmd_demo(args):
    from .demo import run_demo
    run_demo(db_path=args.db, fresh=args.fresh)


def cmd_status(args):
    agent = Agent(db_path=args.db)
    s = agent.status()
    print("状态向量：", s["state_vector"])
    print(f"距吸引子：{s['attractor_distance']}（参考容差 {s['reference_tolerance']}，"
          f"一致性 {s['consistency']}）")
    print("关系：", {k: v for k, v in s["relation"].items() if isinstance(v, (int, float))})
    print("今日主动：", s["active_today"], "；主动开关：", s["proactive_enabled"],
          "；当前级别：", s["permission_level"])
    print("记忆数：", s["memory_count"], "；渲染端：", s["llm_provider"])


def cmd_serve(args):
    from .server import serve
    serve(host=args.host, port=args.port, db_path=args.db, interval=args.interval)


def cmd_wechat(args):
    from .wechat_bridge import run
    run(probe=args.probe)


def cmd_set_proactive(args):
    agent = Agent(db_path=args.db)
    agent.set_proactive(args.value in ("on", "true", "1"))
    print("主动行为总开关：", args.value)


def cmd_set_level(args):
    agent = Agent(db_path=args.db)
    agent.set_level_enabled(args.level.upper(), args.value == "on")
    print(f"{args.level.upper()} 开关：{args.value}")


TASK_NAME = "WithMeAgent"


def _write_resident_files(port: int) -> dict:
    """生成常驻启动三件套：env 变量 bat / 启动 bat / 隐藏窗口 vbs。"""
    deploy = os.path.join(project_root(), "deploy")
    os.makedirs(deploy, exist_ok=True)
    env_keys = ["AGENT_LLM_PROVIDER", "AGENT_LLM_BASE_URL", "AGENT_LLM_MODEL",
                "AGENT_LLM_PROMPT_IN_USER", "LLM_API_KEY", "AGENT_MEMORY_PROVIDER",
                "HF_ENDPOINT"]
    env_lines = ["@echo off"]
    for k in env_keys:
        v = os.environ.get(k, "")
        if v:
            env_lines.append(f"set {k}={v}")
    env_bat = os.path.join(deploy, "withme_env.bat")
    with open(env_bat, "w", encoding="utf-8") as f:
        f.write("\r\n".join(env_lines) + "\r\n")

    pythonw = sys.executable.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw):
        pythonw = sys.executable
    start_bat = os.path.join(deploy, "start_withme.bat")
    with open(start_bat, "w", encoding="utf-8") as f:
        f.write("@echo off\r\n"
                f'call "{env_bat}"\r\n'
                f'cd /d "{project_root()}"\r\n'
                f'"{pythonw}" -m src.main serve --port {port} --interval 60\r\n')
    vbs = os.path.join(deploy, "start_hidden.vbs")
    with open(vbs, "w", encoding="utf-8") as f:
        f.write('CreateObject("Wscript.Shell").Run """' + start_bat + '""", 0, False\n')
    return {"env": env_bat, "start": start_bat, "vbs": vbs}


def cmd_install(args):
    import subprocess
    paths = _write_resident_files(args.port)
    tr = f'wscript.exe "{paths["vbs"]}'
    result = subprocess.run(
        ["schtasks", "/Create", "/TN", TASK_NAME, "/TR", tr,
         "/SC", "ONLOGON", "/F"], capture_output=True,
        encoding="gbk", errors="replace")  # schtasks 在中文 Windows 输出 GBK
    if result.returncode == 0:
        print(f"已注册开机自启计划任务 {TASK_NAME}（登录时启动，无窗口后台运行）")
    else:
        # 无管理员权限时的标准替代：把隐藏启动脚本放进当前用户启动文件夹
        startup = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows",
                               "Start Menu", "Programs", "Startup")
        dest = os.path.join(startup, "withme_autostart.vbs")
        shutil.copyfile(paths["vbs"], dest)
        print(f"计划任务无权限创建，已改用启动文件夹自启：\n  {dest}")
    print(f"  启动脚本：{paths['vbs']}")
    print(f"  环境变量：{paths['env']}（改桥接地址/模型/key 就编辑这个文件）")
    print("卸载：python -m src.main uninstall")


def cmd_reminders(args):
    agent = Agent(db_path=args.db)
    rows = agent.conn.execute(
        "SELECT id, text, due_at, status FROM reminders WHERE role_id=?"
        " ORDER BY due_at", (agent.role_id,)).fetchall()
    if not rows:
        print("没有提醒")
        return
    for r in rows:
        print(f"#{r['id']} [{r['status']}] "
              f"{time.strftime('%m-%d %H:%M', time.localtime(r['due_at']))} {r['text']}")


def cmd_plugins(args):
    agent = Agent(db_path=args.db)
    print(f"动作清单（{len(agent.hands.manifest())} 个对外，手部{'开' if agent.config.hands.enabled else '关'}）：")
    for m in agent.hands.manifest():
        info = agent.plugin_info
        src_tag = next((p["plugin"] for p in info if p.get("status") == "ok"), None)
        print(f"  {m['name']:20s} {m['desc'][:38]}")
    if agent.plugin_info:
        print("插件装载：")
        for p in agent.plugin_info:
            extra = f" x{p['actions']}动作" if p.get("status") == "ok" else (
                f" {p.get('error', '')}" if p.get("status") == "error" else "")
            print(f"  {p['plugin']}: {p['status']}{extra}")


def cmd_reflect(args):
    agent = Agent(db_path=args.db)
    result = agent.maybe_reflect(force=True)
    if not result:
        print("反思未执行（未启用）")
        return
    if result.get("ran"):
        a = result["applied"]
        print("反思完成：")
        for i in a["insights"]:
            print(f"  洞察: {i}")
        if a["new_rule"]:
            print(f"  新习惯: {a['new_rule']}")
        if a["trait_shift"]:
            print(f"  人格微调: {a['trait_shift']}")
    else:
        print("反思失败：", result.get("reason"))


def cmd_uninstall(args):
    import subprocess
    removed = []
    result = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                            capture_output=True, encoding="gbk", errors="replace")
    if result.returncode == 0:
        removed.append("计划任务")
    startup = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows",
                           "Start Menu", "Programs", "Startup", "withme_autostart.vbs")
    if os.path.exists(startup):
        os.remove(startup)
        removed.append("启动文件夹脚本")
    print("已移除：" + "、".join(removed) if removed else "没有已注册的自启项")


def cmd_stickers(args):
    """表情包库：sync 下载 / search 检索 / list 概况。"""
    from . import stickers as sticker_mod
    cfg = load_config(os.path.join(os.environ.get("AGENT_CONFIG_DIR") or
                                   os.path.join(project_root(), "config"), "config.yaml"))
    st = sticker_mod.StickerStore(
        media_dir=(cfg.stickers.dir if os.path.isabs(cfg.stickers.dir)
                   else os.path.join(project_root(), cfg.stickers.dir)),
        site=cfg.stickers.site, packs=list(cfg.stickers.packs or []),
        enabled=bool(cfg.stickers.enabled),
        ffmpeg_path=getattr(cfg.stickers, "ffmpeg_path", "") or "",
        send_mode=getattr(cfg.stickers, "send_mode", "static") or "static",
        log=lambda m: print(f"[stickers] {m}"))
    if args.action == "sync":
        print(st.sync(limit=args.limit))
    elif args.action == "search":
        for it in st.search(args.query, limit=5):
            print(f'{it["id"]}\t{it["emotion"]}\t{"/".join(it["tags"][:4])}')
    else:
        packs = {}
        for it in st.items:
            packs[it["pack"]] = packs.get(it["pack"], 0) + 1
        print({"total": len(st.items), "packs": packs, "dir": st.dir})


def main(argv=None):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="拟人化主动智能体（混沌-收敛框架）")
    parser.add_argument("--db", default=None, help="SQLite 路径（默认 data/agent.db）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat", help="交互对话").set_defaults(func=cmd_chat)
    tick_p = sub.add_parser("tick", help="调度 tick（评估一次主动）")
    tick_p.add_argument("--loop", type=int, default=None)
    tick_p.add_argument("--interval", type=float, default=60.0)
    tick_p.set_defaults(func=cmd_tick)
    demo_p = sub.add_parser("demo", help="端到端演示")
    demo_p.add_argument("--fresh", action="store_true", help="清空后重建演示库")
    demo_p.set_defaults(func=cmd_demo)
    st_p = sub.add_parser("stickers", help="表情包库（sync/search/list）")
    st_p.add_argument("action", choices=["sync", "search", "list"], nargs="?", default="list")
    st_p.add_argument("--query", default="", help="search 的关键词")
    st_p.add_argument("--limit", type=int, default=None, help="sync 只下前 N 张")
    st_p.set_defaults(func=cmd_stickers)
    sub.add_parser("status", help="查看状态").set_defaults(func=cmd_status)
    serve_p = sub.add_parser("serve", help="HTTP API + 网页客户端 + 内置调度线程")
    serve_p.add_argument("--host", default="127.0.0.1", help="默认仅本机；对外用 0.0.0.0 + 令牌")
    serve_p.add_argument("--port", type=int, default=8787)
    serve_p.add_argument("--interval", type=float, default=None,
                         help="内置调度线程间隔（秒）；不给则用配置 tick.fast_seconds"
                              "（控制台可热改，改完下一轮生效）")
    serve_p.set_defaults(func=cmd_serve)
    wx_p = sub.add_parser("wechat", help="微信桥：把小满接到个人微信（需 wcferry + PC 微信，"
                                         "见 docs/wechat_setup.md）")
    wx_p.add_argument("--probe", action="store_true", help="连通性自检，不启动桥")
    wx_p.set_defaults(func=cmd_wechat)
    sp = sub.add_parser("set-proactive", help="总开关 on/off")
    sp.add_argument("value", choices=["on", "off"])
    sp.set_defaults(func=cmd_set_proactive)
    sl = sub.add_parser("set-level", help="分级开关 on/off")
    sl.add_argument("level", choices=["L1", "L2", "L3"])
    sl.add_argument("value", choices=["on", "off"])
    sl.set_defaults(func=cmd_set_level)
    sub.add_parser("reflect", help="手动触发一次反思（人格成长）").set_defaults(func=cmd_reflect)
    sub.add_parser("reminders", help="查看提醒列表").set_defaults(func=cmd_reminders)
    sub.add_parser("plugins", help="查看已装载的动作插件").set_defaults(func=cmd_plugins)
    inst = sub.add_parser("install", help="注册 Windows 开机自启（计划任务，无窗口）")
    inst.add_argument("--port", type=int, default=8787)
    inst.set_defaults(func=cmd_install)
    sub.add_parser("uninstall", help="移除开机自启").set_defaults(func=cmd_uninstall)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
