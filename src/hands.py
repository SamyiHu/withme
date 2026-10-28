"""手部（行动层）：给她能做事的能力。

原则（与闸门同一套哲学）：LLM 不直接执行任何东西——感知层最多"提议"一个动作，
引擎校验参数与权限后执行，结果回填回复。全部执行写 hands_log 留痕。

动作清单：
- reminder.set/list/cancel  定时提醒（到点：桌面通知 + 走主动消息通道）
- note.add                  随手记（进记忆库，可检索）
- web.fetch                 抓网页正文（给她"眼睛"）
- desktop.notify            弹 Windows 通知
- shell.run                 本地命令（默认关闭，hands.allow_shell 显式开启）
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from . import bilibili
from .storage import project_root
from .timeparse import parse_when

MANIFEST = [
    {"name": "reminder.set", "params": {"text": "提醒内容", "when": "时间，如：30分钟后/明天早上9点/晚上8点半"},
     "desc": "用户要求到点提醒时使用"},
    {"name": "reminder.list", "params": {}, "desc": "用户问有哪些提醒时使用"},
    {"name": "reminder.cancel", "params": {"text": "要取消的提醒关键词"}, "desc": "用户要求取消提醒时使用"},
    {"name": "note.add", "params": {"text": "要记的内容"}, "desc": "用户说\"记一下/别忘了\"时使用"},
    {"name": "web.search", "params": {"query": "搜索词"},
     "desc": "查实时信息/新闻/不了解的梗/事实核查时使用（联网搜索）"},
    {"name": "web.fetch", "params": {"url": "http(s) 地址"}, "desc": "用户发来具体链接、想看该网页内容时使用"},
    {"name": "image.draw", "params": {"prompt": "想画的画面内容描述"},
     "desc": "用户想要你画一张图/生成一张图片/做个表情包时使用（生成后发给用户）"},
    {"name": "bilibili.watch", "params": {"url": "B站视频链接或BV号"},
     "desc": "用户发来B站视频想聊、或想知道视频内容时使用（读信息+弹幕+热评）"},
    {"name": "bilibili.comments", "params": {"url": "B站视频链接或BV号"},
     "desc": "用户想看某视频的评论区/网友怎么说时使用"},
    {"name": "bilibili.hot", "params": {},
     "desc": "刷B站热门视频速览（看看现在大家在看什么，可以挑感兴趣的聊/评价）"},
    {"name": "meme.generate", "params": {"template": "模板关键词，如 摸头/真香/咬",
                                          "texts": "文字列表，如 ['我才不','真香']",
                                          "avatar": "要谁的头像：我/对方/对方头像，可选"},
     "desc": "发表情包/梗图时使用（300+模板：摸头、啃、举牌、台词图等，含头像梗图）"},
    {"name": "desktop.notify", "params": {"title": "标题", "text": "内容"},
     "desc": "需要立即在桌面弹通知提醒用户时使用"},
    {"name": "shell.run", "params": {"command": "命令"},
     "desc": "执行本地命令（默认禁用，需 hands.allow_shell）"},
]


class Hands:

    def __init__(self, config, conn, role_id: str, memory=None, notifier=None):
        self.cfg = config          # AppConfig
        self.conn = conn
        self.role_id = role_id
        self.memory = memory
        self.notifier = notifier or self._toast_windows
        # 动作注册表：内置与插件走同一条注册路径（插件经 src/plugins.py 装载）
        self.avatar_getter = None   # 桥注入：(who)->bytes，表情包头像梗图用
        self._actions = {}
        for entry in MANIFEST:
            method = {"reminder.set": "_reminder_set", "reminder.list": "_reminder_list",
                      "reminder.cancel": "_reminder_cancel", "note.add": "_note_add",
                      "web.search": "_web_search", "web.fetch": "_web_fetch", "image.draw": "_image_draw",
                      "bilibili.watch": "_bilibili_watch",
                      "bilibili.comments": "_bilibili_comments",
                      "bilibili.hot": "_bilibili_hot", "meme.generate": "_meme_generate",
                      "desktop.notify": "_desktop_notify",
                      "shell.run": "_shell_run"}[entry["name"]]
            self.register_action(entry["name"], entry["params"], entry["desc"],
                                 getattr(self, method), source="builtin")

    # ---------------- 感知提示用清单 ----------------

    def register_action(self, name: str, params: dict, desc: str, handler,
                        source: str = "builtin"):
        """注册一个动作（内置与插件共用）。插件动作由 src/plugins.py 调用。"""
        self._actions[name] = {"params": params or {}, "desc": desc or "",
                               "handler": handler, "source": source}

    def plugin_summary(self) -> list:
        return [{"name": n, "source": a["source"], "desc": a["desc"][:40]}
                for n, a in self._actions.items()]

    def manifest(self) -> list:
        if not self.cfg.hands.enabled:
            return []
        items = [{"name": n, "params": a["params"], "desc": a["desc"]}
                 for n, a in self._actions.items()]
        if not self.cfg.hands.allow_shell:
            items = [m for m in items if m["name"] != "shell.run"]
        return items

    # ---------------- 执行入口 ----------------

    def execute(self, name: str, params: dict, now: float = None) -> dict:
        now = time.time() if now is None else now
        params = params if isinstance(params, dict) else {}
        result = {"ok": False, "summary": "未知动作"}
        entry = self._actions.get(name)
        if not self.cfg.hands.enabled:
            result = {"ok": False, "summary": "手部未启用"}
        elif entry is None:
            result = {"ok": False, "summary": f"未知动作 {name}"}
        else:
            try:
                result = entry["handler"](params, now) or {"ok": False, "summary": "无结果"}
            except Exception as e:  # noqa: BLE001  手部失败不影响对话
                result = {"ok": False, "summary": f"执行失败：{e!r}"}
        self._log(name, params, result, now)
        return result

    # ---------------- 动作实现 ----------------

    def _reminder_set(self, p, now):
        text = str(p.get("text", "")).strip()[:80]
        when_raw = str(p.get("when", "")).strip()
        if not text or not when_raw:
            return {"ok": False, "summary": "缺少 text 或 when"}
        due = parse_when(when_raw, now)
        if due is None:
            return {"ok": False, "summary": f"时间没听懂：{when_raw}"}
        if due <= now:
            return {"ok": False, "summary": "时间已是过去"}
        cur = self.conn.execute(
            "INSERT INTO reminders(role_id, text, due_at, status, created_at)"
            " VALUES(?,?,?,'open',?)", (self.role_id, text, due, now))
        self.conn.commit()
        rid = cur.lastrowid
        local = time.strftime("%m-%d %H:%M", time.localtime(due))
        return {"ok": True, "summary": f"已设提醒#{rid}：{local}「{text}」", "reminder_id": rid,
                "due_at": due}

    def _reminder_list(self, p, now):
        rows = self.conn.execute(
            "SELECT id, text, due_at FROM reminders WHERE role_id=? AND status='open'"
            " ORDER BY due_at LIMIT 10", (self.role_id,)).fetchall()
        if not rows:
            return {"ok": True, "summary": "当前没有待办的提醒"}
        lines = [f"#{r['id']} {time.strftime('%m-%d %H:%M', time.localtime(r['due_at']))}"
                 f" {r['text']}" for r in rows]
        return {"ok": True, "summary": "待办提醒：\n" + "\n".join(lines)}

    def _reminder_cancel(self, p, now):
        key = str(p.get("text", "")).strip()
        if not key:
            return {"ok": False, "summary": "缺少要取消的提醒关键词"}
        rows = self.conn.execute(
            "SELECT id, text FROM reminders WHERE role_id=? AND status='open'", (self.role_id,)).fetchall()
        hits = [r for r in rows if key in r["text"]]
        if not hits:
            return {"ok": False, "summary": f"没找到包含「{key}」的提醒"}
        for r in hits:
            self.conn.execute("UPDATE reminders SET status='cancelled' WHERE id=?", (r["id"],))
        self.conn.commit()
        return {"ok": True, "summary": f"已取消 {len(hits)} 条提醒（{key}）"}

    def _note_add(self, p, now):
        text = str(p.get("text", "")).strip()[:120]
        if not text:
            return {"ok": False, "summary": "缺少内容"}
        if self.memory is not None:
            self.memory.add(self.role_id, "event", text, salience=0.7, now=now, verbatim=True)
        return {"ok": True, "summary": f"记下了：{text}"}

    def _web_search(self, p, now):
        """联网搜索（智谱 Web Search Pro）：实时信息/新梗/事实核查。"""
        if not self.cfg.web_search.enabled:
            return {"ok": False, "summary": "搜索未开启（web_search.enabled=false）"}
        query = str(p.get("query", "")).strip()[:120]
        if not query:
            return {"ok": False, "summary": "缺少搜索词"}
        items, used, errors = [], "", []
        for provider in (self.cfg.web_search.chain or ["bigmodel"]):
            try:
                if provider == "bigmodel":
                    from .websearch import search
                    items = search(query, self.cfg.web_search)
                elif provider == "cush":
                    from .cush_search import search as cush_search, CushSearchError
                    try:
                        items = cush_search(
                            query, self.cfg.web_search.cush_exe,
                            limit=self.cfg.web_search.max_results,
                            timeout_s=self.cfg.web_search.cush_timeout_seconds)
                    except CushSearchError as e:
                        items = []
                        errors.append(f"cush: {e}")
                else:
                    continue
            except Exception as e:  # noqa: BLE001
                items = []
                errors.append(f"{provider}: {e}")
            if items:
                used = provider
                break
        if not items:
            return {"ok": False, "summary": f"「{query}」没搜到（{'; '.join(errors) or '无结果'}）"}
        # 每条带发布日期（已知时）：模型对照"当前时间"判断新旧，回复能说"据X月X日消息"
        parts = [f"《{it['title']}》" + (f"（{it['date']}）" if it.get("date") else "")
                 + f"{it['content'][:100]}" for it in items]
        return {"ok": True,
                "summary": f"搜索「{query}」[{used}]：" + "；".join(parts)}

    def _web_fetch(self, p, now):
        url = str(p.get("url", "")).strip()
        if not re.match(r"^https?://", url):
            return {"ok": False, "summary": "只支持 http(s) 地址"}
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 WithMeAgent"})
        with urllib.request.urlopen(req, timeout=self.cfg.hands.web_timeout) as resp:
            raw = resp.read(self.cfg.hands.web_max_bytes + 1)
        if len(raw) > self.cfg.hands.web_max_bytes:
            raw = raw[:self.cfg.hands.web_max_bytes]
        # 中文站大多是 GBK 系：先按声明编码，再 utf-8/gbk 严格试错，最后才 replace
        charset = (resp.headers.get_content_charset() or "").strip().lower()
        html = None
        for enc in [e for e in (charset, "utf-8", "gbk") if e]:
            try:
                html = raw.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if html is None:
            html = raw.decode("utf-8", "replace")
        text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        excerpt = text[:600]
        return {"ok": True, "summary": f"网页内容（{url}）：{excerpt}"}

    def _image_draw(self, p, now):
        prompt = str(p.get("prompt", "")).strip()[:200]
        if not prompt:
            return {"ok": False, "summary": "缺少要画的内容"}
        if not self.cfg.image_gen.enabled:
            return {"ok": False, "summary": "画图未开启（image_gen.enabled=false）"}
        try:
            from .imagegen import generate_image  # 延迟导入（核心零依赖）
            out = generate_image(prompt, self.cfg.image_gen, self.cfg.llm,
                                 os.path.join(project_root(), "data", "wechat_media"))
        except Exception as e:  # noqa: BLE001
            msg = f"画图失败：{e}"
            if "1301" in msg or "contentFilter" in msg:
                # 上游（CogView）内容安全拦截：版权角色/真人/敏感题材，不是故障。
                # 给模型可行动的出路——换描述性说法重画，别只会复述"内容不安全"
                return {"ok": False, "summary": (
                    "被上游内容安全拦截了（版权角色/真人明星/敏感题材都会被拒，"
                    "如哆啦A梦、米老鼠）。别直接说画不了：去掉角色名和品牌，"
                    "只描述外形特征重画一次（如「蓝色圆脸机器猫」「穿红裙的大耳朵"
                    "卡通老鼠」）；再失败就坦白说这个题材画不了。")}
            return {"ok": False, "summary": msg}
        return {"ok": True, "summary": f"画好了（{prompt[:40]}），把图发给用户",
                "image": out["path"] or out["url"]}

    def _bilibili_watch(self, p, now):
        url = str(p.get("url", "")).strip()
        if not url:
            return {"ok": False, "summary": "缺少视频链接"}
        try:
            describe = self._describe_cover  # 看封面（视觉回落到对话模型，失败降级）
            summary = bilibili.watch_summary(url, describe_cover=describe)
            return {"ok": True, "summary": summary}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "summary": f"看不了：{e}"}

    def _describe_cover(self, cover_url: str) -> str:
        raw = bilibili.default_fetch(cover_url, binary=True)[0]
        if len(raw) > 900_000:
            raise LLMClientError("封面过大")
        from .llm_client import OpenAICompatibleClient  # 延迟导入
        client = OpenAICompatibleClient(self.cfg.llm)
        return client.render_image(
            "这是一张B站视频封面。用两三句话描述画面内容、风格与大致题材。", raw)

    def _bilibili_comments(self, p, now):
        url = str(p.get("url", "")).strip()
        if not url:
            return {"ok": False, "summary": "缺少视频链接"}
        try:
            summary = bilibili.comments_summary(url)
            return {"ok": True, "summary": summary}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "summary": f"评论读不了：{e}"}

    def _bilibili_hot(self, p, now):
        """刷B站热门：速览标题+数据。给"主动找话题/像群友一样刷视频"用。"""
        try:
            items = bilibili.popular(max_items=6)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "summary": f"热门刷不了：{e}"}
        if not items:
            return {"ok": False, "summary": "热门列表为空"}
        lines = []
        for it in items:
            play = it.get("play") or 0
            play_s = f"{play // 10000}万" if play >= 10000 else str(play)
            lines.append(f"《{it['title'][:40]}》{it.get('owner', '')} "
                         f"播放{play_s} 赞{it.get('like', 0)}")
        return {"ok": True, "summary": "B站热门：" + "；".join(lines)}

    def _meme_generate(self, p, now):
        """表情包（MemeCrafters/meme-generator）：模板模糊匹配 + 可带头像。"""
        try:
            from meme_generator import get_meme, get_memes, Image
        except ImportError:
            return {"ok": False, "summary": "表情包库未安装（pip install meme-generator）"}
        template = str(p.get("template", "")).strip()
        texts = p.get("texts")
        if isinstance(texts, str):
            texts = [t for t in texts.split("|") if t]
        texts = [str(t)[:40] for t in (texts or [])][:4]
        key = self._meme_match(template)
        if not key:
            keys = ", ".join(self._meme_keys_cache()[:30])
            return {"ok": False, "summary": f"没有「{template}」模板。可用的有：{keys}…"}
        imgs = []
        who = str(p.get("avatar", "")).strip()
        if self.avatar_getter:
            # 没指定头像时默认拿 bot 自己的（自嘲梗符合人设）；petpet 类必须有图
            target = ("对方" if who in ("对方", "用户", "对方头像", "他的", "她的")
                      else "我") if who else "我"
            try:
                avatar = self.avatar_getter(target)
                if avatar:
                    imgs = [Image("avatar.img", avatar)]
            except Exception:  # noqa: BLE001  头像失败不挡文字梗图
                pass
        try:
            data = get_meme(key).generate(images=imgs, texts=texts, options={})
        except Exception as e:  # noqa: BLE001  pyo3 异常 str() 会炸，用类名
            return {"ok": False,
                    "summary": f"表情包生成失败（{key}）：{type(e).__name__}"}
        if not isinstance(data, (bytes, bytearray)):
            # 此 rust 绑定不抛异常而是返回错误对象（ImageAssetMissing 等）
            name = type(data).__name__
            if "Asset" in name:
                return {"ok": False,
                        "summary": "表情包模板资源还没下载完（后台在下，稍后再试）"}
            if "ImageNumber" in name:
                return {"ok": False, "summary": f"「{template}」这模板需要头像图，这次没拿到"}
            if "TextNumber" in name:
                return {"ok": False, "summary": f"「{template}」需要配文字（texts 参数）"}
            return {"ok": False, "summary": f"表情包生成失败（{key}）：{name}"}
        ext = "gif" if data[:3] == bytes([0x47, 0x49, 0x46]) else "png"
        out = os.path.join(project_root(), "data", "wechat_media",
                           f"meme_{int(now * 1000)}.{ext}")
        with open(out, "wb") as f:
            f.write(data)
        return {"ok": True, "summary": f"表情包做好了（{key}），把图发给用户",
                "image": out}

    _MEME_KEYS = None

    def _meme_keys_cache(self):
        if self._MEME_KEYS is None:
            try:
                from meme_generator import get_memes
                self._MEME_KEYS = [m.key for m in get_memes()]
            except Exception:  # noqa: BLE001
                self._MEME_KEYS = []
        return self._MEME_KEYS

    def _meme_match(self, query: str) -> str:
        """模板名/关键词模糊匹配（query 是中文口语，如'摸头'->petpet）。"""
        if not query:
            return ""
        q = query.strip().lower()
        try:
            from meme_generator import get_memes
            for m in get_memes():
                if m.key.lower() == q:
                    return m.key
            for m in get_memes():
                kws = list(m.info.keywords or []) + [m.key]
                if any(q in str(k).lower() or str(k).lower() in q for k in kws):
                    return m.key
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _desktop_notify(self, p, now):
        title = str(p.get("title", "小满"))[:40]
        text = str(p.get("text", ""))[:120]
        if not text:
            return {"ok": False, "summary": "缺少通知内容"}
        ok = self.notifier(title, text)
        return {"ok": bool(ok), "summary": f"桌面通知：{title} - {text}" if ok else "通知失败"}

    def _shell_run(self, p, now):
        if not self.cfg.hands.allow_shell:
            return {"ok": False, "summary": "shell 动作未开启（hands.allow_shell）"}
        cmd = str(p.get("command", "")).strip()
        if not cmd:
            return {"ok": False, "summary": "缺少命令"}
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=10,
                           encoding="utf-8", errors="replace")
        out = ((r.stdout or "") + (r.stderr or ""))[:400]
        return {"ok": r.returncode == 0, "summary": f"exit={r.returncode} {out}"}

    # ---------------- 到点提醒（调度调用） ----------------

    def due_reminders(self, now: float) -> list:
        rows = self.conn.execute(
            "SELECT * FROM reminders WHERE role_id=? AND status='open' AND due_at<=?",
            (self.role_id, now)).fetchall()
        return [dict(r) for r in rows]

    def mark_done(self, reminder_id: int):
        self.conn.execute("UPDATE reminders SET status='done' WHERE id=?", (reminder_id,))
        self.conn.commit()

    # ---------------- 桌面通知（Windows toast，无依赖） ----------------

    @staticmethod
    def _toast_windows(title: str, text: str) -> bool:
        if sys.platform != "win32":
            print(f"[notify] {title}: {text}")
            return True
        t = title.replace("'", "''")
        b = text.replace("'", "''")
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "Add-Type -AssemblyName System.Drawing; "
            "$n = New-Object System.Windows.Forms.NotifyIcon; "
            "$n.Icon = [System.Drawing.SystemIcons]::Information; "
            "$n.Visible = $true; "
            f"$n.ShowBalloonTip(8000, '{t}', '{b}', 'Info'); "
            "Start-Sleep -Seconds 9; $n.Dispose()")
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, timeout=15)
            return True
        except Exception:  # noqa: BLE001
            print(f"[notify] {title}: {text}")
            return False

    def _log(self, action, params, result, now):
        self.conn.execute(
            "INSERT INTO hands_log(role_id, action, params, result, created_at)"
            " VALUES(?,?,?,?,?)",
            (self.role_id, action, json.dumps(params, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False), now))
        self.conn.commit()
