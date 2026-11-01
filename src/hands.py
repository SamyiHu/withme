"""手部（行动层）：给她能做事的能力。

原则（与闸门同一套哲学）：LLM 不直接执行任何东西——感知层最多"提议"一个动作，
引擎校验参数与权限后执行，结果回填回复。全部执行写 hands_log 留痕。

动作清单：
- reminder.set/list/cancel  定时提醒（到点：桌面通知 + 走主动消息通道）
- note.add                  随手记（进记忆库，可检索）
- web.fetch                 抓网页正文（给她"眼睛"）
- shopping.link             按型号生成京东/淘宝搜索直达链接（不编商品页）
- image.draw               画图（生成后发图）
- song.sing                点唱：后台生成 55 秒歌声，好了发语音气泡
- desktop.notify            弹 Windows 通知
- shell.run                 本地命令（默认关闭，hands.allow_shell 显式开启）
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

from . import bilibili


def _post_worker(url: str, payload: bytes, timeout: int,
                 retries: int = 12, delay: float = 5.0) -> dict:
    """调换色 worker，503（预热中/忙）和连接失败自动重试，最多约 60 秒。"""
    import urllib.error
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (502, 503, 504) and attempt < retries - 1:
                print(f"[song] worker 未就绪（HTTP {e.code}），{delay:g}s 后重试",
                      flush=True)
                time.sleep(delay)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries - 1:
                print(f"[song] worker 连接失败，{delay:g}s 后重试：{e!r}", flush=True)
                time.sleep(delay)
                continue
            raise
    raise last
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
    {"name": "shopping.link", "params": {"keyword": "商品名/型号，如：米家空气净化器4 Pro",
                                         "site": "可选：jd（默认）/ taobao"},
     "desc": "对方要购买链接/京东淘宝链接时使用：按型号生成商城搜索直达链接"
             "（不编造商品页 ID，链接一定点得开）"},
    {"name": "image.draw", "params": {"prompt": "想画的画面内容描述"},
     "desc": "用户想要你画一张图/生成一张图片/做个表情包时使用（生成后发给用户）"},
    {"name": "song.sing", "params": {"lyrics": "对方给了歌词就原样填入（可多行）；没给才自己写 2-4 句短句",
                                      "style": "用户说的曲风/情绪，如：温柔民谣/轻快流行/伤感慢歌",
                                      "voice": "可选：用户指定的音色，如 女声/男声/低沉男声；不写用后端默认",
                                      "duration": "可选：唱多久，如 30 / 45秒 / 1分钟；不写用默认",
                                      "title": "可选，歌名或主题"},
     "desc": "对方明确点歌/让你唱歌且已经说了（或确认了）曲风时使用。"
             "style 必须来自用户或用户说「随便」后你挑的具体风格；对方没说曲风就留空，"
             "让回复先问一句想听什么风格。"
             "对方把歌词发给你时，lyrics 原样填入（最多 12 行，长歌词记得同时给 duration）。"},
    {"name": "song.cover", "params": {"title": "歌名，如：晴天",
                                       "artist": "可选：歌手名，如：周杰伦",
                                       "gender": "可选：原唱性别 male/female；知道就填，用于自动定调",
                                       "section": "可选：副歌/高潮/chorus=自动找副歌；不填=整首或按 start",
                                       "lyric": "可选：从这句歌词开始唱，如「窗外的天气」；系统查 LRC 时间轴",
                                       "acappella": "可选：清唱/不要伴奏 true",
                                       "duet": "可选：双轨合唱 true（同一条人声再跑一轨，副轨降调增加低音层）",
                                       "duet_pitch": "可选：双轨副轨降几个半音，默认 -12（低八度），范围 -12..-1",
                                       "duration": "可选：翻唱时长秒数，如 60 / 90；不写=整首（section=副歌时默认 30 秒）",
                                       "start": "可选：从第几秒开始，如 80（1 分 20 秒）；可与 duration 组合自定义唱段"},
     "desc": "对方说「翻唱/唱《歌名》/用你的声音唱xxx」且指的是现成的歌曲时使用（不是让你自己编歌）。"
             "系统会自动搜歌→下载→用你的音色翻唱→发回。全首约 2 分钟处理好；"
             "给 duration 参数可以只翻一段（更快）。知道原唱歌手性别时填 gender，男调会自动升到她的音区，女调保持原调。"
             "对方说“唱高潮/副歌”就填 section=副歌（自动定位）；说“从第几秒唱多久”就填 start/duration；"
             "说“从‘某句歌词’开始唱”就把那句词填 lyric（自动查 LRC 时间轴）。"
             "对方说“清唱/不要伴奏/纯人声”就填 acappella=true。"
             "对方说“双轨/合唱/加一层低音”就填 duet=true（同一条人声再跑一轨，"
             "副轨默认降 12 个半音（低八度）；不需要训练第二个音色）。"
             "只有对方明确让你唱/用你的声音翻唱时才用；对方说「你点一首/选一首/推荐一首」给别人唱、"
             "或让你点歌但唱歌的人不是你时，不要用这个动作，直接用文字回歌名。"},
    {"name": "task.status", "params": {},
     "desc": "用户问「刚才让你做的呢/唱到哪了/画好了吗/任务进度」时使用；"
             "会列出正在跑和最近完成的后台任务"},
    {"name": "task.cancel", "params": {"kind": "可选：song/image/all，默认 all"},
     "desc": "用户明确说「别唱了/取消/不用画了」时使用，取消正在跑的后台任务"},
    {"name": "task.retry", "params": {"kind": "可选：song/image/all，默认 all"},
     "desc": "用户说「刚才那首重来/再试一次/画失败了再来」时使用，重试最近失败的任务"},
    {"name": "memory.recall", "params": {"query": "要回忆的关键词/话题/人名",
                                          "limit": "可选：最多几条，默认 8"},
     "desc": "用户提起很久以前的事、要你回忆旧对话/旧任务时使用；"
             "会按关键词翻聊天记录和任务记录"},
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


# 商城搜索直达：不编造商品页 ID——搜索链接永远点得开，商品页 ID 编一个就是死链。
_SHOP_SITES = {"jd": ("京东", "https://search.jd.com/Search?keyword={}"),
               "taobao": ("淘宝", "https://s.taobao.com/search?q={}")}


def shopping_link(keyword: str, site: str = "jd") -> str:
    """商品/型号 -> 商城搜索直达链接（缺关键词返回空串）。"""
    kw = re.sub(r"\s*(?:京东|淘宝|天猫)\s*", " ", str(keyword or "")).strip()
    if not kw:
        return ""
    key = "taobao" if str(site or "").lower().startswith(("tao", "淘", "天")) else "jd"
    return _SHOP_SITES[key][1].format(urllib.parse.quote(kw[:60]))


class Hands:

    def __init__(self, config, conn, role_id: str, memory=None, notifier=None,
                 task_owner: str = ""):
        self.cfg = config          # AppConfig
        self.conn = conn
        self.role_id = role_id
        self.memory = memory
        self.notifier = notifier or self._toast_windows
        # 任务先落库再执行：桥重启后，上一个 bridge 进程留下的 running 任务
        # 不会永远显示"在唱"，而是被标成 failed 等控制台/健康面板处理。
        from .taskstore import TaskStore
        self._db_lock = threading.Lock()
        self.tasks = TaskStore(conn, role_id, lock=self._db_lock, owner=task_owner)
        self.orphaned_tasks = []
        if task_owner:
            try:
                self.tasks.fail_orphans(before=time.time(), owner=task_owner)
                self.orphaned_tasks = list(getattr(self.tasks, "last_orphans", []))
            except Exception as e:  # noqa: BLE001  清理失败不挡启动
                print(f"[task] 孤儿任务清理失败：{e!r}")
        # 当前这一轮是谁在让我做事：权限硬闸与任务归属用。
        self.turn_context = {}
        # 慢动作的后台投递回调（桥注入）：result={"action","ok","summary","image"}。
        self.async_sender = None
        self.async_jobs = {}
        self._async_lock = threading.Lock()
        self._async_actions = {"image.draw", "bilibili.watch", "web.search"}
        # 动作注册表：内置与插件走同一条注册路径（插件经 src/plugins.py 装载）
        self.avatar_getter = None   # 桥注入：(who)->bytes，表情包头像梗图用
        self.image_verifier = None  # Agent 注入：(prompt, path)->bool|None，出图自检
        # 唱歌：桥注入的投递回调（info: {path,duration_ms,lyrics} 或 {error}）。
        # 生成走后台线程，动作本身立即返回，不阻塞这一轮回复。
        self.song_sender = None
        self.song_jobs = {}
        self._song_lock = threading.Lock()
        self._song_running = False
        self._actions = {}
        for entry in MANIFEST:
            method = {"reminder.set": "_reminder_set", "reminder.list": "_reminder_list",
                      "reminder.cancel": "_reminder_cancel", "note.add": "_note_add",
                      "web.search": "_web_search", "web.fetch": "_web_fetch", "image.draw": "_image_draw",
                      "shopping.link": "_shopping_link",
                      "song.sing": "_song_sing", "song.cover": "_song_cover",
                      "task.status": "_task_status", "task.cancel": "_task_cancel",
                      "task.retry": "_task_retry",
                      "memory.recall": "_memory_recall",
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

    def snapshot(self) -> list:
        """控制台展示用：动作名 / 来源 / 说明 / 参数键（handler 不出进程）。"""
        return [{"name": n, "source": a["source"], "desc": a["desc"],
                 "params": list((a["params"] or {}).keys())}
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
        # 权限硬闸：群友的消息可以触发点歌/画图，但不能借她的手跑本地命令。
        # 只在“明确标了不是创造者”时拦；测试/命令行直接调用（空 context）保持旧行为。
        ctx = self.turn_context or {}
        if name in ("shell.run",) and ctx and ctx.get("is_owner") is False:
            result = {"ok": False, "summary": "这个动作只有创造者能让我做"}
        elif not self.cfg.hands.enabled:
            result = {"ok": False, "summary": "手部未启用"}
        elif entry is None:
            result = {"ok": False, "summary": f"未知动作 {name}"}
        elif (name in self._async_actions and self.async_sender is not None
              and not params.get("_sync")):
            # 画图这类几十秒的慢动作：先回“在弄了”，完成后由桥作为新事件投递。
            # 这样前台对话不会被一张图堵住（全双工最小版）。
            result = self._start_async(name, params, now)
        else:
            try:
                result = entry["handler"](params, now) or {"ok": False, "summary": "无结果"}
            except Exception as e:  # noqa: BLE001  手部失败不影响对话
                result = {"ok": False, "summary": f"执行失败：{e!r}"}
        self._log(name, params, result, now)
        return result

    # ---------------- 后台动作（慢手部） ----------------

    def _task_meta(self) -> dict:
        ctx = self.turn_context or {}
        return {"requester": str(ctx.get("requester") or ""),
                "speaker": str(ctx.get("speaker") or ""),
                "role": str(ctx.get("role") or ""),
                "is_owner": bool(ctx.get("is_owner", False)),
                "room": str(ctx.get("room") or "")}

    def _task_params(self, params: dict) -> dict:
        out = dict(params or {})
        out["_meta"] = self._task_meta()
        if not out.get("_request"):
            out["_request"] = str((self.turn_context or {}).get("text") or "")[:300]
        return out

    def _start_async(self, name: str, params: dict, now: float) -> dict:
        job_id = uuid.uuid4().hex[:8]
        task = {}
        try:
            task = self.tasks.create(name, self._task_params(params))
        except Exception as e:  # noqa: BLE001  留痕失败不挡执行
            print(f"[task] 建后台动作失败：{e!r}")
        with self._async_lock:
            self.async_jobs[job_id] = {"status": "running", "action": name,
                                       "task_id": task.get("id", ""),
                                       "cancelled": False, "started_at": now}
            while len(self.async_jobs) > 20:
                self.async_jobs.pop(next(iter(self.async_jobs)))
        threading.Thread(target=self._async_worker,
                         args=(job_id, name, dict(params), now,
                               task.get("id", "")),
                         daemon=True, name="hand-async").start()
        label = {"image.draw": "在画了，画好就发你",
                 "bilibili.watch": "在看了，看完跟你讲",
                 "web.search": "在查了，查到跟你说"}.get(
                     name, "在弄了，好了发你")
        return {"ok": True, "pending": True, "job_id": job_id,
                "task_id": task.get("id", ""),
                "summary": label + "（这轮先不用等）"}

    def _async_worker(self, job_id: str, name: str, params: dict,
                      now: float, task_id: str):
        self._task_update(task_id, status="running", stage="working")
        self._task_heartbeat(task_id, stage="working")
        try:
            entry = self._actions.get(name)
            result = entry["handler"](params, now) if entry else {
                "ok": False, "summary": f"未知动作 {name}"}
            if self._async_cancelled(job_id):
                self._task_update(task_id, status="cancelled", stage="cancelled",
                                  error="用户取消")
                return
            if result.get("ok"):
                self._task_update(task_id, status="ready", stage="done",
                                  result={"summary": result.get("summary"),
                                          "image": result.get("image")})
            else:
                self._task_update(task_id, status="failed", stage="failed",
                                  error=str(result.get("summary") or "")[:500])
            self.async_jobs[job_id] = {"status": "done" if result.get("ok") else "failed",
                                       "action": name, "task_id": task_id,
                                       "result": result}
            if self.async_sender is not None:
                try:
                    self.async_sender({"job_id": job_id, "action": name,
                                       "task_id": task_id, "result": result})
                except Exception as e:  # noqa: BLE001  投递失败保留任务结果
                    print(f"[task] 后台动作投递失败：{e!r}")
        except Exception as e:  # noqa: BLE001
            self._task_update(task_id, status="failed", stage="failed", error=repr(e))
            self.async_jobs[job_id] = {"status": "failed", "action": name,
                                       "task_id": task_id, "error": repr(e)}
            if self.async_sender is not None:
                try:
                    self.async_sender({"job_id": job_id, "action": name,
                                       "task_id": task_id,
                                       "result": {"ok": False,
                                                  "summary": f"执行失败：{e!r}"}})
                except Exception:
                    pass

    def _async_cancelled(self, job_id: str) -> bool:
        with self._async_lock:
            return bool(self.async_jobs.get(job_id, {}).get("cancelled"))

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
        # 链接必须带上：2026-11-01 实测群里要"京东链接"，搜索结果只喂了标题+摘要，
        # 模型手里根本没有 URL，只能回"链接我就不发外链了，你自己搜型号"。
        parts = []
        for it in items:
            seg = (f"《{it['title']}》" + (f"（{it['date']}）" if it.get("date") else "")
                   + f"{it['content'][:100]}")
            if it.get("link"):
                seg += f" 链接：{it['link']}"
            parts.append(seg)
        return {"ok": True,
                "summary": f"搜索「{query}」[{used}]：" + "；".join(parts)}

    def _shopping_link(self, p, now):
        """对方要购买链接：给商城搜索直达链接（含型号），不臆造商品页。"""
        kw = str(p.get("keyword", "")).strip()
        site = str(p.get("site", "jd")).strip() or "jd"
        url = shopping_link(kw, site)
        if not url:
            return {"ok": False, "summary": "缺少商品名或型号"}
        name = _SHOP_SITES["taobao" if site.lower().startswith(("tao", "淘", "天"))
                           else "jd"][0]
        return {"ok": True, "summary": f"{name}搜索「{kw}」直达链接：{url}"}

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
        from .imagegen import generate_image, sanitize_prompt  # 延迟导入（核心零依赖）
        draw_prompt, swapped = sanitize_prompt(prompt)
        try:
            out = generate_image(draw_prompt, self.cfg.image_gen, self.cfg.llm,
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
        if self._drawn_mismatch(draw_prompt, out.get("path")):
            return {"ok": False, "summary": (
                "图是生成了，但跟要画的对不上（上游碰到版权角色/真人/品牌会"
                "静默换成一张无关的图，不报错）。这张没发出去。别硬说画好了，"
                "也别直接说画不了：去掉角色名和品牌，只写外形、颜色、动作再画"
                "一次；再对不上就坦白这个题材画不出来。")}
        summary = f"画好了（{draw_prompt[:40]}"
        if swapped:
            summary += "；按外形描述画的"
        summary += "），把图发给用户"
        return {"ok": True, "summary": summary,
                "image": out["path"] or out["url"]}

    def _drawn_mismatch(self, prompt, path) -> bool:
        """出图自检：上游对版权角色/真人会静默返回无关图（200 + 正常 URL）。
        自检只做减法——未注入、没落盘、视觉端出错，一律当通过。"""
        fn = getattr(self, "image_verifier", None)
        if fn is None or not path or not os.path.isfile(str(path)):
            return False
        try:
            return fn(prompt, path) is False
        except Exception:  # noqa: BLE001  自检失败不挡正常发图
            return False

    # ---------------- 唱歌 ----------------

    def song_running(self) -> bool:
        with self._song_lock:
            return self._song_running

    def cancel_song_jobs(self) -> int:
        """用户纠正“不是让你唱/搞错了”时取消还在跑的点歌/翻唱，避免发错歌。"""
        with self._song_lock:
            n = 0
            for job in self.song_jobs.values():
                if job.get("status") in ("searching", "downloading",
                                         "converting", "running"):
                    job["cancelled"] = True
                    n += 1
            return n

    def _song_cancelled(self, job_id: str) -> bool:
        return bool(self.song_jobs.get(job_id, {}).get("cancelled"))

    def _task_update(self, task_id: str, **fields):
        if not task_id:
            return
        try:
            self.tasks.update(task_id, **fields)
        except Exception as e:  # noqa: BLE001  任务留痕失败不影响执行
            print(f"[task] 更新失败 {task_id}: {e!r}")

    def _task_heartbeat(self, task_id: str, stage: str = None):
        if not task_id:
            return
        try:
            self.tasks.heartbeat(task_id, stage=stage)
        except Exception as e:  # noqa: BLE001
            print(f"[task] 心跳失败 {task_id}: {e!r}")

    @staticmethod
    def _task_title(task: dict) -> str:
        try:
            p = json.loads(task.get("params") or "{}")
        except Exception:  # noqa: BLE001
            p = {}
        meta = p.get("_meta") or {}
        title = p.get("title") or p.get("prompt") or p.get("query") or ""
        if not title and task.get("kind") == "song.sing":
            title = (str(p.get("lyrics") or "").splitlines() or [""])[0][:24]
        who = meta.get("speaker") or meta.get("requester") or ""
        return (str(title)[:40] + (" / " + str(who) if who else "")).strip(" /")

    def _task_status(self, p, now):
        rows = self.tasks.recent(8)
        if not rows:
            return {"ok": True, "summary": "现在没有在跑的任务，最近也没有任务记录"}
        icons = {"queued": "排队", "running": "进行中", "ready": "已完成",
                 "failed": "失败", "cancelled": "已取消"}
        lines = []
        for t in rows[:6]:
            title = self._task_title(t) or t.get("kind")
            stage = f"/{t['stage']}" if t.get("stage") else ""
            err = f"（{str(t.get('error') or '')[:40]}）" if t.get("error") else ""
            lines.append(f"{icons.get(t['status'], t['status'])}·{t['kind']}{stage}"
                         f"：{title}{err}")
        return {"ok": True, "summary": "任务：" + "；".join(lines)}

    def _can_control_task(self, task: dict) -> bool:
        ctx = self.turn_context or {}
        if not ctx or ctx.get("is_owner"):
            return True
        try:
            meta = (json.loads(task.get("params") or "{}").get("_meta") or {})
        except Exception:  # noqa: BLE001
            meta = {}
        requester = str(meta.get("requester") or "")
        return bool(requester and requester == str(ctx.get("requester") or ""))

    @staticmethod
    def _task_kind(value: str) -> str:
        k = str(value or "all").strip().lower()
        if k in ("唱歌", "点歌", "翻唱", "song", "sing", "cover"):
            return "song"
        if k in ("画图", "图片", "image", "draw", "画"):
            return "image"
        if k in ("视频", "看视频", "bilibili", "b站", "video"):
            return "bilibili"
        return k

    def _task_cancel(self, p, now):
        kind = self._task_kind(p.get("kind") or "all")
        active = [t for t in self.tasks.recent(30)
                  if t.get("status") in ("queued", "running")
                  and (kind in ("all", "", t.get("kind")) or kind in str(t.get("kind")))]
        if not active:
            return {"ok": True, "summary": "现在没有可取消的任务"}
        allowed = [t for t in active if self._can_control_task(t)]
        if not allowed:
            return {"ok": False, "summary": "这个任务不是你让我做的，我不能替你取消"}
        n = 0
        for t in allowed:
            if str(t.get("kind", "")).startswith("song"):
                n += self.cancel_song_jobs()
            self._task_update(t["id"], status="cancelled", stage="cancelled",
                              error="用户取消")
        with self._async_lock:
            for job in self.async_jobs.values():
                if job.get("status") == "running":
                    if kind in ("all", "", job.get("action")):
                        job["cancelled"] = True
        return {"ok": True, "summary": f"已取消 {len(allowed)} 个任务"}

    def _task_retry(self, p, now):
        """重试最近失败的同类任务；重试会新建一条任务，不复活旧记录。"""
        kind = self._task_kind(p.get("kind") or "all")
        failed = [t for t in self.tasks.recent(30)
                  if t.get("status") == "failed"
                  and (kind in ("all", "", t.get("kind")) or kind in str(t.get("kind")))]
        if not failed:
            return {"ok": False, "summary": "没有可重试的失败任务"}
        if not self._can_control_task(failed[0]):
            return {"ok": False, "summary": "这个任务不是你让我做的，我不能替你重试"}
        t = failed[0]
        try:
            params = json.loads(t.get("params") or "{}")
        except Exception:  # noqa: BLE001
            params = {}
        params.pop("_meta", None)
        kind_name = str(t.get("kind") or "")
        try:
            if kind_name == "song.cover":
                out = self._song_cover(params, now)
            elif kind_name == "song.sing":
                out = self._song_sing(params, now)
            elif kind_name == "image.draw":
                params["_sync"] = False
                out = self.execute("image.draw", params, now)
            elif kind_name == "bilibili.watch":
                params["_sync"] = False
                out = self.execute("bilibili.watch", params, now)
            elif kind_name == "web.search":
                params["_sync"] = False
                out = self.execute("web.search", params, now)
            else:
                return {"ok": False, "summary": f"这个任务不会重试：{kind_name}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "summary": f"重试失败：{e!r}"}
        if out.get("ok"):
            return {"ok": True,
                    "summary": "重试已开始：" + str(out.get("summary") or "")}
        return out

    def _memory_recall(self, p, now):
        """轻量链式召回：按关键词翻本会话的旧消息和任务，不把全量历史塞 prompt。"""
        query = str(p.get("query") or "").strip()[:60]
        if not query:
            return {"ok": False, "summary": "缺少要回忆的关键词"}
        try:
            limit = int(p.get("limit") or 8)
        except (TypeError, ValueError):
            limit = 8
        limit = max(1, min(limit, 20))
        like = "%" + query + "%"
        lines = []
        try:
            rows = self.conn.execute(
                "SELECT sender, speaker, speaker_role, content, created_at FROM messages"
                " WHERE role_id=? AND content LIKE ? ORDER BY id DESC LIMIT ?",
                (self.role_id, like, limit)).fetchall()
            for r in reversed(rows):
                who = ("你" if r["sender"] == "agent" else
                       (r["speaker"] or ("创造者" if r["speaker_role"] == "owner" else "对方")))
                when = time.strftime("%m-%d %H:%M", time.localtime(r["created_at"] or 0))
                lines.append(f"[{when}] {who}：{(r['content'] or '')[:80]}")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "summary": f"翻聊天记录失败：{e!r}"}
        try:
            for t in self.tasks.recent(50):
                blob = str(t.get("params") or "") + str(t.get("result") or "")
                if query not in blob:
                    continue
                lines.append(f"[任务/{t['status']}] {self._task_title(t) or t['kind']}")
                if len(lines) >= limit:
                    break
        except Exception:  # noqa: BLE001  任务翻不到不影响消息召回
            pass
        if not lines:
            return {"ok": False, "summary": f"没找到跟「{query}」有关的旧记录"}
        return {"ok": True,
                "summary": f"关于「{query}」我记得：" + "；".join(lines[-limit:])}

    def _song_sing(self, p, now):
        """点唱：立即返回一句"在唱了"，后台生成完再由桥发语音气泡。"""
        from . import songgen  # 延迟导入（核心链不依赖）
        cfg = self.cfg.song_gen
        if not cfg.enabled:
            return {"ok": False, "summary": "唱歌未开启（song_gen.enabled=false）"}
        if not songgen.health_check(cfg):
            return {"ok": False, "summary": (
                "唱歌服务没开（本地 ACE-Step 未启动）。别答应唱歌：直接说这次唱不了，"
                "或者先聊别的")}
        lyrics = songgen.sanitize_lyrics(str(p.get("lyrics") or ""), cfg.max_lyrics_chars)
        style = str(p.get("style") or "").strip()[:120]
        if not style and getattr(cfg, "style", ""):
            style = str(cfg.style).strip()[:120]   # 后端显式配置的兜底（默认留空）
        if not style:
            return {"ok": False, "need_style": True, "summary": (
                "对方还没说想听什么曲风。先问一句「想听什么风格的？」"
                "（别直接唱，也别自己定一个默认风格）；对方说随便/都行/你定，"
                "再由你挑一个具体风格写进 style 重新调用。")}
        voice = str(p.get("voice") or "").strip()[:60] or str(getattr(cfg, "voice", "") or "")
        title = str(p.get("title") or "").strip()[:40]
        duration = None
        if p.get("duration") not in (None, ""):
            duration = songgen.parse_duration(
                p.get("duration"), default=None, lo=10.0,
                hi=float(getattr(cfg, "max_duration_seconds", 300)))
        with self._song_lock:
            if self._song_running:
                return {"ok": True, "summary": "已经在唱了，这首还没唱完",
                        "singing": True}
            self._song_running = True
        job_id = uuid.uuid4().hex[:8]
        task = {}
        try:
            task = self.tasks.create("song.sing", self._task_params({
                "lyrics": lyrics, "style": style, "voice": voice,
                "duration": duration, "title": title}))
        except Exception as e:  # noqa: BLE001  留痕失败不挡唱歌
            print(f"[task] 建任务失败：{e!r}")
        self.song_jobs[job_id] = {"status": "running", "lyrics": lyrics,
                                  "style": style, "voice": voice, "title": title,
                                  "duration": duration, "started_at": now,
                                  "task_id": task.get("id", "")}
        while len(self.song_jobs) > 20:   # 只留最近 20 条，防长跑堆积
            self.song_jobs.pop(next(iter(self.song_jobs)))
        threading.Thread(target=self._song_worker,
                         args=(job_id, lyrics, style, voice, duration, title),
                         daemon=True, name="song-worker").start()
        first = lyrics.splitlines()[0] if lyrics else "这首歌"
        if getattr(cfg, "voice_clone_enabled", False):
            eta = max(30, int(getattr(cfg, "voice_clone_eta_seconds", 150)))
            wait = f"换她的音色要慢一点，约 {max(1, eta // 60)}-{max(2, eta // 60 + 1)} 分钟后到"
        elif duration:
            wait = "约 %d 秒后到" % max(15, int(duration * 0.6))
        else:
            wait = "约 20 秒后发语音"
        return {"ok": True,
                "summary": f"在唱了：{first}…（{wait}）",
                "singing": True, "job_id": job_id, "lyrics": lyrics,
                "duration": duration}

    def _song_worker(self, job_id, lyrics, style, voice, duration, title):
        """后台生成；成功/失败都通过桥注入的 song_sender 投递。"""
        from . import songgen
        cfg = self.cfg.song_gen
        sender = getattr(self, "song_sender", None)
        task_id = self.song_jobs.get(job_id, {}).get("task_id", "")
        self._task_update(task_id, status="running", stage="generating")
        self._task_heartbeat(task_id, stage="generating")
        try:
            info = songgen.generate_song(lyrics, style, cfg, out_dir=None,
                                         voice=voice, duration=duration,
                                         logger=lambda m: print(f"[song] {m}"))
            info.update({"job_id": job_id, "title": title})
            self._task_heartbeat(task_id, stage="voice_clone")
            info = self._maybe_voice_clone(info, cfg)
            if self._song_cancelled(job_id):
                self.song_jobs[job_id] = {"status": "cancelled", "title": title,
                                          "job_id": job_id}
                self._task_update(task_id, status="cancelled", stage="cancelled",
                                  error="用户取消")
                return
            self.song_jobs[job_id] = {"status": "done", **info}
            self._task_update(task_id, status="ready", stage="done", result={
                "path": info.get("path"), "duration_ms": info.get("duration_ms"),
                "title": title, "voice_cloned": bool(info.get("voice_cloned"))})
            if sender is not None:
                try:
                    sender(info)
                except Exception as e:  # noqa: BLE001  生成成功但投递失败，任务仍是 ready
                    print(f"[song] 投递回调失败：{e!r}")
            else:
                print(f"[song] 生成完成（桥未注入投递回调）：{info.get('path')}")
        except Exception as e:  # noqa: BLE001  后台失败不影响对话
            self.song_jobs[job_id] = {"status": "failed", "error": repr(e),
                                      "lyrics": lyrics, "title": title}
            self._task_update(task_id, status="failed", stage="failed", error=repr(e))
            print(f"[song] 生成失败：{e!r}")
            if sender is not None:
                try:
                    sender({"error": str(e), "job_id": job_id, "lyrics": lyrics})
                except Exception as e2:  # noqa: BLE001
                    print(f"[song] 投递失败回调也失败：{e2!r}")
        finally:
            with self._song_lock:
                self._song_running = False

    def _maybe_voice_clone(self, info: dict, cfg) -> dict:
        """本地换音色：一个 HTTP 调用，worker 在 GPU 上做完分离+换色+混音。"""
        if not getattr(cfg, "voice_clone_enabled", False):
            return info
        import json as _json
        import urllib.request
        from . import songgen
        try:
            out_dir = os.path.dirname(info["path"])
            out_mp3 = os.path.join(out_dir, "vc_" + os.path.basename(info["path"]))
            from .silk_audio import ffmpeg_exe
            payload = _json.dumps({
                "input": os.path.abspath(info["path"]),
                "reference": cfg.voice_clone_reference,
                "output": os.path.abspath(out_mp3),
                "steps": int(cfg.voice_clone_steps),
                "vocal_gain": float(cfg.voice_clone_vocal_gain),
                "high_note_protect_hz": float(
                    getattr(cfg, "voice_clone_high_note_protect_hz", 0) or 0),
                "ffmpeg": ffmpeg_exe(),
            }, ensure_ascii=False).encode()
            print("[song] 换音色：worker 开始（分离+换色+混音）", flush=True)
            result = _post_worker(cfg.voice_clone_endpoint.rstrip("/") + "/convert",
                                  payload, int(cfg.voice_clone_timeout))
            if not result.get("ok") or not os.path.exists(result.get("path") or ""):
                raise RuntimeError(str(result.get("error") or result)[:200])
            new_info = dict(info)
            new_info.update({
                "path": result["path"],
                "voice_cloned": True,
                "duration_ms": songgen.audio_duration_ms(result["path"])
                or info.get("duration_ms"),
            })
            print(f"[song] 换音色完成（worker {result.get('elapsed')}s："
                  f"demucs {result.get('demucs_s')}s + vc {result.get('vc_s')}s）",
                  flush=True)
            return new_info
        except Exception as e:
            print(f"[song] 换音色失败，发原声：{e!r}", flush=True)
            return info

    # ---------------- 翻唱（原曲换色） ----------------

    def _song_cover(self, p, now):
        """翻唱：搜歌→下载→vc-worker 换色→后台发回。不经过 ACE-Step。"""
        from . import songgen
        from . import songsearch
        cfg = self.cfg.song_gen
        if not cfg.enabled:
            return {"ok": False, "summary": "唱歌未开启（song_gen.enabled=false）"}
        if not getattr(cfg, "voice_clone_enabled", False):
            return {"ok": False, "summary": (
                "翻唱需要换音色服务（voice_clone_enabled=true）。"
                "别答应翻唱：说翻唱服务没开，只能自己编歌唱")}
        title = str(p.get("title") or "").strip()[:60]
        artist = str(p.get("artist") or "").strip()[:40]
        raw_gender = str(p.get("gender") or "").strip().lower()
        gender = {"男": "male", "男性": "male", "男生": "male",
                  "女": "female", "女性": "female", "女生": "female"}.get(
                      raw_gender, raw_gender)
        if gender not in ("male", "female"):
            gender = ""
        raw_section = str(p.get("section") or "").strip().lower()
        section = "chorus" if raw_section in (
            "副歌", "高潮", "chorus", "hook", "高音", "高潮部分") else ""
        lyric = str(p.get("lyric") or "").strip()[:60]
        raw_ac = p.get("acappella")
        acappella = raw_ac is True or str(raw_ac).strip().lower() in (
            "1", "true", "yes", "on", "清唱", "不要伴奏", "纯人声")
        raw_duet = p.get("duet")
        duet = raw_duet is True or str(raw_duet).strip().lower() in (
            "1", "true", "yes", "on", "双轨", "合唱", "和声")
        try:
            duet_pitch = int(p.get("duet_pitch") or -12)
        except (TypeError, ValueError):
            duet_pitch = -12
        duet_pitch = max(-12, min(0, duet_pitch))
        if not title:
            return {"ok": False, "summary": "缺少歌名（title 参数）"}
        query = f"{title} {artist}".strip()
        duration = songgen.parse_duration(p.get("duration"), default=None, lo=15.0, hi=300.0)
        start = songgen.parse_duration(p.get("start"), default=0, lo=0, hi=600) or 0
        if (section == "chorus" or lyric) and not duration:
            duration = 30.0          # 自动找副歌/按歌词定位时默认唱 30 秒
        with self._song_lock:
            if self._song_running:
                return {"ok": True, "summary": "已经在唱了，这首还没唱完", "singing": True}
            self._song_running = True
        job_id = uuid.uuid4().hex[:8]
        task = {}
        try:
            task = self.tasks.create("song.cover", self._task_params({
                "query": query, "title": title, "artist": artist, "gender": gender,
                "section": section, "lyric": lyric, "acappella": acappella,
                "duet": duet, "duet_pitch": duet_pitch,
                "duration": duration, "start": start, "job_id": job_id,
            }), idempotency_key="cover:%s|%s|%s|%s|%s|%s|%s|%s|%s" % (
                query, gender, section, lyric, acappella, duet, duet_pitch,
                duration, start))
        except Exception as e:  # noqa: BLE001
            print(f"[task] 建翻唱任务失败：{e!r}")
        self.song_jobs[job_id] = {"status": "searching", "title": title,
                                  "artist": artist, "gender": gender,
                                  "section": section, "lyric": lyric,
                                  "acappella": acappella,
                                  "started_at": now,
                                  "task_id": task.get("id", "")}
        threading.Thread(target=self._song_cover_worker,
                         args=(job_id, query, title, artist, gender, section,
                               lyric, acappella, duet, duet_pitch, duration, start),
                         daemon=True, name="song-cover").start()
        if lyric:
            full = f"从「{lyric[:12]}」开始唱 {int(duration)} 秒"
        elif section == "chorus":
            full = f"副歌约 {int(duration)} 秒"
        elif not duration:
            full = "全首"
        else:
            full = f"{int(duration)}秒"
        if acappella:
            full = "清唱·" + full
        if duet:
            full = "双轨·" + full
        eta = "约 1 分钟" if (section or lyric or acappella) else (
            "约 2 分钟" if not duration else f"约 {int(duration * 1.2)} 秒")
        return {"ok": True, "cover": True,
                "summary": f"在找《{title}》了，翻唱{full}（{eta}后到）",
                "job_id": job_id}

    def _song_cover_worker(self, job_id, query, title, artist, gender, section,
                           lyric, acappella, duet, duet_pitch, duration, start):
        """翻唱后台：搜歌→下载→裁剪→vc-worker 换色→投递。"""
        from . import songgen
        from . import songsearch
        cfg = self.cfg.song_gen
        sender = getattr(self, "song_sender", None)
        task_id = self.song_jobs.get(job_id, {}).get("task_id", "")
        self._task_update(task_id, status="running", stage="searching")
        self._task_heartbeat(task_id, stage="searching")
        try:
            # 1) 搜歌+下载
            out_dir = songgen._out_dir(cfg)
            import time as _t
            t0 = _t.time()
            src = songsearch.find_and_download(
                query, out_dir, title=title, artist=artist,
                # 她的音色是女声：优先女声源；清唱必须是独唱，双人/对唱会
                # 让自动定调按低音声部算，整首被推高（珊瑚海实测 +7）。
                prefer_female=True, solo=acappella,
                min_duration_s=max(30.0, float(start or 0)
                                   + float(duration or 30.0) + 10.0))
            print(f"[song-cover] 下载《{src['name']}》{src['size']//1024}KB "
                  f"({round(_t.time()-t0,1)}s，源={src.get('source')}"
                  f"{'，女声版' if src.get('is_female') else ''})", flush=True)
            self.song_jobs[job_id] = {"status": "downloading", **src, "job_id": job_id}
            self._task_heartbeat(task_id, stage="downloading")

            src_path = src["path"]
            # 2.4) 按歌词定位：LRC 行时间 -> 起点
            if lyric:
                try:
                    sid = src.get("id") if src.get("source") == "netease" else None
                    sec, matched = songsearch.find_lyric_start(
                        title, artist, lyric, sid)
                    if sec is not None:
                        start = sec
                        print(f"[song-cover] 歌词定位：{sec:.1f}s「{matched}」",
                              flush=True)
                    else:
                        lyric = ""
                        print("[song-cover] 歌词没匹配到，回退副歌/开头", flush=True)
                except Exception as e:  # noqa: BLE001
                    lyric = ""
                    print(f"[song-cover] 歌词定位失败，回退：{e!r}", flush=True)

            # 2.5) 副歌/高潮：按响度 + 重复度自动定位起点
            if not lyric and section == "chorus":
                try:
                    start = songsearch.find_chorus(src_path, duration or 30.0)
                    print(f"[song-cover] 副歌定位：{start:.1f}s 起", flush=True)
                except Exception as e:  # noqa: BLE001
                    start = 0
                    print(f"[song-cover] 副歌定位失败，从 0s 唱：{e!r}", flush=True)

            # 2) 裁剪（可选）
            if self._song_cancelled(job_id):
                print(f"[song-cover] 已取消（{src['name']}），不换色", flush=True)
                self.song_jobs[job_id] = {"status": "cancelled", "title": title,
                                          "job_id": job_id}
                self._task_update(task_id, status="cancelled", stage="cancelled",
                                  error="用户取消")
                return
            if duration or start > 0:
                clip_path = os.path.join(out_dir, "clip_%s.mp3" % job_id)
                src_path = songsearch.clip_audio(
                    src_path, clip_path, start_s=start, duration_s=duration)
                print(f"[song-cover] 裁剪到 {duration or '末尾'}s", flush=True)

            # 3) vc-worker 换色（跟 song.sing 共用同一个 worker）
            import json as _json
            import urllib.request
            from .silk_audio import ffmpeg_exe
            out_mp3 = os.path.join(out_dir, "cover_%s_%s.mp3" % (
                _t.strftime("%Y%m%d_%H%M%S"), job_id))
            # 用了女声翻唱版就不按原唱男声再移调；让 worker 按源音频自己判
            worker_gender = "" if src.get("is_female") else gender
            payload = _json.dumps({
                "input": os.path.abspath(src_path),
                "reference": cfg.voice_clone_reference,
                "output": os.path.abspath(out_mp3),
                "steps": int(cfg.voice_clone_steps),
                "vocal_gain": float(cfg.voice_clone_vocal_gain),
                "gender": worker_gender,
                "acappella": bool(acappella),
                "duet": bool(duet),
                "duet_pitch": int(duet_pitch),
                "ffmpeg": ffmpeg_exe(),
            }, ensure_ascii=False).encode()
            print(f"[song-cover] 换色中（{src['name']}）...", flush=True)
            self._task_heartbeat(task_id, stage="converting")
            result = _post_worker(cfg.voice_clone_endpoint.rstrip("/") + "/convert",
                                  payload, int(cfg.voice_clone_timeout))
            if not result.get("ok"):
                raise RuntimeError(str(result.get("error") or result)[:200])
            if self._song_cancelled(job_id):
                print(f"[song-cover] 已取消，丢弃成品：{result['path']}", flush=True)
                self.song_jobs[job_id] = {"status": "cancelled", "title": title,
                                          "job_id": job_id}
                self._task_update(task_id, status="cancelled", stage="cancelled",
                                  error="用户取消")
                return
            info = {
                "path": result["path"],
                "duration_ms": songgen.audio_duration_ms(result["path"]),
                "voice_cloned": True,
                "cover": True,
                "duet": bool(duet),
                "title": title,
                "artist": src.get("artist", artist),
                "job_id": job_id,
            }
            self.song_jobs[job_id] = {"status": "done", **info}
            self._task_update(task_id, status="ready", stage="done", result={
                "path": info.get("path"), "duration_ms": info.get("duration_ms"),
                "title": title, "cover": True, "acappella": bool(acappella),
                "duet": bool(duet)})
            if sender:
                try:
                    sender(info)
                except Exception as e:  # noqa: BLE001  生成成功但投递失败，任务仍是 ready
                    print(f"[song-cover] 投递回调失败：{e!r}")
            else:
                print(f"[song-cover] 完成（桥未注入回调）：{result['path']}", flush=True)
            # 清理源文件
            for f in (src["path"], src_path if src_path != src["path"] else None):
                if f and os.path.exists(f) and f != result["path"]:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        except Exception as e:
            self.song_jobs[job_id] = {"status": "failed", "error": repr(e), "title": title}
            self._task_update(task_id, status="failed", stage="failed", error=repr(e))
            print(f"[song-cover] 失败：{e!r}", flush=True)
            if sender:
                try:
                    sender({"error": str(e), "cover": True, "title": title, "job_id": job_id})
                except Exception:
                    pass
        finally:
            with self._song_lock:
                self._song_running = False

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
        with self._db_lock:
            self.conn.execute(
                "INSERT INTO hands_log(role_id, action, params, result, created_at)"
                " VALUES(?,?,?,?,?)",
                (self.role_id, action, json.dumps(params, ensure_ascii=False),
                 json.dumps(result, ensure_ascii=False), now))
            self.conn.commit()
