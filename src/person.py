"""人物卡 + 信任引擎（v1）。

卡片是记忆的物化视图：身份/关系/策略是稳定结构，聊天流水仍写 memories。
信任按 person_events 做有界 EMA，不由记忆检索直接修改。
"""
import json
import re
import time


def _clamp(x, lo=0.02, hi=0.98):
    return max(lo, min(hi, float(x)))


def _move(cur, target, alpha):
    return _clamp(float(cur) + (float(target) - float(cur)) * float(alpha))


class PersonStore:
    DEFAULTS = {
        "creator": dict(affinity=0.8, familiarity=1.0, trust_identity=1.0,
                        trust_benevolence=0.85, trust_competence=0.7,
                        trust_disclosure=1.0, caution=0.05, attachment=1.0,
                        authority="creator"),
        "friend": dict(affinity=0.55, familiarity=0.4, trust_identity=0.8,
                       trust_benevolence=0.6, trust_competence=0.5,
                       trust_disclosure=0.35, caution=0.15, attachment=0.0,
                       authority="none"),
        "group_member": dict(affinity=0.3, familiarity=0.1, trust_identity=0.7,
                             trust_benevolence=0.5, trust_competence=0.4,
                             trust_disclosure=0.1, caution=0.3, attachment=0.0,
                             authority="none"),
        "agent": dict(affinity=0.25, familiarity=0.1, trust_identity=0.4,
                      trust_benevolence=0.35, trust_competence=0.45,
                      trust_disclosure=0.0, caution=0.4, attachment=0.0,
                      authority="none"),
        "stranger": dict(affinity=0.1, familiarity=0.0, trust_identity=0.3,
                         trust_benevolence=0.3, trust_competence=0.3,
                         trust_disclosure=0.0, caution=0.5, attachment=0.0,
                         authority="none"),
    }

    def __init__(self, conn, role_id, persona=None):
        self.conn = conn
        self.role_id = role_id
        self.persona = persona

    def ensure(self, wxid, name="", kind="group_member", now=None):
        now = time.time() if now is None else now
        wxid = str(wxid or "").strip()
        if not wxid:
            return None
        row = self.conn.execute(
            "SELECT id FROM persons WHERE role_id=? AND wxid=?",
            (self.role_id, wxid)).fetchone()
        if not row and name:
            # 兼容旧库回填的 name: 伪 wxid：真实 wxid 到达时归并到同一张卡
            row = self.conn.execute(
                "SELECT id FROM persons WHERE role_id=? AND name=? AND wxid LIKE 'name:%'"
                " LIMIT 1", (self.role_id, name)).fetchone()
            if row:
                self.conn.execute("UPDATE persons SET wxid=? WHERE id=?", (wxid, row["id"]))
        if row:
            pid = int(row["id"])
            self.conn.execute(
                "UPDATE persons SET name=?, last_seen=? WHERE id=?",
                (name or "", now, pid))
            if not self.conn.execute(
                    "SELECT 1 FROM person_cards WHERE person_id=?", (pid,)).fetchone():
                self._new_card(pid, kind, now)
            self.conn.commit()
            return pid
        cur = self.conn.execute(
            "INSERT INTO persons(role_id, wxid, name, kind, first_seen, last_seen)"
            " VALUES(?,?,?,?,?,?)", (self.role_id, wxid, name or "", kind, now, now))
        pid = int(cur.lastrowid)
        self._new_card(pid, kind, now)
        self.conn.commit()
        return pid

    def _new_card(self, person_id, kind, now):
        d = dict(self.DEFAULTS.get(kind, self.DEFAULTS["group_member"]))
        self.conn.execute(
            "INSERT INTO person_cards(person_id, affinity, familiarity, trust_identity,"
            " trust_benevolence, trust_competence, trust_disclosure, caution, attachment,"
            " authority, updated_at, version) VALUES(?,?,?,?,?,?,?,?,?,?,?,1)",
            (person_id, d["affinity"], d["familiarity"], d["trust_identity"],
             d["trust_benevolence"], d["trust_competence"], d["trust_disclosure"],
             d["caution"], d["attachment"], d["authority"], now))

    def ensure_creator(self, now=None):
        creator = getattr(self.persona, "creator", None) or {}
        wxid = str(creator.get("wxid") or "").strip()
        name = str(creator.get("name") or "").strip()
        if not wxid:
            return None
        return self.ensure(wxid, name, "creator", now=now)

    def get(self, person_id):
        row = self.conn.execute(
            "SELECT p.*, c.* FROM persons p JOIN person_cards c ON c.person_id=p.id"
            " WHERE p.id=? AND p.role_id=?", (int(person_id), self.role_id)).fetchone()
        return dict(row) if row else None

    def get_by_wxid(self, wxid):
        row = self.conn.execute(
            "SELECT id FROM persons WHERE role_id=? AND wxid=?",
            (self.role_id, str(wxid or ""))).fetchone()
        return self.get(row["id"]) if row else None

    def attach_unscoped_memories(self, person_id, scope):
        """把旧库没有 person_id 的记忆归到创造者/当前群，避免新过滤后失忆。"""
        if person_id is None:
            return 0
        cur = self.conn.execute(
            "UPDATE memories SET person_id=?, scope=? WHERE role_id=?"
            " AND person_id IS NULL", (int(person_id), scope, self.role_id))
        self.conn.commit()
        return cur.rowcount

    def list_cards(self, limit: int = 30):
        rows = self.conn.execute(
            "SELECT p.id, p.wxid, p.name, p.kind, c.affinity, c.familiarity,"
            " c.trust_identity, c.trust_benevolence, c.trust_competence,"
            " c.trust_disclosure, c.caution, c.attachment, c.authority"
            " FROM persons p JOIN person_cards c ON c.person_id=p.id"
            " WHERE p.role_id=? ORDER BY c.updated_at DESC LIMIT ?",
            (self.role_id, int(limit))).fetchall()
        out = []
        for r in rows:
            out.append({
                "person_id": r["id"], "wxid": r["wxid"], "name": r["name"],
                "kind": r["kind"], "affinity": round(r["affinity"], 2),
                "familiarity": round(r["familiarity"], 2),
                "trust_identity": round(r["trust_identity"], 2),
                "trust_benevolence": round(r["trust_benevolence"], 2),
                "trust_competence": round(r["trust_competence"], 2),
                "trust_disclosure": round(r["trust_disclosure"], 2),
                "caution": round(r["caution"], 2), "attachment": round(r["attachment"], 2),
                "authority": r["authority"],
            })
        return out

    def summary(self, person_id):
        p = self.get(person_id)
        if not p:
            return None
        return {
            "姓名": p.get("name") or "",
            "身份": p.get("kind") or "unknown",
            "亲近": round(float(p.get("affinity") or 0), 2),
            "信任": round(float(p.get("trust_benevolence") or 0), 2),
            "身份可信": round(float(p.get("trust_identity") or 0), 2),
            "能力可信": round(float(p.get("trust_competence") or 0), 2),
            "可披露": round(float(p.get("trust_disclosure") or 0), 2),
            "戒备": round(float(p.get("caution") or 0), 2),
            "关系": "创造者" if p.get("authority") == "creator" else "群友",
        }

    def _event_type(self, text, from_owner):
        t = str(text or "")
        if re.search(r"挑拨|骗你|操纵|带节奏|冒充|撒谎|造谣|利用你", t):
            return "manipulation"
        if re.search(r"傻|蠢|滚|废物|骂|他妈|闭嘴|垃圾", t):
            return "insult"
        if re.search(r"谢谢|帮我|帮忙|辛苦|靠谱|厉害|赞", t):
            return "help"
        if re.search(r"保证|答应|一定|约好|说好", t):
            return "promise"
        return "chat"

    def observe(self, person_id, text, from_owner=False, source_message_id=None,
                now=None):
        now = time.time() if now is None else now
        p = self.get(person_id)
        if not p:
            return None
        kind = p.get("kind") or "group_member"
        etype = self._event_type(text, from_owner)
        if kind == "creator" and etype in ("insult", "manipulation"):
            etype = "chat"   # 创造者的批评不降低信任/不抬高戒备
        before = {k: float(p.get(k) or 0) for k in
                  ("affinity", "familiarity", "trust_benevolence",
                   "trust_competence", "trust_identity", "caution")}
        after = dict(before)
        if etype == "help":
            after["affinity"] = _move(after["affinity"], 0.85, 0.05)
            after["trust_benevolence"] = _move(after["trust_benevolence"], 0.95, 0.08)
            after["trust_competence"] = _move(after["trust_competence"], 0.8, 0.04)
            after["caution"] = _move(after["caution"], 0.05, 0.02)
        elif etype == "insult":
            after["affinity"] = _move(after["affinity"], 0.05, 0.10)
            after["trust_benevolence"] = _move(after["trust_benevolence"], 0.05, 0.18)
            after["caution"] = _move(after["caution"], 0.95, 0.20)
        elif etype == "manipulation":
            after["trust_benevolence"] = _move(after["trust_benevolence"], 0.02, 0.25)
            after["caution"] = _move(after["caution"], 0.98, 0.25)
            after["trust_identity"] = _move(float(p.get("trust_identity") or 0), 0.1, 0.2)
        elif etype == "promise":
            after["trust_benevolence"] = _move(after["trust_benevolence"], 0.9, 0.05)
        # 熟悉度：每次互动微增；亲近度只由事件缓慢变化
        after["familiarity"] = _clamp(float(after["familiarity"]) + 0.01)
        if from_owner:
            after["affinity"] = _clamp(float(after["affinity"]) + 0.005)
        self.conn.execute(
            "UPDATE person_cards SET affinity=?, familiarity=?, trust_identity=?,"
            " trust_benevolence=?, trust_competence=?, caution=?, updated_at=?,"
            " version=version+1 WHERE person_id=?",
            (after["affinity"], after["familiarity"], after["trust_identity"],
             after["trust_benevolence"], after["trust_competence"], after["caution"],
             now, int(person_id)))
        if etype != "chat":
            self.conn.execute(
                "INSERT INTO person_events(role_id, person_id, event_type, weight,"
                " content, source_message_id, value_before, value_after, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (self.role_id, int(person_id), etype,
                 {"help": 0.8, "insult": -0.7, "manipulation": -1.0,
                  "promise": 0.4}.get(etype, 0.0), str(text or "")[:240],
                 source_message_id, json.dumps(before, ensure_ascii=False),
                 json.dumps(after, ensure_ascii=False), now))
        self.conn.commit()
        return etype
