"""开源版导出：git 跟踪文件 − .open-source-exclude 排除 → 目标目录 + 敏感信息扫描。

用法（在仓库根执行）：
  python tools/export_open_source.py --out ../withme-open-source
  python tools/export_open_source.py --out ../withme-open-source --zip

导出规则：
  - 文件全集 = `git ls-files`（未跟踪的运行产物天然不进包）；
  - 再按 `.open-source-exclude` 逐条排除（支持目录前缀与 fnmatch 通配）；
  - 复制的是工作区当前内容——先提交或确认改动再导出（有未提交改动时仅提示）。

导出完成后自动扫描（命中即非零退出）：
  - `.env` 里的真实 key 值（原样字符串检索）；
  - `.open-source-banned.txt`（本地禁词清单，不入库）里的真实昵称/wxid 等；
  - 通用形态：sk- 密钥、真实形态 wxid（长串含数字）、本机用户路径、硬编码 api_key。
"""

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXCLUDE_FILE = REPO / ".open-source-exclude"
BANNED_FILE = REPO / ".open-source-banned.txt"
ENV_FILE = REPO / ".env"

# 绝不允许出现在开源包里的文件/目录（排除清单失效时的最后一道闸）
MUST_NOT_SHIP = [
    ".env",
    "config/config.yaml",
    "config/persona.yaml",
]
MUST_NOT_SHIP_DIRS = ["data/", "logs/", "wechatauto_logs/", "reports/"]

# 通用敏感形态（正则，说明文字见 SCAN_LABELS）
RE_WXID = re.compile(r"wxid_[a-z0-9]{10,}")
RE_SK = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")
RE_WIN_PATH = re.compile(r"[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"']+")
RE_HARD_KEY = re.compile(r"api[_-]?key\s*[:=]\s*[\"'][A-Za-z0-9]{16,}", re.I)
RE_LONG_SECRET = re.compile(r"[\"'][A-Za-z0-9]{28,}[\"']")

SCAN_LABELS = {
    "wxid": "疑似真实 wxid（长串含字母数字）",
    "sk": "sk- 形态密钥",
    "winpath": "本机用户绝对路径（泄露用户名）",
    "hardkey": "硬编码 api_key 赋值",
    "longsecret": "疑似硬编码长密钥串",
}

TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".txt", ".sql", ".html", ".js",
    ".mjs", ".bat", ".vbs", ".service", ".csv", ".sh", ".cfg", ".ini", ".example",
}
TEXT_SUFFIXES.add("")  # 无后缀文件（如 .open-source-exclude）也当文本看


def load_patterns(path: Path) -> list:
    pats = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            pats.append(line.replace("\\", "/"))
    return pats


def excluded(rel: str, pats: list) -> bool:
    for pat in pats:
        if pat.endswith("/"):
            if rel.startswith(pat) or rel + "/" == pat:
                return True
        elif fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel + "/", pat):
            return True
        elif rel == pat or rel.startswith(pat + "/"):
            return True
    return False


def tracked_files() -> list:
    # 已跟踪 + 未被忽略的未跟踪文件（= 将来会进仓库的全集），
    # 去重后排序；被 .gitignore 忽略的运行产物/密钥不会出现
    cached = subprocess.run(["git", "ls-files", "-z", "--cached"], cwd=REPO,
                            capture_output=True, check=True).stdout
    others = subprocess.run(["git", "ls-files", "-z", "--others", "--exclude-standard"],
                            cwd=REPO, capture_output=True, check=True).stdout
    files = set()
    for blob in (cached, others):
        files.update(p for p in blob.decode("utf-8", "replace").split("\0") if p)
    return sorted(files)


# .env 里只有凭据类变量（名字含 key/token/secret/password）的值才按密钥对待，
# 否则 AGENT_LLM_PROVIDER=openai_compatible 这类枚举值会全网误报
RE_CRED_NAME = re.compile(r"key|token|secret|password", re.I)


def env_secret_values() -> list:
    if not ENV_FILE.exists():
        return []
    vals = []
    for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            v = v.strip()
            if RE_CRED_NAME.search(k) and len(v) >= 8:
                vals.append(v)
    return vals


def banned_literals() -> list:
    if not BANNED_FILE.exists():
        return []
    return [ln.strip() for ln in
            BANNED_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def scan_file(path: Path, secrets: list, banned: list) -> list:
    """返回 [(kind, detail)]；kind ∈ SCAN_LABELS / 'secret' / 'banned'。"""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    hits = []
    for i, v in enumerate(secrets):
        if v in text:
            hits.append(("secret", f".env 第{i + 1}个 key 的真实值"))
    for b in banned:
        if b in text:
            hits.append(("banned", f"禁词「{b}」"))
    for m in RE_WXID.finditer(text):
        if any(c.isdigit() for c in m.group()[5:]):
            hits.append(("wxid", m.group()))
    for m in RE_SK.finditer(text):
        hits.append(("sk", m.group()[:24] + "…"))
    for m in RE_WIN_PATH.finditer(text):
        hits.append(("winpath", m.group()))
    for m in RE_HARD_KEY.finditer(text):
        hits.append(("hardkey", m.group()[:40]))
    for m in RE_LONG_SECRET.finditer(text):
        hits.append(("longsecret", m.group()[:20] + "…"))
    return hits


def main():
    ap = argparse.ArgumentParser(description="导出开源版并扫描敏感信息")
    ap.add_argument("--out", default=str(REPO.parent / "withme-open-source"),
                    help="输出目录（默认仓库同级 withme-open-source）")
    ap.add_argument("--zip", action="store_true", help="额外打一个 zip 包")
    args = ap.parse_args()

    pats = load_patterns(EXCLUDE_FILE)
    files = [f for f in tracked_files() if not excluded(f, pats)]
    out = Path(args.out).resolve()

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, check=True).stdout.decode("utf-8", "replace").strip()
    if dirty:
        print("[提示] 工作区有未提交改动，导出的是工作区当前内容：")
        for ln in dirty.splitlines()[:10]:
            print("       " + ln)

    # 最后一道闸：排除清单失效也拦得住
    for f in files:
        for bad in MUST_NOT_SHIP:
            if f == bad:
                sys.exit(f"[拒绝] {bad} 进入导出清单——检查 .open-source-exclude")
        for d in MUST_NOT_SHIP_DIRS:
            if f.startswith(d):
                sys.exit(f"[拒绝] {f} 属于 {d}——检查 .open-source-exclude")

    if out.exists():
        # 快照目录可能被 git init 过（当作公开仓工作副本用），.git 对象文件在
        # Windows 上是只读的，rmtree 会 PermissionError——先去只读位再删
        import stat

        def _force_rm(func, path, _exc):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(out, onerror=_force_rm)
    copied = 0
    for f in files:
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, dst)
        copied += 1

    print(f"[导出] {copied} 个文件 → {out}")

    secrets = env_secret_values()
    banned = banned_literals()
    problems = []
    for f in files:
        for kind, detail in scan_file(out / f, secrets, banned):
            problems.append((f, kind, detail))

    top = {}
    for f in files:
        key = f.split("/", 1)[0] if "/" in f else "(根目录)"
        top[key] = top.get(key, 0) + 1
    print("[构成] " + ", ".join(f"{k}:{v}" for k, v in sorted(top.items())))

    if problems:
        print(f"\n[失败] 扫描命中 {len(problems)} 处：")
        for f, kind, detail in problems[:40]:
            label = SCAN_LABELS.get(kind, kind)
            print(f"  {f}: [{label}] {detail}")
        if len(problems) > 40:
            print(f"  … 共 {len(problems)} 处")
        sys.exit(1)
    print("[扫描] 无密钥 / wxid / 本机路径 / 禁词命中，通过。")

    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out.parent,
                                      base_dir=out.name)
        print(f"[打包] {archive}")


if __name__ == "__main__":
    main()
