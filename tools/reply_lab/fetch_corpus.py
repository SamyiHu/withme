"""拉公开语料（LCCC toy 样本）到 data/reply_lab/。

只下公开研究数据，放 data/ 下（gitignore），不随仓库分发。
    python -m tools.reply_lab.fetch_corpus
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "https://github.com/thu-coai/CDial-GPT"


def main():
    dst_dir = os.path.join(ROOT, "data", "reply_lab")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "lccc_toy.json")
    if os.path.exists(dst):
        print("已存在：", dst)
        return 0
    tmp = tempfile.mkdtemp(prefix="lccc_")
    try:
        print("克隆", REPO, "（--depth 1）…")
        subprocess.run(["git", "clone", "--depth", "1", REPO, os.path.join(tmp, "cdial")],
                       check=True)
        src = os.path.join(tmp, "cdial", "data", "toy_data.json")
        shutil.copy(src, dst)
        print("已保存：", dst, os.path.getsize(dst), "bytes")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
