"""表情包库（贴纸）：检索 / 挑选 / [sticker:xxx] 解析 / 桥接发送。

数据源 kiriyuki.com（作者：B站 可以叫我云大人），元数据在
assets/stickers/nekowhale.json；测试全部离线，不碰网络。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import stickers as st_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeStore(st_mod.StickerStore):
    """不联网：不加载元数据，手工塞几条。"""

    def __init__(self, media_dir):
        self.enabled = True
        self.log = lambda m: None
        self.site = "https://example.invalid"
        self.packs = []
        self.dir = media_dir
        self.meta_path = ""
        self.items = [
            {"id": "大肥鱼/1N01_摸鱼", "pack": "大肥鱼", "name": "1N01_摸鱼",
             "emotion": "开心", "tags": ["摸鱼", "偷懒", "上班"], "use": "上班摸鱼时",
             "desc": "", "en": "", "meme": "", "dir": "", "hash": ""},
            {"id": "大肥鱼/2N01_无语", "pack": "大肥鱼", "name": "2N01_无语",
             "emotion": "无语", "tags": ["无语", "翻白眼"], "use": "无话可说时",
             "desc": "", "en": "", "meme": "", "dir": "", "hash": ""},
        ]
        self.by_id = {i["id"]: i for i in self.items}
        self._manifest = []
        self._cache = {}

    def download(self, item):
        path = os.path.join(self.dir, item["id"].replace("/", "__") + ".gif")
        os.makedirs(self.dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"GIF89a" + b"0" * 100)
        return path

    def download_thumb(self, item):
        path = os.path.join(self.dir, item["id"].replace("/", "__") + ".png")
        os.makedirs(self.dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
        return path


def test_search_and_pick():
    tmp = make_tmpdir()
    try:
        st = FakeStore(tmp)
        hits = st.search("摸鱼", limit=3)
        assert hits and hits[0]["emotion"] == "开心", hits
        # 名字/标签命中优先，不乱随机
        assert st.pick("摸鱼")["name"] == "1N01_摸鱼"
        assert st.pick("无语")["name"] == "2N01_无语"
        assert st.pick("完全不存在的词") is None
    finally:
        rmtree(tmp)


def test_extract_sticker_marker():
    tmp = make_tmpdir()
    try:
        st = FakeStore(tmp)
        clean, paths = st_mod.extract_stickers("今天真不想干活\n[sticker:摸鱼]", st)
        assert clean == "今天真不想干活", repr(clean)
        assert len(paths) == 1 and os.path.exists(paths[0])
        assert paths[0].endswith(".png"), "static 模式发缩略图"
        # 一条回复最多发一张
        clean2, paths2 = st_mod.extract_stickers(
            "[sticker:摸鱼] 笑死 [sticker:无语]", st)
        assert len(paths2) == 1, paths2
        # 认不出的关键词：只去掉标记，不发图、不报错
        clean3, paths3 = st_mod.extract_stickers("嗯 [sticker:不存在的词]", st)
        assert clean3 == "嗯" and not paths3
    finally:
        rmtree(tmp)


def test_real_meta_loads_offline():
    """仓库里带的元数据能离线加载（不联网也能检索）。"""
    st = st_mod.StickerStore(
        media_dir=os.path.join(make_tmpdir(), "s"),
        packs=["大肥鱼"], log=lambda m: None)
    assert len(st.items) > 100, len(st.items)
    hits = st.search("无语", limit=3)
    assert hits, "离线检索要有结果"


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_stickers"))
