"""mem0 接入探针：桥接 LLM（自定义类，system 并入 user）+ fastembed 中文向量 + qdrant 本地库。
验证：事实抽取走桥接、语义检索换说法也能命中。"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mem0 import Memory
from mem0.configs.base import MemoryConfig, VectorStoreConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.base import LLMBase

from src.perception import extract_json


class BridgeLLM(LLMBase):
    """桥接专用 LLM：桥接丢弃 system 角色，所以把所有消息合并进 user，只回 JSON。"""

    def __init__(self, config=None):
        super().__init__(config)
        self.base_url = os.environ.get("AGENT_LLM_BASE_URL", "")
        self.model = os.environ.get("AGENT_LLM_MODEL", "")
        self.api_key = os.environ.get("LLM_API_KEY", "")

    def generate_response(self, messages, response_format=None, tools=None,
                          tool_choice="auto", **kwargs):
        parts = [f"【{m.get('role')}】\n{m['content']}" for m in messages if m.get("content")]
        text = "\n\n".join(parts) + "\n\n只输出要求的 JSON 对象，禁止任何其他文字。"
        body = {"model": self.model,
                "messages": [{"role": "user", "content": text}],
                "temperature": 0.1, "max_tokens": 1200}
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
                obj = extract_json(raw)
                return json.dumps(obj, ensure_ascii=False) if obj is not None else raw
            except Exception as e:  # noqa: BLE001
                if attempt == 1:
                    raise
                print(f"  [retry] {e!r}")


def build_memory():
    cfg = MemoryConfig(
        vector_store=VectorStoreConfig(provider="qdrant", config={
            "collection_name": "probe", "embedding_model_dims": 512,
            "path": os.path.join(ROOT, "data", "mem0_probe")}),
        llm={"provider": "openai", "config": {"api_key": "dummy"}},
        embedder=EmbedderConfig(provider="fastembed",
                                config={"model": "BAAI/bge-small-zh-v1.5"}),
        history_db_path=os.path.join(ROOT, "data", "mem0_probe_history.db"),
    )
    mem = Memory(cfg)
    mem.llm = BridgeLLM()
    return mem


if __name__ == "__main__":
    m = build_memory()
    print("== add 1")
    r1 = m.add([{"role": "user", "content": "我打算下个月开始晨跑，先定个小目标三公里"}],
               user_id="hu")
    print(json.dumps(r1, ensure_ascii=False)[:400])
    print("== add 2（含已存在的相似事实，测去重/更新）")
    r2 = m.add([{"role": "user", "content": "晨跑那个，我改成跑五公里了，还是下个月开始"}],
               user_id="hu")
    print(json.dumps(r2, ensure_ascii=False)[:400])
    print("== add 3")
    m.add([{"role": "user", "content": "我们家猫最近总半夜跑酷，吵得我睡不好"}], user_id="hu")
    print("== 语义检索（无共同关键词）")
    for q in ("运动锻炼的事", "宠物", "睡眠不好"):
        res = m.search(query=q, filters={"user_id": "hu"}, limit=3)
        print(f"Q:{q} ->", [(r["memory"][:36], round(r["score"], 3)) for r in res["results"]])
    print("== 全量")
    print([r["memory"][:40] for r in m.get_all(filters={"user_id": "hu"})["results"]])
