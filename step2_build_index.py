"""
第二步：把parsed_content.json里的每个chunk转成向量，建FAISS索引
运行前：确保 config.py 里填好了智谱API Key
运行：python step2_build_index.py
"""

import json
import time
import numpy as np
import faiss
from tqdm import tqdm
from zhipuai import ZhipuAI

from config import ZHIPU_API_KEY

INPUT_PATH = "data/parsed_content_tagged.json"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/chunk_meta.json"

EMBEDDING_MODEL = "embedding-3"
MIN_CONTENT_LEN = 10  # 过短的内容块（比如空表格行）直接跳过，减少噪音

client = ZhipuAI(api_key=ZHIPU_API_KEY)


def get_embedding(text: str):
    """调用智谱embedding接口，返回向量（list of float）"""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def build_index():
    with open(INPUT_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    # 过滤掉太短的内容块
    chunks = [c for c in chunks if len(c["content"].strip()) >= MIN_CONTENT_LEN]
    print(f"共 {len(chunks)} 个有效内容块，开始向量化...")

    vectors = []
    valid_meta = []

    for c in tqdm(chunks):
        try:
            vec = get_embedding(c["content"])
            vectors.append(vec)
            valid_meta.append(c)
        except Exception as e:
            print(f"\n[跳过] 第{c['page']}页 embedding失败: {e}")
        time.sleep(0.05)  # 简单限速，避免请求过快触发限流

    vectors = np.array(vectors, dtype="float32")
    print(f"\n向量维度: {vectors.shape}")

    # 用内积做相似度（配合归一化向量，等价于余弦相似度）
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(valid_meta, f, ensure_ascii=False, indent=2)

    print(f"索引已保存到 {INDEX_PATH}")
    print(f"元数据已保存到 {META_PATH}")
    print(f"共索引 {len(valid_meta)} 个内容块")


if __name__ == "__main__":
    build_index()
